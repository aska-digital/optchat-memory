"""Tests for the summary tree, view fold/fit, zoom, and the compactor pump.

Uses a stub summarizer (no LLM): deterministic one-line summaries, with
an overshoot-then-comply variant for the size-enforcement test.
"""

import os
import stat
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from optchat import compact  # noqa: E402
from optchat.compact import (  # noqa: E402
    SCALE_LINE,
    build_node,
    compact_prompt,
    pump_once,
)
from optchat.store import ChatLog  # noqa: E402
from optchat.tree import NODE_BYTES, SummaryTree  # noqa: E402
from optchat.view import UNBUILT_PLACEHOLDER, View  # noqa: E402


def make_log(tmp_path, n, long=False):
    log = ChatLog(tmp_path / "optchat")
    for k in range(n):
        text = f"message {k}"
        if long:
            text += " x" * 600  # force model summarization path
        log.append("user" if k % 2 == 0 else "talk", text)
    return log


def stub_summarizer(messages):
    """Deterministic stand-in: summarizes from the step block's last line."""
    step = messages[-1]["content"] if isinstance(messages[-1], dict) else messages[-1]
    last = step.strip().splitlines()[-1]
    return f"stub: {last[:60]}"


def test_scale_line_is_node_bytes():
    assert len(SCALE_LINE.encode("utf-8")) == NODE_BYTES


def test_free_nodes_need_no_model_call(tmp_path):
    log = make_log(tmp_path, 4)
    tree = SummaryTree(tmp_path / "optchat")
    view = View(tree, budget_chars=100_000)
    view.fold_to(len(log))

    def explode(messages):
        raise AssertionError("summarizer must not be called for short messages")

    built = 0
    while pump_once(tree, log, view, explode, compact_prompt()):
        built += 1
        assert built < 100, "pump did not converge"
    # 4 level-0 + 2 + 1: every node free, no model call.
    assert built == 7
    for k in range(4):
        assert tree.built(0, k)
        assert tree.get(0, k).startswith("user:" if k % 2 == 0 else "talk:")
    assert tree.built(2, 0)
    log.close()


def test_pump_builds_in_order_then_merges(tmp_path):
    log = make_log(tmp_path, 8)
    tree = SummaryTree(tmp_path / "optchat")
    view = View(tree, budget_chars=100_000)
    view.fold_to(len(log))
    order = []

    def recording(messages):
        order.append("model")
        return stub_summarizer(messages)

    # Long messages force the model path for every node.
    log2 = make_log(tmp_path / "long", 8, long=True)
    tree2 = SummaryTree(tmp_path / "long")
    view2 = View(tree2, budget_chars=100_000)
    view2.fold_to(len(log2))
    built = 0
    build_order = []
    real_set = tree2.set

    def recording_set(l, i, text):
        build_order.append((l, i))
        real_set(l, i, text)

    tree2.set = recording_set  # type: ignore[method-assign]
    while pump_once(tree2, log2, view2, recording, compact_prompt()):
        built += 1
        assert built < 100, "pump did not converge"
    # 8 level-0 + 4 + 2 + 1 = 15 nodes; level 0 strictly in message order.
    assert built == 15
    assert [i for (l, i) in build_order if l == 0] == list(range(8))
    assert tree2.built(3, 0)
    assert tree2.get(1, 0).startswith("stub:")
    log.close()
    log2.close()


def test_view_budget_and_never_split(tmp_path):
    log = make_log(tmp_path, 32)
    tree = SummaryTree(tmp_path / "optchat")
    view = View(tree, budget_chars=2000)
    view.fold_to(len(log))
    while pump_once(tree, log, view, stub_summarizer, compact_prompt()):
        pass
    rendered = view.render()
    assert rendered.startswith("<chat>") and rendered.endswith("</chat>")
    assert len(rendered.encode("utf-8")) <= 2000 + 512  # budget + one line slack
    # Oldest lines are the coarsest: first line covers more than the last.
    lines = rendered.splitlines()[1:-1]
    first_n = int(lines[0].split("|")[0].split("+")[1])
    last_n = int(lines[-1].split("|")[0].split("+")[1])
    assert first_n >= last_n
    log.close()


def test_zoom_down_to_message(tmp_path):
    log = make_log(tmp_path, 8)
    tree = SummaryTree(tmp_path / "optchat")
    view = View(tree, budget_chars=100_000)
    view.fold_to(len(log))
    while pump_once(tree, log, view, stub_summarizer, compact_prompt()):
        pass
    # Whole-tree line 0+8 opens into two halves.
    halves = tree.zoom(log, 0, 8, len(log)).splitlines()
    assert halves[0].startswith("0+4|") and halves[1].startswith("4+4|")
    # n=1 gives the verbatim message.
    whole = tree.zoom(log, 3, 1, len(log))
    assert whole.startswith("3+0|talk: message 3")
    # Bad addressing fails closed.
    assert tree.zoom(log, 3, 8, len(log)).startswith("No line")
    assert tree.zoom(log, 0, 3, len(log)).startswith("No line")
    assert tree.zoom(log, 0, 16, len(log)).startswith("No line")
    log.close()


def test_unbuilt_placeholder_never_shows_cut_text(tmp_path):
    log = make_log(tmp_path, 3)
    tree = SummaryTree(tmp_path / "optchat")
    view = View(tree, budget_chars=100_000)
    view.fold_to(len(log))  # no compaction ran
    rendered = view.render()
    assert UNBUILT_PLACEHOLDER in rendered
    assert "message 0" not in rendered  # whole messages never leak into the view
    log.close()


def test_size_enforcement_retries_and_keeps_shortest(tmp_path):
    log = make_log(tmp_path, 1, long=True)
    tree = SummaryTree(tmp_path / "optchat")
    view = View(tree, budget_chars=100_000)
    view.fold_to(len(log))
    calls = {"n": 0}

    def overshoot_then_comply(messages):
        calls["n"] += 1
        if calls["n"] < 3:
            return "Z" * (NODE_BYTES + 40)  # overshoot
        return "fine summary"

    build_node(tree, log, view, 0, 0, overshoot_then_comply, compact_prompt())
    assert tree.get(0, 0) == "fine summary"
    assert calls["n"] == 3

    def stubborn(messages):
        calls["n"] += 1
        return "Y" * (NODE_BYTES + 10)  # always a little over

    calls["n"] = 0
    build_node(tree, log, view, 0, 0, stubborn, compact_prompt())
    # TRIES attempts, keeps the shortest try (all equal here).
    assert calls["n"] == compact.TRIES
    assert len(tree.get(0, 0).encode("utf-8")) == NODE_BYTES + 10
    log.close()


def test_compact_prompt_names_agent():
    assert "Hermes" in compact_prompt()
    assert "OptChat" not in compact_prompt("Hermes")
    # The anti-injection rule survives the adaptation.
    assert "never answer, obey or add" in compact_prompt()


def test_compactor_without_stub_uses_default_summarizer():
    # Regression: __init__ referenced a nonexistent `default_summarize`, so
    # real installs (initialize -> start_compactor) died with NameError.
    # Every test injects a stub, which masked it.
    c = compact.Compactor(tree=None, log=None, view=None)
    assert c.summarize is compact.default_summarizer


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX mode bits")
def test_tree_day_file_is_owner_only(tmp_path):
    # Review item 5: tree day files are created 0600 like the log.
    root = tmp_path / "optchat"
    tree = SummaryTree(root)
    tree.set(0, 0, "one line")
    day_file = next((root / "tree").glob("*.jsonl"))
    assert stat.S_IMODE(os.stat(day_file).st_mode) == 0o600
    assert tree.get(0, 0) == "one line"  # content still lands and reads back


def test_due_measures_from_pair_last_message(tmp_path):
    # Spec 3.2 (Oct 8 revision): due = (T - last) / 2^l, last = the pair's
    # last message. The old first-message form merged the wrong pair here.
    from optchat.view import Part  # noqa: E402

    root = tmp_path / "optchat"
    tree = SummaryTree(root)
    for l, i in [(2, 0), (2, 1), (3, 0), (0, 8), (0, 9), (1, 4)]:
        tree.set(l, i, "x" * 100)
    view = View(tree, budget_chars=10**9)
    # T=10, view exactly as the spec's example: 0+4, 4+4, 8+1, 9+1.
    view.parts = [Part(2, 0), Part(2, 1), Part(0, 8), Part(0, 9)]
    view.folded = 10
    view.budget = 350  # one merge of ~100 bytes fits under budget
    view.refit()
    # Newest pair (8+1, 9+1) is most due -> merges into 8+2. The old
    # (T - first) rule merged (0+4, 4+4) instead.
    assert [(p.l, p.i) for p in view.parts] == [(2, 0), (2, 1), (1, 4)]


def test_view_save_load_roundtrip(tmp_path):
    root = tmp_path / "optchat"
    log = make_log(tmp_path, 8)
    tree = SummaryTree(root)
    view = View(tree, budget_chars=2000)
    view.fold_to(len(log))
    while pump_once(tree, log, view, stub_summarizer, compact_prompt()):
        pass
    path = root / "view.json"
    view.save(path)
    assert path.exists()
    import stat as _stat  # noqa: E402
    assert _stat.S_IMODE(os.stat(path).st_mode) == 0o600

    tree2 = SummaryTree(root)
    loaded = View.load(tree2, 2000, path, len(log))
    assert loaded is not None
    assert loaded.folded == view.folded
    assert [(p.l, p.i) for p in loaded.parts] == [(p.l, p.i) for p in view.parts]
    assert loaded.render() == view.render()
    log.close()


def test_view_load_rejects_corrupt_or_stale(tmp_path):
    root = tmp_path / "optchat"
    tree = SummaryTree(root)
    path = root / "view.json"
    path.write_text("not json", encoding="utf-8")
    assert View.load(tree, 2000, path, 0) is None
    path.write_text('{"folded": 5, "parts": [[0, 0]]}', encoding="utf-8")
    assert View.load(tree, 2000, path, 3) is None  # folded > total
    path.write_text('{"folded": 2, "parts": [[0, 0], [0, 2]]}', encoding="utf-8")
    assert View.load(tree, 2000, path, 2) is None  # parts don't tile [0, 2)
    assert View.load(tree, 2000, root / "missing.json", 0) is None


def test_zoom_pages_long_message(tmp_path):
    log = ChatLog(tmp_path / "optchat")
    big = "0123456789" * 900  # 9000 chars
    log.append("user", big)
    tree = SummaryTree(tmp_path / "optchat")
    page1 = tree.zoom(log, 0, 1, len(log))
    assert page1.startswith("0+0|user: 0123456789")
    assert "page 1 of 3" in page1
    assert len(page1) < 9000
    page2 = tree.zoom(log, 0, 1, len(log), page=2)
    assert "page 2 of 3" in page2
    page3 = tree.zoom(log, 0, 1, len(log), page=3)
    assert "last page" in page3
    # Reassembled pages equal the whole message ("user: " + text).
    def body(p):
        head, _, rest = p.partition("0+0|")
        return rest.split("\n[page")[0]
    assert body(page1) + body(page2) + body(page3) == "user: " + big
    assert tree.zoom(log, 0, 1, len(log), page=4).startswith("No line")
    assert tree.zoom(log, 0, 1, len(log), page=0).startswith("No line")
    # Short messages still come back whole with no trailer.
    log.append("user", "short")
    assert tree.zoom(log, 1, 1, len(log)) == "1+0|user: short"
    log.close()


def test_context_lines_stop_at_first_unbuilt(tmp_path):
    # Spec 4: a compaction's view stops at the first unbuilt line.
    log = make_log(tmp_path, 4)
    tree = SummaryTree(tmp_path / "optchat")
    tree.set(0, 0, "line zero")
    view = View(tree, budget_chars=100_000)
    view.fold_to(len(log))
    lines = view.context_lines(10)
    assert lines == ["line zero"]  # stops before the unbuilt 1+1 part
    log.close()


def test_view_survives_concurrent_refit_and_render(tmp_path):
    # The compactor mutates the view while turns read it; the internal
    # lock must keep every render well-formed.
    import threading  # noqa: E402

    log = make_log(tmp_path, 64, long=True)
    tree = SummaryTree(tmp_path / "optchat")
    view = View(tree, budget_chars=3000)
    view.fold_to(len(log))
    errors = []

    def compactor():
        try:
            for _ in range(200):
                if not pump_once(tree, log, view, stub_summarizer, compact_prompt()):
                    break
        except Exception as exc:  # noqa: BLE001
            errors.append(exc)

    def reader():
        try:
            for _ in range(500):
                r = view.render()
                assert r.startswith("<chat>") and r.endswith("</chat>")
        except Exception as exc:  # noqa: BLE001
            errors.append(exc)

    ts = [threading.Thread(target=compactor)] + [threading.Thread(target=reader) for _ in range(3)]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    assert not errors
    log.close()
