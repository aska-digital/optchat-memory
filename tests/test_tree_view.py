"""Tests for the summary tree, view fold/fit, zoom, and the compactor pump.

Uses a stub summarizer (no LLM): deterministic one-line summaries, with
an overshoot-then-comply variant for the size-enforcement test.
"""

import sys
from pathlib import Path

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
