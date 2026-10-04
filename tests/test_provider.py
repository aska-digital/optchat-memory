"""Tests for the MemoryProvider wiring (optchat/provider.py).

Imports the real ``agent.memory_provider`` ABC from the hermes-agent
checkout; skipped when the checkout isn't on disk.
"""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

HERMES_CHECKOUT = Path.home() / "workspace" / "hermes-agent-draft"
if HERMES_CHECKOUT.is_dir():
    sys.path.insert(0, str(HERMES_CHECKOUT))

try:
    from optchat.provider import OptChatMemoryProvider, VIEW_DOC  # noqa: E402
    from agent.memory_provider import MemoryProvider  # noqa: E402
except ImportError:
    pytest.skip("hermes-agent checkout not available", allow_module_level=True)


@pytest.fixture
def provider(tmp_path, monkeypatch):
    home = tmp_path / "hermes-home"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    p = OptChatMemoryProvider()
    assert isinstance(p, MemoryProvider)
    assert p.name == "optchat"
    assert p.is_available()
    p.save_config({"compact_enabled": False}, str(home))  # no background thread in tests
    p.initialize("sess-1", hermes_home=str(home), platform="cli", agent_context="primary")
    yield p, str(home)
    p.shutdown()


def test_sync_turn_logs_kinds_and_only_new_tail(provider):
    p, _ = provider
    messages = [
        {"role": "system", "content": "be helpful"},  # never logged
        {"role": "user", "content": "deploy it"},
        {"role": "assistant", "content": "on it",
         "tool_calls": [{"function": {"name": "deploy", "arguments": '{"x": 1}'}}]},
        {"role": "tool", "content": "done " * 100},
    ]
    p.sync_turn("deploy it", "on it", session_id="sess-1", messages=messages)
    p.sync_turn("deploy it", "on it", session_id="sess-1", messages=messages)  # idempotent
    state = p._state()
    assert len(state.log) == 4  # system skipped, no double-log
    kinds = [m["kind"] for m in state.log.iter_all()]
    assert kinds == ["user", "talk", "tool", "echo"]
    tool = state.log.read(2)
    assert tool["text"] == 'deploy({"x": 1})'


def test_echo_capped_at_cap_chars(provider):
    p, _ = provider
    big = "E" * 60_000
    p.sync_turn("q", "a", session_id="sess-1",
                messages=[{"role": "user", "content": "q"},
                          {"role": "tool", "content": big}])
    state = p._state()
    echo = state.log.read(1)
    assert len(echo["text"]) < 60_000
    assert "optchat: cut" in echo["text"]


def test_non_primary_context_does_not_write(provider):
    p, home = provider
    p.initialize("cron-1", hermes_home=home, platform="cli", agent_context="cron")
    before = len(p._state().log)
    p.sync_turn("x", "y", session_id="cron-1",
                messages=[{"role": "user", "content": "x"}])
    assert len(p._state().log) == before


def test_prefetch_returns_view_and_zoom_tool_roundtrips(provider):
    p, _ = provider
    p.sync_turn("remember the ship name", "noted", session_id="sess-1",
                messages=[{"role": "user", "content": "remember the ship name"},
                          {"role": "assistant", "content": "noted"}])
    # Compaction is disabled in tests; drive the pump manually with a stub.
    from optchat.compact import compact_prompt, pump_once
    state = p._state()
    while pump_once(state.tree, state.log, state.view,
                    lambda msgs: "stub summary", compact_prompt()):
        pass
    view = p.prefetch("ship", session_id="sess-1")
    assert "<chat>" in view and "</chat>" in view
    assert "remember the ship name" in view  # short message: verbatim free node
    assert p.recall_status().count == 2

    out = json.loads(p.handle_tool_call("optchat_zoom", {"id": 0, "n": 1}))
    assert out["result"].startswith("0+0|user: remember the ship name")
    out = json.loads(p.handle_tool_call("optchat_zoom", {"id": 0, "n": 2}))
    assert out["result"].splitlines() == [
        "0+1|user: remember the ship name",
        "1+1|talk: noted",
    ]
    out = json.loads(p.handle_tool_call("optchat_zoom", {"id": 0, "n": 4}))
    assert "No line" in out["result"]  # id+n must not exceed total
    out = json.loads(p.handle_tool_call("optchat_date", {"id": 1}))
    assert "2026" in out["result"]
    with pytest.raises(NotImplementedError):  # ABC contract: manager wraps it as tool_error
        p.handle_tool_call("bogus", {})


def test_system_prompt_block_is_byte_stable(provider):
    p, _ = provider
    assert p.system_prompt_block() == VIEW_DOC
    assert p.system_prompt_block() == p.system_prompt_block()


def test_session_switch_does_not_relog_history(provider):
    p, _ = provider
    msgs = [{"role": "user", "content": "hi"}]
    p.sync_turn("hi", "", session_id="sess-1", messages=msgs)
    assert len(p._state().log) == 1
    p.on_session_switch("sess-2")
    p.sync_turn("hi", "", session_id="sess-2", messages=msgs)
    assert len(p._state().log) == 1  # history not re-logged into the new session
    p.sync_turn("hi2", "", session_id="sess-2",
                messages=msgs + [{"role": "user", "content": "hi2"}])
    assert len(p._state().log) == 2


def test_config_roundtrip_and_validation(provider):
    p, home = provider
    schema = p.get_config_schema()
    assert {f["key"] for f in schema} >= {"view_budget_chars", "node_bytes", "compact_enabled"}
    p.save_config({"view_budget_chars": 8000, "bogus_key": 1}, home)
    raw = json.loads((Path(home) / "optchat" / "config.json").read_text())
    assert raw["view_budget_chars"] == 8000
    assert "bogus_key" not in raw
