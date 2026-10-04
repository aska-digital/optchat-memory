"""OptChat-style memory as a Hermes MemoryProvider.

What this is — and honestly isn't
----------------------------------
OptChat's full design replaces the agent's turn loop: every user message
starts a *fresh* model call whose context is ``[system][view][message]``,
with the summary tree as the *only* memory. A Hermes plugin cannot do
that — and must not try: per-conversation prompt caching is sacred here
(``agent/AGENTS.md``), and the core turn loop is off-limits to plugins
(``plugins/AGENTS.md``: "Plugins never touch core").

So this provider implements the OptChat *memory substrate* — the part
that predates it as OptMem and that the spec's own history describes:
an append-only verbatim log plus a binary summary tree, surfaced as
*recall* through the MemoryProvider ABC:

- ``sync_turn`` appends each turn's messages to the log (kinds ``user`` /
  ``talk`` / ``tool`` / ``echo``, tool results capped at ``CAP`` chars).
- A background compactor (``compact.py``) builds the tree in order with
  a cheap auxiliary model (``call_llm(task="optchat_compact")`` — pin a
  model under ``auxiliary:`` in config.yaml).
- ``prefetch`` returns the current *view*: a fixed-budget tiling of the
  whole log, oldest first, recent messages one line each, older ones
  coarser — the agent's long-term recall for the turn.
- ``optchat_zoom(id, n)`` / ``optchat_date(id)`` tools let the agent
  open any line down to the verbatim message.
- ``system_prompt_block`` is a byte-stable VIEW_DOC telling the agent
  how to read the view and to zoom before acting on a summary.

Recent turns stay in native conversation context (and in Hermes's own
compression pipeline); the tree is the deep past. Nothing here mutates
past context or rebuilds the system prompt mid-conversation.
"""

from __future__ import annotations

import json
import logging
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional

from agent.memory_provider import (
    MemoryProvider,
    RecallStatus,
    is_core_memory_provider,
    spawn_context_thread,
)

from .compact import Compactor, compact_prompt
from .store import ChatLog, LogLocked
from .tree import SummaryTree
from .view import View

logger = logging.getLogger(__name__)

PROVIDER_NAME = "optchat"

# Max chars of one tool result kept in the log (head + tail), spec constant CAP.
CAP_CHARS = 30_000

DEFAULT_CONFIG = {
    "view_budget_chars": 16000,  # prefetch view budget (~4k tokens)
    "node_bytes": 512,  # spec NODE
    "compact_enabled": True,
}

VIEW_DOC = """The <memory-context> block below holds this agent's long-term memory: the whole conversation history, oldest first, as one-line summaries. Each line is
  id+n|text   the n messages from id on, summarized (newlines shown as spaces)
A summary tags each item with its kind: user (the user's words), talk (the agent's replies), tool (tool calls), echo (their results), note (imported history). A short message is its own line, word for word. Recent lines cover one message each; the older the messages, the more a line covers.
A message not summarized yet shows as "(not summarized yet: zoom it)". No message appears in full, not even the last ones.
Navigating: optchat_zoom(id, n) opens line id+n into the two lines of n/2 messages it was made from; optchat_zoom(id, 1) gives message id in full. Zoom whenever a summary only mentions something you need, such as what your last reply said, a decision, a past attempt or where a file is, before you act, guess or ask. optchat_date(id) gives the date and time of message id."""


def _content_text(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for part in content:
            if isinstance(part, dict):
                if part.get("type") == "text":
                    parts.append(str(part.get("text", "")))
                elif "text" in part:
                    parts.append(str(part["text"]))
            elif isinstance(part, str):
                parts.append(part)
        return "".join(parts)
    return str(content)


def _cap_echo(text: str, cap: int = CAP_CHARS) -> str:
    if len(text) <= cap:
        return text
    head = cap * 3 // 4
    tail = cap - head
    return f"{text[:head]}\n...[optchat: cut {len(text) - cap} chars]...\n{text[-tail:]}"


class _HomeState:
    """All per-profile state. Keyed by hermes_home_key(); never shared across profiles."""

    def __init__(self, hermes_home: str, config: dict) -> None:
        self.hermes_home = hermes_home
        self.config = config
        self.root = Path(hermes_home) / "optchat"
        self.lock = threading.RLock()
        self.log = ChatLog(self.root)
        self.tree = SummaryTree(self.root)
        self.view = View(self.tree, budget_chars=int(config["view_budget_chars"]))
        with self.lock:
            self.view.fold_to(len(self.log))
        self.compactor: Optional[Compactor] = None
        # (session_id) -> {"last_idx": int, "primary": bool}
        self.sessions: Dict[str, Dict[str, Any]] = {}
        self._last_prefetch_lines = 0

    def start_compactor(self) -> None:
        if not self.config.get("compact_enabled", True):
            return
        self.compactor = Compactor(
            self.tree, self.log, self.view, node_bytes=int(self.config["node_bytes"])
        )
        self.compactor.start()

    def stop(self) -> None:
        if self.compactor:
            self.compactor.stop()
            self.compactor = None
        self.log.close()


class OptChatMemoryProvider(MemoryProvider):
    """MemoryProvider ABC implementation. One instance per process; state is per home."""

    def __init__(self) -> None:
        self._homes: Dict[str, _HomeState] = {}
        self._homes_lock = threading.RLock()

    # -- helpers ---------------------------------------------------------

    @staticmethod
    def _config_path(hermes_home: str) -> Path:
        return Path(hermes_home) / "optchat" / "config.json"

    def _load_config(self, hermes_home: str) -> dict:
        cfg = dict(DEFAULT_CONFIG)
        try:
            raw = self._config_path(hermes_home).read_text(encoding="utf-8")
            cfg.update(json.loads(raw))
        except (OSError, ValueError):
            pass
        return cfg

    def _home_key(self, hermes_home: Optional[str] = None) -> str:
        from hermes_constants import hermes_home_key

        return hermes_home_key(hermes_home)

    def _state(self, hermes_home: Optional[str] = None) -> Optional[_HomeState]:
        """State for the calling profile (ContextVar-bound home when omitted)."""
        try:
            key = self._home_key(hermes_home)
        except Exception:
            key = None
        with self._homes_lock:
            if key and key in self._homes:
                return self._homes[key]
            if len(self._homes) == 1:
                return next(iter(self._homes.values()))
            return None

    def _session(self, state: _HomeState, session_id: str) -> Dict[str, Any]:
        sess = state.sessions.get(session_id)
        if sess is None:
            # Unknown session (e.g. provider loaded mid-conversation): backfill
            # from the start so the log captures the history it missed.
            sess = {"last_idx": 0, "primary": True}
            state.sessions[session_id] = sess
        return sess

    # -- ABC: core lifecycle ----------------------------------------------

    @property
    def name(self) -> str:
        return PROVIDER_NAME

    def is_available(self) -> bool:
        # Local files only: no credentials, no network. Nothing to check.
        return True

    def initialize(self, session_id: str, **kwargs) -> None:
        hermes_home = kwargs.get("hermes_home")
        if not hermes_home:
            raise RuntimeError("optchat: initialize() requires hermes_home")
        key = self._home_key(hermes_home)
        with self._homes_lock:
            if key not in self._homes:
                try:
                    state = _HomeState(hermes_home, self._load_config(hermes_home))
                except LogLocked as exc:
                    raise RuntimeError(str(exc)) from exc
                state.start_compactor()
                self._homes[key] = state
            state = self._homes[key]
        agent_context = kwargs.get("agent_context", "primary")
        with state.lock:
            state.sessions[session_id or "default"] = {
                "last_idx": len(state.log),
                "primary": agent_context == "primary",
            }

    def system_prompt_block(self) -> str:
        # Static and byte-stable for the life of a conversation (caching invariant).
        return VIEW_DOC

    def prefetch(self, query: str, *, session_id: str = "") -> str:
        state = self._state()
        if state is None:
            return ""
        with state.lock:
            state.view.fold_to(len(state.log))
            state.view.refit()
            if len(state.log) == 0:
                return ""
            body = state.view.render()
            self._last_prefetch_lines = body.count("\n")
        header = (
            f"[optchat memory: {len(state.log)} messages remembered; "
            "older lines are coarser summaries — zoom before acting on one]"
        )
        return header + "\n" + body

    def queue_prefetch(self, query: str, *, session_id: str = "") -> None:
        state = self._state()
        if state and state.compactor:
            state.compactor.notify()

    def recall_status(self) -> Optional[RecallStatus]:
        state = self._state()
        if state is None or len(state.log) == 0:
            return None
        return RecallStatus(provider_label="optchat", count=len(state.log))

    def sync_turn(
        self,
        user_content: str,
        assistant_content: str,
        *,
        session_id: str = "",
        messages: Optional[List[Dict[str, Any]]] = None,
        turn_author: Optional[Dict[str, Any]] = None,
    ) -> None:
        state = self._state()
        if state is None:
            return
        sid = session_id or "default"
        with state.lock:
            sess = self._session(state, sid)
            if not sess.get("primary", True):
                return  # cron/subagent/flush contexts don't write
            if messages:
                entries = list(self._entries_from_messages(messages[sess["last_idx"]:]))
                sess["last_idx"] = len(messages)
            else:
                entries = []
                if (user_content or "").strip():
                    entries.append(("user", user_content))
                if (assistant_content or "").strip():
                    entries.append(("talk", assistant_content))
            for kind, text in entries:
                state.log.append(kind, text, session=sid)
            if entries and state.compactor:
                state.compactor.notify()

    @staticmethod
    def _entries_from_messages(messages: List[Dict[str, Any]]):
        for m in messages:
            role = m.get("role")
            if role == "user":
                text = _content_text(m.get("content"))
                if text.strip():
                    yield ("user", text)
            elif role == "assistant":
                text = _content_text(m.get("content"))
                if text.strip():
                    yield ("talk", text)
                for tc in m.get("tool_calls") or []:
                    fn = (tc.get("function") or {}) if isinstance(tc, dict) else {}
                    name = fn.get("name", "?")
                    args = fn.get("arguments", "")
                    yield ("tool", f"{name}({args})")
            elif role == "tool":
                text = _content_text(m.get("content"))
                if text.strip():
                    yield ("echo", _cap_echo(text))
            # "system" rows are the harness's, not the chat: never logged.

    def get_tool_schemas(self) -> List[Dict[str, Any]]:
        return [
            {
                "name": "optchat_zoom",
                "description": (
                    "Open the memory line id+n into the two lines of n/2 messages it was "
                    "made from; n = 1 gives message id in full. Zoom before acting on a summary."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "id": {"type": "integer", "description": "First message id of the line."},
                        "n": {"type": "integer", "description": "How many messages the line covers (power of 2)."},
                    },
                    "required": ["id", "n"],
                },
            },
            {
                "name": "optchat_date",
                "description": "The date and time of memory message id.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "id": {"type": "integer", "description": "Message id."},
                    },
                    "required": ["id"],
                },
            },
        ]

    def handle_tool_call(self, tool_name: str, args: Dict[str, Any], **kwargs) -> str:
        state = self._state()
        if state is None:
            return json.dumps({"error": "optchat memory is not initialized for this profile"})
        with state.lock:
            if tool_name == "optchat_zoom":
                try:
                    msg_id = int(args["id"])
                    n = int(args["n"])
                except (KeyError, TypeError, ValueError):
                    return json.dumps({"error": "optchat_zoom needs integer id and n"})
                return json.dumps({"result": state.tree.zoom(state.log, msg_id, n, len(state.log))})
            if tool_name == "optchat_date":
                try:
                    msg_id = int(args["id"])
                except (KeyError, TypeError, ValueError):
                    return json.dumps({"error": "optchat_date needs integer id"})
                return json.dumps({"result": state.tree.message_date(state.log, msg_id)})
        raise NotImplementedError(f"Provider {self.name} does not handle tool {tool_name}")

    def shutdown(self) -> None:
        with self._homes_lock:
            homes = list(self._homes.values())
            self._homes.clear()
        for state in homes:
            with state.lock:
                state.stop()

    # -- ABC: optional hooks -----------------------------------------------

    def on_session_switch(self, new_session_id: str, *, parent_session_id: str = "",
                          reset: bool = False, rewound: bool = False, **kwargs) -> None:
        state = self._state()
        if state is None:
            return
        with state.lock:
            # A new conversation continues the same endless log; don't re-log history.
            state.sessions[new_session_id] = {"last_idx": len(state.log), "primary": True}

    def on_pre_compress(self, messages: List[Dict[str, Any]]) -> str:
        # The tree already is the long-term store; nothing extra to distill here.
        return ""

    def identity_signature(self) -> Dict[str, Any]:
        return {"optchat.provider": PROVIDER_NAME}

    # -- ABC: setup ----------------------------------------------------------

    def get_config_schema(self) -> List[Dict[str, Any]]:
        return [
            {
                "key": "view_budget_chars",
                "description": "Prefetch view budget in chars (~4 chars/token). Larger = deeper recall, more context per turn.",
                "type": "integer", "default": DEFAULT_CONFIG["view_budget_chars"],
                "minimum": 2000, "maximum": 120000, "step": 1000,
            },
            {
                "key": "node_bytes",
                "description": "Target size of one summary-tree line in bytes (OptChat NODE).",
                "type": "integer", "default": DEFAULT_CONFIG["node_bytes"],
                "minimum": 128, "maximum": 2048, "step": 64,
            },
            {
                "key": "compact_enabled",
                "description": "Run the background summary-tree compactor (needs an auxiliary model route).",
                "type": "boolean", "default": True, "choices": ["true", "false"],
            },
        ]

    def save_config(self, values: Dict[str, Any], hermes_home: str) -> None:
        path = self._config_path(hermes_home)
        path.parent.mkdir(parents=True, exist_ok=True)
        current: dict = {}
        try:
            current = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            pass
        merged = {**current, **{k: v for k, v in values.items() if k in DEFAULT_CONFIG}}
        path.write_text(json.dumps(merged, indent=2) + "\n", encoding="utf-8")
        path.chmod(0o600)

    def backup_paths(self) -> List[str]:
        return []  # state lives under HERMES_HOME/optchat; backup covers it.
