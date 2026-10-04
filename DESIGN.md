# optchat-memory — Design (DRAFT)

## Why a standalone repo, not an in-tree PR

Two Hermes policies decide the shape:

1. **No new in-tree memory providers** (`plugins/AGENTS.md`, May 2026).
   `plugins/memory/` is closed; new backends ship as standalone repos
   implementing the `MemoryProvider` ABC and enter through the
   `plugin-catalog/`. This repo is that standalone backend.
2. **No runtime overrides of Hermes core** (catalog admission rule 9).
   A listed plugin extends Hermes only through public surfaces.

## The honest scoping: OptMem-subset, not full OptChat

OptChat's full design *is the turn loop*: every user message starts a
fresh model call whose context is `[system][view][message]`, and the
summary tree is the *only* memory the agent has. A Hermes plugin cannot
do that, and must not try:

- Per-conversation prompt caching is sacred (`agent/AGENTS.md`); the
  system prompt is byte-stable for the life of a conversation, and
  context compression is the one sanctioned cache break. Rebuilding the
  context from a computed view every turn would torch the cache on
  every message.
- "Plugins never touch core" (`plugins/AGENTS.md`): the turn loop in
  `agent/turn_*.py` is off-limits.

So this provider implements the OptChat **memory substrate** — the part
of the lineage that predates it as OptMem (which the spec itself
describes as "an append-only log of short notes plus a summary tree,
which an agent reads at the start of each session"). Concretely:

| OptChat spec | Here |
|---|---|
| Log: every message verbatim, append-only | `store.py`: per-day JSONL under `<home>/optchat/main/`, fsync per line, single-writer fcntl lock, torn lines skipped |
| Tree: binary summaries, `id+n` addressing, free nodes | `tree.py`: same addressing; free-node rule for level 0 *and* merges |
| Compactor: in-order pump, context-aware summaries, SCALE + retry | `compact.py`: same pump rules; `call_llm(task="optchat_compact")` so the model is user-pinned; sequential (one node at a time) — turns never wait on it here, so the spec's JOBS-way parallelism is deferred |
| View: fold + most-due-pair fit, summaries only, never cut text | `view.py`: same fold/fit; unbuilt parts render `(not summarized yet: zoom it)` |
| Turn loop: fresh call per message with the view | **Not taken.** `prefetch()` returns the view as per-turn `<memory-context>` recall instead; recent turns stay in native context |
| `zoom` / `date` tools | `optchat_zoom` / `optchat_date` via `get_tool_schemas` / `handle_tool_call` |
| System prompt: MASTER + VIEW_DOC | `system_prompt_block()` returns a byte-stable VIEW_DOC only; Hermes owns the rest |

What the agent loses vs full OptChat: instructions "sticking" without
files still works (user corrections survive up the tree and are
recalled), but there is no fresh-context-per-turn and no immunity to
context rot *within* a long session — Hermes's own compression still
governs that. What it gains: months of verbatim, zoomable history
without touching the core loop or the cache invariant.

## Concurrency and profile scope

- One process may serve several profiles: all state (`ChatLog`,
  `SummaryTree`, `View`, `Compactor`) is keyed by `hermes_home_key()`,
  resolved per call; `initialize()` never caches a home as "the" home.
- Background work starts via `spawn_context_thread` (contextvars are
  copied; an unbound worker would silently land on the wrong profile).
- `sync_turn` skips writes for non-primary `agent_context`
  (cron / subagent / flush).
- The log takes a non-blocking exclusive lock at open: a second process
  on the same profile fails loudly (`LogLocked`) instead of
  interleaving ids — the spec's Unix-socket rule, via fcntl.

## Failure posture

- The compactor is best-effort and sequential: a failed node logs and
  retries; the view serves whatever is built, with placeholders for the
  rest. Prefetch never blocks on compaction.
- `prefetch` is fast by construction: it renders the cached view
  (refreshed incrementally), never calls the model.
- Summarizer failures raise inside `build_node` and are caught by the
  compactor loop (warn + 10s backoff), never by a turn.

## Tests

`tests/`: log durability (torn lines, reload, lock contention), tree
addressing and zoom (including failure-closed addressing), pump
ordering (level 0 strictly in message order), view budget behavior
(oldest coarsest, never split, no whole messages in the view),
size-enforcement retries, and provider wiring against the real
`MemoryProvider` ABC (kinds mapping, echo capping, idempotent tails,
non-primary skip, session-switch rebinding, config round-trip).
The summarizer is a stub; no test hits the network.
