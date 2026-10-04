# plugin-catalog: add optchat-memory (OptChat-style endless memory provider)

## What

Adds `optchat-memory` to the plugin catalog: a standalone memory-provider
plugin implementing an OptChat-style endless memory, which is an
append-only verbatim log plus a binary summary tree, surfaced as recall
through the `MemoryProvider` ABC. Design follows Victor Taelin's OptChat spec.

Per the in-tree policy (no new memory providers under `plugins/`), this
ships as a standalone repo installed via the catalog. The plugin code
itself is not part of this diff. Only the catalog entry is.

## How it works

- `sync_turn` appends each turn's messages to
  `<home>/optchat/main/YYYY-MM-DD.jsonl` (kinds `user` / `talk` / `tool` /
  `echo`). Tool results are capped at 30k chars. Every line is fsync'd.
  One process holds the write lock, and torn lines are skipped at load.
  The log is never edited or deleted.
- A background compactor builds the binary summary tree in order
  (`tree/YYYY-MM-DD.jsonl`) through `call_llm(task="optchat_compact")`,
  so operators can pin a cheap model under `auxiliary:` in config.yaml.
  Free nodes (short messages, small merges) need no model call. Node
  addressing is `id+n`: first message id plus span, a power of two.
- `prefetch()` returns the current *view*: a fixed-budget tiling of the
  whole log, oldest first, with recent messages one line each and older
  ones coarser with age. Unbuilt parts render as a zoom prompt. Whole
  messages never enter the view, and no cut text is ever shown.
- Tools `optchat_zoom(id, n)` and `optchat_date(id)` open any line down to
  the verbatim message. `system_prompt_block()` is a byte-stable VIEW_DOC,
  so the prompt-caching invariant holds.

## Deliberate scoping

Full OptChat replaces the agent's turn loop (a fresh call per message
with the computed view). A plugin cannot do that without breaking the
prompt-caching invariant and the plugins-never-touch-core rule, so this
is the memory *substrate*: the OptMem-shaped subset the spec's own
history describes, namely the log plus the tree as *recall* rather than
as the conversation itself. Recent turns stay in native context and in
Hermes's own compression pipeline, while the tree is the deep past. The
plugin repo's `DESIGN.md` documents this trade explicitly, including what
the agent gives up versus full OptChat (no fresh-context-per-turn) and
what it gains (months of verbatim, zoomable history with zero core changes).

## Notes for reviewers

- `sha` is currently a placeholder. It will be replaced with the publish
  commit's full 40-hex SHA before this PR is opened (admission rule 2).
  The plugin repo is not published yet. This PR is a draft for review of
  the entry shape and the design, with 21 passing tests in the repo.
- No secrets, no network calls, no new env vars: local files only, plus
  the operator's existing auxiliary model route for compaction.
- `capabilities.provides_tools` lists the two memory tools the provider
  registers through `get_tool_schemas`. No hooks or middleware.
- Background work starts via `spawn_context_thread`. All state is keyed
  by `hermes_home_key()` per call, never cached from `initialize()`.

> Automated posting by agentic team with human oversight.
