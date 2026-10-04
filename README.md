# optchat-memory

OptChat-style endless memory for [Hermes Agent](https://github.com/NousResearch/hermes-agent):
an append-only verbatim log plus a binary summary tree, exposed as *recall*
through Hermes's `MemoryProvider` ABC. Based on Victor Taelin's
[OptChat spec](https://gist.github.com/VictorTaelin/91837951a5ce5b38f341ec1ba1df6449).

## What it does

- **Logs everything, forever.** Every turn's user messages, agent replies,
  tool calls and tool results are appended to
  `<hermes_home>/optchat/main/YYYY-MM-DD.jsonl` (one JSON object per line,
  fsync'd; torn lines skipped at load). Nothing is ever edited or deleted.
- **Compresses in the background.** A compactor thread builds a binary tree
  of one-line summaries (`tree/YYYY-MM-DD.jsonl`): each message becomes a
  line, adjacent lines merge pairwise, up the tree. Short messages stay
  verbatim; long ones are summarized with a cheap auxiliary model
  (`call_llm(task="optchat_compact")` — pin a model under `auxiliary:` in
  config.yaml).
- **Recalls through the view.** `prefetch()` returns a fixed-budget tiling
  of the whole log, oldest first: recent messages one line each, older
  ones coarser with age. The agent gets two tools to navigate it:
  `optchat_zoom(id, n)` opens any line down to the verbatim message,
  `optchat_date(id)` gives a message's timestamp.

Recent turns stay in Hermes's native conversation context; the tree is
the deep past. See `DESIGN.md` for the honest scoping (what a plugin
can and cannot take from the OptChat spec).

## Install

Via the Hermes plugin catalog:

```
hermes plugins install optchat-memory
```

or drop this package directory into `$HERMES_HOME/plugins/optchat/`.
Then activate:

```
hermes memory setup   # choose optchat
```

or set `memory.provider: optchat` in config.yaml.

## Configuration (`hermes memory setup` fields)

| key | default | meaning |
|---|---|---|
| `view_budget_chars` | 16000 | Prefetch view budget (~4 chars/token). Larger = deeper recall per turn. |
| `node_bytes` | 512 | Target size of one summary-tree line, in bytes. |
| `compact_enabled` | true | Run the background compactor (needs an auxiliary model route). |

Non-secret settings live in `<hermes_home>/optchat/config.json`. To use a
cheap model for compaction, pin the task route in config.yaml:

```yaml
auxiliary:
  optchat_compact:
    provider: <your-cheap-provider>
    model: <model>
```

## Layout

```
optchat/
  __init__.py   register(ctx) — memory-provider discovery entry point
  provider.py   OptChatMemoryProvider (the ABC wiring)
  store.py      append-only JSONL log
  tree.py       binary summary tree + zoom addressing
  view.py       view fold / most-due-pair fit / rendering
  compact.py    background compactor (COMPACT prompt, pump, retries)
tests/          pytest suite (stubbed summarizer; ABC wired against a real checkout)
```

## License

Apache-2.0
