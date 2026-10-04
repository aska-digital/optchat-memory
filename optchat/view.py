"""The view (OptChat spec section 5).

The view is a list of tree nodes ("parts") that tiles the whole chat
``[0, T)``, oldest first. It is what the agent's recall sees.

Rendering: one line per part, ``id+n|text`` (newlines shown as spaces),
inside ``<chat>`` tags. No dates on lines (the agent calls ``date(id)``
when it needs one). **The view never holds a whole message** — only
summaries; a part whose node isn't built yet renders as
``id+1|(not summarized yet: zoom it)``.

How it changes (spec section 5.2 — the easiest part to get wrong):
on a new message, append ``part(0, i)`` and ``fit()``; on a node built,
``fit()``. ``fit()`` repeatedly merges the *most due* adjacent pair —
oldest relative to its size, ``due = (T - start) / 2**(l+2)`` — among
pairs whose parent is already built. Never split: once merged, a part
stays merged; the view only appends at the end and coarsens. A line at
level ``l`` changes about once every ``2**l`` messages, so consecutive
views share most of their bytes (this is what made the design cheap;
here it keeps prefetch output stable and small).

Deliberately NOT done (spec section 5.3): showing recent messages whole
(one big tool result would permanently erase old detail), or cut text
for unsummarized messages (the agent would act on half a message).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from .tree import SummaryTree

UNBUILT_PLACEHOLDER = "(not summarized yet: zoom it)"


@dataclass
class Part:
    l: int  # level
    i: int  # index within the level

    def address(self) -> tuple[int, int]:
        return SummaryTree.address(self.l, self.i)


class View:
    """Incrementally maintained tiling. ``fold_to`` / ``refit`` keep it live;
    the provider serializes all access."""

    def __init__(self, tree: SummaryTree, budget_chars: int) -> None:
        self.tree = tree
        self.budget = budget_chars
        self.parts: list[Part] = []
        self.folded = 0  # messages folded so far
        self._render_cache: Optional[str] = None

    # -- incremental maintenance ----------------------------------------

    def _part_size(self, part: Part) -> int:
        text = self.tree.get(part.l, part.i)
        if text is None:
            text = UNBUILT_PLACEHOLDER
        return len(text.encode("utf-8"))

    def _size(self) -> int:
        return sum(self._part_size(p) for p in self.parts)

    def fold_to(self, total: int) -> None:
        """Append ``part(0, i)`` for every new message, then fit."""
        if total < self.folded:
            # Log truncation should not happen (append-only); refold cleanly.
            self.parts = []
            self.folded = 0
        while self.folded < total:
            self.parts.append(Part(0, self.folded))
            self.folded += 1
        self.refit()

    def refit(self) -> None:
        """Merge the most due pair until under budget. Parents enter only once built."""
        self._render_cache = None
        total = self.folded
        size = self._size()
        while size > self.budget:
            best = -1
            best_due = -1.0
            for idx in range(len(self.parts) - 1):
                a, b = self.parts[idx], self.parts[idx + 1]
                if a.l != b.l or b.i != a.i + 1 or a.i % 2 != 0:
                    continue
                if not self.tree.built(a.l + 1, a.i // 2):
                    continue
                start = a.i * (1 << a.l)
                due = (total - start) / float(1 << (a.l + 2))
                if due > best_due:
                    best_due = due
                    best = idx
            if best < 0:
                break  # wait until a parent is built
            a = self.parts[best]
            self.parts[best : best + 2] = [Part(a.l + 1, a.i // 2)]
            size = self._size()

    # -- compactor support (spec section 4.1, rule 3) ---------------------

    def first_unbuilt_message(self) -> int:
        """First message whose view line is not a built summary (== total when all built)."""
        for part in self.parts:
            if not self.tree.built(part.l, part.i):
                return part.i * (1 << part.l)
        return self.folded

    # -- rendering --------------------------------------------------------

    def render(self) -> str:
        """The ``<chat>`` block the agent sees. Cached until the next refit."""
        if self._render_cache is not None:
            return self._render_cache
        lines = ["<chat>"]
        for part in self.parts:
            msg_id, n = part.address()
            text = self.tree.get(part.l, part.i)
            if text is None:
                text = UNBUILT_PLACEHOLDER
            flat = text.replace("\n", " ")
            lines.append(f"{msg_id}+{n}|{flat}")
        lines.append("</chat>")
        self._render_cache = "\n".join(lines)
        return self._render_cache
