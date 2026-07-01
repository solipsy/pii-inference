"""Post-processing helpers to reduce clutter in classification results.

The engine emits token-level spans, so a single real-world entity often arrives
as several adjacent spans (``FIRSTNAME`` + ``LASTNAME`` -> a person, ``$`` +
``400 billion`` -> an amount). :func:`merge_entities` folds these into single
spans, and :func:`dedupe_entities` collapses repeats. Both operate on any object
exposing ``start``/``end``/``score``/``label`` (the C ``Entity`` or a
:class:`Span`), so they can be chained.

Merging is separator-aware on purpose: ``Indiana. South Korea`` are two
``STATE`` spans across a sentence boundary and must *not* merge, while
``Executive Assistants`` (two ``OCCUPATION`` spans joined by a space) should.
Only an allow-list of ordered label pairs, across a small set of connecting
characters, is ever merged.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Iterable, Protocol, Sequence


class _EntityLike(Protocol):
    start: int
    end: int
    score: float
    label: str


# Name components, in the order they typically appear. Any run of these joined
# by a name separator collapses into a single PERSON.
NAME_LABELS = frozenset({"PREFIX", "FIRSTNAME", "MIDDLENAME", "LASTNAME", "SUFFIX"})
MONEY_LABELS = frozenset({"CURRENCYSYMBOL", "AMOUNT", "CURRENCYCODE"})
# Tier-2 "location grouping" (opt-in via address=True).
ADDRESS_LABELS = frozenset(
    {"BUILDINGNUMBER", "STREET", "SECONDARYADDRESS", "CITY", "STATE", "COUNTY", "ZIPCODE", "COUNTRY"}
)

# Allowed connecting characters between spans, per family. None of these permit
# a newline or a sentence-ending ". " — those signal genuinely separate spans.
_NAME_SEP = re.compile(r"[ '\-.]{1,2}\Z")   # space, apostrophe, hyphen, initial dot
_MONEY_SEP = re.compile(r" ?\Z")            # adjacent or single space
_WS_SEP = re.compile(r" +\Z")               # whitespace only (same-label runs)
_ADDR_SEP = re.compile(r"[ ,]{1,2}\Z")      # space and/or comma


@dataclass(frozen=True)
class Span:
    """A (possibly merged) entity span. Mirrors the C ``Entity`` interface.

    ``start``/``end`` are UTF-8 byte offsets. ``parts`` holds the original
    entities that were merged (a single-element tuple for un-merged spans), so
    the token-level detail is always recoverable.
    """

    start: int
    end: int
    score: float
    label: str
    parts: tuple = field(default_factory=tuple, repr=False)

    def text(self, source: str) -> str:
        """Return the matched substring given the original text."""
        return source.encode("utf-8")[self.start : self.end].decode("utf-8", "replace")

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"Span(start={self.start}, end={self.end}, score={self.score:.3f}, label={self.label!r})"


def _pair_family(prev_label: str, next_label: str, sep: str, address: bool) -> str | None:
    """Family a (prev, next) pair merges into across ``sep``, or None."""
    if prev_label in NAME_LABELS and next_label in NAME_LABELS and _NAME_SEP.match(sep):
        return "PERSON"
    if prev_label in MONEY_LABELS and next_label in MONEY_LABELS and _MONEY_SEP.match(sep):
        return "MONEY"
    if address and prev_label in ADDRESS_LABELS and next_label in ADDRESS_LABELS and _ADDR_SEP.match(sep):
        return "ADDRESS"
    if prev_label == next_label and _WS_SEP.match(sep):
        return "SAME"  # multi-word single entity, e.g. "Executive Assistants"
    return None


def merge_entities(
    entities: Iterable[_EntityLike], text: str, *, address: bool = False
) -> list[Span]:
    """Merge adjacent same-family spans into single :class:`Span` objects.

    Tier 1 (always on): PERSON (name components), MONEY (currency + amount), and
    runs of the same label joined by whitespace. Tier 2 (``address=True``):
    also group street/city/state/... runs into a single ``ADDRESS`` span.

    A merged span's score is the minimum of its parts' scores (conservative).
    Input need not be sorted; output is sorted by start offset.
    """
    ents = sorted(entities, key=lambda e: (e.start, e.end))
    raw = text.encode("utf-8")
    out: list[Span] = []

    i = 0
    n = len(ents)
    while i < n:
        start = ents[i].start
        end = ents[i].end
        labels = [ents[i].label]
        scores = [ents[i].score]
        parts = [ents[i]]
        family: str | None = None

        j = i
        while j + 1 < n:
            nxt = ents[j + 1]
            if nxt.start < end:  # overlapping — leave for dedupe, don't chain
                break
            sep = raw[end:nxt.start].decode("utf-8", "replace")
            fam = _pair_family(labels[-1], nxt.label, sep, address)
            if fam is None or (family is not None and fam != family):
                break
            family = fam
            end = nxt.end
            labels.append(nxt.label)
            scores.append(nxt.score)
            parts.append(nxt)
            j += 1

        if len(parts) == 1:
            merged_label = labels[0]
        elif family == "SAME":
            merged_label = labels[0]
        else:
            merged_label = family  # PERSON / MONEY / ADDRESS

        out.append(Span(start, end, min(scores), merged_label, tuple(parts)))
        i = j + 1

    return out


def _span_text(e: _EntityLike, text: str) -> str:
    raw = text.encode("utf-8")
    return raw[e.start : e.end].decode("utf-8", "replace")


def dedupe_entities(
    entities: Sequence[_EntityLike], text: str, *, by: str = "value"
) -> list[_EntityLike]:
    """Drop redundant spans, preserving first-seen order.

    ``by="value"``: collapse repeats of the same ``(label, casefolded text)`` —
    e.g. a person named five times becomes one entry. ``by="overlap"``: drop any
    span fully contained within another kept span (keeps the widest).
    """
    if by == "value":
        seen: set[tuple[str, str]] = set()
        out: list[_EntityLike] = []
        for e in entities:
            key = (e.label, _span_text(e, text).casefold())
            if key in seen:
                continue
            seen.add(key)
            out.append(e)
        return out

    if by == "overlap":
        # Keep widest spans first; drop any later span contained in a kept one.
        ordered = sorted(entities, key=lambda e: (e.start, -(e.end - e.start)))
        kept: list[_EntityLike] = []
        for e in ordered:
            if any(k.start <= e.start and e.end <= k.end and k is not e for k in kept):
                continue
            kept.append(e)
        kept.sort(key=lambda e: e.start)
        return kept

    raise ValueError(f"unknown dedupe mode: {by!r} (expected 'value' or 'overlap')")
