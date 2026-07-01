"""Tests for the merge/dedupe post-processing helpers (no model required)."""

from dataclasses import dataclass

from privacy_filter import Span, dedupe_entities, merge_entities


@dataclass
class E:
    """A minimal Entity stand-in (duck-typed) for building test inputs."""

    start: int
    end: int
    score: float
    label: str


def spans_from(text: str, *segments):
    """Build entities from (label, substring) pairs, locating byte offsets."""
    raw = text.encode("utf-8")
    ents = []
    cursor = 0
    for label, sub in segments:
        idx = raw.index(sub.encode("utf-8"), cursor)
        ents.append(E(idx, idx + len(sub.encode("utf-8")), 0.9, label))
        cursor = idx + len(sub.encode("utf-8"))
    return ents


def labels_texts(spans, text):
    return [(s.label, s.text(text)) for s in spans]


def test_person_merge():
    text = "Contact Mr. Marc Crowe today."
    ents = spans_from(text, ("PREFIX", "Mr."), ("FIRSTNAME", "Marc"), ("LASTNAME", "Crowe"))
    merged = merge_entities(ents, text)
    assert labels_texts(merged, text) == [("PERSON", "Mr. Marc Crowe")]
    # component detail is preserved
    assert len(merged[0].parts) == 3
    assert isinstance(merged[0], Span)


def test_firstname_lastname_merge():
    text = "hi Anne Horel here"
    ents = spans_from(text, ("FIRSTNAME", "Anne"), ("LASTNAME", "Horel"))
    assert labels_texts(merge_entities(ents, text), text) == [("PERSON", "Anne Horel")]


def test_money_merge_adjacent():
    text = "raised $400 billion last year"
    ents = spans_from(text, ("CURRENCYSYMBOL", "$"), ("AMOUNT", "400 billion"))
    assert labels_texts(merge_entities(ents, text), text) == [("MONEY", "$400 billion")]


def test_same_label_whitespace_merges():
    text = "hiring Executive Assistants now"
    ents = spans_from(text, ("OCCUPATION", "Executive"), ("OCCUPATION", "Assistants"))
    assert labels_texts(merge_entities(ents, text), text) == [("OCCUPATION", "Executive Assistants")]


def test_sentence_boundary_does_not_merge():
    # "Indiana. South Korea" — same label but separated by a sentence boundary.
    text = "moved from Indiana. South Korea is far."
    ents = spans_from(text, ("STATE", "Indiana"), ("STATE", "South Korea"))
    merged = merge_entities(ents, text)
    assert [s.label for s in merged] == ["STATE", "STATE"]  # unchanged


def test_slash_separated_cities_do_not_merge():
    text = "serving Cold Lake/Bonnyville region"
    ents = spans_from(text, ("CITY", "Cold Lake"), ("CITY", "Bonnyville"))
    assert len(merge_entities(ents, text)) == 2


def test_unrelated_labels_do_not_merge():
    text = "a VFX Producer role"  # VFX mislabeled; must not fold into OCCUPATION
    ents = spans_from(text, ("GENDER", "VFX"), ("OCCUPATION", "Producer"))
    assert len(merge_entities(ents, text)) == 2


def test_address_tier_opt_in():
    text = "6005 Monticello Drive, Montgomery, Alabama"
    ents = spans_from(
        text,
        ("BUILDINGNUMBER", "6005"),
        ("STREET", "Monticello Drive"),
        ("CITY", "Montgomery"),
        ("STATE", "Alabama"),
    )
    # off by default
    assert len(merge_entities(ents, text)) > 1
    # on when requested
    merged = merge_entities(ents, text, address=True)
    assert labels_texts(merged, text) == [("ADDRESS", "6005 Monticello Drive, Montgomery, Alabama")]


def test_dedupe_by_value():
    text = "Anne and anne and ANNE"
    ents = spans_from(text, ("FIRSTNAME", "Anne"), ("FIRSTNAME", "anne"), ("FIRSTNAME", "ANNE"))
    deduped = dedupe_entities(ents, text, by="value")
    assert len(deduped) == 1  # case-insensitive collapse


def test_dedupe_by_overlap():
    text = "Marc Crowe"
    wide = Span(0, 10, 0.9, "PERSON")
    narrow = Span(0, 4, 0.9, "FIRSTNAME")
    kept = dedupe_entities([wide, narrow], text, by="overlap")
    assert kept == [wide]


def test_merge_then_dedupe_chain():
    text = "Anne Horel met Anne Horel"
    ents = spans_from(
        text,
        ("FIRSTNAME", "Anne"), ("LASTNAME", "Horel"),
        ("FIRSTNAME", "Anne"), ("LASTNAME", "Horel"),
    )
    merged = merge_entities(ents, text)
    assert len(merged) == 2
    deduped = dedupe_entities(merged, text, by="value")
    assert labels_texts(deduped, text) == [("PERSON", "Anne Horel")]
