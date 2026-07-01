"""Tests for the privacy_filter bindings.

Model-dependent tests are gated behind the PF_TEST_MODEL env var (path to a
GGUF), so the suite passes in CI without a model.
"""

import os

import pytest

import privacy_filter
from privacy_filter import Entity, PrivacyFilter

MODEL = os.environ.get("PF_TEST_MODEL")
needs_model = pytest.mark.skipif(not MODEL, reason="set PF_TEST_MODEL to a GGUF path")

SAMPLE = "Contact Jane Doe at jane.doe@acme.com or +1-202-555-0142."


def test_abi_version_matches_header():
    assert privacy_filter.abi_version() == privacy_filter._core.__abi_version__


def test_load_missing_model_raises():
    with pytest.raises(RuntimeError):
        PrivacyFilter("/nonexistent/model.gguf")


@needs_model
def test_classify_returns_valid_entities():
    n_bytes = len(SAMPLE.encode("utf-8"))
    with PrivacyFilter(MODEL, device="cpu") as pf:
        ents = pf.classify(SAMPLE, threshold=0.0)
    assert isinstance(ents, list)
    for e in ents:
        assert isinstance(e, Entity)
        assert 0 <= e.start <= e.end <= n_bytes
        assert e.label  # non-empty label
        # text() slices the original by byte offset
        assert e.text(SAMPLE) == SAMPLE.encode()[e.start : e.end].decode("utf-8")


@needs_model
def test_tokenize_shapes():
    with PrivacyFilter(MODEL) as pf:
        ids, offsets = pf.tokenize(SAMPLE)
    assert len(ids) == len(offsets)
    for start, end in offsets:
        assert 0 <= start <= end <= len(SAMPLE.encode("utf-8"))


@needs_model
def test_use_after_close_raises():
    pf = PrivacyFilter(MODEL)
    pf.close()
    pf.close()  # idempotent
    with pytest.raises(RuntimeError):
        pf.classify(SAMPLE)
