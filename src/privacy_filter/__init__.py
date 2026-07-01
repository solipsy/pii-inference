"""Python bindings for privacy-filter.cpp — fast GGML PII/NER token classification.

Example
-------
>>> from privacy_filter import PrivacyFilter
>>> with PrivacyFilter("model.gguf", device="cpu") as pf:
...     for e in pf.classify("Email me at jane@acme.com", threshold=0.5):
...         print(e.label, e.start, e.end, e.text("Email me at jane@acme.com"))
"""

from ._core import Entity, PrivacyFilter, abi_version
from .merge import Span, dedupe_entities, merge_entities

__version__ = "0.1.1"
__all__ = [
    "PrivacyFilter",
    "Entity",
    "abi_version",
    "Span",
    "merge_entities",
    "dedupe_entities",
    "__version__",
]
