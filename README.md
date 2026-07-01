# privacy-filter

> Fast PII/NER detection for Python — thin bindings over [`privacy-filter.cpp`](https://github.com/localai-org/privacy-filter.cpp), a minimal GGML inference engine for OpenAI's privacy-filter token-classification models.

[![Wheels](https://github.com/solipsy/pii-inference/actions/workflows/wheels.yml/badge.svg)](https://github.com/solipsy/pii-inference/actions/workflows/wheels.yml)
[![Python](https://img.shields.io/badge/python-3.12%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Platforms](https://img.shields.io/badge/platforms-Linux%20x86__64%20%7C%20macOS%20arm64-lightgrey.svg)](#installation)

Detect names, emails, phone numbers and other PII with **precise UTF-8 byte offsets**,
far faster than a stock Hugging Face Transformers pipeline. The upstream `pf` library and
`ggml` are statically linked into a single compiled extension via
[nanobind](https://github.com/wjakob/nanobind) +
[scikit-build-core](https://github.com/scikit-build/scikit-build-core), so an installed
wheel is fully self-contained.

## Features

- 🔎 **Entity spans with byte offsets** — every detection carries `start`/`end`/`score`/`label`.
- 🧩 **Merging & dedup helpers** — fold token-level spans into `PERSON` / `MONEY` / `ADDRESS` and collapse repeats.
- 🌍 **Multilingual** — works on non-ASCII text, offsets stay correct.
- 📦 **Self-contained wheels** — one `abi3` wheel per platform, no external `libggml` to locate.
- ⚡ **Releases the GIL** during inference; load a model once and reuse it.

> **Scope:** this release is **CPU-only** and targets **CPython 3.12+**. Prebuilt wheels
> are published for **Linux x86_64** (manylinux) and **macOS arm64** (Apple Silicon).

## Installation

### From PyPI (coming soon)

> 🚧 **Not yet published.** Once released, install with:
>
> ```bash
> pip install privacy-filter
> ```

### From a prebuilt wheel (GitHub Releases)

Each [release](https://github.com/solipsy/pii-inference/releases) attaches `abi3` wheels
for Linux x86_64 and macOS arm64. Grab the one for your platform:

```bash
pip install https://github.com/solipsy/pii-inference/releases/download/v0.1.0/privacy_filter-0.1.0-cp312-abi3-macosx_11_0_arm64.whl
```

### From source

Building from source compiles the C++ engine, so you need a **C++17 compiler** and
**CMake ≥ 3.21** (CMake/Ninja are fetched automatically by the build). The upstream
engine and its nested `ggml` are git submodules:

```bash
git clone --recursive https://github.com/solipsy/pii-inference.git
cd pii-inference
pip install .          # or: uv sync && uv pip install -e . --no-build-isolation
```

Already cloned without `--recursive`? Run `git submodule update --init --recursive`.

## Quick start

You supply a GGUF model at runtime (it is not bundled — see
[Getting a model](docs/privacy-filter.md#getting-a-model)):

```python
from privacy_filter import PrivacyFilter

text = "Contact Jane Doe at jane.doe@acme.com or +1-202-555-0142."

with PrivacyFilter("model.gguf", device="cpu", n_threads=0) as pf:
    for e in pf.classify(text, threshold=0.5):
        print(f"{e.label:12} {e.score:.3f}  {e.text(text)!r}")
```

```text
FIRSTNAME    0.589  'Jane'
LASTNAME     0.855  'Doe'
EMAIL        0.991  'jane.doe@acme.com'
PHONE        0.987  '+1-202-555-0142'
```

`Entity` exposes `.start`/`.end` (UTF-8 **byte** offsets), `.score`, `.label`, and
`.text(source)`. There are also `merge_entities()` / `dedupe_entities()` post-processing
helpers, tokenization, long-document windowing, and device selection.

## Documentation

Full usage and API reference live in **[docs/privacy-filter.md](docs/privacy-filter.md)**:

| Topic | |
|---|---|
| [Quick start](docs/privacy-filter.md#quick-start) · [Detecting entities](docs/privacy-filter.md#detecting-entities) | classify, thresholds |
| [Byte offsets & redaction](docs/privacy-filter.md#working-with-byte-offsets) | non-ASCII-safe slicing |
| [Merging & deduplication](docs/privacy-filter.md#reducing-clutter-merging--deduplication) | PERSON / MONEY / ADDRESS, dedup |
| [Tokenization](docs/privacy-filter.md#tokenization) · [Windowing](docs/privacy-filter.md#long-documents--windowing) | lower-level access |
| [API reference](docs/privacy-filter.md#api-reference) | `PrivacyFilter`, `Entity`, `Span`, functions |
| [Building wheels](docs/privacy-filter.md#building-wheels-for-distribution) | cibuildwheel, CI |

## Testing

```bash
uv run pytest                                   # model-free tests
PF_TEST_MODEL=/path/to/model.gguf uv run pytest # full suite incl. classify/tokenize
```

## Roadmap

- [ ] Publish to PyPI
- [ ] Windows wheel
- [ ] Optional GPU builds (CUDA / Vulkan / Metal)
- [ ] Expose per-token `logits` (needs a small upstream addition)

## License

MIT — see [LICENSE](LICENSE). Bindings for the upstream
[`privacy-filter.cpp`](https://github.com/localai-org/privacy-filter.cpp) project.
