# privacy_filter — Python bindings for privacy-filter.cpp

`privacy_filter` is a Python package that exposes
[privacy-filter.cpp](https://github.com/localai-org/privacy-filter.cpp) — a minimal
GGML inference engine for OpenAI's privacy-filter PII/NER token-classification model
family — to Python. It detects personally identifiable information (names, emails,
phone numbers, etc.) with precise UTF-8 byte offsets, far faster than a stock
Hugging Face Transformers pipeline.

- **Distribution name:** `privacy-filter`
- **Import name:** `privacy_filter`
- **License:** MIT

---

## Table of contents

- [How it works](#how-it-works)
- [Installation](#installation)
  - [Build requirements](#build-requirements)
  - [Install from source](#install-from-source)
  - [Getting a model](#getting-a-model)
- [Usage](#usage)
  - [Quick start](#quick-start)
  - [Detecting entities](#detecting-entities)
  - [Working with byte offsets](#working-with-byte-offsets)
  - [Tokenization](#tokenization)
  - [Long documents & windowing](#long-documents--windowing)
  - [Choosing a device](#choosing-a-device)
  - [Lifetime & resource management](#lifetime--resource-management)
  - [Error handling](#error-handling)
- [API reference](#api-reference)
- [Building wheels for distribution](#building-wheels-for-distribution)
- [Testing](#testing)
- [Project layout](#project-layout)
- [Design notes & limitations](#design-notes--limitations)

---

## How it works

The upstream project ships a deliberately small, FFI-friendly **flat C API**
(`include/pf.h`): an opaque handle plus a handful of functions with caller-owned
buffers. This package wraps that C API with [nanobind](https://github.com/wjakob/nanobind)
and builds it with [scikit-build-core](https://github.com/scikit-build/scikit-build-core)
(a CMake-based build backend).

The upstream C++ library (`pf`) and its GGML dependency are **statically linked into a
single compiled extension** (`privacy_filter._core`). The resulting `.so` has no
external `libggml`/`libpf` runtime dependency, so a built wheel is self-contained.

```
your Python code
      │
      ▼
privacy_filter (pure-Python re-exports)
      │
      ▼
privacy_filter._core         ← nanobind extension (binding.cpp)
      │
      ▼
pf  +  ggml   (static, compiled from the extern/privacy-filter.cpp submodule)
```

---

## Installation

### Build requirements

Installing from source compiles the C++ engine, so you need:

| Requirement | Version | Notes |
|---|---|---|
| Python | ≥ 3.12 | The extension targets the CPython **stable ABI (abi3)** for 3.12+. |
| C++ compiler | C++17 | GCC 13 / Clang 15+ known-good. |
| CMake | ≥ 3.21 | Pulled automatically if you build via `pip`/`uv` with build isolation. |
| Git | any | Needed to fetch the vendored submodules. |

Build-time Python packages (`scikit-build-core`, `nanobind`) are declared in
`pyproject.toml` and fetched automatically unless you use `--no-build-isolation`.

> **CPU-only:** this release builds a portable CPU engine. CUDA/Vulkan are compiled
> out (`PF_CUDA=OFF`, `PF_VULKAN=OFF`). See [Choosing a device](#choosing-a-device).

### Install from source

The upstream engine and its nested `ggml` are **git submodules**, so clone
recursively:

```bash
git clone --recursive <this-repo-url>
cd pii-inference
```

If you already cloned without `--recursive`:

```bash
git submodule update --init --recursive
```

Then install. Using [uv](https://docs.astral.sh/uv/) (recommended here):

```bash
# Create/populate the environment with dev + build tools
uv sync
uv pip install "scikit-build-core>=0.10" "nanobind>=2.0"   # build backend deps

# Editable install (fast rebuilds during development)
uv pip install -e . --no-build-isolation
```

Or with plain `pip` (build isolation fetches the build deps for you):

```bash
pip install .
```

Verify:

```bash
python -c "import privacy_filter; print(privacy_filter.__version__, privacy_filter.abi_version())"
# -> 0.1.0 1
```

### Getting a model

The package does **not** bundle a model — you supply a GGUF file at runtime. Converted
GGUFs for the privacy-filter family (English, multilingual, Nemotron) are published on
Hugging Face; you can also convert weights yourself with the upstream `scripts/convert.py`.

**Download a model:** the multilingual GGUFs used during development are at
<https://huggingface.co/LocalAI-io/privacy-filter-multilingual-GGUF/tree/main>. That repo
provides `privacy-filter-multilingual-q8.gguf` (8-bit, smaller/faster) and
`privacy-filter-multilingual-f16.gguf` (full precision). Download one and point
`PrivacyFilter(...)` at the local path:

```bash
# e.g. with the huggingface-cli (pip install huggingface_hub)
huggingface-cli download LocalAI-io/privacy-filter-multilingual-GGUF \
    privacy-filter-multilingual-q8.gguf --local-dir ./models
```

```python
pf = PrivacyFilter("./models/privacy-filter-multilingual-q8.gguf")
```

---

## Usage

### Quick start

```python
from privacy_filter import PrivacyFilter

text = "Contact Jane Doe at jane.doe@acme.com or +1-202-555-0142."

with PrivacyFilter("model.gguf", device="cpu", n_threads=0) as pf:
    for e in pf.classify(text, threshold=0.5):
        print(f"{e.label:12} {e.score:.3f}  {e.text(text)!r}")
```

Output:

```
FIRSTNAME    0.589  'Jane'
LASTNAME     0.855  'Doe'
EMAIL        0.991  'jane.doe@acme.com'
PHONE        0.987  '+1-202-555-0142'
```

### Detecting entities

`classify(text, threshold=0.0)` returns a `list[Entity]`. Each `Entity` has:

| Attribute | Type | Meaning |
|---|---|---|
| `start` | `int` | Start **byte** offset into the UTF-8 text |
| `end` | `int` | End **byte** offset (exclusive) |
| `score` | `float` | Confidence in `[0, 1]` |
| `label` | `str` | Entity category, e.g. `EMAIL`, `FIRSTNAME`, `PHONE` |

Entities scoring below `threshold` are dropped by the engine. Use a higher threshold
(e.g. `0.5`) to trade recall for precision:

```python
strict   = pf.classify(text, threshold=0.8)   # high-precision
everything = pf.classify(text, threshold=0.0)  # every candidate span
```

### Working with byte offsets

`start`/`end` are **UTF-8 byte offsets**, matching the C engine. This matters for
non-ASCII text, where byte offsets differ from Python string (code-point) indices.
Use `Entity.text(source)` to recover the matched substring safely:

```python
text = "Écrivez à José at jose@exemplo.es"
with PrivacyFilter("model.gguf") as pf:
    for e in pf.classify(text):
        # Correct — slices UTF-8 bytes and decodes:
        print(e.text(text))
        # Equivalent manual form:
        print(text.encode("utf-8")[e.start:e.end].decode("utf-8"))
        # WRONG for non-ASCII — do not index the str directly with byte offsets:
        # text[e.start:e.end]
```

To redact detected spans, work in bytes and process right-to-left so earlier offsets
stay valid:

```python
def redact(text: str, ents) -> str:
    buf = bytearray(text.encode("utf-8"))
    for e in sorted(ents, key=lambda x: x.start, reverse=True):
        buf[e.start:e.end] = f"[{e.label}]".encode("utf-8")
    return buf.decode("utf-8")

with PrivacyFilter("model.gguf") as pf:
    print(redact(text, pf.classify(text, threshold=0.5)))
# -> "Contact [FIRSTNAME] [LASTNAME] at [EMAIL] or [PHONE]."
```

### Tokenization

`tokenize(text)` exposes the model's tokenizer for lower-level use. It returns a
`(ids, offsets)` tuple, where `offsets[i]` is the `(start_byte, end_byte)` span of
token `ids[i]`:

```python
ids, offsets = pf.tokenize("Jane at jane@acme.com")
print(len(ids), "tokens")
for tok_id, (s, e) in zip(ids, offsets):
    print(tok_id, repr(text.encode()[s:e].decode()))
```

### Long documents & windowing

The model processes up to `max_forward_tokens` (default 4096) per forward pass. Longer
inputs are automatically run as overlapping windows and stitched together. Tune the
window with `set_window`:

```python
with PrivacyFilter("model.gguf") as pf:
    pf.set_window(8192)                 # larger window; must be > 2048 to window
    ents = pf.classify(very_long_document, threshold=0.5)
```

### Choosing a device

`device` accepts `"cpu"` (default), `"gpu"`, `"cuda"`, or `"vulkan"`, optionally with
`":N"` to select the Nth matching GPU (e.g. `"cuda:1"`). `n_threads <= 0` picks a
sensible default (CPU only).

```python
PrivacyFilter("model.gguf", device="cpu", n_threads=8)
```

> **Note:** GPU backends require a build compiled with `PF_CUDA`/`PF_VULKAN` enabled.
> The wheels produced by this project are CPU-only, so requesting `"cuda"`/`"vulkan"`
> there will fail at load time. GPU builds are a planned separate variant.

### Lifetime & resource management

`PrivacyFilter` owns native resources (the loaded model). Free them by either using it
as a **context manager** (recommended) or calling `close()`:

```python
with PrivacyFilter("model.gguf") as pf:
    ...
# freed automatically on block exit

pf = PrivacyFilter("model.gguf")
try:
    ...
finally:
    pf.close()   # idempotent; also runs on garbage collection
```

Calling any method after `close()` raises `RuntimeError`. Loading a model is
comparatively expensive — load once and reuse the instance for many `classify` calls.

### Error handling

Failures raise `RuntimeError` with the underlying engine message:

```python
try:
    pf = PrivacyFilter("/nonexistent/model.gguf")
except RuntimeError as e:
    print("load failed:", e)
    # -> load failed: pf_load: failed to read GGUF: /nonexistent/model.gguf
```

---

## API reference

### `class PrivacyFilter(gguf_path, device="cpu", n_threads=0)`

Load a GGUF model. Raises `RuntimeError` if the model cannot be loaded.

| Method | Description |
|---|---|
| `classify(text, threshold=0.0) -> list[Entity]` | Detect PII spans; drop those below `threshold`. Releases the GIL during inference. |
| `tokenize(text) -> tuple[list[int], list[tuple[int, int]]]` | Token IDs and their `(start, end)` byte offsets. |
| `set_window(max_forward_tokens: int) -> None` | Max tokens per forward pass (default 4096); must be > 2048 to window. |
| `close() -> None` | Free the model. Idempotent. |
| `__enter__` / `__exit__` | Context-manager support (calls `close()` on exit). |

### `class Entity`

Read-only span: `start: int`, `end: int` (UTF-8 byte offsets), `score: float`,
`label: str`, plus `text(source: str) -> str` to extract the matched substring.

### Module functions

- `privacy_filter.abi_version() -> int` — ABI version of the linked engine.
- `privacy_filter.__version__` — package version string.

---

## Building wheels for distribution

The package builds a **single stable-ABI (`abi3`) wheel per platform** that works on
CPython 3.12+ (`wheel.py-api = "cp312"` in `pyproject.toml`), so you do not need one
wheel per Python minor version.

### Build a wheel locally

```bash
# with uv
uv build --wheel

# or with the PyPA build frontend
python -m build --wheel
```

The wheel lands in `dist/`, named like
`privacy_filter-0.1.0-cp312-abi3-linux_x86_64.whl`, and can be installed with:

```bash
pip install dist/privacy_filter-0.1.0-cp312-abi3-*.whl
```

> A locally built Linux wheel is tagged `linux_x86_64` and is **not** portable to
> other machines/distros. For redistributable Linux wheels, build under `manylinux`
> (see below) to produce a `manylinux_*` tag.

### Portable/redistributable wheels (cibuildwheel)

For wheels you can publish, use [cibuildwheel](https://cibuildwheel.pypa.io/), which
builds inside `manylinux` containers and repairs the wheel with `auditwheel`. Because
the engine statically links `ggml`, there are no extra shared libraries to bundle.

Minimal `pyproject.toml` additions:

```toml
[tool.cibuildwheel]
build = "cp312-*"                 # one abi3 wheel; abi3 covers 3.12+
build-frontend = "build"
# Submodules must be present in the build container:
before-all = "git submodule update --init --recursive"

[tool.cibuildwheel.linux]
# manylinux image with a C++17 toolchain and CMake is required.
```

Then:

```bash
pipx run cibuildwheel --platform linux
```

Notes and caveats:

- **Submodules:** the build needs `extern/privacy-filter.cpp` and its nested `ggml`.
  Ensure they are checked out in the build environment (the `before-all` hook above,
  or `actions/checkout` with `submodules: recursive` in CI).
- **CPU portability:** verify the CPU baseline of the upstream build. If it enables
  aggressive ISA flags (e.g. AVX-512), a wheel built on a modern host may crash with
  `SIGILL` on older CPUs. For broadly portable wheels, constrain the CPU features at
  build time.
- **GPU:** CUDA/Vulkan wheels are out of scope for this baseline; they would be
  separate builds with `PF_CUDA`/`PF_VULKAN` enabled and their own platform tags.

### Install requirements summary

| Scenario | Needs a compiler? | Needs submodules? |
|---|---|---|
| `pip install <prebuilt-wheel>.whl` | No | No |
| `pip install .` / `uv pip install -e .` (from source) | Yes (C++17 + CMake) | Yes (recursive) |

---

## Testing

```bash
uv run pytest                                   # model-free tests (import, ABI, error path)
PF_TEST_MODEL=/path/to/model.gguf uv run pytest # full suite incl. classify/tokenize
```

Model-dependent tests are skipped unless `PF_TEST_MODEL` points at a GGUF file, so CI
without a model still passes.

### Processing a SQLite database

`tests/classify_database.py` is a runnable script (not collected by pytest — it needs a
local model and database) that runs inference over texts stored in a SQLite table and
prints each text followed by the detected entities. It is handy for eyeballing model
behaviour over real data:

```bash
python tests/classify_database.py \
    --db /path/to/data.sqlite \
    --table classifications --column cleaned_text \
    --model /path/to/model.gguf \
    --limit 200 --threshold 0.5
```

All flags have defaults (see `--help`); the model path also falls back to
`PF_TEST_MODEL`. Output is `TEXT` / `INFERENCE` blocks per row, ending with an
entity-count summary.

---

## Project layout

```
pyproject.toml               # scikit-build-core backend + project metadata
CMakeLists.txt               # links pf + ggml statically into the _core extension
extern/privacy-filter.cpp/   # upstream engine (git submodule; nests ggml)
src/privacy_filter/
  __init__.py                # public re-exports (PrivacyFilter, Entity, abi_version)
  binding.cpp                # nanobind glue over include/pf.h
  _core.pyi                  # type stubs for the compiled extension
  py.typed                   # PEP 561 marker
tests/test_bindings.py       # pytest suite (model-gated)
docs/privacy-filter.md       # this document
```

---

## Design notes & limitations

- **Static linking.** `CMakeLists.txt` sets `BUILD_SHARED_LIBS OFF` and
  `POSITION_INDEPENDENT_CODE ON` so `pf` and `ggml` are absorbed into a single
  extension. Without this, `import privacy_filter` fails with
  `libggml.so.0: cannot open shared object file`.
- **Memory ownership.** Buffers returned by the C API (`pf_classify`, `pf_tokenize`)
  are copied into Python objects and freed immediately; entity `label` strings (valid
  only until the context is freed) are copied to Python `str`.
- **GIL.** `classify` and `tokenize` release the GIL during native inference, so other
  Python threads can run concurrently.
- **`logits` not exposed.** The upstream public C API returns per-token logits as a
  flat `n * n_labels` buffer but provides no way to query `n_labels`, so the buffer
  cannot be reliably reshaped from the public API alone. Exposing this would require a
  small upstream addition (e.g. `pf_n_labels(ctx)`).
- **CPU-only.** GPU backends are compiled out in this release.
```
