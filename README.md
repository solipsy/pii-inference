# privacy-filter (Python bindings)

Python bindings for [privacy-filter.cpp](https://github.com/localai-org/privacy-filter.cpp) —
a minimal GGML inference engine for OpenAI's privacy-filter PII/NER
token-classification models. Detects personally identifiable information with
precise UTF-8 byte offsets, far faster than stock Transformers.

The bindings statically link the upstream `pf` library and `ggml` into a single
compiled extension (via [nanobind](https://github.com/wjakob/nanobind) +
[scikit-build-core](https://github.com/scikit-build/scikit-build-core)).

## Requirements

- Python ≥ 3.12
- A C++17 compiler and CMake ≥ 3.21 (only for building from source)
- A privacy-filter GGUF model file (not bundled — see upstream Hugging Face collections)

This first release is **CPU-only**. CUDA/Vulkan variants are planned as separate builds.

## Install (from source)

```bash
git clone --recursive <this repo>
cd pii-inference
uv sync                              # create/populate the venv
uv pip install -e . --no-build-isolation
```

If you cloned without `--recursive`:

```bash
git submodule update --init --recursive
```

## Usage

```python
from privacy_filter import PrivacyFilter

text = "Contact Jane Doe at jane.doe@acme.com or +1-202-555-0142."

with PrivacyFilter("model.gguf", device="cpu", n_threads=0) as pf:
    for e in pf.classify(text, threshold=0.5):
        print(f"{e.label:12} {e.score:.3f}  {e.text(text)!r}")
```

`Entity` exposes `.start`, `.end` (UTF-8 **byte** offsets), `.score`, `.label`,
and `.text(source)` to recover the matched substring.

### Other methods

- `pf.tokenize(text) -> (ids, offsets)` — token IDs and byte-offset `(start, end)` pairs.
- `pf.set_window(max_forward_tokens)` — tune windowing for long documents (must be > 2048 to window).
- `pf.close()` — free the model (also done on context-manager exit / GC).

## Testing

```bash
uv run pytest                        # model-free tests
PF_TEST_MODEL=/path/to/model.gguf uv run pytest   # full suite
```

## Notes

- `device` accepts `"cpu"`, `"gpu"`, `"cuda"`, `"vulkan"` (optionally `":N"` to pick a GPU),
  but GPU backends require a build with `PF_CUDA`/`PF_VULKAN` enabled.
- Per-token `logits` are not yet exposed: the public C API provides no way to
  query the label count needed to reshape the buffer. Tracked for a future release.
