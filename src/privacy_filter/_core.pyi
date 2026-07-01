"""Type stubs for the compiled _core nanobind extension."""

__abi_version__: int

def abi_version() -> int:
    """Runtime ABI version of the linked pf library."""

class Entity:
    """A detected PII span. start/end are UTF-8 byte offsets into the source text."""

    @property
    def start(self) -> int: ...
    @property
    def end(self) -> int: ...
    @property
    def score(self) -> float: ...
    @property
    def label(self) -> str: ...
    def text(self, source: str) -> str:
        """Return the matched substring given the original text."""

    def __repr__(self) -> str: ...

class PrivacyFilter:
    """Loaded privacy-filter model. Use as a context manager or call close()."""

    def __init__(
        self, gguf_path: str, device: str = "cpu", n_threads: int = 0
    ) -> None:
        """Load a GGUF model.

        device: 'cpu' | 'gpu' | 'cuda' | 'vulkan' (optionally ':N' to pick a GPU).
        n_threads <= 0 picks a default (CPU only).
        """

    def classify(self, text: str, threshold: float = 0.0) -> list[Entity]:
        """Detect PII entities in text. Entities scoring below threshold are dropped."""

    def tokenize(self, text: str) -> tuple[list[int], list[tuple[int, int]]]:
        """Tokenize text; returns (ids, offsets) with byte-offset (start, end) pairs."""

    def set_window(self, max_forward_tokens: int) -> None:
        """Set max tokens per forward pass (default 4096); must be > 2048 to window."""

    def close(self) -> None:
        """Free the model. Idempotent."""

    def __enter__(self) -> PrivacyFilter: ...
    def __exit__(self, *args: object) -> None: ...
