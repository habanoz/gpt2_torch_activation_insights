import torch
from typing import Any

class AllocatedMemContext:
    """
    Context manager which captures the allocated GPU memory at context exit and the change between
    enter and exit.

    Only includes `allocated_bytes.all.`-prefixed keys in `memory_stats` with all readings converted
    to GiB.

    Example:

    ```python

    ```
    """

    def __init__(self) -> None:
        # Ensure CUDA libraries are loaded:
        torch.cuda.current_blas_handle()

        self.before: dict[str, int] = {}
        self.after: dict[str, int] = {}
        self.delta: dict[str, int] = {}

        self._mem_key_prefix = "allocated_bytes.all."

    def _get_mem_dict(self) -> dict[str, int]:
        return {
            k.replace(self._mem_key_prefix, ""): v
            for k, v in torch.cuda.memory_stats().items()
            if self._mem_key_prefix in k
        }

    def __enter__(self) -> "AllocatedMemContext":
        self.before = self._get_mem_dict()
        return self

    def __exit__(self, *args: Any, **kwargs: Any) -> None:
        self.after = self._get_mem_dict()
        self.delta = {k: v - self.before[k] for k, v in self.after.items()}