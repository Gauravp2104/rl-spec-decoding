"""Device selection for mini_aurora.

Every module calls ``get_device()`` instead of hard-coding ``.cuda()`` so the same
code runs on a Mac (MPS) and on a CUDA machine (BP-09).
"""

import os

# Must be set before torch initialises MPS / dynamo.
_ENV_DEFAULTS = {
    "PYTORCH_ENABLE_MPS_FALLBACK": "1",
    "TORCHDYNAMO_DISABLE": "1",
    "WANDB_MODE": "offline",
    "TOKENIZERS_PARALLELISM": "false",
}


def configure_env() -> None:
    """Set Mac-friendly defaults without overriding values the user already set."""
    for key, value in _ENV_DEFAULTS.items():
        os.environ.setdefault(key, value)


configure_env()

import torch  # noqa: E402


def get_device(prefer: str | None = None) -> torch.device:
    """Return ``prefer`` (or ``$MINI_AURORA_DEVICE``) if given, else cuda → mps → cpu."""
    prefer = prefer or os.environ.get("MINI_AURORA_DEVICE")
    if prefer:
        return torch.device(prefer)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def get_dtype(device: torch.device) -> torch.dtype:
    """bf16 on CUDA/MPS, fp32 on CPU."""
    return torch.float32 if device.type == "cpu" else torch.bfloat16


def empty_cache(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.empty_cache()
    elif device.type == "mps":
        torch.mps.empty_cache()


def memory_allocated_gb(device: torch.device) -> float:
    """Memory currently held by tensors on ``device`` (0 on CPU)."""
    if device.type == "cuda":
        return torch.cuda.memory_allocated(device) / 1e9
    if device.type == "mps":
        return torch.mps.current_allocated_memory() / 1e9
    return 0.0
