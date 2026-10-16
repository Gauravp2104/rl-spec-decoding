"""Scaled-down Aurora reproduction that runs on a single device (CUDA, MPS or CPU)."""

from mini_aurora.device import configure_env, get_device, get_dtype

__all__ = ["configure_env", "get_device", "get_dtype"]
