"""BP-01 smoke tests: Aurora's model code imports and runs a forward/backward on the local device."""

import os

import torch

from mini_aurora.device import get_device, get_dtype


def test_env_defaults():
    assert os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] == "1"
    assert os.environ["TORCHDYNAMO_DISABLE"] == "1"


def test_get_device():
    device = get_device()
    if torch.backends.mps.is_available() and not torch.cuda.is_available():
        assert device.type == "mps"
    assert get_device("cpu").type == "cpu"
    x = torch.ones(4, device=device, dtype=get_dtype(device))
    assert x.sum().item() == 4


def test_aurora_imports_without_cluster_stack():
    import aurora.models.draft.llama3_eagle  # noqa: F401
    import aurora.models.eagle3  # noqa: F401
    from aurora.models import Eagle3Model  # noqa: F401
    from aurora.models.ops.loss_mask import compute_assistant_loss_mask  # noqa: F401
