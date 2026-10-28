import torch


def reference(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """FP32 mathematical reference; run.py disables TF32 globally."""
    return torch.matmul(a.float(), b.float())
