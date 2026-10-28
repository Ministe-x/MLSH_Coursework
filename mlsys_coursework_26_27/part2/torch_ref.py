import math
import torch


def reference(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Transparent unfused oracle: FP16 storage, FP32 score/softmax/reductions.

    Deliberately does not call scaled_dot_product_attention or FlashAttention.
    """
    scores = torch.matmul(q.float(), k.float().transpose(-2, -1)) / math.sqrt(q.shape[-1])
    future = torch.ones(q.shape[-2], k.shape[-2], device=q.device,
                        dtype=torch.bool).triu(diagonal=1)
    scores = scores.masked_fill(future, float("-inf"))
    probabilities = torch.softmax(scores, dim=-1)
    return torch.matmul(probabilities, v.float()).to(torch.float16)
