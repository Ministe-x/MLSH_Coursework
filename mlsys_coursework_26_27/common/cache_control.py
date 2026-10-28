"""L2 eviction setup for reproducible cold-cache CUDA measurements.

The eviction kernel accesses a separate buffer. It does not alter the problem
inputs, output or workspace. Eviction must be queued before the start event.
"""
import math

import torch

MIB = 1024 * 1024
DEFAULT_MIN_FLUSH_BYTES = 256 * MIB
# Conservative for the course's RTX 2080 Ti / 4080 / 5090 GPUs when neither
# property API is available. The assumption is reported, never presented as
# a measured hardware property.
UNKNOWN_L2_ASSUMPTION_BYTES = 128 * MIB


def query_l2_cache_size(device=None):
    """Return (actual bytes or None, property source) without GPU work."""
    if device is None:
        device = torch.cuda.current_device()
    if not isinstance(device, int):
        device = torch.device(device).index
        if device is None:
            device = torch.cuda.current_device()
    properties = torch.cuda.get_device_properties(device)
    for name in ("L2_cache_size", "l2_cache_size", "l2_cache_size_bytes"):
        value = getattr(properties, name, None)
        if isinstance(value, int) and value > 0:
            return value, f"torch.cuda.get_device_properties.{name}"
    try:
        from cuda.bindings import driver
        status, = driver.cuInit(0)
        if int(status) != 0:
            raise RuntimeError(f"cuInit returned {int(status)}")
        status, cuda_device = driver.cuDeviceGet(device)
        if int(status) != 0:
            raise RuntimeError(f"cuDeviceGet returned {int(status)}")
        status, size = driver.cuDeviceGetAttribute(
            driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_L2_CACHE_SIZE,
            cuda_device,
        )
        if int(status) == 0 and int(size) > 0:
            return int(size), "cuda.bindings.driver.CU_DEVICE_ATTRIBUTE_L2_CACHE_SIZE"
        raise RuntimeError(f"cuDeviceGetAttribute returned {int(status)} / {size}")
    except (ImportError, AttributeError, RuntimeError, TypeError, ValueError) as error:
        return None, f"unavailable ({type(error).__name__}: {error})"


def resolve_cache_config(cache_mode="cold", flush_mib=None, device=None):
    """Resolve actual eviction bytes; an override must cover at least 2x L2.

    The default is max(256 MiB, 2x detected L2). If detection is unavailable,
    assume 128 MiB of L2 and record that conservative fallback explicitly.
    Warm mode performs no eviction and records zero flush bytes.
    """
    if cache_mode not in ("cold", "warm"):
        raise ValueError("cache_mode must be cold or warm")
    if flush_mib is not None and (not math.isfinite(flush_mib) or flush_mib <= 0):
        raise ValueError("--flush-mib must be finite and positive")
    if cache_mode == "warm":
        return {"cache_mode": "warm", "flush_bytes": 0,
                "l2_cache_bytes": None, "l2_source": "not queried in warm mode",
                "fallback_l2_assumption_bytes": 0}
    actual_l2, source = query_l2_cache_size(device)
    assumed_l2 = actual_l2 if actual_l2 is not None else UNKNOWN_L2_ASSUMPTION_BYTES
    minimum = 2 * assumed_l2
    size = max(DEFAULT_MIN_FLUSH_BYTES, minimum) if flush_mib is None else math.ceil(flush_mib * MIB)
    if size < minimum:
        qualifier = "detected" if actual_l2 is not None else "conservatively assumed"
        raise ValueError(f"--flush-mib must be at least {minimum / MIB:g} MiB "
                         f"(2x the {qualifier} L2 size)")
    return {"cache_mode": "cold", "flush_bytes": size,
            "l2_cache_bytes": actual_l2, "l2_source": source,
            "fallback_l2_assumption_bytes": 0 if actual_l2 is not None else assumed_l2}


class L2Evictor:
    """Allocate once and reuse a separate read/write buffer for L2 eviction."""

    def __init__(self, flush_bytes, device=None):
        if not isinstance(flush_bytes, int) or flush_bytes <= 0:
            raise ValueError("flush_bytes must be a positive integer")
        self.flush_bytes = flush_bytes
        self.buffer = torch.empty(flush_bytes, device=device or "cuda", dtype=torch.uint8)
        self.buffer.zero_()

    def __call__(self):
        # Read-modify-write each byte instead of relying on cudaMemset's cache
        # behavior. This operation uses PyTorch's current CUDA stream.
        self.buffer.add_(1)
