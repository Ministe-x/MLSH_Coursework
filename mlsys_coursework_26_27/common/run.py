#!/usr/bin/env python3
"""Compile a CUDA implementation, check against PyTorch, then time valid cases.

This is a trusted local development harness, NOT a sandbox for student code.
"""
import argparse
import ctypes
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import shutil
import statistics
import subprocess
import sys
from datetime import datetime, timezone

import torch
from cases import GEMM_PERFORMANCE, ATTENTION_PERFORMANCE
from reference_backends import select_reference
from cache_control import L2Evictor, resolve_cache_config

ROOT = Path(__file__).resolve().parents[1]
TOLERANCES = {"gemm": (1e-3, 1e-3), "attention": (1e-2, 1e-2)}  # atol, rtol
GUARD = 256


def build(part, implementation, source_override=None):
    directory = ROOT / ("part1" if part == "gemm" else "part2")
    source = (Path(source_override).resolve() if source_override else
              ROOT / "examples" / f"{part}_naive.cu" if implementation == "example" else
              directory / "solve.cu")
    if not source.is_file():
        raise FileNotFoundError(source)
    nvcc = os.environ.get("NVCC") or shutil.which("nvcc")
    if not nvcc and os.environ.get("CUDA_HOME"):
        nvcc = str(Path(os.environ["CUDA_HOME"]) / "bin" / "nvcc")
    if not nvcc:
        raise RuntimeError("nvcc not found. Set CUDA_HOME and PATH, or NVCC.")
    major, minor = torch.cuda.get_device_capability()
    arch = os.environ.get("CUDA_ARCH", f"sm_{major}{minor}")
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    library = ROOT / "build" / f"{part}_{implementation}_{digest[:12]}.so"
    library.parent.mkdir(exist_ok=True)
    command = [nvcc, "-O3", "-std=c++17", f"-arch={arch}", "-lineinfo",
               "--shared", "-Xcompiler", "-fPIC", "-I", str(directory),
               str(directory / "binding.cu"), str(source), "-o", str(library)]
    print("BUILD:", " ".join(command), flush=True)
    subprocess.run(command, check=True)
    return library, source, digest, arch, subprocess.check_output(
        [nvcc, "--version"], text=True).strip()


class Kernel:
    def __init__(self, library, part):
        self.part = part
        self.lib = ctypes.CDLL(str(library))
        ptr, integer = ctypes.c_void_p, ctypes.c_int
        self.lib.invoke.argtypes = ([ptr] * 3 + [integer] * 3 + [ptr] if part == "gemm"
                                   else [ptr] * 4 + [integer] * 4 + [ptr] * 2)
        self.lib.invoke.restype = integer
        self.lib.error_string.argtypes = [integer]
        self.lib.error_string.restype = ctypes.c_char_p
        if part == "attention":
            self.lib.scratch_size.argtypes = [integer] * 4
            self.lib.scratch_size.restype = ctypes.c_size_t

    def scratch_bytes(self, shape):
        if self.part == "gemm":
            return 0
        size = self.lib.scratch_size(*shape)
        b, h, s, _ = shape
        cap = 2 * b * h * s * s * 4
        if size > cap:
            raise RuntimeError(f"workspace {size} bytes exceeds allowed {cap}")
        return size

    def __call__(self, inputs, output, shape, scratch):
        args = [tensor.data_ptr() for tensor in inputs] + [output.data_ptr(), *shape]
        if self.part == "attention":
            args.append(scratch.data_ptr() if scratch.numel() else None)
        args.append(torch.cuda.current_stream().cuda_stream)
        result = self.lib.invoke(*args)
        if result:
            raise RuntimeError(self.lib.error_string(result).decode())


def make_inputs(part, shape, kind, seed):
    generator = torch.Generator(device="cuda").manual_seed(seed)
    if part == "gemm":
        m, n, k = shape
        a = torch.randn(m, k, generator=generator, device="cuda") * 0.5
        b = torch.randn(k, n, generator=generator, device="cuda") * 0.5
        if kind == "zeros":
            a.zero_()
        elif kind == "identity":
            a.copy_(torch.eye(m, device="cuda"))
        elif kind == "signed":
            a.copy_((torch.arange(k, device="cuda") % 2 * 2 - 1).float().expand(m, k))
        return (a, b), (m, n), torch.float32
    inputs = tuple(torch.randn(*shape, device="cuda", generator=generator,
                               dtype=torch.float32).to(torch.float16) for _ in range(3))
    q, k, v = inputs
    if kind == "zero_qk":
        q.zero_(); k.zero_()
    elif kind == "zero_v":
        v.zero_()
    elif kind == "large_logits":
        q.mul_(16); k.mul_(16)
    elif kind == "constant_qk":
        q.fill_(1); k.fill_(1)
    return inputs, shape, torch.float16


def reference_for(part):
    path = ROOT / ("part1" if part == "gemm" else "part2") / "torch_ref.py"
    spec = importlib.util.spec_from_file_location(f"{part}_reference", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.reference


def benchmark(call, warmup, repeats, iterations, *, cache_mode="warm", cache_evictor=None):
    """Return median and batch means while excluding cold-cache eviction time.

    The default remains warm for callers of the original four-argument API.
    Course runs explicitly request cold mode and supply one reusable evictor.
    """
    if cache_mode not in ("cold", "warm"):
        raise ValueError("cache_mode must be cold or warm")
    if cache_mode == "cold" and cache_evictor is None:
        raise ValueError("cold timing requires a preallocated cache_evictor")
    stream = torch.cuda.current_stream()
    with torch.cuda.stream(stream):
        for _ in range(warmup):
            call()
        torch.cuda.synchronize()
        samples = []
        if cache_mode == "warm":
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            for _ in range(repeats):
                start.record(stream)
                for _ in range(iterations):
                    call()
                end.record(stream)
                end.synchronize()
                samples.append(start.elapsed_time(end) / iterations)
        else:
            # Allocate event pairs once. Queue flush/start/solve/end in the
            # same stream for every call. Only synchronize after a whole batch.
            pairs = [(torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
                     for _ in range(iterations)]
            for _ in range(repeats):
                for start, end in pairs:
                    cache_evictor()
                    start.record(stream)
                    call()
                    end.record(stream)
                pairs[-1][1].synchronize()
                samples.append(sum(start.elapsed_time(end) for start, end in pairs) / iterations)
    return statistics.median(samples), samples


def benchmark_calls(call, warmup, iterations, *, cache_mode="cold", cache_evictor=None):
    """Warm up, measure each call once, and return their arithmetic mean."""
    if cache_mode not in ("cold", "warm"):
        raise ValueError("cache_mode must be cold or warm")
    if cache_mode == "cold" and cache_evictor is None:
        raise ValueError("cold timing requires a preallocated cache_evictor")
    stream = torch.cuda.current_stream()
    with torch.cuda.stream(stream):
        for _ in range(warmup):
            call()
        torch.cuda.synchronize()
        pairs = [(torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
                 for _ in range(iterations)]
        for start, end in pairs:
            if cache_mode == "cold":
                cache_evictor()
            start.record(stream)
            call()
            end.record(stream)
        pairs[-1][1].synchronize()
        samples = [start.elapsed_time(end) for start, end in pairs]
    return statistics.mean(samples), samples


def check_case(kernel, reference, case, seed, timing, args):
    name, shape, kind = case
    inputs, output_shape, dtype = make_inputs(args.part, shape, kind, seed)
    snapshots = tuple(t.clone() for t in inputs)
    expected = reference(*inputs)
    count = math.prod(output_shape)
    allocation = torch.full((count + 2 * GUARD,), 8192., device="cuda", dtype=dtype)
    output = allocation[GUARD:-GUARD].view(output_shape)
    scratch_bytes = kernel.scratch_bytes(shape)
    scratch_allocation = torch.full((scratch_bytes + 2 * GUARD,), 165,
                                    device="cuda", dtype=torch.uint8)
    scratch = scratch_allocation[GUARD:-GUARD]
    output.fill_(float("nan"))
    # Exercise the supplied non-default stream. This is not a proof of stream
    # correctness; a stream-aware consumer below also checks the observed result.
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        kernel(inputs, output, shape, scratch)
        stream_observed_output = output.clone()
    torch.cuda.current_stream().wait_stream(stream)
    torch.cuda.synchronize()
    atol, rtol = TOLERANCES[args.part]
    torch.testing.assert_close(output, expected, atol=atol, rtol=rtol,
                               equal_nan=False, check_dtype=True)
    torch.testing.assert_close(stream_observed_output, expected, atol=atol, rtol=rtol,
                               equal_nan=False, check_dtype=True)
    if args.part == "attention":
        # Position zero attends only to V[0]; this catches strict-vs-inclusive
        # mask errors even when Q and K are arbitrary.
        torch.testing.assert_close(output[:, :, 0], inputs[2][:, :, 0],
                                   atol=atol, rtol=rtol)
        if kind in ("zero_qk", "constant_qk"):
            prefix_mean = inputs[2].float().cumsum(dim=-2) / torch.arange(
                1, shape[2] + 1, device="cuda").view(1, 1, -1, 1)
            torch.testing.assert_close(output, prefix_mean.half(), atol=atol, rtol=rtol)
        if name == "odd":
            prefix = shape[2] // 2
            changed = tuple(t.clone() for t in inputs)
            changed[1][:, :, prefix:].mul_(-4)
            changed[2][:, :, prefix:].add_(3)
            altered_output = torch.full_like(output, float("nan"))
            kernel(changed, altered_output, shape, scratch)
            torch.cuda.synchronize()
            torch.testing.assert_close(altered_output[:, :, :prefix], output[:, :, :prefix],
                                       atol=atol, rtol=rtol)
    if not bool(torch.isfinite(output).all()):
        raise AssertionError("Output contains non-finite values")
    for actual, saved in zip(inputs, snapshots):
        if not torch.equal(actual, saved):
            raise AssertionError("An input was modified")
    for storage, value in [(allocation, 8192.), (scratch_allocation, 165)]:
        if not bool((storage[:GUARD] == value).all() and (storage[-GUARD:] == value).all()):
            raise AssertionError("Output/workspace guard was overwritten")
    max_error = float((output.float() - expected.float()).abs().max())
    error_rms = float((output.float() - expected.float()).square().mean().sqrt())
    reference_rms = float(expected.float().square().mean().sqrt())
    normalized_rmse = error_rms / max(reference_rms, 1e-12)
    record = {"name": name, "shape": shape, "kind": kind, "seed": seed,
              "passed": True, "max_abs_error": max_error,
              "normalized_rmse": normalized_rmse, "workspace_bytes": scratch_bytes}
    if timing:
        # Output/scratch setup, allocation, reference and validation are excluded.
        call = lambda: kernel(inputs, output, shape, scratch)
        cache_mode = getattr(args, "cache_mode", "cold")
        flush_bytes = getattr(args, "flush_bytes", None)
        if cache_mode == "cold" and flush_bytes is None:
            flush_bytes = resolve_cache_config(cache_mode, getattr(args, "flush_mib", None),
                                               output.device)["flush_bytes"]
        # Allocate the independent buffer once for this case, outside timing.
        evictor = L2Evictor(flush_bytes, output.device) if cache_mode == "cold" else None
        if getattr(args, "repeats", None) is not None:
            # Reproduce explicitly configured historical calibration runs.
            latency_ms, samples = benchmark(call, args.warmup, args.repeats, args.iterations,
                                            cache_mode=cache_mode, cache_evictor=evictor)
            record["median_ms"] = latency_ms
        else:
            latency_ms, samples = benchmark_calls(call, args.warmup, args.iterations,
                                                  cache_mode=cache_mode, cache_evictor=evictor)
            record["mean_ms"] = latency_ms
        torch.testing.assert_close(output, expected, atol=atol, rtol=rtol, equal_nan=False)
        for actual, saved in zip(inputs, snapshots):
            if not torch.equal(actual, saved):
                raise AssertionError("An input was modified during timing")
        for storage, value in [(allocation, 8192.), (scratch_allocation, 165)]:
            if not bool((storage[:GUARD] == value).all() and (storage[-GUARD:] == value).all()):
                raise AssertionError("Output/workspace guard was overwritten during timing")
        record.update(samples_ms=samples)
        if args.part == "gemm":
            m, n, k = shape
            record["tflops"] = 2 * m * n * k / (latency_ms * 1e9)
        else:
            b, h, s, d = shape
            # Useful causal matmul FLOPs; the deliberately naive baseline
            # executes extra upper-triangle work, not included in this count.
            record["useful_causal_matmul_tflops"] = 2 * b * h * s * (s + 1) * d / (latency_ms * 1e9)
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--part", choices=["gemm", "attention"], required=True)
    parser.add_argument("--implementation", choices=["student", "example"], default="student")
    parser.add_argument("--source", help="Optional standalone .cu implementation to test")
    parser.add_argument("--reference", choices=["auto", "fast", "torch"], default="auto",
                        help="auto: compiled GEMM and instructor CuTe attention if available; torch: original oracle")
    parser.add_argument("--suite", choices=["correctness", "performance", "all"], default="all",
                        help="Same 12 cases: all/performance check then time; correctness skips timing")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=10,
                        help="Number of individually timed calls; their mean is reported")
    parser.add_argument("--cache-mode", choices=["cold", "warm"], default="cold",
                        help="cold evicts L2 before each timed call; warm disables active eviction")
    parser.add_argument("--flush-mib", type=float,
                        help="Cold-mode eviction buffer MiB; default max(256 MiB, 2x L2), override >=2x L2")
    parser.add_argument("--json", type=Path)
    parser.add_argument("--case", help="Run only the named public case (development only)")
    args = parser.parse_args()
    if min(args.warmup, args.iterations) < 1:
        parser.error("warmup and iterations must be positive")
    if not torch.cuda.is_available():
        parser.error("A CUDA-enabled PyTorch installation and NVIDIA GPU are required")
    torch.cuda.set_device(args.device)
    try:
        cache_config = resolve_cache_config(args.cache_mode, args.flush_mib, args.device)
    except ValueError as error:
        parser.error(str(error))
    args.flush_bytes = cache_config["flush_bytes"]
    print(f"CACHE {args.cache_mode}: eviction buffer {args.flush_bytes / (1024 * 1024):g} MiB", flush=True)
    if cache_config["fallback_l2_assumption_bytes"]:
        print(f"L2 size unavailable; conservatively assuming "
              f"{cache_config['fallback_l2_assumption_bytes'] / (1024 * 1024):g} MiB "
              f"({cache_config['l2_source']})", flush=True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
    library, source, digest, arch, compiler = build(args.part, args.implementation, args.source)
    kernel = Kernel(library, args.part)
    reference = select_reference(args.part, args.reference, reference_for(args.part))
    print(f"REFERENCE {args.part}: {reference.backend_name}", flush=True)
    performance = GEMM_PERFORMANCE if args.part == "gemm" else ATTENTION_PERFORMANCE
    selected = [(case, args.suite != "correctness") for case in performance]
    if args.case:
        selected = [(c, t) for c, t in selected if c[0] == args.case]
    if not selected:
        parser.error("No matching cases")
    output = {"schema_version": 1, "part": args.part, "implementation": args.implementation,
              "task_spec": "gemm_fp32_v1" if args.part == "gemm" else "attention_causal_fp16_v1",
              "reference_backend": reference.backend_name,
              "reference_source_sha256": reference.source_hash,
              "harness_sha256": hashlib.sha256(
                  Path(__file__).read_bytes() + (ROOT / "common/cases.py").read_bytes() +
                  (ROOT / "common/reference_backends.py").read_bytes() +
                  (ROOT / "common/cache_control.py").read_bytes() +
                  (ROOT / ("part1" if args.part == "gemm" else "part2") / "torch_ref.py").read_bytes()).hexdigest(),
              "source": str(source), "source_sha256": digest, "architecture": arch,
              "gpu": torch.cuda.get_device_name(), "device": args.device,
              "gpu_uuid": str(getattr(torch.cuda.get_device_properties(args.device), "uuid", "unknown")),
              "torch": torch.__version__, "torch_cuda": torch.version.cuda, "compiler": compiler,
              "utc": datetime.now(timezone.utc).isoformat(), "suite": args.suite,
              "seed": args.seed, "case_filter": args.case,
              "atol": TOLERANCES[args.part][0], "rtol": TOLERANCES[args.part][1],
              "timing": {"warmup": args.warmup, "iterations": args.iterations,
                         "aggregation": "arithmetic_mean",
                         "cache_mode": args.cache_mode, "flush_bytes": args.flush_bytes,
                         "method": ("CUDA events per solve, L2 eviction before each call excluded; arithmetic mean of calls"
                                    if args.cache_mode == "cold" else
                                    "CUDA events per solve without active eviction; arithmetic mean of calls")},
              "cache_detection": {key: value for key, value in cache_config.items()
                                  if key not in ("cache_mode", "flush_bytes")}, "cases": []}
    failures = 0
    for case, timing in selected:
        # Stable across --suite and --case selections. One row per shape.
        seed = args.seed + int(hashlib.sha256(case[0].encode()).hexdigest()[:8], 16)
        try:
            result = check_case(kernel, reference, case, seed, timing, args)
            suffix = f" mean={result['mean_ms']:.6f} ms" if timing else ""
            print(f"PASS {case[0]} {case[1]} seed={seed} max_abs={result['max_abs_error']:.3g} nrmse={result['normalized_rmse']:.3g}{suffix}", flush=True)
        except Exception as error:
            failures += 1
            result = {"name": case[0], "shape": case[1], "seed": seed,
                      "passed": False, "error": str(error)}
            print(f"FAIL {case[0]} seed={seed}: {error}", flush=True)
        result["category"] = "performance" if timing else "correctness"
        output["cases"].append(result)
    output["all_passed"] = failures == 0
    destination = args.json or ROOT / "results" / f"{args.part}_{args.implementation}.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(output, indent=2) + "\n")
    print(f"{len(output['cases']) - failures}/{len(output['cases'])} passed; results: {destination}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
