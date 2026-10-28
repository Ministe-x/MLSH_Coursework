#!/usr/bin/env python3
"""Compare a correct implementation with the naive CUDA baseline.

This reports speedups, not automatically awarded marks. Both files must contain
the complete suite, using matching inputs, GPU and timing settings.
"""
import argparse
import json
import math
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "common"))
from cases import GEMM_PERFORMANCE, ATTENTION_PERFORMANCE


def validate(data):
    if data.get("schema_version") != 1 or not data.get("all_passed"):
        raise ValueError("Both runs must pass every test, with schema_version=1")
    if data.get("suite") not in ("all", "performance") or data.get("case_filter"):
        raise ValueError("Use --suite all or performance without --case for both runs")
    timing = data.get("timing", {})
    mode, flush_bytes = timing.get("cache_mode"), timing.get("flush_bytes")
    if mode not in ("cold", "warm") or not isinstance(flush_bytes, int):
        raise ValueError("Cache policy/size metadata is missing; rerun with the current harness")
    if (mode == "cold" and flush_bytes <= 0) or (mode == "warm" and flush_bytes != 0):
        raise ValueError("Invalid eviction-buffer size for the recorded cache policy")
    performance = GEMM_PERFORMANCE if data["part"] == "gemm" else ATTENTION_PERFORMANCE
    expected = [(name, tuple(shape), "performance") for name, shape, _ in performance]
    found = [(row["name"], tuple(row["shape"]), row["category"]) for row in data["cases"]]
    if sorted(expected) != sorted(found):
        raise ValueError("Result does not contain the complete current case manifest")
    if not all(row["passed"] for row in data["cases"]):
        raise ValueError("A case failed")
    return {row["name"]: row for row in data["cases"] if row["category"] == "performance"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline", type=Path)
    parser.add_argument("candidate", type=Path)
    args = parser.parse_args()
    baseline, candidate = [json.loads(p.read_text()) for p in (args.baseline, args.candidate)]
    if baseline.get("implementation") != "example":
        raise ValueError("The baseline must be the instructor's naive example run")
    # References run outside timing. Student-only packages may use the explicit
    # Torch attention oracle while the instructor baseline uses validated CuTe.
    # Require identical task/tolerances/harness, not identical reference engines.
    # The full timing dictionary comparison also requires matching cache mode,
    # eviction buffer size, event methodology, warmups and sample counts.
    for key in ("part", "task_spec", "harness_sha256", "gpu", "gpu_uuid", "architecture", "compiler", "torch", "seed", "timing", "atol", "rtol"):
        if baseline.get(key) != candidate.get(key):
            raise ValueError(f"Mismatching {key}; rerun both under identical conditions")
    base_rows, candidate_rows = validate(baseline), validate(candidate)
    speedups = []
    print(f"Cache policy: {baseline['timing']['cache_mode']}; "
          f"eviction buffer: {baseline['timing']['flush_bytes'] / (1024 * 1024):g} MiB")
    print(f"{'Case':20s} {'Naive ms':>12s} {'Candidate ms':>14s} {'Speedup':>10s}")
    for name, base in base_rows.items():
        row = candidate_rows[name]
        if row["seed"] != base["seed"]:
            raise ValueError(f"Input seed mismatch for {name}")
        base_ms = base.get("mean_ms", base.get("median_ms"))
        candidate_ms = row.get("mean_ms", row.get("median_ms"))
        if not (isinstance(base_ms, (int, float)) and math.isfinite(base_ms) and base_ms > 0 and
                isinstance(candidate_ms, (int, float)) and math.isfinite(candidate_ms) and candidate_ms > 0):
            raise ValueError("Latency must be finite and positive")
        speedup = base_ms / candidate_ms
        speedups.append(speedup)
        print(f"{name:20s} {base_ms:12.6f} {candidate_ms:14.6f} {speedup:9.3f}x")
    geometric_mean = math.exp(sum(map(math.log, speedups)) / len(speedups))
    print(f"Geometric-mean speedup: {geometric_mean:.3f}x")
    print("Development evidence only; staff rerun code on controlled hardware for grading.")


if __name__ == "__main__":
    main()
