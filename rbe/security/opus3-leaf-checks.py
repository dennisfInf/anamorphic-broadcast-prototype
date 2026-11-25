#!/usr/bin/env python3
"""Independent third-audit LLWE check; not a lattice-hardness estimator.

Widths use exp(-pi*x*x/s*s). The leaf uses only secret leakage F,
so its nonzero-entry budget is n*B rather than (3*m_A+n)*B.
"""
import json
import math
from pathlib import Path
import os

PHI = 256
B = 30
N_K = 2**30
EPS_BITS = 208
SIGMA = 16
SIGMA_STAR = 12
SIGMA_TILDE = 2**36
BETA = 2**15


def check(n, samples, weight):
    dimension = PHI * (n + samples)
    eta_squared = (
        math.log(2 * dimension)
        + EPS_BITS * math.log(2)
        + math.log1p(2.0**-EPS_BITS)
    ) / math.pi
    lhs = weight * PHI**2 * BETA**2 / (
        SIGMA_TILDE**2 - SIGMA**2
    ) + SIGMA**-2
    rhs = 1 / (2 * eta_squared + SIGMA_STAR**2)
    smoothing_lower = math.sqrt(2 * eta_squared)
    return {
        "rank": n,
        "module_samples": samples,
        "scalar_samples": PHI * samples,
        "smoothing_dimension": dimension,
        "nonzero_leakage_entries": weight,
        "eta_squared_upper": eta_squared,
        "smoothing_lower": smoothing_lower,
        "smoothing_ok": SIGMA_STAR >= smoothing_lower,
        "llwe_lhs": lhs,
        "llwe_rhs": rhs,
        "llwe_ratio": lhs / rhs,
        "llwe_ok": lhs <= rhs,
    }


def profile(n):
    base = 6 * n + B
    return {
        "rank": n,
        "honest_key_query_cap": N_K,
        "base": check(n, base, 7 * n * B),
        "leaf_padded": check(n, base + N_K, n * B),
        "leaf_minimal": check(n, n + B + N_K, n * B),
        "generic_weight_at_leaf_for_comparison": check(
            n, base + N_K, 7 * n * B
        ),
    }


if __name__ == "__main__":
    result = {
        "source": "RBE Appendix A.3, PDF pp.18-19; LSW Example 1 and Theorem 3, PDF pp.15-16",
        "epsilon_bits": EPS_BITS,
        "sigma": SIGMA,
        "sigma_star": SIGMA_STAR,
        "sigma_tilde": SIGMA_TILDE,
        "profiles": [profile(n) for n in (7, 9, 11, 13, 14, 16)],
    }
    (Path(os.environ.get("RBE_CHECK_OUTPUT", Path(__file__).parent)) / "opus3-leaf-checks.json").write_text(json.dumps(result, indent=2) + "\n")
    final = result["profiles"][-1]
    for label in ("base", "leaf_padded", "leaf_minimal", "generic_weight_at_leaf_for_comparison"):
        entry = final[label]
        print(f"{label}: m={entry['module_samples']}, w={entry['nonzero_leakage_entries']}, "
              f"sqrt(2)*eta<={entry['smoothing_lower']:.15f}, "
              f"LLWE ratio={entry['llwe_ratio']:.15f}")
