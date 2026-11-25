#!/usr/bin/env python3
"""Reproduce the conservative depth-16 public-residual decision bound."""
import json
import runpy
from pathlib import Path
import os

root = Path(__file__).resolve().parents[1]
evaluate = runpy.run_path(str(root / 'correctness' / 'calculate.py'))['evaluate']
profile = evaluate(depth=16)
# Termwise: the public residual has no final e_ell*x or codec*x term,
# and its leaf gadget remainder is bounded by b/2 rather than b.
one = int(profile['ordinary_total_bound'])
result = {
    'depth': 16,
    'one_public_residual_noise_bound': one,
    'two_public_residual_noise_bound': 2 * one,
    'q_over_4_minus_1': profile['q_over_4_minus_1'],
    'decision_radius_sufficient': 2 * one < profile['q_over_4_minus_1'],
    'scope': 'Termwise bound on the source Gaussian norm events, with the separately stated sampler allowance. Does not certify the original q/1024 diagnostic threshold.',
}
assert result['decision_radius_sufficient']
(Path(os.environ.get('RBE_CHECK_OUTPUT', Path(__file__).parent)) / 'opus3-probe-bound.json').write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps(result, indent=2))
