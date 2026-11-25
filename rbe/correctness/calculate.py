#!/usr/bin/env python3
"""Reproduce conservative correctness bounds; no lattice-hardness estimate."""
import json
import math
from pathlib import Path
import os


def evaluate(n=16, phi=256, depth=162, batch=30, sigma=16,
             sigma_tilde=2**36, key_width=2**19, root_drop=13,
             other_drop=8, coefficient_error=160, queries=2**32):
    q = 281474976694273
    ell_a = 2
    b = 65536
    ma, mb, mc = ell_a*n, n, 2*ell_a*n
    # q = 1 mod 2^(D+1) for both chosen codecs: centered codec error <=2^(D-1).
    for drop in (root_drop, other_drop):
        assert q % 2**(drop+1) == 1
    kr, ko = 2**(root_drop-1), 2**(other_drop-1)
    # Source Theorem 5.2, after normalizing all encryption products.
    ordinary = ((3*ell_a+1)*depth+2)*phi**1.5*n*sigma*b/2 + math.sqrt(phi)*sigma_tilde
    # Every polynomial factor and every vector entry is accounted for.
    compression = ko + phi*(b/2)*(3*ma*kr+(3*(depth-1)*ma+mb)*ko)
    covert = (1+mc*phi*key_width/2)*(coefficient_error+kr)
    gaussians_per_query = phi*((depth+1)*n+depth*3*ma+mb+batch)
    # Leading term of source failure bound; omitted terms are listed separately.
    log2_ordinary_leading_failure = math.log2(queries*batch)-phi
    log2_ordinary_second_failure = math.log2(queries*batch*(depth+2))-phi*n
    log2_ordinary_third_failure = math.log2(queries*batch*depth)-3*ell_a*phi*n
    log2_covert_error_union = math.log2(2*queries*mc*phi)-math.pi*(coefficient_error/sigma)**2/math.log(2)
    return {
        'parameters': dict(q=q,n=n,phi=phi,depth=depth,batch=batch,sigma=sigma,
                           sigma_tilde=sigma_tilde,key_width=key_width,root_drop=root_drop,
                           other_drop=other_drop,coefficient_error=coefficient_error,queries=queries),
        'dimensions': dict(m_A=ma,m_B=mb,m_C=mc),
        'q_over_4_minus_1': q/4-1,
        'ordinary_noise_bound': ordinary,
        'ordinary_compression_bound': compression,
        'ordinary_total_bound': ordinary+compression,
        'ordinary_fraction_of_decoding_radius': (ordinary+compression)/(q/4-1),
        'covert_noise_compression_bound': covert,
        'covert_fraction_of_decoding_radius': covert/(q/4-1),
        'log2_ordinary_failure_terms_across_queries': [log2_ordinary_leading_failure, log2_ordinary_second_failure, log2_ordinary_third_failure],
        'log2_covert_error_union_across_queries': log2_covert_error_union,
        'sampled_gaussian_coefficients_per_query': gaussians_per_query,
        'log2_total_sampler_error_one_experiment_TV_2_neg_250': math.log2(queries*gaussians_per_query)-250,
        'log2_total_sampler_error_two_experiments_TV_2_neg_250': math.log2(2*queries*gaussians_per_query)-250,
        'ordinary_correctness_inequality_passes': ordinary+compression < q/4-1,
        'covert_correctness_inequality_passes': covert < q/4-1,
        'scope': 'Ideal Gaussian correctness plus stated per-coefficient sampler TV. Rank failure and hardness are separate.'
    }


if __name__ == '__main__':
    result = evaluate()
    here = Path(os.environ.get('RBE_CHECK_OUTPUT', Path(__file__).resolve().parent))
    (here/'bounds.json').write_text(json.dumps(result,indent=2)+'\n')
    (here/'codec-sweep.json').write_text(json.dumps([evaluate(other_drop=d) for d in [8,9,10,11]],indent=2)+'\n')
    print(json.dumps(result,indent=2))
