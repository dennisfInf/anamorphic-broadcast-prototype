#!/usr/bin/env python3
"""Sum separate conditional ideal-randomness bounds; NOT an attack-cost conversion."""
import json, math
from pathlib import Path
import os
p=Path(os.environ.get('RBE_CHECK_OUTPUT', Path(__file__).parent))
profile=next(a for a in json.loads((p/'analytical-profiles.json').read_text()) if a['n']==16)
Q=2**32; NK=2**30; H=163; B=30; g=4; eps=2.0**-208
primitive=2.0**-192
# A separate hypothesis covers the enlarged ordinary leaf instance.
leaf_primitive=2.0**-192
ordinary_common_hybrid=NK*primitive+Q*(H-1)*primitive+Q*leaf_primitive+Q*(4*H*eps/(1-eps)+16*B*eps)
# Message IND-CPA compares each message branch to the common random endpoint.
ordinary=2*ordinary_common_hybrid
mode=2*Q*primitive+2.0**profile['mode_stat_loss_log2']
anamorphic_message=mode
ordinary_failure=Q*B*(2.0**-256+(162+2)*2.0**(-256*16)+162*2.0**(-6*256*16))
covert_failure=g*2.0**profile['hash_delta_log2']+2.0**profile['rank_failure_log2']+2.0**profile['covert_tail_union_log2']
# Covers three pairs of security experiments and extra correctness sampling.
sampler=4*2.0**profile['sampler_two_experiments_loss_log2']
components={'ordinary_privacy':ordinary,'mode_privacy':mode,'anamorphic_message_privacy':anamorphic_message,'ordinary_decryption_failure':ordinary_failure,'covert_decryption_failure':covert_failure,'conservative_sampler_allowance':sampler}
total=sum(components.values())
report={
    'uninstantiated_bounds': {
        'mode_privacy': 'Q*(a_n + a_n_minus_g) + delta_mode',
        'anamorphic_message_privacy': '2*Q*a_n_minus_g + delta_mode',
        'anamorphic_message_primitive_coefficient': 2*Q,
        'ordinary_message_privacy': '2*N_K*a_key + 2*Q*(H-1)*a_G + 2*Q*a_G_plus + delta_ordinary',
        'ordinary_privacy_admissibility': 'At every challenge, occupied selected leaves use distinct honest public-key values (A1), with inadmissible challenges rejected in every hybrid and unconditioned key sampling. The low-level prototype does not enforce this registry restriction.',
        'resource_scope': 'Each primitive advantage is evaluated at its complete reduction resources, including transcript simulation, LLWE reduction and any rank padding.',
        'mode_each_primitive_coefficient': Q,
        'ordinary_key_coefficient': 2*NK,
        'ordinary_gaussian_coefficient': 2*Q*H,
        'ordinary_base_gaussian_coefficient': 2*Q*(H-1),
        'ordinary_leaf_gaussian_coefficient': 2*Q,
        'ordinary_base_samples_module': profile['mstar'],
        'ordinary_leaf_samples_module': profile['mstar']+NK,
        'ordinary_gaussian_coefficient_scope': 'Total coefficient for the coarser bound using a_G_plus for every Gaussian step; the precise bound splits base and leaf instances.',
        'mode_statistical_loss_log2': profile['mode_stat_loss_log2'],
        'ordinary_statistical_loss_log2': profile['ordinary_stat_loss_log2'],
        'implementation_terms': 'Sampler approximation and aggregate CSPRNG replacement must be accounted for separately from these ideal-distribution reduction bounds.',
    },
    'illustrative_primitive_advantage': '2^-192; an explicit hypothesis, not established by estimated attack costs or a generic time/advantage conversion.',
    'assumption': 'Under A1, each relevant primitive distinguishing advantage, including a_G_plus with 126+N_K module samples, is <= 2^-192 at the full reduction resources, including the complete key pool, transcript simulation, LLWE reduction and any rank padding. Estimator costs do not imply this inequality.',
    'randomness_model': 'Ideal independent uniform random words. All numerical fields in this report use this model.',
    'implementation_total_requires': 'Add a separate aggregate CSPRNG distinguishing term for the complete compared transcripts. This term is not numerically bounded here and is not included in the sampler allowance.',
    'honest_key_query_limit': NK,
    'ciphertext_query_limit': Q,
    'ordinary_common_hybrid_distance_log2': math.log2(ordinary_common_hybrid),
    'scope': 'Sum of separate reviewed-game security and correctness bounds, not a new arbitrary composition theorem.',
    'identity_space_scope': 'For ordinary privacy, registration and challenge identities lie in one prefix-free identity space; fixed-length encodings suffice. The API does not enforce this precondition.',
    'ordinary_exposure_scope': 'No honest secret key is disclosed. A1 separately requires pairwise distinct honest public-key values among occupied selected leaves for the challenge identity.',
    'channel_exposure_scope': 'Ordinary keys are generated honestly and may be exposed. Intended covert keys and sender coins remain secret; the ordinary key-replacement hybrid is not used.',
    'registration_input_scope': 'Prescribed honest key-generation outputs in predecomposed representation. Arbitrary untrusted predecomposed public keys are outside this implementation profile.',
    'components_log2': {k:math.log2(v) for k,v in components.items()},
    'ideal_randomness_sum_log2': math.log2(total),
    'sum_log2': math.log2(total),
    'sum_strictly_below_2_minus_151': total<2.0**-151,
    'sum_strictly_below_2_minus_129': total<2.0**-129,
    'sum_strictly_below_2_minus_128': total<2.0**-128,
}
(p/'aggregate-bounds.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
