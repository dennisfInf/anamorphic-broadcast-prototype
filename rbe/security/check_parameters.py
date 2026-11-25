#!/usr/bin/env python3
"""Analytical parameter checks, not a lattice-hardness estimator.
Gaussian widths use exp(-pi*x^2/s^2), i.e. stddev=s/sqrt(2*pi).
Source: KLSS26 Lemma3.7/Cor5.6; reviewed RBE extension equations.
"""
import json, math
from pathlib import Path
import os

QMOD=281474976694273
PHI=256
EPS_BITS=208
QENC=2**32
KEY_QUERY_LIMIT=2**30
DEPTH_HYBRIDS=163

def eta2(dim):
    # GPV smoothing upper bound, with log1p avoiding giant intermediates.
    return (math.log(2*dim) + EPS_BITS*math.log(2) + math.log1p(2.0**(-EPS_BITS)))/math.pi

def calculate(n,g=4,B=30,Kbits=19,sigma=16,sigma_star=12,sigma_tilde=2**36):
    mA=2*n; mC=4*n; mstar=6*n+B; b=2**16; beta=b/2
    bar2=sigma_tilde**2-sigma**2
    # Use the source convolution: D_sigma + D_bar approximately D_tilde.
    rows=[]
    for label,r,w,samples in [('ordinary-source',n,7*n*B,mstar),('mode-root',n,(6*n+1)*B,mstar),('anamorphic-root',n-g,(6*n+1)*B,mstar),('ordinary-leaf',n,n*B,mstar+KEY_QUERY_LIMIT)]:
        # The leaf embeds every honest-key response, with secret-only leakage.
        dim=PHI*(r+samples)
        lhs=w*PHI**2*beta**2/bar2 + 1/sigma**2
        rhs=1/(2*eta2(dim)+sigma_star**2)
        rows.append(dict(label=label,rank=r,samples_module=samples,scalar_dimension=PHI*r,scalar_samples=PHI*samples,nonzero_leakage_entries=w,smoothing_lower=math.sqrt(2*eta2(dim)),sigma_star=sigma_star,llwe_lhs=lhs,llwe_rhs=rhs,llwe_ratio=lhs/rhs,smoothing_ok=sigma_star>=math.sqrt(2*eta2(dim)),llwe_ok=lhs<=rhs))
    # Stable delta = 1/2 sqrt((1+x)^phi -1) with x=(q^n-1)/K^mC.
    logx=n*math.log(QMOD)-mC*Kbits*math.log(2)
    x=math.exp(logx)
    delta=.5*math.sqrt(math.expm1(PHI*math.log1p(x)))
    rho=PHI*sum(QMOD**(j-n) for j in range(g))
    err=((7*162+2)*PHI**1.5*n*sigma*b/2)+math.sqrt(PHI)*sigma_tilde
    root_kappa=4096; other_kappa=128; E=160
    root_compression=PHI*(b/2)*(3*mA)*root_kappa
    other_compression=other_kappa*(1+PHI*(b/2)*(3*(162-1)*mA+n))
    covert_bound=(1+mC*PHI*2**Kbits/2)*(E+root_kappa)
    # Per-coefficient discrete Gaussian Chernoff bound: 2 exp(-pi E^2/sigma^2)
    log2_error_tail=1-math.pi*(E/sigma)**2/math.log(2)
    log2_carrier_tail_union=math.log2(QENC*mC*PHI)+log2_error_tail
    hash_setup_loss=2*g*delta
    llwe_stat_loss=QENC*2*(4*2.0**(-EPS_BITS)/(1-2.0**(-EPS_BITS)))
    conv_loss=2*QENC*B*16*2.0**(-EPS_BITS)
    ordinary_common_hybrid_stat=QENC*(4*DEPTH_HYBRIDS*2.0**(-EPS_BITS)/(1-2.0**(-EPS_BITS))+16*B*2.0**(-EPS_BITS))
    ordinary_stat=2*ordinary_common_hybrid_stat
    gaussian_draws=PHI*((162+1)*n+162*3*mA+n+B)
    sampler_loss=2*QENC*gaussian_draws*2.0**(-250)
    return dict(n=n,g=g,B=B,q=QMOD,phi=PHI,mA=mA,mC=mC,mstar=mstar,K=2**Kbits,sigma=sigma,sigma_star=sigma_star,sigma_star_stddev=sigma_star/math.sqrt(2*math.pi),sigma_tilde=sigma_tilde,log2_queries=32,eps_bits=EPS_BITS,llwe=rows,hash_delta_log2=math.log2(delta),rank_failure_log2=math.log2(rho),ordinary_uncompressed_bound=err,ordinary_with_compression_bound=err+root_compression+other_compression,ordinary_with_compression_ratio=(err+root_compression+other_compression)/(QMOD/4-1),covert_bound=covert_bound,covert_bound_ratio=covert_bound/(QMOD/4-1),covert_tail_union_log2=log2_carrier_tail_union,mode_stat_loss_log2=math.log2(hash_setup_loss+llwe_stat_loss+conv_loss),ordinary_stat_loss_log2=math.log2(ordinary_stat),ordinary_common_hybrid_stat_loss_log2=math.log2(ordinary_common_hybrid_stat),honest_key_generation_query_limit=KEY_QUERY_LIMIT,ordinary_ind_cpa_bounded_lwe_coefficient=2*KEY_QUERY_LIMIT,ordinary_ind_cpa_gaussian_lwe_coefficient=2*QENC*DEPTH_HYBRIDS,ordinary_ind_cpa_base_gaussian_lwe_coefficient=2*QENC*(DEPTH_HYBRIDS-1),ordinary_ind_cpa_leaf_gaussian_lwe_coefficient=2*QENC,ordinary_leaf_samples_module=mstar+KEY_QUERY_LIMIT,ordinary_privacy_admissibility="At each challenge, occupied selected leaves use distinct honest public-key values (A1); reject inadmissible challenges in every hybrid without conditioning key sampling. No honest secret key is disclosed. For ordinary privacy, registration and challenge identities lie in one prefix-free identity space.",gaussian_draws_per_ciphertext=gaussian_draws,sampler_two_experiments_loss_log2=math.log2(sampler_loss),mode_privacy_under_128_if_each_lwe_adv_2_minus_192=math.log2(hash_setup_loss+llwe_stat_loss+conv_loss+2*QENC*2.0**(-192))<-128)

if __name__=='__main__':
    reports=[calculate(n) for n in [7,9,11,13,14,16]]
    (Path(os.environ.get('RBE_CHECK_OUTPUT', Path(__file__).parent)) / 'analytical-profiles.json').write_text(json.dumps(reports,indent=2)+'\n')
    for a in reports:
        print('n=%d, reduced=%d, LLWE=%s, log2(delta)=%.2f, ordinary ratio=%.3f, covert ratio=%.3f, statlog2=%.2f'%(a['n'],a['n']-a['g'],all(z['llwe_ok'] and z['smoothing_ok'] for z in a['llwe']),a['hash_delta_log2'],a['ordinary_with_compression_ratio'],a['covert_bound_ratio'],a['mode_stat_loss_log2']))
