# RBE correctness and parameter alignment

The selected profile has module rank n=16. This audit records the source arithmetic and finite sufficient bounds. Earlier n=13 experiments and native-sampler diagnostics belong to the original full parameter package. Current source hashes and corrected-source tests are recorded in `../provenance.json` and `../../../research/history/experiments/implementation-results/rbe/results/review-fixes-*.log`. `../../../research/history/experiments/implementation-results/rbe/correctness/live-code-alignment.json` is an older static-check snapshot, with its original hashes retained. Lattice-hardness estimates are separate.

## Source parameters

The pinned source uses ring degree 256, prime modulus 281474976694273, module rank 7, gadget base 65536, two retained gadget digits, three branches, 30 subregistries, identity depth 162, small Gaussian parameter 16, and large requested Gaussian parameter 33519950000. Thus m_A=14, m_B=7, m_C=28. Ordinary user-key and key-error coefficients are sampled from the 65536 consecutive integers -32767 through 32768; the claimed bound b/2=32768 holds. Both encryption secrets r_j and small errors are Gaussian-sampled, despite a stale comment suggesting a uniform distribution for r.

The source encodes bit 1 as floor(q/2), drops 16 bits from ordinary blocks, and has a separate optional aggressive 43-bit message-block codec. The aligned profile disables that aggressive option. The previous extension kept native inverse-NTT scaling, used K=2^22 and dropped 13 root bits, but retained 16 lower bits. Successful tests did not certify these choices jointly.

## Normalization

The authors' key generation and non-root registry updates normalize before inverse NTT; encryption does not. This multiplies encryption's signal terms by the ring degree. Principal ordinary terms still cancel, but gadget approximation and user-key-noise terms inherit a factor of 256. The paper's error bound does not describe this arithmetic directly.

The aligned copy normalizes the NTT product immediately before inversion at the three encryption sites: each c_j, c_ell, and d. This yields precisely the paper's ring-product equations. The affine constraints consequently use H^T r_0=Delta*a, with no factor of 256. Decryption and key generation already use the required normalization. Both ordinary and anamorphic modes use the same corrected host arithmetic. This is an explicit modification of the authors' prototype, not a claim that the old measurements instantiated the paper exactly.

The preliminary n=13 tests checked this normalization change with the earlier samplers and seeded setup. The current n=16 tests use explicit public matrices and the JointGaussian sampler; their execution is recorded in `../../../research/history/experiments/implementation-results/rbe/results/review-fixes-extension-tests.log`.

## Finite correctness bounds for the selected n=16 profile

Take n=16, phi=256, ell=162, B=30, b=65536, m_A=32, m_B=16, m_C=64, sigma=16, and sigma_tilde=2^36. Take covert coefficient width K=2^19, root drop count 13, other drop counts 8, and ordinary non-aggressive message compression. All ordinary message coefficients remain bits.

After normalization, the source Theorem 5.2 gives an uncompressed coefficient bound

    E_ord = ((3*ell_A+1)*ell+2)*phi^(3/2)*n*sigma*b/2 + sqrt(phi)*sigma_tilde
          = 40,132,174,413,824.

Unlike the source Appendix A.2 display, the compression bound must count every entry of each inner product. With root reconstruction error kappa_root and other reconstruction error kappa_other,

    E_compress = kappa_other
               + phi*(b/2)*(3*m_A*kappa_root
                            +(3*(ell-1)*m_A+m_B)*kappa_other).

Here q is 1 modulo both 2^14 and 2^9. The centered codec therefore has kappa_root=4096 and kappa_other=128. The compression bound is 19,911,468,384,384; the sum is 60,043,642,798,208, below q/4-1=70,368,744,173,567.25. It uses 85.33% of the sufficient decoding radius. Dropping nine lower bits does not satisfy this inequality (108.94% of the radius); eight is the selected value. No independence or uniformity of the compression error is assumed.

The ordinary bound applies to the affine-uniform root because r_0 cancels identically. No Gaussian norm event is imposed on r_0. Lower secrets and all errors retain their original Gaussian distributions. There are only ell lower-secret norm events and one leaf-error norm event; retaining the source coefficient ell+2 rather than ell+1 is conservative. The resulting ideal-Gaussian correctness failure bound per ciphertext is

    B * (2^-phi + (ell+2)*2^(-phi*n) + ell*2^(-3*ell_A*phi*n)).

For at most Q=2^32 ciphertexts the leading term is approximately 2^-219.09. The other two terms are much smaller. Sampler approximation error is additional, not hidden in this bound.

For each intended covert recipient, a coefficient of the residual is Delta*a plus e[f]-k^T e and its compressed counterpart. Conditioning on |e|_infinity<=160, its error is bounded by

    (1 + m_C*phi*K/2)*(160+4096) = 18,279,380,816,032,

which uses 25.98% of the decoding radius. This is a worst-case bound for all covert keys in the support; it does not assume typical keys or random reconstruction errors. The selected JointGaussian implementation has hard cutoff 8*sigma=128 for these coefficients, so 160 is a conservative 10*sigma threshold; in actual sampling, this coefficient-bound event holds deterministically. For the ideal D_Z,16 distribution used by the proof, the elementary tail bound 2*exp(-pi*(160/16)^2), union bounded over Q*m_C*phi coefficients, is below 2^-406. Rank failure and statistical public-key replacement must be included separately using the security analysis. Exact finite numbers are in bounds.json, generated by calculate.py.

## Samplers

The native CDF uses floating-point probabilities, 64-bit cumulative masses, and tailcut 3.63*s. It assigns one of 2^64 outcomes to an extra endpoint because its final CDF entry is 2^64-1 and the sample comparison is strict. The native Gaussian parameter 16 distribution has mean about -1.30e-14 and variance 40.74366543152453 in this build; this is close numerically to 16^2/(2*pi), but closeness at ordinary numerical precision is not a 128-bit statistical certificate.

The native Micciancio-Walter sampler targets only a lower bound on width. For requested width 33519950000 it combines 16 base samples, and its ideal realized width is 33521655375.538002. Its maximum support under the actual base CDF is 487850419875; saturating integer arithmetic is inactive at those parameters. The exact discrete convolution is not the exact requested Gaussian. These native-sampler diagnostics are retained in the original full parameter package.

The aligned sampler has now been independently audited in `../security/sampler-analysis.txt`. Its exact output distribution has per-coefficient statistical distance less than 2^-250 from the prescribed Gaussian, assuming MPFR directed-rounding correctness and uniform RNG words. Integer CDF and rejection sampling realize the same normalized, downward-rounded weights. Rejection iterations do not add a separate approximation term: the bound applies to each returned sample. The program uses a CSPRNG, whose computational assumption is separate.

There are 4,660,736 Gaussian coefficients per full-depth encryption at n=16. A union bound over Q=2^32 calls gives less than 2^-195.84 total sampler distance in one experiment, and less than 2^-194.84 when comparing two experiments. This count conservatively includes the initially sampled root vector later discarded in anamorphic mode. Ordinary user keys, covert keys, explicit public matrices, and terminators use bounded or modular uniform sampling; registering users introduces no Gaussian approximation terms. The earlier hypothetical 2^-192 sampler calculation is preserved only in the exploratory n=13 artifacts.

## Explicit public setup

The new copy stores independently sampled public A and B matrices, without disclosing a short generator seed. All algorithms use those arrays directly. Terminators remain sampled by the original setup. This matches the explicit-uniform-matrix argument; it avoids conflating a public seed expansion with statistical uniformity.

At 48 packed bits per coefficient, A is 2,359,296 bytes and B is 393,216 bytes. The 30 terminators add 737,280 bytes. Total coefficient payload is 3,489,792 bytes, excluding dimension/framing metadata; the corresponding u64 arrays use 4,653,056 bytes. See explicit-public-setup.json. The corrected n=16 test suites and five-workload benchmark have completed; see `../provenance.json` and `../../../research/history/experiments/implementation-results/rbe/results/`.

## Live-code verification and correlation conditions

The checked live copy uses JointGaussian for both Gaussian widths, explicit public PP=(MatNTT,MatNTT), n=16, sigma=16, sigma_tilde=2^36, default root drop 13, other drop 8, and g<=4. The API rejects K other than 2^19 and disallows aggressive pruning, root drops above 13, and encryption/witness depths outside 1 through 162. Smaller root drop counts preserve the inequalities. The current benchmark fixes the exact selected root drop count 13. Each encryption product is normalized before inversion. The CRT constraint has no ring-degree multiplier.

The bounds require ordinary keys and witnesses produced by the stated algorithms. Key generation samples x,z coefficientwise in [-32767,32768]. Pruned base-b decomposition produces digits in [-32768,32767]; after the sign used by the witness, every witness coefficient has magnitude at most 32768. Its omitted digit gives a gadget residual f of the same magnitude. Thus f-z has magnitude at most b=65536. All these bounds hold for every registry value, including adaptively updated registries; no distributional claim about a witness is needed.

For a fixed ordinary recipient, the uncompressed error is

    eta = e_tilde - sum_{j=1}^{ell-1} r_j^T f_j
          - r_ell^T(f_ell-z) - sum_{j=0}^{ell-1} e_j^T u_j - e_ell^T x.

The root r_0 disappears exactly. Registry witnesses, f_j, x and z may be mutually correlated and are fixed before the fresh encryption coins. The source bound only applies norm inequalities to these fixed bounded vectors and a union bound to Gaussian vector norms; it does not assume independence among witness coordinates or compression errors. D_R,s is explicitly the product of phi coefficient Gaussians in the source definition. The norm-tail dimensions are phi for each message error polynomial, phi*n for lower secrets and the leaf error, and 3*ell_A*phi*n for each internal error vector. Consequently there is no additional factor phi missing from the displayed vector-tail union. The factor B is conservative because the lower random vectors are shared across subregistries. For adaptive ciphertext queries, conditioning on the preceding history preserves the fresh-coin distributions, so a Q-fold union bound remains valid.

For covert correctness, no distributional assumption on the intended key beyond its coefficient support is used. The single root-error coefficient event bounds all intended keys at once; multiplying again by the number of recipients would be unnecessary. Rank failure is an independent setup issue handled in the parameter/security report.

## Serialized sizes

The full-depth selected profile serializes 19,950,080 ciphertext bytes in both modes. The root contains 96*256 coefficients at 35 bits (107,520 bytes). The other 161 internal blocks contain 96*256 coefficients each at 40 bits (19,783,680 bytes); the leaf contributes 20,480 bytes and the message block 38,400 bytes. These are exact payload counts for the existing per-block serialize function; it writes no separate mode flag or framing header. The previous n=7 extension used 7,009,216 bytes with root drop 13 and lower drop 16. The authors' original all-drop-16 profile used 7,005,184 bytes. The selected profile is 2.846 times the previous extension's ciphertext size, primarily because it increases the rank and retains more precision at the lower levels.

For each recipient, each identity-branch family, and each 256-bit payload polynomial, the selected covert public key uses 24,576 packed bytes and the secret key uses 38,912 bytes. These are computed by the actual public-mod-q and coefficient-box packing conventions. Multiply by ceil(L/256) for longer payloads and by three when all first-identity-digit families are provisioned. The previous n=7, K=2^22 extension used 10,752 public and 19,712 secret bytes per such unit. See serialized-sizes.json for all component counts.

These counts should not be compared as an equal-security benchmark: the previous profile was a functionality profile, while the selected profile pays for the jointly checked inequalities and the reduced-rank hardness target. All final performance comparisons should measure ordinary and anamorphic modes under this same selected host profile.

## Limits

These inequalities certify correctness for the specified ideal distributions and stated sampler-distance budget. They do not establish lattice hardness, the Leaky-LWE reduction parameters, side-channel security, or production readiness. Those are independent checks. No empirical decryption success rate substitutes for any of these bounds.
