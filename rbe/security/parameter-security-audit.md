# Joint RBE/anamorphic parameter audit

Scope: mathematical security conditions and conservative numerical parameter checks. This is not a certification of an implementation or a proof of concrete lattice hardness. Formal manuscript files were not edited.

## Source baseline and why it does not transfer unchanged

The authors' pinned `params/param_selection.ipynb` (upstream commit 5f1e7416893de7950d231f11b1207dd5bc02924d, cell 6) records full estimator outputs of 304.6 bits for their bounded-key LWE family and 134.4 bits for the Gaussian LWE family. Its parameters are phi=256, n=7, ell_A=2, B=30, q=281474976694273, b=65536, sigma=16, sigma_star=2^3.75 and sigma_tilde about 2^34.964301. These recorded outputs are source evidence, not a new estimate. The latest run by the estimation agent is recorded separately.

Our affine restriction consumes g=|S| ceil(L/phi) module dimensions. Its strongest relevant Gaussian LWE family has rank n-g, rather than n. Therefore the old prototype with n=7 and g=2 or 4 leaves ranks 5 or 3. The source's 134.4-bit estimate at rank 7 cannot support a 128-bit claim for these smaller ranks. Even preserving rank 7 would not give a comfortable many-ciphertext reduction margin.

## Hardness instances and sample counts

1. Ordinary user public keys: module LWE of rank n, with n module samples, modulus q and coefficient-uniform secret/error on the exact source interval (-b/2,b/2] intersect Z, namely {-32767,...,32768}, with 65536 outcomes. The scalarized estimator input has dimension and sample count phi*n.
2. Ordinary non-leaf ciphertext steps: LLWE at rank n with m_star=6n+B module samples. Its Gaussian LWE reduction uses secret/error D_sigma_star and leakage weight at most 7nB, with coefficient bound beta=b/2.
3. Ordinary leaf step: rank-n Gaussian LWE with up to m_plus=m_star+N_K module samples. The reduction embeds all honest key-generation responses before the adaptive challenge. Its leakage is (F;0), with at most nB nonzero entries. The direct minimum n+B+N_K is padded to m_plus.
4. Anamorphic mode and hidden-message privacy: root error-only LLWE at ranks n and n-g, with the same m_star and w_root=(6n+1)B. Thus estimate the reduced rank using all m_star samples; do not reuse the source's sample count 72 when n changes.

Estimation uses scalarized structured/module instances and attack cost models; it is evidence under the modeling assumptions, not a reduction from module LWE to unstructured LWE or a proof that all attacks cost that much.

## Exact source sufficient inequalities

For each instance of rank r and module sample count m, let D=phi*(r+m). Use m=m_star for the root and non-leaf steps and m=m_plus for the ordinary leaf. For eps=2^-208 use the smoothing upper bound eta^2 <= log(2D(1+1/eps))/pi. A conservative sufficient condition is:

- sigma_star >= sqrt(2)*eta;
- w*phi^2*beta^2 / bar_sigma^2 + 1/sigma^2 <= 1/(2*eta^2+sigma_star^2).

Use w_ord=7nB for ordinary non-leaf steps, w_leaf=nB for the ordinary leaf, and w_root=(6n+1)B for the channel. We choose bar_sigma^2=sigma_tilde^2-sigma^2 and the convolution D_bar_sigma + D_sigma approximately D_sigma_tilde, so the same noise specification works for both reductions. Sigma=16 also exceeds sqrt(2)*eta_eps(Z^phi), as required for convolution. Gaussian width uses density proportional to exp(-pi*x^2/s^2), so the estimator standard deviation is s/sqrt(2*pi), not s.

Source: KLSS26 Lemma 3.7, Corollary 5.6 and the underlying Lai--Swarnakar--Woo Leaky LWE Theorem 3. The dimension used here is conservatively at least the dimensions in the source smoothing prerequisites.

## Explicit operating envelope

- At most 2^30 registered users, supported by B=30. Separately impose at most N_K=2^30 honest key-generation oracle queries; the population bound does not itself cap oracle invocations.
- At most Q=2^32 challenge ciphertexts within one family, with the intended covert group, first identity branch and payload length fixed before setup.
- Identity length at most 162 in the ternary tree.
- For ordinary privacy, A1 holds at each challenge: occupied selected leaves for its identity use distinct honest public-key values. The low-level prototype permits reuse and does not enforce A1. A registry manager maintaining one subregistry per identity satisfies A1 automatically.
- g_max=4. Examples: at most four recipients for an L<=256-bit payload, or at most two recipients for L<=512. The software must enforce g<=4, not merely g<n.
- For channel privacy, ordinary keys are generated honestly and may be exposed as in the channel theorem; intended covert keys and sender randomness remain secret. The ordinary RBE security game discloses no secret key corresponding to any honest key-generation-oracle response replaced in the global key hybrid. Merely keeping the eventual target key secret is insufficient. The channel proof does not use that key-replacement hybrid.
- Public matrices are sampled explicitly uniformly. A seeded expansion would need a computational pseudorandomness/hybrid argument and is not silently covered by the literal uniform-matrix proof.

## Analytically feasible candidate family

Retain q=281474976694273, phi=256, b=65536, ell_A=2, B=30, sigma=16. Set sigma_star=12, sigma_tilde=2^36, K=2^19, eps=2^-208 and g_max=4. Increase n until the reduced-rank lattice estimate meets the chosen primitive target. Candidate n=16 gives m_A=32, m_C=64, m_star=126 and rank n-g=12.

For n=16, source-aligned normalized arithmetic and root bit-dropping 13 / all other component bit-dropping 8 yield:

- Ordinary non-leaf, enlarged-leaf and channel LLWE inequalities pass. The largest ratio is 0.9900673140023465, for the leaf at m_plus=126+2^30 and leakage weight nB=480.
- Per-intended-key statistical distance delta_K approximately 2^-221.
- For channel privacy, fixed-setup hashing, LLWE statistical and convolution terms combined are below 2^-166 for Q<=2^32. Ordinary message IND-CPA has its separately stated statistical bound below 2^-164.
- Conservative ordinary residual bound, including all witness-vector lengths in compression, is about 85.3% of q/4-1.
- Conservative covert error bound with E=160 is about 26.0% of q/4-1.
- The source ordinary error-norm failure probability union-bounded over 2^32 encryptions is about 2^-219, assuming the actual Gaussian samplers meet their stated distributions to the allocated statistical tolerance.
- Covert Gaussian error tails are smaller still; the common coefficient event is enough for all recipient keys, so no unjustified independence assumption is needed.

If each relevant primitive distinguishing advantage were at most 2^-192 at the running times and resources of all reductions, including transcript simulation, the LLWE reduction and rank padding, the channel advantage bound is below 2^-128 with a large margin. This is a conditional theorem calculation. A reported 192-bit estimated attack cost is not itself such an advantage theorem. For ordinary RBE under A1, globally replace the N_K honest public keys once and then apply H=163 depth hybrids per ciphertext: H-1 base-instance steps and one enlarged-leaf step. The two message branches give coefficients 2N_K, 2Q(H-1) and 2Q for bounded-key, base Gaussian and enlarged-leaf Gaussian advantages, respectively. The separate 2^-192 hypothesis must cover the enlarged leaf at its full reduction resources as well.

`check_parameters.py` reproduces all numerical inequality checks and writes `analytical-profiles.json`. It is intentionally separate from the lattice estimator.

## Implementation conditions

The source prototype's three encryption inverse-NTT calls leave a factor phi on the root and lower-level signal terms, whereas key generation, registry operations and decryption use normalized arithmetic. Normalizing those encryption products gives the equations analyzed in the source and eliminates the extraneous phi in the channel's affine constraint. Without this change, the lower approximation/user-key error terms must be separately analyzed with phi amplification; the calculations above do not cover them. The correctness agent independently identified and implemented this alignment.

The original 64-bit floating-point CDF and tail truncation cannot simply be declared 128-bit statistically faithful across the operating envelope. Likewise the original large-noise MW sampler only guarantees some realized width in a range, not the precise requested width. New sampled distributions must be specified and their approximation error budgeted. Increasing dimensions and passing functional tests alone does not discharge this obligation.

The original root/lower compression settings are not appropriate for a strict all-parameter proof. In particular the source's Appendix A.2 compression bound omits witness vector lengths. The updated joint parameters use the complete coefficient-vector bound, and both ordinary and anamorphic modes must use identical codecs and rank-failure fallback.

## Sources

- KLSS26 paper: https://eprint.iacr.org/2026/717
- Authors' prototype: https://github.com/jnsiemer/scalable_rbe_prototype
- Leaky LWE published reprint: https://aaltodoc.aalto.fi/bitstreams/3059de93-bf66-48bb-866f-c471c831a980/download
- Local reviewed construction: ../../../research/draft-sections/06_rbe.tex

## Independent estimator update

The estimation agent completed the full estimator suite for n=16, g=4, remaining rank 12, sigma_star=12 under the pinned estimator: the conservative classical Core-SVP minimum is 219.0 bits and the corresponding quantum Core-SVP cost model gives 198.75 bits. These are actual estimator runs, not interpolations from the source baseline. The candidate n=13,g=4 gives only about 149.8 classical bits and is therefore rejected against the chosen 192-bit attack-cost screening threshold. This threshold and the separate 2^-192 primitive-advantage hypothesis are independent modeling choices; neither implies the other. The detailed estimator artifacts and full-suite results are maintained under the separate estimation directory; these figures must not be reported as an exhaustive quantum-attack analysis.

The n=16 profile's scalarized reduced instance has dimension 3,072 and 32,256 samples. The non-leaf ordinary Gaussian instance has dimension 4,096 with the same 32,256 samples, and the bounded-user-key instance has dimension/sample count 4,096. The enlarged leaf instead has up to 274,877,939,200 scalar samples. Its estimator executions and failures are recorded separately in `../../../research/history/experiments/implementation-results/rbe/results/opus3-leaf-estimator.json`; the old small-sample estimates do not cover it.

## Meaning of each selected parameter

| Parameter | Value | Reason |
|---|---:|---|
| Ring degree phi | 256 | Retain the source cyclotomic ring and existing NTT machinery. |
| q | 281474976694273 | Retain the source prime modulus, congruent to 1 modulo 512. |
| Module rank n | 16 | Reserve four constrained dimensions and retain rank 12 for the weakest LWE estimate. |
| ell_A, m_A | 2, 32 | Source approximate gadget design, with m_A=n*ell_A. |
| m_B | 16 | Source user-key matrix width equals n. |
| m_C | 64 | Two inactive ternary branches, 2m_A. |
| m_star | 126 | All root columns plus terminators, 3m_A+B. |
| m_plus | 126+N_K | Proof-only ordinary leaf sample bound, embedding every honest key response. |
| b | 65536 | Source gadget radix; q<b^3, and the lowest gadget digit is omitted. |
| B | 30 | Up to 2^30 ordinary registered users in the source wrapper. |
| Maximum identity length | 162 | Ternary identities cover at least 2^256 identity strings. |
| sigma | 16 | Gaussian parameter for ordinary lower random vectors and small errors. |
| sigma_tilde | 2^36 | Enlarged message noise supports both source and root LLWE reductions with eps=2^-208. |
| sigma_star | 12 | Reduced-to Gaussian LWE parameter, not a noise added by the encryption implementation. |
| K | 2^19 | Covert-key coefficient box width; balances statistical public-key uniformity against decryption error. |
| g_max | 4 | Enforced bound on |S|*ceil(L/256), rather than the merely algebraic bound g<n. |
| Root bits dropped | 13 | Rigorous root compression and covert recovery margin. |
| Other bits dropped | 8 | Includes all lower blocks and message block, satisfying the full ordinary witness-vector compression bound. |
| eps | 2^-208 | Internal smoothing/convolution statistical tolerance. |
| Q | 2^32 | Explicit ciphertext budget used for statistical and hybrid accounting. |
| N_K | 2^30 | Separate limit on honest key-generation oracle calls; not implied by the registered population. |

The user-facing payload/group examples should be exact: four recipients with one 256-bit block, two with two blocks (up to 512 bits), or one with four blocks (up to 1,024 bits). A 257-bit message consumes two blocks, not one. Sending to more recipients or longer payloads requires increasing n and rerunning both the analytic checks and hardness estimator.

## Final statistical-budget refinement

The selected internal smoothing/convolution tolerance is **eps=2^-208**. This leaves sigma_star=12 unchanged and passes the base ordinary-step inequality with lhs/rhs=0.961032457151613. The enlarged leaf is checked separately below. Write Q<=2^32 for ciphertext challenges, N_K<=2^30 for honest key-generation queries, and H=163 for depth hybrids. N_K is an explicit oracle-query allowance, separate from the 2^30 registered population.

The source leaf transition needs an additional registration condition. In Figure 3, the adversary may register one honest key for the same identity in two selected subregistries. Public witnesses then give

    R_t = d_t - sum_(j<ell) c_j^T*u_(t,j)
        = r_ell^T*y_t + Delta*m_t + nu_t.

If y_t=y_tprime comes from reusing the same key response, subtracting these public residuals cancels the shared pad. The result reveals per-slot message differences and distinguishes the claimed independent-uniform endpoint. When the slots carry one common message, that message cancels as well. The counterexample therefore leaves unrestricted common-message confidentiality unsettled. The retained probe and its exact-codec control are in `implementation/extension/tests/opus3_duplicate_key.rs`.

We restrict this proof route by requiring A1 at every challenge: the honest public-key values at occupied selected leaves for the target identity are pairwise distinct. Reject an inadmissible challenge in every hybrid. The source registers raw vectors while the prototype registers canonical prescribed encodings. Map each returned value to its first key-oracle occurrence, retaining the underlying raw pool vector privately; A1 then makes the selected indices distinct. All matrix columns remain independently sampled, including accidental equalities. There is no conditioning on collision freedom and no extra collision-loss claim. This changes the ordinary-security scope; it does not repair the unrestricted source BLE theorem or add enforcement to the prototype.

For ordinary privacy, registration and challenge identities must also lie in one prefix-free identity space, for example one fixed length. This additional condition is not needed by the root-only channel games. A sufficient local condition for the leaf step is that a challenged identity cannot be a strict prefix or strict extension of a registered identity in a selected subregistry. Otherwise the terminal node may be an internal hash or a traversal may pass a key leaf. The source's registered-only prefix-freeness wording (PDF page 10) is insufficient for arbitrary challenge identities. This package states the precondition explicitly.

Fix either ordinary-message branch under A1. Appendix A.3 globally replaces at most N_K honest key responses in G_-4 to G_-3 (PDF page 18), at cost N_K*a_key and no LLWE statistical term. Each ciphertext uses at most B convolution replacements and ell+1<=H LLWE replacements (page 19). Non-leaf replacements use the base 126-sample instance. For the final replacement, the reduction embeds the public user-key matrix, terminators and all N_K key-response columns before the adaptive challenge. A1 makes its selected key-column indices distinct. The leaf has at most m_plus=126+N_K samples and leakage weight nB, since L_ell=(F;0). Semi-adaptive LLWE permits choosing that leakage after the matrix is known and before its challenge response is received. In each adjacent non-leaf step, absorb the full product noise bar_e=sum_j bar_e_j into Lambda_j+bar_e; the individual support-restricted bar_e_j alone would leave continuing-slot leakage unmasked. This exact regrouping does not add noises or change the convolution count. The explicit simulator and encoding correspondence are given in `opus3-leaf-derivation.txt`.

Let a_G and a_G_plus be the base and enlarged-leaf Gaussian advantages, at their complete reduction resources. Lemma 3.7 charges a_G (or a_G_plus) plus 4*eps/(1-eps) per LLWE step, and Corollary 3.4 charges 16*eps_c per convolution. Thus

    D_branch <= N_K*a_key + Q*(H-1)*a_G + Q*a_G_plus
                + Q*(4H*eps/(1-eps) + 16B*eps_c),
    Adv_ordinary <= 2*N_K*a_key + 2*Q*(H-1)*a_G + 2*Q*a_G_plus
                    + 8QH*eps/(1-eps) + 32QB*eps_c.

Both message branches reach the same common hybrid with uniformized honest-key responses and uniform uncompressed ciphertexts. No honest secret key is disclosed. Compression applies the same deterministic map to both endpoints. The channel game instead generates ordinary keys honestly, may expose them, and replaces only the root; it does not use this leaf transition.

The H coefficient still counts the LLWE steps. The printed Corollary 5.6 uses N_K, while the global key replacement has no LLWE call. The explicit count gives the unchanged ordinary statistical term 2^-164.8553. Source: [author manuscript, Figure 3, Appendix A.3 and Lemma 3.7](https://kclpure.kcl.ac.uk/ws/portalfiles/portal/373075977/main.pdf). The complete corrected simulation and exact independent calculations are in `opus3-leaf-derivation.txt` and `opus3-leaf-checks.json`.

For m_plus=126+2^30, the smoothing dimension is 256*(16+m_plus). The leaf's smoothing lower bound is 10.440018513374458<12 and its LLWE ratio is 0.9900673140023465<1. Using the generic weight7nB would fail at 1.0009246380257668. The secret-only leaf weight is therefore essential; no encryption distribution or ciphertext dimension changes.

For anamorphic-message privacy, each message branch reaches the same message-independent endpoint through Q reduced-rank LLWE replacements. Hence Adv_am <= 2Q*a_(n-g)+delta_mode. The two-sided setup, LLWE and convolution allowance is the same delta_mode as for mode privacy. Sampler approximation and CSPRNG terms remain separate.

Both reductions can use **the same** proof-only decomposition bar_sigma^2=sigma_tilde^2-sigma^2 and sigma_add=sigma=16. The ordinary proof requires this added width; the root proof permits any independent added noise satisfying its convolution condition and therefore also permits 16. Both convolution widths exceed their smoothing thresholds. A half-and-half root decomposition is optional and is not used in this certificate. No encryption distribution changes between these proof-only decompositions.

The MPFR/integer-CDF Gaussian sampler samples each specified discrete Gaussian within total variation tau+2^-251/(1-tau)<2^-250, where tau=exp(-64*pi)/(8*pi). Across two experiments with 2^32 full-depth ciphertexts, its approximation contributes less than 2^-194.84. See `sampler-analysis.txt` for the audit and exact integer-CDF equivalence.

## Final source-distribution and guard check

The native ordinary key/error sampler draws Uniform::new_inclusive(0,65535) and subtracts 32767. Its exact support is {-32767,...,32768}, with 65536 points, matching the source definition U(R_b)=(-b/2,b/2] for b=65536. The coefficient bound remains b/2=32768. There is no need to modify this sampler or the bounded-key LWE reduction. The authors' estimator notebook instead used the inclusive interval [-32768,32768], with 65537 points, so an exact implementation estimate must use the actual native endpoints. The independent estimation agent is recording that separate exact input.

A final code inspection confirmed the K=2^19 guard, g<=4 guard, refusal of aggressive message compression, and depth/codec checks. The fixed profile uses normalized encryption products, explicit uniform public matrices, and the audited Gaussian samplers. These checks address the specific parameter/distribution issues found in this audit; they are not a general implementation or side-channel audit. Registration inputs in this implementation profile are outputs of the prescribed honest key generator, stored as precomputed negative gadget decompositions. The update interface does not validate bounded coefficients of arbitrary externally supplied predecomposed public keys; consequently this profile does not implement the source's broader untrusted raw-public-key interface.

## Aggregate conditional calculation

Under A1, if each relevant primitive advantage, including a_G_plus at 126+N_K samples, is at most 2^-192 at the complete reduction resources, ordinary message IND-CPA has bound about 2^-151.64891. These resources include the full key pool, transcript simulation, LLWE reduction and any rank padding. The numerical value is unchanged under this stronger, explicitly enlarged assumption. Under ideal independent randomness, the sum of the separate ordinary-privacy, mode-privacy, anamorphic-message-privacy and decryption-failure bounds, together with a conservative sampler allowance, has log2 -151.63121381410429 and is strictly below 2^-151. This includes Q<=2^32, the separately imposed N_K<=2^30, depth162, g<=4 and eps=eps_c=2^-208. `aggregate-bounds.py` reproduces the calculation. This is neither an operation-count-to-advantage conversion nor a new theorem for an arbitrary composed game. Ordinary privacy discloses no secret key corresponding to any honest key-generation-oracle response replaced globally; channel privacy permits exposure of honestly generated ordinary keys and keeps intended covert keys and sender randomness secret. Inferring ordinary-message privacy in the anamorphic branch through mode privacy likewise retains the unexposed intended-covert-key restriction. An implementation-level total must additionally include a separate aggregate CSPRNG distinguishing term for the complete compared transcripts. That term is not numerically bounded here and is not part of the Gaussian sampler approximation allowance.
