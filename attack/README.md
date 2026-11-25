# Simulation of the attack on DPPY25 (Section 3, Proposition 3.1)

`dppy25_attack.py` runs the mode experiment of Proposition 3.1 against the Dual-Regev
anamorphic broadcast construction of Do, Persiano, Phan and Yung (EUROCRYPT 2025,
DPPY25), and the same experiment against Construction 1 of the paper (the construction
that `basic/` implements, here re-implemented at toy size). `test_dppy25_attack.py`
tests the implementation. Python 3.8 or later, standard library only; matplotlib only
for `--plot`.

```sh
python3 attack/dppy25_attack.py                       # 1000 experiments per mode, about 40 s
python3 attack/dppy25_attack.py --plot dppy25-attack --json dppy25-attack.json
python3 attack/test_dppy25_attack.py                  # 29 tests, about 11 s
```

`experiments/run_all.sh` runs the simulation too, after the benchmarks (the part
`attack` of `SCHEMES`, with `ATTACK_TRIALS` and `ATTACK_SEED`), and writes `attack.txt`
(the report below), `attack.json`, `attack.log` and `attack.pdf` / `attack.png` into the
result folder. So the bench image's default run includes it, and
`docker run --rm -v "$PWD/results:/results" -e SCHEMES=attack anamorphic-bench` runs it
alone into a new `results/run-<UTC time>`. `test_all.sh` runs the tests where python3
is installed, which includes the dev image.

## What it does

Each experiment generates fresh keys, gives the adversary the ordinary secret key and
answers one encryption query, in the ordinary or the anamorphic mode:

- **DPPY25**, as defined in their paper (Dual Regev in their Section 2.3, the anonymous
  multi-channel broadcast AnonMCBE in 3.4, the anamorphic broadcast mode in 4.2) and
  recalled in Section 3. Ordinary keys: A uniform in Z_q^{n x m}, e ~ D_{Z^m,r},
  u = A e; an ordinary ciphertext encrypts a fresh uniform bit, (p, c) =
  (A^T s + x, u^T s + x' + Delta b). Broadcast setup: u_Broad uniform and short x_i with
  A_Broad x_i = -u_Broad, dk_i = (1 || x_i) for N = 6 users. The query carries one hidden
  bit am_j to each of two disjoint channels of three recipients: y is uniform subject to
  <dk_i - r_j, y> = 0 with r_j = (am_j, 0, ..., 0) for every channel j and recipient i in
  aset_j, and act = y + e_Broad with rounded Gaussian e_Broad. A recipient outputs 0 if
  <dk_i, act> is closer to 0 than to y_1, else 1.
- **Construction 1** on the Dual-Regev scheme of Definition 2.4 (binary k, b = A k)
  with |S| = 3 recipients of L = 2 bits (d = 6 equations): binary khat_{i,a},
  bhat_{i,a} = A khat_{i,a}, s uniform subject to (a_1 - bhat_{i,a})^T s = Delta am_a
  for every (i, a), then the ordinary encryption of a uniform bit with this s. A
  recipient rounds r[1] - khat_{i,a}^T r.

The adversary D computes the residual z = c - <key, p> mod q with the ordinary key (e,
or k) and outputs 1 ("ordinary") if z is within circular distance E of 0 or of
Delta = floor(q/2). The script reports Pr[D = 1] in both modes with Wilson 95%
intervals and the advantage with a Newcombe 95% interval, for E = floor(q/5) as in
Proposition 3.1 and for E = 8 standard deviations of the decryption error
eta = x' - <key, x>, which the adversary computes from its key (sigma_eta =
sigma (1 + ||key||^2)^(1/2), for sigma^2 the variance of a rounded error), next to the
proposition's lower bound 1 - (4E + 2)/q on the advantage against DPPY25. It also
checks DPPY25's parameter conditions and every ciphertext (ordinary decryption of
ordinary ciphertexts, and of both modes for Construction 1; hidden decryption by every
recipient), counts the event Bad of the proof (d = (-e, 1) in the span W of the
constraint vectors), and compares the two modes' residuals with Kolmogorov-Smirnov
tests, of z itself and of its distance from {0, Delta}; the statistic of the latter is
the best advantage of a threshold test over all E.

## How the simulation follows DPPY25

It differs from their definitions in four places, none of which the attack uses:

- **Transposition.** Their A is m x n with u = A^T e; here A is n x m with u = A e,
  and A_Broad x_i = -u_Broad is their A_Broad^T x_i = -u_Broad. The ciphertexts are the
  same vectors.
- **Keys without a trapdoor.** Their KeyGen samples x_i from D_{Lambda^perp_{-u}(A_Broad), S}
  with the trapdoor of A_Broad. The simulation plants the keys instead: x_i ~ D_{Z^m,r}
  first, then u_Broad uniform and A_Broad uniform subject to A_Broad x_i = -u_Broad for
  all i. Above the smoothing parameter of Lambda^perp(A_Broad) this has the same
  distribution (the GPV argument), and the default r = 6 is above the usual estimate of
  that parameter, 4.6 (see below). Their Gaussian parameter S may be a matrix; the
  simulation uses the spherical r.
- **Decoding against act_1.** Their Decrypt compares <dk_i, ct> with 0 and with y_1,
  which the recipient sees only as act_1 = y_1 + e_1, so the simulation compares with
  act_1. The difference is noise of a single coordinate.
- **No ⊥.** aDec returns a bit, never ⊥; the experiment never asks for ⊥.

Their parameter conditions (their Section 2.3, with omega(sqrt(log n)) instantiated by
sqrt(log2 n)) hold at the defaults, and the report prints them:

```
q >= 5 r (m + 1): 1.074e+09 against 3.075e+04, holds
m >= 2 n log2 q: 1024 against 960, holds
alpha = sigma/q <= 1/(r sqrt(m + 1) sqrt(log2 n)): 2.98e-09 against 0.002603, holds
r >= smoothing of Lambda^perp(A), about q^(n/m) sqrt(ln(2m(1 + 2^40))/pi): 6 against 4.642, holds
```

The last line is not one of their conditions: it is the one under which the planted
keys follow trapdoor sampling, with the Gaussian-heuristic estimate of the smoothing
parameter at epsilon = 2^-40. Earlier versions of this script used n = 32, which
breaks m >= 2 n log2 q (1920 needed) at this q; the script now warns outside the
conditions.

## Tests

`test_dppy25_attack.py` (29 tests, standard library `unittest`, about 11 s) checks the
implementation from the bottom up, at the default parameters where the property is
exact and at small moduli where it is a distribution:

- linear algebra and samplers: solutions of M x = t satisfy it and are uniform over all
  solutions (chi-square over the 25 solutions of an equation over Z_5), rank deficiency
  is refused, row-space membership is exact, D_{Z,r} has mass proportional to
  exp(-pi x^2 / r^2) and variance r^2 / (2 pi), the errors are rounded Gaussians;
- the ordinary scheme: u = A e, p = A^T s + x, c = u^T s + x' + Delta b, the residual
  is Delta b + x' - e^T x, and decryption is correct;
- the broadcast keys: (u_Broad | A_Broad) dk_i = 0 with x_i of the expected norm, and
  A_Broad uniform;
- anamorphic encryption: every constraint <dk_i - r_j, y> = 0 holds exactly, so
  <dk_i, y> = am_j y_1; y is uniform over all solutions; act = y + e_Broad with one
  rounded-Gaussian e_Broad, and <dk_i, act> = am_j y_1 + <dk_i, e_Broad>;
- anamorphic decryption: the rule on hand-checked values, and all 300 hidden bits of 50
  queries decoded by their recipients;
- what the attack reads: z = c - e^T p = <d, act> with d = (-e, 1); z uniform when d
  is outside W, with and without the broadcast noise; and on the event Bad, built by
  hand (e_1 = 0, x_1 = (-e_2, ..., -e_m, 1), am_1 = 1, so d = dk_1 - r_1), z is only
  the broadcast noise, as the proof of Proposition 3.1 says;
- the experiment at n = 8, m = 256, q = 65521 (which meet the conditions too), 1500
  trials: D accepts every ordinary ciphertext and an anamorphic one at the rate
  (4E + 2)/q for both thresholds, and accepts both modes of Construction 1 alike;
- the default parameters meet the conditions (and n = 32 does not), the statistics
  helpers reproduce Newcombe's published intervals, and the command line works.

Statistical tests use fixed seeds and reject at p = 1e-4, so they are deterministic.
To check that the tests can fail, each of twelve planted bugs was run against them,
and each made tests fail: the constraint (am_j, x_i) instead of (1 - am_j, x_i), the 1
at the end of dk_i, the decoding rule inverted, A_Broad x_i = +u_Broad, the residual
c + e^T p, a non-uniform y, the broadcast noise left out (three ways), D_{Z,r} with
standard deviation r, u computed from |e|, and Bad never detected.

## Result (default parameters, seed 2025)

```
Functionality (every ciphertext checked):
  DPPY25: ordinary decryption 1000/1000, hidden bits 6000/6000 decoded by their recipients
  Construction 1: ordinary decryption 2000/2000 (both modes), hidden bits 6000/6000 decoded
  Bad (d in W) in DPPY25's queries: 0 of 1000

Residual test D, output 1 = 'ordinary' if z is within E of 0 or Delta (Wilson / Newcombe 95% intervals):
                                                   E         Pr[D=1 | ord.]        Pr[D=1 | anam.]                advantage  1-(4E+2)/q
  DPPY25, E = floor(q/5)                   2.147e+08  1.000 [0.996, 1.000]  0.791 [0.765, 0.815]  +0.209 [+0.185, +0.235]    0.200000
  DPPY25, E = 8 sd of the error                 1967  1.000 [0.996, 1.000]  0.000 [0.000, 0.004]  +1.000 [+0.995, +1.000]    0.999993
  Construction 1, E = floor(q/5)           2.147e+08  1.000 [0.996, 1.000]  1.000 [0.996, 1.000]  +0.000 [-0.004, +0.004]           -
  Construction 1, E = 8 sd of the error        582.8  1.000 [0.996, 1.000]  1.000 [0.996, 1.000]  +0.000 [-0.004, +0.004]           -

Kolmogorov-Smirnov tests, ordinary against anamorphic mode: of the residual z (its whole distribution),
and of its distance from {0, Delta} (the statistic is the best advantage of a threshold test over all E):
  DPPY25: residual D = 0.279 (p = 1.31e-34), distance D = 1.000 (p = 0)
  Construction 1: residual D = 0.031 (p = 0.716), distance D = 0.055 (p = 0.0939)
```

Against DPPY25 the residual of an anamorphic ciphertext is uniform: with E = floor(q/5),
D accepts it with probability 0.791 [0.765, 0.815], against the exact 4/5 of a uniform
residual, and accepts every ordinary ciphertext, an advantage of 0.209 [0.185, 0.235]
around the proposition's 1 - (4E + 2)/q = 0.2. With the tight threshold the advantage
is 1.000 [0.995, 1.000]: no anamorphic residual of the 1000 came within 1967 of 0 or
Delta. Bad never occurred, as its probability is at most q^(1-n) + q^(-n) + 2^(-Omega(n)).
Against Construction 1 the same test accepts both modes alike, and neither
Kolmogorov-Smirnov test separates their residuals: the constrained s cancels in
c - k^T r exactly as the uniform one does. The recipients of both schemes decoded every
hidden bit, so the attack is not an artefact of a broken implementation.

`--plot` draws this as one column (3.5 in): for each scheme, the fraction of residuals
within distance x of {0, Delta} against x/q on a logarithmic axis, ordinary in blue,
anamorphic in orange, with the two thresholds; Pr[D = 1] is each curve's value at E/q.

## Parameters and limits

Defaults: n = 16, m = 1024, q = 2^30 - 35 = 1073741789 (prime), errors rounded
Gaussians of standard deviation 3.2 (`--sigma`, that is alpha q), DPPY25's keys from
D_{Z,r} with r = 6 (`--r`), channels `--groups 3,3`, Construction 1
`--recipients 3 --bits 2`, `--trials 1000`, `--seed 2025`. These are toy parameters
without security: the attack uses only the ordinary key and the decryption residual,
and its advantage depends on the parameters only through E. Construction 1 runs with
the same A in each experiment as DPPY25's ordinary key, which leaves each scheme's own
distribution unchanged. Randomness comes from Python's `random.Random`, seeded for
reproducibility; it is a simulation, not an implementation for use.
