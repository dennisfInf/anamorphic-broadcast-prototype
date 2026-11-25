# Basic one-ciphertext construction in Rust

The maintained Rust implementation, with the independent Python core retained in [`one_ciphertext.py`](../../research/history/prototypes/python-reference/basic/one_ciphertext.py). It implements ordinary Dual-Regev encryption, the affine one-ciphertext anamorphic channel, group delivery of a common L-bit message, both decryption paths, seed expansion, and the native ciphertext codec. This uses the screened n1152 candidate; the historical n160 Python runner and its results remain in [`python-reference/basic/`](../../research/history/prototypes/python-reference/basic/README.md). Manuscript integration is separate.

Since 2026-10-26 every heavy operation exists in a single-thread and a multi-thread mode with identical results, and the rounded normal errors come from a precomputed 256-bit base table whose statistical distance to the manuscript's error distribution is bounded. Later the same day the arithmetic moved to established crates: `crypto-bigint` for every operation modulo q, GMP and MPFR through `rug` for primality and the multiprecision floats of the noise table, `bitvec` for key bits and `clap` for the command lines. The matrix products are now exact integer sums without hand-written reduction, budget or SIMD code. See [Execution modes](#execution-modes), [Arithmetic and kernels](#arithmetic-and-kernels), [Noise base table](#noise-base-table) and [Measured performance](#measured-performance).

## Run

Use the included `Cargo.lock` (Rust 2024 edition). Like Epoque, the crate needs a C compiler, make and the build prerequisites for `gmp-mpfr-sys`, which builds GMP and MPFR for `rug` (the first build takes several minutes). From this directory:

```sh
cargo test --locked
cargo test --locked --release
cargo clippy --locked --all-targets -- -D warnings
cargo run --locked --release --bin benchmark -- --repetitions 4 --warmups 1 --output /tmp/basic-rust-benchmark.json
cargo run --locked --release --bin experiments -- --reps 50 --output /tmp/basic-experiments.json
```

`experiments` times KGen, encryption and decryption and, for group sizes 128, 64, ..., 2 (profile g256, L = 2, 4, ..., 128 bits) and half the message length,
aKGen (seed keys, in batches of eight users), the group setup, aEnc and aDec, as
means over `--reps` runs. See [../experiments/README.md](../experiments/README.md).

The benchmark defaults to one and four recipients, each receiving two hidden bits. `--rows 1:1,4:2,4:8` selects other rows. Each row needs `recipients * bits <= 32` and at most `--population` recipients (8 by default). The library accepts validated parameters separately. The output file must be new; existing results are never overwritten. Benchmark JSON is checkpointed before work and after every complete row; `complete: false` means the run is unfinished. Timings exclude encoding and checks, and include encryption coin sampling. Setup, directory generation and audience-dependent elimination are reported separately. All timed ciphertexts, including warmups, are checked after timing. No secret keys or sender randomness are written to the result.

The benchmark measures both execution modes by default (`--modes single,multi`), interleaved iteration by iteration with the same seeds, and fails if the modes ever produce different keys or ciphertexts. `--population N` sets the number of registered hidden users (default 8, recipients per row at most N), and `--one-time N` repeats key generation, anamorphic key generation and the noise-table construction (default 3). Anamorphic encryption is timed in total and split into the broadcast step (`HiddenPlan::sample_secret`) and the Dual-Regev part (`encrypt_from_seed`). The multi-thread mode uses rayon's global pool, so `RAYON_NUM_THREADS` sets its size.

The noise base table can be inspected and checked against an independent MPFR evaluation, its exact statistical distance to the ideal law computed, and the noise that encryption adds tested statistically:

```sh
cargo run --locked --release --bin noise_table -- --dump /tmp/basic-noise-table.txt
cc -O2 tools/noise_table_mpfr.c -lmpfr -lgmp -o /tmp/noise_table_mpfr  # with Homebrew add -I$(brew --prefix)/include -L$(brew --prefix)/lib
/tmp/noise_table_mpfr var:34972416 111285 | cmp - /tmp/basic-noise-table.txt
cc -O2 tools/noise_distance_mpfr.c -lmpfr -lgmp -o /tmp/noise_distance_mpfr
/tmp/noise_distance_mpfr var:34972416 /tmp/basic-noise-table.txt 30358
cargo run --locked --release --example noise_stats -- --ciphertexts 330000 --output /tmp/noise-stats.json
python3 tools/noise_stats.py /tmp/noise-stats.json  # needs numpy and scipy
```

`noise_table` builds the maintained profile from its exact variance by default. `--sd S` builds the table for the double `S` instead, and the MPFR tools take the same width as `var:V` or as a decimal double.

To regenerate the Python fixtures from the independent reference:

```sh
python3 -B tests/regenerate_fixtures.py
cargo test --locked --test reference
```

The fixture generator accepts an explicit reference path when the Python implementation is elsewhere. These are public, deliberately small test instances, not secure parameters.

## Execution modes

Every operation that touches the public matrix takes a `Mode`:

| Operation | `Mode::Single` | `Mode::Multi` |
| --- | --- | --- |
| `keygen_with` | draws A and k from the caller's RNG, then `b = A k` | same draws, `b = A k` split by rows |
| `anamorphic_keygen_with` | draws all user keys, then every `b_{i,a} = A k_{i,a}` in one pass over A | same, split by rows |
| `encrypt_with`, `HiddenPlan::encrypt_with`, `encrypt_from_seed`, `native_from_coins_with` | one pass over A | header columns split by blocks, each block also samples its own noise |
| `NoiseTable::new`, `noise_table_with` | sequential | blocks of 4096 entries in parallel |

The functions without a mode argument (`keygen`, `anamorphic_keygen`, `encrypt`, `HiddenPlan::encrypt`, `native_from_coins`) keep their signatures and run single-threaded. For the n1152 profile the group setup (`prepare_hidden`), the broadcast step (`HiddenPlan::sample_secret`) and decryption take well under two milliseconds; with the 256 constraints of the g256 experiments the group setup takes about half a second and hidden decryption with L = 128 about 12 ms on one thread. Since 7 November 2026 they also have mode variants: `prepare_hidden_with` and `AffineFiber::new_with` clear each pivot column of the Gauss-Jordan elimination on the rayon pool, `HiddenPlan::sample_secret_with` (and `encrypt_with`) computes the pivot coordinates of s in parallel, and `hidden_decrypt_with`, `hidden_residuals_with` and `expand_seed_with` read the L key blocks of a seed and form their products with the header in parallel. The functions without the suffix run them single-threaded, and `src/bin/experiments.rs` uses the variants in its mode. Results never depend on the mode or on the number of threads. `tests/optimized.rs` checks this for 1, 2, 3 and 7 threads, `tests/modes.rs` checks the new variants for the same counts (128 constraints on n = 600), and the benchmark checks it for every timed ciphertext.

## Arithmetic and kernels

All arithmetic stays exact modulo q, and `tests/optimized.rs` compares keys, directories, eliminations, affine solutions, ciphertexts for explicit coins and residuals bit for bit with a frozen copy of the previous code (`tests/legacy/basic_v1.rs`), up to the production profile.

* Field. `src/field.rs` wraps `crypto-bigint` (RustCrypto): remainders use its precomputed reciprocal of q (division by an invariant single-limb divisor), sums, differences and negations use `add_mod`, `sub_mod` and `neg_mod`, and inverses use the constant-time `invert_odd_mod`. A product of two field elements is below 2^64 because q < 2^32, so it is formed exactly in u64 and then reduced. Primality of q is GMP's test through `rug`, which is certain below 2^51 and therefore for every u32 modulus.
* `A^T s` adds eight matrix rows per pass into exact integer accumulators, u64 when n (q - 1)^2 fits (for q = 67537879 up to 4044 rows, more than n = 1152) and u128 otherwise, and reduces each header coordinate once with `crypto-bigint`. The pass over the 140 MB matrix is the remaining cost.
* Binary products (`b = A k` in KGen and aKGen, `k^T r` in both decryptions) turn each key bit into a 32-bit mask (all ones or zero), so that `x & mask` equals `x k_j`, sum the selected entries exactly in u64 (u128 if m (q - 1) did not fit) and reduce once.
* The group setup runs the same Gauss-Jordan elimination with the same pivot rule, only on the columns right of the current pivot. A row update `x - c p` is computed as `(q - c) p + x`, which is exact in u64, and reduced by `crypto-bigint`.
* For the broadcast step the setup stores the reduced coefficients of the free columns contiguously and, for each payload bit a, the pivot offset `Delta * sum_j T[:, (j-1)L + a]`. Sampling `s` then costs n - d uniform field elements and d dot products of length n - d.
* Seed expansion reads the SHAKE256 stream through `bitvec` (least significant bit first).

The loops are portable Rust that the compiler vectorizes for the build target. There is no `unsafe` code, no intrinsics and no runtime CPU dispatch any more. No established crate offers modular matrix-vector products for a modulus chosen at run time, so the loops themselves remain, as plain exact integer sums.

## Noise base table

Definition 2.4 of the manuscript draws every error from N(0, q^2 sigma^2) rounded to the nearest integer. The library now samples exactly this distribution from a precomputed cumulative distribution table (CDT) of `|e|` with 256-bit entries, `T_k = floor(2^256 P(|e| <= k) / P(|e| <= B))`, plus a guide table indexed by the leading 16 bits of the uniform input and one sign bit (see `src/noise.rs` for the construction and the error analysis). The construction runs entirely in MPFR (through `rug`): 384-bit floats, every operation rounded toward zero, MPFR's correctly rounded `exp`, and the positive series for the mass of each integer. The table is byte-identical to the one the previous hand-written 384-bit arithmetic produced. The sampling step itself stays in `src/noise.rs`, because no crate samples from a 256-bit table, and the normal samplers of crates such as `rand_distr` work in double precision, which would replace the 2^-241 bound by one near 2^-53. For the maintained profile, s = sqrt(1152 * 30358) = 5913.748..., taken exactly from the integer variance n (m + 1) = 34972416 (`Parameters::with_noise_variance`, see [Sampler verification](#sampler-verification)).

| Quantity | Value |
| --- | ---: |
| Largest error B (18.8 s) | 111285 |
| Ideal mass beyond B | at most 2^-260 |
| Statistical distance of one sample to the rounded normal | 2^-241.84 exactly (bound 2^-239.2) |
| Statistical distance of one ciphertext's m + 1 errors | at most 2^-226.95 (bound 2^-224.3) |
| Table plus guide table | 3,823,268 bytes |
| One-time construction (cached per width) | 0.21 s single thread, 0.14 s two threads (0.16 s and 0.09 s with the former hand-written arithmetic) |
| Cost per sample on this machine | 11.0 ns, against 17.3 ns for the previous `rand_distr::Normal` path |

The per-ciphertext distance is the `epsilon_smp` of the manuscript's remark on approximate Gaussian sampling, so it adds at most 2^-226.95 to the correctness failure per encryption and 2 Q 2^-226.95 to each privacy advantage. These distances are measured against the manuscript's exact width. They hold for parameters built with `Parameters::with_noise_variance`. With `Parameters::new` the width is the given double, and the double nearest sqrt(34972416) adds 2^-58.73 per error (2^-43.84 per ciphertext). The entries are bit-identical to an MPFR evaluation at 1024-bit precision (`tools/noise_table_mpfr.c`) for this profile (exact and double width) and for s = 0.25, 3, 77.5 and 1000, and `tests/noise_table.rs` pins their digests.

Randomness layout: encryption first draws s from the caller's RNG as before (n field elements, or the n - d free coordinates for aEnc), then a 32-byte noise seed. Header error j comes from ChaCha12 stream `floor(j / 1024)` of that seed and the scalar error from the next stream, so the errors are fixed by the seed whatever the mode or thread count. `noise_from_seed` returns them for tests. A sample uses one 64-bit word, and three more only when its leading 64 bits tie with a table entry (probability about 2^-47).

## Sampler verification

On 2026-10-26 the sampler was checked in four independent ways (records in [`../benchmarks/2026-10-26/sampler-verification/`](../benchmarks/2026-10-26/sampler-verification/)):

1. Target width. The manuscript sets (q sigma)^2 = n (m + 1) = 34972416. The double nearest its square root is 2^-57.68 too large, which moves every error by 2^-58.73 in statistical distance. The table is therefore built from the exact variance: it only needs 1/s^2 = 1/V, so no square root is rounded (`Parameters::with_noise_variance`, `NoiseTable::with_variance`). An integer variance with an exact double square root gives the same table as that double (tested for V = 9).
2. Every table entry. All 111285 entries equal `floor(2^256 erf((k + 1/2)/(s sqrt 2)) / erf((B + 1/2)/(s sqrt 2)))` as evaluated by MPFR's `erf` at 1024 bits, a method independent of the series in `src/noise.rs` (both now use MPFR, but different functions and precisions; rechecked after the move to MPFR on 2026-10-26). `tools/noise_distance_mpfr.c` then evaluates the exact total variation distance between the sampled law and the ideal rounded normal law at 1024 bits: 2^-241.84 per error (2^-241.85 for the double width against its own double). Against the exact width, the double-width table is 2^-58.73 away, matching the analytic value.
3. Exact inversion. A unit test (`sampler_inverts_the_production_table_exactly`) runs the sampler on 2464034 chosen 256-bit inputs, every entry and its two neighbours, both edges of all 65536 guide buckets and 2000000 random inputs (332963 of them through the comparison of the lower 192 bits), and compares the output and the number of words drawn with a plain binary search over the full entries. All agree, also for six smaller widths. So with uniform input bits the sampler returns `|e| = k` with probability exactly `(T_k - T_{k-1}) / 2^256`, with a fair sign for `k != 0`.
4. The noise encryption adds. `examples/noise_stats.rs` draws errors through `noise_from_seed`, the ChaCha12 path of encryption, and `tools/noise_stats.py` runs 32 tests against probabilities that scipy computes independently: chi-square and G-test on every value of `|e|`, equiprobable bins, Kolmogorov-Smirnov, tails beyond 3 to 6.5 s, sign balance and symmetry for every magnitude, the first four moments, lag 1 to 4 correlations, correlations across streams, seeds and the scalar stream, independence of consecutive signs and magnitude classes, and uniformity over the 1024 positions of a stream. For 1.0018 x 10^10 errors of the maintained profile the smallest p-value is 0.108, and 10^9 errors each at s = 1 and s = 77.5 also pass (smallest p-values 0.081 and 0.0097 among 32 tests).

Exactness assumes uniform input bits. The bits come from ChaCha12 keyed by a 32-byte seed from the caller's RNG, so the errors are computationally, not statistically, close to the ideal law.

## Measured performance

Measured on 2026-10-26 on this cloud machine (Intel Xeon at 2.1 GHz, two cores, AVX2 kernels, Rust 1.97.0), for the first version of that day; the [current code](#after-moving-the-arithmetic-to-crates) was measured separately. The previous code (the initial commit, single thread) and the new code ran alternately, three times each, with 21 measured repetitions after three warmups per run and the default 8 registered users. The values are medians over all runs. Raw results, including quartiles, are in [`../benchmarks/2026-10-26/basic/`](../benchmarks/2026-10-26/basic/).

| Operation | S:L | Before | Single thread | Multi thread (2) |
| --- | --- | ---: | ---: | ---: |
| Ordinary encryption | 1:2 | 40.5 ms | 7.23 ms | 4.57 ms |
| | 4:8 | 39.4 ms | 7.95 ms | 4.78 ms |
| Anamorphic encryption | 1:2 | 40.7 ms | 7.26 ms | 4.61 ms |
| | 4:2 | 41.2 ms | 7.44 ms | 4.41 ms |
| | 4:8 | 39.7 ms | 7.93 ms | 4.78 ms |
| | 8:4 | 41.5 ms | 7.90 ms | 4.89 ms |
| of which the broadcast step | 1:2 to 8:4 | not separable | 0.010 to 0.029 ms | 0.011 to 0.028 ms |
| Anamorphic key generation, 8 users | L = 2 | 988 ms | 59.6 ms | 37.6 ms |
| | L = 4 | 2,063 ms | 105 ms | 54.4 ms |
| | L = 8 | 4,096 ms | 196 ms | 101 ms |
| Key generation | | 271 to 293 ms | 178 to 182 ms | 174 to 187 ms |
| Group setup | d = 2 | 0.041 ms | 0.029 ms | same code |
| | d = 8 | 0.444 ms | 0.151 ms | same code |
| | d = 32 | 6.17 to 6.20 ms | 0.82 to 1.02 ms | same code |
| Decryption | | 0.036 to 0.041 ms | 0.025 to 0.033 ms | same code |
| Hidden decryption, seed key | L = 2 | 0.150 to 0.166 ms | 0.043 ms | same code |
| | L = 8 | 0.523 ms | 0.157 ms | same code |
| Hidden decryption, expanded key | L = 2 | 0.067 to 0.068 ms | 0.039 to 0.041 ms | same code |
| | L = 8 | 0.234 ms | 0.176 ms | same code |

Encryption is now about 5 times faster on one thread and 8 to 9 times faster on two, anamorphic key generation 16 to 21 times on one thread and 26 to 41 times on two. Encryption is bound by reading the 140 MB matrix once, so more threads help as far as memory bandwidth allows. Key generation is bound by drawing the 35 million matrix entries from the caller's RNG, which is sequential by design, so it gains little from a second thread. The broadcast step stays below 0.6% of an anamorphic encryption. Sizes are unchanged: the ciphertext has 121,432 bytes, the anamorphic public key has L n field elements per user (for L = 2 that is 2,304 elements, 73,728 bits at four bytes each), and a user secret key has 256 bits as a seed or L m bits expanded.

### After moving the arithmetic to crates

The table above records the first version of 2026-10-26, with hand-written Barrett reduction, lazy-reduction budgets and AVX2 kernels. The current code takes its arithmetic from crates (see [Arithmetic and kernels](#arithmetic-and-kernels)). Both versions were measured on the same machine (Intel Xeon at 2.1 GHz, two cores, Rust 1.97.0), alternately, three runs each, with 11 measured repetitions after two warmups per run, rows 1:2, 4:2, 4:8 and 8:4 and 8 registered users. The values are pooled medians. Raw results are in [`../benchmarks/2026-10-26/crates/`](../benchmarks/2026-10-26/crates/).

| Operation | S:L | Hand-written, single | Crates, single | Hand-written, multi (2) | Crates, multi (2) |
| --- | --- | ---: | ---: | ---: | ---: |
| Ordinary encryption | 1:2 | 7.10 ms | 8.01 ms | 4.54 ms | 5.95 ms |
| | 4:2 | 6.58 ms | 7.86 ms | 4.03 ms | 5.65 ms |
| | 4:8 | 6.83 ms | 8.13 ms | 4.36 ms | 5.40 ms |
| | 8:4 | 7.66 ms | 7.94 ms | 4.73 ms | 5.11 ms |
| Anamorphic encryption | 1:2 | 7.21 ms | 8.03 ms | 4.36 ms | 5.79 ms |
| | 4:2 | 6.53 ms | 7.98 ms | 3.92 ms | 5.28 ms |
| | 4:8 | 6.94 ms | 8.25 ms | 4.21 ms | 5.47 ms |
| | 8:4 | 7.73 ms | 7.76 ms | 4.58 ms | 4.78 ms |
| of which the broadcast step | all rows | 0.011 to 0.029 ms | 0.010 to 0.032 ms | 0.013 to 0.030 ms | 0.010 to 0.032 ms |
| Anamorphic key generation, 8 users | L = 2 | 52.4 ms | 151 ms | 26.9 ms | 80.2 ms |
| | L = 4 | 93.8 ms | 292 ms | 50.3 ms | 168 ms |
| | L = 8 | 191 ms | 562 ms | 105 ms | 294 ms |
| Key generation | | 185 to 210 ms | 183 to 197 ms | 183 to 190 ms | 169 to 178 ms |
| Noise table, once per standard deviation | | 154 to 164 ms | 200 to 217 ms | 88 to 92 ms | 120 to 135 ms |
| Group setup | d = 2 | 0.030 ms | 0.050 ms | same code | same code |
| | d = 8 | 0.145 ms | 0.586 ms | same code | same code |
| | d = 32 | 0.82 to 1.02 ms | 8.3 to 8.9 ms | same code | same code |
| Decryption | | 0.025 to 0.032 ms | 0.029 to 0.030 ms | same code | same code |
| Hidden decryption, seed key | L = 2 | 0.041 to 0.053 ms | 0.114 to 0.124 ms | same code | same code |
| | L = 8 | 0.154 ms | 0.454 ms | same code | same code |
| Hidden decryption, expanded key | L = 2 | 0.038 to 0.042 ms | 0.048 ms | same code | same code |
| | L = 8 | 0.139 ms | 0.173 ms | same code | same code |

Per message the cost is small: encryption is 0 to 22% slower on one thread and 5 to 40% slower on two, ordinary decryption is unchanged, and hidden decryption with a seed key takes about 0.07 to 0.3 ms longer because the key bits are now read one by one through `bitvec`. The one-time operations lost more. Anamorphic key generation is about 3 times slower, because the AVX2 kernel shared every matrix load among eight keys and summed in 32-bit lanes. The group setup is up to 10 times slower, because every element update now ends in a constant-time `crypto-bigint` reduction instead of lazily reduced u64 rows, but it still stays below 9 ms for 32 constraints. The noise table takes about 1.3 times as long in MPFR. Key generation is unchanged, since drawing the matrix from the caller's RNG dominates it. As before, every timed ciphertext was decrypted and round-tripped through the codec (624 ordinary decryptions, 12,792 hidden bits and 468 round trips over the three runs), and the two modes produced identical ciphertexts in all 156 iterations.

## Parameters and correspondence

The maintained performance profile is:

| Quantity | Value |
|---|---:|
| Native dimension n | 1152 |
| Minimum remaining dimension | 1120 |
| Header length m | 30357 |
| Prime q | 67537879 |
| Maximum constraints | 32 |
| Absolute noise standard deviation | sqrt(1152 * 30358) |
| Registered hidden users | 8 |
| Ciphertext bytes | 121432 |
| Matrix storage (excluding allocator overhead) | 139885056 bytes |
| Noise base table and guide table | 3823268 bytes |

These dimensions are unchanged by the port. The estimator and ideal statistical/correctness analysis remain in `../../lab-resource/2026-10-18_21-32-02-mythos-audit/local-fixes/evidence/basic-parameters/`. They analyse the ideal error distribution. The finite sampler's distance to it is bounded in [Noise base table](#noise-base-table), and neither proves a deployment security level.

`PublicKey` stores A in a contiguous row-major array. `HiddenPlan` borrows its public key, so it cannot accidentally encrypt under another key or copy the matrix per query. The matrix is sampled once uniformly and is never resampled on rank failure. The plan constructs rows A[:,0] - directory[i][a], solves the same affine equations by modular elimination, stores the free-column coefficients and per-bit pivot offsets, and samples each free coordinate uniformly. A deficient plan takes exactly the ordinary encryption path, without retries. All selected recipients get the same L-bit payload.

The ciphertext remains m header fields followed by one scalar. Each canonical field is encoded in fixed-width little endian; there is no audience, mode, key, dimension, or diagnostic tag. Decoders reject wrong lengths and noncanonical fields. SHAKE256 expansion retains the exact domain string, trailing zero byte, LE64(n,m,L) context and low-bit-first stream layout, including blocks that cross byte boundaries.

The library supports odd prime moduli that fit in u32. This deliberate implementation boundary includes the reviewed profile and the differential tests. Dimension arithmetic is checked; field operations are `crypto-bigint` operations, and products and sums use exact u64 or u128 intermediates whose bounds are checked. This is not a port of Python's unlimited-integer parameter range.

## Recorded run (code before 2026-10-26)

On this workstation, with Rust 1.96.1 in release mode, four measured trials after one warmup gave:

| Recipients | Hidden bits | Ordinary encryption | Anamorphic encryption | Decode including seed expansion |
|---:|---:|---:|---:|---:|
| 1 | 2 | 10.381 ms | 10.362 ms | 0.068 ms |
| 4 | 2 | 10.334 ms | 10.327 ms | 0.068 ms |

Raw samples and setup/preprocessing costs are in [recorded benchmarks](../../research/history/experiments/implementation-results/basic/benchmarks.json). A separate [capacity run](../../research/history/experiments/implementation-results/basic/capacity-benchmarks.json) checks one recipient with one bit and four recipients with eight bits (all 32 constraints). Across the four rows, all 32 measured and eight warmup ciphertexts passed ordinary decryption and codec checks; every selected hidden recipient recovered its common payload through both key representations.

The n1152 Python comparison in the [review package](../../lab-resource/2026-10-18_21-32-02-mythos-audit/local-fixes/code/basic/README.md) measured about 2.6–2.7 seconds per encryption under the same parameters. The retained historical runner in [`python-reference/basic/`](../../research/history/prototypes/python-reference/basic/README.md) uses n160 and must not be used for that timing comparison. This compares complete implementations: packed memory, arithmetic, validation and finite sampling differ. It is not an isolated estimate of language overhead. Neither the matrix dimensions nor ciphertext size decreased.

`tests/paper_conformance.rs` checks Construction 1 of the manuscript equation by
equation without the Python fixtures: `b_{i,a} = A k_{i,a}`, the column order of
`H_S`, `H_S^T s = Delta * am`, both ciphertext equations, the exact ordinary and
hidden residuals for every group member, the bound `s_max`, and the ordinary
fallback on rank deficiency.

`tests/optimized.rs` compares the optimized code with the frozen previous code in `tests/legacy/basic_v1.rs` on six parameter profiles (including q = 3 and moduli near 2^31 and 2^32) and on the production profile, checks the random-number consumption of every step, and checks that both modes and all thread counts agree. `tests/noise_table.rs` checks the base table against the MPFR digests and the sampler against its table with a chi-square test, a sign test and the variance s^2 + 1/12. Unit tests in `src/` compare every `crypto-bigint` field operation with wide integer arithmetic (boundary values and 20,000 random cases for each of six moduli from 3 to 4294967291), check the kernels in both accumulator widths and at the u64 bound, check that Gauss-Jordan returns the reduced form with its transformation (also for rank-deficient input), check the truncating MPFR helpers of the noise table, and check the comparison of all 256 bits when leading words tie.

`tests/sanity.rs` audits the values of real runs in both modes, on a small profile and (with `--release`) on the production profile. Every quantity must follow its law, and nothing may be zero, constant or repeated unless the protocol says so. `A`, `b`, the directory, `H_S`, `s` and the header must be uniform modulo q (mean, spread, zeros), key blocks must be balanced and distinct, and the errors must follow the rounded normal law of the table (spread, zero frequency, bound). The header and the scalar must carry exactly their errors, both residuals must equal the message plus the predicted noise, keys outside the audience (and the members, on ordinary ciphertexts) must see residuals spread over Z_q, every noise stream must differ, and the rank fallback must return the ordinary ciphertext. Zeros occur only where the protocol puts them: the target `Delta * am` of a zero bit, and an error with its small probability under the rounded normal law.

The independent checks cover 18 explicit-coin Python vectors, two SHAKE vectors, all small payload/audience combinations, every 2-by-3 matrix over F_3 for affine-fiber sampling, exact ordinary fallback, malformed inputs, rounded-noise boundaries, and arithmetic with a near-u32-maximum prime and signed-i64-extreme errors. See [recorded validation](../../research/history/experiments/implementation-results/basic/validation.json) for final commands, outputs and source provenance.

## Randomness and limits

The caller supplies the library RNG. For a fresh local demonstration, use `rand::rng()`; reproducible tests and benchmarks instead use public `StdRng::seed_from_u64` seeds. Public benchmark seeds are unsuitable for actual keys or messages. Pinning Cargo.lock records the exact finite sampler versions, but does not imply portable byte-identical random streams across environments.

Both implementations target independent rounded continuous normal errors. Rust samples them from the 256-bit base table described above, within the stated statistical distance and with errors bounded by B, and Python uses `random.Random.normalvariate`. Their finite distributions and random-bit consumption differ. The Rust encryption draws s and then a 32-byte noise seed from the caller's RNG, and expands the seed with ChaCha12, the generator behind rand's `StdRng`. Cross-language checks therefore use explicit encryption coins to verify equations, residuals and bytes, and separately check seed expansion and affine sampling. They do not claim the two numerical Gaussian samplers are identical. `round_noise` still rejects nonfinite or out-of-range floating-point values rather than saturating or resampling them, and `rounded_noise` now draws from the base table.

The implementation is variable time, including the table lookups of the noise sampler. SHAKE-based and ChaCha12-based PRG substitution, memory hygiene and side channels are not covered by a combined deployment guarantee. Hidden decryption returns L bits even for a nonrecipient; no authentication or reliable presence test has been added. A source-level port and passing tests do not close the existing finite-security obligations. The Rust-specific randomness/validation implementation has local review only; it has not received a new Pro audit. Promotion to maintained code does not change that review status or integrate the manuscript. Recorded results above are preserved from the port; fresh integration checks are indexed in [project progress](../../lab-resource/progress.md#rust-implementations-promoted-to-maintained-code).
