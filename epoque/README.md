# Epoque with the anamorphic broadcast, in Rust

This crate implements the identity-based encryption of Epoque (Boyen, Haines and
Mueller, [ePrint 2021/304](https://eprint.iacr.org/2021/304), Section 5) with
Epoque's own trapdoor key generation, and embeds the one-ciphertext anamorphic
broadcast of Construction 1 of "Anamorphic Broadcast Encryption, Revisited"
through A0, as the manuscript's Construction 3 describes. It contains what the
encryption and decryption benchmarks need: KeyGen with the master shortcut key,
Extr for any identity in Z_q^n, ordinary and anamorphic encryption, the three
decoders (master shortcut key, identity key, anamorphic key) and the ciphertext
encoding. It is not the authors' code and not a voting system: there are no
trustees, commitments, ballot proofs or complaints.

## Run

Use Rust/Cargo with a C compiler, make and the build prerequisites for
`gmp-mpfr-sys`. Cargo builds GMP and MPFR through `rug`, so the first build takes
several minutes. All protocol code is Rust. Modular arithmetic comes from
established crates (see [Sampling and arithmetic](#sampling-and-arithmetic)), and
the command lines use `clap`. No Python is needed to run the crate.

```sh
cargo test --locked
cargo clippy --locked --all-targets -- -D warnings
cargo fmt --check
cargo run --locked --release --bin run_epoque -- --profile small --output /tmp/epoque-smoke
cargo run --locked --release --bin run_epoque -- --profile full --output /tmp/epoque-full
cargo run --locked --release --bin bench_ab -- --profile full --reps 10 --output /tmp/epoque-ab.json
cargo run --locked --release --bin experiments -- --profile full --reps 50 --output /tmp/epoque-experiments.json
# The full-profile tests (about 2 GB, six to eight minutes each here):
cargo test --locked --release -- --ignored --nocapture
```

`experiments` times KeyGen, Enc and Dec and, for group sizes 32, 16, 8 and 4 (profile full32), each with the longest message that fits and half of it, aKGen, the
group setup, aEnc and aDec, as means over `--reps` runs, and records the key and
ciphertext sizes (see [../experiments/README.md](../experiments/README.md)). The programs take
`--identity random` (the default, a uniformly random identity
in Z_q^n) or `--identity unit` (the identity (1, 0, ..., 0), whose matrix H is the
identity matrix). Each `run_epoque` output directory must be new. The runner
checkpoints completed phases in `timings.json`, writes an ordinary and an
anamorphic ciphertext, reads them back and decrypts them with the master shortcut
key, with the identity key and (anamorphic) by each of the four recipients with
its own key. It saves no keys or dense matrices, and a failed or interrupted run
leaves `success` false. The small profile is a functional test without security.
The full profile needs about 2 GiB of memory.

## Construction correspondence

### Epoque, Section 5

| Epoque | Code |
|---|---|
| Parameters, eq. (17) | `Profile` and `TrapdoorProfile`, with the conditions under [Parameters](#parameters) |
| Identity encoding H with full-rank differences, eq. (19) | `identity.rs`: `IdentityEncoding`, `Identity::matrix`, `transpose_times`, `times` |
| prm = (A0, A1, B, u), eq. (20), msk = T_A0, eqs. (21)-(22), T_B, eq. (28) | `trapdoor::setup_trapdoor`: TrapGen for A0 and for B, A1 and U uniform |
| Master shortcut key, Section 5.2.1 and eq. (29): `[A0 \| A1 + H(id) B] d = u` for every id | d2_j <- SamplePre(B, T_B, 0), d1_j <- SamplePre(A0, T_A0, u_j - A1 d2_j), the `ShortcutKey` |
| Extr, eq. (23), F_id = `[A0 \| A1 + H(id) B]` | `MasterTrapdoor::extract(pk, id, rng)`: e2_j from D_{Z^m,s}, e1_j <- SamplePre(A0, T_A0, u_j - (A1 + H(id) B) e2_j) |
| Enc, eqs. (24)-(26) | `encrypt_ordinary(pk, id, m, rng)` and `encrypt_with_coins`: CT = (c0, c1) with c0 = U^T s + x + floor(q/2) m and c1 = F_id^T s + [y; R^T y], the `Ciphertext` with `c0()` and `c1()` |
| Dec, eq. (27): c0 - d^T c1 | `decrypt_ordinary` (master shortcut key d = (d1; d2), every identity) and `trapdoor::decrypt_identity` (identity key) |

The paper prints the target of d1 as `-A1 d2`. Eq. (29) holds only with
`u - A1 d2`, which the code uses (the manuscript notes the same in Appendix A.5).
Because B d2 = 0, the master shortcut key decrypts every identity, and the
identity enters encryption only through the second half of c1.

The ciphertext is Epoque's two-part CT = (c0, c1): c0 has one entry per ordinary
bit (Epoque's c0, c0', ... for the vectors u, u', ...), and c1 has length 2m. Its
first half `A0^T s + y` is what aDec reads. Errors are drawn as in the project's
candidate: `y` and `x` are normals of standard deviation `q/(65 C)` rounded to the
nearest integer, and `R` is a fresh dense m by m sign matrix, so the second half of
c1 carries `R^T y`. The wire format is `c0 || c1` with fixed-width little-endian
canonical residues and no mode marker.

### Identity encoding

`identity.rs` implements the full-rank-difference encoding of Agrawal, Boneh and
Boyen (EUROCRYPT 2010, Section 5), which Epoque uses. A monic irreducible f of
degree n over Z_q is fixed. The identity id = (id_0, ..., id_(n-1)) is read as
g(X) = id_0 + id_1 X + ... + id_(n-1) X^(n-1), and row i of H(id) holds the
coefficients of X^i g(X) mod f(X). H(id) is the matrix of multiplication by g in
the field Z_q[X]/(f), so H is linear and H(id) - H(id') = H(id - id') is
invertible whenever id != id'. The unit identity gives H = I.

f depends only on (n, q), so every party derives the same encoding without
extra public data. For each prime power d that divides n exactly, the first
monic polynomial of degree d in a fixed pseudorandom sequence that passes
Ben-Or's irreducibility test is taken. These parts have pairwise coprime degrees
and are joined by composed sums, whose roots are the sums of the parts' roots
(computed from power sums with Newton's identities, valid because q > n, which
`Profile::new` checks). A composed sum of irreducible polynomials of coprime
degrees is irreducible, so f is irreducible of degree n. For n = 1360 = 16 * 5 * 17
the parts have degrees 16, 5 and 17.

Encryption needs only `H(id)^T s`, the coefficients of s(X) g(X) mod f(X), which
`Identity::transpose_times` computes from the rows with exact sums: n^2
multiply-adds, small next to the m n of `A^T s`. Extr needs `H(id) (B E2)`.

### Construction 1 inside Epoque (Construction 3)

| Construction 1 | Code |
|---|---|
| aKGen(pk, N, L): ask_i = (k̂_(i,1), ..., k̂_(i,L)) binary, apk = (b̂_(i,a) = A0 k̂_(i,a)) | `covert_keygen(pk, N, L, rng)` returns `(CovertPublicKey, Vec<CovertSecretKey>)`. Its two steps are `covert_secret_keygen` (one user's ask_i, stored as L m bits) and `CovertPublicKey::from_secret_keys` (apk). `with_capacity` and `append` run aKGen in batches |
| aEnc(pk, apk, S, m, am), steps 1, 2 and 4: H_S with column (j - 1)L + a equal to a1 - b̂_(i_j,a), where a1 is the first column of A0 | `prepare_hidden(pk, &apk, S)`, from public inputs only, once per group S = {i_1 < ... < i_t}: H_S and its elimination |
| aEnc, steps 3 and 5 to 7: if rank(H_S) < tL encrypt ordinarily, else s uniform with H_S^T s = floor(q/2) (am, ..., am), then Enc | `HiddenPlan::encrypt(id, m, am, rng)`: am is repeated once per recipient, and Epoque's encryption uses fresh errors |
| aDec(ask_i, C): Round(c1[1] - k̂_(i,a)^T c1[1..m]) for each a | `decrypt_covert(&ask_i, ct)` returns the user's L bits (the nearer of 0 and floor(q/2) in circular distance, ties to 0). `covert_residual` returns the residuals |

The anamorphic algorithms use neither trapdoor and leave KeyGen and Extr
unchanged. For a recipient, `c1[1] - k̂^T c1[1..m] = (a1 - A0 k̂)^T s + y[1] - k̂^T y
= floor(q/2) am_a + y[1] - k̂^T y`. For a user outside S the term `(a1 - A0 k̂)^T s`
is uniform. Both ordinary decoders cancel s exactly, so an anamorphic ciphertext
decrypts like an ordinary one built from the same errors (eq. (9)).

`src/lib.rs` contains the scheme, `src/identity.rs` the identity encoding,
`src/trapdoor.rs` KeyGen and Extr, `src/arithmetic.rs` the exact matrix products,
the affine sampler and the field type over `crypto-bigint`, and `src/sampling.rs`
the certified error sampler. Library callers supply their random generator. The
programs use buffered operating-system bytes and never replace dense matrices or
signs by a seeded expansion.

### Threads

Every parallel step runs on rayon's thread pool, as in basic and RBE:
`RAYON_NUM_THREADS` sets its size (default: every logical CPU), and
`arithmetic::threads()` reports it. `arithmetic::with_threads(n, f)` runs `f` on a
pool of n threads, as the tests do. Jobs below 2^16 elementary steps stay on the
calling thread. Integer sums are exact and floating-point ones keep their order of
operations, so no result and no coin depends on the thread count.

- Encryption (`encrypt_ordinary`, `HiddenPlan::encrypt`): `A0^T s`, `A1^T s` and
  `B^T (H(id)^T s)` split the columns of the matrix over the threads and check the
  residues of A while reading it; the m + J normals are certified in parallel
  (`ErrorSampler::sample_many`); every column block of `R^T y` splits its sign
  groups over the threads and adds their exact partial sums in order; `OsBits`
  fills a block of R (6741 by 2048 bytes at full32) from the OS on all threads,
  after the bytes it has buffered; and aEnc's broadcast step computes the pivot
  coordinates of s in parallel.
- Decryption with the master shortcut key or an identity key: the 2J inner
  products of length m (`wide_right_product` with one row) are split over the
  threads.
- aDec (`decrypt_covert`): the one-row binary product splits the 64-entry blocks
  of `c1[1..m]` over the threads and adds the exact partial sums of the L keys.
- Group setup (`prepare_hidden`): every pivot step of the elimination of H_S
  updates the other rows in parallel.
- KeyGen and Extr: the dense products, TrapGen's Gram matrix, its Cholesky factor
  (in panels of 64 columns, every entry computed as in the row-by-row algorithm)
  and the products L g of the perturbations (after their normals are drawn, in the
  same order as before) are parallel.
- On the calling thread: `H(id)^T s`, the uniform matrices and binary keys, and
  every sampler of KeyGen and Extr that draws coins (normals, D_Z, the gadget
  sampler).

Encryption draws its coins in the same order as before, on the calling thread: one
64-bit word per normal, the first words of all m + J normals in order, then R in
column blocks of 2048, one `fill_bytes` request per block. A normal whose first 64
bits do not decide its value (probability about 2^-55 per normal at full32's
width) is certified again on the calling thread after all first words, from its
first word followed by fresh words. Each normal still uses its own independent
uniform bits, so the distribution is exact; only the position of those further
words in the stream differs from sampling one by one, and without such a normal
the two streams are identical (`tests/threads.rs`).

## Trapdoors and preimage sampling

TrapGen is the gadget trapdoor of Micciancio and Peikert (EUROCRYPT 2012):
`A = [Abar | P | G - Abar R]` with `Abar` (n by 3n) and `P` uniform,
`G = I_n (x) (1, b, ..., b^(k-1))` with `b^k > q`, and `R` uniform in
`{-1,0,1}^(3n by nk)`, so `A [R; 0; I] = G`. A0 has no identity block: Epoque's
secret `s` is uniform and A0 must be indistinguishable from uniform. Writing
`Abar = [A1' | A2']` with `A1'` invertible (n by n), each column of
`A1'^-1 Abar R` is `r1 + M r2` with `M` uniform: LWE with a ternary secret of
dimension 2n, ternary error and only n samples. Whoever solves it recovers R,
the master key. Abar has width 3n because width 2n would leave this instance at
about 87 bits (core-SVP) for the full profile, below the ciphertext's 128, while
width 3n makes it underdetermined, at about 248 bits (see [Parameters](#parameters)).

SamplePre follows MP12's SampleD: a perturbation with parameter
`sqrt(s^2 I - sigma_G^2 T T^T)`, `T = [R; 0; I]` (a continuous Gaussian drawn
block by block with the Cholesky factor of the 3n by 3n conditional covariance,
then randomized rounding of width r), a Gaussian sample from the coset of
`Lambda^perp(G)` by Klein's algorithm on the basis `S_k`, and `x = p + T z`.
TrapGen redraws `R` if `s` is too small for its largest singular value. A
statistical trapdoor would need Abar wide enough for the leftover hash lemma with
ternary R (about n log2 q / log2 3, near 32,000 columns) and the factorization
of a covariance matrix of that dimension, so the trapdoor is computational.

These samplers use `f64`, as practical trapdoor implementations do. The
Box-Muller radius comes from a 128-bit uniform, so no Gaussian tail beyond
probability 2^-128 is cut off (53-bit uniforms would stop at 8.6 standard
deviations, about 2^-53 per pair of normals). The remaining deviations are
rounding errors of relative size about 2^-50 per sample. By standard
Rényi-divergence arguments they change the probability of any event, such as
recovering the trapdoor from the exposed keys, only negligibly. The exposed keys
are still not 2^-128-close to ideal Gaussians in statistical distance. Every batch
of preimages is checked exactly (`A x_j = t_j`). Setup checks `B D2 = 0` and
`A0 D1 + A1 D2 = U`, and Extr checks `F_id e_j = u_j`, on every entry. Key
columns whose noise bound exceeds C are redrawn with probability below 2^-128 per
column, so the redraw is statistically invisible. T_B is dropped after setup.
Trapdoor keys use the exact `wide_right_product`. The dense products, the Gram
matrix and its Cholesky factor, and the products L g of the perturbations (which
draw no coins) run on rayon's pool (see [Threads](#threads)),
with the same results bit for bit. The trapdoor sampling stays on the caller's
single random source, on the calling thread.

## Parameters

Both profiles satisfy these conditions, which `TrapdoorProfile::new` checks:

- `m >= ceil(n log2 q) + 2*128 + 2 ceil(log2 g)`, Construction 1's leftover-hash
  condition for its binary keys, so that `b̂ = A0 k̂` looks uniform. Both presets
  meet it with equality. Also `m >= 3n + nk` for the trapdoor blocks.
- `s^2 >= sigma_G^2 (s1^2 + 1) + r^2`, with `r = eta_eps(Z^m)` for eps = 2^-128,
  `sigma_G = r sqrt(b^2 + 1)` and `s1 = 1.1 sqrt(2/3) (sqrt(3n) + sqrt(nk))`.
- `C >= sqrt(1 + (|d1| + sqrt(f m) |d2|)^2)` for key columns of the expected norm
  `sqrt(m) s / sqrt(2 pi)` up to a 2^-128 tail, where `|R d2|^2 <= f m |d2|^2` for
  the encryption signs except with probability 2^-128. Realized keys are checked
  against C.
- `q/(65 C) >= 2 sqrt(n) / sqrt(2 pi)`: the rounded-normal errors of standard
  deviation `q/(65 C)` meet Regev's width condition.
- `3n(q - 1) < 2^53`, so the f64 trapdoor products are exact, and `q > n` for the
  identity encoding.

| Parameter | Small (tests only) | Full |
|---|---:|---:|
| n | 16 | 1360 |
| m | 671 | 51325 |
| q | 47835289 (about 2^25.5) | 200639082569 (about 2^37.6) |
| J / g (largest \|S\| L) | 8 / 8 | 128 / 8 |
| C | 230587 | 104904148 |
| Gadget base b, digits k | 4, 13 | 4, 19 |
| Abar width / gadget columns nk / padding | 48 / 208 / 415 | 4080 / 25840 / 21405 |
| Preimage width s (key standard deviation) | 438 (175) | 4700 (1875) |
| Rounding width r, gadget width sigma_G | 5.53, 22.78 | 5.65, 23.29 |
| Error standard deviation q/(65 C) | 3.19 | 29.42 |
| Ciphertext bytes (2m + J residues) | 5400 | 513890 |
| Public matrices A0, A1, B | 0.26 MB | 1.68 GB |

`tools/trapdoor_parameters.py` derives both presets: the smallest prime q that meets
all conditions. Base 4 keeps q within half a bit of base 2 with half of its
trapdoor columns. Every key column's decryption noise has standard deviation at
most q/65, so a bit fails only if the sign bound fails (2^-128) or a Gaussian
exceeds 16.25 standard deviations (about 2^-195).

The full profile's n is the smallest multiple of 8 at which the ciphertext's LWE
instance reaches 128 bits core-SVP in both attacks below, with every other value
derived as above. `tools/security_estimate.py` computes the estimates and repeats
the search (primal uSVP and dual, 2016 model, core-SVP 0.292 beta classical and
0.265 beta quantum). It gives 118 bits for Kyber-512 and 182 for Kyber-768,
matching the published figures. The earlier full profile, n = 1984, reached 206
bits (classical) with 770730-byte ciphertexts and 3.67 GB of public matrices.

| Instance | beta | Classical | Quantum |
|---|---:|---:|---:|
| Ciphertext LWE, dimension n - g = 1352, error sd 29.4 | 441 | 129 | 117 |
| Trapdoor: ternary LWE, secret dimension 2n, n samples | 850 | 248 | 225 |
| Trapdoor with Abar of width 2n (not used) | 297 | 87 | 79 |

The dual attack gives 128 and 247 bits for the first two rows. These are not
lattice-estimator results and must be confirmed, in particular against hybrid
attacks on ternary secrets. They are attack estimates, not the premise of the
manuscript's reduction (see [Security boundary](#security-boundary)).

Epoque's Table 2 dimensions (m = 12n, n <= 400) are not offered. Binary anamorphic
keys give only m = 12n bits of entropy there, about a third of what the
leftover-hash condition needs at a q where trapdoor keys decrypt, and Table 2's
q = 4093 cannot decrypt with trapdoor keys at all.

## Security boundary

Whoever holds T_A0 can invert the first half of c1, `A0^T s + y` (MP12 inversion,
`MasterTrapdoor::recover_secret`), recompute `H_S^T s` from the public anamorphic
keys and read the hidden message, as the manuscript's Appendix A.7 shows and
`tests/trapdoor.rs` checks. The extraction master key must never reach the party
from whom the anamorphic message is hidden. The programs drop it after
extracting the one identity key they use.

The manuscript's security experiment exposes the public parameters, the
anamorphic public keys, all J shortcut vectors and the anamorphic keys outside S,
and keeps both trapdoors and the identity keys hidden. Lemma 12 (Appendix A.5)
first replaces the 2J preimage calls by exact Gaussians (distances ε0 and εB),
which removes T_A0, and then replaces A0 by a uniform matrix at cost δ0. The
manuscript takes δ0 from ABB10's statistical TrapGen. With MP12's computational
trapdoor, δ0 is instead the advantage of a distinguisher between the accepted A0
and a uniform matrix: at most nk times the advantage against the trapdoor's
ternary LWE instance above, plus the probability that TrapGen redraws R. The proof
needs no other change, because T_A0 is gone before A0 is replaced, and its bound
then holds against efficient distinguishers. B needs no such step, because the
simulator runs the host's own generator for (B, T_B). The f64 rounding enters the
sampler distances ε0 and εB. The norm-based key redraw is public and independent
of the trapdoor, so the simulator applies it too. This is the specified key
generation with a standard argument, not a new security theorem.

Proposition 14 (Appendix A.9) reduces anamorphic security to LWE in dimension
n - g whose noise is the host's divided by M + 1, where M bounds the operator norm
of the exposed keys' noise multiplier T = D1 + R D2. The reduction must produce c0
from the LWE samples through T, so it can only start from smaller noise. Here
each column of T has norm close to C, about 2^27, so M is at least that, the
reduced noise has standard deviation far below 1, and this premise fails for both
presets. It holds for the
asymptotic family of Appendix A.11. A concrete 128-bit choice that meets it needs q
of at least about 2^70 (n around 2800), beyond the `u64` residues and the exact
`f64` trapdoor products of this crate. The presets are therefore chosen by the
attack estimates above.

## Sampling and arithmetic

Rounded-normal sampling certifies the inverse-CDF bin of each error using
outward-rounded MPFR `erfc` bounds. If a random word hits a bin boundary, it
extends that same random prefix until the whole interval lies inside one bin. A
floating-point inverse normal CDF (`statrs`) only locates an initial search
point and cannot accept an output. Under ideal independent bits and correct MPFR
bounds, these decisions realize the stated law. The output has unbounded integer
support, with no tail clipping or resampling.

Every operation modulo q goes through `crypto-bigint` (RustCrypto): `add_mod`,
`sub_mod`, `neg_mod`, `mul_mod`, remainders of 128-bit integers and the
constant-time inversion `invert_odd_mod`. Primality of q is GMP's test (trial
division, Baillie-PSW and Miller-Rabin rounds). Baillie-PSW has no pseudoprimes
below 2^64, and GMP reports primes below about 2^51, such as the full profile's
q, as certain. Uniform residues are `rand`'s unbiased `random_range`. Dense
products are exact integer sums, `u128` for `A^T s` and `H(id)^T s` and `i128` for
products with keys (runs of terms in `i64` while they provably cannot overflow),
with the overflow bound checked before the loop, reduced once per entry. Products with the binary anamorphic keys (`binary_right_product`)
take the keys as packed bits and add the selected entries exactly in `u64`. For a
batch of keys, each block of 64 entries of a row first gets the sums of all subsets
of its groups of 4 or 8 entries, so a key needs 16 or 8 table lookups per block. One
pass over A0 serves 512 users (about 30 times faster per core than the `i128`
product). Sign products use eight-error lookup tables of exact integer
sums and fresh blocks of independent signs. If the error magnitudes exceed the
fast-path bound, the same computation uses arbitrary integers. No error is
reduced, clipped or rejected to make a machine-integer path fit. The modular
operations of `crypto-bigint` are constant time, but the samplers and the
surrounding loops are variable time. This is not side-channel-hardened or
zeroizing code.

## Measured execution (indicative)

On 2 November 2026, one full run of the trapdoor setup (release) completed in a
2-core x86-64 Linux container with Rust 1.97.0. crates.io was unreachable there,
so the crate was built against local API-compatible stand-ins for its
dependencies, not the locked crates. The trapdoor code depends on them only for
uniform integers and field operations. The run predates the identity encoding and
Construction 1's per-user interface (it used identity 1) and the current
presets. The identity adds `H(id)^T s` to each encryption (n^2 = 3.9 * 10^6
multiply-adds) and `H(id) (B E2)` to Extr (n^2 J = 5.0 * 10^8 multiply-adds). The
3n trapdoor block makes TrapGen's Gram matrix 2.25 times and its Cholesky factor
3.4 times as costly (dimension 5952 instead of 3968), and the presets grew by 7% in
q and 200 columns in m. Re-measure with the lockfile.

| Phase | Seconds |
|---|---:|
| TrapGen A0 / TrapGen B (one draw of R each) | 212.4 / 216.1 |
| Uniform A1 and U | 9.2 |
| D2 (128 preimages under B) / D1 (128 preimages under A0) | 36.4 / 40.9 |
| Exact relation checks | 13.6 |
| Setup total | 529.0 |
| Extr, identity 1 (128 columns) | 68.5 |
| Ordinary / anamorphic encryption | 3.5 / 3.7 |
| Each decryption (shortcut key, identity key) | 0.02-0.03 |
| Complete run | 606.2 |

Peak resident memory was 4.0 GiB. All 128 ordinary bits were recovered with the
master shortcut key and with the identity key from both ciphertexts, and the four
recipients each recovered `[1,0]`. No key column was redrawn.

## Encryption and decryption benchmark

`src/bin/bench_ab.rs` times Epoque encryption alone and with the anamorphic
broadcast inside, for the workloads `|S|:L` = `1:1, 2:2, 4:2` by default:

```sh
cargo run --locked --release --bin bench_ab -- --profile full --reps 10 --output /tmp/epoque-ab.json
```

It runs the trapdoor setup once, extracts the key of one identity (random by
default, `--identity unit` for H = I) and drops T_A0. It then performs the steps
of `encrypt_ordinary` and `HiddenPlan::encrypt` through the public API, so that
the broadcast part (sampling `s` subject to `H_S^T s = floor(q/2) (am, ..., am)`)
is timed separately from the Epoque part (exact errors, signs and matrix
products). A control re-times the two library calls. It also times aKGen (the
binary keys ask_i and the public vectors `A0 k̂`) and aEnc's preparation
(`H_S` and its elimination), and reports their sizes. Decryption is timed with
the master shortcut key, with the identity key and for all recipients with their
own keys. Every ciphertext is checked. The results of the earlier code (identity
1, reverse setup) are in [../BENCHMARKS.md](../BENCHMARKS.md).

## Tests

`cargo test` runs 29 unit tests and 23 integration tests. Three more tests use the
full profile and are opt-in (`--release -- --ignored`).

- Unit tests (`src/`). Field operations against wide integer arithmetic, exact
  products (the packed binary product against the `i128` one, on several threads
  and with the `u128` path), the same results for 1, 2, 3 and 8 threads from every
  parallel product, from both sign-table paths and from one-row binary products
  (masked sums, both table widths, `u64` and `u128` sums), checked against direct
  sums, the split of work into ranges, the affine sampler and rank deficiency
  (the same elimination and solutions for 1 to 8 threads), sign layouts, the certified
  error sampler (refinement, unbounded tails, independence from the search hint,
  batched sampling equal to one-by-one sampling for 1 to 7 threads, and a batched
  normal that needs further words taking them after all first words),
  the trapdoor samplers (moments, gadget cosets, Cholesky factor, the panelled
  factor equal bit for bit to the row-by-row one for 1, 3 and 8 threads, the
  perturbation's cross-covariance, perturbations equal in values and coins to the
  earlier loop for 1, 2 and 5 threads, spherical preimages) and the identity
  encoding: Ben-Or on known polynomials, composed sums, and irreducible moduli of
  degree n for several (n, q).
- `tests/trapdoor.rs`. H(id) is rebuilt from the public modulus f with the test's
  own polynomial arithmetic and must equal `Identity::matrix`, with H(unit) = I and
  `transpose_times` and `times` matching the matrix. f passes Rabin's
  irreducibility test, written independently of the library's Ben-Or test (itself
  checked on known reducible and irreducible polynomials), for five (n, q)
  including composed sums. H is linear, and H(a) - H(b) has full rank for random
  pairs and pairs that differ in one coordinate. With arbitrary integers, `B D2 =
  0`, `A0 D1 + A1 D2 = U`, `F_id D = U` and `F_id E = U` hold on every entry for
  the unit and random identities. Keys have width s and stay below C. For six
  group shapes (1x1, 2x2, 4x2, 8x1, 2x4, 1x8 recipients by bits), with aKGen for
  N = 2|S| users and every hidden message, every ordinary and anamorphic
  ciphertext decrypts with the master shortcut key and with the key of its
  identity, and every recipient recovers the hidden message with its own key. A
  key of another identity agrees with the message only by chance. aKGen equals its
  two steps coin for coin, apk is A0 k̂ exactly, H_S has the columns of
  Construction 1 in order, and malformed keys, groups and cross-parameter inputs
  are refused. A rank-deficient group falls back to `encrypt_ordinary` with the
  same coins, and invalid requests draw no coins. Same-error ordinary and
  anamorphic ciphertexts have identical ordinary residuals. T_A0 recovers s and
  the hidden constraint.
- `tests/correctness.rs`. All 2^8 ordinary messages decrypt with both ordinary
  keys, alternating between the unit and a random identity. For six group shapes
  and every hidden message (three repetitions), every recipient recovers the
  hidden message with its own key, the ciphertext still decrypts ordinarily, and
  users outside S agree with the hidden bits only at chance rate. The largest
  decoding noise is reported against q/4.
- `tests/paper_conformance.rs`. Equation by equation with arbitrary integers, for a
  random identity: KeyGen, eq. (29), Extr, Enc with `(A1 + H(id) B)`, aKGen,
  H_S, the hidden constraint, both ordinary residuals and every recipient's
  residual.
- `tests/sanity.rs`. A value audit of real runs: A0, A1, B, U, H(id), H_S, s, both
  halves of c1 and c0 are uniform modulo q (mean, spread, sixteen buckets, zeros),
  D1 and D2 are width-s Gaussians, the errors follow the rounded normal law, and R
  is a fresh dense sign matrix. It recomputes c0, c1 and every residual from the coins,
  checks that `HiddenPlan::encrypt` and `encrypt_ordinary` are exactly the audited
  steps, checks the rank fallback, and checks that keys outside the audience see
  residuals spread over Z_q.
- `tests/threads.rs`. On a profile where every parallel step of encryption is
  split (n = 256, m = 2^14, no security), `sample_errors` gives the errors and
  `R^T y` of one-by-one sampling and leaves the generator at the same position,
  and ordinary and anamorphic encryption and hidden decryption give the same
  ciphertexts, bits and coin use for 1, 2 and 5 threads.
- `tests/reference.rs`. Independent Python/Arb fixtures with explicit coins (unit
  identity, two users with two bits each): ciphertexts, residuals and bytes for
  six cases including errors beyond `i64`, affine plans and late pivots, sign
  layouts, 13 rounded-normal draws, all two-bit payloads, the exact fallback and
  its coin use, malformed inputs and the two decoders' rounding rules.

Fixtures are checked in, and Python is only needed to regenerate them:

```sh
# From this directory, using the existing project environment with numpy/python-flint:
../../research/history/.venv/bin/python -B tests/regenerate_fixtures.py \
  --reference ../../lab-resource/2026-10-18_18-40-18-epoque-improvement/practical-run \
  --output /tmp/epoque-python-fixtures.json
cmp tests/fixtures/python.json /tmp/epoque-python-fixtures.json
```

The fixture's key-sampler section belonged to the removed reverse setup and was
dropped from the fixture and the generator on 4 November 2026. On 5 November 2026
its ciphertexts were rewritten from (v, w, c) to Epoque's (c0, c1) = (c, (v; w)),
and its bytes to the order c0 || c1. The generator now emits this layout. No
value changed. The Python/Arb and Rust/MPFR implementations use different
certified searches, and the tests compare their outputs for a shared
infinite-bit prefix, not identical production RNG consumption.

Checks on 5 November 2026, after the switch to Epoque's (c0, c1) and the n = 1360
profile: crates.io was unreachable, so the crate was built against local
API-compatible stand-ins for its dependencies. All 43 tests passed in debug and
release builds. The three opt-in full-profile tests passed in a 2-core container
in 446, 364 and 358 seconds with a peak resident memory of 2.0 GiB. Their largest
decoding noise was 0.20 of q/4 for the shortcut key, 0.18 for identity keys and
3 * 10^-7 for the recipients, and users outside S matched the hidden bits at rate
0.51. clippy with `-D warnings`, rustfmt and rustdoc were clean, and the three
programs ran on the small profile (`run_epoque` with a random and with the unit
identity). Re-run `cargo test --locked` with the real crates.

## Source and scope

The [Epoque paper](https://eprint.iacr.org/2021/304) reports an implementation of
its IBE component. On 19 October 2026, renewed checks of the ePrint record,
[institutional publication record](https://researchportalplus.anu.edu.au/en/publications/epoque-practical-end-to-end-verifiable-post-quantum-secure-e-voti/),
author publication pages and targeted repository searches did not locate an
author-authenticated public source release. This is a bounded search result,
not proof that no release exists. No author implementation was copied.

The encryption and error sampling follow the project's `practical-run` candidate
([source](../../lab-resource/2026-10-18_18-40-18-epoque-improvement/practical-run/README.md)),
especially `runtime.py`, `kernels.c` and the reference encryption and sampling
modules. Earlier versions of this crate also had a reverse setup (shortcut
vectors sampled first, no extraction) restricted to identity 1. Its recorded runs
remain in
[implementation-results](../../research/history/experiments/implementation-results/epoque/full/timings.json).
The old n160 Python package is retained in
[`python-reference/epoque/`](../../research/history/prototypes/python-reference/epoque/README.md).

The crate covers the fixed-group model with all J shortcut vectors exposed. It
does not implement trustee keys, commitments, ballot proofs, complaint handling,
a voting service, presence authentication or adaptive-group security. A new Pro
review of this code is pending, and earlier reviews cover earlier code only.
Full finite-security and side-channel certification remain open.
