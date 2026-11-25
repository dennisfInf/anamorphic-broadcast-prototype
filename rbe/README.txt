RBE with a one-ciphertext anamorphic channel
============================================

This package adds the one-ciphertext anamorphic broadcast channel to the
batched laconic encryption path of the scalable-RBE prototype by Kloos, Lai,
Siemer and Swarnakar. The host is vendored in implementation/upstream: the
authors' revision 5f1e7416893de7950d231f11b1207dd5bc02924d of
https://github.com/jnsiemer/scalable_rbe_prototype (MPL-2.0) with the rank-16
joint profile of the paper's Sec. 7.2 applied. implementation/n16-joint-profile.patch
recreates that tree from the upstream revision; the extension's Cargo.toml points
at it with a path dependency.

Start with implementation/README.md and implementation/extension/src/lib.rs.
Family::generate computes the covert public keys from the inactive root columns.
Family::context prepares the fixed group's CRT linear systems. Context::apply
replaces only r[0] of fresh Coins and preserves the ordinary fallback on rank
failure. Both modes then call the host's enc_wo_randomness and the same byte
codec. Family::decode recovers the common hidden payload for each recipient.

Host profile (joint profile, Sec. 7.2)
--------------------------------------
Ring Z[X]/(X^256+1), q=281474976694273, n=16, B=30, K=3, gadget radix 65536,
m_A=32, m_B=16, inactive carrier width m_C=64. The root block drops 13 bits,
every other block 8 (q = 1 mod 2^13, so the rounding error is at most 4096 and
128). Explicit uniform public matrices. Gaussian coins from the certified
JointGaussian sampler (s=16, message error s~=2^36). Encryption normalises every
product once before the inverse NTT, so the channel's target is Delta*am exactly
as in eq. (3). Covert key width K=2^19 (coefficients in [-2^18, 2^18)).

Rank budget
-----------
Each recipient and each 256-bit payload block fixes one of the root's n ring
coordinates, so g = |S| ceil(L/256) constraints leave the channel's LWE secret
rank n - g. lib.rs sets MIN_ROOT_RANK = 12 and MAX_CONSTRAINTS = N - MIN_ROOT_RANK
= 4, so the root keeps rank 12 or more, which the paper's Core-SVP estimate
(219.0 classical, 198.75 quantum bits) assumes. The bound is derived from the
host's rank and lib.rs refuses to compile against a host of rank 12 or less.

Between 19 October and 5 November 2026 the extension ran on the unmodified
upstream crate (rank 7). There, four constraints left root rank 3, about 27 bits
in the same Core-SVP model, so that build is superseded. BENCHMARKS.md keeps its
measurements as a record.

Status of the analysis files
----------------------------
correctness/, security/, selected-profile.json and provenance.json were written
for this rank-16 profile and describe the build again.

Reproduction
------------
Use Rust/Cargo with the GMP/MPFR build prerequisites (a C compiler, make, m4).
Run tests serially in a disposable copy, because they clear the local registry_db:

  cd implementation/extension
  RAYON_NUM_THREADS=1 cargo test --locked --release -- --test-threads=1

run-benchmarks.sh NEW_DIRECTORY runs the encryption benchmark into a fresh
output directory. reproduce.sh NEW_DIRECTORY runs the tests and the benchmark.
The experiments over N = 2^4 to 2^16 anamorphic users (cargo run --bin
experiments) are described in ../experiments/README.md.
