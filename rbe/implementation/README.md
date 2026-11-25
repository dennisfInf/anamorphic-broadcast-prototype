# RBE with an anamorphic broadcast channel

Start with `extension/src/lib.rs`. It adds the channel to the authors' RBE
implementation in `upstream/`: revision `5f1e7416893de7950d231f11b1207dd5bc02924d` of
<https://github.com/jnsiemer/scalable_rbe_prototype> with the rank-16 joint profile of
the paper's Sec. 7.2 applied. `n16-joint-profile.patch` recreates `upstream/` from that
revision (`git apply` in a clone); the Gaussian sampler is in
`upstream/src/joint_dgs.rs`. `extension/src/main.rs` is the encryption benchmark,
`extension/src/bin/experiments.rs` the experiments for trade-offs between group size and message length, and `extension/tests/`
check the complete encryption path.

## Host facts the channel relies on

| Quantity | Value |
| --- | --- |
| Ring | Z_q[X]/(X^256+1), q = 281474976694273 |
| Module rank n | 16 |
| Branches K, subregistries B | 3, 30 |
| Gadget base, retained digits | 65536, 2 (so m_A = 32, m_C = 64) |
| Dropped bits | 13 in the root block, 8 in every other block |
| Samplers | `JointGaussian` (s = 16, message error s~ = 2^36) |
| Public parameters | explicit uniform matrices `A` and `B` in NTT form |

1. `PP` holds `A` and `B`; `Family::generate` reads the inactive root columns of `A`.
2. Encryption normalises every product once before the inverse NTT, so the root block
   is `c_0 = r_0^T A + r_1^T (i (x) G) + e_0` as in eq. (10), and the channel's target
   is `Delta*am` as in eq. (3) (checked in `tests/paper_conformance.rs`).
3. `decompress` returns canonical residues. With q = 1 mod 2^13 the rounding error is
   at most 4096 in the root block and 128 elsewhere.
4. `setup` already publishes all 30 terminators, so encryption works before the first
   registration. It requires a fresh `registry_db`.

## Rank budget

Each recipient and each 256-bit payload block consumes one constraint on the root
randomness, so `g = |S| ceil(L/256)` constraints leave the channel's LWE secret rank
`n - g`. `lib.rs` sets `MIN_ROOT_RANK = 12` and derives
`MAX_CONSTRAINTS = N - MIN_ROOT_RANK = 4`, so the root keeps rank 12 or more, as the
paper's Core-SVP estimate (219.0 classical, 198.75 quantum bits) assumes. One ciphertext
therefore carries a common payload to four recipients with up to 256 bits, two with up
to 512 bits, or one with up to 1024 bits.

A compile-time assertion rejects any host of rank 12 or less. The unmodified upstream
crate (rank 7), on which this extension ran between 19 October and 5 November 2026,
left root rank 3 at g = 4: about 27 classical bits in the same Core-SVP model. That
build is superseded; `BENCHMARKS.md` keeps its measurements as a record.

## Construction flow

Choose the covert group, payload length and first identity branch before setup.

1. `Host::setup` creates the public matrices, the initial state and the registry.
   `keygen` and `update` register ordinary keys.
2. `Family::generate` makes the covert keys from the inactive root columns.
   `family.context(&group)` prepares one linear system per CRT coordinate.
3. `Coins::sample` samples all ordinary encryption randomness, exactly as `RBE::enc`.
   `context.apply` replaces only the root vector, or leaves the coins unchanged if
   the setup's constraint matrix is rank deficient.
4. `coins.encrypt` runs the host's `enc_wo_randomness`. Both modes use the same codec.
5. `host.dec` recovers the ordinary message. `family.decode` recovers the covert
   payload for each intended recipient.

`Family::decode` returns bits without detecting or authenticating a hidden payload.
Sample fresh `Coins` for every published ciphertext.

## Build and check

From `implementation/extension`, with Rust (edition 2024) and the GMP/MPFR build
prerequisites (a C compiler, make, m4), because the sampler uses `rug`. Tests and
benchmarks clear a local `registry_db`, so run them sequentially in a disposable copy
of this package:

```sh
export RAYON_NUM_THREADS=1
cargo test --locked --release -- --test-threads=1
cargo test --locked --release --manifest-path ../upstream/Cargo.toml -- --test-threads=1
cargo clippy --locked --release --all-targets -- -D warnings
RAYON_NUM_THREADS=1 cargo run --locked --release -- --reps 50 --output /tmp/rbe-ab.json
RAYON_NUM_THREADS=1 cargo run --locked --release --bin experiments -- --reps 50 --output /tmp/rbe-experiments.json
# another host rank, set at build time: here rank 9 for two recipients
RBE_RANK=9 RAYON_NUM_THREADS=1 cargo run --locked --release --bin experiments -- --reps 50 --groups 2 --output /tmp/rbe9-experiments.json
```

`RBE_RANK` (read by `upstream/build.rs`, default 16) sets the host's module rank at
build time. Ranks 8 to 15 keep the joint profile's pruning and noise width and keep the
authors' rank 7 free ([`MIN_ROOT_RANK`] = 7), so rank 7 + |S| carries |S| recipients at
the security level of the authors' benchmark host. `experiments/run_all.sh` runs one to
six recipients on ranks 8 to 13 this way. `RBE_RANK=7` is the authors' host itself
(`PRUNE_CT_BY = 16`, their `SIGMA_TILDE` of 33,519,950,000 rounded to 2^35 because the
joint sampler needs a power of two) with up to six constraints, each taking one of its
seven ranks: use it to compare costs, not as a secure configuration. The test suites
need at least four constraints, so run them at the default rank 16 or at ranks 11 to 15.

`src/bin/experiments.rs` times the host's Setup with KeyGen and registration,
`RBE::enc` and `RBE::dec` and, for up to `MAX_CONSTRAINTS` recipients (1 to 4 at rank 16, one to six on hosts of rank 7 + |S| via `RBE_RANK`), each with a 256-bit and a 128-bit message, aKGen, the group
setup, aEnc and aDec, as means over `--reps` runs. It also records the ordinary and
anamorphic key sizes, the public parameters and state, the group matrix and the
serialized ciphertexts of both modes (see
[../../experiments/README.md](../../experiments/README.md)). `cargo run` without
`--bin` still runs the encryption benchmark of `src/main.rs`.

The benchmark defaults to identity depth 162 and the workloads `|S|:L` =
`1:256, 2:256, 4:256, 2:512, 1:1024`. Use `--workloads` and `--depth` to change
them. It times the authors' `RBE::enc`, the same encryption through `Coins`
(control), and RBE with the broadcast inside, split into coin sampling, the
broadcast (affine root sampling) and the host encryption. It also times anamorphic
key generation and the group setup and reports all key, setup and ciphertext sizes.
Every timed ciphertext is decrypted by the ordinary receiver and by every intended
hidden recipient. The certified sampler makes coin sampling the largest part of an
encryption at this profile.

`tests/paper_conformance.rs` checks the rank budget, `b = C k`, the carrier equation of
the real ciphertext up to the codec's rounding, the constraint in every CRT slot and
the exact residual `Delta*am + e_C[0] - e_C^T k`. `tests/sanity.rs` audits the values of
real anamorphic ciphertexts at depth 162 on all three branches and three group shapes.
`tests/integration.rs` covers both root codecs (8 and 13 dropped bits), the full
registry root relation, the wire format and the rank-failure fallback, and
`tests/audit.rs` the decompression and the top wire symbol; its residual diagnostic is
opt-in (`--ignored`).

## Scope

`tests/opus3_duplicate_key.rs` records the unrestricted BLE limitation: reusing
one key at two selected leaves exposes their per-slot message difference from
public data. The reference API accepts honestly generated registration inputs and
does not implement registry merging or side-channel hardening.

The analysis files under `../correctness` and `../security`, `../selected-profile.json`
and `../provenance.json` were written for this rank-16 profile.
