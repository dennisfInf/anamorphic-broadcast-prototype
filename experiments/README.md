# Experiments: ordinary and anamorphic costs across the number of equations d = |S| L

Each implementation has an `experiments` binary that times the host's ordinary
setup, encryption and decryption and the anamorphic setup, encryption and
decryption over repeated runs, across the number of equations d = |S| L that a
group's message imposes (|S| recipients of an L-bit message), and times the one step
in which aEnc differs from Enc on its own. `run_all.sh` runs all three, pinned to the
machine's performance cores where it can tell them apart, `summarize.py` turns their
JSON output into tables per scheme, and at the end the script runs the simulation of
the attack on DPPY25 ([attack/](../attack/README.md), about 80 s, not timed and not
pinned).

```sh
sh experiments/run_all.sh /path/outside/code/experiments-run
```

The output directory must not exist yet. It receives `basic.json`, one
`rbe-rankR.json` per RBE host rank, `epoque.json`, a log per scheme (progress lines,
`tail -f` it), `environment.txt`, `summary.md` / `summary.tex` / `summary.csv` /
`summary-sizes.csv`, the figures, and the attack simulation's report `attack.txt` with
`attack.json`, `attack.log` and its figure `attack.pdf` / `attack.png`. A part that
fails does not stop the others, and the summary covers every result file that was
written. Each binary rewrites its JSON file after
every finished part, so an interrupted run keeps what it measured
(`"complete": false`).

## What is measured

| | Basic (Dual-Regev) | RBE (authors' prototype, rank 7 + \|S\|) | Epoque |
|---|---|---|---|
| Ordinary setup | KGen: uniform A, binary k, b = A k | Setup, KeyGen and registration (Upd) of the receiver | KeyGen with the master shortcut key (`setup_trapdoor`) |
| Ordinary encryption | Dual-Regev encryption of one bit (the steps of `encrypt_with`) | `RBE::enc` (samples its own coins) | the steps of `encrypt_ordinary` for a random identity |
| Ordinary decryption | rounding of c - k^T r | `RBE::dec` with the receiver's witness | Dec with the master shortcut key |
| Also reported | noise-table construction | host setup, KeyGen, registration and witness generation apart | identity-key extraction, Dec with the identity key |
| aKGen (the \|S\| recipients) | seeds, SHAKE256 expansion, b = A k per key block | covert keys and public columns b = C k (A from the explicit public parameters) | binary keys, b = A0 k per key block |
| Group setup | `prepare_hidden` | `Family::context` | `prepare_hidden` |
| Anamorphic setup | aKGen + group setup, run by run | same | same |
| aEnc | the steps of `HiddenPlan::encrypt_with` | `Coins::sample_without_root`, `Context::apply`, host encryption | the steps of `HiddenPlan::encrypt` |
| Overhead: the differing step alone | s subject to H_S^T s = Delta am (`sample_secret_with`) minus s uniform | the constrained r_0 (`Context::sample`) minus the Gaussian r_0 of Enc (`Coins::sample_root`) | s subject to H_S^T s = floor(q/2) am (`AffineSampler::sample`) minus s uniform |
| aDec | one fixed recipient, from its seed (and from its expanded key) | one fixed recipient | one fixed recipient |
| Trades \|S\| x L | sweep 2 x 1, 2 x 2, ..., 2 x 128 (d = 2 to 256), checks 128 x 2 and 16 x 16, profile g256 | 1 to 6 recipients of 256 bits (d = \|S\| ring equations) on hosts of rank 7 + \|S\| (8 to 13), identity depth 162 | sweep 2 x 1, ..., 2 x 32 (d = 2 to 64), checks 32 x 2 and 8 x 8, profile full32 |

**The overhead of aEnc.** Enc and aEnc differ in one step: basic and Epoque draw the
secret s uniformly in Enc and subject to the group's equations in aEnc; RBE draws a
Gaussian root r_0 in Enc and computes r_0 subject to the equations in aEnc (aEnc's coins
come from `Coins::sample_without_root`, which skips the Gaussian r_0 that the broadcast
step would overwrite). Everything else is the same code. Each binary times that step
alone, `--step-reps` times per trade (2000 by default) in pairs, the two variants in
alternating order, and reports the paired differences (`step.overhead` in the JSON
files, with `step.replaced` and `step.broadcast`). For RBE the overhead can be
negative: computing r_0 from the equations can be cheaper than sampling the Gaussian
r_0, and the summary says so where it is. RBE's step is single-threaded in both
variants; with `THREADS` above 1, `RBE::enc` draws r_0 in parallel with the coins of
the other levels, so skipping it saves less wall-clock time than its sequential cost,
and the wall-clock overhead lies between the step overhead and the cost of
`Context::sample`. Run RBE with `THREADS=1` for the step overhead to be its
wall-clock overhead.

Enc and aEnc are timed in the same rounds as well. For every trade, after `--warmup`
untimed rounds (2 by default), each round encrypts fresh messages once with Enc and
once with aEnc, Enc first in even rounds and aEnc first in odd ones, so that drift of
the machine (clock and temperature after the long setups, other load) falls on both
alike. The summary gives Enc and aEnc of these rounds side by side, as mean ± standard
deviation, and the difference of the totals aEnc − Enc, round by round, as a check:
its interval should contain the step overhead. It is far wider, because one RBE or
Epoque encryption varies from run to run by milliseconds, far more than the step
costs, so the totals cannot resolve the overhead and can come out negative. The
summary and the figure give overheads as medians with distribution-free 95% intervals,
because a step of microseconds is now and then hit by an interrupt or a preemption,
which would dominate its mean. Each row of the JSON files also keeps the step as
timed inside aEnc (`broadcast_step`, and for basic and Epoque the uniform s inside Enc,
`replaced_step`).

Every loop of fast operations (aKGen, the group setup, the encryption rounds, the
decryptions) runs at least `--reps` times (`--keygen-reps` for aKGen) and then until
`--min-time` seconds (10 by default) have passed, at most 20000 times, so millisecond
operations get hundreds to thousands of runs while the slow ones keep `--reps`. The
decryptions are timed in a loop of their own over the last 8 rounds' ciphertexts
(RBE: 2), which were all checked before.

Every ciphertext is checked outside the timers: ordinary decryption, and for
anamorphic ciphertexts hidden decryption by every recipient in S. A failed check
aborts the run. Times are wall clock. Every scheme runs on `THREADS` worker threads:
basic its matrix work, group setup, broadcast step and hidden decryption on rayon's
pool (`BASIC_MODE=single` for one thread), RBE its host on rayon's pool (the authors'
`RBE::enc` samples and encrypts the levels of the update path in parallel; their
decryption and the extension's broadcast step are sequential), and Epoque its dense
products, the certification of its normals, the sums of R^T y, the OS bytes of R,
ordinary and hidden decryption and the group setup, all on rayon's pool as well.
Results do not depend on the thread count. `environment.txt` records it with the CPUs.

## The machine

A hybrid CPU such as the i5-1235U (two performance cores with two threads each,
eight efficiency cores) runs the same code at different speeds on its two kinds of
core, and a run left to the scheduler can move between them. `run_all.sh` therefore
pins every measured run with `taskset` (the builds before them stay unpinned):

- `CPUS=auto` (the default) pins to the performance cores that Linux reports in
  `/sys/devices/cpu_core/cpus` (on the i5-1235U usually CPUs 0-3) and does not pin on a
  CPU that is not hybrid. `CPUS=0-3` or `CPUS=0,2` pins to a list, `CPUS=all` not at all.
- `THREADS` defaults to the number of pinned CPUs (4 on the i5-1235U), at most 10. In
  basic and Epoque the differing step runs on its own between the other steps of Enc
  and aEnc, on the same pool, so its overhead is the wall-clock overhead at any thread
  count. In RBE it is exact with one thread (see above): `SCHEMES=rbe THREADS=1` gives
  it in a run of its own. Threads on separate cores avoid sharing a core's two
  hardware threads: `CPUS=0,2 THREADS=2` if `lscpu --extended` shows CPUs 0 and 1 on
  one core and 2 and 3 on the other.
- Fix the clock where possible, on the host (not in the container): the `performance`
  governor and turbo off (`echo 1 > /sys/devices/system/cpu/intel_pstate/no_turbo`),
  the laptop on mains power, nothing else running. The paired rounds cancel slow drift
  between Enc and aEnc, but not its effect on the absolute times. `environment.txt`
  records the pinned CPUs, the performance and efficiency cores, the governor, the
  energy-performance preference, turbo and the power supply as the container sees them.

Pinning needs Linux on the machine itself (native Linux, Docker Engine). Under WSL 2
(Docker Desktop on Windows) the container's CPUs are virtual, Windows decides which
physical core runs each of them, and `/sys/devices/cpu_core` does not exist, so
`CPUS` cannot select performance cores there; `run_all.sh` warns when it detects WSL.
Take the paper's numbers on native Linux, or at least report the setup.

## Groups

Every recipient and hidden bit adds one linear equation on the encryption randomness,
and each parameter set reserves a fixed number of them, so the largest group is
(reserved equations) / (equations per recipient):

- basic, profile `g256`: n = 1384 with n_min = 1128 reserves 256 equations, from 128
  recipients of 2 bits to 2 recipients of 128 bits (the maintained n = 1152 profile
  reserves 32).
- Epoque, profile `full32`: n = 1424 with g = 64, 32 recipients of 2 bits, and J = 192
  ordinary bits per ciphertext, the ballot length of Epoque's own benchmarks (the
  maintained n = 1360 preset has g = 8, 4 recipients, J = 128). Epoque's own parameters
  (n = 100 to 400, q = 4093, m = 12 n) cannot carry the channel: m - n log2 q is below
  one bit, so the leftover hash for the anamorphic keys fails, and even with error
  standard deviation 1 the core-SVP estimate stays below 80 bits.
- RBE: every recipient of a message of up to 256 bits needs one ring equation, and
  every ring equation removes one rank from the encryption randomness. The rank-16
  joint profile keeps root rank 12 for 4 recipients. By default (`RBE_PROFILE=scaled`)
  |S| recipients run on a host of rank 7 + |S|, so the root keeps the authors' rank 7
  and with it the security level of their benchmark host. `RBE_PROFILE=rank7` runs 1 to
  6 recipients on the authors' rank-7 host itself, leaving root rank 6 down to 1, below
  the paper's estimate (cost only). The host's rank is fixed at build time by
  `RBE_RANK`, so the script builds once per rank.

For basic and Epoque, the group setup and aEnc depend on d = |S| L only (H_S has d
columns) and aDec on L only, so the binaries vary d instead of listing splits of one d:
by default a sweep with |S| = 2 and L = 1, 2, 4, ... up to the budget, and two checks
that repeat the largest d with d/2 recipients of 2 bits and with |S| close to L. The
checks should match the sweep's last row; aKGen, which streams A once per batch of 8
users in basic, may differ a little. `--trades 2:1,2:2,...` and `--checks 128:2,...`
(RECIPIENTS:BITS) replace the defaults, and `--groups 128,64,...` runs the earlier
layout instead (for every |S| the longest L that fits and half of it). For RBE, a
message of up to 256 bits takes one ring equation per recipient whatever its length,
so d = |S| and the binary runs one 256-bit message per recipient (`--bits` adds other
lengths). Every result row records |S|, L, d (`constraints`) and whether it belongs to
the sweep or the checks (`role`). For each trade, aKGen generates the keys of exactly
the |S| recipients, and aDec is timed for one fixed recipient (the first of S). Every
ciphertext is still decrypted untimed by all recipients and by the ordinary receiver.
Keys of different users are independent, so aKGen for N users costs N times the
reported cost per recipient. The new profiles keep the maintained profiles' security
estimate for the dimension left after the equations (core-SVP,
`epoque/tools/security_estimate.py` for Epoque, the same model at the maintained
profile's level for basic), but basic's should be confirmed with the estimator behind
its maintained profile.

## Settings

`run_all.sh` reads these environment variables:

| Variable | Default | Meaning |
|---|---|---|
| `SCHEMES` | `basic rbe epoque attack` | parts to run, in order; `attack` is the simulation of Section 3 (`SCHEMES=attack` runs it alone) |
| `ATTACK_TRIALS` | 1000 | experiments per mode of the attack simulation |
| `ATTACK_SEED` | 2025 | its random seed (the default reproduces `attack/README.md`) |
| `CPUS` | `auto` | CPUs every measured run is pinned to: `auto` (the performance cores of a hybrid CPU, else none), `all`, or a list such as `0-3` |
| `THREADS` | pinned CPUs, at most 10 | worker threads of every scheme (`RAYON_NUM_THREADS`; at most the CPUs the runs may use) |
| `REPS` | 50 | runs of every measured algorithm, at least |
| `MIN_TIME` | 10 | seconds that every loop of fast operations runs at least, after its `REPS` runs (at most 20000 runs) |
| `STEP_REPS` | 2000 | paired iterations of the differing step alone, per trade |
| `WARMUP` | 2 | untimed encryption rounds (Enc and aEnc) per trade before the measured ones |
| `KEYGEN_REPS` | `REPS` | runs of aKGen for the group's recipients, at least |
| `SETUP_REPS` | `REPS` | runs of Epoque's ordinary setup |
| `BASIC_PROFILE` | `g256` | `n1152` is the maintained profile with at most 4 recipients of 8 bits |
| `BASIC_MODE` | `multi` | basic's matrix work on `THREADS` threads, or `single` |
| `RBE_PROFILE` | `scaled` | `scaled`: \|S\| = 1 to 6 on rank 7 + \|S\|, `rank7`: 1 to 6 on the authors' rank-7 host (cost only), `rank16`: 1 to 4 on the joint profile |
| `EPOQUE_PROFILE` | `full32` | `full` is the maintained n = 1360 preset (4 recipients of 2 bits), `small` a test-sized profile without security |

The binaries take the same options directly, for example
`cargo run --locked --release --manifest-path epoque/Cargo.toml --bin experiments -- --help`
(basic and RBE list theirs in the error for an unknown option): `--reps`, `--warmup`,
`--keygen-reps`, `--step-reps`, `--min-time`, for basic and Epoque `--trades`,
`--checks` and `--groups`, for RBE `--groups`, `--bits` and `--depth`. The RBE binary
must run in a scratch directory because the host keeps its registry in
`./registry_db`, which `run_all.sh` handles.

## Sizes

Every binary also writes `sizes_bytes` into its JSON file, and the summary lists them
per scheme (`summary.md`, and one row per item in `summary-sizes.csv`). Ciphertext
sizes are measured: the first ciphertext of each mode is serialized with the
implementation's own encoding. Key sizes follow the keys' encodings.

| Item | Basic (Dual-Regev) | RBE | Epoque |
|---|---|---|---|
| `ordinary_public_key` | (A, b), 4 bytes per field element | y = B x + e*, 48-bit packed | public parameters (A0, A1, B, U) |
| `ordinary_secret_key` | binary k, one bit per entry | x, 16 bits per coefficient | master shortcut key (D1, D2) at the width of its largest entry |
| `public_parameters`, `state` | | (A, B) and the 30 root digests, 48-bit packed | |
| `ordinary_ciphertext`, `anamorphic_ciphertext` | `serialize` | `serialize` (compressed blocks) | `encode` |
| `anamorphic_public_key_per_bit` | n field elements | | n field elements |
| `anamorphic_secret_key_per_bit` | (`_expanded`: m bits) | | m bits |
| `anamorphic_public_key_per_user` | L n field elements | packed b = C k, all payload blocks | L n field elements |
| `anamorphic_secret_key_per_user` | 32-byte seed (`_expanded`: L m bits) | k with 19-bit coefficients | L m bits |
| `anamorphic_group_matrix` | H_S, n by \|S\| L | H_S, n by g ring elements | H_S, n by \|S\| L |

The items per user and `anamorphic_group_matrix` are those of the largest trade (the
largest d, and among its splits the one with the most recipients); per hidden bit,
a user's keys grow linearly with L.

Both ciphertexts of a scheme have the same size by construction; the
binaries measure them separately.

## Figures

`plot.py` draws the paper figures from one result folder. It needs matplotlib, which
the bench image includes; `run_all.sh` calls it at the end where matplotlib is
installed. For a folder from the container:

```sh
python3 experiments/plot.py /path/to/run-YYYYMMDD-HHMMSS --output /path/to/figure
```

Both figures are 7.16 in wide, for a `figure*`, with one panel per host and the same
x axis: d = |S| L on a logarithmic axis for basic and Epoque, |S| (one ring equation
each) for RBE. `figure-costs.pdf` gives the mean times of the ordinary setup,
encryption and decryption (dashed) and of aSetup (aKGen plus group setup), aEnc and
aDec (markers: filled and joined for the sweep, hollow for the checks, the other splits
of the largest d) on one logarithmic time axis. `figure-overhead.pdf` gives the
overhead of aEnc in microseconds, the differing step timed alone, as the median of the
paired differences with a distribution-free 95% interval, on a linear axis per panel
with a zero line, and the median Enc of each host in the panel's title. The script
also prints every overhead next to the difference of the two totals of the same rounds
and flags a row whose 99.9% interval of that difference excludes the overhead. Result
folders from before the step was timed alone fall back to the overhead timed inside
aEnc, with a warning. Fonts are embedded as TrueType, and a 300 dpi PNG accompanies
each PDF. The script prints the worker threads of every result file and warns if the
schemes differ.

`summary.tex` has the same rows as LaTeX tables (booktabs): d, |S|, L, Enc and aEnc of
the same rounds (mean ± standard deviation), the step overhead with its 95% interval in
microseconds, and aDec. Check rows are starred.

## Cost of a full run

Without the 2^16-user key generation, a full run is dominated by the ordinary setups
(50 trapdoor setups of a few minutes each for Epoque) and by RBE's encryptions at depth
162 (about 1.7 s each). `MIN_TIME` adds up to about 40 s per trade (four loops of fast
operations of at least 10 s each) and the step loop well under a minute, about half an
hour over the 24 trades, and the attack simulation about 80 s. Expect a few hours in
total on a current laptop, more with fewer threads. A first pass with `REPS=3 MIN_TIME=0 STEP_REPS=100
EPOQUE_PROFILE=small` takes minutes and checks the setup.

## Status

The three binaries and the scripts were built and run here on small settings
against local stand-ins for the crates (crates.io was unreachable), including a
complete `run_all.sh` pass. Their timings are not representative. Run them with
the locked crates for the paper's numbers.

On 5 November 2026 the binaries gained the size reporting above and RBE moved back to
the vendored rank-16 host. Those changes were made without network access to
crates.io and have not been compiled yet: run `cargo build --locked --release --bins`
in each crate first (and `summarize.py` was tested on synthetic result files).

Later on 5 November 2026 the merged tree (this rank-16 RBE and the two-component Epoque) was
compiled and tested here against the same stand-ins, which gained the few functions the rank-16
host and the size reporting use (see the merge addendum in `BENCHMARKS.md`). All binaries
build, and one-run passes of the three binaries and `summarize.py` produced complete results
with sizes. A build with the locked crates is still to be done.

Later on 5 November 2026 the binaries switched from N = 2^4 to 2^16 users to the full and
half recipient group above, with keys only for the recipients and aDec by one fixed
recipient. They were built against the same stand-ins and run once with small settings.

On 7 November 2026 `run_all.sh` gained `THREADS` (default 10) for all three schemes;
RBE had run with `RAYON_NUM_THREADS=1` before. Epoque moved its parallel steps to
rayon's pool (the `rayon` lock entries are those of basic and RBE), so
`RAYON_NUM_THREADS` sets all three. Its encryption (products, normals, R^T y, the OS
bytes of R, the broadcast step), ordinary and hidden decryption and the group setup
became parallel, as did TrapGen's Cholesky factor and the perturbations' products L g
in KeyGen and Extr, with unchanged results and coin order (see the Threads section of
`epoque/README.md`); its JSON records the pool size. Earlier files list the logical
CPUs as `threads`, although Epoque's encryption and decryption then ran on one thread.
basic gained mode variants of the group setup, the broadcast step and hidden
decryption (`prepare_hidden_with`, `HiddenPlan::sample_secret_with`,
`hidden_decrypt_with`), which its experiments use in multi mode. RBE's code is
unchanged: the authors' encryption already uses rayon; their decryption and the
extension's broadcast step and aDec stay sequential.

Checks here, on two cores against the crate stand-ins and the real `rayon` 1.12.0
(built from its GitHub sources): every Epoque test passes except the four that need
MPFR beyond `f64` and integers beyond `i128`, which the stand-ins lack (the three
refinement tests of the sampler, one of them new, and the Arb normals fixture), and the
three opt-in tests of the n = 1360 profile pass. basic's dependencies could not be
built here; its changed functions were compiled from their exact text with stand-in
types and the real rayon, and agree between the modes at 256 constraints on 1384
coordinates and for L up to 128 at m = 36974. `tests/modes.rs` checks the variants in
the crate itself; run `test_all.sh`. With a uniform public key of the full32
dimensions, two threads took Epoque's encryption from about 1.6 s to 0.8 s,
decryption with a random wide key from about 30 ms to 17 ms, the group setup of 64
equations from about 32 ms to 20 ms and aDec from about 0.10 ms to 0.06 ms (stand-in
timings, so only the ratios are meaningful). The 10-thread timings are still to be
measured with the locked crates. `plot.py` was run on the result folder of 7 November.

On 12 November 2026 the three binaries moved to timing Enc and aEnc in the same rounds
(see What is measured), after the runs of 6, 7 and 11 November on the i5-1235U had put
aEnc below Enc for RBE (down to −3.8% at rank 8) and Epoque (down to −2.1%). Enc had
been timed in one block right after the setups and aEnc in later blocks, and the first
block was the slowest: at RBE rank 8 on 11 November the first ten ordinary encryptions
took 762 ms on average and the last ten 704 ms, against about 700 ms in every aEnc
block, and Epoque's ordinary block on 6 November fell from 1830 ms to 1764 ms, against
about 1750 ms for aEnc. `run_all.sh` passes `WARMUP`, `summarize.py` adds the overhead
table and `plot.py` draws the overhead from the paired rounds. Checked here without
crates.io: the previous and the new binaries were built, without warnings, against
stand-ins with the crates' signatures and run on a simulated machine with such a
warm-up drift. The previous binaries put RBE and Epoque at −1% to −3%; the new ones
gave every overhead positive and inside the interval of the paired totals, and a dry
run of `run_all.sh` passed `--warmup` to every scheme. Timings with the locked crates
are still to be measured.

Later on 12 November 2026, after review, the binaries moved to the layout above: the
step in which Enc and aEnc differ is timed alone in paired iterations (`step`) and is
the reported overhead, every loop of fast operations runs at least `--min-time`
seconds, the decryptions are timed in loops of their own, and basic and Epoque sweep
d = |S| L with |S| = 2 and two checks of the largest d instead of full and half
groups (RBE keeps 1 to 6 recipients of 256 bits; its 128-bit rows took the same ring
equations and were dropped). RBE's aEnc no longer draws the Gaussian r_0 that the
broadcast step overwrites (`Coins::sample_without_root`, with `Coins::sample_root` for
the rank-failure fallback and a test of both in `tests/integration.rs`); `main.rs`
(`rbe-bench-ab`) still draws it, as recorded in `BENCHMARKS.md`. `run_all.sh` pins the
measured runs to the performance cores (`CPUS`), builds before pinning, passes
`MIN_TIME` and `STEP_REPS`, and records the governor, turbo and power supply;
`summarize.py` and `plot.py` report the step overhead in microseconds by d, and
`summarize.py` also writes `summary.tex`. Checked here without crates.io: the binaries
built without warnings against the stand-ins and ran on the simulated machine through
a complete `run_all.sh` pass pinned to one CPU; the real RBE `lib.rs`, `main.rs`, the
experiments binary and `tests/integration.rs` type-check against stand-ins with the
crates' signatures (clippy clean); the summary and figures were also produced from
result files of the two earlier layouts. The new RBE test and all timings still need
the locked crates (`test_all.sh`, then `run_all.sh`).

Also on 12 November 2026, `run_all.sh` gained the part `attack`, last by default: the
simulation of Section 3 (`attack/dppy25_attack.py`, `ATTACK_TRIALS`, `ATTACK_SEED`),
which writes `attack.txt`, `attack.json` and `attack.pdf` into the result folder;
`summarize.py` skips JSON files that no experiments binary wrote. A dry run of
`SCHEMES="basic attack"` against the stand-ins produced all of them.

Later the same day the simulation's defaults moved from n = 32 to n = 16 and from
|S| = 3, L = 4 to L = 2 for Construction 1, so that they meet DPPY25's parameter
conditions (m >= 2 n log2 q failed at n = 32), which its report now prints;
`attack/test_dppy25_attack.py` tests the implementation (29 tests, run by `test_all.sh`
where python3 is installed, which the dev image now has).

