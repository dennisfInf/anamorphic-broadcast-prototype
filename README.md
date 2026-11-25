# Anamorphic broadcast encryption: implementations and experiments

This repository contains Rust implementations of the one-ciphertext anamorphic
broadcast channel (Construction 1 of the accompanying paper) inside three host
schemes, the scripts that run the paper's experiments, and a Python simulation of
the distinguishing attack on the Dual-Regev construction of Do, Persiano, Phan and
Yung (DPPY25).

## How this project was made

This project was created through vibe coding with Claude Opus 5.5.

It is research code. It is not meant for production use (see
[Scope and limitations](#scope-and-limitations)).

## Contents

| Path | What it contains |
| --- | --- |
| [`basic/`](basic/README.md) | Standalone Dual-Regev encryption with Construction 1 (crate `basic-anamorphic`). The maintained profile has n = 1152 and allows up to 32 hidden equations. The experiments use profile `g256` (n = 1384, 256 equations). Every heavy operation has a single-thread and a multi-thread mode. Errors come from a 256-bit table noise sampler. |
| [`rbe/`](rbe/implementation/README.md) | Construction 1 inside lattice-based registration-based encryption (RBE). `rbe/implementation/extension/` (crate `rbe-anamorphic-review`) adds the channel to the third-party scalable-RBE prototype vendored in `rbe/implementation/upstream/`. The host uses the rank-16 joint profile by default; `RBE_RANK` selects another module rank at build time. `rbe/correctness/` and `rbe/security/` hold the parameter analyses. |
| [`epoque/`](epoque/README.md) | The identity-based encryption of Epoque (ePrint 2021/304, Section 5) with MP12 trapdoor key generation and Construction 1 embedded through A0 (crate `epoque-shortcut`). Profiles: `small` (tests only, no security), `full` (n = 1360) and `full32` (n = 1424, used by the experiments). |
| [`attack/`](attack/README.md) | Simulation of the attack of Section 3 (Proposition 3.1) on DPPY25, next to Construction 1, with its test suite. Python standard library only. |
| [`experiments/`](experiments/README.md) | `run_all.sh` runs every experiment of the paper. `summarize.py` turns the results into Markdown, CSV and LaTeX tables, and `plot.py` draws the figures. |
| `Dockerfile`, `.dockerignore` | A container build with three targets: `dev`, `bench` and the runtime image. |
| `test_all.sh` | Runs every test suite. |

## Requirements

### Option A: Docker (recommended)

- Docker Engine on Linux, or Docker Desktop on Windows or macOS (Linux containers).
  The Dockerfile uses BuildKit cache mounts, which current Docker versions enable by default.
- Internet access during `docker build`. The build pulls the base images, the crates
  from crates.io and, for the `bench` image, matplotlib from PyPI.
- Time: the first build takes several minutes, mostly to compile and test GMP and MPFR.
- Memory: Epoque's full-size profiles need several GB of RAM. Give Docker at least
  5 GB. Docker Desktop gets half of the computer's memory by default.

Nothing else is needed on the host: no Rust toolchain, GMP, MPFR or Python.

### Option B: native build

- **Rust** with Cargo, version 1.96 or later. The code was tested with 1.96.1 and
  1.97.0, uses edition 2024, and must be built with the checked-in lockfiles (`--locked`).
- **A C compiler, `make` and `m4`.** All three crates depend on `rug`, whose
  `gmp-mpfr-sys` dependency builds GMP and MPFR from source. The first build takes
  several minutes. Later builds reuse the cached libraries (`GMP_MPFR_SYS_CACHE`
  sets the cache location).
  - Debian/Ubuntu: `sudo apt install build-essential m4`
  - macOS: the Xcode command line tools (`xcode-select --install`)
  - Windows: use Docker Desktop or WSL 2. A native Windows build needs the
    MSYS2/MinGW setup described in the `gmp-mpfr-sys` documentation.
- **Python 3.8 or later.** The attack simulation, its tests and `summarize.py` use
  only the standard library.
- **Optional tools:**
  - `matplotlib` for the figures (`experiments/plot.py`, and `attack/dppy25_attack.py --plot`)
  - `numpy` and `scipy` for `basic/tools/noise_stats.py`
  - the GMP and MPFR development headers (`libgmp-dev libmpfr-dev`, or `brew install mpfr`) for the C tools in `basic/tools/`
  - `taskset` (util-linux), which `run_all.sh` uses for CPU pinning

All commands below are run from the repository root unless stated otherwise.

## Quick start with Docker

```sh
# 1. Build the development image and run every test suite (about 10-20 minutes)
docker build --target dev -t anamorphic-dev .
docker run --rm anamorphic-dev sh test_all.sh

# 2. Build the benchmark image and do a short trial run of all experiments
docker build --target bench -t anamorphic-bench .
mkdir -p results
docker run --rm -v "$PWD/results:/results" \
  -e REPS=3 -e MIN_TIME=0 -e STEP_REPS=100 -e EPOQUE_PROFILE=small \
  anamorphic-bench
```

The trial run writes a new folder `results/run-<UTC time>/` with one JSON file and
log per scheme, the summary tables, the figures and the attack report (see
[Outputs](#outputs)). It takes minutes rather than hours. Its numbers only check
the setup and are not meant for reporting. For the paper's settings, see
[Reproducing the paper's experiments](#reproducing-the-papers-experiments).

In Windows PowerShell, write `"${PWD}\results:/results"` instead of
`"$PWD/results:/results"`, and `mkdir results` instead of `mkdir -p results`.

## Quick start without Docker

```sh
# Every test suite (about 10-20 minutes after the first build)
sh test_all.sh

# Short runs of each part
cargo run --locked --release --manifest-path basic/Cargo.toml --bin benchmark -- \
  --repetitions 4 --warmups 1 --output /tmp/basic-benchmark.json
cargo run --locked --release --manifest-path epoque/Cargo.toml --bin run_epoque -- \
  --profile small --output /tmp/epoque-small
python3 attack/dppy25_attack.py
```

`test_all.sh` runs the release-mode tests of `basic/` and `epoque/`, the attack
simulation's tests (skipped with a note if `python3` is missing), and the tests of the
RBE extension and of the vendored RBE host. The RBE tests run one at a time on a single
thread, in a temporary copy of `rbe/implementation`, because they delete
`./registry_db` (see [rbe/](#rbe-rbe-with-construction-1)). A failing suite does not
stop the others. The script ends with `All test suites passed.`, or with
`FAILED: <suites>` and a nonzero exit status. `FULL=1 sh test_all.sh` adds Epoque's
three opt-in tests of the full n = 1360 profile. They need about 2 GB of memory
(budget 5 GB) and take several minutes each.

Tip: `export CARGO_TARGET_DIR="$PWD/target"` makes all crates share one build
directory, as the Docker image does. Without it, `test_all.sh` compiles the RBE
crates and all their dependencies from scratch in its temporary copy on every run.

## Running the implementations

General rules:

- Benchmark and experiment programs refuse debug builds. Always pass `--release`.
- Write every run to a new output path. `basic`'s `benchmark` and Epoque's
  `run_epoque` refuse existing outputs, and the other programs overwrite them.
- `RAYON_NUM_THREADS` sets the number of worker threads of all three implementations.
  The default is every logical CPU.
- Linting, as used during development (needs the `clippy` and `rustfmt` components):
  `cargo clippy --locked --all-targets --manifest-path <crate>/Cargo.toml -- -D warnings`
  and `cargo fmt --check --manifest-path <crate>/Cargo.toml`.

### basic/: Dual-Regev with Construction 1

```sh
# Tests (a few production-size tests run only in release mode)
cargo test --locked --release --manifest-path basic/Cargo.toml

# Benchmark: ordinary vs. anamorphic encryption in both execution modes
cargo run --locked --release --manifest-path basic/Cargo.toml --bin benchmark -- \
  --repetitions 4 --warmups 1 --modes single,multi --output /tmp/basic-benchmark.json

# Experiments for the paper (normally started by experiments/run_all.sh)
cargo run --locked --release --manifest-path basic/Cargo.toml --bin experiments -- \
  --reps 50 --output /tmp/basic-experiments.json
```

`benchmark` options (`--help` lists them):

| Option | Default | Meaning |
| --- | --- | --- |
| `--repetitions N` | 21 | measured repetitions per row (at least 4) |
| `--warmups N` | 3 | unmeasured repetitions before them |
| `--one-time N` | 3 | repetitions of key generation, anamorphic key generation and the noise table |
| `--population N` | 8 | registered hidden users (1 to 64) |
| `--rows S:L,...` | `1:2,4:2` | recipients and hidden bits per row. Needs S <= population and S x L <= 32 |
| `--modes ...` | `single,multi` | execution modes. Results must be identical in both |
| `--output FILE` | `results/benchmarks.json` | result file, which must not exist yet |

Every timed ciphertext is decrypted afterwards by the ordinary receiver and by every
intended recipient. If the two modes ever produce different keys or ciphertexts, the
benchmark fails.

`experiments` options: `--profile g256|n1152` (default `g256`), `--mode single|multi`
(default `multi`), `--reps` (50), `--warmup` (2), `--keygen-reps`, `--step-reps`
(2000), `--min-time SECONDS` (10), `--trades S:L,...`, `--checks S:L,...`,
`--groups S,...` and `--output`. The program prints this list when it meets an
unknown option.
[experiments/README.md](experiments/README.md) defines what is measured.

Optional checks of the noise sampler, from inside `basic/`
([basic/README.md](basic/README.md) explains each step):

```sh
cd basic
cargo run --locked --release --bin noise_table -- --dump /tmp/basic-noise-table.txt
# Independent MPFR evaluation of every table entry (needs the GMP/MPFR headers;
# with Homebrew, add -I$(brew --prefix)/include -L$(brew --prefix)/lib)
cc -O2 tools/noise_table_mpfr.c -lmpfr -lgmp -o /tmp/noise_table_mpfr
/tmp/noise_table_mpfr var:34972416 111285 | cmp - /tmp/basic-noise-table.txt
# Exact statistical distance of the table to the ideal rounded normal law
cc -O2 tools/noise_distance_mpfr.c -lmpfr -lgmp -o /tmp/noise_distance_mpfr
/tmp/noise_distance_mpfr var:34972416 /tmp/basic-noise-table.txt 30358
# Statistical tests of the noise that encryption adds (needs numpy and scipy).
# The recorded run used --ciphertexts 330000, about 10^10 errors.
cargo run --locked --release --example noise_stats -- --ciphertexts 1000 --output /tmp/noise-stats.json
python3 tools/noise_stats.py /tmp/noise-stats.json
cd ..
```

### epoque/: Epoque IBE with Construction 1

```sh
# Tests
cargo test --locked --release --manifest-path epoque/Cargo.toml
# Opt-in tests of the full n = 1360 profile (about 2 GB of memory, several minutes each)
cargo test --locked --release --manifest-path epoque/Cargo.toml -- --ignored --nocapture

# End-to-end run: trapdoor setup, key extraction, an ordinary and an anamorphic
# ciphertext, and every decryption (four recipients of two hidden bits)
cargo run --locked --release --manifest-path epoque/Cargo.toml --bin run_epoque -- \
  --profile small --output /tmp/epoque-small     # test-sized, no security, quick
cargo run --locked --release --manifest-path epoque/Cargo.toml --bin run_epoque -- \
  --profile full --output /tmp/epoque-full       # n = 1360, 1.7 GB of public matrices

# Encryption benchmark with and without the broadcast
cargo run --locked --release --manifest-path epoque/Cargo.toml --bin bench_ab -- \
  --profile full --reps 10 --output /tmp/epoque-ab.json

# Experiments for the paper (normally started by experiments/run_all.sh)
cargo run --locked --release --manifest-path epoque/Cargo.toml --bin experiments -- \
  --profile full32 --reps 50 --output /tmp/epoque-experiments.json

# Parameter derivation and attack estimates (standard library Python, seconds)
python3 epoque/tools/trapdoor_parameters.py
python3 epoque/tools/security_estimate.py
```

- `run_epoque` takes `--profile small|full` (default `small`),
  `--identity random|unit` (default `random`) and `--output DIR`. The directory must
  be new. The program writes `timings.json` (with `"success": true` at the end of a
  complete run), `ordinary.bin` and `anamorphic.bin`. It saves no keys.
- `bench_ab` takes `--profile`, `--identity`, `--reps` (10), `--control` (2),
  `--workloads S:L,...` (default `1:1,2:2,4:2`) and `--output`.
- `experiments` takes `--profile small|full|full32` (default `full32`), `--identity`,
  `--reps`, `--warmup`, `--setup-reps`, `--keygen-reps`, `--step-reps`, `--min-time`,
  `--trades`, `--checks`, `--groups` and `--output`. Every program accepts `--help`.

### rbe/: RBE with Construction 1

The RBE host keeps its registry in `./registry_db` in the current working directory.
It creates that directory and clears it. Therefore:

- run the programs from an empty scratch directory with `--manifest-path`;
- run the tests serially (`--test-threads=1`) in a disposable copy of
  `rbe/implementation`, because Cargo runs tests inside the package directory.

Tests (this is what `test_all.sh` does):

```sh
repo=$PWD
scratch=$(mktemp -d)
cp -R rbe/implementation "$scratch/"
cd "$scratch/implementation/extension"
RAYON_NUM_THREADS=1 cargo test --locked --release -- --test-threads=1
RAYON_NUM_THREADS=1 cargo test --locked --release --manifest-path ../upstream/Cargo.toml -- --test-threads=1
cd "$repo" && rm -rf "$scratch"
```

Programs, run from a scratch working directory:

```sh
repo=$PWD
manifest="$repo/rbe/implementation/extension/Cargo.toml"
mkdir -p /tmp/rbe-work && cd /tmp/rbe-work

# Encryption benchmark: RBE alone vs. RBE with the broadcast inside
RAYON_NUM_THREADS=1 cargo run --locked --release --manifest-path "$manifest" -- \
  --reps 50 --output /tmp/rbe-ab.json

# Experiments on the default rank-16 host (1 to 4 recipients)
RAYON_NUM_THREADS=1 cargo run --locked --release --manifest-path "$manifest" \
  --bin experiments -- --reps 50 --output /tmp/rbe-experiments.json

# The same on a host of module rank 9 (set at build time), two recipients
RBE_RANK=9 RAYON_NUM_THREADS=1 cargo run --locked --release --manifest-path "$manifest" \
  --bin experiments -- --reps 50 --groups 2 --output /tmp/rbe9-experiments.json
cd "$repo"
```

- **Host rank.** `RBE_RANK` (7 to 16, default 16) sets the module rank of the vendored
  host at build time. Every recipient of a payload of up to 256 bits takes one rank of
  the root, so the number of allowed constraints depends on the rank:
  - rank 16 keeps root rank 12, as the paper's profile requires, and allows at most 4 constraints;
  - ranks 8 to 15 keep the original authors' rank 7 free and allow `RBE_RANK - 7` constraints;
  - rank 7 is the original authors' benchmark host and allows up to 6 constraints. It is for cost comparisons only, not a secure configuration.

  The test suites need at least four constraints, so run them at rank 16 (the default) or at ranks 11 to 15.
- **Benchmark options** (default binary `rbe-anamorphic-review`): `--reps` (50, at
  least 2), `--depth` (identity depth 1 to 162, default 162), `--workloads S:L,...`
  (default `1:256,2:256,4:256,2:512,1:1024`) and `--output`.
- **Experiments options:** `--reps`, `--warmup`, `--keygen-reps`, `--step-reps`,
  `--min-time`, `--groups` (default: every allowed group size), `--bits` (default 256),
  `--depth` and `--output`.
- **Helper scripts.** `sh rbe/run-benchmarks.sh NEW_DIR` runs the benchmark, and
  `sh rbe/reproduce.sh NEW_DIR` runs the tests and the benchmark. Both build and run
  inside `rbe/implementation/extension` and create `registry_db` there, so start them
  from a disposable copy of the repository.
- **Analysis scripts** (standard library Python). By default they overwrite the
  checked-in JSON files next to them, so point `RBE_CHECK_OUTPUT` at a scratch
  directory. `aggregate-bounds.py` reads the output of `check_parameters.py`, so keep
  this order:

  ```sh
  export RBE_CHECK_OUTPUT=/tmp/rbe-checks && mkdir -p "$RBE_CHECK_OUTPUT"
  python3 rbe/correctness/calculate.py
  python3 rbe/security/check_parameters.py
  python3 rbe/security/aggregate-bounds.py
  python3 rbe/security/opus3-leaf-checks.py
  python3 rbe/security/opus3-probe-bound.py
  # The results should be byte-identical to the checked-in files:
  for f in bounds codec-sweep; do cmp "$RBE_CHECK_OUTPUT/$f.json" rbe/correctness/$f.json; done
  for f in analytical-profiles aggregate-bounds opus3-leaf-checks opus3-probe-bound; do
    cmp "$RBE_CHECK_OUTPUT/$f.json" rbe/security/$f.json
  done
  ```
- **Recreating the vendored host.** `rbe/implementation/upstream/` equals revision
  `5f1e7416893de7950d231f11b1207dd5bc02924d` of
  <https://github.com/jnsiemer/scalable_rbe_prototype> with
  `rbe/implementation/n16-joint-profile.patch` applied (`git apply` in a clone at that
  revision).

### attack/: the attack on DPPY25

```sh
python3 attack/dppy25_attack.py                  # 1000 experiments per mode, seed 2025, about 35-80 s
python3 attack/dppy25_attack.py --plot /tmp/attack --json /tmp/attack.json   # also writes the figure (matplotlib)
python3 attack/test_dppy25_attack.py             # 29 tests, about 10 s
```

The default run reproduces the result recorded in [attack/README.md](attack/README.md).
Against DPPY25 the residual test has advantage 0.209 [0.185, 0.235] at E = floor(q/5)
and 1.000 at the tight threshold. Against Construction 1 both modes look alike.
Options: `--trials`, `--seed`, `--n`, `--m`, `--q`, `--sigma`, `--r`, `--groups`,
`--recipients`, `--bits`, `--plot PREFIX`, `--json PATH` and `--progress`.

## Reproducing the paper's experiments

`experiments/run_all.sh NEW_OUTPUT_DIR` runs, in this order:

1. basic, profile `g256`: d = |S| L = 2 to 256 hidden equations;
2. RBE: 1 to 6 recipients on hosts of rank 7 + |S|, rebuilding the host once per rank;
3. Epoque, profile `full32`: d = 2 to 64;
4. the attack simulation.

It then writes the summary tables and, where matplotlib is installed, the figures. A
part that fails does not stop the others. The exit status is 0 only if every part
passed.

**With Docker** (recommended):

```sh
docker build --target bench -t anamorphic-bench .
mkdir -p results
docker run -d --name bench -v "$PWD/results:/results" anamorphic-bench
docker logs -f bench    # follow the progress; Ctrl-C stops only the viewer
docker wait bench       # waits for the end; prints 0 if every part passed
docker rm bench
```

**Natively:**

```sh
sh experiments/run_all.sh /path/to/new-run-folder   # must not exist yet; keep it outside the repository
```

With the default settings the full run takes several hours. The Epoque setups (50
trapdoor setups of a few minutes each) and RBE's encryptions dominate. A trial pass
with `REPS=3 MIN_TIME=0 STEP_REPS=100 EPOQUE_PROFILE=small` checks the whole setup much
faster.

### Settings

`run_all.sh` reads these environment variables. With Docker, pass them as `-e NAME=value`.

| Variable | Default | Meaning |
| --- | --- | --- |
| `SCHEMES` | `basic rbe epoque attack` | parts to run, in order (`SCHEMES=attack` runs only the simulation) |
| `CPUS` | `auto` | CPUs that every measured run is pinned to: `auto` (the performance cores of a hybrid Intel CPU, else no pinning), `all` (no pinning), or a list such as `0-3` |
| `THREADS` | the CPUs the runs may use (the pinned ones, else all), at most 10 | worker threads of every scheme (`RAYON_NUM_THREADS`) |
| `REPS` | 50 | minimum runs of every measured algorithm |
| `MIN_TIME` | 10 | seconds that every loop of fast operations runs at least, after its `REPS` runs (at most 20000 runs) |
| `STEP_REPS` | 2000 | paired iterations of the step in which aEnc differs from Enc, timed alone, per trade |
| `WARMUP` | 2 | untimed encryption rounds per trade |
| `KEYGEN_REPS` | `REPS` | runs of aKGen |
| `SETUP_REPS` | `REPS` | runs of Epoque's ordinary setup |
| `BASIC_PROFILE` | `g256` | `n1152` is the maintained profile (at most 32 equations) |
| `BASIC_MODE` | `multi` | basic's matrix work on `THREADS` threads, or `single` |
| `RBE_PROFILE` | `scaled` | `scaled`: 1 to 6 recipients on rank 7 + \|S\|; `rank7`: 1 to 6 on the original authors' rank-7 host (cost only); `rank16`: 1 to 4 on the joint profile |
| `EPOQUE_PROFILE` | `full32` | `full` (n = 1360, 4 recipients of 2 bits) or `small` (test-sized, no security) |
| `ATTACK_TRIALS` | 1000 | experiments per mode of the attack simulation |
| `ATTACK_SEED` | 2025 | its random seed (the default reproduces `attack/README.md`) |

For RBE's step overhead to equal its wall-clock overhead, run it on one thread:
`SCHEMES=rbe THREADS=1`.

### Outputs

| File | Content |
| --- | --- |
| `environment.txt` | date, kernel, Rust version, CPUs and pinning, governor, turbo, power supply, memory, all settings |
| `basic.json`, `rbe-rank<R>.json`, `epoque.json` | raw results per scheme (and per RBE rank). Each is rewritten after every finished part; `"complete": false` marks an interrupted run |
| `basic.log`, `rbe-rank<R>.log`, `epoque.log` | build output and progress lines (`tail -f` them) |
| `summary.md`, `summary.csv`, `summary-sizes.csv`, `summary.tex` | tables of all results: times, the overhead of aEnc, key and ciphertext sizes |
| `figure-costs.pdf`, `figure-overhead.pdf` (with `.png`) | the paper figures |
| `attack.txt`, `attack.json`, `attack.log`, `attack.pdf`, `attack.png` | the attack simulation's report, counts and figure |

To redo the tables or figures of a finished run:

```sh
python3 experiments/summarize.py RUN/*.json --output RUN/summary   # skips JSON files that are not experiment results
python3 experiments/plot.py RUN --output RUN/figure                # needs matplotlib
```

### Measurement hygiene

[experiments/README.md](experiments/README.md) explains the measurement method in detail.

- On a hybrid CPU (for example an i5-1235U), `CPUS=auto` pins every measured run to
  the performance cores that Linux reports in `/sys/devices/cpu_core/cpus`.
- Fix the clock on the host where possible: the `performance` governor, turbo off,
  mains power, and nothing else running.
- Pinning needs Linux on the machine itself: native Linux, or Docker Engine on Linux.
  Under WSL 2 (Docker Desktop on Windows) the CPUs are virtual and cannot be pinned to
  physical cores. `run_all.sh` warns there. Take the paper's numbers on native Linux,
  or report the setup.

## Docker reference

| Target | Build command | Contains | Default command |
| --- | --- | --- | --- |
| `dev` | `docker build --target dev -t anamorphic-dev .` | toolchain, sources in `/src`, prebuilt test binaries, python3 | `bash` |
| `bench` | `docker build --target bench -t anamorphic-bench .` | toolchain, sources, python3 with matplotlib | `sh experiments/run_all.sh /results/run-<UTC time>` |
| runtime (last stage) | `docker build -t anamorphic .` | the release programs only | `basic-benchmark --help` |

```sh
# dev image
docker run --rm anamorphic-dev sh test_all.sh
docker run --rm -e FULL=1 anamorphic-dev sh test_all.sh
docker run --rm anamorphic-dev cargo test --locked --release --manifest-path basic/Cargo.toml
docker run --rm -it anamorphic-dev          # interactive shell in /src

# runtime image: single programs, results written to ./results
mkdir -p results
docker run --rm -v "$PWD/results:/results" anamorphic basic-benchmark --output /results/basic.json
docker run --rm -v "$PWD/results:/results" anamorphic epoque-run --profile small --output /results/epoque-small
docker run --rm -v "$PWD/results:/results" -e RAYON_NUM_THREADS=1 anamorphic rbe-bench-ab --reps 50 --output /results/rbe-ab.json
```

Programs in the runtime image:

| Program | Source |
| --- | --- |
| `basic-benchmark` | `basic/src/bin/benchmark.rs` |
| `basic-noise-table` | `basic/src/bin/noise_table.rs` |
| `basic-noise-stats` | `basic/examples/noise_stats.rs` |
| `epoque-run` | `epoque/src/bin/run_epoque.rs` |
| `epoque-bench-ab` | `epoque/src/bin/bench_ab.rs` |
| `rbe-bench-ab` | `rbe/implementation/extension/src/main.rs` |

`--build-arg RUST_VERSION=1.96` selects another Rust release (default 1.97, Debian
trixie base). The runtime image links GMP and MPFR statically. RBE's `registry_db`
stays inside the container. With Docker on Linux, the files written to `results/`
belong to root: `sudo chown -R "$USER": results`.

## Resources and running times (indicative)

| Task | Time | Memory |
| --- | --- | --- |
| First build of a crate | several minutes (GMP/MPFR) | modest |
| `test_all.sh` | about 10-20 minutes | modest |
| `FULL=1` additions (three Epoque tests) | several minutes each | about 2 GB (budget 5 GB) |
| `attack/dppy25_attack.py` with defaults | about 35-80 s | small |
| `attack/test_dppy25_attack.py` | about 10 s | small |
| `run_epoque --profile full` | several minutes | about 2 GB (1.7 GB of public matrices) |
| `run_all.sh`, trial settings | minutes rather than hours | modest |
| `run_all.sh`, default settings | several hours | several GB (Epoque `full32`: about 1.8 GB of public matrices plus its trapdoor setup) |

The ciphertexts are 121,432 bytes for basic (n1152), about 0.51 MB for Epoque `full`,
and about 20 MB for RBE at depth 162 on the rank-16 host.

## Troubleshooting

- **`--locked` fails or the lock file "needs to be updated".** Use Rust 1.96 or
  later and leave `Cargo.toml` and `Cargo.lock` unchanged. The lockfiles pin every
  dependency.
- **`gmp-mpfr-sys` fails to build.** Install a C compiler, `make` and `m4`. On
  Windows, use Docker Desktop or WSL 2.
- **`run benchmarks with --release` / `run experiments with --release`.** Add `--release`.
- **An RBE run panics about the registry or the database.** A leftover
  `registry_db` is in the working directory, or two RBE runs share one. Delete
  `registry_db` or use a fresh scratch directory. Run the RBE tests with
  `--test-threads=1`.
- **Epoque's full profile is killed.** It ran out of memory. Give the machine or
  Docker more memory, or use `--profile small` / `EPOQUE_PROFILE=small`.
- **`cannot pin to CPUs ...` or `taskset (util-linux) is missing`.** Set `CPUS=all`
  or a valid CPU list, or install util-linux.
- **The output already exists.** Choose a new path. This applies to `basic`'s
  `benchmark`, `run_epoque`, the folder given to `run_all.sh` and the RBE helper scripts.

## Notes on the included documentation

- The READMEs in `basic/`, `epoque/`, `rbe/`, `attack/` and `experiments/` describe the
  constructions, parameters, tests and measurement methods in detail. They also
  correspond the code to the paper equation by equation.
- Some documents refer to records that are not part of this repository: paths
  beginning with `../research/` or `../lab-resource/`, and `benchmarks/2026-...`
  folders.
- The test fixtures (`basic/tests/fixtures/python.json` and
  `epoque/tests/fixtures/python.json`) are checked in, and the Rust tests use them as
  they are. Regenerating them with `*/tests/regenerate_fixtures.py` needs the
  independent Python reference implementations, which are not included.

## Scope and limitations

These are research implementations:

- The code is variable time. It is not side-channel hardened and does not zeroize secrets.
- Hidden decryption returns bits even for a non-recipient. There is no authentication
  or presence test.
- The RBE API assumes honestly generated registrations and does not implement
  registry merging.
- The Epoque crate is not a voting system: it has no trustees, commitments, ballot
  proofs or complaint handling.
- The security levels quoted for the parameter sets are attack estimates with stated
  models and caveats. The per-crate READMEs and `rbe/security/` give them.
- Some tests and benchmarks use fixed public seeds for reproducibility. Never use
  those seeds for real keys or messages.

## Third-party code and licenses

- `rbe/implementation/upstream/` is the scalable-RBE prototype by Kloos, Lai, Siemer
  and Swarnakar (<https://github.com/jnsiemer/scalable_rbe_prototype>, revision
  5f1e741), licensed under the Mozilla Public License 2.0 (`rbe/LICENSE`,
  `rbe/implementation/upstream/LICENSE`). Its modifications are recorded in
  `rbe/implementation/n16-joint-profile.patch`. The extension crate in
  `rbe/implementation/extension/` is also MPL-2.0.
- The dependencies from crates.io come under their own licenses. Their versions are
  pinned in each crate's `Cargo.lock`.
- The remaining code (`basic/`, `epoque/`, `attack/`, `experiments/`) carries no
  separate license file.
