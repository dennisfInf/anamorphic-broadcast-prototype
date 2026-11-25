# syntax=docker/dockerfile:1
#
# basic/, epoque/ and rbe/ in a Linux container, on Linux with Docker Engine or on
# Windows with Docker Desktop (Linux containers, the default WSL 2 backend). Nothing
# else is needed on the host: no Rust, MSYS2, GMP or MPFR. The build needs internet
# access for the base images, crates.io (RBE's host is vendored in
# rbe/implementation/upstream) and, for the bench image's matplotlib, PyPI. The
# first build takes several minutes, mostly to compile and test GMP and MPFR.
#
# All benchmarks in one run: the bench image runs experiments/run_all.sh, that is
# basic, then RBE at ranks 8 to 13 (it rebuilds the host once per rank), then Epoque,
# one after the other, pinned to the performance cores of a hybrid CPU (CPUS=auto) and
# each on THREADS worker threads (default: the pinned CPUs, at most 10; see
# experiments/README.md for pinning under Docker Desktop). Each run writes a new
# folder results/run-<UTC time> with one JSON file and log per scheme and RBE rank,
# environment.txt, summary.md and the paper figures (figure-costs.pdf,
# figure-overhead.pdf), and last the attack simulation of attack/ (attack.txt,
# attack.pdf; -e SCHEMES=attack runs it alone). On Linux, from this folder:
#
#   docker build --target bench -t anamorphic-bench .
#   mkdir -p results
#   docker run -d --name bench -v "$PWD/results:/results" anamorphic-bench
#   docker logs -f bench    (progress, Ctrl-C stops only the viewer)
#   docker wait bench       (waits for the end, prints 0 if every scheme passed)
#   docker rm bench
#
# The settings of run_all.sh are environment variables, for example
# -e REPS=3 -e MIN_TIME=0 -e EPOQUE_PROFILE=small for a check that takes minutes. Every
# scheme runs on THREADS worker threads (for example -e THREADS=1), which
# environment.txt records with the CPUs the container sees and the pinning. On Linux the
# containers use all cores and all memory unless limited, and the files in results/
# belong to root (sudo chown -R "$USER": results). In PowerShell, write
# "${PWD}\results" instead of "$PWD/results".
#
# Single programs run in the smaller runtime image, built in PowerShell from this
# folder with
#
#   docker build -t anamorphic .
#
# Run a program. Write its output under /results, which the -v option maps to
# the folder .\results on Windows:
#
#   mkdir results
#   docker run --rm -v "${PWD}\results:/results" anamorphic basic-benchmark --output /results/basic.json
#   docker run --rm -v "${PWD}\results:/results" anamorphic epoque-run --profile small --output /results/epoque-small
#   docker run --rm -v "${PWD}\results:/results" -e RAYON_NUM_THREADS=1 anamorphic rbe-bench-ab --reps 50 --output /results/rbe-ab.json
#
# Programs, each with the options described in the READMEs:
#   basic-benchmark     basic/src/bin/benchmark.rs   (--help lists the options)
#   basic-noise-table   basic/src/bin/noise_table.rs
#   basic-noise-stats   basic/examples/noise_stats.rs
#   epoque-run          epoque/src/bin/run_epoque.rs
#   epoque-bench-ab     epoque/src/bin/bench_ab.rs
#   rbe-bench-ab        rbe/implementation/extension/src/main.rs (--reps, --depth, --workloads, --output)
# -e RAYON_NUM_THREADS=N sets the thread count of all three implementations (basic's
# multi-thread mode, RBE's host and Epoque); the default is every logical CPU.
# Epoque's full profile needs about 5 GB of memory; Docker Desktop gets half of
# the computer's memory by default.
#
# For the tests and the cargo commands of the READMEs, build the development
# image instead. It keeps the toolchain, the sources and prebuilt test binaries.
# test_all.sh runs every test suite (FULL=1 adds Epoque's full-profile tests,
# which need about 5 GB of memory):
#
#   docker build --target dev -t anamorphic-dev .
#   docker run --rm anamorphic-dev sh test_all.sh
#   docker run --rm -e FULL=1 anamorphic-dev sh test_all.sh
#   docker run --rm anamorphic-dev cargo test --locked --release --manifest-path basic/Cargo.toml
#   docker run --rm -it anamorphic-dev          (a shell in /src)
#
# --build-arg RUST_VERSION=1.99 selects another Rust release; the code was
# tested with 1.96 and 1.97.

ARG RUST_VERSION=1.97
ARG DEBIAN=trixie

FROM rust:${RUST_VERSION}-slim-${DEBIAN} AS build
# rug builds GMP and MPFR from source, which needs a C compiler, make and m4.
RUN apt-get update \
 && apt-get install -y --no-install-recommends build-essential m4 \
 && rm -rf /var/lib/apt/lists/*
# One target directory for the three crates. GMP and MPFR are kept in a build
# cache, so a rebuild after a source change does not compile them again.
ENV CARGO_TARGET_DIR=/src/target \
    GMP_MPFR_SYS_CACHE=/var/cache/gmp-mpfr-sys
WORKDIR /src
COPY . .
# A checkout on Windows may have turned LF into CRLF; shell scripts need LF.
RUN find . -name '*.sh' -exec sed -i 's/\r$//' {} +
RUN --mount=type=cache,target=/var/cache/gmp-mpfr-sys \
    cargo build --locked --release --manifest-path basic/Cargo.toml --bins --examples \
 && cargo build --locked --release --manifest-path epoque/Cargo.toml --bins \
 && cargo build --locked --release --manifest-path rbe/implementation/extension/Cargo.toml

FROM build AS dev
# python3 for the tests of the attack simulation in test_all.sh.
RUN apt-get update \
 && apt-get install -y --no-install-recommends python3 \
 && rm -rf /var/lib/apt/lists/*
RUN --mount=type=cache,target=/var/cache/gmp-mpfr-sys \
    cargo test --locked --release --no-run --manifest-path basic/Cargo.toml \
 && cargo test --locked --release --no-run --manifest-path epoque/Cargo.toml \
 && cargo test --locked --release --no-run --manifest-path rbe/implementation/extension/Cargo.toml \
 && cargo test --locked --release --no-run --manifest-path rbe/implementation/upstream/Cargo.toml
CMD ["bash"]

# run_all.sh rebuilds the RBE host for every rank with cargo, summarizes the JSON
# files with Python and draws the figures with matplotlib, so this image keeps the
# toolchain and adds python3 with matplotlib (in a virtual environment on PATH).
FROM build AS bench
RUN apt-get update \
 && apt-get install -y --no-install-recommends python3 python3-venv \
 && rm -rf /var/lib/apt/lists/* \
 && python3 -m venv /opt/plot \
 && /opt/plot/bin/pip install --no-cache-dir "matplotlib>=3.8"
ENV PATH=/opt/plot/bin:$PATH
CMD ["sh", "-c", "exec sh experiments/run_all.sh /results/run-$(date -u +%Y%m%d-%H%M%S)"]

FROM debian:${DEBIAN}-slim AS runtime
# GMP and MPFR are linked statically; the programs need only the C library.
COPY --from=build /src/target/release/benchmark /usr/local/bin/basic-benchmark
COPY --from=build /src/target/release/noise_table /usr/local/bin/basic-noise-table
COPY --from=build /src/target/release/examples/noise_stats /usr/local/bin/basic-noise-stats
COPY --from=build /src/target/release/run_epoque /usr/local/bin/epoque-run
COPY --from=build /src/target/release/bench_ab /usr/local/bin/epoque-bench-ab
COPY --from=build /src/target/release/rbe-anamorphic-review /usr/local/bin/rbe-bench-ab
# Scratch files such as RBE's registry_db stay inside the container.
WORKDIR /work
CMD ["basic-benchmark", "--help"]