#!/bin/sh
# Runs the test suites of basic/, epoque/ and rbe/ (release builds, locked crates).
#
#   sh test_all.sh             every default suite (about 10 to 20 minutes)
#   FULL=1 sh test_all.sh      also Epoque's three opt-in tests of the full n = 1360
#                              profile (about 5 GB of memory, a few minutes each)
#
# In Docker, from this folder:
#   docker build --target dev -t anamorphic-dev .
#   docker run --rm anamorphic-dev sh test_all.sh
#   docker run --rm -e FULL=1 anamorphic-dev sh test_all.sh
#
# The attack simulation's tests (attack/test_dppy25_attack.py) need python3, which the
# dev image has; elsewhere they are skipped with a note if python3 is missing.
#
# RBE's tests clear ./registry_db in their working directory, so they run one at a
# time (and with RAYON_NUM_THREADS=1, as its README prescribes) in a scratch copy of
# rbe/implementation. A failing suite does not stop the others; the exit status is
# nonzero if any failed, and the last lines list them.
set -u
code=$(cd "$(dirname "$0")" && pwd)
failed=""
run() {
  name=$1
  shift
  echo "== $name ($(date -u +%H:%M:%S) UTC)" >&2
  if ! "$@"; then
    failed="$failed $name"
  fi
}

epoque_extra=""
if [ "${FULL:-0}" = 1 ]; then
  epoque_extra="--include-ignored"
fi
run basic cargo test --locked --release --manifest-path "$code/basic/Cargo.toml"
if command -v python3 > /dev/null 2>&1; then
  run attack python3 "$code/attack/test_dppy25_attack.py"
else
  echo "== attack: skipped, python3 is not installed" >&2
fi
run epoque cargo test --locked --release --manifest-path "$code/epoque/Cargo.toml" -- $epoque_extra

scratch=$(mktemp -d)
cp -R "$code/rbe/implementation" "$scratch/"
cd "$scratch/implementation/extension"
run rbe-extension env RAYON_NUM_THREADS=1 cargo test --locked --release -- --test-threads=1
run rbe-upstream env RAYON_NUM_THREADS=1 cargo test --locked --release \
  --manifest-path ../upstream/Cargo.toml -- --test-threads=1
cd "$code"
rm -rf "$scratch"

if [ -n "$failed" ]; then
  echo "FAILED:$failed" >&2
  exit 1
fi
echo "All test suites passed." >&2
