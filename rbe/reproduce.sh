#!/bin/sh
# Tests and benchmark for the RBE channel on the vendored rank-16 host (upstream/).
# Run from a disposable copy: tests and benchmark clear registry_db.
set -eu
if [ "$#" -ne 1 ] || [ -z "$1" ]; then
  echo "Usage: $0 NEW_OUTPUT_DIR" >&2
  exit 2
fi
case "$1" in
  /*) output=$1 ;;
  *) output=$PWD/$1 ;;
esac
mkdir -p "$(dirname "$output")"
mkdir "$output"
here=$(cd "$(dirname "$0")" && pwd)
cd "$here/implementation/extension"
export RAYON_NUM_THREADS=1
cargo test --locked --release -- --test-threads=1 --nocapture > "$output/tests.log" 2>&1
cd "$here"
sh run-benchmarks.sh "$output/benchmarks"
