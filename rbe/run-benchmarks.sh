#!/bin/sh
# Encryption benchmark of RBE alone and RBE with the anamorphic broadcast inside.
# Run from a disposable copy: the benchmark creates and clears registry_db.
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
cd "$(dirname "$0")/implementation/extension"
export RAYON_NUM_THREADS=1
cargo build --locked --release
cargo run --locked --release -q -- --reps 50 --output "$output/rbe-ab-benchmark.json" \
  2> "$output/benchmark.log"
