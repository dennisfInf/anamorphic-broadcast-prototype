#!/bin/sh
# Runs the experiments of basic/, rbe/ and epoque/, summarizes them, and runs the
# simulation of the attack on DPPY25 in attack/.
#
#   sh experiments/run_all.sh NEW_OUTPUT_DIR
#
# NEW_OUTPUT_DIR must not exist yet. Put it outside code/: every run writes JSON
# results, logs and, for RBE, a temporary registry_db there. Settings come from the
# environment (defaults in brackets):
#   SCHEMES         parts to run, in order: basic, rbe, epoque and attack (the
#                   simulation, not timed, so never pinned) ["basic rbe epoque attack"]
#   ATTACK_TRIALS   experiments per mode of the attack simulation [1000]
#   ATTACK_SEED     its random seed                    [2025]
#   CPUS            CPUs every run is pinned to (taskset): auto, all, or a CPU list
#                   such as 0-3. auto pins to the performance cores of a hybrid Intel
#                   CPU (/sys/devices/cpu_core/cpus) and does not pin elsewhere [auto]
#   THREADS         worker threads of every scheme     [the pinned CPUs, at most 10]
#   REPS            runs of every measured algorithm   [50]
#   WARMUP          untimed encryption rounds per trade [2]
#   MIN_TIME        seconds that every loop of fast operations runs at least, after
#                   its REPS runs (at most 20000 runs) [10]
#   STEP_REPS       paired iterations of the step in which Enc and aEnc differ,
#                   timed alone per trade               [2000]
#   KEYGEN_REPS     runs of aKGen (largest N each)     [REPS]
#   SETUP_REPS      runs of Epoque's ordinary setup    [REPS]
#   BASIC_PROFILE   basic parameters: g256 | n1152      [g256]
#   BASIC_MODE      basic matrix work: single | multi  [multi]
#   EPOQUE_PROFILE  Epoque: full32 | full | small      [full32]
#   RBE_PROFILE     RBE hosts: scaled | rank7 | rank16   [scaled]
# For basic and Epoque every sender-side cost depends on the number of equations
# d = |S| L only, and aDec on L only, so their binaries sweep d: |S| = 2 recipients
# with L = 1, 2, 4, ... bits up to the parameters' budget (basic g256: d = 2 to 256,
# Epoque full32: d = 2 to 64), and repeat the largest d with two other splits (basic
# 128 x 2 and 16 x 16, Epoque 32 x 2 and 8 x 8) as a check. aKGen generates keys for
# the group's recipients only. RBE runs 1 to 6 recipients of 256 bits (one ring
# equation each). RBE_PROFILE=scaled runs |S| recipients on a host of rank 7 + |S|
# (8 to 13), so the authors' rank 7 stays free and the security level of their
# benchmark host is kept, one build per rank (RBE_RANK). rank7 runs 1 to 6 recipients
# on the authors' rank-7 host (cost only, each recipient takes one of its seven ranks),
# rank16 runs 1 to 4 on the paper's joint profile. RBE runs at identity depth 162,
# Epoque for a random identity.
# Enc and aEnc are timed in the same rounds, Enc first in even rounds and aEnc first in
# odd ones, so that drift of the machine falls on both alike. The step in which they
# differ (the secret s, or RBE's root r_0) is also timed alone, STEP_REPS times per
# trade; its difference is aEnc's overhead (step.overhead in the JSON files).
# THREADS sets the rayon pool of all three (RAYON_NUM_THREADS): basic's matrix work
# (BASIC_MODE=multi), RBE's host and Epoque's parallel steps. Results do not depend
# on it; it should not exceed the CPUs the runs may use. The binaries are built
# before each run, unpinned, so that only the measured runs are pinned.
# environment.txt records the CPUs, the CPU frequency governor, turbo and power.
set -eu
if [ "$#" -ne 1 ] || [ -z "$1" ]; then
  echo "Usage: $0 NEW_OUTPUT_DIR" >&2
  exit 2
fi
case "$1" in
  /*) output=$1 ;;
  *) output=$PWD/$1 ;;
esac
code=$(cd "$(dirname "$0")/.." && pwd)

positive() {
  case "$2" in
    '' | *[!0-9]* | 0*) echo "$1 must be a positive integer" >&2; exit 2 ;;
  esac
}
# The number of CPUs in a CPU list such as 0-3,8,10-11.
count_cpus() {
  echo "$1" | awk -F, '{ n = 0; for (i = 1; i <= NF; i++) { k = split($i, r, "-");
    n += (k == 2 ? r[2] - r[1] + 1 : 1) } print n }'
}
# The first line of a file, or "unknown" where it cannot be read.
first_line() {
  if [ -r "$1" ]; then head -n 1 "$1"; else echo unknown; fi
}

SCHEMES=${SCHEMES:-"basic rbe epoque attack"}
ATTACK_TRIALS=${ATTACK_TRIALS:-1000}
positive ATTACK_TRIALS "$ATTACK_TRIALS"
ATTACK_SEED=${ATTACK_SEED:-2025}
case "$ATTACK_SEED" in
  '' | *[!0-9]*) echo "ATTACK_SEED must be a non-negative integer" >&2; exit 2 ;;
esac
CPUS=${CPUS:-auto}
REPS=${REPS:-50}
WARMUP=${WARMUP:-2}
case "$WARMUP" in
  '' | *[!0-9]*) echo "WARMUP must be a non-negative integer" >&2; exit 2 ;;
esac
MIN_TIME=${MIN_TIME:-10}
case "$MIN_TIME" in
  '' | *[!0-9.]* | *.*.* | .) echo "MIN_TIME must be a non-negative number of seconds" >&2; exit 2 ;;
esac
STEP_REPS=${STEP_REPS:-2000}
positive STEP_REPS "$STEP_REPS"
KEYGEN_REPS=${KEYGEN_REPS:-$REPS}
SETUP_REPS=${SETUP_REPS:-$REPS}
BASIC_PROFILE=${BASIC_PROFILE:-g256}
BASIC_MODE=${BASIC_MODE:-multi}
EPOQUE_PROFILE=${EPOQUE_PROFILE:-full32}
RBE_PROFILE=${RBE_PROFILE:-scaled}
# rank:groups pairs of the RBE runs, one build of the host per rank.
case "$RBE_PROFILE" in
  scaled) rbe_plan="8:1 9:2 10:3 11:4 12:5 13:6" ;;
  rank7) rbe_plan="7:6,5,4,3,2,1" ;;
  rank16) rbe_plan="16:4,3,2,1" ;;
  *) echo "RBE_PROFILE must be scaled, rank7 or rank16" >&2; exit 2 ;;
esac

# The CPUs to pin to: on a hybrid Intel CPU (2 performance and 8 efficiency cores on
# an i5-1235U, for example) a run left to the scheduler can land on either kind.
hybrid=""
if [ -r /sys/devices/cpu_core/cpus ]; then
  hybrid=$(first_line /sys/devices/cpu_core/cpus)
fi
case "$CPUS" in
  auto) pinned=$hybrid ;;
  all) pinned="" ;;
  *[!0-9,-]* | '') echo "CPUS must be auto, all or a CPU list such as 0-3" >&2; exit 2 ;;
  *) pinned=$CPUS ;;
esac
pin=""
if [ -n "$pinned" ]; then
  if ! command -v taskset > /dev/null 2>&1; then
    if [ "$CPUS" != auto ]; then
      echo "CPUS=$CPUS needs taskset (util-linux)" >&2
      exit 2
    fi
    echo "warning: taskset (util-linux) is missing, the runs are not pinned" >&2
    pinned=""
  elif ! taskset -c "$pinned" true 2> /dev/null; then
    echo "cannot pin to CPUs $pinned (not available here?)" >&2
    exit 2
  else
    pin="taskset -c $pinned"
  fi
fi
available=$(nproc 2>/dev/null || echo 1)
if [ -n "$pinned" ]; then
  available=$(count_cpus "$pinned")
fi
THREADS=${THREADS:-$(( available < 10 ? available : 10 ))}
positive THREADS "$THREADS"
if [ "$THREADS" -gt "$available" ]; then
  echo "warning: THREADS=$THREADS exceeds the $available CPUs the runs may use" >&2
fi
case "$(uname -r)" in
  *[Mm]icrosoft* | *WSL*)
    echo "warning: this is WSL 2: its CPUs are virtual and Windows decides which cores run" \
      "them, so CPUS cannot select performance cores (see experiments/README.md)" >&2 ;;
esac

mkdir -p "$(dirname "$output")"
mkdir "$output"
common="--reps $REPS --warmup $WARMUP --keygen-reps $KEYGEN_REPS --min-time $MIN_TIME --step-reps $STEP_REPS"
{
  echo "date: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "host: $(uname -srvm)"
  echo "rustc: $(rustc --version)"
  echo "schemes: $SCHEMES"
  echo "threads: $THREADS (RAYON_NUM_THREADS of every scheme)"
  echo "logical CPUs: $(nproc 2>/dev/null || echo unknown)"
  echo "pinned to CPUs: ${pinned:-none} (CPUS=$CPUS)"
  echo "performance cores: ${hybrid:-not reported (not a hybrid CPU, or no /sys/devices/cpu_core)}"
  echo "efficiency cores: $(first_line /sys/devices/cpu_atom/cpus)"
  echo "cpu: $(sed -n 's/^model name[[:space:]]*: //p' /proc/cpuinfo 2>/dev/null | head -n 1)"
  echo "frequency governor (cpu0): $(first_line /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor)"
  echo "energy-performance preference (cpu0): $(first_line /sys/devices/system/cpu/cpu0/cpufreq/energy_performance_preference)"
  echo "intel_pstate no_turbo: $(first_line /sys/devices/system/cpu/intel_pstate/no_turbo)"
  echo "boost (acpi-cpufreq): $(first_line /sys/devices/system/cpu/cpufreq/boost)"
  for supply in /sys/class/power_supply/A*/online; do
    if [ -r "$supply" ]; then
      echo "on mains power ($(basename "$(dirname "$supply")")): $(first_line "$supply")"
    fi
  done
  echo "memory: $(sed -n 's/^MemTotal:[[:space:]]*//p' /proc/meminfo 2>/dev/null)"
  echo "options: $common, setup reps $SETUP_REPS, basic profile $BASIC_PROFILE, basic mode $BASIC_MODE, Epoque profile $EPOQUE_PROFILE, RBE profile $RBE_PROFILE ($rbe_plan), attack trials $ATTACK_TRIALS, attack seed $ATTACK_SEED"
} > "$output/environment.txt"

run_scheme() {
  case "$1" in
    basic)
      manifest="$code/basic/Cargo.toml"
      cargo build --locked --release --quiet --manifest-path "$manifest" --bin experiments \
        2> "$output/basic.log"
      RAYON_NUM_THREADS=$THREADS $pin cargo run --locked --release --quiet --manifest-path "$manifest" \
        --bin experiments -- $common --profile "$BASIC_PROFILE" --mode "$BASIC_MODE" \
        --output "$output/basic.json" 2>> "$output/basic.log"
      ;;
    rbe)
      # The RBE host keeps its registry in ./registry_db: run in a scratch directory.
      manifest="$code/rbe/implementation/extension/Cargo.toml"
      mkdir -p "$output/rbe-work"
      status=0
      for item in $rbe_plan; do
        rank=${item%%:*}
        groups=${item#*:}
        log="$output/rbe-rank$rank.log"
        echo "   RBE rank $rank, |S| = $groups, log: $log" >&2
        (cd "$output/rbe-work" \
          && RBE_RANK=$rank cargo build --locked --release --quiet --manifest-path "$manifest" \
            --bin experiments \
          && RBE_RANK=$rank RAYON_NUM_THREADS=$THREADS $pin cargo run --locked --release --quiet \
            --manifest-path "$manifest" --bin experiments -- $common --groups "$groups" \
            --output "$output/rbe-rank$rank.json") 2> "$log" || status=1
      done
      rm -rf "$output/rbe-work"
      return $status
      ;;
    epoque)
      manifest="$code/epoque/Cargo.toml"
      cargo build --locked --release --quiet --manifest-path "$manifest" --bin experiments \
        2> "$output/epoque.log"
      RAYON_NUM_THREADS=$THREADS $pin cargo run --locked --release --quiet \
        --manifest-path "$manifest" \
        --bin experiments -- $common --setup-reps "$SETUP_REPS" \
        --profile "$EPOQUE_PROFILE" --output "$output/epoque.json" 2>> "$output/epoque.log"
      ;;
    attack)
      # Proposition 3.1 on DPPY25 and Construction 1: attack.txt is the report,
      # attack.json the counts, attack.pdf and attack.png the figure (with matplotlib).
      python3 "$code/attack/dppy25_attack.py" --trials "$ATTACK_TRIALS" --seed "$ATTACK_SEED" \
        --json "$output/attack.json" --plot "$output/attack" --progress \
        > "$output/attack.txt" 2> "$output/attack.log"
      ;;
    *)
      echo "unknown part $1 (use basic, rbe, epoque, attack)" >&2
      return 2
      ;;
  esac
}

# A failed scheme does not stop the others. The summary covers what was written.
failed=0
results=""
for scheme in $SCHEMES; do
  echo "== $scheme ($(date -u +%H:%M:%S) UTC), log: $output/$scheme.log" >&2
  if ! run_scheme "$scheme"; then
    echo "$scheme failed, see $output/$scheme.log" >&2
    failed=1
  fi
  case "$scheme" in
    # The simulation's report is attack.txt; its JSON file is no benchmark result.
    attack) continue ;;
    rbe)
      files=""
      for item in $rbe_plan; do
        files="$files $output/rbe-rank${item%%:*}.json"
      done
      ;;
    *) files="$output/$scheme.json" ;;
  esac
  for file in $files; do
    if [ -f "$file" ]; then
      results="$results $file"
    fi
  done
done
if [ -n "$results" ]; then
  python3 "$code/experiments/summarize.py" $results --output "$output/summary"
  # The paper figures, where matplotlib is installed (the bench image has it).
  if python3 -c 'import matplotlib' 2> /dev/null; then
    python3 "$code/experiments/plot.py" "$output" --output "$output/figure" || true
  fi
fi
echo "Results in $output (summary.md, summary.tex, summary.csv, summary-sizes.csv, one JSON file and log per scheme and RBE rank, attack.txt and attack.pdf)." >&2
exit $failed
