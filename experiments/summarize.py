"""Summarize the JSON results of the experiments binaries (basic, rbe, epoque).

    python3 experiments/summarize.py basic.json rbe.json epoque.json --output summary

writes summary.csv (one row per scheme, trade and operation, times in ms),
summary-sizes.csv (one row per scheme and size item, in bytes), summary.md (per
scheme the ordinary algorithms, the anamorphic ones per trade |S|, L with d = |S| L
equations, Enc and aEnc of the same rounds side by side, the overhead of aEnc, and
the key and ciphertext sizes) and summary.tex (the same rows as LaTeX tables, booktabs
rules). Standard library only.

The overhead is the step in which aEnc differs from Enc, timed alone in paired
iterations (`step` in every row): the constrained secret s of aEnc against the
uniform s of Enc (basic, Epoque), or the constrained root r_0 of aEnc against the
Gaussian r_0 of Enc (RBE, where it can be negative). It is given as the median of the
paired differences with a distribution-free 95% interval (order statistics), because a
step of microseconds is now and then hit by an interrupt or a preemption, which would
dominate its mean. The paired difference of the totals aEnc - Enc of the same round is
the check: its interval should contain the overhead, but it is far wider, because a
whole encryption varies from run to run by much more. Files from before the step was
timed alone fall back to the overhead timed inside aEnc (`encryption_overhead`), or to
a warning for files from before the paired rounds.
"""

import argparse
import csv
import json
import math
import sys

ORDINARY = [("setup", "Setup"), ("encryption", "Enc"), ("decryption", "Dec")]
ANAMORPHIC = [
    ("keygen", "aKGen"),
    ("group_setup", "Group setup"),
    ("setup", "Anamorphic setup"),
    ("ordinary_encryption", "Enc"),
    ("encryption", "aEnc"),
    ("decryption", "aDec"),
]
NAMES = {"basic": "Basic (Dual-Regev)", "rbe": "RBE", "epoque": "Epoque"}
SIZES = [
    ("ordinary_public_key", "Ordinary public key"),
    ("ordinary_secret_key", "Ordinary secret key"),
    ("public_parameters", "Public parameters"),
    ("state", "Registry state (root digests)"),
    ("ordinary_ciphertext", "Ordinary ciphertext"),
    ("anamorphic_ciphertext", "Anamorphic ciphertext"),
    ("anamorphic_public_key_per_bit", "Anamorphic public key, per hidden bit"),
    ("anamorphic_secret_key_per_bit", "Anamorphic secret key, per hidden bit"),
    ("anamorphic_secret_key_per_bit_expanded", "Anamorphic secret key, per hidden bit, expanded"),
    ("anamorphic_public_key_per_user", "Anamorphic public key, per user of the largest trade"),
    ("anamorphic_secret_key_per_user", "Anamorphic secret key, per user of the largest trade"),
    ("anamorphic_secret_key_per_user_expanded",
     "Anamorphic secret key, per user of the largest trade, expanded"),
    ("anamorphic_group_matrix", "Group setup output H_S of the largest trade"),
]
# Result files written before sizes_bytes existed carried sizes_bits.
LEGACY_SIZES = {
    "public_key": "ordinary_public_key",
    "ciphertext": "ordinary_ciphertext",
    "apk_per_user": "anamorphic_public_key_per_user",
    "ask_per_user": "anamorphic_secret_key_per_user",
    "ask_per_user_seed": "anamorphic_secret_key_per_user",
    "ask_per_user_expanded": "anamorphic_secret_key_per_user_expanded",
    "host_state": "state",
}


def sizes(report):
    """Key and ciphertext sizes in bytes, from sizes_bytes or the older sizes_bits."""
    if "sizes_bytes" in report:
        return dict(report["sizes_bytes"])
    out = {}
    for key, bits in report.get("sizes_bits", {}).items():
        out[LEGACY_SIZES.get(key, key)] = (bits + 7) // 8
    return out


def readable_bytes(count):
    """A byte count with separators and a decimal unit."""
    for limit, unit in ((1e3, "B"), (1e6, "kB"), (1e9, "MB"), (1e12, "GB")):
        if count < limit:
            scale = limit / 1e3
            return f"{count:,} B" if unit == "B" else f"{count:,} B ({count / scale:.3g} {unit})"
    return f"{count:,} B ({count / 1e12:.3g} TB)"


def readable(ms):
    """A duration in ms as a short string with a fitting unit."""
    if ms < 1:
        return f"{ms * 1000:.3g} µs"
    if ms < 1000:
        return f"{ms:.3g} ms"
    seconds = ms / 1000
    for limit, size, unit in ((120, 1, "s"), (7200, 60, "min"), (172800, 3600, "h"),
                              (63072000, 86400, "days")):
        if seconds < limit:
            return f"{seconds / size:.3g} {unit}"
    return f"{seconds / 31536000:.3g} years"


def cell(stats):
    """Mean and standard deviation (the mean alone for a single run)."""
    if not stats:
        return "n/a"
    if stats["runs"] < 2:
        return readable(stats["mean_ms"])
    return f"{readable(stats['mean_ms'])} ± {readable(stats['sd_ms'])}"


def signed(ms):
    """A duration in ms that may be negative, with its sign."""
    if math.isinf(ms):
        return "−∞" if ms < 0 else "+∞"
    return ("−" if ms < 0 else "+") + readable(abs(ms))


def median_interval(xs, level=0.95):
    """The median of xs and a distribution-free interval for it: the order statistics
    s_(j) and s_(n+1-j) with the largest j for which P(Binomial(n, 1/2) < j) is at most
    (1 - level) / 2, so the coverage is at least `level`. Infinite below 6 samples."""
    s = sorted(xs)
    n = len(s)
    median = (s[(n - 1) // 2] + s[n // 2]) / 2
    # The binomial tail in logarithms: 2^-n underflows for the long series of the
    # decryption loops.
    tail, j = 0.0, 0
    while j < n:
        term = math.exp(math.lgamma(n + 1) - math.lgamma(j + 1) - math.lgamma(n - j + 1)
                        - n * math.log(2))
        if tail + term > (1 - level) / 2:
            break
        tail += term
        j += 1
    if j == 0:
        return median, -math.inf, math.inf
    return median, s[j - 1], s[n - j]


def bounds(low, high, show):
    """An interval as [low, high], or a note where the rounds are too few for one."""
    if not (math.isfinite(low) and math.isfinite(high)):
        return "[none below 6 runs]"
    return f"[{show(low)}, {show(high)}]"


def percent(ratio):
    return ("−" if ratio < 0 else "+") + f"{abs(100 * ratio):.3g}%"


def sentence(text):
    """Text with its first letter capitalized."""
    return text[:1].upper() + text[1:]


def group_label(row):
    """(length, |S|, L, equations) of a row (older files: N users, no length)."""
    if "recipients" in row:
        return (row.get("length", row.get("set", "")), row["recipients"], row.get("bits", ""),
                row.get("constraints", ""))
    return f"N={row['users']}", row["users"], "", ""


def scheme_label(report):
    """The scheme's name, with the host rank for RBE runs that record it."""
    rank = report.get("config", {}).get("rank")
    return report["scheme"] if rank is None else f"{report['scheme']} (rank {rank})"


def step_overhead(row):
    """The step overhead of a row: (median, low, high) of its 95% interval, in ms."""
    return median_interval(row["step"]["overhead"]["samples_ms"])


def totals_difference(row, level=0.95):
    """aEnc minus Enc of the same round: (median, low, high), in ms."""
    return median_interval(row["encryption_difference"]["samples_ms"], level)


def step_table(report):
    """The overhead of aEnc per row: the differing step timed alone, with the totals'
    difference of the same rounds as the check."""
    rows = report.get("anamorphic", [])
    if not all("step" in row for row in rows):
        return insitu_table(report)
    config = report.get("config", {})
    definitions = report.get("definitions", {})
    rbe = report["scheme"] == "rbe"
    lines = [
        "**Overhead of aEnc over Enc.** "
        + sentence(definitions.get("anamorphic.step",
                                   "the step in which aEnc differs from Enc, alone"))
        + f" ({config.get('step_reps', rows[0]['step']['iterations'])} paired iterations per "
        "row). Medians, the overhead with a distribution-free 95% interval. The totals' "
        "difference aEnc − Enc of the same encryption round is the check: its interval "
        "should contain the overhead but is far wider, since a whole encryption varies "
        "from run to run by much more than the step."
        + (" For RBE the overhead can be negative: aEnc computes r_0 instead of drawing the "
           "Gaussian r_0 of Enc." if rbe else ""),
        "",
        "| \\|S\\| | L | d | Role | Replaced step | Broadcast step | Overhead [95%] "
        "| Overhead / Enc | aEnc − Enc of the totals [95%] |",
        "|---:|---:|---:|---|---:|---:|---:|---:|---:|",
    ]
    flagged = False
    for row in rows:
        _, count, bits, equations = group_label(row)
        o, o_low, o_high = step_overhead(row)
        d, d_low, d_high = totals_difference(row)
        # Flag only beyond chance: a 95% interval misses in one row of twenty.
        _, d_far_low, d_far_high = totals_difference(row, level=0.999)
        mark = ""
        if not d_far_low <= o <= d_far_high:
            mark, flagged = " †", True
        enc = row["ordinary_encryption"]["median_ms"]
        lines.append(
            f"| {count} | {bits} | {equations} | {row.get('role', '')} "
            f"| {readable(row['step']['replaced']['median_ms'])} "
            f"| {readable(row['step']['broadcast']['median_ms'])} "
            f"| {signed(o)} {bounds(o_low, o_high, signed)} "
            f"| {percent(o / enc)} "
            f"| {signed(d)} {bounds(d_low, d_high, signed)}{mark} |")
    lines.append("")
    if flagged:
        lines += ["† Even the 99.9% interval of the totals' difference excludes the overhead: "
                  "Enc and aEnc of these rows differ in more than the step (or the machine "
                  "was disturbed during the rounds). Check the samples before using them.", ""]
    lows = [(step_overhead(row)[1], row) for row in rows]
    highs = [(step_overhead(row)[2], row) for row in rows]
    low, low_row = min(lows, key=lambda x: x[0])
    high, high_row = max(highs, key=lambda x: x[0])
    widths = [totals_difference(row) for row in rows]
    half = [(h - lo) / 2 for _, lo, h in widths if math.isfinite(lo) and math.isfinite(h)]
    if math.isfinite(low) and math.isfinite(high):
        lines += [
            "Over all rows the overhead lies between "
            f"{signed(low)} (\\|S\\| = {low_row['recipients']}, L = {low_row['bits']}) and "
            f"{signed(high)} (\\|S\\| = {high_row['recipients']}, L = {high_row['bits']}), "
            "the extreme 95% bounds; the upper one is "
            f"{percent(high / high_row['ordinary_encryption']['median_ms'])} of that row's "
            "Enc."
            + (" The totals of a round resolve differences of about "
               f"±{readable(min(half))} to ±{readable(max(half))} (half the width of "
               "their 95% interval)." if half else ""),
            "",
        ]
    return lines


def paired_overhead(row):
    """Per round of a row: the overhead timed inside aEnc (ms), the overhead relative
    to Enc of the same round, and aEnc minus Enc, each as (median, low, high)."""
    overhead = row["encryption_overhead"]["samples_ms"]
    enc = row["ordinary_encryption"]["samples_ms"]
    difference = row["encryption_difference"]["samples_ms"]
    relative = [o / e for o, e in zip(overhead, enc)]
    return median_interval(overhead), median_interval(relative), median_interval(difference)


def insitu_table(report):
    """The overhead timed inside aEnc, for files from before the step was timed alone."""
    rows = report.get("anamorphic", [])
    needed = ("encryption_overhead", "ordinary_encryption", "encryption_difference")
    if not all(key in row for row in rows for key in needed):
        return ["**Encryption overhead: not available.** This file predates timing Enc and "
                "aEnc in the same rounds. Its Enc and aEnc were timed in separate blocks, so "
                "aEnc − Enc includes the drift of the machine between the blocks and can come "
                "out negative. Re-run the experiments binary.", ""]
    definitions = report.get("definitions", {})
    config = report.get("config", {})
    lines = [
        "Encryption overhead of aEnc over Enc, timed inside aEnc (a file from before the "
        "step was timed alone). In each of the "
        f"{rows[0]['ordinary_encryption']['runs']} rounds per row (after "
        f"{config.get('warmup_rounds', 'the')} untimed ones) fresh messages were encrypted "
        "once with Enc and once with aEnc, in alternating order. Overhead: "
        + definitions.get("anamorphic.encryption_overhead", "aEnc's work beyond Enc")
        + ". Medians over the rounds with distribution-free 95% intervals.",
        "",
        "| \\|S\\| | L | Broadcast step | Overhead [95%] | Overhead / Enc [95%] "
        "| aEnc − Enc of the totals [95%] |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        _, count, bits, _ = group_label(row)
        (o, o_low, o_high), (r, r_low, r_high), (d, d_low, d_high) = paired_overhead(row)
        lines.append(
            f"| {count} | {bits} | {readable(row['broadcast_step']['median_ms'])} "
            f"| {signed(o)} {bounds(o_low, o_high, signed)} "
            f"| {percent(r)} {bounds(r_low, r_high, percent)} "
            f"| {signed(d)} {bounds(d_low, d_high, signed)} |")
    lines.append("")
    return lines


def stats_items(row):
    """(operation, stats) of a row, the step's series as step.replaced and so on."""
    for key, stats in row.items():
        if isinstance(stats, dict) and "mean_ms" in stats:
            yield key, stats
        elif key == "step" and isinstance(stats, dict):
            for part, inner in stats.items():
                if isinstance(inner, dict) and "mean_ms" in inner:
                    yield f"step.{part}", inner


def csv_rows(report):
    scheme = scheme_label(report)
    for key, stats in report.get("ordinary", {}).items():
        yield [scheme, "", "", "", "", "", f"ordinary.{key}"] + numbers(stats)
    for row in report.get("anamorphic", []):
        length, count, bits, equations = group_label(row)
        for key, stats in stats_items(row):
            yield ([scheme, count, bits, equations, length, row.get("role", ""),
                    f"anamorphic.{key}"] + numbers(stats))


def numbers(stats):
    return [stats["runs"]] + [
        f"{stats[k]:.6f}" for k in ("mean_ms", "sd_ms", "median_ms", "min_ms", "max_ms")
    ]


def markdown(report):
    scheme = report["scheme"]
    config = report.get("config", {})
    rank = config.get("rank")
    lines = [f"## {NAMES.get(scheme, scheme)}" + ("" if rank is None else f", rank {rank}"), ""]
    if not report.get("complete"):
        lines += ["**Incomplete run: the tables below hold the parts that finished.**", ""]
    details = ", ".join(
        f"{k} {v}"
        for k, v in config.items()
        if k in ("profile", "reps", "warmup_rounds", "keygen_reps", "setup_reps", "step_reps",
                 "min_time_s", "group_size", "payload_bits", "identity_depth", "mode")
    )
    lines += [f"Configuration: {details}.", ""]
    # Worker threads as each binary reports them: basic in its configuration,
    # Epoque as `threads` and RBE's host as `rayon_threads` in the environment.
    environment = report.get("environment", {})
    threads = environment.get("threads", environment.get("rayon_threads", config.get("threads")))
    if threads is not None:
        cpus = environment.get("logical_cpus")
        lines += [f"Worker threads: {threads}"
                  + ("" if cpus is None else f" (logical CPUs available: {cpus})") + ".", ""]
    ordinary = report.get("ordinary", {})
    if ordinary:
        lines += [
            "Ordinary algorithms, mean ± standard deviation (Enc and Dec over the rounds "
            "of all trades):",
            "",
            "| " + " | ".join(label for _, label in ORDINARY) + " |",
            "|" + "---:|" * len(ORDINARY),
            "| " + " | ".join(cell(ordinary.get(key)) for key, _ in ORDINARY) + " |",
            "",
        ]
        extras = [k for k in ordinary if k not in dict(ORDINARY)]
        if extras:
            lines += [
                "Also measured: "
                + ", ".join(f"{k} {cell(ordinary[k])}" for k in extras)
                + ".",
                "",
            ]
    rows = report.get("anamorphic", [])
    if rows:
        lines += [
            "Anamorphic algorithms per trade: \\|S\\| recipients of an L-bit message, d = "
            "\\|S\\| L equations (RBE: one ring equation per recipient), mean ± standard "
            "deviation. Sweep rows vary d, check rows repeat a d with another split. Enc is "
            "ordinary encryption in the same rounds as aEnc. aKGen generates the keys of the "
            "group's recipients only, aDec is one fixed recipient:",
            "",
            "| \\|S\\| | L | d | Role | " + " | ".join(label for _, label in ANAMORPHIC)
            + " | aKGen per recipient |",
            "|---:|---:|---:|---|" + "---:|" * (len(ANAMORPHIC) + 1),
        ]
        for row in rows:
            length, count, bits, equations = group_label(row)
            per_user = row["keygen"]["mean_ms"] / count
            lines.append(
                f"| {count} | {bits} | {equations} | {row.get('role', length)} | "
                + " | ".join(cell(row.get(key)) for key, _ in ANAMORPHIC)
                + f" | {readable(per_user)} |"
            )
        lines += [
            "",
            "Each recipient's keys are independent, so aKGen for N users takes N times the "
            "cost per recipient.",
            "",
        ]
        lines += step_table(report)
    measured = sizes(report)
    if measured:
        known = [(key, label) for key, label in SIZES if key in measured]
        known += [(key, key) for key in measured if key not in dict(SIZES)]
        lines += [
            "Sizes (both ciphertexts as serialized by the implementation):",
            "",
            "| Item | Size |",
            "|---|---:|",
        ]
        lines += [f"| {label} | {readable_bytes(measured[key])} |" for key, label in known]
        lines.append("")
    return lines


def tex_ms(ms):
    """A duration in ms for a LaTeX cell, three significant digits."""
    return f"{ms:.3g}" if abs(ms) >= 1 else f"{ms:.3f}"


def tex_us(ms):
    """A duration in ms as signed microseconds for a LaTeX cell."""
    us = ms * 1e3
    text = f"{abs(us):.3g}" if abs(us) >= 1 else f"{abs(us):.2f}"
    return ("-" if us < 0 else "+") + text


def latex(report):
    """The anamorphic rows of one result file as a LaTeX table (booktabs)."""
    rows = report.get("anamorphic", [])
    if not rows or not all("step" in row for row in rows):
        return []
    rank = report.get("config", {}).get("rank")
    name = NAMES.get(report["scheme"], report["scheme"]) + ("" if rank is None else f", rank {rank}")
    lines = [
        f"% {name}: Enc and aEnc in the same rounds (mean $\\pm$ sd, ms), the step overhead",
        "% timed alone (median and distribution-free 95% interval, microseconds), aDec (mean, ms).",
        "% Check rows (other splits of a d of the sweep) are marked with an asterisk.",
        "\\begin{tabular}{rrrrrrr}",
        "\\toprule",
        "$d$ & $|S|$ & $L$ & Enc (ms) & aEnc (ms) & Overhead ($\\mu$s) & aDec (ms) \\\\",
        "\\midrule",
    ]
    for row in rows:
        o, o_low, o_high = step_overhead(row)
        interval = (f" $[{tex_us(o_low)}, {tex_us(o_high)}]$"
                    if math.isfinite(o_low) and math.isfinite(o_high) else "")
        enc, aenc = row["ordinary_encryption"], row["encryption"]
        star = "$^*$" if row.get("role") == "check" else ""
        lines.append(
            f"{row.get('constraints', '')}{star} & {row['recipients']} & {row['bits']} "
            f"& ${tex_ms(enc['mean_ms'])} \\pm {tex_ms(enc['sd_ms'])}$ "
            f"& ${tex_ms(aenc['mean_ms'])} \\pm {tex_ms(aenc['sd_ms'])}$ "
            f"& ${tex_us(o)}${interval} & {tex_ms(row['decryption']['mean_ms'])} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", ""]
    return lines


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("results", nargs="+", help="JSON files written by the experiments binaries")
    parser.add_argument("--output", required=True,
                        help="output prefix (writes PREFIX.csv, PREFIX-sizes.csv, PREFIX.md, PREFIX.tex)")
    args = parser.parse_args()
    reports = []
    for path in args.results:
        try:
            with open(path, encoding="utf-8") as f:
                report = json.load(f)
        except (OSError, json.JSONDecodeError) as error:
            print(f"skipping {path}: {error}", file=sys.stderr)
            continue
        # The attack simulation's attack.json, for example, holds no benchmark.
        if isinstance(report, dict) and "scheme" in report:
            reports.append(report)
        else:
            print(f"skipping {path}: not a result file of an experiments binary",
                  file=sys.stderr)
    with open(args.output + ".csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["scheme", "recipients", "bits", "equations", "length", "role",
                         "operation", "runs", "mean_ms", "sd_ms",
                         "median_ms", "min_ms", "max_ms"])
        for report in reports:
            writer.writerows(csv_rows(report))
    with open(args.output + "-sizes.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["scheme", "item", "bytes"])
        for report in reports:
            for key, count in sizes(report).items():
                writer.writerow([scheme_label(report), key, count])
    lines = ["# Experiment summary", ""]
    tex = ["% Generated by experiments/summarize.py; needs \\usepackage{booktabs}.", ""]
    for report in reports:
        lines += markdown(report)
        tex += latex(report)
        rows = report.get("anamorphic", [])
        if any("encryption_overhead" not in row and "step" not in row for row in rows):
            print(f"warning: {scheme_label(report)} timed Enc and aEnc in separate blocks "
                  "(a file from before the paired rounds); its aEnc - Enc includes drift",
                  file=sys.stderr)
        elif any("step" not in row for row in rows):
            print(f"warning: {scheme_label(report)} predates timing the differing step alone; "
                  "its overhead is the one timed inside aEnc", file=sys.stderr)
    with open(args.output + ".md", "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    with open(args.output + ".tex", "w", encoding="utf-8") as f:
        f.write("\n".join(tex) + "\n")
    print(f"wrote {args.output}.csv, {args.output}-sizes.csv, {args.output}.md and "
          f"{args.output}.tex", file=sys.stderr)


if __name__ == "__main__":
    main()
