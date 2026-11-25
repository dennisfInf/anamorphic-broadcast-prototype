#!/usr/bin/env python3
"""Paper figures of one run_all.sh result folder (IEEE two-column layout).

    python3 experiments/plot.py RUN_DIR [--output PREFIX]

Writes two PDF files (fonts embedded as TrueType, Times-like STIX) and a PNG preview
of each, both for a figure* (7.16 in wide) with one panel per host scheme:

* PREFIX-costs.pdf: the mean time of the ordinary algorithms (dashed) and of the
  anamorphic ones (markers) against the number of equations d = |S| L that a group's
  message imposes, on one logarithmic time axis shared by the panels. For basic and
  Epoque every sender-side cost depends on d only and aDec on L only: filled markers,
  joined, are the sweep (|S| = 2 recipients, L = d/2 bits), hollow ones the checks
  (other splits of the largest d). For RBE, d is the number of recipients, one ring
  equation each, on a host of rank 7 + |S|.
* PREFIX-overhead.pdf: the overhead of aEnc over Enc in microseconds, against d as
  above, on a linear axis per panel with a zero line. It is the step in which aEnc
  differs from Enc, timed alone in paired iterations (the constrained secret s
  against the uniform s of Enc for basic and Epoque, the constrained root r_0 against
  the Gaussian r_0 of Enc for RBE, where it can be negative): the median of the paired
  differences with a distribution-free 95% interval, since a rare interrupt during the
  short step would dominate a mean. The panel titles give the median Enc. Files from
  before the step was timed alone fall back to the overhead timed inside aEnc.

It also prints the worker threads every result file records, and warns if they
differ between the schemes, and prints every overhead next to the paired difference
of the totals aEnc - Enc of the same rounds (flagging rows whose 99.9% interval for the
latter excludes the former). Needs matplotlib (the bench image has it).
"""
import argparse
import json
import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

# Validated categorical slots 1 to 3 (all pairs, light mode, white surface): one hue
# per algorithm, with a marker shape as the second channel for print and CVD.
OPERATIONS = [
    ("setup", "Setup", "#1baf7a", "^"),
    ("encryption", "Encryption", "#2a78d6", "o"),
    ("decryption", "Decryption", "#eb6834", "s"),
]
INK, SECONDARY, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, AXIS, SURFACE = "#e1e0d9", "#c3c2b7", "#ffffff"
SCHEMES = [
    ("basic", "Basic (Dual-Regev)"),
    ("rbe", "RBE"),
    ("epoque", "Epoque (IBE)"),
]


def style():
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["STIXGeneral", "Times New Roman", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 8,
        "axes.titlesize": 8,
        "axes.labelsize": 8,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 7,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "axes.edgecolor": AXIS,
        "axes.linewidth": 0.6,
        "axes.labelcolor": INK,
        "axes.titlecolor": INK,
        "text.color": INK,
        "xtick.color": AXIS,
        "ytick.color": AXIS,
        "xtick.labelcolor": SECONDARY,
        "ytick.labelcolor": SECONDARY,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.minor.width": 0.4,
        "ytick.minor.width": 0.4,
        "grid.color": GRID,
        "grid.linewidth": 0.5,
        "grid.linestyle": "-",
        "savefig.facecolor": SURFACE,
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
    })


def load(run):
    """The result files of a run folder, by scheme (RBE has one file per rank)."""
    reports = {}
    for path in sorted(Path(run).glob("*.json")):
        try:
            report = json.loads(path.read_text())
        except (OSError, ValueError):
            continue
        if isinstance(report, dict) and "scheme" in report:
            reports.setdefault(report["scheme"], []).append(report)
    return reports


def threads(report):
    environment = report.get("environment", {})
    config = report.get("config", {})
    return environment.get("threads", environment.get("rayon_threads", config.get("threads")))


def points(reports):
    """One point per anamorphic row: |S|, L, d, whether it belongs to the sweep, the
    anamorphic and the ordinary means and standard deviations (ms) and the run counts,
    the ordinary ones taken from the same file (for RBE, the same host rank), the
    median Enc of the row's rounds, and the overhead series: the paired differences
    of the step timed alone (`step`), or for older files the overhead timed inside
    aEnc (`insitu`), with aEnc minus Enc of the same rounds (`difference`)."""
    out = []
    for report in reports:
        ordinary = report.get("ordinary", {})
        for row in report.get("anamorphic", []):
            # Files from before the sweep have no role: their rows count as the sweep.
            point = {"recipients": row["recipients"], "bits": row["bits"],
                     "d": row.get("constraints") or row["recipients"] * row["bits"],
                     "sweep": row.get("role", "sweep") == "sweep"}
            for key, _, _, _ in OPERATIONS:
                for prefix, source in (("a", row), ("o", ordinary)):
                    stats = source.get(key)
                    if stats:
                        point[prefix + key] = (stats["mean_ms"], stats["sd_ms"], stats["runs"])
            if "ordinary_encryption" in row:
                point["enc"] = row["ordinary_encryption"]["median_ms"]
            if "step" in row:
                point["step"] = row["step"]["overhead"]["samples_ms"]
            elif "encryption_overhead" in row:
                point["insitu"] = row["encryption_overhead"]["samples_ms"]
            if "encryption_difference" in row:
                point["difference"] = row["encryption_difference"]["samples_ms"]
            out.append(point)
    return sorted(out, key=lambda p: (p["d"], not p["sweep"], p["recipients"]))


def median_interval(xs, level=0.95):
    """The median of xs and a distribution-free interval for it: the order statistics
    s_(j) and s_(n+1-j) with the largest j for which P(Binomial(n, 1/2) < j) is at most
    (1 - level) / 2, so the coverage is at least `level`. Infinite below 6 samples."""
    s = sorted(xs)
    n = len(s)
    median = (s[(n - 1) // 2] + s[n // 2]) / 2
    # The binomial tail in logarithms: 2^-n underflows for long series.
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


def readable(ms):
    """A duration in ms as a short string with a fitting unit."""
    if abs(ms) < 1:
        return f"{ms * 1000:.3g} µs"
    if abs(ms) < 1000:
        return f"{ms:.3g} ms"
    return f"{ms / 1000:.3g} s"


def x_axis(ax, scheme, rows):
    """d on a logarithmic axis of base 2 for basic and Epoque, linear for RBE."""
    ds = sorted({p["d"] for p in rows})
    if scheme == "rbe":
        ax.set_xticks(ds)
        ax.set_xlim(ds[0] - 0.5, ds[-1] + 0.5)
        ax.set_xlabel("Recipients |S| = ring equations d (host rank 7 + |S|)",
                      color=SECONDARY, labelpad=2)
        return
    ax.set_xscale("log", base=2)
    # At most six labels: every other power of two on long sweeps.
    step = 1 if len(ds) <= 6 else 2
    shown = ds[::-step][::-1]
    ax.set_xticks(shown)
    ax.set_xticklabels([str(d) for d in shown])
    ax.minorticks_off()
    ax.set_xlim(ds[0] / 1.35, ds[-1] * 1.35)
    ax.set_xlabel("Equations d = |S| L (sweep: |S| = 2)", color=SECONDARY, labelpad=2)


def frame(ax):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.set_axisbelow(True)


def time_label(value, _):
    if value >= 1000:
        return f"{value / 1000:g} s"
    if value < 1:
        return f"{value * 1000:g} µs"
    return f"{value:g} ms"


def costs(reports, prefix):
    present = [(s, title) for s, title in SCHEMES if s in reports]
    fig, axes = plt.subplots(1, len(present), figsize=(7.16, 2.45), sharey=True,
                             gridspec_kw={"wspace": 0.08})
    axes = [axes] if len(present) == 1 else list(axes)
    values = []
    for ax, (scheme, title) in zip(axes, present):
        rows = points(reports[scheme])
        ds = sorted({p["d"] for p in rows})
        for key, _, color, marker in OPERATIONS:
            # Ordinary scheme: one mean per host (per rank for RBE), dashed.
            ordinary = {}
            for p in rows:
                if "o" + key in p:
                    ordinary[p["d"]] = p["o" + key][0]
            if ordinary:
                xs = sorted(ordinary)
                ys = [ordinary[x] for x in xs]
                if len(set(ys)) == 1 and scheme != "rbe":
                    xs = [ds[0], ds[-1]]
                    ys = [ys[0], ys[0]]
                ax.plot(xs, ys, color=color, lw=1.0, ls=(0, (4, 2)), zorder=2)
                values += ys
            # Anamorphic: the sweep filled and joined, the checks hollow.
            for sweep in (True, False):
                chosen = [p for p in rows if p["sweep"] == sweep and "a" + key in p]
                xs = [p["d"] for p in chosen]
                ys = [p["a" + key][0] for p in chosen]
                values += ys
                if sweep:
                    ax.plot(xs, ys, color=color, lw=1.1, marker=marker, ms=4.2,
                            mfc=color, mec=SURFACE, mew=0.7, zorder=4)
                else:
                    ax.plot(xs, ys, ls="none", marker=marker, ms=3.6, mfc=SURFACE,
                            mec=color, mew=0.9, zorder=3)
        x_axis(ax, scheme, rows)
        ax.set_title(title, pad=4)
        ax.grid(True, axis="y", which="major")
        frame(ax)
    axes[0].set_yscale("log")
    low = 10 ** math.floor(math.log10(min(values)))
    high = 10 ** math.ceil(math.log10(max(values)))
    axes[0].set_ylim(low, high)
    axes[0].yaxis.set_major_locator(matplotlib.ticker.LogLocator(base=10, numticks=16))
    axes[0].yaxis.set_major_formatter(plt.FuncFormatter(time_label))
    axes[0].set_ylabel("Mean time", color=SECONDARY)
    for ax in axes:
        ax.yaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    keys = [Line2D([], [], color=color, lw=1.1, marker=marker, ms=4.2, mfc=color, mec=SURFACE,
                   mew=0.7, label=label) for _, label, color, marker in OPERATIONS]
    keys += [
        Line2D([], [], color=MUTED, lw=1.0, ls=(0, (4, 2)), label="ordinary scheme"),
        Line2D([], [], color=MUTED, lw=1.1, marker="o", ms=4.2, mfc=MUTED, mec=SURFACE, mew=0.7,
               label="anamorphic, sweep"),
        Line2D([], [], ls="none", marker="o", ms=3.6, mfc=SURFACE, mec=MUTED, mew=0.9,
               label="anamorphic, other split of d"),
    ]
    fig.legend(handles=keys, loc="upper center", ncol=6, frameon=False,
               bbox_to_anchor=(0.5, 1.0), handlelength=2.2, columnspacing=1.4,
               handletextpad=0.5)
    fig.subplots_adjust(left=0.088, right=0.995, top=0.80, bottom=0.17)
    save(fig, prefix + "-costs")


def overhead_of(p):
    """(median, low, high) of a point's overhead in ms, and whether it is the step
    timed alone (else the overhead timed inside aEnc, from an older file)."""
    if "step" in p:
        return median_interval(p["step"]), True
    return median_interval(p["insitu"]), False


def interval(low, high):
    """[low, high] as durations, or a note where the runs are too few for one."""
    if not (math.isfinite(low) and math.isfinite(high)):
        return "[no 95% interval below 6 runs]"
    return f"[{readable(low)}, {readable(high)}]"


def report_line(title, p):
    """One printed line per row: the overhead and the check of the totals."""
    (o, o_low, o_high), alone = overhead_of(p)
    line = (f"{title}, |S| = {p['recipients']}, L = {p['bits']} (d = {p['d']}, "
            f"{'sweep' if p['sweep'] else 'check'}): "
            f"{'step overhead' if alone else 'overhead inside aEnc'} {readable(o)} "
            f"{interval(o_low, o_high)}")
    if "enc" in p:
        line += f", {100 * o / p['enc']:+.3g}% of Enc ({readable(p['enc'])})"
    if "difference" in p:
        d, d_low, d_high = median_interval(p["difference"])
        line += f"; aEnc - Enc of the totals {readable(d)} {interval(d_low, d_high)}"
        # Beyond chance only: a 95% interval misses in one row of twenty.
        _, far_low, far_high = median_interval(p["difference"], level=0.999)
        if not far_low <= o <= far_high:
            line += " (even its 99.9% interval excludes the overhead: check these rounds)"
    return line + " (medians, 95% intervals)"


def overhead(reports, prefix):
    present = [(s, title) for s, title in SCHEMES if s in reports]
    fig, axes = plt.subplots(1, len(present), figsize=(7.16, 2.2),
                             gridspec_kw={"wspace": 0.3})
    axes = [axes] if len(present) == 1 else list(axes)
    color = OPERATIONS[1][2]
    missing, older, open_ended = [], [], False
    for ax, (scheme, title) in zip(axes, present):
        rows = [p for p in points(reports[scheme]) if "step" in p or "insitu" in p]
        missing += [f"{title}, |S| = {p['recipients']}, L = {p['bits']}"
                    for p in points(reports[scheme]) if p not in rows]
        if not rows:
            ax.set_visible(False)
            continue
        entries = []
        for p in rows:
            (o, low, high), alone = overhead_of(p)
            if not alone:
                older.append(f"{title}, |S| = {p['recipients']}, L = {p['bits']}")
            print(report_line(title, p))
            entries.append((p["d"], 1e3 * o, 1e3 * low, 1e3 * high, p["sweep"]))
        for x, y, low, high, sweep in entries:
            if math.isfinite(low) and math.isfinite(high):
                ax.errorbar(x, y, yerr=[[y - low], [high - y]], fmt="none", ecolor=color,
                            elinewidth=0.7, capsize=0, zorder=2)
            else:
                open_ended = True
        sweep = sorted((x, y) for x, y, _, _, s in entries if s)
        ax.plot([x for x, _ in sweep], [y for _, y in sweep], color=color, lw=1.1, marker="o",
                ms=4.2, mfc=color, mec=SURFACE, mew=0.7, zorder=4)
        checks = [(x, y) for x, y, _, _, s in entries if not s]
        ax.plot([x for x, _ in checks], [y for _, y in checks], ls="none", marker="o", ms=3.6,
                mfc=SURFACE, mec=color, mew=0.9, zorder=3)
        ax.axhline(0, color=SECONDARY, lw=0.6, zorder=1)
        spread = [v for _, y, low, high, _ in entries for v in (y, low, high) if math.isfinite(v)]
        top, bottom = max(spread + [0.0]), min(spread + [0.0])
        margin = 0.08 * (top - bottom or 1.0)
        ax.set_ylim(bottom - margin, top + margin)
        ax.yaxis.set_major_formatter(
            plt.FuncFormatter(lambda v, _: f"{v:+g}".replace("-", "−") if v else "0"))
        x_axis(ax, scheme, rows)
        encs = sorted(p["enc"] for p in rows if "enc" in p)
        enc = f", Enc {readable(encs[len(encs) // 2])}" if encs else ""
        ax.set_title(title + enc, pad=4)
        ax.grid(True, axis="y")
        frame(ax)
    axes[0].set_ylabel("aEnc overhead (µs)", color=SECONDARY)
    if open_ended:
        print("note: rows with fewer than 6 runs have no 95% interval; their points are "
              "drawn without one", file=sys.stderr)
    if older:
        print("warning: no step timed alone for " + "; ".join(older) + "; these points are "
              "the overhead timed inside aEnc (a file from before the step benchmark)",
              file=sys.stderr)
    if missing:
        print("warning: no overhead series for " + "; ".join(missing) + " (Enc and aEnc "
              "were timed in separate blocks: re-run the experiments binaries)",
              file=sys.stderr)
    keys = [
        Line2D([], [], color=color, lw=1.1, marker="o", ms=4.2, mfc=color, mec=SURFACE, mew=0.7,
               label="sweep"),
        Line2D([], [], ls="none", marker="o", ms=3.6, mfc=SURFACE, mec=color, mew=0.9,
               label="other split of d"),
        Line2D([], [], color=color, lw=0.7, label="95% interval of the median"),
    ]
    fig.legend(handles=keys, loc="upper center", ncol=3, frameon=False,
               bbox_to_anchor=(0.5, 1.0), handlelength=2.0, columnspacing=1.4)
    fig.subplots_adjust(left=0.07, right=0.995, top=0.78, bottom=0.19)
    save(fig, prefix + "-overhead")


def save(fig, name):
    fig.savefig(name + ".pdf")
    fig.savefig(name + ".png", dpi=300)
    plt.close(fig)
    print(f"wrote {name}.pdf and {name}.png")


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("run", help="folder written by run_all.sh")
    parser.add_argument("--output", default=None,
                        help="file name prefix (default: RUN/figure)")
    args = parser.parse_args()
    reports = load(args.run)
    if not any(s in reports for s, _ in SCHEMES):
        sys.exit(f"no basic, rbe or epoque result files in {args.run}")
    counts = {}
    for scheme, title in SCHEMES:
        for report in reports.get(scheme, []):
            counts.setdefault(title, set()).add(threads(report))
    for title, values in counts.items():
        print(f"{title}: worker threads {', '.join(str(v) for v in sorted(values, key=str))}")
    if len({v for values in counts.values() for v in values}) > 1:
        print("warning: the schemes ran with different thread counts", file=sys.stderr)
    style()
    prefix = args.output or str(Path(args.run) / "figure")
    costs(reports, prefix)
    overhead(reports, prefix)


if __name__ == "__main__":
    main()
