#!/usr/bin/env python3
"""Statistical tests for the output of examples/noise_stats.rs.

Every test compares the counts with the ideal rounded normal law
round(N(0, s^2)), whose probabilities are computed here independently of the
Rust table (scipy's erf and erfc), or tests a symmetry or independence that the
law and the ChaCha12 streams imply. For a correct sampler every p-value is
uniform on [0, 1]. The script reports all of them and fails if any is below
1e-6.

    python3 tools/noise_stats.py stats.json
"""

import json
import math
import sys

import numpy as np
from scipy import special, stats


def two_sided_normal(z):
    return 2.0 * stats.norm.sf(abs(z))


def poisson_two_sided(observed, mean):
    low = stats.poisson.cdf(observed, mean)
    high = stats.poisson.sf(observed - 1, mean)
    return min(1.0, 2.0 * min(low, high))


def main(path):
    with open(path) as f:
        d = json.load(f)
    s = math.sqrt(d["variance"]) if d["variance"] is not None else d["sd"]
    width = f"variance {d['variance']}" if d["variance"] is not None else f"sd {d['sd']!r}"
    pos = np.array(d["pos"], dtype=np.float64)
    neg = np.array(d["neg"], dtype=np.float64)
    mag = pos + neg
    n = mag.sum()
    bound = d["bound"]
    k = np.arange(bound + 1, dtype=np.float64)
    c = 1.0 / (s * math.sqrt(2.0))
    # Ideal law of |e|: P(0) = erf(c/2), P(k) = erfc((k - 1/2) c) - erfc((k + 1/2) c).
    p = np.empty(bound + 1)
    p[0] = special.erf(0.5 * c)
    p[1:] = special.erfc((k[1:] - 0.5) * c) - special.erfc((k[1:] + 0.5) * c)
    tail_above = lambda K: special.erfc((K + 0.5) * c)  # P(|e| > K)

    e2 = float(np.sum(k**2 * p))
    e4 = float(np.sum(k**4 * p))
    e6 = float(np.sum(k**6 * p))
    e8 = float(np.sum(k**8 * p))
    var2 = e4 - e2 * e2
    results = []

    def record(name, statistic, pvalue):
        results.append((name, statistic, pvalue))

    print(f"{width}: s = {s!r}, {int(n)} errors from {d['ciphertexts']} seeds "
          f"({d['seconds']:.0f} s, {d['threads']} threads)")

    # 1. Chi-square and G-test on every value of |e| with at least 50 expected,
    #    the rest merged into one tail bin.
    expected = n * p
    small = np.nonzero(expected < 50)[0]
    cut = int(small[0]) if small.size else bound + 1
    obs = np.append(mag[:cut], mag[cut:].sum())
    exp = np.append(expected[:cut], n * tail_above(cut - 1))
    chi2 = float(np.sum((obs - exp) ** 2 / exp))
    dof = len(obs) - 1
    record(f"chi-square on |e| = 0..{cut - 1} plus tail ({dof} dof)", chi2, stats.chi2.sf(chi2, dof))
    nz = obs > 0
    g = float(2.0 * np.sum(obs[nz] * np.log(obs[nz] / exp[nz])))
    record(f"G-test on the same bins ({dof} dof)", g, stats.chi2.sf(g, dof))

    # 2. Coarse chi-square on 200 bins of about equal probability.
    cdf = np.cumsum(p)
    edges = np.searchsorted(cdf, np.arange(1, 200) / 200.0)
    edges = np.unique(np.concatenate(([0], edges + 1, [bound + 1])))
    obs_c = np.add.reduceat(mag, edges[:-1])
    exp_c = n * np.add.reduceat(p, edges[:-1])
    chi2_c = float(np.sum((obs_c - exp_c) ** 2 / exp_c))
    record(f"chi-square on {len(obs_c)} equiprobable bins of |e|", chi2_c, stats.chi2.sf(chi2_c, len(obs_c) - 1))

    # 3. Kolmogorov-Smirnov distance for |e| (asymptotic, conservative for a discrete law).
    exact_cdf = 1.0 - special.erfc((k + 0.5) * c)
    ks = float(np.max(np.abs(np.cumsum(mag) / n - exact_cdf)))
    record("Kolmogorov-Smirnov sqrt(N) D on |e|", ks * math.sqrt(n), stats.kstwobign.sf(ks * math.sqrt(n)))

    # 4. Tails beyond t standard deviations (Poisson).
    for t in (3.0, 4.0, 5.0, 5.5, 6.0, 6.5):
        K = int(math.floor(t * s))
        observed = int(mag[K + 1:].sum())
        mean = n * tail_above(K)
        record(f"tail |e| > {K} ({t} s): {observed} observed, {mean:.1f} expected", observed,
               poisson_two_sided(observed, mean))

    # 5. Sign: overall balance and symmetry for every magnitude (pos+neg >= 20, rest merged).
    plus, minus = pos[1:].sum(), neg[1:].sum()
    record(f"sign balance: {int(plus)} positive, {int(minus)} negative",
           (plus - minus) / math.sqrt(plus + minus), two_sided_normal((plus - minus) / math.sqrt(plus + minus)))
    tot = pos[1:] + neg[1:]
    big = tot >= 20
    dpos = np.append(pos[1:][big], pos[1:][~big].sum())
    dneg = np.append(neg[1:][big], neg[1:][~big].sum())
    keep = dpos + dneg > 0
    sym = float(np.sum((dpos[keep] - dneg[keep]) ** 2 / (dpos[keep] + dneg[keep])))
    record(f"sign symmetry per magnitude ({int(keep.sum())} dof)", sym, stats.chi2.sf(sym, int(keep.sum())))

    # 6. Moments against the exact moments of the law.
    m = [float(int(x)) for x in d["moments"]]
    record("mean", m[1] / math.sqrt(n * e2), two_sided_normal(m[1] / math.sqrt(n * e2)))
    z = (m[2] / n - e2) / math.sqrt(var2 / n)
    record(f"variance {m[2] / n:.4f} (exact {e2:.4f}, s^2 + 1/12 = {s * s + 1 / 12:.4f})", z, two_sided_normal(z))
    z = m[3] / math.sqrt(n * e6)
    record("third moment", z, two_sided_normal(z))
    z = (m[4] / n - e4) / math.sqrt((e8 - e4 * e4) / n)
    record("fourth moment", z, two_sided_normal(z))

    # 7. Correlations: within streams (lags), across streams, across seeds, header vs scalar.
    def correlation(name, entry):
        total, pairs = float(int(entry["sum"])), entry["pairs"]
        z = total / (math.sqrt(pairs) * e2)
        record(f"{name}: r = {total / (pairs * e2):+.2e} over {pairs} pairs", z, two_sided_normal(z))

    for lag, entry in enumerate(d["lag"], start=1):
        correlation(f"lag-{lag} correlation within a stream", entry)
    correlation("last error of a stream vs first of the next", d["boundary"])
    correlation("same position in consecutive streams", d["aligned"])
    correlation("first error of consecutive seeds", d["seeds"])
    correlation("header error 0 vs the scalar error", d["header_scalar"])

    # 8. Independence of consecutive errors: signs (-, 0, +) and quartile classes of |e|.
    for name, table in (("signs", d["sign_pairs"]), ("quartile classes", d["class_pairs"])):
        table = np.array(table, dtype=np.float64)
        # Small widths can leave a quartile class empty, which carries no information.
        table = table[table.sum(axis=1) > 0][:, table.sum(axis=0) > 0]
        chi2_t, p_t, dof_t, _ = stats.chi2_contingency(table, correction=False)
        record(f"consecutive {name}, contingency ({dof_t} dof)", chi2_t, p_t)
    # The marginal of the sign classes is the law itself: P(e = 0) = p[0].
    signs = np.array(d["sign_pairs"], dtype=np.float64)
    zeros = signs[:, 1].sum()
    total = signs.sum()
    z = (zeros - total * p[0]) / math.sqrt(total * p[0] * (1 - p[0]))
    record(f"P(e = 0) = {zeros / total:.6e} (exact {p[0]:.6e})", z, two_sided_normal(z))

    # 9. Position inside a stream: count, zeros, mean and mean square per position.
    position = d["position"]
    z0 = z1 = z2 = 0.0
    for count, zero_count, total_e, total_e2 in position:
        total_e, total_e2 = float(int(total_e)), float(int(total_e2))
        z0 += ((zero_count - count * p[0]) / math.sqrt(count * p[0] * (1 - p[0]))) ** 2
        z1 += (total_e / math.sqrt(count * e2)) ** 2
        z2 += ((total_e2 / count - e2) / math.sqrt(var2 / count)) ** 2
    positions = len(position)
    record(f"zeros by position in a stream ({positions} dof)", z0, stats.chi2.sf(z0, positions))
    record(f"mean by position in a stream ({positions} dof)", z1, stats.chi2.sf(z1, positions))
    record(f"mean square by position in a stream ({positions} dof)", z2, stats.chi2.sf(z2, positions))

    # 10. The scalar error (its own stream).
    sc = d["scalar"]
    count, total_e, total_e2 = sc["count"], float(int(sc["sum"])), float(int(sc["sum2"]))
    z = (total_e2 / count - e2) / math.sqrt(var2 / count)
    record(f"scalar error variance ({count} samples)", z, two_sided_normal(z))
    z = total_e / math.sqrt(count * e2)
    record("scalar error mean", z, two_sided_normal(z))

    width_name = max(len(r[0]) for r in results)
    for name, statistic, pvalue in results:
        print(f"  {name:<{width_name}}  stat {statistic:>14.4f}  p = {pvalue:.4f}")
    worst = min(r[2] for r in results)
    print(f"{len(results)} tests, smallest p-value {worst:.4g}")
    if worst < 1e-6:
        print("FAIL: a p-value is below 1e-6")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
