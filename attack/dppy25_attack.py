#!/usr/bin/env python3
"""Simulation of the distinguishing attack of Section 3 (Proposition 3.1).

    python3 attack/dppy25_attack.py [--trials 1000] [--seed 2025] [--plot PREFIX]

The adversary of Proposition 3.1 holds the ordinary secret key, makes one encryption
query and outputs 1 ("ordinary") if the decryption residual z = c - <key, p> lies
within circular distance E of 0 or Delta = floor(q/2). The script runs the mode
experiment --trials times per mode, each time with fresh keys and one query, for

* DPPY25, the Dual-Regev anamorphic broadcast mode of Do, Persiano, Phan and Yung
  (EUROCRYPT 2025: Dual Regev in their Section 2.3, AnonMCBE in 3.4, the broadcast
  mode in 4.2), as Section 3 recalls it: ordinary keys e ~ D_{Z^m,r}, u = A e (their
  A is the transpose of ours); broadcast keys dk_i = (1 || x_i) with
  A_Broad x_i = -u_Broad; an anamorphic ciphertext is act = y + e_Broad with y uniform
  subject to <dk_i - r_j, y> = 0 for every channel j and recipient i in aset_j, where
  r_j = (am_j, 0, ..., 0), and e_Broad rounded Gaussian; recipient i outputs 0 if
  <dk_i, act> is closer to 0 than to y_1, which it reads as act_1 = y_1 + e_1;
* Construction 1 of the paper on the Dual-Regev scheme of Definition 2.4 (binary k,
  b = A k): s uniform subject to (a_1 - bhat_{i,a})^T s = Delta am_a for every
  recipient i in S and bit a, then ordinary encryption with this s,

and prints, for E = floor(q/5) (as in Proposition 3.1) and for E = 8 standard
deviations of the decryption error (which the adversary can compute from its key),
Pr[D = 1] in both modes with Wilson 95% intervals, the advantage with a Newcombe 95%
interval, and the bound 1 - (4E + 2)/q. It checks every ciphertext (ordinary
decryption of ordinary ciphertexts, hidden decryption by every recipient), counts the
event Bad (d in W) of the proof, and compares the residual distributions of the two
modes with a Kolmogorov-Smirnov test. --plot PREFIX draws the distribution of the
residual's distance from {0, Delta} (needs matplotlib), --json PATH writes the counts.

The parameters are toy-sized and give no security, but meet DPPY25's conditions
q >= 5 r (m + 1), m >= 2 n log2 q and alpha <= 1/(r sqrt(m + 1) omega(sqrt(log n))),
which the report checks; the attack does not depend on them beyond the bound E on the
decryption error. test_dppy25_attack.py tests the implementation. A_Broad is generated
with planted keys:
x_i ~ D_{Z^m,r} first, then A_Broad uniform subject to A_Broad x_i = -u_Broad for all i,
which above the smoothing parameter is distributed as trapdoor sampling. The attack
uses none of A_Broad, u_Broad and x_i. Standard library only (random.Random: a
simulation, not cryptographic randomness).
"""
import argparse
import bisect
import json
import math
import random
import sys
import time
from operator import mul


# --- Arithmetic modulo q -----------------------------------------------------------


def is_prime(q):
    """Deterministic Miller-Rabin for q < 3.3e24."""
    if q < 2:
        return False
    for p in (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41):
        if q % p == 0:
            return q == p
    d, s = q - 1, 0
    while d % 2 == 0:
        d, s = d // 2, s + 1
    for a in (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41):
        x = pow(a, d, q)
        if x in (1, q - 1):
            continue
        for _ in range(s - 1):
            x = x * x % q
            if x == q - 1:
                break
        else:
            return False
    return True


def circular(z, q):
    """The distance of z from 0 modulo q."""
    z %= q
    return min(z, q - z)


def dot(a, b, q):
    return sum(map(mul, a, b)) % q


class AffineSampler:
    """Uniform solutions x in Z_q^N of M x = t for a k x N matrix M of full row rank:
    the reduced row echelon form R = T M fixes the k pivot coordinates, the other
    N - k coordinates are uniform."""

    def __init__(self, rows, q):
        self.q = q
        k, width = len(rows), len(rows[0])
        r = [[x % q for x in row] for row in rows]
        t = [[int(i == j) for j in range(k)] for i in range(k)]
        pivots, row = [], 0
        for col in range(width):
            if row == k:
                break
            found = next((i for i in range(row, k) if r[i][col]), None)
            if found is None:
                continue
            r[row], r[found], t[row], t[found] = r[found], r[row], t[found], t[row]
            inverse = pow(r[row][col], -1, q)
            r[row] = [x * inverse % q for x in r[row]]
            t[row] = [x * inverse % q for x in t[row]]
            for i in range(k):
                if i != row and r[i][col]:
                    f = r[i][col]
                    r[i] = [(a - f * b) % q for a, b in zip(r[i], r[row])]
                    t[i] = [(a - f * b) % q for a, b in zip(t[i], t[row])]
            pivots.append(col)
            row += 1
        if row < k:
            raise ValueError("rank deficient system")
        self.width, self.rows, self.transform, self.pivots = width, r, t, pivots
        chosen = set(pivots)
        self.free = [c for c in range(width) if c not in chosen]
        self.free_rows = [[row[c] for c in self.free] for row in r]

    def sample(self, target, rng):
        q = self.q
        x = [0] * self.width
        free = [rng.randrange(q) for _ in self.free]
        for c, v in zip(self.free, free):
            x[c] = v
        for p, t, f in zip(self.pivots, self.transform, self.free_rows):
            x[p] = (dot(t, target, q) - sum(map(mul, f, free))) % q
        return x

    def spans(self, v):
        """Whether v lies in the row space of M."""
        q = self.q
        rest = [x % q for x in v]
        for p, row in zip(self.pivots, self.rows):
            f = rest[p]
            if f:
                rest = [(a - f * b) % q for a, b in zip(rest, row)]
        return not any(rest)


class Gaussians:
    """Rounded continuous Gaussians (the errors) and the discrete Gaussian D_{Z,r}
    with mass proportional to exp(-pi x^2 / r^2) (DPPY25's keys), by inversion of a
    table over [-12 r, 12 r]."""

    def __init__(self, rng, sigma, r):
        self.rng, self.sigma = rng, sigma
        tail = math.ceil(12 * r)
        self.values = list(range(-tail, tail + 1))
        weights = [math.exp(-math.pi * x * x / (r * r)) for x in self.values]
        total, running, self.cumulative = sum(weights), 0.0, []
        for w in weights:
            running += w
            self.cumulative.append(running / total)

    def errors(self, count):
        return [round(self.rng.gauss(0.0, self.sigma)) for _ in range(count)]

    def discrete(self, count):
        u = self.rng.random
        values, cumulative = self.values, self.cumulative
        return [values[min(bisect.bisect_right(cumulative, u()), len(values) - 1)]
                for _ in range(count)]


def times_matrix(a, s, q):
    """A^T s for A given by its n rows of length m."""
    acc = [0] * len(a[0])
    for row, si in zip(a, s):
        if si:
            acc = [x + si * y for x, y in zip(acc, row)]
    return [x % q for x in acc]


def matrix_times(a, v, q):
    """A v for A given by its rows."""
    return [dot(row, v, q) for row in a]


def dual_regev_encrypt(a, public, s, bit, noise, q):
    """(p, c) = (A^T s + x, <public, s> + x' + Delta bit), the errors rounded
    Gaussians: Definition 2.4, and the ordinary encryption of DPPY25 with u for b."""
    x = noise.errors(len(a[0]) + 1)
    p = [(v + e) % q for v, e in zip(times_matrix(a, s, q), x)]
    c = (dot(public, s, q) + x[-1] + (q // 2) * bit) % q
    return p, c


def residual(key, ciphertext, q):
    """z = c - <key, p> mod q, the residual before rounding."""
    p, c = ciphertext
    return (c - dot(key, p, q)) % q


def nearest_bit(z, q):
    """Round(z): the bit whose encoding 0 or Delta is nearest to z."""
    return int(circular(z - q // 2, q) < circular(z, q))


def distance(z, q):
    """The circular distance of z from the nearer of 0 and Delta."""
    return min(circular(z, q), circular(z - q // 2, q))


# --- The two schemes ---------------------------------------------------------------


def dppy25_setup(a, rng, noise, groups, q):
    """Ordinary GPV keys (e, u = A e) and the broadcast keys dk_i = (1 || x_i) with
    A_Broad x_i = -u_Broad (planted), for sum(groups) anamorphic users."""
    n, m = len(a), len(a[0])
    e = noise.discrete(m)
    u = matrix_times(a, e, q)
    users = sum(groups)
    xs = [noise.discrete(m) for _ in range(users)]
    u_broad = [rng.randrange(q) for _ in range(n)]
    planted = AffineSampler(xs, q)
    a_broad = [planted.sample([-v % q] * users, rng) for v in u_broad]
    for x in xs:
        assert matrix_times(a_broad, x, q) == [-v % q for v in u_broad]
    return {"e": e, "u": u, "xs": xs, "a_broad": a_broad, "u_broad": u_broad}


def dppy25_query(keys, groups, bits, rng, noise, q):
    """DPPY25's anamorphic encryption of one hidden bit per channel: y uniform subject
    to <dk_i - r_j, y> = 0 for every channel j and recipient i in aset_j, plus noise.
    Returns the ciphertext, read as (p, c) with p the first m coordinates, and the
    constraint sampler (whose row space is W)."""
    rows, first = [], 0
    for size, bit in zip(groups, bits):
        for i in range(first, first + size):
            rows.append([(1 - bit) % q] + keys["xs"][i])
        first += size
    constraints = AffineSampler(rows, q)
    y = constraints.sample([0] * len(rows), rng)
    act = [(v + e) % q for v, e in zip(y, noise.errors(len(y)))]
    return (act[:-1], act[-1]), constraints


def dppy25_decode(x, ciphertext, q):
    """<dk_i, act> is close to 0 for hidden bit 0 and to act[1] for hidden bit 1."""
    p, c = ciphertext
    act = p + [c]
    w = (act[0] + dot(x, act[1:], q)) % q
    return int(circular(w - act[0], q) < circular(w, q))


def dppy25_conditions(n, m, q, sigma, r):
    """DPPY25's parameter conditions for its Dual-Regev scheme ([7], Section 2.3), with
    omega(sqrt(log n)) instantiated by sqrt(log2 n), and the smoothing condition under
    which the planted keys follow trapdoor sampling. Rows: (condition, holds, value,
    bound)."""
    smoothing = math.sqrt(math.log(2 * m * (1 + 2 ** 40)) / math.pi)
    alpha = sigma / q
    return [
        ("q >= 5 r (m + 1)", q >= 5 * r * (m + 1), q, 5 * r * (m + 1)),
        ("m >= 2 n log2 q", m >= 2 * n * math.log2(q), m, 2 * n * math.log2(q)),
        ("alpha = sigma/q <= 1/(r sqrt(m + 1) sqrt(log2 n))",
         alpha <= 1 / (r * math.sqrt(m + 1) * math.sqrt(math.log2(n))), alpha,
         1 / (r * math.sqrt(m + 1) * math.sqrt(math.log2(n)))),
        ("r >= smoothing of Lambda^perp(A), about q^(n/m) sqrt(ln(2m(1 + 2^40))/pi)",
         r >= q ** (n / m) * smoothing, r, q ** (n / m) * smoothing),
    ]


def construction1_setup(a, rng, recipients, bits, q):
    """Definition 2.4 (binary k, b = A k) and the anamorphic keys of Construction 1:
    binary khat_{i,a} with bhat_{i,a} = A khat_{i,a}, and the group's sampler for s,
    one equation (a_1 - bhat_{i,a})^T s = Delta am_a per recipient and bit."""
    m = len(a[0])
    k = [rng.getrandbits(1) for _ in range(m)]
    b = matrix_times(a, k, q)
    a1 = [row[0] for row in a]
    khat = [[[rng.getrandbits(1) for _ in range(m)] for _ in range(bits)]
            for _ in range(recipients)]
    h = [[(x - y) % q for x, y in zip(a1, matrix_times(a, key, q))]
         for keys in khat for key in keys]
    return {"k": k, "b": b, "khat": khat, "sampler": AffineSampler(h, q)}


def construction1_decode(key, ciphertext, q):
    """Round(r[1] - khat^T r)."""
    p, _ = ciphertext
    return nearest_bit((p[0] - dot(key, p, q)) % q, q)


# --- Statistics --------------------------------------------------------------------


def wilson(k, n, z=1.959964):
    """Wilson score interval for a binomial proportion k/n."""
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return max(0.0, centre - half), min(1.0, centre + half)


def newcombe(k1, n1, k2, n2):
    """Newcombe's hybrid score interval for p1 - p2."""
    p1, p2 = k1 / n1, k2 / n2
    (l1, u1), (l2, u2) = wilson(k1, n1), wilson(k2, n2)
    d = p1 - p2
    return d - math.hypot(p1 - l1, u2 - p2), d + math.hypot(u1 - p1, p2 - l2)


def kolmogorov_smirnov(xs, ys):
    """Two-sample statistic and its asymptotic p-value."""
    xs, ys = sorted(xs), sorted(ys)
    i = j = 0
    statistic = 0.0
    while i < len(xs) and j < len(ys):
        v = min(xs[i], ys[j])
        while i < len(xs) and xs[i] == v:
            i += 1
        while j < len(ys) and ys[j] == v:
            j += 1
        statistic = max(statistic, abs(i / len(xs) - j / len(ys)))
    ne = len(xs) * len(ys) / (len(xs) + len(ys))
    lam = (math.sqrt(ne) + 0.12 + 0.11 / math.sqrt(ne)) * statistic
    if lam < 1e-3:
        return statistic, 1.0
    p = 2 * sum((-1) ** (k - 1) * math.exp(-2 * k * k * lam * lam) for k in range(1, 101))
    return statistic, min(1.0, max(0.0, p))


# --- The experiment ----------------------------------------------------------------


def run(args):
    q, n, m = args.q, args.n, args.m
    rng = random.Random(args.seed)
    noise = Gaussians(rng, args.sigma, args.r)
    variance = args.sigma ** 2 + 1 / 12  # of a rounded Gaussian, closely
    loose = q // 5
    results = {
        "dppy25": {"ordinary": [], "anamorphic": []},
        "construction1": {"ordinary": [], "anamorphic": []},
    }
    tight = {"dppy25": [], "construction1": []}
    checks = {"dppy25_ordinary": 0, "dppy25_hidden": 0, "dppy25_hidden_total": 0,
              "c1_ordinary": 0, "c1_hidden": 0, "c1_hidden_total": 0, "bad": 0}
    start = time.time()
    for trial in range(args.trials):
        # One uniform A per trial serves both schemes' ordinary public keys.
        a = [[rng.randrange(q) for _ in range(m)] for _ in range(n)]

        # DPPY25: the ordinary oracle encrypts a fresh uniform bit, the anamorphic one
        # answers the query of one hidden bit per channel.
        keys = dppy25_setup(a, rng, noise, args.groups, q)
        e = keys["e"]
        tight["dppy25"].append(math.ceil(8 * math.sqrt(variance * (1 + sum(v * v for v in e)))))
        bit = rng.getrandbits(1)
        s = [rng.randrange(q) for _ in range(n)]
        ordinary = dual_regev_encrypt(a, keys["u"], s, bit, noise, q)
        z = residual(e, ordinary, q)
        checks["dppy25_ordinary"] += nearest_bit(z, q) == bit
        results["dppy25"]["ordinary"].append(z)
        hidden = [rng.getrandbits(1) for _ in args.groups]
        anamorphic, constraints = dppy25_query(keys, args.groups, hidden, rng, noise, q)
        results["dppy25"]["anamorphic"].append(residual(e, anamorphic, q))
        first = 0
        for size, value in zip(args.groups, hidden):
            for i in range(first, first + size):
                checks["dppy25_hidden"] += dppy25_decode(keys["xs"][i], anamorphic, q) == value
                checks["dppy25_hidden_total"] += 1
            first += size
        # Bad: d = (-e, 1) in W, the span of the constraint vectors.
        checks["bad"] += constraints.spans([-v for v in e] + [1])

        # Construction 1: Enc(m) against aEnc(m, am) for the group S.
        c1 = construction1_setup(a, rng, args.recipients, args.bits, q)
        k = c1["k"]
        tight["construction1"].append(math.ceil(8 * math.sqrt(variance * (1 + sum(k)))))
        message = rng.getrandbits(1)
        s = [rng.randrange(q) for _ in range(n)]
        ordinary = dual_regev_encrypt(a, c1["b"], s, message, noise, q)
        z = residual(k, ordinary, q)
        checks["c1_ordinary"] += nearest_bit(z, q) == message
        results["construction1"]["ordinary"].append(z)
        am = [rng.getrandbits(1) for _ in range(args.bits)]
        target = [(q // 2) * v for _ in range(args.recipients) for v in am]
        s = c1["sampler"].sample(target, rng)
        anamorphic = dual_regev_encrypt(a, c1["b"], s, message, noise, q)
        z = residual(k, anamorphic, q)
        checks["c1_ordinary"] += nearest_bit(z, q) == message
        results["construction1"]["anamorphic"].append(z)
        for keys_of_user in c1["khat"]:
            for key, value in zip(keys_of_user, am):
                checks["c1_hidden"] += construction1_decode(key, anamorphic, q) == value
                checks["c1_hidden_total"] += 1
        if args.progress and (trial + 1) % max(1, args.trials // 10) == 0:
            print(f"  {trial + 1}/{args.trials} experiments, {time.time() - start:.0f} s",
                  file=sys.stderr)
    return results, tight, checks, loose, time.time() - start


def report(args, results, tight, checks, loose, seconds):
    q, trials = args.q, args.trials
    d = args.recipients * args.bits
    print("Proposition 3.1: the residual test against DPPY25 and against Construction 1")
    print(f"  n = {args.n}, m = {args.m}, q = {q} (prime), Delta = floor(q/2); errors rounded "
          f"Gaussians of standard deviation {args.sigma}; DPPY25 keys e, x_i ~ D_(Z^m, r), "
          f"r = {args.r}")
    print(f"  DPPY25: {sum(args.groups)} anamorphic users, one query with "
          f"{len(args.groups)} channels of {', '.join(map(str, args.groups))} recipients, "
          "one hidden bit each")
    print(f"  Construction 1: |S| = {args.recipients} recipients, L = {args.bits} hidden bits "
          f"(d = {d} equations)")
    print(f"  {trials} experiments per mode, each with fresh keys and one query "
          f"(seed {args.seed}, {seconds:.0f} s)")
    print("  DPPY25's parameter conditions:")
    conditions = dppy25_conditions(args.n, args.m, q, args.sigma, args.r)
    for name, holds, value, bound in conditions:
        print(f"    {name}: {value:.4g} against {bound:.4g}, {'holds' if holds else 'FAILS'}")
    print()
    print("Functionality (every ciphertext checked):")
    print(f"  DPPY25: ordinary decryption {checks['dppy25_ordinary']}/{trials}, hidden bits "
          f"{checks['dppy25_hidden']}/{checks['dppy25_hidden_total']} decoded by their recipients")
    print(f"  Construction 1: ordinary decryption {checks['c1_ordinary']}/{2 * trials} (both "
          f"modes), hidden bits {checks['c1_hidden']}/{checks['c1_hidden_total']} decoded")
    print(f"  Bad (d in W) in DPPY25's queries: {checks['bad']} of {trials}")
    print()
    print("Residual test D, output 1 = 'ordinary' if z is within E of 0 or Delta "
          "(Wilson / Newcombe 95% intervals):")
    rows = []
    summary = {"dppy25 conditions": [{"condition": name, "holds": holds, "value": value,
                                      "bound": bound}
                                     for name, holds, value, bound in conditions]}
    for scheme, label in (("dppy25", "DPPY25"), ("construction1", "Construction 1")):
        ordinary = results[scheme]["ordinary"]
        anamorphic = results[scheme]["anamorphic"]
        for name, thresholds in (("floor(q/5)", [loose] * trials),
                                 ("8 sd of the error", tight[scheme])):
            hits_o = sum(distance(z, q) <= t for z, t in zip(ordinary, thresholds))
            hits_a = sum(distance(z, q) <= t for z, t in zip(anamorphic, thresholds))
            low, high = newcombe(hits_o, trials, hits_a, trials)
            # Proposition 3.1's lower bound on the advantage holds for DPPY25 only.
            bound = (1 - sum((4 * t + 2) / q for t in thresholds) / trials
                     if scheme == "dppy25" else None)
            rows.append((f"{label}, E = {name}", hits_o, hits_a, low, high, bound,
                         sum(thresholds) / trials))
            summary[f"{scheme}, E = {name}"] = {
                "E_mean": sum(thresholds) / trials, "ordinary_accepts": hits_o,
                "anamorphic_accepts": hits_a, "trials": trials,
                "advantage": (hits_o - hits_a) / trials, "advantage_95": [low, high]}
            if bound is not None:
                summary[f"{scheme}, E = {name}"]["bound_1_minus_4E_plus_2_over_q"] = bound
    width = max(len(r[0]) for r in rows)
    print(f"  {'':{width}}  {'E':>11}  {'Pr[D=1 | ord.]':>21}  {'Pr[D=1 | anam.]':>21}  "
          f"{'advantage':>23}  {'1-(4E+2)/q':>10}")
    for label, hits_o, hits_a, low, high, bound, e in rows:
        po, pa = hits_o / trials, hits_a / trials
        (ol, oh), (al, ah) = wilson(hits_o, trials), wilson(hits_a, trials)
        print(f"  {label:{width}}  {e:11.4g}  {po:.3f} [{ol:.3f}, {oh:.3f}]  "
              f"{pa:.3f} [{al:.3f}, {ah:.3f}]  {po - pa:+.3f} [{low:+.3f}, {high:+.3f}]  "
              + (f"{bound:10.6f}" if bound is not None else f"{'-':>10}"))
    print()
    print("Kolmogorov-Smirnov tests, ordinary against anamorphic mode: of the residual z "
          "(its whole distribution),")
    print("and of its distance from {0, Delta} (the statistic is the best advantage of "
          "a threshold test over all E):")
    for scheme, label in (("dppy25", "DPPY25"), ("construction1", "Construction 1")):
        ordinary, anamorphic = results[scheme]["ordinary"], results[scheme]["anamorphic"]
        whole = kolmogorov_smirnov(ordinary, anamorphic)
        near = kolmogorov_smirnov([distance(z, q) for z in ordinary],
                                  [distance(z, q) for z in anamorphic])
        summary[f"{scheme}, Kolmogorov-Smirnov"] = {
            "residual": {"statistic": whole[0], "p_value": whole[1]},
            "distance": {"statistic": near[0], "p_value": near[1]}}
        print(f"  {label}: residual D = {whole[0]:.3f} (p = {whole[1]:.3g}), "
              f"distance D = {near[0]:.3f} (p = {near[1]:.3g})")
    return summary


def plot(args, results, tight, loose, prefix):
    """The distribution of the distance of z from {0, Delta}, as a fraction of q, in
    both modes, with the two thresholds E/q (E = floor(q/5), and E = 8 standard
    deviations of the decryption error, the median over the keys): Pr[D = 1] is each
    curve's value at E/q."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    q = args.q
    # Categorical slots 1 and 2 of the validated palette, text in ink tokens.
    ordinary_color, anamorphic_color = "#2a78d6", "#eb6834"
    ink, secondary, grid, axis = "#0b0b0b", "#52514e", "#e1e0d9", "#c3c2b7"
    plt.rcParams.update({
        "font.family": "serif", "font.serif": ["STIXGeneral", "Times New Roman", "DejaVu Serif"],
        "mathtext.fontset": "stix", "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8,
        "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7, "pdf.fonttype": 42,
        "axes.edgecolor": axis, "axes.linewidth": 0.6, "axes.labelcolor": ink,
        "text.color": ink, "xtick.color": axis, "ytick.color": axis,
        "xtick.labelcolor": secondary, "ytick.labelcolor": secondary,
        "grid.color": grid, "grid.linewidth": 0.5,
    })
    fig, axes = plt.subplots(1, 2, figsize=(3.5, 1.75), sharey=True,
                             gridspec_kw={"wspace": 0.08})
    floor = 0.5 / q
    for ax, (scheme, title) in zip(axes, (("dppy25", "DPPY25"),
                                          ("construction1", "Construction 1"))):
        for mode, color, label, style in (("ordinary", ordinary_color, "ordinary", "-"),
                                          ("anamorphic", anamorphic_color, "anamorphic",
                                           (0, (3, 1.5)))):
            xs = sorted(max(distance(z, q) / q, floor) for z in results[scheme][mode])
            ys = [(i + 1) / len(xs) for i in range(len(xs))]
            ax.step([floor] + xs + [0.25], [0.0] + ys + [1.0], where="post", color=color,
                    lw=1.2, ls=style, label=label, zorder=3)
        median_tight = sorted(tight[scheme])[len(tight[scheme]) // 2]
        # The thresholds, labelled above the frame where no curve runs.
        for value, text in ((loose / q, r"$\lfloor q/5\rfloor$"),
                            (median_tight / q, r"$8\sigma_\eta$")):
            ax.axvline(value, color=secondary, lw=0.6, ls=(0, (1, 1.5)), zorder=2)
            ax.annotate(text, (value, 1.0), xycoords=("data", "axes fraction"),
                        xytext=(0, 2), textcoords="offset points", ha="center", va="bottom",
                        fontsize=6.5, color=secondary)
        ax.set_xscale("log")
        ax.set_xlim(floor, 0.25)
        ax.set_ylim(0, 1.02)
        ax.set_title(title, pad=11)
        ax.grid(True, axis="y")
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    axes[0].set_ylabel("Fraction with distance $\\leq x$", color=secondary)
    fig.supxlabel("Distance $x$ of $z$ from $\\{0,\\Delta\\}$, as a fraction of $q$",
                  fontsize=8, color=secondary, y=0.02)
    # The middle of the DPPY25 panel is empty: the curves run at 0 and 1 there.
    axes[0].legend(loc="center", bbox_to_anchor=(0.61, 0.5), frameon=False,
                   handlelength=1.8, borderaxespad=0.2)
    fig.subplots_adjust(left=0.13, right=0.98, top=0.83, bottom=0.25)
    for ext, options in (("pdf", {}), ("png", {"dpi": 300})):
        fig.savefig(f"{prefix}.{ext}", **options)
    plt.close(fig)
    print(f"wrote {prefix}.pdf and {prefix}.png", file=sys.stderr)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--trials", type=int, default=1000, help="experiments per mode")
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument("--n", type=int, default=16)
    parser.add_argument("--m", type=int, default=1024)
    parser.add_argument("--q", type=int, default=1073741789, help="an odd prime (2^30 - 35)")
    parser.add_argument("--sigma", type=float, default=3.2,
                        help="standard deviation of the rounded Gaussian errors")
    parser.add_argument("--r", type=float, default=6.0,
                        help="parameter of D_(Z,r) for DPPY25's keys e and x_i")
    parser.add_argument("--groups", type=lambda v: [int(x) for x in v.split(",")],
                        default=[3, 3], help="recipients per DPPY25 channel")
    parser.add_argument("--recipients", type=int, default=3, help="|S| for Construction 1")
    parser.add_argument("--bits", type=int, default=2, help="L for Construction 1")
    parser.add_argument("--plot", metavar="PREFIX", help="write PREFIX.pdf and PREFIX.png")
    parser.add_argument("--json", metavar="PATH", help="write the counts as JSON")
    parser.add_argument("--progress", action="store_true", help="progress on stderr")
    args = parser.parse_args()
    if not (is_prime(args.q) and args.q > 2):
        parser.error("q must be an odd prime")
    if args.trials < 1 or args.n < 2 or args.m <= args.n or min(args.groups) < 1:
        parser.error("need trials >= 1, n >= 2, m > n and recipients >= 1 per channel")
    if args.recipients * args.bits > args.n - 1 or sum(args.groups) > args.m:
        parser.error("need |S| L < n for Construction 1 and at most m DPPY25 users")
    failing = [name for name, holds, _, _ in
               dppy25_conditions(args.n, args.m, args.q, args.sigma, args.r) if not holds]
    if failing:
        print("warning: outside DPPY25's parameter conditions: " + "; ".join(failing),
              file=sys.stderr)
    results, tight, checks, loose, seconds = run(args)
    summary = report(args, results, tight, checks, loose, seconds)
    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump({"parameters": {k: v for k, v in vars(args).items()
                                      if k not in ("plot", "json", "progress")},
                       "checks": checks, "tests": summary}, f, indent=2)
        print(f"wrote {args.json}", file=sys.stderr)
    if args.plot:
        try:
            plot(args, results, tight, loose, args.plot)
        except ImportError:
            print("matplotlib is not installed: no plot", file=sys.stderr)
    failed = (checks["dppy25_ordinary"] < args.trials
              or checks["c1_ordinary"] < 2 * args.trials
              or checks["c1_hidden"] < checks["c1_hidden_total"])
    if failed:
        print("warning: some ordinary or Construction 1 decryptions failed", file=sys.stderr)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
