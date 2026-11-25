#!/usr/bin/env python3
"""Tests of the DPPY25 implementation in dppy25_attack.py (standard library unittest).

    python3 attack/test_dppy25_attack.py        (about 11 s)
    python3 -m unittest discover -s attack -v

They check the implementation against the construction of Do, Persiano, Phan and Yung
(EUROCRYPT 2025): Dual Regev in their Section 2.3, the anonymous multi-channel
broadcast AnonMCBE in 3.4 and the anamorphic broadcast mode in 4.2, as Section 3 of
our paper recalls them. Bottom up:

* the linear algebra and the samplers: solutions of M x = t satisfy it and are uniform
  over all solutions, rank deficiency is refused, row-space membership is exact,
  D_{Z,r} has mass proportional to exp(-pi x^2 / r^2), the errors are rounded Gaussians;
* the ordinary scheme: u = A e, p = A^T s + x, c = u^T s + x' + Delta b, and the
  residual c - e^T p = Delta b + x' - e^T x;
* the broadcast keys: dk_i = (1 || x_i) is a short vector with
  u_Broad + A_Broad x_i = 0, and A_Broad is uniform;
* anamorphic encryption: every constraint <dk_i - r_j, y> = 0 holds exactly, y is
  uniform over all solutions, and <dk_i, act> = am_j y_1 + <dk_i, e_Broad>;
* anamorphic decryption: every recipient decodes its channel's bit;
* what the attack reads: z = c - e^T p = <d, y + e_Broad> with d = (-e, 1), uniform
  when d is outside the span W of the constraints, and only noise on the event Bad
  (d in W) that the proof of Proposition 3.1 excludes;
* the experiment: D accepts every ordinary ciphertext and an anamorphic one with
  probability (4E + 2)/q, while both modes of Construction 1 look alike;
* the default parameters meet DPPY25's conditions, and the command line works.

Statistical tests use fixed seeds and reject at p = 1e-4, so they are deterministic.
"""
import argparse
import contextlib
import io
import json
import math
import os
import random
import subprocess
import sys
import tempfile
import unittest
from collections import Counter
from operator import mul

HERE = os.path.dirname(os.path.abspath(__file__))
SCRIPT = os.path.join(HERE, "dppy25_attack.py")
sys.path.insert(0, HERE)
import dppy25_attack as dppy  # noqa: E402

Q = 1073741789  # the default modulus, 2^30 - 35
N, M, SIGMA, R = 16, 1024, 3.2, 6.0  # the other defaults
GROUPS = [3, 3]  # two channels of three recipients


def chi_square(observed, expected):
    return sum((o - e) ** 2 / e for o, e in zip(observed, expected))


def chi_square_limit(df, z=3.719):
    """The upper 1e-4 point of chi-square with df degrees of freedom (Wilson-Hilferty)."""
    a = 2 / (9 * df)
    return df * (1 - a + z * math.sqrt(a)) ** 3


def uniform_matrix(rng, rows, columns, q):
    return [[rng.randrange(q) for _ in range(columns)] for _ in range(rows)]


def norm(v):
    return math.sqrt(sum(x * x for x in v))


class ZeroNoise:
    """Errors that are all zero, so that a ciphertext shows the randomness y itself."""

    def errors(self, count):
        return [0] * count


class RecordingNoise:
    """The script's samplers, keeping every error vector they hand out."""

    def __init__(self, inner):
        self.inner, self.drawn = inner, []

    def errors(self, count):
        self.drawn.append(self.inner.errors(count))
        return self.drawn[-1]

    def discrete(self, count):
        return self.inner.discrete(count)


def channel_of(groups, bits):
    """The hidden bit of every user, in the order of the keys."""
    return [bit for size, bit in zip(groups, bits) for _ in range(size)]


class ArithmeticTest(unittest.TestCase):
    def test_primality(self):
        self.assertEqual(Q, 2 ** 30 - 35)
        for prime in (2, 3, 65521, Q, 2 ** 31 - 1, 2 ** 61 - 1):
            self.assertTrue(dppy.is_prime(prime), prime)
        # 3215031751 is a strong pseudoprime to the bases 2, 3, 5 and 7.
        for composite in (1, 561, 65535, 3215031751, 3 * Q, (2 ** 31 - 1) * Q):
            self.assertFalse(dppy.is_prime(composite), composite)

    def test_rounding_to_the_encodings(self):
        q = 101  # Delta = 50
        self.assertEqual((dppy.circular(3, q), dppy.circular(98, q)), (3, 3))
        self.assertEqual((dppy.distance(52, q), dppy.distance(99, q)), (2, 2))
        self.assertEqual([dppy.nearest_bit(z, q) for z in (0, 24, 26, 50, 75, 77, 100)],
                         [0, 0, 1, 1, 1, 0, 0])


class AffineSamplerTest(unittest.TestCase):
    def test_solutions_satisfy_the_system(self):
        rng = random.Random(1)
        for q, k, width in ((Q, 6, M + 1), (65521, 12, N), (Q, 1, 2)):
            rows = uniform_matrix(rng, k, width, q)
            sampler = dppy.AffineSampler(rows, q)
            for _ in range(10):
                target = [rng.randrange(q) for _ in range(k)]
                x = sampler.sample(target, rng)
                self.assertEqual([dppy.dot(row, x, q) for row in rows], target)

    def test_solutions_are_uniform(self):
        # x_1 + 2 x_2 + 3 x_3 = 4 over Z_5 has 25 solutions, all equally likely.
        q, rows, draws = 5, [[1, 2, 3]], 25 * 400
        sampler, rng = dppy.AffineSampler(rows, q), random.Random(2)
        counts = Counter(tuple(sampler.sample([4], rng)) for _ in range(draws))
        self.assertEqual(len(counts), 25)
        self.assertTrue(all(dppy.dot(rows[0], x, q) == 4 for x in counts))
        self.assertLess(chi_square(counts.values(), [draws / 25] * 25), chi_square_limit(24))

    def test_rank_deficiency_and_row_space(self):
        q = 65521
        with self.assertRaises(ValueError):
            dppy.AffineSampler([[1, 2, 3], [2, 4, 6]], q)
        sampler = dppy.AffineSampler([[1, 2, 3, 4], [0, 1, 5, 7]], q)
        self.assertTrue(sampler.spans([3, 7, 14, 19]))  # 3 (1, 2, 3, 4) + (0, 1, 5, 7)
        self.assertTrue(sampler.spans([-1, -2, -3, -4]))
        self.assertFalse(sampler.spans([0, 0, 0, 1]))


class SamplerTest(unittest.TestCase):
    def test_discrete_gaussian(self):
        draws = 200_000
        counts = Counter(dppy.Gaussians(random.Random(3), SIGMA, R).discrete(draws))
        weight = {x: math.exp(-math.pi * x * x / (R * R)) for x in range(-100, 101)}
        total = sum(weight.values())
        support = range(-8, 9)
        observed = [counts[x] for x in support]
        expected = [draws * weight[x] / total for x in support]
        observed.append(draws - sum(observed))
        expected.append(draws - sum(expected))
        self.assertLess(chi_square(observed, expected), chi_square_limit(len(observed) - 1))
        mean = sum(x * c for x, c in counts.items()) / draws
        variance = sum((x - mean) ** 2 * c for x, c in counts.items()) / (draws - 1)
        self.assertLess(abs(mean), 4 * math.sqrt(R * R / (2 * math.pi) / draws))
        # Above the smoothing parameter of Z the variance of D_{Z,r} is r^2 / (2 pi).
        self.assertAlmostEqual(variance, R * R / (2 * math.pi), delta=0.05 * R * R / (2 * math.pi))
        self.assertLessEqual(max(map(abs, counts)), math.ceil(12 * R))

    def test_rounded_gaussian_errors(self):
        draws = 100_000
        xs = dppy.Gaussians(random.Random(4), SIGMA, R).errors(draws)
        self.assertTrue(all(isinstance(x, int) for x in xs))
        mean = sum(xs) / draws
        variance = sum((x - mean) ** 2 for x in xs) / (draws - 1)
        self.assertLess(abs(mean), 4 * SIGMA / math.sqrt(draws))
        self.assertAlmostEqual(variance, SIGMA ** 2 + 1 / 12, delta=0.03 * SIGMA ** 2)


class DPPY25Fixture(unittest.TestCase):
    """A DPPY25 setup at the default parameters: A, (e, u) and the broadcast keys."""

    def setUp(self):
        self.rng = random.Random(5)
        self.noise = RecordingNoise(dppy.Gaussians(self.rng, SIGMA, R))
        self.a = uniform_matrix(self.rng, N, M, Q)
        self.keys = dppy.dppy25_setup(self.a, self.rng, self.noise, GROUPS, Q)
        self.e = self.keys["e"]


class OrdinarySchemeTest(DPPY25Fixture):
    def test_public_key_is_the_syndrome_of_the_secret_key(self):
        self.assertEqual(self.keys["u"], dppy.matrix_times(self.a, self.e, Q))
        self.assertLessEqual(max(map(abs, self.e)), math.ceil(12 * R))
        expected = math.sqrt(M * R * R / (2 * math.pi))
        self.assertLess(abs(norm(self.e) - expected), 0.1 * expected)

    def test_encryption_and_decryption(self):
        for bit in (0, 1, 1, 0, 1):
            s = [self.rng.randrange(Q) for _ in range(N)]
            p, c = dppy.dual_regev_encrypt(self.a, self.keys["u"], s, bit, self.noise, Q)
            x = self.noise.drawn[-1]
            self.assertEqual(p, [(v + w) % Q for v, w in zip(dppy.times_matrix(self.a, s, Q), x)])
            self.assertEqual(c, (dppy.dot(self.keys["u"], s, Q) + x[-1] + (Q // 2) * bit) % Q)
            eta = x[-1] - sum(map(mul, self.e, x))
            z = dppy.residual(self.e, (p, c), Q)
            self.assertEqual(z, ((Q // 2) * bit + eta) % Q)
            self.assertLess(abs(eta), 8 * SIGMA * math.sqrt(1 + norm(self.e) ** 2))
            self.assertEqual(dppy.nearest_bit(z, Q), bit)


class BroadcastKeysTest(DPPY25Fixture):
    def test_keys_are_short_vectors_in_the_kernel(self):
        # AnonMCBE's KeyGen: x_i ~ D_{Lambda^perp_{-u}(A_Broad), r}, dk_i = (1 || x_i),
        # so (u_Broad | A_Broad) dk_i = 0.
        self.assertEqual(len(self.keys["xs"]), sum(GROUPS))
        self.assertEqual(len(self.keys["a_broad"]), N)
        expected = math.sqrt(M * R * R / (2 * math.pi))
        for x in self.keys["xs"]:
            self.assertEqual(len(x), M)
            image = dppy.matrix_times(self.keys["a_broad"], x, Q)
            self.assertEqual([(u + v) % Q for u, v in zip(self.keys["u_broad"], image)], [0] * N)
            self.assertLess(abs(norm(x) - expected), 0.1 * expected)
        self.assertEqual(len(set(map(tuple, self.keys["xs"]))), len(self.keys["xs"]))

    def test_broadcast_matrix_and_vector_are_uniform(self):
        # The planted keys leave A_Broad uniform subject to its equations, so its entries,
        # like those of u_Broad, are uniform over Z_q.
        q, rng = 101, random.Random(6)
        noise = dppy.Gaussians(rng, 1.0, R)
        entries = Counter()
        for _ in range(60):
            keys = dppy.dppy25_setup(uniform_matrix(rng, 4, 40, q), rng, noise, [2, 1], q)
            for row in keys["a_broad"]:
                entries.update(row)
        drawn = sum(entries.values())
        self.assertEqual(drawn, 60 * 4 * 40)
        self.assertLess(chi_square([entries[v] for v in range(q)], [drawn / q] * q),
                        chi_square_limit(q - 1))


class AnamorphicEncryptionTest(DPPY25Fixture):
    def test_every_constraint_holds_exactly(self):
        # y orthogonal to dk_i - r_j with r_j = (am_j, 0, ..., 0), so <dk_i, y> = am_j y_1.
        for bits in ([0, 0], [0, 1], [1, 0], [1, 1]):
            (p, c), _ = dppy.dppy25_query(self.keys, GROUPS, bits, self.rng, ZeroNoise(), Q)
            y = p + [c]
            self.assertEqual(len(y), M + 1)
            for x, bit in zip(self.keys["xs"], channel_of(GROUPS, bits)):
                dk, r = [1] + x, [bit] + [0] * M
                self.assertEqual(dppy.dot([a - b for a, b in zip(dk, r)], y, Q), 0)
                self.assertEqual(dppy.dot(dk, y, Q), bit * y[0] % Q)

    def test_ciphertext_is_randomness_plus_broadcast_noise(self):
        noise = RecordingNoise(dppy.Gaussians(self.rng, SIGMA, R))
        bits = [1, 0]
        (p, c), constraints = dppy.dppy25_query(self.keys, GROUPS, bits, self.rng, noise, Q)
        self.assertEqual(len(noise.drawn), 1, "act = y + e_Broad draws e_Broad once")
        act, e_broad = p + [c], noise.drawn[-1]
        self.assertEqual(len(e_broad), M + 1)
        self.assertLessEqual(max(map(abs, e_broad)), 8 * SIGMA)
        # e_Broad is rounded Gaussian: variance sigma^2 + 1/12, within 20% (4.5 sd).
        variance = sum(x * x for x in e_broad) / len(e_broad)
        self.assertAlmostEqual(variance, SIGMA ** 2 + 1 / 12, delta=0.2 * SIGMA ** 2)
        y = [(a - b) % Q for a, b in zip(act, e_broad)]
        for x, bit in zip(self.keys["xs"], channel_of(GROUPS, bits)):
            dk = [1] + x
            # Correctness of AnonMCBE: <dk_i, act> = am_j y_1 + <dk_i, e_Broad>.
            small = sum(map(mul, dk, e_broad))
            self.assertEqual(dppy.dot(dk, act, Q), (bit * y[0] + small) % Q)
            self.assertLess(abs(small), 8 * SIGMA * norm(dk))
        self.assertEqual(len(constraints.pivots), sum(GROUPS))

    def test_randomness_is_uniform_on_the_solutions(self):
        # One recipient per channel over Z_5 with m = 3: the two constraints
        # (1 - am_j, x_i) leave 25 vectors y, all equally likely.
        q, keys, draws = 5, {"xs": [[1, 2, 0], [0, 1, 3]]}, 25 * 400
        rng = random.Random(7)
        counts = Counter()
        for _ in range(draws):
            (p, c), _ = dppy.dppy25_query(keys, [1, 1], [1, 0], rng, ZeroNoise(), q)
            counts[tuple(p + [c])] += 1
        constraints = [[0, 1, 2, 0], [1, 0, 1, 3]]
        self.assertEqual(len(counts), 25)
        for y in counts:
            self.assertEqual([dppy.dot(w, y, q) for w in constraints], [0, 0])
        self.assertLess(chi_square(counts.values(), [draws / 25] * 25), chi_square_limit(24))


class AnamorphicDecryptionTest(DPPY25Fixture):
    def test_every_recipient_decodes_its_channel(self):
        wrong = total = 0
        for _ in range(50):
            bits = [self.rng.getrandbits(1) for _ in GROUPS]
            ct, _ = dppy.dppy25_query(self.keys, GROUPS, bits, self.rng, self.noise, Q)
            for x, bit in zip(self.keys["xs"], channel_of(GROUPS, bits)):
                wrong += dppy.dppy25_decode(x, ct, Q) != bit
                total += 1
        self.assertEqual((wrong, total), (0, 300))

    def test_decoding_rule(self):
        # 0 if <dk_i, act> is closer to 0 than to act_1, else 1. With dk = (1, 3) and
        # act = (40, a), <dk, act> = 40 + 3 a mod 101.
        q, x = 101, [3]
        self.assertEqual(dppy.dppy25_decode(x, ([40], 88), q), 0)  # 304 = 1 mod 101
        self.assertEqual(dppy.dppy25_decode(x, ([40], 1), q), 1)  # 43, next to 40
        self.assertEqual(dppy.dppy25_decode(x, ([40], 0), q), 1)  # 40 itself


class ResidualTest(DPPY25Fixture):
    def test_residual_is_the_inner_product_with_d(self):
        (p, c), constraints = dppy.dppy25_query(self.keys, GROUPS, [0, 1], self.rng,
                                                self.noise, Q)
        d = [-v for v in self.e] + [1]
        self.assertEqual(dppy.residual(self.e, (p, c), Q), dppy.dot(d, p + [c], Q))
        self.assertFalse(constraints.spans(d))

    def test_residual_is_uniform_outside_bad(self):
        # Over Z_11 with m = 4, d = (-e, 1) is outside W, so <d, y> is uniform, with the
        # broadcast noise and without it.
        q, keys, e = 11, {"xs": [[1, 0, 2, 5], [3, 1, 0, 4]]}, [2, -1, 0, 1]
        d = [-v for v in e] + [1]
        rng, draws = random.Random(8), 11 * 600
        for noise in (ZeroNoise(), dppy.Gaussians(rng, SIGMA, R)):
            counts = Counter()
            for _ in range(draws):
                ct, constraints = dppy.dppy25_query(keys, [1, 1], [0, 1], rng, noise, q)
                counts[dppy.residual(e, ct, q)] += 1
            self.assertFalse(constraints.spans(d))
            self.assertLess(chi_square([counts[v] for v in range(q)], [draws / q] * q),
                            chi_square_limit(q - 1))

    def test_bad_event_leaves_only_noise(self):
        # The proof's exception: with e_1 = 0, the first key x_1 = v = (-e_2, ..., -e_m, 1)
        # and am_1 = 1, d = (0, v) = dk_1 - r_1 lies in W. Then <d, y> = 0 and the residual
        # is the broadcast noise alone, as for an ordinary encryption of 0.
        noise = dppy.Gaussians(self.rng, SIGMA, R)
        e = [0] + noise.discrete(M - 1)
        v = [-x for x in e[1:]] + [1]
        keys = {"xs": [v] + [noise.discrete(M) for _ in range(sum(GROUPS) - 1)]}
        d = [-x for x in e] + [1]
        for _ in range(10):
            ct, constraints = dppy.dppy25_query(keys, GROUPS, [1, self.rng.getrandbits(1)],
                                                self.rng, noise, Q)
            self.assertTrue(constraints.spans(d))
            self.assertLess(dppy.circular(dppy.residual(e, ct, Q), Q), 8 * SIGMA * norm(d))
        ct, constraints = dppy.dppy25_query(keys, GROUPS, [0, 0], self.rng, noise, Q)
        self.assertFalse(constraints.spans(d))


class ExperimentTest(unittest.TestCase):
    """The experiment of Proposition 3.1 at small parameters (which also meet DPPY25's
    conditions): D accepts every ordinary ciphertext, and an anamorphic one with
    probability (4E + 2)/q."""

    @classmethod
    def setUpClass(cls):
        cls.args = argparse.Namespace(trials=1500, seed=9, n=8, m=256, q=65521, sigma=SIGMA,
                                      r=R, groups=GROUPS, recipients=2, bits=2,
                                      progress=False)
        cls.results, cls.tight, cls.checks, cls.loose, _ = dppy.run(cls.args)
        with contextlib.redirect_stdout(io.StringIO()):
            cls.summary = dppy.report(cls.args, cls.results, cls.tight, cls.checks,
                                      cls.loose, 0.0)

    def assert_rate(self, accepted, p):
        trials = self.args.trials
        self.assertLess(abs(accepted / trials - p), 4 * math.sqrt(p * (1 - p) / trials) + 1e-9)

    def test_conditions(self):
        a = self.args
        self.assertTrue(all(row[1] for row in dppy.dppy25_conditions(a.n, a.m, a.q, a.sigma, a.r)))

    def test_dppy25_against_proposition_3_1(self):
        q, trials = self.args.q, self.args.trials
        self.assertEqual(self.checks["dppy25_ordinary"], trials)
        self.assertEqual(self.checks["bad"], 0)
        loose = self.summary["dppy25, E = floor(q/5)"]
        self.assertEqual(loose["ordinary_accepts"], trials)
        self.assert_rate(loose["anamorphic_accepts"], (4 * self.loose + 2) / q)
        tight = self.summary["dppy25, E = 8 sd of the error"]
        self.assertEqual(tight["ordinary_accepts"], trials)
        self.assert_rate(tight["anamorphic_accepts"],
                         sum((4 * t + 2) / q for t in self.tight["dppy25"]) / trials)
        self.assertGreater(tight["advantage"], 0.9)

    def test_construction1_modes_look_alike(self):
        trials = self.args.trials
        self.assertEqual(self.checks["c1_ordinary"], 2 * trials)
        self.assertEqual(self.checks["c1_hidden"], self.checks["c1_hidden_total"])
        for threshold in ("floor(q/5)", "8 sd of the error"):
            row = self.summary[f"construction1, E = {threshold}"]
            self.assertEqual((row["ordinary_accepts"], row["anamorphic_accepts"]),
                             (trials, trials))
        self.assertGreater(self.summary["construction1, Kolmogorov-Smirnov"]["residual"]
                           ["p_value"], 1e-4)


class ParameterTest(unittest.TestCase):
    def test_defaults_meet_dppy25_conditions(self):
        for name, holds, value, bound in dppy.dppy25_conditions(N, M, Q, SIGMA, R):
            self.assertTrue(holds, f"{name}: {value} against {bound}")

    def test_previous_default_failed_one(self):
        # n = 32 at this q needs m >= 1920.
        failing = [row[0] for row in dppy.dppy25_conditions(32, M, Q, SIGMA, R) if not row[1]]
        self.assertIn("m >= 2 n log2 q", failing)


class StatisticsTest(unittest.TestCase):
    def test_wilson_and_newcombe(self):
        # Newcombe (1998): the score interval for 81/263, and his hybrid score interval
        # for 56/70 - 48/80.
        low, high = dppy.wilson(81, 263)
        self.assertAlmostEqual(low, 0.2553, places=4)
        self.assertAlmostEqual(high, 0.3662, places=4)
        low, high = dppy.newcombe(56, 70, 48, 80)
        self.assertAlmostEqual(low, 0.0524, places=4)
        self.assertAlmostEqual(high, 0.3339, places=4)
        low, high = dppy.wilson(0, 1000)
        self.assertLess(low, 1e-12)
        self.assertAlmostEqual(high, 3.8415 / 1003.8415, places=6)

    def test_kolmogorov_smirnov(self):
        same = list(range(100))
        self.assertEqual(dppy.kolmogorov_smirnov(same, same), (0.0, 1.0))
        statistic, p = dppy.kolmogorov_smirnov(list(range(100)), list(range(100, 200)))
        self.assertEqual(statistic, 1.0)
        self.assertLess(p, 1e-20)


class CommandLineTest(unittest.TestCase):
    def test_small_run(self):
        with tempfile.TemporaryDirectory() as folder:
            path = os.path.join(folder, "out.json")
            done = subprocess.run(
                [sys.executable, SCRIPT, "--trials", "20", "--n", "8", "--m", "256", "--q",
                 "65521", "--recipients", "2", "--bits", "2", "--json", path],
                capture_output=True, text=True, timeout=300)
            self.assertEqual(done.returncode, 0, done.stderr)
            self.assertNotIn("warning", done.stderr)
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
        self.assertEqual(data["checks"]["bad"], 0)
        self.assertTrue(all(row["holds"] for row in data["tests"]["dppy25 conditions"]))
        self.assertIn("DPPY25's parameter conditions", done.stdout)

    def test_rejects_bad_arguments(self):
        for extra in (["--q", "65535"], ["--recipients", "4", "--bits", "4"]):
            done = subprocess.run([sys.executable, SCRIPT, "--trials", "1", *extra],
                                  capture_output=True, text=True, timeout=60)
            self.assertEqual(done.returncode, 2, extra)

    def test_warns_outside_the_conditions(self):
        done = subprocess.run([sys.executable, SCRIPT, "--trials", "2", "--n", "32"],
                              capture_output=True, text=True, timeout=300)
        self.assertEqual(done.returncode, 0, done.stderr)
        self.assertIn("outside DPPY25's parameter conditions", done.stderr)


if __name__ == "__main__":
    unittest.main()
