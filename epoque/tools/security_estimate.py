"""Attack estimates for the trapdoor-setup presets and the choice of the full profile's n.

Core-SVP in the 2016 model of Alkim, Ducas, Poeppelmann and Schwabe (NewHope):
BKZ with block size beta costs 2^(0.292 beta) classically and 2^(0.265 beta)
quantumly. The primal attack is the uSVP embedding with the secret scaled to the
error's width, the dual attack counts the 2^(0.2075 beta) short vectors of one
sieve against the 1/eps^2 that distinguishing needs. The same code gives 118 bits
for Kyber-512 and 182 for Kyber-768 (the published figures). These are not
lattice-estimator results, and hybrid attacks on the ternary trapdoor are not covered.

Two instances matter for each profile:
  ciphertext  LWE in dimension n - g (aEnc fixes g linear combinations of s) with the
              rounded-normal error of standard deviation q / (65 C), uniform secret;
  trapdoor    MP12's A0 = [Abar | G - Abar R] with Abar of width 3n: ternary LWE with
              secret dimension 2n and only n samples.
The full profile takes the smallest n (in steps of 8) whose ciphertext instance
reaches 128 bits classically in both attacks, with every other value derived by
tools/trapdoor_parameters.py. Usage:
    python3 tools/security_estimate.py
"""
import math

from trapdoor_parameters import derive

TARGET_BITS = 128
TERNARY_SD = math.sqrt(2 / 3)


def delta(beta):
    return ((math.pi * beta) ** (1 / beta) * beta / (2 * math.pi * math.e)) ** (1 / (2 * (beta - 1)))


def primal_beta(n, m_max, q, sd_e, sd_s):
    """Smallest beta for which some m <= m_max meets sd_e sqrt(beta) <= delta^(2 beta - d - 1) vol^(1/d)."""
    log_nu = math.log(sd_e / sd_s)
    step = max(1, m_max // 400)
    samples = list(range(step, m_max + 1, step)) + [m_max]
    for beta in range(50, 5000):
        ld = math.log(delta(beta))
        lhs = math.log(sd_e) + 0.5 * math.log(beta)
        for m in samples:
            d = n + m + 1
            if lhs <= (2 * beta - d - 1) * ld + (m * math.log(q) + n * log_nu) / d:
                return beta
    raise ValueError("no block size up to 5000")


def dual_bits(n, m_max, q, sd_e, sd_s):
    """Smallest classical cost max(1, 1 / (eps^2 2^(0.2075 beta))) 2^(0.292 beta), in bits."""
    log_nu = math.log(sd_e / sd_s)
    step = max(1, m_max // 300)
    samples = list(range(step, m_max + 1, step)) + [m_max]
    best = float("inf")
    for beta in range(50, 5000, 2):
        if 0.292 * beta > best:
            break
        ld = math.log(delta(beta))
        for m in samples:
            d = m + n
            length = math.exp((d - 1) * ld + n * (math.log(q) - log_nu) / d)
            tau = length * sd_e / q
            log2_eps = 2 - 2 * math.pi ** 2 * tau ** 2 / math.log(2)
            best = min(best, 0.292 * beta + max(0.0, -2 * log2_eps - 0.2075 * beta))
    return best


def instances(p):
    """(name, primal beta, dual bits) of the ciphertext and trapdoor instances of a profile."""
    n, q, g = p["n"], p["q"], p["g"]
    sd = q / (65 * p["c"])
    return [
        ("ciphertext LWE, dimension n - g", primal_beta(n - g, 4 * n, q, sd, sd), dual_bits(n - g, 4 * n, q, sd, sd)),
        ("trapdoor ternary LWE, 2n by n", primal_beta(2 * n, n, q, TERNARY_SD, TERNARY_SD),
         dual_bits(2 * n, n, q, TERNARY_SD, TERNARY_SD)),
    ]


def ciphertext_bits(n):
    p = derive(n, 128, 8, 4)
    sd = p["q"] / (65 * p["c"])
    return min(0.292 * primal_beta(n - 8, 4 * n, p["q"], sd, sd), dual_bits(n - 8, 4 * n, p["q"], sd, sd))


if __name__ == "__main__":
    for name, n, sd in (("Kyber-512", 512, math.sqrt(1.5)), ("Kyber-768", 768, 1.0)):
        print(f"{name}: {0.292 * primal_beta(n, n, 3329, sd, sd):.1f} bits classical (calibration)")
    n = 1024
    while ciphertext_bits(n) < TARGET_BITS:
        n += 8
    print(f"Smallest n (steps of 8) with a {TARGET_BITS}-bit ciphertext instance: {n}")
    for n in (1360, 1984):
        p = derive(n, 128, 8, 4)
        size = (2 * p["m"] + 128) * ((p["q"].bit_length() + 7) // 8)
        print(f"n = {n}: m = {p['m']}, q = 2^{math.log2(p['q']):.2f}, error sd {p['q'] / (65 * p['c']):.2f}, "
              f"ciphertext {size} bytes")
        for name, beta, dual in instances(p):
            print(f"  {name}: beta {beta}, classical {0.292 * beta:.0f}, quantum {0.265 * beta:.0f}, "
                  f"dual {dual:.0f}")
