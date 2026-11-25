"""Derive the trapdoor-setup presets (TrapdoorProfile::small and ::full).

For a dimension n, J ordinary bits, g hidden constraints and gadget base b, find
the smallest prime q (fixed point) such that
  m = max(ceil(n log2 q) + 2*128 + 2 ceil(log2 g), 3n + nk)   (Construction 1, MP12)
  s = ceil(sqrt(sigma_G^2 (s1^2 + 1) + r^2))                   (MP12 SampleD)
  C = ceil(noise bound of keys of expected norm sqrt(m) s / sqrt(2 pi))
  q >= 65 C * 2 sqrt(n) / sqrt(2 pi)                           (Regev width for q/(65C))
with r = eta_eps(Z^m) for eps = 2^-128, sigma_G = r sqrt(b^2 + 1), s1 = 1.1 sqrt(2/3)
(sqrt(3n) + sqrt(nk)) and k the number of base-b digits of q. Abar has 3n columns, so
the trapdoor's ternary LWE has secret dimension 2n against n samples. The conditions
mirror TrapdoorProfile::new in src/trapdoor.rs, which re-checks them. Usage:
    python3 tools/trapdoor_parameters.py
"""
import math, sys

LAMBDA = 128
SMOOTHING_BITS = 128      # epsilon = 2^-128 for eta_eps(Z^m)
KEY_NORM_BITS = 128       # per key column: P(norm above bound) <= 2^-128, resampled if it happens
SIGN_NORM_BITS = 128      # per encryption and bit: P(|R d|^2 > f m |d|^2) <= 2^-128
TRAPDOOR_SLACK = 1.1      # s1(R) <= 1.1 sqrt(2/3)(sqrt(3n) + sqrt(w)), else TrapGen resamples R
ABAR = 3                  # Abar has ABAR * n columns (rows of the trapdoor R)

def is_prime(x):
    if x < 2: return False
    for p in (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37):
        if x % p == 0: return x == p
    d, s = x - 1, 0
    while d % 2 == 0: d //= 2; s += 1
    for a in (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37):
        y = pow(a, d, x)
        if y in (1, x - 1): continue
        for _ in range(s - 1):
            y = y * y % x
            if y == x - 1: break
        else: return False
    return True

def next_prime(x):
    x = max(3, math.ceil(x)) | 1
    while not is_prime(x): x += 2
    return x

def rounding_width(m):            # eta_eps(Z^m), eps = 2^-128
    return math.sqrt(math.log(2 * m * (1 + 2.0 ** SMOOTHING_BITS)) / math.pi)

def digits(q, b):                 # smallest k with b^k > q
    k, x = 0, 1
    while x <= q: x *= b; k += 1
    return k

def chi2_factor(m, bits):         # smallest f = 1+e with m (e - ln(1+e)) / 2 >= bits ln 2
    target = 2 * bits * math.log(2) / m
    lo, hi = 0.0, 1.0
    while hi - math.log1p(hi) < target: hi *= 2
    for _ in range(200):
        mid = (lo + hi) / 2
        if mid - math.log1p(mid) >= target: hi = mid
        else: lo = mid
    return 1 + hi

def construction1_m(n, q, g):     # ceil(n log2 q) + 2 lambda + 2 ceil(log2 g), as in the paper
    return (q ** n).bit_length() + 2 * LAMBDA + 2 * math.ceil(math.log2(g))

def width(n, w, m, b):
    r = rounding_width(m)
    sigma_g = r * math.sqrt(b * b + 1)
    s1 = TRAPDOOR_SLACK * math.sqrt(2 / 3) * (math.sqrt(ABAR * n) + math.sqrt(w))
    return math.sqrt(sigma_g ** 2 * (s1 ** 2 + 1) + r ** 2), r, sigma_g, s1

def c_bound(m, s):
    expected = math.sqrt(m) * s / math.sqrt(2 * math.pi)
    f_d = math.sqrt(chi2_factor(m, KEY_NORM_BITS))
    f_r = chi2_factor(m, SIGN_NORM_BITS)
    return math.ceil(math.sqrt(1 + (f_d * expected * (1 + math.sqrt(f_r * m))) ** 2))

def regev_q(n, c):                # q/(65 C) >= 2 sqrt(n) / sqrt(2 pi)
    return 65 * c * 2 * math.sqrt(n) / math.sqrt(2 * math.pi)

def derive(n, j, g, b):
    q = 1 << 20
    for _ in range(100):
        k = digits(q, b); w = n * k
        m = max(construction1_m(n, q, g), ABAR * n + w)
        s_min, r, sigma_g, s1 = width(n, w, m, b)
        s = math.ceil(s_min)
        c = c_bound(m, s)
        qn = next_prime(regev_q(n, c))
        if qn == q: break
        q = qn
    k = digits(q, b); w = n * k
    m = max(construction1_m(n, q, g), ABAR * n + w)
    s_min, r, sigma_g, s1 = width(n, w, m, b)
    s = math.ceil(s_min); c = c_bound(m, s)
    assert q >= regev_q(n, c) and m >= construction1_m(n, q, g) and m >= ABAR * n + w
    return dict(n=n, m=m, q=q, j=j, g=g, c=c, base=b, k=k, w=w, pad=m - ABAR * n - w, s=s,
                r=r, sigma_g=sigma_g, s1=s1, error_sd=q / (65 * c))

if __name__ == "__main__":
    print("Gadget bases for n = 1360 (the presets use b = 4: q within 0.5 bit of b = 2,")
    print("with half of b = 2's trapdoor columns):")
    for b in (2, 3, 4, 8, 16):
        p = derive(1360, 128, 8, b)
        print(f"  b={b:2d}: m={p['m']} (nk={p['w']}, padding={p['pad']}), q=2^{math.log2(p['q']):.2f}, "
              f"s={p['s']}, C={p['c']}")
    for name, (n, j, g) in (("small", (16, 8, 8)), ("full", (1360, 128, 8))):
        p = derive(n, j, g, 4)
        print(f"{name}: Profile::new({n}, {p['m']}, {p['q']}, {j}, {g}, {p['c']}), base 4, width {p['s']}.0"
              f"  [q = 2^{math.log2(p['q']):.2f}, nk = {p['w']}, padding = {p['pad']}, r = {p['r']:.3f},"
              f" sigma_G = {p['sigma_g']:.2f}, error sd = {p['error_sd']:.2f}]")
