//! Table-based sampler for the rounded normal error `round(N(0, s^2))`.
//!
//! The paper's error distribution is the normal distribution with standard
//! deviation `s = q sigma`, rounded to the nearest integer. The width is given
//! either as a double, used exactly, or as an integer variance `V` with
//! `s = sqrt(V)` exactly. The table only needs `1/s^2`, so an exact variance
//! never rounds a square root. The manuscript's `(q sigma)^2 = n (m + 1)` is such
//! an integer, while the double nearest `sqrt(34972416)` is `2^-57.7` too large
//! (about `2^-58.7` of statistical distance per sample). This module stores
//! the cumulative distribution of `|e|` as a base table of 256-bit fixed-point
//! values (a cumulative distribution table, CDT) together with a small guide
//! table indexed by the leading bits of the uniform input.
//!
//! Sampling draws a uniform 256-bit number `U` lazily, 64 bits at a time, and
//! returns the first `k` with `U < T_k`. The guide table maps the leading 16
//! bits of `U` to the range of candidate entries. For the maintained profile
//! about 77% of draws are decided by the guide table alone, the others read one
//! or a few leading words of the table, and only a `2^-47` fraction needs the
//! lower 192 bits. One sample therefore costs one 64-bit random word, one guide
//! lookup, rarely a table read, and one sign bit. The output follows exactly
//! the distribution encoded by the table, whose statistical distance to the
//! ideal rounded normal distribution is bounded per table (below `2^-239` per
//! sample for the maintained profile). No crate offers this sampling step at
//! 256-bit resolution, so it stays here; all multiprecision arithmetic of the
//! table construction is done by MPFR (through `rug`).
//!
//! # How the table is computed
//!
//! For `k >= 0` let `w(k) = int_{k-1/2}^{k+1/2} exp(-x^2/(2 s^2)) dx`, the
//! unnormalised mass of `k`. With `u = x - k` it factors as
//! `w(k) = exp(-k^2/(2 s^2)) J(k)`, where
//! `J(k) = int_{-1/2}^{1/2} exp(-k u / s^2 - u^2 / (2 s^2)) du`.
//! Expanding the odd-free part `cosh(k u / s^2)` gives the positive series
//! `J(k) = exp(-x0) sum_e y^e c_e` with `y = (k/s^2)^2`, `x0 = 1/(8 s^2)` and
//! `c_e = 2^(-2e-1) S_e / (2e)!`, `S_e = sum_j t_j`, `t_0 = 2/(2e+1)`,
//! `t_j = t_{j-1} 2 x0 / (2e+2j+1)` (a lower incomplete gamma series).
//! The common factor `exp(-x0)` cancels in the normalisation and is dropped.
//! `exp(-k^2/(2s^2))` follows the recurrence `rho(k) = rho(k-1) f_k`,
//! `f_k = f_{k-1} exp(-1/s^2)`, `f_1 = exp(-1/(2 s^2))`.
//!
//! Every value is an MPFR float with a 384-bit mantissa, and every operation,
//! including MPFR's correctly rounded `exp`, rounds toward zero. Every series
//! has positive terms only, so the relative error of every cumulative value
//! stays below `2^-356` (each operation loses at most `2^-383`, and no value
//! passes through more than about `2^20` operations), far below the `2^-256`
//! resolution of the entries. MPFR's `erf` would give the entries directly, but
//! at this precision it costs about 0.1 ms per entry, so the series is used and
//! `tools/noise_table_mpfr.c` recomputes the entries with MPFR's `erf` instead;
//! `tests/noise_table.rs` checks that both agree bit for bit.
//!
//! The table covers `|e| <= B` with `B` chosen so that the ideal mass beyond `B`
//! is below `2^-260`, normalised by the mass of `[-B, B]`. Its statistical
//! distance to the ideal distribution is therefore at most
//! `B 2^-256 (1 + 2^-100) + 2^-260`.

use crate::{Error, Result};
use chacha20::ChaCha12Rng;
use rand::{Rng, SeedableRng};
use rug::{
    Assign, Float,
    float::Round,
    integer::Order,
    ops::{AddAssignRound, MulAssignRound},
};
use std::cmp::Ordering;
use std::sync::{Arc, Mutex, OnceLock};

/// Leading bits of the uniform word that index the guide table.
const GUIDE_BITS: u32 = 16;
/// Base-2 logarithm of the largest ideal tail mass the table may drop.
const TAIL_LOG2: f64 = -260.0;
/// Entries computed per independent block when the table is built.
const BUILD_BLOCK: usize = 4096;

/// Number of error coordinates drawn from one ChaCha12 stream.
pub(crate) const NOISE_CHUNK: usize = 1024;

/// The width `s` of the unrounded normal distribution. The table only needs
/// `1/s^2`, so an integer variance is used exactly, without a square root.
#[derive(Clone, Copy, Debug)]
pub(crate) enum Width {
    /// `s` is the exact value of this double.
    Sd(f64),
    /// `s` is the exact square root of this integer.
    Variance(u64),
}

impl Width {
    /// Cache key: two widths share a table exactly when their keys agree.
    fn key(self) -> (bool, u64) {
        match self {
            Width::Sd(sd) => (false, sd.to_bits()),
            Width::Variance(v) => (true, v),
        }
    }
    /// `s` rounded to a double, for the tail bound and for reporting.
    fn approx(self) -> f64 {
        match self {
            Width::Sd(sd) => sd,
            Width::Variance(v) => (v as f64).sqrt(),
        }
    }
    /// `1/s^2`, rounded toward zero (`s^2` itself is exact in both cases).
    fn inv_s2(self) -> Float {
        match self {
            Width::Sd(sd) => {
                let s = Float::with_val(PRECISION, sd);
                recip(&mul(&s, &s))
            }
            Width::Variance(v) => recip(&Float::with_val(PRECISION, v)),
        }
    }
}

// ---------------------------------------------------------------------------
// MPFR arithmetic (through `rug`) at a fixed precision, rounding toward zero.

/// Mantissa bits of every value in the table construction.
const PRECISION: u32 = 384;
/// Every operation truncates: positive results never exceed the exact value.
const ROUND: Round = Round::Zero;
/// A term below `2^-GAP` of the running sum ends a series.
const GAP: i32 = PRECISION as i32 + 8;

fn mul(a: &Float, b: &Float) -> Float {
    Float::with_val_round(PRECISION, a * b, ROUND).0
}

fn add(a: &Float, b: &Float) -> Float {
    Float::with_val_round(PRECISION, a + b, ROUND).0
}

fn mul_u64(a: &Float, k: u64) -> Float {
    Float::with_val_round(PRECISION, a * k, ROUND).0
}

fn div_u64(a: &Float, k: u64) -> Float {
    Float::with_val_round(PRECISION, a / k, ROUND).0
}

fn recip(a: &Float) -> Float {
    let mut x = a.clone();
    x.recip_round(ROUND);
    x
}

/// `a 2^k`, exact.
fn ldexp(a: &Float, k: i32) -> Float {
    Float::with_val(PRECISION, a << k)
}

/// `exp(-v)`, correctly rounded toward zero by MPFR.
fn exp_negative(v: &Float) -> Float {
    let mut x = Float::with_val(PRECISION, -v);
    x.exp_round(ROUND);
    x
}

/// `true` if `a < b 2^-gap` is certain from the exponents alone.
fn negligible(a: &Float, b: &Float, gap: i32) -> bool {
    match (a.get_exp(), b.get_exp()) {
        _ if a.is_zero() => true,
        (Some(ea), Some(eb)) => ea < eb - gap,
        _ => false,
    }
}

/// `floor(x 2^256)` as four little-endian words if `0 <= x < 1`.
fn fixed256(x: &Float) -> Option<[u64; 4]> {
    if *x >= 1 {
        return None;
    }
    let (scaled, _) = Float::with_val(PRECISION, x << 256u32).to_integer_round(Round::Down)?;
    let digits = scaled.to_digits::<u64>(Order::Lsf);
    let mut words = [0u64; 4];
    words[..digits.len()].copy_from_slice(&digits);
    Some(words)
}

/// Series coefficients `c_e` of `J`, up to the first one that is negligible at
/// the largest `y` used by the table (where the terms also decay by half).
fn series_coefficients(x0: &Float, y_max: &Float) -> Vec<Float> {
    let two_x = ldexp(x0, 1);
    let x0_approx = x0.to_f64();
    let mut coefficients = Vec::new();
    let mut sum_at_max = Float::new(PRECISION);
    let mut previous_term = Float::new(PRECISION);
    let mut factorial = Float::with_val(PRECISION, 1);
    for e in 0u64.. {
        if e > 0 {
            factorial = mul_u64(&mul_u64(&factorial, 2 * e - 1), 2 * e);
        }
        let mut t = div_u64(&Float::with_val(PRECISION, 2), 2 * e + 1);
        let mut s = t.clone();
        for j in 1u64.. {
            t = div_u64(&mul(&t, &two_x), 2 * e + 2 * j + 1);
            let halving = 4.0 * x0_approx <= (2 * e + 2 * j + 3) as f64 - 1.0;
            if t.is_zero() || (halving && negligible(&t, &s, GAP)) {
                break;
            }
            s = add(&s, &t);
        }
        let c = mul(&ldexp(&s, -2 * e as i32 - 1), &recip(&factorial));
        let mut term = c.clone();
        for _ in 0..e {
            term = mul(&term, y_max);
        }
        coefficients.push(c);
        let decaying = e > 0 && ldexp(&term, 1) <= previous_term;
        if decaying && negligible(&term, &sum_at_max, GAP) {
            break;
        }
        sum_at_max = add(&sum_at_max, &term);
        previous_term = term;
    }
    coefficients
}

// ---------------------------------------------------------------------------

/// The base table for one error width.
pub struct NoiseTable {
    sd: f64,
    /// The exact variance, if the table was built from one.
    variance: Option<u64>,
    /// Largest `|e|` the table can return.
    bound: usize,
    /// Most significant 64 bits of `T_k` for `k < bound`.
    hi: Vec<u64>,
    /// Remaining 192 bits of `T_k`, most significant first.
    lo: Vec<[u64; 3]>,
    /// `guide[g]` is the first `k` whose `hi[k]` has leading bits at least `g`.
    guide: Vec<u32>,
    distance_log2: f64,
    tail_log2: f64,
}

impl std::fmt::Debug for NoiseTable {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("NoiseTable")
            .field("sd", &self.sd)
            .field("variance", &self.variance)
            .field("bound", &self.bound)
            .field("bytes", &self.bytes())
            .field("distance_log2", &self.distance_log2)
            .finish()
    }
}

/// Upper bound on `log2 P(|N(0, s^2)| > B + 1/2)` from `erfc(z) <= exp(-z^2)/(z sqrt(pi))`.
fn tail_log2(sd: f64, bound: usize) -> f64 {
    let z = (bound as f64 + 0.5) / (sd * std::f64::consts::SQRT_2);
    (-(z * z) - (z * std::f64::consts::PI.sqrt()).ln()) / std::f64::consts::LN_2
}

impl NoiseTable {
    /// Builds the table. The work is split into fixed blocks of entries, which
    /// run on the rayon pool when `parallel` is set; the table is identical
    /// either way.
    pub(crate) fn build(width: Width, parallel: bool) -> Result<Self> {
        let valid = match width {
            Width::Sd(sd) => sd.is_finite() && sd > 0.0,
            Width::Variance(v) => v > 0,
        };
        if !valid {
            return Err(Error("invalid noise standard deviation"));
        }
        // The bound and the tail estimate only need s to a few digits. The double
        // nearest an exact sqrt(V) is within 2^-53 of it, which moves the tail
        // estimate by about 2^-45 bits, far inside the slack of the erfc bound.
        let sd = width.approx();
        // Smallest B with tail mass below 2^-260 (and B >= 1).
        let mut bound = ((13.0 * std::f64::consts::SQRT_2 * sd).floor() as usize).max(1);
        while tail_log2(sd, bound) > TAIL_LOG2 {
            bound += 1;
        }
        if bound > u32::MAX as usize - 2 {
            return Err(Error("noise standard deviation too large for the table"));
        }
        let tail = tail_log2(sd, bound);
        let inv_s2 = width.inv_s2();
        let x0 = ldexp(&inv_s2, -3);
        let y = |k: u64| {
            let t = mul_u64(&inv_s2, k);
            mul(&t, &t)
        };
        let coefficients = series_coefficients(&x0, &y(bound as u64));
        let half_inv_s2 = ldexp(&inv_s2, -1);
        let step = exp_negative(&inv_s2);
        // Unnormalised masses w(k) for k = 0..=bound, in independent blocks.
        let blocks: Vec<usize> = (0..=bound).step_by(BUILD_BLOCK).collect();
        let block_masses = |&start: &usize| -> Vec<Float> {
            let stop = (start + BUILD_BLOCK).min(bound + 1);
            let k0 = start as u64;
            // rho(k0) = exp(-k0^2/(2s^2)), f_{k0+1} = exp(-(2 k0 + 1)/(2 s^2)).
            let mut rho = exp_negative(&mul_u64(&mul_u64(&half_inv_s2, k0), k0));
            let mut factor = exp_negative(&mul_u64(&half_inv_s2, 2 * k0 + 1));
            let (last, rest) = coefficients.split_last().expect("at least one coefficient");
            // The same truncating MPFR operations as `mul` and `add`, in place.
            let (mut yk, mut j) = (Float::new(PRECISION), Float::new(PRECISION));
            let mut out = Vec::with_capacity(stop - start);
            for k in start..stop {
                if k > start {
                    rho.mul_assign_round(&factor, ROUND);
                    factor.mul_assign_round(&step, ROUND);
                }
                // yk = (k / s^2)^2 and J(k) by Horner's rule.
                yk.assign(&inv_s2);
                yk.mul_assign_round(k as u64, ROUND);
                yk.square_round(ROUND);
                j.assign(last);
                for c in rest.iter().rev() {
                    j.mul_assign_round(&yk, ROUND);
                    j.add_assign_round(c, ROUND);
                }
                let mut mass = mul(&rho, &j);
                if k > 0 {
                    mass <<= 1u32;
                }
                out.push(mass);
            }
            out
        };
        let masses: Vec<Vec<Float>> = if parallel {
            use rayon::prelude::*;
            blocks.par_iter().map(block_masses).collect()
        } else {
            blocks.iter().map(block_masses).collect()
        };
        // Cumulative sums in a fixed order, then normalisation by the total.
        let mut cumulative = Vec::with_capacity(bound + 1);
        let mut running = Float::new(PRECISION);
        for mass in masses.iter().flatten() {
            running = add(&running, mass);
            cumulative.push(running.clone());
        }
        let inv_total = recip(&running);
        let mut hi = Vec::with_capacity(bound);
        let mut lo = Vec::with_capacity(bound);
        let mut previous = [0u64; 4];
        for c in &cumulative[..bound] {
            let limbs = fixed256(&mul(c, &inv_total)).unwrap_or([u64::MAX; 4]);
            // Monotone by construction, checked because the sampler relies on it.
            if limbs.iter().rev().cmp(previous.iter().rev()) == Ordering::Less {
                return Err(Error("noise table is not monotone"));
            }
            previous = limbs;
            hi.push(limbs[3]);
            lo.push([limbs[2], limbs[1], limbs[0]]);
        }
        let mut guide = Vec::with_capacity((1 << GUIDE_BITS) + 1);
        let mut k = 0usize;
        for g in 0..=(1u64 << GUIDE_BITS) {
            while k < bound && (hi[k] >> (64 - GUIDE_BITS)) < g {
                k += 1;
            }
            guide.push(k as u32);
        }
        // Each entry is within 2^-256 (1 + 2^-100) of the exact normalised value,
        // so |Q(k) - P(k)| <= 2^-255 (1 + 2^-100) and summing over the 2B inner
        // differences gives the bound below.
        let distance = (bound as f64).log2() - 256.0 + 1e-30;
        let distance_log2 =
            (2f64.powf(distance) * (1.0 + 2f64.powi(-100)) + 2f64.powf(tail)).log2();
        Ok(Self {
            sd,
            variance: match width {
                Width::Variance(v) => Some(v),
                Width::Sd(_) => None,
            },
            bound,
            hi,
            lo,
            guide,
            distance_log2,
            tail_log2: tail,
        })
    }

    /// Builds a table without the shared cache (for measurements). Normally use
    /// [`crate::noise_table`], which builds once per standard deviation.
    pub fn new(sd: f64, mode: crate::Mode) -> Result<Self> {
        Self::build(Width::Sd(sd), mode == crate::Mode::Multi)
    }

    /// [`NoiseTable::new`] for the standard deviation `sqrt(variance)`, used
    /// exactly (the double nearest a square root is not the square root).
    pub fn with_variance(variance: u64, mode: crate::Mode) -> Result<Self> {
        Self::build(Width::Variance(variance), mode == crate::Mode::Multi)
    }

    /// The standard deviation of the unrounded normal distribution (the
    /// nearest double if the table was built from an exact variance).
    pub fn sd(&self) -> f64 {
        self.sd
    }
    /// The exact variance the table was built from, if any.
    pub fn variance(&self) -> Option<u64> {
        self.variance
    }
    /// Largest absolute value the sampler returns.
    pub fn bound(&self) -> usize {
        self.bound
    }
    /// Number of cumulative entries stored.
    pub fn entries(&self) -> usize {
        self.hi.len()
    }
    /// Memory used by the base table and the guide table.
    pub fn bytes(&self) -> usize {
        self.hi.len() * 8 + self.lo.len() * 24 + self.guide.len() * 4
    }
    /// Base-2 logarithm of the upper bound on the statistical distance between
    /// one sample and the ideal rounded normal distribution.
    pub fn distance_log2(&self) -> f64 {
        self.distance_log2
    }
    /// Base-2 logarithm of the ideal mass beyond the bound (included above).
    pub fn tail_log2(&self) -> f64 {
        self.tail_log2
    }
    /// The 256-bit cumulative entry `T_k` as big-endian limbs, for `k < bound`.
    pub fn entry(&self, k: usize) -> [u64; 4] {
        [self.hi[k], self.lo[k][0], self.lo[k][1], self.lo[k][2]]
    }

    /// Samples `|e|`. Returns the first `k` with `U < T_k` (and `bound` if none),
    /// where `U` is uniform on 256 bits and drawn lazily from `rng`.
    #[inline]
    fn sample_magnitude<R: Rng + ?Sized>(&self, rng: &mut R) -> usize {
        let u = rng.next_u64();
        let g = (u >> (64 - GUIDE_BITS)) as usize;
        let start = self.guide[g] as usize;
        let stop = self.guide[g + 1] as usize;
        // Entries before `start` lie below u. Entry `stop` (or the bound itself)
        // lies above u, because its leading bits exceed those of u.
        if start == stop {
            return start;
        }
        let window = &self.hi[start..stop];
        let first_not_below = start
            + if window.len() <= 8 {
                window.iter().take_while(|&&h| h < u).count()
            } else {
                window.partition_point(|&h| h < u)
            };
        if first_not_below == stop || self.hi[first_not_below] > u {
            return first_not_below;
        }
        // The leading words agree on a run of entries: compare the remaining 192
        // bits of U, drawn now, with those of T_k.
        let first_above =
            first_not_below + self.hi[first_not_below..stop].partition_point(|&h| h <= u);
        let rest = [rng.next_u64(), rng.next_u64(), rng.next_u64()];
        (first_not_below..first_above)
            .find(|&k| rest < self.lo[k])
            .unwrap_or(first_above)
    }

    /// Samples one signed error. `signs` buffers sign bits drawn from `rng`.
    #[inline]
    pub(crate) fn sample<R: Rng + ?Sized>(&self, rng: &mut R, signs: &mut SignBits) -> i64 {
        let magnitude = self.sample_magnitude(rng) as i64;
        if magnitude == 0 {
            0
        } else if signs.next(rng) {
            -magnitude
        } else {
            magnitude
        }
    }

    /// Fills `out` with the errors of one ChaCha12 stream of `seed`.
    pub(crate) fn fill_stream(&self, seed: &[u8; 32], stream: u64, out: &mut [i64]) {
        let mut rng = ChaCha12Rng::from_seed(*seed);
        rng.set_stream(stream);
        let mut signs = SignBits::default();
        for e in out {
            *e = self.sample(&mut rng, &mut signs);
        }
    }
}

/// Sign bits taken one at a time from 64-bit words of the stream.
#[derive(Default)]
pub(crate) struct SignBits {
    word: u64,
    left: u32,
}

impl SignBits {
    #[inline]
    fn next<R: Rng + ?Sized>(&mut self, rng: &mut R) -> bool {
        if self.left == 0 {
            self.word = rng.next_u64();
            self.left = 64;
        }
        let bit = self.word & 1 == 1;
        self.word >>= 1;
        self.left -= 1;
        bit
    }
}

type Cache = Mutex<Vec<((bool, u64), Arc<NoiseTable>)>>;
static TABLES: OnceLock<Cache> = OnceLock::new();

/// The shared table for a width, built on first use.
pub(crate) fn table(width: Width, parallel: bool) -> Result<Arc<NoiseTable>> {
    let cache = TABLES.get_or_init(|| Mutex::new(Vec::new()));
    let mut tables = cache
        .lock()
        .map_err(|_| Error("noise table cache poisoned"))?;
    if let Some((_, table)) = tables.iter().find(|(key, _)| *key == width.key()) {
        return Ok(Arc::clone(table));
    }
    let table = Arc::new(NoiseTable::build(width, parallel)?);
    tables.push((width.key(), Arc::clone(&table)));
    Ok(table)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mpfr_operations_truncate_and_convert_exactly() {
        // 1/3 + 1/3 + 1/3 stays just below 1 because every step truncates.
        let third = recip(&Float::with_val(PRECISION, 3));
        let sum = add(&add(&third, &third), &third);
        assert!(sum < 1);
        assert!(mul_u64(&recip(&Float::with_val(PRECISION, 10)), 10) < 1);
        assert_eq!(div_u64(&Float::with_val(PRECISION, 7), 7), 1);
        let e = exp_negative(&Float::with_val(PRECISION, -1));
        assert!((e.to_f64() - std::f64::consts::E).abs() < 1e-15);
        let small = exp_negative(&Float::with_val(PRECISION, 30));
        assert!((small.to_f64() / (-30f64).exp() - 1.0).abs() < 1e-14);
        assert!(fixed256(&Float::with_val(PRECISION, 1)).is_none());
        assert_eq!(
            fixed256(&Float::with_val(PRECISION, 0.5)),
            Some([0, 0, 0, 1u64 << 63])
        );
        assert_eq!(fixed256(&Float::new(PRECISION)), Some([0; 4]));
        let tiny = ldexp(&Float::with_val(PRECISION, 3), -256);
        assert_eq!(fixed256(&tiny), Some([3, 0, 0, 0]));
        assert!(negligible(&tiny, &third, 200));
        assert!(!negligible(&third, &tiny, 200));
    }

    #[test]
    fn small_tables_match_double_precision_erf() {
        // erf from its Taylor series in f64, adequate to 1e-13 for x <= 3. Beyond
        // 6 it equals 1 in double precision.
        fn erf(x: f64) -> f64 {
            if x > 6.0 {
                return 1.0;
            }
            let mut term = x;
            let mut sum = x;
            for n in 1..200 {
                term *= -x * x / n as f64;
                let add = term / (2 * n + 1) as f64;
                sum += add;
                if add.abs() < 1e-18 {
                    break;
                }
            }
            sum * 2.0 / std::f64::consts::PI.sqrt()
        }
        for sd in [0.25, 0.7, 1.0, 3.0] {
            let table = NoiseTable::build(Width::Sd(sd), false).unwrap();
            let total = erf((table.bound() as f64 + 0.5) / (sd * std::f64::consts::SQRT_2));
            for k in 0..table.bound().min(12) {
                let x = (k as f64 + 0.5) / (sd * std::f64::consts::SQRT_2);
                if x > 3.0 {
                    break;
                }
                let expected = erf(x) / total;
                let got = table.entry(k)[0] as f64 / 2f64.powi(64);
                assert!(
                    (got - expected).abs() < 1e-12,
                    "sd {sd} k {k}: {got} vs {expected}"
                );
            }
            assert!(table.distance_log2() < -238.0);
        }
    }

    /// Returns scripted 64-bit words.
    struct Script(Vec<u64>, usize);
    impl rand::TryRng for Script {
        type Error = std::convert::Infallible;
        fn try_next_u32(&mut self) -> std::result::Result<u32, Self::Error> {
            Ok(self.try_next_u64()? as u32)
        }
        fn try_next_u64(&mut self) -> std::result::Result<u64, Self::Error> {
            self.1 += 1;
            Ok(self.0[self.1 - 1])
        }
        fn try_fill_bytes(&mut self, _: &mut [u8]) -> std::result::Result<(), Self::Error> {
            unreachable!("the sampler only draws words")
        }
    }

    /// A table whose entries share leading words, to exercise the comparison of
    /// the lower 192 bits. T = (5,0,0,1), (5,0,0,2), (5,3,0,0), (9,0,0,0), bound 4.
    fn crafted() -> NoiseTable {
        let hi = vec![5, 5, 5, 9];
        let lo = vec![[0, 0, 1], [0, 0, 2], [3, 0, 0], [0, 0, 0]];
        let mut guide = vec![4u32; (1 << GUIDE_BITS) + 1];
        guide[0] = 0;
        NoiseTable {
            sd: 1.0,
            variance: None,
            bound: 4,
            hi,
            lo,
            guide,
            distance_log2: 0.0,
            tail_log2: 0.0,
        }
    }

    #[test]
    fn ties_in_the_leading_word_compare_all_256_bits() {
        let table = crafted();
        // (u, lower words) -> first k with U < T_k, and the words drawn.
        let cases: [(u64, [u64; 3], usize, usize); 8] = [
            (4, [0, 0, 0], 0, 1),
            (5, [0, 0, 0], 0, 4),
            (5, [0, 0, 1], 1, 4),
            (5, [0, 0, 2], 2, 4),
            (5, [2, u64::MAX, u64::MAX], 2, 4),
            (5, [3, 0, 0], 3, 4),
            (9, [0, 0, 0], 4, 4),
            (10, [0, 0, 0], 4, 1),
        ];
        for (u, rest, expected, drawn) in cases {
            let mut rng = Script(vec![u, rest[0], rest[1], rest[2]], 0);
            assert_eq!(
                table.sample_magnitude(&mut rng),
                expected,
                "u {u} rest {rest:?}"
            );
            assert_eq!(rng.1, drawn);
        }
        // A window longer than eight entries takes the binary search.
        let mut long = crafted();
        long.hi = (0..20).map(|k| 10 * k + 10).collect();
        long.lo = vec![[0; 3]; 20];
        long.bound = 20;
        long.guide = vec![20; (1 << GUIDE_BITS) + 1];
        long.guide[0] = 0;
        for (u, expected) in [
            (0, 0),
            (9, 0),
            (10, 1),
            (11, 1),
            (105, 10),
            (199, 19),
            (200, 20),
        ] {
            let mut rng = Script(vec![u, 0, 0, 0], 0);
            assert_eq!(long.sample_magnitude(&mut rng), expected, "u {u}");
        }
    }

    /// 256-bit `x + 1` and `x - 1` on big-endian words, `None` on wraparound.
    fn successor(x: [u64; 4]) -> Option<[u64; 4]> {
        let mut y = x;
        for i in (0..4).rev() {
            let (v, carry) = y[i].overflowing_add(1);
            y[i] = v;
            if !carry {
                return Some(y);
            }
        }
        None
    }
    fn predecessor(x: [u64; 4]) -> Option<[u64; 4]> {
        let mut y = x;
        for i in (0..4).rev() {
            let (v, borrow) = y[i].overflowing_sub(1);
            y[i] = v;
            if !borrow {
                return Some(y);
            }
        }
        None
    }

    /// Checks `sample_magnitude` on the 256-bit input `u` (big-endian words)
    /// against a plain binary search over the full entries: the answer must be
    /// the number of entries `T_k <= u`, and the sampler must draw the lower
    /// three words exactly when the leading word of `u` equals a leading word
    /// of the table.
    fn check_inversion(table: &NoiseTable, u: [u64; 4]) -> bool {
        let (mut low, mut high) = (0usize, table.entries());
        while low < high {
            let mid = (low + high) / 2;
            if table.entry(mid) <= u {
                low = mid + 1;
            } else {
                high = mid;
            }
        }
        let mut rng = Script(u.to_vec(), 0);
        assert_eq!(table.sample_magnitude(&mut rng), low, "U = {u:016x?}");
        let tie = table.hi.binary_search(&u[0]).is_ok();
        assert_eq!(
            rng.1,
            if tie { 4 } else { 1 },
            "words drawn for U = {u:016x?}"
        );
        tie
    }

    /// Every entry from both sides, every guide bucket edge, the extremes and
    /// random inputs. Together with the table check against MPFR this shows that
    /// the sampler returns k with probability exactly (T_k - T_{k-1}) / 2^256.
    fn check_exact_inversion(table: &NoiseTable, random: usize) {
        let (mut inputs, mut ties) = (0usize, 0usize);
        let mut check = |u: [u64; 4]| {
            inputs += 1;
            ties += usize::from(check_inversion(table, u));
        };
        for k in 0..table.entries() {
            let t = table.entry(k);
            check(t);
            if let Some(u) = predecessor(t) {
                check(u);
            }
            if let Some(u) = successor(t) {
                check(u);
            }
        }
        for g in 0..1u64 << GUIDE_BITS {
            let first = g << (64 - GUIDE_BITS);
            let last = first | ((1u64 << (64 - GUIDE_BITS)) - 1);
            check([first, 0, 0, 0]);
            check([last, u64::MAX, u64::MAX, u64::MAX]);
        }
        let mut rng = rand::rngs::StdRng::seed_from_u64(0x5a3b1e);
        for _ in 0..random {
            check(std::array::from_fn(|_| rng.next_u64()));
        }
        let width = match table.variance {
            Some(v) => format!("variance {v}"),
            None => format!("sd {}", table.sd),
        };
        eprintln!(
            "{width}: {inputs} inputs agree with the 256-bit search ({ties} through the lower words), {} entries",
            table.entries()
        );
    }

    #[test]
    fn sampler_inverts_small_tables_exactly() {
        for width in [
            Width::Sd(0.25),
            Width::Sd(0.7),
            Width::Variance(1),
            Width::Variance(3),
            Width::Sd(77.5),
        ] {
            let table = NoiseTable::build(width, false).unwrap();
            check_exact_inversion(&table, 100_000);
        }
    }

    #[test]
    #[cfg_attr(debug_assertions, ignore = "large tables: run with --release")]
    fn sampler_inverts_the_production_table_exactly() {
        for width in [Width::Variance(1152 * 30358), Width::Sd(1000.0)] {
            let table = NoiseTable::build(width, true).unwrap();
            check_exact_inversion(&table, 2_000_000);
        }
    }

    #[test]
    fn an_exact_variance_equals_its_exact_square_root() {
        // sqrt(9) = 3 is a double, so both widths give the same table.
        let a = NoiseTable::build(Width::Variance(9), false).unwrap();
        let b = NoiseTable::build(Width::Sd(3.0), false).unwrap();
        assert_eq!((a.hi, a.lo, a.guide), (b.hi, b.lo, b.guide));
        assert_eq!((a.variance, b.variance), (Some(9), None));
        // sqrt(2) is not, and its nearest double gives a different table.
        let c = NoiseTable::build(Width::Variance(2), false).unwrap();
        let d = NoiseTable::build(Width::Sd(2f64.sqrt()), false).unwrap();
        assert_eq!(c.bound, d.bound);
        assert_ne!(c.lo, d.lo);
    }

    #[test]
    fn parallel_and_sequential_builds_agree() {
        let a = NoiseTable::build(Width::Sd(5913.748050094796), false).unwrap();
        let b = NoiseTable::build(Width::Sd(5913.748050094796), true).unwrap();
        assert_eq!(a.hi, b.hi);
        assert_eq!(a.lo, b.lo);
        assert_eq!(a.guide, b.guide);
        assert_eq!(a.bound(), 111_285);
        assert!(a.distance_log2() < -239.0);
    }
}
