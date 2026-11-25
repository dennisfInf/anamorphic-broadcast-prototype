// SPDX-License-Identifier: MPL-2.0
//! Reference Gaussian sampler for the joint parameter profile.
//!
//! Targets D_{Z,s}(x) proportional to exp(-pi*(x/s)^2), with integer power-of-two
//! s, truncation at 8s, and downward-certified 256-bit acceptance probabilities.
//! The width s is a Gaussian parameter, not its standard deviation. Small widths
//! use an integer CDF; large widths use rejection with the same integer masses.
//!
//! This is variable-time research code. The distance bound is in the package
//! file `security/sampler-analysis.txt`; the sampler is not side-channel hardened.
use crate::dgs::DGS;
use rand::{
    Rng,
    distr::{Distribution, Uniform},
    rngs::ThreadRng,
};
use rug::{
    Float, Integer,
    float::{Constant, Round},
    integer::Order,
};

const PRECISION: u32 = 384;
const PROB_BITS: u32 = 256;

// The zero point has mass 2^256, which does not fit in four limbs.
#[derive(Clone, Debug, PartialEq, Eq)]
enum Threshold {
    Always,
    Fixed([u64; 4]),
}

// The sum of the 256-bit masses needs a fifth limb.
#[derive(Clone, Debug)]
struct IntegerCdf {
    cumulative: Vec<[u64; 5]>,
    total: [u64; 5],
    high_mask: u64,
}

fn limbs5(value: &Integer) -> [u64; 5] {
    let digits = value.to_digits::<u64>(Order::Lsf);
    assert!(digits.len() <= 5);
    let mut limbs = [0; 5];
    limbs[..digits.len()].copy_from_slice(&digits);
    limbs
}

fn less5(a: &[u64; 5], b: &[u64; 5]) -> bool {
    for j in (0..5).rev() {
        if a[j] < b[j] {
            return true;
        }
        if a[j] > b[j] {
            return false;
        }
    }
    false
}

#[derive(Clone, Debug)]
pub struct JointGaussian {
    width: u64,
    modulus: u64,
    cutoff: u64,
    proposal: Uniform<u64>,
    cdf: Option<IntegerCdf>,
    pi_low: Float,
    pi_high: Float,
}

impl JointGaussian {
    fn bounds(&self, abs_x: u64) -> (Integer, Integer) {
        // x^2 and division by s^2 are exact at this precision for the accepted
        // parameter domain. Only pi, its product, and exp need rounding bounds.
        let square = Integer::from(abs_x) * Integer::from(abs_x);
        let mut ratio = Float::with_val(PRECISION, square);
        ratio >>= 2 * self.width.trailing_zeros();
        let mut low = -Float::with_val_round(PRECISION, &ratio * &self.pi_high, Round::Up).0;
        let mut high = -Float::with_val_round(PRECISION, &ratio * &self.pi_low, Round::Down).0;
        low.exp_round(Round::Down);
        high.exp_round(Round::Up);
        low <<= PROB_BITS;
        high <<= PROB_BITS;
        (
            low.to_integer_round(Round::Down).unwrap().0,
            high.to_integer_round(Round::Down).unwrap().0,
        )
    }

    fn threshold(&self, abs_x: u64) -> Threshold {
        if abs_x == 0 {
            return Threshold::Always;
        }
        let (low, high) = self.bounds(abs_x);
        // If the true scaled mass straddles an integer, using the lower floor
        // loses less than two units, already charged in the distance bound.
        assert!(
            high >= low && Integer::from(&high - &low) <= 1,
            "Gaussian acceptance interval was not certified"
        );
        let digits = low.to_digits::<u64>(Order::Lsf);
        assert!(digits.len() <= 4);
        let mut limbs = [0; 4];
        limbs[..digits.len()].copy_from_slice(&digits);
        Threshold::Fixed(limbs)
    }

    fn accepts(threshold: &Threshold, words: [u64; 4]) -> bool {
        match threshold {
            Threshold::Always => true,
            Threshold::Fixed(t) => {
                for j in (0..4).rev() {
                    if words[j] < t[j] {
                        return true;
                    }
                    if words[j] > t[j] {
                        return false;
                    }
                }
                false
            }
        }
    }

    /// Returns an integer in [-8s, 8s] under the certified truncated law.
    /// A failed proposal is resampled internally; it is not a decoding failure.
    pub fn sample_signed(&self, rng: &mut ThreadRng) -> i64 {
        if let Some(cdf) = &self.cdf {
            // Exact sampling from the integer weights used by rejection below.
            // Rejection against the total avoids any modulo bias.
            loop {
                let mut value = [
                    rng.next_u64(),
                    rng.next_u64(),
                    rng.next_u64(),
                    rng.next_u64(),
                    rng.next_u64(),
                ];
                value[4] &= cdf.high_mask;
                if !less5(&value, &cdf.total) {
                    continue;
                }
                let index = cdf.cumulative.partition_point(|c| !less5(&value, c));
                return index as i64 - self.cutoff as i64;
            }
        }
        loop {
            let proposed = self.proposal.sample(rng) as i64 - self.cutoff as i64;
            let abs_x = proposed.unsigned_abs();
            let threshold = self.threshold(abs_x);
            if threshold == Threshold::Always {
                return proposed;
            }
            let words = [
                rng.next_u64(),
                rng.next_u64(),
                rng.next_u64(),
                rng.next_u64(),
            ];
            if Self::accepts(&threshold, words) {
                return proposed;
            }
        }
    }
}

impl DGS for JointGaussian {
    /// Requires an integer power-of-two width and an unambiguous centered lift.
    /// Panics if these requirements or a probability-interval check fail.
    fn init(s: f64, mod_q: u64) -> Self {
        assert!(s.is_finite() && s >= 16.0 && s <= (1u64 << 48) as f64);
        let width = s as u64;
        assert_eq!(width as f64, s);
        assert!(
            width.is_power_of_two(),
            "profile Gaussian width must be a power of two"
        );
        let cutoff = 8 * width;
        assert!(
            cutoff < mod_q / 2,
            "truncation must have an unambiguous centered lift"
        );
        let mut this = Self {
            width,
            modulus: mod_q,
            cutoff,
            proposal: Uniform::new_inclusive(0, 2 * cutoff).unwrap(),
            cdf: None,
            pi_low: Float::with_val_round(PRECISION, Constant::Pi, Round::Down).0,
            pi_high: Float::with_val_round(PRECISION, Constant::Pi, Round::Up).0,
        };
        if cutoff <= 4096 {
            let table: Vec<_> = (0..=cutoff).map(|x| this.threshold(x)).collect();
            let mut sum = Integer::from(0);
            let mut cumulative = Vec::new();
            for x in -(cutoff as i64)..=cutoff as i64 {
                let mass = match &table[x.unsigned_abs() as usize] {
                    Threshold::Always => Integer::from(1) << PROB_BITS,
                    Threshold::Fixed(words) => Integer::from_digits(words, Order::Lsf),
                };
                sum += mass;
                cumulative.push(limbs5(&sum));
            }
            let high_bits = sum.significant_bits() - 256;
            assert!(high_bits > 0 && high_bits < 64);
            this.cdf = Some(IntegerCdf {
                cumulative,
                total: limbs5(&sum),
                high_mask: (1u64 << high_bits) - 1,
            });
        }
        this
    }

    fn sample(&self, rng: &mut ThreadRng) -> u64 {
        let value = self.sample_signed(rng);
        if value < 0 {
            self.modulus - value.unsigned_abs()
        } else {
            value as u64
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    const Q: u64 = 281_474_976_694_273;

    #[test]
    fn probability_thresholds_are_certified_and_symmetric() {
        for width in [16u64, 1 << 36] {
            let sampler = JointGaussian::init(width as f64, Q);
            let points: Vec<u64> = if width == 16 {
                (1..=8 * width).collect()
            } else {
                vec![
                    1,
                    width / 4,
                    width / 2,
                    width,
                    2 * width,
                    4 * width,
                    8 * width,
                ]
            };
            for x in points {
                let (lo, hi) = sampler.bounds(x);
                assert!(Integer::from(&hi - &lo) <= 1);
                assert_eq!(
                    sampler.threshold(x),
                    sampler.threshold((-(x as i64)).unsigned_abs())
                );
            }
        }
    }

    #[test]
    fn strict_comparison_handles_zero_and_equality() {
        assert!(!JointGaussian::accepts(&Threshold::Fixed([0; 4]), [0; 4]));
        assert!(!JointGaussian::accepts(
            &Threshold::Fixed([1, 0, 0, 0]),
            [1, 0, 0, 0]
        ));
        assert!(JointGaussian::accepts(
            &Threshold::Fixed([1, 0, 0, 0]),
            [0; 4]
        ));
        assert!(JointGaussian::accepts(&Threshold::Always, [u64::MAX; 4]));
    }

    #[test]
    fn samples_are_centered_lifts_and_within_support() {
        for width in [16u64, 1 << 36] {
            let sampler = JointGaussian::init(width as f64, Q);
            for _ in 0..128 {
                let x = sampler.sample(&mut rand::rng());
                assert!(x < Q && x.min(Q - x) <= 8 * width);
            }
        }
    }
}
