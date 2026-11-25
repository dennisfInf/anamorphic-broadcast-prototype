//! The certified sampler for the encryption errors, as in the Python/Arb reference.
//!
//! All acceptance decisions use outward-rounded MPFR intervals. Ambiguous
//! decisions extend one retained uniform bit string, rather than restarting
//! or rounding its probability. Termination is almost sure in the ideal
//! independent-random-bit model. These research samplers are variable time.

use crate::arithmetic::map_ranges;
use crate::{Error, Result};
use rand::{Rng, TryRng};
use rug::float::Round;
use rug::ops::DivAssignRound;
use rug::{Float, Integer};
use statrs::distribution::{ContinuousCDF, Normal};
use std::convert::Infallible;

/// The coins of a sample certified on a worker thread: its first word, then
/// zero words, recording that more were asked for (the result is then discarded).
struct FirstWord {
    word: Option<u64>,
    exhausted: bool,
}

impl TryRng for FirstWord {
    type Error = Infallible;
    fn try_next_u32(&mut self) -> std::result::Result<u32, Infallible> {
        Ok(self.try_next_u64()? as u32)
    }
    fn try_next_u64(&mut self) -> std::result::Result<u64, Infallible> {
        Ok(self.word.take().unwrap_or_else(|| {
            self.exhausted = true;
            0
        }))
    }
    fn try_fill_bytes(&mut self, dst: &mut [u8]) -> std::result::Result<(), Infallible> {
        for chunk in dst.chunks_mut(8) {
            chunk.copy_from_slice(&self.try_next_u64()?.to_le_bytes()[..chunk.len()]);
        }
        Ok(())
    }
}

/// A first word already drawn, then the caller's generator.
struct Prefixed<'a, R: ?Sized> {
    word: Option<u64>,
    rest: &'a mut R,
}

impl<R: Rng + ?Sized> TryRng for Prefixed<'_, R> {
    type Error = Infallible;
    fn try_next_u32(&mut self) -> std::result::Result<u32, Infallible> {
        Ok(self.try_next_u64()? as u32)
    }
    fn try_next_u64(&mut self) -> std::result::Result<u64, Infallible> {
        Ok(self.word.take().unwrap_or_else(|| self.rest.next_u64()))
    }
    fn try_fill_bytes(&mut self, dst: &mut [u8]) -> std::result::Result<(), Infallible> {
        for chunk in dst.chunks_mut(8) {
            chunk.copy_from_slice(&self.try_next_u64()?.to_le_bytes()[..chunk.len()]);
        }
        Ok(())
    }
}

#[derive(Clone)]
struct Bounds {
    lo: Float,
    hi: Float,
}

/// A retained prefix of one uniform real in [0, 1].
struct UniformPrefix {
    prefix: Integer,
    bits: u32,
}

impl UniformPrefix {
    fn from_word(word: u64) -> Self {
        Self {
            prefix: Integer::from(word),
            bits: 64,
        }
    }

    fn extend<R: Rng + ?Sized>(&mut self, rng: &mut R) -> Result<()> {
        let extra = self.bits.max(64);
        self.bits = self
            .bits
            .checked_add(extra)
            .ok_or(Error("uniform precision exhausted"))?;
        for _ in 0..extra / 64 {
            self.prefix <<= 64;
            self.prefix += rng.next_u64();
        }
        Ok(())
    }

    fn bounds(&self, precision: u32) -> Bounds {
        // precision exceeds bits, so these dyadic endpoints are exact.
        let mut lo = Float::with_val(precision, &self.prefix);
        let mut upper = self.prefix.clone();
        upper += 1;
        let mut hi = Float::with_val(precision, upper);
        lo >>= self.bits;
        hi >>= self.bits;
        Bounds { lo, hi }
    }
}

/// Exact integer rounding floor(N(0, (q/denominator)²) + 1/2).
///
/// Integer outputs have unbounded support. A floating-point inverse normal CDF
/// (`statrs`) only suggests a starting point for the certified search; it is
/// never used to accept an output, set a tail cutoff, or draw a sample.
pub struct ErrorSampler {
    q: u64,
    denominator: u64,
    refinements: u64,
}

impl ErrorSampler {
    pub fn new(q: u64, denominator: u64) -> Result<Self> {
        if q == 0 || denominator == 0 {
            return Err(Error(
                "positive Gaussian numerator and denominator required",
            ));
        }
        Ok(Self {
            q,
            denominator,
            refinements: 0,
        })
    }

    pub fn refinements(&self) -> u64 {
        self.refinements
    }

    pub fn sample<R: Rng + ?Sized>(&mut self, rng: &mut R) -> Result<Integer> {
        let word = rng.next_u64();
        // Work with the lower tail so conversion to f64 never rounds p to 1.
        let lower_word = word.min(u64::MAX - word);
        let p = (lower_word as f64 + 0.5) / 18446744073709551616.0;
        let z = Normal::standard().inverse_cdf(p);
        let z = if word < 1 << 63 { z } else { -z };
        let guess = (z * (self.q as f64 / self.denominator as f64) + 0.5).floor();
        let candidate = Float::with_val(64, guess)
            .to_integer_round(Round::Down)
            .ok_or(Error("nonfinite Gaussian search hint"))?
            .0;
        self.search(UniformPrefix::from_word(word), candidate, rng)
    }

    /// `count` samples, the same as `count` calls of `sample` whenever no sample
    /// needs more than its first 64-bit word (as for all but a fraction of about
    /// 2^-55 of the samples at the Epoque profiles' widths). The first words are
    /// drawn in order on the calling thread, and the samples are certified from
    /// them on up to `threads()` threads. A sample that needs further words is
    /// certified again afterwards on the calling thread, in order, from its first
    /// word followed by fresh words of `rng`. Every sample still uses its own
    /// independent uniform bits, so the output distribution is exact; the result
    /// does not depend on the thread count.
    pub fn sample_many<R: Rng + ?Sized>(&mut self, count: usize, rng: &mut R) -> Result<Vec<Integer>> {
        let mut words = Vec::new();
        words
            .try_reserve_exact(count)
            .map_err(|_| Error("allocation failed"))?;
        words.extend((0..count).map(|_| rng.next_u64()));
        let (q, denominator) = (self.q, self.denominator);
        let parts = map_ranges(count, 1 << 14, |range| {
            let mut sampler = ErrorSampler {
                q,
                denominator,
                refinements: 0,
            };
            words[range]
                .iter()
                .map(|&word| {
                    let mut coins = FirstWord {
                        word: Some(word),
                        exhausted: false,
                    };
                    let value = sampler.sample(&mut coins);
                    // A sample that asked for more than its first word is redone below.
                    value.ok().filter(|_| !coins.exhausted)
                })
                .collect::<Vec<_>>()
        });
        let mut out = Vec::new();
        out.try_reserve_exact(count)
            .map_err(|_| Error("allocation failed"))?;
        for (value, &word) in parts.into_iter().flatten().zip(&words) {
            out.push(match value {
                Some(value) => value,
                None => self.sample(&mut Prefixed {
                    word: Some(word),
                    rest: &mut *rng,
                })?,
            });
        }
        Ok(out)
    }

    fn search<R: Rng + ?Sized>(
        &mut self,
        mut uniform: UniformPrefix,
        candidate: Integer,
        rng: &mut R,
    ) -> Result<Integer> {
        let (mut lo, mut hi);
        let mut step = Integer::from(1);
        if self.below_cdf(&mut uniform, &candidate, rng)? {
            hi = candidate;
            loop {
                lo = Integer::from(&hi - &step);
                if !self.below_cdf(&mut uniform, &lo, rng)? {
                    break;
                }
                hi = lo;
                step <<= 1;
            }
        } else {
            lo = candidate;
            loop {
                hi = Integer::from(&lo + &step);
                if self.below_cdf(&mut uniform, &hi, rng)? {
                    break;
                }
                lo = hi;
                step <<= 1;
            }
        }
        loop {
            let mut gap = Integer::from(&hi - &lo);
            if gap == 1 {
                return Ok(hi);
            }
            gap >>= 1;
            let mid = &lo + gap;
            if self.below_cdf(&mut uniform, &mid, rng)? {
                hi = mid;
            } else {
                lo = mid;
            }
        }
    }

    // Certify U < Phi((k+1/2)/(q/denominator)).
    fn below_cdf<R: Rng + ?Sized>(
        &mut self,
        uniform: &mut UniformPrefix,
        k: &Integer,
        rng: &mut R,
    ) -> Result<bool> {
        loop {
            let precision = uniform
                .bits
                .checked_add(64)
                .ok_or(Error("error CDF precision exhausted"))?;
            let u = uniform.bounds(precision);
            let cdf = self.cdf(k, precision);
            if u.hi < cdf.lo {
                return Ok(true);
            }
            if u.lo > cdf.hi {
                return Ok(false);
            }
            uniform.extend(rng)?;
            self.refinements += 1;
        }
    }

    fn cdf(&self, k: &Integer, precision: u32) -> Bounds {
        // Phi((k+1/2)/sd) = erfc(-(2k+1)*denominator/(2q*sqrt(2)))/2.
        // All integer arithmetic is exact, including arbitrarily large k.
        let mut numerator = k.clone();
        numerator <<= 1;
        numerator += 1;
        numerator *= self.denominator;
        numerator = -numerator;
        let mut divisor = Integer::from(self.q);
        divisor <<= 1;
        let mut x_lo = Float::with_val_round(precision, &numerator, Round::Down).0;
        let mut x_hi = Float::with_val_round(precision, &numerator, Round::Up).0;
        x_lo.div_assign_round(&divisor, Round::Down);
        x_hi.div_assign_round(&divisor, Round::Up);
        let mut sqrt_lo = Float::with_val(precision, 2);
        let mut sqrt_hi = sqrt_lo.clone();
        sqrt_lo.sqrt_round(Round::Down);
        sqrt_hi.sqrt_round(Round::Up);
        if numerator > 0 {
            x_lo.div_assign_round(&sqrt_hi, Round::Down);
            x_hi.div_assign_round(&sqrt_lo, Round::Up);
        } else {
            x_lo.div_assign_round(&sqrt_lo, Round::Down);
            x_hi.div_assign_round(&sqrt_hi, Round::Up);
        }
        // erfc is decreasing, so reverse endpoints again.
        let mut lo = x_hi;
        let mut hi = x_lo;
        lo.erfc_round(Round::Down);
        hi.erfc_round(Round::Up);
        lo >>= 1;
        hi >>= 1;
        Bounds { lo, hi }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::arithmetic::with_threads;
    use rand::{SeedableRng, rngs::StdRng};
    use rug::ops::AddAssignRound;

    #[test]
    fn invalid_widths_are_rejected() {
        assert!(ErrorSampler::new(0, 1).is_err());
        assert!(ErrorSampler::new(1, 0).is_err());
    }

    #[test]
    fn error_search_is_independent_of_starting_hint() {
        let mut sampler = ErrorSampler::new(65537, 66560).unwrap();
        for word in [
            0,
            1,
            1 << 62,
            (1 << 63) - 1,
            1 << 63,
            u64::MAX - 1,
            u64::MAX,
        ] {
            let mut rng1 = StdRng::seed_from_u64(word);
            let mut rng2 = StdRng::seed_from_u64(word);
            let a = sampler
                .search(UniformPrefix::from_word(word), Integer::from(0), &mut rng1)
                .unwrap();
            let b = sampler
                .search(
                    UniformPrefix::from_word(word),
                    Integer::from(10000),
                    &mut rng2,
                )
                .unwrap();
            assert_eq!(a, b);
        }
    }

    #[test]
    fn error_boundary_refinement_and_unbounded_tail() {
        let mut sampler = ErrorSampler::new(65537, 66560).unwrap();
        let cdf = sampler.cdf(&Integer::from(0), 256);
        let mut p_lo = cdf.lo;
        let mut p_hi = cdf.hi;
        p_lo <<= 64;
        p_hi <<= 64;
        let word = p_lo.to_integer_round(Round::Down).unwrap().0;
        assert_eq!(word, p_hi.to_integer_round(Round::Down).unwrap().0);
        let mut rng = StdRng::seed_from_u64(51);
        let answer = sampler
            .search(
                UniformPrefix::from_word(word.to_u64().unwrap()),
                Integer::from(0),
                &mut rng,
            )
            .unwrap();
        assert!(answer == 0 || answer == 1);
        assert!(sampler.refinements > 0);
        // U near 2^-1024 deliberately exercises a tail beyond the initial
        // 64-bit prefix and a result beyond any 12-standard-deviation cutoff.
        let tiny = UniformPrefix {
            prefix: Integer::from(1),
            bits: 1024,
        };
        let tail = sampler.search(tiny, Integer::from(0), &mut rng).unwrap();
        assert!(tail < -30);
    }

    /// Words in a fixed order, then a panic (a sampler asked for too many).
    struct Words(Vec<u64>, usize);
    impl TryRng for Words {
        type Error = Infallible;
        fn try_next_u32(&mut self) -> std::result::Result<u32, Infallible> {
            Ok(self.try_next_u64()? as u32)
        }
        fn try_next_u64(&mut self) -> std::result::Result<u64, Infallible> {
            self.1 += 1;
            Ok(self.0[self.1 - 1])
        }
        fn try_fill_bytes(&mut self, _: &mut [u8]) -> std::result::Result<(), Infallible> {
            unreachable!("the sampler draws words")
        }
    }

    #[test]
    fn batched_samples_match_one_by_one_sampling_for_every_thread_count() {
        // 600 normals at the full32 width: enough to be split over the threads.
        let (q, denominator) = (220_301_035_189, 65 * 112_566_231);
        let mut one_by_one = ErrorSampler::new(q, denominator).unwrap();
        let mut rng = StdRng::seed_from_u64(7);
        let expected: Vec<Integer> = (0..600)
            .map(|_| one_by_one.sample(&mut rng).unwrap())
            .collect();
        let next = rng.next_u64();
        for threads in [1, 2, 3, 7] {
            let mut sampler = ErrorSampler::new(q, denominator).unwrap();
            let mut rng = StdRng::seed_from_u64(7);
            let batch = with_threads(threads, || sampler.sample_many(600, &mut rng)).unwrap();
            assert_eq!(batch, expected, "{threads} threads");
            assert_eq!(rng.next_u64(), next, "{threads} threads: coins used");
        }
    }

    #[test]
    fn a_batched_sample_that_needs_more_words_takes_them_after_all_first_words() {
        let mut reference = ErrorSampler::new(65537, 66560).unwrap();
        // A first word whose 64-bit interval contains the boundary between 0 and 1.
        let cdf = reference.cdf(&Integer::from(0), 256);
        let mut p = cdf.lo;
        p <<= 64;
        let boundary = p.to_integer_round(Round::Down).unwrap().0.to_u64().unwrap();
        let mut seeded = StdRng::seed_from_u64(9);
        let mut first: Vec<u64> = (0..400).map(|_| seeded.next_u64()).collect();
        first[150] = boundary;
        let further: Vec<u64> = (0..64).map(|_| seeded.next_u64()).collect();
        // Every sample from its own first word; the boundary one continues with `further`.
        let expected: Vec<Integer> = first
            .iter()
            .map(|&word| {
                let tail = if word == boundary { &further[..] } else { &further[..0] };
                let mut coins = Words([&[word][..], tail].concat(), 0);
                reference.sample(&mut coins).unwrap()
            })
            .collect();
        assert!(reference.refinements() > 0);
        let mut coins = Words([&[boundary][..], &further].concat(), 0);
        ErrorSampler::new(65537, 66560).unwrap().sample(&mut coins).unwrap();
        let used = 400 + coins.1 - 1;
        for threads in [1, 3] {
            let mut sampler = ErrorSampler::new(65537, 66560).unwrap();
            let mut stream = Words([&first[..], &further].concat(), 0);
            let batch = with_threads(threads, || sampler.sample_many(400, &mut stream)).unwrap();
            assert_eq!(batch, expected, "{threads} threads");
            assert_eq!(stream.1, used, "{threads} threads: words used");
            assert!(sampler.refinements() > 0);
        }
    }

    #[test]
    fn normal_cdf_has_correct_symmetry_and_monotonicity() {
        let sampler = ErrorSampler::new(178282599284737, 5256875).unwrap();
        let left = sampler.cdf(&Integer::from(-1), 128);
        let right = sampler.cdf(&Integer::from(0), 128);
        assert!(left.hi < 0.5 && right.lo > 0.5);
        let mut sum_lo = left.lo;
        let mut sum_hi = left.hi;
        sum_lo.add_assign_round(right.lo, Round::Down);
        sum_hi.add_assign_round(right.hi, Round::Up);
        assert!(sum_lo <= 1 && sum_hi >= 1);
    }
}
