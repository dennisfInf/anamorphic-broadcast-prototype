//! Arithmetic modulo an odd modulus q < 2^32, delegated to `crypto-bigint`.
//!
//! Values are kept canonical in `u32`. Every reduction, modular addition,
//! subtraction, negation and inversion is a `crypto-bigint` operation:
//! remainders use its precomputed reciprocal of q (division by an invariant
//! single-limb divisor), sums and differences use `add_mod`, `sub_mod` and
//! `neg_mod`, and inverses use its constant-time `invert_odd_mod`. A product of
//! two canonical values is below 2^64 because q < 2^32, so it is formed exactly
//! in `u64` and then reduced.
//!
//! The kernels accumulate exact integer sums and pass them to [`Field::reduce`]
//! or [`Field::reduce_wide`]. [`Field::products_fit_u64`] decides once per call
//! whether a `u64` accumulator is wide enough, otherwise the kernels use `u128`.

use crypto_bigint::{Limb, NonZero, Odd, Reciprocal, U64, U128};

/// The modulus and the `crypto-bigint` data derived from it once.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Field {
    q: u32,
    modulus: NonZero<U64>,
    odd: Odd<U64>,
    reciprocal: Reciprocal,
}

impl Field {
    /// `q` must be odd and at least 3. Inversion additionally needs q prime,
    /// which the public constructors check.
    pub(crate) fn new(q: u32) -> Self {
        assert!(q >= 3 && q % 2 == 1, "modulus must be odd and at least 3");
        let wide = U64::from_u32(q);
        Self {
            q,
            modulus: NonZero::new(wide).expect("q is nonzero"),
            odd: Odd::new(wide).expect("q is odd"),
            reciprocal: Reciprocal::new(NonZero::new(Limb::from_u32(q)).expect("q is nonzero")),
        }
    }

    pub(crate) fn q(&self) -> u32 {
        self.q
    }

    /// Whether `count` products of canonical values, plus one more canonical
    /// value, always fit in a `u64`. Then `u64` accumulators are exact.
    pub(crate) fn products_fit_u64(&self, count: usize) -> bool {
        let top = u64::from(self.q - 1);
        u64::try_from(count)
            .ok()
            .and_then(|count| (top * top).checked_mul(count))
            .and_then(|sum| sum.checked_add(top))
            .is_some()
    }

    /// Whether `count` canonical values always sum to at most `u64::MAX`.
    pub(crate) fn values_fit_u64(&self, count: usize) -> bool {
        u64::try_from(count)
            .ok()
            .and_then(|count| u64::from(self.q - 1).checked_mul(count))
            .is_some()
    }

    fn canonical(remainder: Limb) -> u32 {
        u32::try_from(remainder.0).expect("a remainder modulo q < 2^32 fits in u32")
    }

    fn element(&self, a: u32) -> U64 {
        debug_assert!(a < self.q, "field elements are canonical");
        U64::from_u32(a)
    }

    fn value(x: U64) -> u32 {
        u32::try_from(u64::from(x)).expect("a canonical residue fits in u32")
    }

    /// `x mod q`.
    #[inline]
    pub(crate) fn reduce(&self, x: u64) -> u32 {
        Self::canonical(U64::from_u64(x).rem_limb_with_reciprocal(&self.reciprocal))
    }

    /// `x mod q` for a 128-bit sum.
    #[inline]
    pub(crate) fn reduce_wide(&self, x: u128) -> u32 {
        Self::canonical(U128::from_u128(x).rem_limb_with_reciprocal(&self.reciprocal))
    }

    #[inline]
    pub(crate) fn add(&self, a: u32, b: u32) -> u32 {
        Self::value(self.element(a).add_mod(&self.element(b), &self.modulus))
    }

    #[inline]
    pub(crate) fn sub(&self, a: u32, b: u32) -> u32 {
        Self::value(self.element(a).sub_mod(&self.element(b), &self.modulus))
    }

    #[inline]
    pub(crate) fn neg(&self, a: u32) -> u32 {
        Self::value(self.element(a).neg_mod(&self.modulus))
    }

    /// `a b mod q`: the product is exact in `u64` since a, b < q < 2^32.
    #[inline]
    pub(crate) fn mul(&self, a: u32, b: u32) -> u32 {
        debug_assert!(a < self.q && b < self.q, "field elements are canonical");
        self.reduce(u64::from(a) * u64::from(b))
    }

    /// `a b + c mod q`: the value is exact in `u64` since it is at most
    /// `(q - 1)^2 + q - 1 < 2^64`.
    #[inline]
    pub(crate) fn mul_add(&self, a: u32, b: u32, c: u32) -> u32 {
        debug_assert!(a < self.q && b < self.q && c < self.q, "canonical");
        self.reduce(u64::from(a) * u64::from(b) + u64::from(c))
    }

    /// The inverse of a nonzero element, for prime q.
    pub(crate) fn inverse(&self, a: u32) -> u32 {
        Self::value(
            self.element(a)
                .invert_odd_mod(&self.odd)
                .expect("nonzero elements of a prime field are invertible"),
        )
    }

    /// The residue of a signed integer.
    #[inline]
    pub(crate) fn residue(&self, e: i64) -> u32 {
        let magnitude = self.reduce(e.unsigned_abs());
        if e < 0 {
            self.neg(magnitude)
        } else {
            magnitude
        }
    }
}

#[cfg(test)]
mod tests {
    use super::Field;

    const MODULI: [u32; 6] = [3, 257, 12289, 67537879, 2147483647, 4294967291];

    #[test]
    fn accumulator_bounds_are_tight() {
        for q in MODULI {
            let f = Field::new(q);
            let top = u128::from(q - 1);
            // The largest count accepted by products_fit_u64.
            let k = (u128::from(u64::MAX) - top) / (top * top);
            let k = usize::try_from(k).unwrap_or(usize::MAX);
            assert!(f.products_fit_u64(k));
            if k < usize::MAX {
                assert!(!f.products_fit_u64(k + 1));
            }
            let l = usize::try_from(u128::from(u64::MAX) / top).unwrap_or(usize::MAX);
            assert!(f.values_fit_u64(l));
            if l < usize::MAX {
                assert!(!f.values_fit_u64(l + 1));
            }
        }
        // The maintained profile: 1152 products per header coordinate fit in u64.
        assert!(Field::new(67537879).products_fit_u64(4044));
        assert!(!Field::new(67537879).products_fit_u64(4045));
        assert!(!Field::new(4294967291).products_fit_u64(2));
    }

    #[test]
    fn field_operations_match_wide_arithmetic() {
        for q in MODULI {
            let f = Field::new(q);
            let samples = [0, 1, 2, q / 2, q - 2, q - 1];
            for &a in &samples {
                for &b in &samples {
                    let (wa, wb, wq) = (i128::from(a), i128::from(b), i128::from(q));
                    assert_eq!(i128::from(f.add(a, b)), (wa + wb).rem_euclid(wq));
                    assert_eq!(i128::from(f.sub(a, b)), (wa - wb).rem_euclid(wq));
                    assert_eq!(i128::from(f.mul(a, b)), (wa * wb).rem_euclid(wq));
                }
                assert_eq!(f.add(a, f.neg(a)), 0);
                if a != 0 {
                    assert_eq!(f.mul(a, f.inverse(a)), 1);
                }
            }
            for e in [
                i64::MIN,
                i64::MIN + 1,
                -(i64::from(q)) - 1,
                -(i64::from(q)),
                -1,
                0,
                1,
                i64::from(q) - 1,
                i64::from(q),
                i64::MAX,
            ] {
                let expected = i128::from(e).rem_euclid(i128::from(q)) as u32;
                assert_eq!(f.residue(e), expected, "q {q} e {e}");
            }
            for x in [
                0u64,
                1,
                u64::from(q) - 1,
                u64::from(q),
                u64::from(q) * 2 + 3,
                u64::MAX - 1,
                u64::MAX,
            ] {
                assert_eq!(u64::from(f.reduce(x)), x % u64::from(q));
            }
            for x in [
                0u128,
                u128::from(u64::MAX) + 1,
                u128::from(q) << 64,
                (u128::from(q) << 70) - 1,
                u128::MAX - 1,
                u128::MAX,
            ] {
                assert_eq!(u128::from(f.reduce_wide(x)), x % u128::from(q));
            }
        }
    }

    #[test]
    fn random_operations_match_wide_arithmetic() {
        use rand::{RngExt, SeedableRng, rngs::StdRng};
        let mut rng = StdRng::seed_from_u64(11);
        for q in MODULI {
            let f = Field::new(q);
            for _ in 0..20_000 {
                let (a, b) = (rng.random_range(0..q), rng.random_range(0..q));
                let x: u64 = rng.random();
                let w: u128 = rng.random();
                let e: i64 = rng.random();
                assert_eq!(
                    u64::from(f.mul(a, b)),
                    u64::from(a) * u64::from(b) % u64::from(q)
                );
                assert_eq!(u64::from(f.reduce(x)), x % u64::from(q));
                assert_eq!(u128::from(f.reduce_wide(w)), w % u128::from(q));
                let c = rng.random_range(0..q);
                assert_eq!(
                    u128::from(f.mul_add(a, b, c)),
                    (u128::from(a) * u128::from(b) + u128::from(c)) % u128::from(q)
                );
                assert_eq!(
                    i128::from(f.residue(e)),
                    i128::from(e).rem_euclid(i128::from(q))
                );
                if a != 0 {
                    assert_eq!(f.mul(a, f.inverse(a)), 1);
                }
            }
        }
    }
}
