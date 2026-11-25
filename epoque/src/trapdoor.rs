//! Epoque key generation with lattice trapdoors (Boyen-Haines-Mueller, Section 5).
//!
//! `setup_trapdoor` is Epoque's KeyGen (prm of eq. 20, msk = T_A0 of eqs. 21-22,
//! T_B of eq. 28) with the master shortcut key of Section 5.2.1 (eq. 29);
//! `MasterTrapdoor::extract` is Extr (eq. 23) for any identity id in Z_q^n, with
//! the full-rank-difference matrix H(id) of `crate::identity`.
//!
//! - (A0, T_A0) and (B, T_B) come from TrapGen; A1 and U = [u_1 .. u_J] are uniform.
//! - d2_j <- SamplePre(B, T_B, 0) and d1_j <- SamplePre(A0, T_A0, u_j - A1 d2_j)
//!   (the paper prints the target as -A1 d2; eq. 29 needs u - A1 d2). Then
//!   B d2_j = 0 and A0 d1_j + (A1 + H(id) B) d2_j = u_j for every identity, so
//!   (D1, D2) is the master shortcut key used for ordinary decryption.
//! - Extr samples e2_j from D_{Z^m,s} and e1_j <- SamplePre(A0, T_A0,
//!   u_j - (A1 + H(id) B) e2_j) (SampleLeft), so [A0 | A1 + H(id) B] (e1_j; e2_j) = u_j.
//!
//! TrapGen is the gadget trapdoor of Micciancio and Peikert (EUROCRYPT 2012):
//! A = [Abar | P | G - Abar R] with Abar (n by 3n) and P uniform, the gadget
//! G = I_n (x) (1, b, .., b^(k-1)) with b^k > q, and R uniform in {-1,0,1}^(3n by nk),
//! so A [R; 0; I] = G. There is no identity block: Epoque's secret s is uniform,
//! and A0 must be indistinguishable from uniform. Writing Abar = [A1' | A2'] with A1'
//! invertible (n by n), each column of A1'^-1 Abar R is r1 + M r2 with M uniform: LWE
//! with a ternary secret of dimension 2n, ternary error and only n samples. Width 3n
//! makes this instance underdetermined (about 380 bits core-SVP for the full profile,
//! against about 140 for width 2n). Whoever recovers R holds the master key.
//! SamplePre draws a perturbation p with parameter sqrt(s^2 I - sigma_G^2 T T^T),
//! T = [R; 0; I] (Peikert's convolution: a continuous Gaussian, then randomized
//! rounding of width r), samples z in the coset of Lambda^perp(G) with Klein's
//! algorithm on the basis S_k, and returns x = p + T z with A x = target. It
//! follows MP12's SampleD; the output is a discrete Gaussian of parameter s on
//! the coset. These samplers use `f64`, as practical trapdoor implementations
//! do. The Box-Muller radius comes from a 128-bit uniform, so no Gaussian tail
//! beyond probability 2^-128 is cut off. The remaining deviations are rounding
//! errors of relative size about 2^-50 per sample, which by Renyi-divergence
//! arguments change the probability of any event (such as recovering the trapdoor
//! from the exposed keys) only negligibly. Every preimage is checked exactly, and
//! the encryption errors still come from the certified sampler of `crate::sampling`.
//!
//! Security boundary: whoever holds T_A0 can invert c1[1..m] = A0^T s + y
//! (`MasterTrapdoor::recover_secret`) and so read every hidden constraint
//! H_S^T s = floor(q/2) am. The extraction master key must never reach the party
//! from whom the anamorphic message is hidden. T_B is erased after setup.
use crate::arithmetic::{for_each_row_block, transpose_matvec, uniform, wide_right_product};
use crate::{
    Ciphertext, Error, Identity, Profile, PublicKey, Result, Setup, ShortcutKey, decode_residual,
    key_residual,
};
use rand::{Rng, RngExt};
use std::f64::consts::PI;

/// Security parameter of Construction 1's leftover-hash condition.
const LAMBDA: usize = 128;
/// The rounding width is the smoothing parameter of Z^m for epsilon = 2^-128.
const SMOOTHING_BITS: f64 = 128.0;
/// A shortcut or identity key column exceeds its norm bound with probability at
/// most 2^-128 (and is then resampled), so the redraw is statistically invisible.
const KEY_NORM_BITS: f64 = 128.0;
/// |R d|^2 <= f m |d|^2 for the encryption signs R, except with probability 2^-128.
const SIGN_NORM_BITS: f64 = 128.0;
/// Slack on the expected largest singular value of the ternary trapdoor R.
const TRAPDOOR_SLACK: f64 = 1.1;
/// One-dimensional samplers keep the integers within TAIL widths of the center;
/// the omitted mass is below exp(-pi TAIL^2) < 2^-190 of the total.
const TAIL: f64 = 6.5;
const TRAPDOOR_ATTEMPTS: usize = 64;
const KEY_ATTEMPTS: usize = 64;
/// Abar has ABAR n columns, the rows of the trapdoor R (see the module documentation).
const ABAR: usize = 3;

/// Parameters of the trapdoor setup: an Epoque profile, the gadget base and the
/// preimage width s (the Gaussian parameter of every shortcut and identity key).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TrapdoorProfile {
    profile: Profile,
    base: u64,
    digits: usize,
    width: f64,
    rounding: f64,
    gadget: f64,
}

impl TrapdoorProfile {
    /// Validates the conditions the trapdoor setup and Construction 1 rely on:
    /// - m >= ceil(n log2 q) + 2*128 + 2 ceil(log2 g) (leftover hash for b = A0 k
    ///   with Construction 1's binary keys), and m >= 3n + nk for the trapdoor blocks;
    /// - s^2 >= sigma_G^2 (s1^2 + 1) + r^2, where r = eta_eps(Z^m) with eps = 2^-128,
    ///   sigma_G = r sqrt(b^2 + 1) is the gadget width, and
    ///   s1 = 1.1 sqrt(2/3) (sqrt(3n) + sqrt(nk)) bounds the trapdoor's largest
    ///   singular value (TrapGen resamples R when it does not hold);
    /// - C bounds sqrt(1 + (|d1| + sqrt(f m) |d2|)^2) for keys of expected norm
    ///   sqrt(m) s / sqrt(2 pi) up to a 2^-128 tail. Then the decryption noise has
    ///   standard deviation at most about q/65 unless the sign bound fails
    ///   (probability 2^-128), and a bit fails with probability below 2^-189;
    /// - the error standard deviation q/(65 C) is at least 2 sqrt(n)/sqrt(2 pi)
    ///   (Regev's width condition, alpha q >= 2 sqrt(n)).
    pub fn new(profile: Profile, base: u64, width: f64) -> Result<Self> {
        let (n, m, q) = (profile.n, profile.m, profile.q);
        if !(2..=1 << 16).contains(&base) || base >= q || !width.is_finite() || width <= 0.0 {
            return Err(Error("invalid gadget base or preimage width"));
        }
        let digits = gadget_digits(q, base);
        let columns = n
            .checked_mul(digits)
            .and_then(|x| x.checked_add(ABAR * n))
            .ok_or(Error("trapdoor dimensions overflow"))?;
        if columns > m {
            return Err(Error("m is too small for the gadget trapdoor"));
        }
        if m < construction1_bits(n, q, profile.maximum_constraints)? {
            return Err(Error(
                "m is below Construction 1's leftover-hash condition for binary keys",
            ));
        }
        // The trapdoor products are exact in f64 below 2^53.
        if ((ABAR * n) as u128) * u128::from(q - 1) >= 1 << 53 {
            return Err(Error("3n(q-1) must stay below 2^53"));
        }
        let rounding = rounding_width(m);
        let gadget = rounding * ((base * base + 1) as f64).sqrt();
        let s1 = trapdoor_singular_bound(n, n * digits);
        if width * width < gadget * gadget * (s1 * s1 + 1.0) + rounding * rounding {
            return Err(Error("preimage width is below the trapdoor sampling bound"));
        }
        if width > 1e8 {
            return Err(Error("preimage width too large for i32 keys"));
        }
        if (profile.c_bound as f64) < expected_noise_bound(m, width) {
            return Err(Error("C is below the noise bound of width-s keys"));
        }
        if (q as f64) / (profile.denominator() as f64) < 2.0 * (n as f64).sqrt() / (2.0 * PI).sqrt()
        {
            return Err(Error("error width q/(65C) is below Regev's condition"));
        }
        Ok(Self {
            profile,
            base,
            digits,
            width,
            rounding,
            gadget,
        })
    }
    /// Test-sized parameters (n = 16): fast arithmetic checks, no security.
    pub fn small() -> Self {
        let p =
            Profile::new(16, 671, 47_835_289, 8, 8, 230_587).expect("fixed trapdoor smoke profile");
        Self::new(p, 4, 438.0).expect("fixed trapdoor smoke profile")
    }
    /// n = 1360, the smallest dimension (in steps of 8) at which the ciphertext's LWE
    /// instance reaches 128 bits core-SVP, with Construction 1's conditions (see the
    /// README, tools/trapdoor_parameters.py and tools/security_estimate.py).
    pub fn full() -> Self {
        let p = Profile::new(1360, 51_325, 200_639_082_569, 128, 8, 104_904_148)
            .expect("fixed trapdoor full profile");
        Self::new(p, 4, 4700.0).expect("fixed trapdoor full profile")
    }
    pub fn profile(&self) -> Profile {
        self.profile
    }
    pub fn base(&self) -> u64 {
        self.base
    }
    /// k, the number of base-b digits of q.
    pub fn digits(&self) -> usize {
        self.digits
    }
    /// nk, the number of gadget columns.
    pub fn gadget_columns(&self) -> usize {
        self.profile.n * self.digits
    }
    /// 3n, the columns of Abar and rows of the trapdoor R.
    pub fn trapdoor_rows(&self) -> usize {
        ABAR * self.profile.n
    }
    /// Uniform columns between Abar and the gadget block.
    pub fn padding(&self) -> usize {
        self.profile.m - self.trapdoor_rows() - self.gadget_columns()
    }
    /// s: every key coordinate has standard deviation about s / sqrt(2 pi).
    pub fn width(&self) -> f64 {
        self.width
    }
    /// r = eta_eps(Z^m) with eps = 2^-128.
    pub fn rounding_width(&self) -> f64 {
        self.rounding
    }
    /// sigma_G = r sqrt(b^2 + 1).
    pub fn gadget_width(&self) -> f64 {
        self.gadget
    }
    /// sqrt(1 + (|first| + sqrt(f m) |second|)^2), which must not exceed C: with
    /// fresh encryption signs R, |first + R second| stays below it except with
    /// probability 2^-128, so the noise of a key column has standard deviation
    /// at most q/65 (C times the error deviation q/(65 C)).
    pub fn noise_bound(&self, first: &[i32], second: &[i32]) -> f64 {
        let norm = |x: &[i32]| {
            x.iter()
                .map(|&v| f64::from(v) * f64::from(v))
                .sum::<f64>()
                .sqrt()
        };
        let m = self.profile.m as f64;
        let sign = (chi2_factor(self.profile.m, SIGN_NORM_BITS) * m).sqrt();
        (1.0 + (norm(first) + sign * norm(second)).powi(2)).sqrt()
    }
    fn within_bound(&self, first: &[i32], second: &[i32]) -> bool {
        self.noise_bound(first, second) <= self.profile.c_bound as f64
    }
}

/// The smallest k with b^k > q.
fn gadget_digits(q: u64, base: u64) -> usize {
    let (mut k, mut power) = (0, 1u128);
    while power <= u128::from(q) {
        power *= u128::from(base);
        k += 1;
    }
    k
}

/// ceil(n log2 q) + 2 lambda + 2 ceil(log2 g): the key entropy (bits) that
/// Construction 1's leftover-hash condition requires; m for binary keys.
fn construction1_bits(n: usize, q: u64, g: usize) -> Result<usize> {
    let bits = n as f64 * (q as f64).log2();
    if (bits - bits.round()).abs() < 1e-6 {
        return Err(Error(
            "n log2 q is too close to an integer to round reliably",
        ));
    }
    let log_g = (usize::BITS - g.saturating_sub(1).leading_zeros()) as usize;
    Ok(bits.ceil() as usize + 2 * LAMBDA + 2 * log_g)
}

/// eta_eps(Z^m) = sqrt(ln(2m(1 + 1/eps)) / pi) for eps = 2^-128.
fn rounding_width(m: usize) -> f64 {
    ((((2 * m) as f64).ln() + SMOOTHING_BITS * std::f64::consts::LN_2) / PI).sqrt()
}

fn trapdoor_singular_bound(n: usize, columns: usize) -> f64 {
    TRAPDOOR_SLACK * (2.0f64 / 3.0).sqrt() * (((ABAR * n) as f64).sqrt() + (columns as f64).sqrt())
}

/// The smallest 1 + e with m (e - ln(1 + e)) / 2 >= bits ln 2: a chi-square tail
/// bound, also valid for sums of squared Rademacher combinations.
fn chi2_factor(m: usize, bits: f64) -> f64 {
    let target = 2.0 * bits * std::f64::consts::LN_2 / m as f64;
    let excess = |e: f64| e - e.ln_1p();
    let (mut low, mut high) = (0.0f64, 1.0f64);
    while excess(high) < target {
        high *= 2.0;
    }
    for _ in 0..200 {
        let middle = 0.5 * (low + high);
        if excess(middle) >= target {
            high = middle;
        } else {
            low = middle;
        }
    }
    1.0 + high
}

/// The noise bound for key columns of expected norm sqrt(m) s / sqrt(2 pi).
fn expected_noise_bound(m: usize, width: f64) -> f64 {
    let norm = (m as f64).sqrt() * width / (2.0 * PI).sqrt();
    let keys = chi2_factor(m, KEY_NORM_BITS).sqrt();
    let sign = (chi2_factor(m, SIGN_NORM_BITS) * m as f64).sqrt();
    (1.0 + (keys * norm * (1.0 + sign)).powi(2)).sqrt()
}

/// A uniform real in [0, 1) with 53 random bits.
fn unit<R: Rng + ?Sized>(rng: &mut R) -> f64 {
    (rng.next_u64() >> 11) as f64 * (1.0 / (1u64 << 53) as f64)
}

/// A uniform real in (0, 1] on the multiples of 2^-128, from two words. As the
/// Box-Muller radius sqrt(-2 ln v), it reaches 13.3 standard deviations, so the
/// tail it cuts off has mass below 2^-128 (a 53-bit uniform stops at 8.6, about
/// 2^-53 per pair).
fn unit_fine<R: Rng + ?Sized>(rng: &mut R) -> f64 {
    const SCALE: f64 = 1.0 / 18_446_744_073_709_551_616.0; // 2^-64
    let (high, low) = (rng.next_u64(), rng.next_u64());
    ((high as f64 + (low as f64 + 1.0) * SCALE) * SCALE).min(1.0)
}

/// Standard normal deviates by the Box-Muller transform, two per pair of uniforms.
#[derive(Default)]
struct Normals {
    spare: Option<f64>,
}
impl Normals {
    fn next<R: Rng + ?Sized>(&mut self, rng: &mut R) -> f64 {
        if let Some(x) = self.spare.take() {
            return x;
        }
        let radius = (-2.0 * unit_fine(rng).ln()).sqrt();
        let (sin, cos) = (2.0 * PI * unit(rng)).sin_cos();
        self.spare = Some(radius * sin);
        radius * cos
    }
}

/// D_{Z,width,center}: mass proportional to exp(-pi (x - center)^2 / width^2),
/// by rejection from the uniform law on the integers within TAIL widths.
fn sample_z<R: Rng + ?Sized>(center: f64, width: f64, rng: &mut R) -> Result<i64> {
    let radius = TAIL * width;
    let (low, high) = ((center - radius).floor(), (center + radius).ceil());
    if !(width > 0.0 && low.is_finite() && high.is_finite() && high - low < 1e15) {
        return Err(Error(
            "one-dimensional Gaussian center or width out of range",
        ));
    }
    let (low, span) = (low as i64, (high - low) as u64 + 1);
    let scale = -PI / (width * width);
    loop {
        let x = low + rng.random_range(0..span) as i64;
        let d = x as f64 - center;
        if unit(rng) < (scale * d * d).exp() {
            return Ok(x);
        }
    }
}

/// A dot product with eight independent accumulators (vectorizes; fixed order).
fn dot(x: &[f64], y: &[f64]) -> f64 {
    let mut lanes = [0.0f64; 8];
    let (xs, ys) = (x.chunks_exact(8), y.chunks_exact(8));
    let (x_rest, y_rest) = (xs.remainder(), ys.remainder());
    for (a, b) in xs.zip(ys) {
        for l in 0..8 {
            lanes[l] += a[l] * b[l];
        }
    }
    let mut total = lanes.iter().sum::<f64>();
    for (a, b) in x_rest.iter().zip(y_rest) {
        total += a * b;
    }
    total
}

/// The same for a ternary row and a real vector.
fn dot_ternary(r: &[i8], y: &[f64]) -> f64 {
    let mut lanes = [0.0f64; 8];
    let (rs, ys) = (r.chunks_exact(8), y.chunks_exact(8));
    let (r_rest, y_rest) = (rs.remainder(), ys.remainder());
    for (a, b) in rs.zip(ys) {
        for l in 0..8 {
            lanes[l] += f64::from(a[l]) * b[l];
        }
    }
    let mut total = lanes.iter().sum::<f64>();
    for (&a, b) in r_rest.iter().zip(y_rest) {
        total += f64::from(a) * b;
    }
    total
}

/// R Y for the trapdoor R (rows by cols, ternary) and `count` vectors Y
/// (row-major, each of length cols). Returns rows by count. Exact when every
/// partial sum is an integer below 2^53.
fn ternary_product(r: &[i8], rows: usize, cols: usize, y: &[f64], count: usize) -> Vec<f64> {
    const TILE: usize = 32;
    let mut out = vec![0.0; rows * count];
    for_each_row_block(&mut out, count, cols * count, |first, block| {
        let block_rows = block.len() / count;
        for tile in (0..block_rows).step_by(TILE) {
            let end = (tile + TILE).min(block_rows);
            for j in 0..count {
                let column = &y[j * cols..(j + 1) * cols];
                for i in tile..end {
                    let row = first + i;
                    block[i * count + j] = dot_ternary(&r[row * cols..(row + 1) * cols], column);
                }
            }
        }
    });
    out
}

/// Gaussian sampling on cosets of Lambda^perp(g^T), g = (1, b, .., b^(k-1)), by
/// Klein's algorithm on the basis S_k (columns b e_i - e_(i+1) and the base-b
/// digits of q), whose Gram-Schmidt vectors have norm at most sqrt(b^2 + 1).
#[derive(Clone, Debug)]
struct Gadget {
    base: u64,
    k: usize,
    basis: Vec<i64>,
    orthogonal: Vec<f64>,
    squared_norms: Vec<f64>,
    width: f64,
}

impl Gadget {
    fn new(tp: &TrapdoorProfile) -> Self {
        let (q, base, k) = (tp.profile.q, tp.base, tp.digits);
        let mut basis = vec![0i64; k * k];
        for i in 0..k - 1 {
            basis[i * k + i] = base as i64;
            basis[i * k + i + 1] = -1;
        }
        let mut rest = q;
        for j in 0..k {
            basis[(k - 1) * k + j] = (rest % base) as i64;
            rest /= base;
        }
        let mut orthogonal = vec![0.0f64; k * k];
        let mut squared_norms = vec![0.0f64; k];
        for i in 0..k {
            let column: Vec<f64> = basis[i * k..(i + 1) * k]
                .iter()
                .map(|&x| x as f64)
                .collect();
            let mut v = column.clone();
            for j in 0..i {
                let previous = &orthogonal[j * k..(j + 1) * k];
                let mu = dot(&column, previous) / squared_norms[j];
                for (x, &y) in v.iter_mut().zip(previous) {
                    *x -= mu * y;
                }
            }
            squared_norms[i] = dot(&v, &v);
            orthogonal[i * k..(i + 1) * k].copy_from_slice(&v);
        }
        Self {
            base,
            k,
            basis,
            orthogonal,
            squared_norms,
            width: tp.gadget,
        }
    }

    /// Write z with <g, z> = value (mod q), distributed as D_{Lambda^perp_value(g^T), sigma_G}.
    fn sample<R: Rng + ?Sized>(
        &self,
        value: u64,
        out: &mut [i64],
        center: &mut [f64],
        rng: &mut R,
    ) -> Result<()> {
        let k = self.k;
        // t = base-b digits of value is one solution; add a lattice point drawn
        // from D_{Lambda, sigma_G, -t} (Klein, last basis vector first).
        let mut rest = value;
        for j in 0..k {
            out[j] = (rest % self.base) as i64;
            rest /= self.base;
            center[j] = -(out[j] as f64);
        }
        for i in (0..k).rev() {
            let direction = &self.orthogonal[i * k..(i + 1) * k];
            let c = dot(center, direction) / self.squared_norms[i];
            let z = sample_z(c, self.width / self.squared_norms[i].sqrt(), rng)?;
            if z != 0 {
                for (j, &entry) in self.basis[i * k..(i + 1) * k].iter().enumerate() {
                    center[j] -= (z * entry) as f64;
                    out[j] += z * entry;
                }
            }
        }
        Ok(())
    }
}

/// R and the Cholesky factor of the conditional perturbation covariance.
struct Trapdoor {
    r: Vec<i8>,
    factor: Vec<f64>,
}

/// Uniform entries of {-1, 0, 1}: forty base-3 digits per accepted 64-bit word.
fn ternary<R: Rng + ?Sized>(len: usize, rng: &mut R) -> Vec<i8> {
    const WORD: u64 = 12_157_665_459_056_928_801; // 3^40
    let mut out = Vec::with_capacity(len);
    while out.len() < len {
        let mut x = rng.next_u64();
        if x >= WORD {
            continue;
        }
        for _ in 0..40.min(len - out.len()) {
            out.push((x % 3) as i8 - 1);
            x /= 3;
        }
    }
    out
}

/// In-place lower Cholesky factor of a symmetric matrix stored row-major (lower
/// triangle filled). Returns false unless it is numerically positive definite.
///
/// Every entry is computed exactly as in the row-by-row algorithm,
/// `L[i][j] = (A[i][j] - dot(L[i][..j], L[j][..j])) / L[j][j]` with the same
/// `dot`, so the factor is the same bit for bit. Columns are taken in panels of
/// 64: the panel's diagonal block row by row, then the panel's columns of all later
/// rows in parallel over the rows (each needs only its own row and finished rows).
fn cholesky(a: &mut [f64], n: usize) -> bool {
    const PANEL: usize = 64;
    for start in (0..n).step_by(PANEL) {
        let end = (start + PANEL).min(n);
        for i in start..end {
            let (done, rest) = a.split_at_mut(i * n);
            let row = &mut rest[..n];
            for j in start..i {
                let previous = &done[j * n..j * n + j + 1];
                let value = (row[j] - dot(&row[..j], &previous[..j])) / previous[j];
                row[j] = value;
            }
            let diagonal = row[i] - dot(&row[..i], &row[..i]);
            if !(diagonal.is_finite() && diagonal > 0.0) {
                return false;
            }
            row[i] = diagonal.sqrt();
        }
        let (done, rest) = a.split_at_mut(end * n);
        let done = &*done;
        for_each_row_block(rest, n, (end - start) * end, |_, block| {
            for row in block.chunks_exact_mut(n) {
                for j in start..end {
                    let previous = &done[j * n..j * n + j + 1];
                    let value = (row[j] - dot(&row[..j], &previous[..j])) / previous[j];
                    row[j] = value;
                }
            }
        });
    }
    true
}

/// Constants of the perturbation: c22 = s^2 - r^2 - sigma_G^2 is the parameter of
/// the gadget block; the Abar block, given it, has parameter (s^2 - r^2) I - kappa R R^T.
struct Perturbation {
    c22: f64,
    kappa: f64,
}
impl Perturbation {
    fn new(tp: &TrapdoorProfile) -> Self {
        let (s2, r2, g2) = (tp.width.powi(2), tp.rounding.powi(2), tp.gadget.powi(2));
        let c22 = s2 - r2 - g2;
        Self {
            c22,
            kappa: g2 * (s2 - r2) / c22,
        }
    }
}

/// The factor L with L L^T = (s^2 - r^2) I - kappa R R^T, or None when R's
/// largest singular value is too large for s (then TrapGen resamples R).
fn perturbation_factor(tp: &TrapdoorProfile, r: &[i8]) -> Option<Vec<f64>> {
    let (rows, w) = (tp.trapdoor_rows(), tp.gadget_columns());
    let constants = Perturbation::new(tp);
    let diagonal = tp.width.powi(2) - tp.rounding.powi(2);
    let mut matrix = vec![0.0f64; rows * rows];
    for_each_row_block(&mut matrix, rows, rows * w / 2, |first, block| {
        for (offset, out) in block.chunks_exact_mut(rows).enumerate() {
            let a = first + offset;
            let ra = &r[a * w..(a + 1) * w];
            for (b, value) in out.iter_mut().enumerate().take(a + 1) {
                let gram: i32 = ra
                    .iter()
                    .zip(&r[b * w..(b + 1) * w])
                    .map(|(&x, &y)| i32::from(x) * i32::from(y))
                    .sum();
                *value = -constants.kappa * f64::from(gram);
            }
            out[a] += diagonal;
        }
    });
    cholesky(&mut matrix, rows).then_some(matrix)
}

/// TrapGen: A = [Abar | P | G - Abar R] (n by m, row-major) and its trapdoor.
fn trapgen<R: Rng + ?Sized>(
    tp: &TrapdoorProfile,
    rng: &mut R,
) -> Result<(Vec<u64>, Trapdoor, usize)> {
    let p = tp.profile;
    let (n, m, q) = (p.n, p.m, p.q);
    let (rows, w, k) = (tp.trapdoor_rows(), tp.gadget_columns(), tp.digits);
    let free = m - w;
    let mut a = Vec::new();
    a.try_reserve_exact(n * m)
        .map_err(|_| Error("allocation failed"))?;
    for _ in 0..n {
        a.extend(uniform(free, q, rng)?);
        a.extend(std::iter::repeat_n(0, w));
    }
    let mut attempts = 0;
    let (r, factor) = loop {
        attempts += 1;
        if attempts > TRAPDOOR_ATTEMPTS {
            return Err(Error("trapdoor singular-value check failed repeatedly"));
        }
        let r = ternary(rows * w, rng);
        if let Some(factor) = perturbation_factor(tp, &r) {
            break (r, factor);
        }
    };
    // G - Abar R, tiled: every partial sum is an integer below 3n(q-1) < 2^53.
    let powers: Vec<u64> = (0..k).map(|j| tp.base.pow(j as u32)).collect();
    const ROWS: usize = 16;
    const COLS: usize = 2048;
    for_each_row_block(&mut a, m, rows * w, |first, block| {
        let block_rows = block.len() / m;
        let mut acc = vec![0.0f64; ROWS * COLS];
        for row_start in (0..block_rows).step_by(ROWS) {
            let rb = (block_rows - row_start).min(ROWS);
            for col_start in (0..w).step_by(COLS) {
                let cb = (w - col_start).min(COLS);
                acc[..rb * cb].fill(0.0);
                for j in 0..rows {
                    let r_row = &r[j * w + col_start..j * w + col_start + cb];
                    for t in 0..rb {
                        let coefficient = block[(row_start + t) * m + j] as f64;
                        for (x, &y) in acc[t * cb..(t + 1) * cb].iter_mut().zip(r_row) {
                            *x += coefficient * f64::from(y);
                        }
                    }
                }
                for t in 0..rb {
                    let i = first + row_start + t;
                    for c in 0..cb {
                        let column = col_start + c;
                        let g = if column / k == i {
                            powers[column % k]
                        } else {
                            0
                        };
                        let product = acc[t * cb + c] as i64;
                        block[(row_start + t) * m + free + column] =
                            p.field.reduce_signed(i128::from(g) - i128::from(product));
                    }
                }
            }
        }
    });
    Ok((a, Trapdoor { r, factor }, attempts))
}

/// `count` perturbations (count by m, row-major) with parameter
/// sqrt(s^2 I - sigma_G^2 T T^T), T = [R; 0; I], by Peikert's convolution: a
/// continuous Gaussian of parameter sqrt(s^2 I - sigma_G^2 T T^T - r^2 I), then
/// D_{Z,r,y_i} on each coordinate. The gadget block is drawn first (parameter
/// sqrt(c22)); given it, the Abar block has mean -(sigma_G^2/c22) R y_G and
/// parameter L, and the padding is independent with parameter sqrt(s^2 - r^2).
fn perturbations<R: Rng + ?Sized>(
    tp: &TrapdoorProfile,
    td: &Trapdoor,
    count: usize,
    rng: &mut R,
) -> Result<Vec<i32>> {
    let p = tp.profile;
    let (rows, w, m) = (tp.trapdoor_rows(), tp.gadget_columns(), p.m);
    let free = m - w;
    let constants = Perturbation::new(tp);
    let to_deviation = 1.0 / (2.0 * PI).sqrt();
    let gadget_scale = constants.c22.sqrt() * to_deviation;
    let padding_scale = (tp.width.powi(2) - tp.rounding.powi(2)).sqrt() * to_deviation;
    let mut normals = Normals::default();
    let mut gadget_part = vec![0.0f64; count * w];
    let mut free_part = vec![0.0f64; count * free];
    let mut g = vec![0.0f64; count * rows];
    // The normals in the order of one perturbation after the other: its gadget
    // block, its g, its padding.
    for j in 0..count {
        for x in &mut gadget_part[j * w..(j + 1) * w] {
            *x = gadget_scale * normals.next(rng);
        }
        for x in &mut g[j * rows..(j + 1) * rows] {
            *x = normals.next(rng);
        }
        for x in &mut free_part[j * free + rows..(j + 1) * free] {
            *x = padding_scale * normals.next(rng);
        }
    }
    // The Abar block L g of every perturbation, which draws no coins, in parallel.
    for_each_row_block(&mut free_part, free, rows * rows / 2, |first, block| {
        for (offset, out) in block.chunks_exact_mut(free).enumerate() {
            let g = &g[(first + offset) * rows..(first + offset + 1) * rows];
            for (a, x) in out[..rows].iter_mut().enumerate() {
                *x = to_deviation * dot(&td.factor[a * rows..a * rows + a + 1], &g[..a + 1]);
            }
        }
    });
    let shift = ternary_product(&td.r, rows, w, &gadget_part, count);
    let ratio = tp.gadget.powi(2) / constants.c22;
    let mut out = vec![0i32; count * m];
    let round = |y: f64, rng: &mut R| -> Result<i32> {
        i32::try_from(sample_z(y, tp.rounding, rng)?).map_err(|_| Error("perturbation overflow"))
    };
    for j in 0..count {
        for a in 0..free {
            let mut y = free_part[j * free + a];
            if a < rows {
                y -= ratio * shift[a * count + j];
            }
            out[j * m + a] = round(y, rng)?;
        }
        for c in 0..w {
            out[j * m + free + c] = round(gadget_part[j * w + c], rng)?;
        }
    }
    Ok(out)
}

/// MP12 SampleD for `count` targets (n by count, row-major): preimages x_j
/// (count by m) with A x_j = t_j (mod q), each distributed as
/// D_{Lambda^perp_{t_j}(A), s}. Every preimage is checked exactly.
fn sample_preimages<R: Rng + ?Sized>(
    tp: &TrapdoorProfile,
    a: &[u64],
    td: &Trapdoor,
    gadget: &Gadget,
    targets: &[u64],
    count: usize,
    rng: &mut R,
) -> Result<Vec<i32>> {
    let p = tp.profile;
    let (n, m, q, k) = (p.n, p.m, p.q, tp.digits);
    let (rows, w) = (tp.trapdoor_rows(), tp.gadget_columns());
    let free = m - w;
    if targets.len() != n * count || targets.iter().any(|&x| x >= q) {
        return Err(Error("invalid preimage targets"));
    }
    let mut x = perturbations(tp, td, count, rng)?;
    let image = wide_right_product(a, n, m, &x, count, q)?;
    let mut z = vec![0i64; count * w];
    let mut center = vec![0.0f64; k];
    for j in 0..count {
        for i in 0..n {
            let value = p.field.sub(targets[i * count + j], image[i * count + j]);
            gadget.sample(
                value,
                &mut z[j * w + i * k..j * w + (i + 1) * k],
                &mut center,
                rng,
            )?;
        }
    }
    let z_real: Vec<f64> = z.iter().map(|&v| v as f64).collect();
    let rz = ternary_product(&td.r, rows, w, &z_real, count);
    let add = |x: &mut i32, v: i64| -> Result<()> {
        *x = i32::try_from(i64::from(*x) + v).map_err(|_| Error("preimage overflow"))?;
        Ok(())
    };
    for j in 0..count {
        for a_index in 0..rows {
            add(&mut x[j * m + a_index], rz[a_index * count + j] as i64)?;
        }
        for c in 0..w {
            add(&mut x[j * m + free + c], z[j * w + c])?;
        }
    }
    if wide_right_product(a, n, m, &x, count, q)? != targets {
        return Err(Error("preimage check failed"));
    }
    Ok(x)
}

/// The extraction master key T_A0 (Epoque's msk), with the data SamplePre needs.
pub struct MasterTrapdoor {
    profile: TrapdoorProfile,
    trapdoor: Trapdoor,
    gadget: Gadget,
}

pub struct TrapdoorSetup {
    /// Public key and the master shortcut key (D1, D2) of Section 5.2.1.
    pub setup: Setup,
    /// T_A0 for Extr. Drop it unless identity keys are needed: whoever holds it
    /// can read every hidden message (see `MasterTrapdoor::recover_secret`).
    pub master: MasterTrapdoor,
    /// TrapGen draws of R for A0 and for B (more than one only if the
    /// singular-value check failed).
    pub trapgen_attempts: [usize; 2],
    /// Shortcut-key columns redrawn because they exceeded the noise bound C.
    pub key_resamples: usize,
}

/// Epoque's KeyGen with trapdoors and the master shortcut key (see the module
/// documentation). The callback receives phase names and counts, never secrets.
pub fn setup_trapdoor<R: Rng + ?Sized>(
    tp: &TrapdoorProfile,
    rng: &mut R,
    mut progress: impl FnMut(&str, usize),
) -> Result<TrapdoorSetup> {
    let p = tp.profile;
    let (n, m, q, jb) = (p.n, p.m, p.q, p.ordinary_bits);
    let gadget = Gadget::new(tp);
    let (a0, master, attempts_a0) = trapgen(tp, rng)?;
    progress("trapgen_a0", attempts_a0);
    let (b, trapdoor_b, attempts_b) = trapgen(tp, rng)?;
    progress("trapgen_b", attempts_b);
    let a1 = uniform(n * m, q, rng)?;
    let u = uniform(n * jb, q, rng)?;
    progress("uniform_a1_u", n);
    let zero = vec![0u64; n * jb];
    let mut d2t = sample_preimages(tp, &b, &trapdoor_b, &gadget, &zero, jb, rng)?;
    progress("shortcut_d2", jb);
    let image = wide_right_product(&a1, n, m, &d2t, jb, q)?;
    let targets: Vec<u64> = u
        .iter()
        .zip(&image)
        .map(|(&x, &y)| p.field.sub(x, y))
        .collect();
    let mut d1t = sample_preimages(tp, &a0, &master, &gadget, &targets, jb, rng)?;
    progress("shortcut_d1", jb);
    let mut resamples = 0;
    for j in 0..jb {
        let mut attempts = 0;
        while !tp.within_bound(&d1t[j * m..(j + 1) * m], &d2t[j * m..(j + 1) * m]) {
            attempts += 1;
            if attempts > KEY_ATTEMPTS {
                return Err(Error("shortcut keys repeatedly exceed the noise bound C"));
            }
            resamples += 1;
            let d2 = sample_preimages(tp, &b, &trapdoor_b, &gadget, &vec![0; n], 1, rng)?;
            let image = wide_right_product(&a1, n, m, &d2, 1, q)?;
            let target: Vec<u64> = (0..n)
                .map(|i| p.field.sub(u[i * jb + j], image[i]))
                .collect();
            let d1 = sample_preimages(tp, &a0, &master, &gadget, &target, 1, rng)?;
            d2t[j * m..(j + 1) * m].copy_from_slice(&d2);
            d1t[j * m..(j + 1) * m].copy_from_slice(&d1);
        }
    }
    // T_B is no longer needed: erase it.
    drop(trapdoor_b);
    // B D2 = 0 and A0 D1 + A1 D2 = U, entry by entry.
    let kernel = wide_right_product(&b, n, m, &d2t, jb, q)?;
    let left = wide_right_product(&a0, n, m, &d1t, jb, q)?;
    let right = wide_right_product(&a1, n, m, &d2t, jb, q)?;
    if kernel.iter().any(|&x| x != 0) || (0..n * jb).any(|i| p.field.add(left[i], right[i]) != u[i])
    {
        return Err(Error("shortcut key relations failed"));
    }
    progress("relations_checked", jb);
    let shortcut = ShortcutKey::from_transposed(p, d1t, d2t)?;
    Ok(TrapdoorSetup {
        setup: Setup {
            public: PublicKey::new(p, a0, a1, b, u)?,
            shortcut,
        },
        master: MasterTrapdoor {
            profile: *tp,
            trapdoor: master,
            gadget,
        },
        trapgen_attempts: [attempts_a0, attempts_b],
        key_resamples: resamples,
    })
}

/// An identity key (e1_j; e2_j) for one identity, stored transposed (J by m).
pub struct IdentityKey {
    profile: Profile,
    identity: Identity,
    e1t: Vec<i32>,
    e2t: Vec<i32>,
    /// Columns redrawn because they exceeded the noise bound C.
    pub resamples: usize,
}
impl IdentityKey {
    /// The identity this key was extracted for.
    pub fn identity(&self) -> &Identity {
        &self.identity
    }
    pub fn e1_transposed(&self) -> &[i32] {
        &self.e1t
    }
    pub fn e2_transposed(&self) -> &[i32] {
        &self.e2t
    }
}

impl MasterTrapdoor {
    pub fn profile(&self) -> TrapdoorProfile {
        self.profile
    }

    /// Spot-check that `pk.a0` is the matrix of this trapdoor: A0 [R; 0; I] = G on
    /// four gadget columns (all n rows). A full check would cost n 3n nk operations.
    fn matches(&self, pk: &PublicKey) -> Result<()> {
        let tp = &self.profile;
        let p = tp.profile;
        if pk.profile != p {
            return Err(Error("master trapdoor belongs to different parameters"));
        }
        let (n, m, w, k) = (p.n, p.m, tp.gadget_columns(), tp.digits);
        let rows = tp.trapdoor_rows();
        let free = m - w;
        let r = &self.trapdoor.r;
        let powers: Vec<u64> = (0..k).map(|j| tp.base.pow(j as u32)).collect();
        for column in [0, k - 1, w / 2, w - 1] {
            for i in 0..n {
                let row = &pk.a0[i * m..(i + 1) * m];
                let mut total = i128::from(row[free + column]);
                for a in 0..rows {
                    total += i128::from(row[a]) * i128::from(r[a * w + column]);
                }
                let g = if column / k == i {
                    powers[column % k]
                } else {
                    0
                };
                if p.field.reduce_signed(total) != g {
                    return Err(Error("master trapdoor does not match A0"));
                }
            }
        }
        Ok(())
    }

    /// Extr for identity `id` (F_id = [A0 | A1 + H(id) B]): e2_j from D_{Z^m,s} and
    /// e1_j <- SamplePre(A0, T_A0, u_j - (A1 + H(id) B) e2_j), so F_id (e1_j; e2_j) = u_j.
    pub fn extract<R: Rng + ?Sized>(
        &self,
        pk: &PublicKey,
        id: &Identity,
        rng: &mut R,
    ) -> Result<IdentityKey> {
        self.matches(pk)?;
        let tp = &self.profile;
        let p = tp.profile;
        if id.profile() != p {
            return Err(Error("identity belongs to different parameters"));
        }
        let (n, m, q, jb) = (p.n, p.m, p.q, p.ordinary_bits);
        let gaussian = |rng: &mut R| -> Result<i32> {
            i32::try_from(sample_z(0.0, tp.width, rng)?).map_err(|_| Error("key overflow"))
        };
        // (A1 + H(id) B) E2 for `count` columns E2 stored transposed.
        let image = |e2t: &[i32], count: usize| -> Result<Vec<u64>> {
            let a1 = wide_right_product(&pk.a1, n, m, e2t, count, q)?;
            let hb = id.times(&wide_right_product(&pk.b, n, m, e2t, count, q)?, count)?;
            Ok(a1
                .iter()
                .zip(&hb)
                .map(|(&x, &y)| p.field.add(x, y))
                .collect())
        };
        let mut e2t = (0..jb * m)
            .map(|_| gaussian(rng))
            .collect::<Result<Vec<_>>>()?;
        let right = image(&e2t, jb)?;
        let all: Vec<u64> =
            pk.u.iter()
                .zip(&right)
                .map(|(&u, &x)| p.field.sub(u, x))
                .collect();
        let mut e1t = sample_preimages(tp, &pk.a0, &self.trapdoor, &self.gadget, &all, jb, rng)?;
        let mut resamples = 0;
        for j in 0..jb {
            let mut attempts = 0;
            while !tp.within_bound(&e1t[j * m..(j + 1) * m], &e2t[j * m..(j + 1) * m]) {
                attempts += 1;
                if attempts > KEY_ATTEMPTS {
                    return Err(Error("identity keys repeatedly exceed the noise bound C"));
                }
                resamples += 1;
                let e2 = (0..m).map(|_| gaussian(rng)).collect::<Result<Vec<_>>>()?;
                let right = image(&e2, 1)?;
                let target: Vec<u64> = (0..n)
                    .map(|i| p.field.sub(pk.u[i * jb + j], right[i]))
                    .collect();
                let e1 =
                    sample_preimages(tp, &pk.a0, &self.trapdoor, &self.gadget, &target, 1, rng)?;
                e2t[j * m..(j + 1) * m].copy_from_slice(&e2);
                e1t[j * m..(j + 1) * m].copy_from_slice(&e1);
            }
        }
        // F_id (e1_j; e2_j) = u_j on every entry.
        let left = wide_right_product(&pk.a0, n, m, &e1t, jb, q)?;
        let right = image(&e2t, jb)?;
        if (0..n * jb).any(|i| p.field.add(left[i], right[i]) != pk.u[i]) {
            return Err(Error("identity key relation failed"));
        }
        Ok(IdentityKey {
            profile: p,
            identity: id.clone(),
            e1t,
            e2t,
            resamples,
        })
    }

    /// MP12 inversion of v = c1[1..m] = A0^T s + y: R^T v_Abar + v_G = G^T s + e
    /// with the small error e = R^T y_Abar + y_G, and S_k^T removes s from each
    /// gadget block, leaving S_k^T e exactly. Returns s. This is why T_A0 must stay
    /// away from the dictator: with s, the hidden constraints H_S^T s = floor(q/2) am
    /// can be read off directly.
    pub fn recover_secret(&self, pk: &PublicKey, ct: &Ciphertext) -> Result<Vec<u64>> {
        self.matches(pk)?;
        let tp = &self.profile;
        let p = tp.profile;
        ct.validate(p)?;
        let (n, m, q, k, base) = (p.n, p.m, p.q, tp.digits, tp.base);
        let (rows, w) = (tp.trapdoor_rows(), tp.gadget_columns());
        let free = m - w;
        let field = &p.field;
        let r = &self.trapdoor.r;
        let lift = |x: u64| -> i128 {
            if x > q / 2 {
                i128::from(x) - i128::from(q)
            } else {
                i128::from(x)
            }
        };
        let digits: Vec<i128> = {
            let mut rest = q;
            (0..k)
                .map(|_| {
                    let d = i128::from(rest % base);
                    rest /= base;
                    d
                })
                .collect()
        };
        let overflow = || Error("inversion overflow");
        let v = &ct.c1[..m];
        let mut s = Vec::with_capacity(n);
        for i in 0..n {
            // The k entries of R^T v_Abar + v_G for gadget block i.
            let block: Vec<u64> = (0..k)
                .map(|j| {
                    let column = i * k + j;
                    let total: i128 = (0..rows)
                        .map(|a| i128::from(r[a * w + column]) * i128::from(v[a]))
                        .sum::<i128>()
                        + i128::from(v[free + column]);
                    field.reduce_signed(total)
                })
                .collect();
            // c = S_k^T block (mod q) = S_k^T e, lifted to (-q/2, q/2].
            let mut c: Vec<i128> = (0..k - 1)
                .map(|t| {
                    let value = field.sub(field.mul(base % q, block[t]), block[t + 1]);
                    lift(value)
                })
                .collect();
            let mut last = 0u64;
            for (j, &d) in digits.iter().enumerate() {
                last = field.add(last, field.mul(d as u64, block[j]));
            }
            c.push(lift(last));
            // e_j = b^j e_0 - D_j with D_(j+1) = b D_j + c_j, and q e_0 = c_(k-1) + sum q_j D_j.
            let mut d = 0i128;
            let mut sum = c[k - 1];
            for j in 0..k {
                sum = digits[j]
                    .checked_mul(d)
                    .and_then(|x| sum.checked_add(x))
                    .ok_or_else(overflow)?;
                if j + 1 < k {
                    d = d
                        .checked_mul(i128::from(base))
                        .and_then(|x| x.checked_add(c[j]))
                        .ok_or_else(overflow)?;
                }
            }
            if sum % i128::from(q) != 0 {
                return Err(Error("inversion failed: error too large"));
            }
            let e0 = sum / i128::from(q);
            s.push(field.reduce_signed(i128::from(block[0]) - e0));
        }
        // The remaining error v - A0^T s must be small.
        let image = transpose_matvec(&pk.a0, n, m, &s, q)?;
        if image
            .iter()
            .zip(v)
            .any(|(&x, &v)| lift(field.sub(v, x)).unsigned_abs() >= u128::from(q / 4))
        {
            return Err(Error("inversion failed: residual is not small"));
        }
        Ok(s)
    }
}

/// `c0 - d_id^T c1` for the identity key d_id = (e1; e2) (Epoque eq. 27).
pub fn identity_residual(key: &IdentityKey, ct: &Ciphertext) -> Result<Vec<u64>> {
    key_residual(key.profile, &key.e1t, &key.e2t, true, ct)
}

/// Decryption with an extracted identity key (Epoque eq. 27).
pub fn decrypt_identity(key: &IdentityKey, ct: &Ciphertext) -> Result<Vec<u8>> {
    Ok(decode_residual(key.profile, identity_residual(key, ct)?))
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::{SeedableRng, rngs::StdRng};

    fn moments(values: impl Iterator<Item = f64>) -> (f64, f64, usize) {
        let values: Vec<f64> = values.collect();
        let count = values.len();
        let mean = values.iter().sum::<f64>() / count as f64;
        let variance = values.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (count - 1) as f64;
        (mean, variance, count)
    }

    #[test]
    fn presets_satisfy_the_stated_conditions() {
        for tp in [TrapdoorProfile::small(), TrapdoorProfile::full()] {
            let p = tp.profile();
            assert!(p.m() >= construction1_bits(p.n(), p.q(), 8).unwrap());
            assert_eq!(
                tp.padding() + tp.trapdoor_rows() + tp.gadget_columns(),
                p.m()
            );
            assert_eq!(tp.trapdoor_rows(), 3 * p.n());
            assert!(u128::from(tp.base()).pow(tp.digits() as u32) > u128::from(p.q()));
            assert!(u128::from(tp.base()).pow(tp.digits() as u32 - 1) <= u128::from(p.q()));
            let sd = p.q() as f64 / p.denominator() as f64;
            assert!(sd * (2.0 * PI).sqrt() >= 2.0 * (p.n() as f64).sqrt());
        }
        let full = TrapdoorProfile::full();
        assert_eq!(full.digits(), 19);
        assert_eq!(full.gadget_columns(), 25_840);
        // One column fewer breaks Construction 1's leftover-hash condition for binary keys.
        let preset = TrapdoorProfile::small();
        let small = preset.profile();
        let short = Profile::new(16, small.m() - 1, small.q(), 8, 8, small.c_bound()).unwrap();
        assert!(TrapdoorProfile::new(short, 4, preset.width()).is_err());
        // A width below the MP12 bound and a C below the key bound are rejected.
        assert!(TrapdoorProfile::new(small, 4, preset.width() - 1.0).is_err());
        let weak = Profile::new(16, small.m(), small.q(), 8, 8, small.c_bound() - 1).unwrap();
        assert!(TrapdoorProfile::new(weak, 4, preset.width()).is_err());
    }

    #[test]
    fn one_dimensional_sampler_has_the_right_moments() {
        let mut rng = StdRng::seed_from_u64(1);
        for (center, width) in [(0.0, 5.6), (0.37, 5.6), (-12.8, 23.0), (1000.5, 412.0)] {
            let (mean, variance, count) =
                moments((0..40_000).map(|_| sample_z(center, width, &mut rng).unwrap() as f64));
            let expected = width * width / (2.0 * PI);
            let sd = expected.sqrt();
            assert!(
                (mean - center).abs() < 5.0 * sd / (count as f64).sqrt(),
                "{mean} {center}"
            );
            assert!(
                (variance / expected - 1.0).abs() < 0.05,
                "{variance} {expected}"
            );
        }
    }

    #[test]
    fn gadget_samples_lie_in_the_coset_with_the_right_width() {
        let mut rng = StdRng::seed_from_u64(2);
        let preset = TrapdoorProfile::small();
        let mut tried = 0;
        for base in [2u64, 3, 4, 7] {
            let p = preset.profile();
            let Ok(tp) = TrapdoorProfile::new(p, base, preset.width()) else {
                continue;
            };
            tried += 1;
            let gadget = Gadget::new(&tp);
            let k = tp.digits();
            let (mut z, mut center) = (vec![0i64; k], vec![0.0f64; k]);
            let mut values = Vec::new();
            for trial in 0..4000u64 {
                let target = (trial * 9_781_237) % p.q();
                gadget
                    .sample(target, &mut z, &mut center, &mut rng)
                    .unwrap();
                let mut total = 0i128;
                let mut power = 1i128;
                for &digit in &z {
                    total += i128::from(digit) * power;
                    power *= i128::from(base);
                }
                assert_eq!(total.rem_euclid(i128::from(p.q())), i128::from(target));
                values.extend(z.iter().map(|&x| x as f64));
            }
            let (_, variance, _) = moments(values.into_iter());
            let expected = tp.gadget_width().powi(2) / (2.0 * PI);
            assert!(
                (variance / expected - 1.0).abs() < 0.05,
                "base {base}: {variance} {expected}"
            );
        }
        assert!(
            tried >= 3,
            "only {tried} bases meet the small profile's conditions"
        );
    }

    #[test]
    fn cholesky_reproduces_the_matrix() {
        let a = [4.0, 0.0, 0.0, 2.0, 5.0, 0.0, -2.0, 1.0, 6.0];
        let mut l = a;
        assert!(cholesky(&mut l, 3));
        for i in 0..3 {
            for j in 0..=i {
                let product: f64 = (0..=j).map(|t| l[i * 3 + t] * l[j * 3 + t]).sum();
                assert!((product - a[i * 3 + j]).abs() < 1e-12);
            }
        }
        let mut indefinite = [1.0, 0.0, 2.0, 1.0];
        assert!(!cholesky(&mut indefinite, 2));
    }

    /// sum over R[a][c] != 0 of R[a][c] x_a x_(m-w+c), averaged over pairs and samples.
    fn linked_covariance(
        x: &[i32],
        r: &[i8],
        rows: usize,
        m: usize,
        w: usize,
        count: usize,
    ) -> (f64, usize) {
        let mut sum = 0.0;
        let mut pairs = 0usize;
        for j in 0..count {
            let sample = &x[j * m..(j + 1) * m];
            for a in 0..rows {
                for c in 0..w {
                    let sign = r[a * w + c];
                    if sign != 0 {
                        sum +=
                            f64::from(sign) * f64::from(sample[a]) * f64::from(sample[m - w + c]);
                        pairs += 1;
                    }
                }
            }
        }
        (sum / pairs as f64, pairs)
    }

    #[test]
    fn perturbations_carry_the_negative_trapdoor_covariance() {
        // Sigma_p = s^2 I - sigma_G^2 T T^T: Abar and gadget coordinates linked by
        // R[a][c] = +-1 have covariance -R[a][c] sigma_G^2/(2 pi).
        let tp = TrapdoorProfile::small();
        let p = tp.profile();
        let mut rng = StdRng::seed_from_u64(4);
        let (_, td, _) = trapgen(&tp, &mut rng).unwrap();
        let count = 2000;
        let x = perturbations(&tp, &td, count, &mut rng).unwrap();
        let (m, w) = (p.m(), tp.gadget_columns());
        let rows = tp.trapdoor_rows();
        let (covariance, pairs) = linked_covariance(&x, &td.r, rows, m, w, count / 10);
        let expected = -tp.gadget_width().powi(2) / (2.0 * PI);
        let error = tp.width().powi(2) / (2.0 * PI) / (pairs as f64).sqrt();
        assert!(
            (covariance - expected).abs() < 5.0 * error,
            "{covariance} vs {expected} +- {error}"
        );
        let (covariance, pairs) = linked_covariance(&x, &td.r, rows, m, w, count);
        let error = tp.width().powi(2) / (2.0 * PI) / (pairs as f64).sqrt();
        assert!(
            (covariance - expected).abs() < 5.0 * error,
            "{covariance} vs {expected} +- {error}"
        );
        assert!(expected.abs() > 6.0 * error);
    }

    #[test]
    fn preimages_are_spherical_gaussians_of_width_s() {
        // x = p + T z: the perturbation cancels the correlation that T z adds, and
        // every block (Abar, padding, gadget) has variance s^2/(2 pi).
        let tp = TrapdoorProfile::small();
        let p = tp.profile();
        let mut rng = StdRng::seed_from_u64(3);
        let (a, td, _) = trapgen(&tp, &mut rng).unwrap();
        let gadget = Gadget::new(&tp);
        let count = 1000;
        let targets = uniform(p.n() * count, p.q(), &mut rng).unwrap();
        let x = sample_preimages(&tp, &a, &td, &gadget, &targets, count, &mut rng).unwrap();
        let (rows, m, w) = (tp.trapdoor_rows(), p.m(), tp.gadget_columns());
        let expected = tp.width().powi(2) / (2.0 * PI);
        let xs = &x;
        for (name, range) in [
            ("abar", 0..rows),
            ("padding", rows..m - w),
            ("gadget", m - w..m),
        ] {
            let (mean, variance, samples) = moments(
                (0..count).flat_map(|j| range.clone().map(move |c| f64::from(xs[j * m + c]))),
            );
            assert!(
                mean.abs() < 5.0 * expected.sqrt() / (samples as f64).sqrt(),
                "{name} mean {mean}"
            );
            assert!(
                (variance / expected - 1.0).abs() < 6.0 * (2.0 / samples as f64).sqrt(),
                "{name}: variance {variance}, expected {expected}"
            );
        }
        // Without the perturbation's off-diagonal block this would be +sigma_G^2/(2 pi).
        let (covariance, pairs) = linked_covariance(&x, &td.r, rows, m, w, count);
        let error = expected / (pairs as f64).sqrt();
        let omitted = tp.gadget_width().powi(2) / (2.0 * PI);
        assert!(covariance.abs() < 4.0 * error, "{covariance} +- {error}");
        assert!(omitted > 5.0 * error);
    }

    /// The row-by-row Cholesky factorization that `cholesky` reorganizes.
    fn cholesky_row_by_row(a: &mut [f64], n: usize) -> bool {
        for i in 0..n {
            let (done, rest) = a.split_at_mut(i * n);
            let row = &mut rest[..n];
            for j in 0..i {
                let previous = &done[j * n..j * n + j + 1];
                row[j] = (row[j] - dot(&row[..j], &previous[..j])) / previous[j];
            }
            let diagonal = row[i] - dot(&row[..i], &row[..i]);
            if !(diagonal.is_finite() && diagonal > 0.0) {
                return false;
            }
            row[i] = diagonal.sqrt();
        }
        true
    }

    #[test]
    fn panelled_cholesky_is_the_row_by_row_factor_bit_for_bit() {
        use crate::arithmetic::with_threads;
        // Symmetric and diagonally dominant, large enough that the panels are split.
        let n = 700;
        let mut rng = StdRng::seed_from_u64(8);
        let mut a = vec![0.0f64; n * n];
        for i in 0..n {
            for j in 0..i {
                a[i * n + j] = unit(&mut rng) * 2.0 - 1.0;
            }
            a[i * n + i] = n as f64 + unit(&mut rng);
        }
        let mut expected = a.clone();
        assert!(cholesky_row_by_row(&mut expected, n));
        for threads in [1, 3, 8] {
            let mut factor = a.clone();
            assert!(with_threads(threads, || cholesky(&mut factor, n)));
            assert!(
                factor.iter().zip(&expected).all(|(x, y)| x.to_bits() == y.to_bits()),
                "{threads} threads"
            );
        }
        // Not positive definite from row 300 on: refused like the row-by-row version.
        a[300 * n + 300] = -1.0;
        let mut expected = a.clone();
        assert!(!cholesky_row_by_row(&mut expected, n));
        assert!(!with_threads(3, || cholesky(&mut a, n)));
    }

    #[test]
    fn perturbations_keep_their_values_and_coins_for_every_thread_count() {
        use crate::arithmetic::with_threads;
        let tp = TrapdoorProfile::small();
        let mut rng = StdRng::seed_from_u64(12);
        let (_, td, _) = trapgen(&tp, &mut rng).unwrap();
        // 4096 perturbations: the products L g are split over the threads.
        let count = 4096;
        let run = |threads| {
            let mut rng = StdRng::seed_from_u64(13);
            let x = with_threads(threads, || perturbations(&tp, &td, count, &mut rng)).unwrap();
            (x, rng.next_u64())
        };
        let one = run(1);
        // The earlier loop: L g computed between drawing g and the padding.
        let reference = {
            let p = tp.profile();
            let (rows, w, m) = (tp.trapdoor_rows(), tp.gadget_columns(), p.m());
            let free = m - w;
            let constants = Perturbation::new(&tp);
            let to_deviation = 1.0 / (2.0 * PI).sqrt();
            let gadget_scale = constants.c22.sqrt() * to_deviation;
            let padding_scale = (tp.width.powi(2) - tp.rounding.powi(2)).sqrt() * to_deviation;
            let mut rng = StdRng::seed_from_u64(13);
            let mut normals = Normals::default();
            let mut gadget_part = vec![0.0f64; count * w];
            let mut free_part = vec![0.0f64; count * free];
            let mut g = vec![0.0f64; rows];
            for j in 0..count {
                for x in &mut gadget_part[j * w..(j + 1) * w] {
                    *x = gadget_scale * normals.next(&mut rng);
                }
                for x in &mut g {
                    *x = normals.next(&mut rng);
                }
                let out = &mut free_part[j * free..(j + 1) * free];
                for (a, x) in out[..rows].iter_mut().enumerate() {
                    *x = to_deviation * dot(&td.factor[a * rows..a * rows + a + 1], &g[..a + 1]);
                }
                for x in &mut out[rows..] {
                    *x = padding_scale * normals.next(&mut rng);
                }
            }
            let shift = ternary_product(&td.r, rows, w, &gadget_part, count);
            let ratio = tp.gadget.powi(2) / constants.c22;
            let mut out = vec![0i32; count * m];
            for j in 0..count {
                for a in 0..free {
                    let mut y = free_part[j * free + a];
                    if a < rows {
                        y -= ratio * shift[a * count + j];
                    }
                    out[j * m + a] = sample_z(y, tp.rounding, &mut rng).unwrap() as i32;
                }
                for c in 0..w {
                    out[j * m + free + c] =
                        sample_z(gadget_part[j * w + c], tp.rounding, &mut rng).unwrap() as i32;
                }
            }
            (out, rng.next_u64())
        };
        assert!(one == reference, "one thread differs from the earlier loop");
        for threads in [2, 5] {
            assert!(run(threads) == one, "{threads} threads");
        }
    }
}
