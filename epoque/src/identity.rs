//! The identity encoding H : Z_q^n -> Z_q^(n x n) of Epoque (BHM21, eq. (19)).
//!
//! H(id) - H(id') must be invertible whenever id != id' (full-rank differences).
//! BHM21 takes the construction of Agrawal, Boneh and Boyen (EUROCRYPT 2010,
//! Section 5), implemented here: fix a monic irreducible f of degree n over Z_q,
//! read id as the polynomial g(X) = id_0 + id_1 X + ... + id_(n-1) X^(n-1), and
//! let row i of H(id) hold the coefficients of X^i g(X) mod f(X). H(id) is then
//! the matrix of multiplication by g in the field `Z_q[X]/(f)`, so H is linear and
//! H(id) - H(id') = H(id - id') is invertible for id != id'. The unit identity
//! (1, 0, ..., 0) gives H = I.
//!
//! f depends only on (n, q), so every party derives the same encoding. For each
//! prime power d that divides n exactly, the first monic polynomial of degree d in a
//! fixed pseudorandom sequence that passes Ben-Or's irreducibility test is taken.
//! The parts are joined by composed sums (whose roots are the sums of the parts'
//! roots). For coprime degrees and q > n a composed sum of irreducible polynomials
//! is irreducible, so f is irreducible of degree n.
use crate::arithmetic::Field;
use crate::{Error, Profile, Result};

/// The encoding for one profile: the coefficients c_0, ..., c_(n-1) of the monic
/// irreducible modulus f(X) = X^n + c_(n-1) X^(n-1) + ... + c_0.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct IdentityEncoding {
    profile: Profile,
    modulus: Vec<u64>,
}

/// An identity id in Z_q^n together with the modulus that defines H(id).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Identity {
    profile: Profile,
    id: Vec<u64>,
    modulus: Vec<u64>,
}

impl IdentityEncoding {
    pub fn new(p: Profile) -> Result<Self> {
        let (n, q) = (p.n, p.q);
        if u128::from(q) <= n as u128 {
            return Err(Error("the identity encoding needs q > n"));
        }
        let field = &p.field;
        let mut modulus: Option<Vec<u64>> = None;
        for degree in prime_power_parts(n) {
            let part = irreducible(field, q, degree)?;
            modulus = Some(match modulus {
                None => part,
                Some(previous) => composed_sum(field, &previous, &part),
            });
        }
        let modulus = modulus.ok_or(Error("the identity encoding needs n >= 2"))?;
        Ok(Self {
            profile: p,
            modulus,
        })
    }

    pub fn profile(&self) -> Profile {
        self.profile
    }

    /// c_0, ..., c_(n-1) of f(X) = X^n + c_(n-1) X^(n-1) + ... + c_0.
    pub fn modulus(&self) -> &[u64] {
        &self.modulus
    }

    /// The identity with coordinates `id` (n canonical residues).
    pub fn identity(&self, id: &[u64]) -> Result<Identity> {
        let p = self.profile;
        if id.len() != p.n || id.iter().any(|&x| x >= p.q) {
            return Err(Error("an identity is n canonical residues"));
        }
        Ok(Identity {
            profile: p,
            id: id.to_vec(),
            modulus: self.modulus.clone(),
        })
    }

    /// The unit identity (1, 0, ..., 0), whose matrix is H = I.
    pub fn unit(&self) -> Identity {
        let mut id = vec![0; self.profile.n];
        id[0] = 1;
        Identity {
            profile: self.profile,
            id,
            modulus: self.modulus.clone(),
        }
    }
}

impl Identity {
    pub fn profile(&self) -> Profile {
        self.profile
    }

    /// The coordinates id_0, ..., id_(n-1).
    pub fn vector(&self) -> &[u64] {
        &self.id
    }

    /// Visit the rows of H(id), X^i g(X) mod f(X) for i = 0, ..., n - 1, in order.
    fn for_each_row(&self, mut visit: impl FnMut(usize, &[u64])) {
        let p = self.profile;
        let (n, field) = (p.n, &p.field);
        let mut row = self.id.clone();
        for i in 0..n {
            visit(i, &row);
            if i + 1 < n {
                // X * row: shift up by one; X^n = -(c_0 + ... + c_(n-1) X^(n-1)).
                let top = row[n - 1];
                for j in (1..n).rev() {
                    row[j] = field.sub(row[j - 1], field.mul(top, self.modulus[j]));
                }
                row[0] = field.neg(field.mul(top, self.modulus[0]));
            }
        }
    }

    /// H(id) as a row-major n by n matrix.
    pub fn matrix(&self) -> Vec<u64> {
        let n = self.profile.n;
        let mut out = Vec::with_capacity(n * n);
        self.for_each_row(|_, row| out.extend_from_slice(row));
        out
    }

    /// H(id)^T s, the coefficients of s(X) g(X) mod f(X), with exact sums reduced once.
    pub fn transpose_times(&self, s: &[u64]) -> Result<Vec<u64>> {
        let p = self.profile;
        if s.len() != p.n || s.iter().any(|&x| x >= p.q) {
            return Err(Error("H(id)^T s needs n canonical residues"));
        }
        let budget = accumulation_budget(p.q);
        let mut acc = vec![0u128; p.n];
        let mut pending = 0;
        self.for_each_row(|i, row| {
            let coefficient = u128::from(s[i]);
            if coefficient == 0 {
                return;
            }
            if pending == budget {
                for x in &mut acc {
                    *x = u128::from(p.field.reduce(*x));
                }
                pending = 0;
            }
            for (sum, &x) in acc.iter_mut().zip(row) {
                *sum += u128::from(x) * coefficient;
            }
            pending += 1;
        });
        Ok(acc.into_iter().map(|x| p.field.reduce(x)).collect())
    }

    /// H(id) X for `count` columns X stored row-major (n by count).
    pub fn times(&self, x: &[u64], count: usize) -> Result<Vec<u64>> {
        let p = self.profile;
        let n = p.n;
        if count == 0 || n.checked_mul(count) != Some(x.len()) || x.iter().any(|&v| v >= p.q) {
            return Err(Error("H(id) X needs n by count canonical residues"));
        }
        let columns: Vec<Vec<u64>> = (0..count)
            .map(|c| (0..n).map(|t| x[t * count + c]).collect())
            .collect();
        let budget = accumulation_budget(p.q);
        let mut out = vec![0u64; n * count];
        self.for_each_row(|i, row| {
            for (c, column) in columns.iter().enumerate() {
                let mut acc = 0u128;
                for (chunk, values) in row.chunks(budget).zip(column.chunks(budget)) {
                    let partial: u128 = chunk
                        .iter()
                        .zip(values)
                        .map(|(&a, &b)| u128::from(a) * u128::from(b))
                        .sum();
                    acc = u128::from(p.field.reduce(acc)) + u128::from(p.field.reduce(partial));
                }
                out[i * count + c] = p.field.reduce(acc);
            }
        });
        Ok(out)
    }
}

/// How many products below (q - 1)^2 a u128 holds on top of a value below q.
fn accumulation_budget(q: u64) -> usize {
    let square = u128::from(q - 1) * u128::from(q - 1);
    usize::try_from(((u128::MAX - u128::from(q)) / square.max(1)).min(1 << 30))
        .expect("budget fits in usize")
        .max(1)
}

/// The prime powers r^e that divide n exactly, in increasing order of r.
fn prime_power_parts(mut n: usize) -> Vec<usize> {
    let mut parts = Vec::new();
    let mut r = 2;
    while r * r <= n {
        if n.is_multiple_of(r) {
            let mut power = 1;
            while n.is_multiple_of(r) {
                n /= r;
                power *= r;
            }
            parts.push(power);
        }
        r += 1;
    }
    if n > 1 {
        parts.push(n);
    }
    parts
}

/// SplitMix64: the fixed stream of candidate coefficients.
struct Stream(u64);
impl Stream {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
}

/// The first monic irreducible polynomial of degree d in the stream for (q, d).
fn irreducible(field: &Field, q: u64, d: usize) -> Result<Vec<u64>> {
    let mut stream = Stream(q ^ (d as u64).rotate_left(32) ^ 0x6964_656e_7469_7479);
    // About one candidate in d is irreducible; the cap is never reached in practice.
    for _ in 0..64 * d + 4096 {
        let candidate: Vec<u64> = (0..d).map(|_| stream.next() % q).collect();
        if is_irreducible(field, q, &candidate) {
            return Ok(candidate);
        }
    }
    Err(Error(
        "no irreducible polynomial found for the identity encoding",
    ))
}

/// Remove leading zero coefficients.
fn trim(v: &mut Vec<u64>) {
    while v.last() == Some(&0) {
        v.pop();
    }
}

/// Reduce `poly` modulo the monic f (lower coefficients `f`) to its low d coefficients.
fn reduce_mod(field: &Field, mut poly: Vec<u64>, f: &[u64]) -> Vec<u64> {
    let d = f.len();
    for k in (d..poly.len()).rev() {
        let top = poly[k];
        if top == 0 {
            continue;
        }
        for (j, &c) in f.iter().enumerate() {
            poly[k - d + j] = field.sub(poly[k - d + j], field.mul(top, c));
        }
    }
    poly.resize(d, 0);
    poly
}

/// a b mod f for a and b of degree below deg f.
fn mul_mod(field: &Field, a: &[u64], b: &[u64], f: &[u64]) -> Vec<u64> {
    let mut product = vec![0u64; a.len() + b.len() - 1];
    for (i, &x) in a.iter().enumerate() {
        if x == 0 {
            continue;
        }
        for (j, &y) in b.iter().enumerate() {
            product[i + j] = field.add(product[i + j], field.mul(x, y));
        }
    }
    reduce_mod(field, product, f)
}

/// X^exponent mod f.
fn x_power(field: &Field, exponent: u64, f: &[u64]) -> Vec<u64> {
    let d = f.len();
    let mut base = vec![0u64; d];
    if d == 1 {
        base[0] = field.neg(f[0]);
    } else {
        base[1] = 1;
    }
    let mut result = vec![0u64; d];
    result[0] = 1;
    let mut e = exponent;
    while e > 0 {
        if e & 1 == 1 {
            result = mul_mod(field, &result, &base, f);
        }
        e >>= 1;
        if e > 0 {
            base = mul_mod(field, &base, &base, f);
        }
    }
    result
}

/// The remainder of a modulo a nonzero trimmed b.
fn remainder(field: &Field, a: &[u64], b: &[u64]) -> Vec<u64> {
    let mut r = a.to_vec();
    trim(&mut r);
    let lead = field.inverse(*b.last().expect("nonzero divisor"));
    while r.len() >= b.len() {
        let shift = r.len() - b.len();
        let factor = field.mul(*r.last().expect("nonempty"), lead);
        for (j, &x) in b.iter().enumerate() {
            r[shift + j] = field.sub(r[shift + j], field.mul(factor, x));
        }
        trim(&mut r);
    }
    r
}

/// Whether gcd(f, t) is a nonzero constant, for monic f and t of lower degree.
fn coprime(field: &Field, f: &[u64], t: &[u64]) -> bool {
    let mut a = f.to_vec();
    a.push(1);
    let mut b = t.to_vec();
    trim(&mut b);
    while !b.is_empty() {
        let r = remainder(field, &a, &b);
        a = b;
        b = r;
    }
    a.len() == 1
}

/// Ben-Or: a monic f of degree d is irreducible exactly when
/// gcd(f, X^(q^i) - X) = 1 for every i from 1 to d/2.
fn is_irreducible(field: &Field, q: u64, f: &[u64]) -> bool {
    let d = f.len();
    if d == 1 {
        return true;
    }
    let minus_x = |h: &[u64]| {
        let mut t = h.to_vec();
        t[1] = field.sub(t[1], 1);
        t
    };
    let xq = x_power(field, q, f);
    if !coprime(field, f, &minus_x(&xq)) {
        return false;
    }
    // h -> h^q mod f is linear over Z_q: h^q = sum_j h_j (X^q)^j.
    let mut powers = Vec::with_capacity(d);
    let mut current = vec![0u64; d];
    current[0] = 1;
    for _ in 0..d {
        let next = mul_mod(field, &current, &xq, f);
        powers.push(current);
        current = next;
    }
    let mut h = xq;
    for _ in 2..=d / 2 {
        let mut next = vec![0u64; d];
        for (j, &c) in h.iter().enumerate() {
            if c == 0 {
                continue;
            }
            for (k, &x) in powers[j].iter().enumerate() {
                next[k] = field.add(next[k], field.mul(c, x));
            }
        }
        h = next;
        if !coprime(field, f, &minus_x(&h)) {
            return false;
        }
    }
    true
}

/// p_0, ..., p_count: the power sums of the roots of the monic f (Newton's identities).
fn power_sums(field: &Field, f: &[u64], count: usize) -> Vec<u64> {
    let d = f.len();
    // Elementary symmetric functions e_i = (-1)^i c_(d-i).
    let e: Vec<u64> = (0..=d)
        .map(|i| match i {
            0 => 1,
            _ if i % 2 == 1 => field.neg(f[d - i]),
            _ => f[d - i],
        })
        .collect();
    let mut p = vec![0u64; count + 1];
    p[0] = field.reduce(d as u128);
    for k in 1..=count {
        let mut total = 0;
        for i in 1..=k.min(d) {
            let term = if i < k {
                field.mul(e[i], p[k - i])
            } else {
                field.mul(field.reduce(k as u128), e[k])
            };
            total = if i % 2 == 1 {
                field.add(total, term)
            } else {
                field.sub(total, term)
            };
        }
        p[k] = total;
    }
    p
}

/// The monic polynomial of degree d whose roots have power sums p_1, ..., p_d (q > d).
fn from_power_sums(field: &Field, p: &[u64], d: usize) -> Vec<u64> {
    let mut e = vec![0u64; d + 1];
    e[0] = 1;
    for k in 1..=d {
        let mut total = 0;
        for i in 1..=k {
            let term = field.mul(e[k - i], p[i]);
            total = if i % 2 == 1 {
                field.add(total, term)
            } else {
                field.sub(total, term)
            };
        }
        e[k] = field.mul(total, field.inverse(field.reduce(k as u128)));
    }
    (0..d)
        .map(|j| {
            let k = d - j;
            if k % 2 == 1 { field.neg(e[k]) } else { e[k] }
        })
        .collect()
}

/// The composed sum of monic f and g: the monic polynomial whose roots are
/// alpha + beta over all roots alpha of f and beta of g.
fn composed_sum(field: &Field, f: &[u64], g: &[u64]) -> Vec<u64> {
    let d = f.len() * g.len();
    let (pf, pg) = (power_sums(field, f, d), power_sums(field, g, d));
    let mut factorial = vec![1u64; d + 1];
    for k in 1..=d {
        factorial[k] = field.mul(factorial[k - 1], field.reduce(k as u128));
    }
    let mut inverse = vec![1u64; d + 1];
    inverse[d] = field.inverse(factorial[d]);
    for k in (1..=d).rev() {
        inverse[k - 1] = field.mul(inverse[k], field.reduce(k as u128));
    }
    // p_k(alpha + beta) = sum_l C(k, l) p_l(alpha) p_(k-l)(beta).
    let a: Vec<u64> = (0..=d).map(|l| field.mul(pf[l], inverse[l])).collect();
    let b: Vec<u64> = (0..=d).map(|l| field.mul(pg[l], inverse[l])).collect();
    let mut p = vec![0u64; d + 1];
    for k in 1..=d {
        let mut total = 0;
        for l in 0..=k {
            total = field.add(total, field.mul(a[l], b[k - l]));
        }
        p[k] = field.mul(total, factorial[k]);
    }
    from_power_sums(field, &p, d)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn field(q: u64) -> Field {
        Field::new(q).unwrap()
    }

    #[test]
    fn ben_or_separates_known_polynomials() {
        let f7 = field(7);
        // X^2 + 1 is irreducible mod 7 (7 = 3 mod 4), X^2 - 1 is not.
        assert!(is_irreducible(&f7, 7, &[1, 0]));
        assert!(!is_irreducible(&f7, 7, &[6, 0]));
        // X^3 - 2: 2 is not a cube mod 7. X^3 - 1 has the root 1.
        assert!(is_irreducible(&f7, 7, &[5, 0, 0]));
        assert!(!is_irreducible(&f7, 7, &[6, 0, 0]));
        // (X^2 + 1)^2 = X^4 + 2X^2 + 1 has no root but is reducible.
        assert!(!is_irreducible(&f7, 7, &[1, 0, 2, 0]));
    }

    #[test]
    fn composed_sums_add_roots() {
        let f = field(65537);
        // (X - 1) and (X - 2) give X - 3.
        assert_eq!(composed_sum(&f, &[65536], &[65535]), vec![65534]);
        // X^2 + 1 (roots +-i) and X - 5 give (X - 5)^2 + 1 = X^2 - 10X + 26.
        assert_eq!(composed_sum(&f, &[1, 0], &[65532]), vec![26, 65527]);
    }

    #[test]
    fn moduli_are_irreducible_of_degree_n() {
        for (n, q) in [(4, 65537), (12, 65537), (16, 39_130_261), (30, 691_947_401)] {
            let p = Profile::new(n, n + 1, q, 1, 1, 1).unwrap();
            let encoding = IdentityEncoding::new(p).unwrap();
            assert_eq!(encoding.modulus().len(), n);
            assert!(is_irreducible(&p.field, q, encoding.modulus()), "n {n}");
        }
        assert_eq!(prime_power_parts(1984), [64, 31]);
        assert_eq!(prime_power_parts(300), [4, 3, 25]);
    }
}
