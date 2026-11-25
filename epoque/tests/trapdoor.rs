//! Epoque's trapdoor KeyGen, master shortcut key and Extr for any identity, with the
//! anamorphic broadcast of Construction 1 inside.
//!
//! Relations are rechecked here with arbitrary integers, independently of the
//! library's products, and H(id) is rebuilt from the public modulus f
//! (`common::rot_f`). The modulus passes Rabin's irreducibility test, written here
//! independently of the library's Ben-Or test, so H has full-rank differences.
//! Every ciphertext must decrypt to its ordinary message with the master shortcut
//! key (D1, D2) and with the identity key extracted for its identity. On the same
//! anamorphic ciphertext every recipient in S must recover the intended hidden
//! message holding only its own key ask_i. The extraction master key T_A0 must
//! expose the hidden constraint (security boundary).
mod common;

use epoque_shortcut::arithmetic::uniform;
use epoque_shortcut::trapdoor::*;
use epoque_shortcut::*;
use rand::{Rng, RngExt, SeedableRng, rngs::StdRng};
use rug::Integer;

fn residue(x: &Integer, q: u64) -> u64 {
    integer_mod(x, q)
}

fn small_setup(seed: u64) -> (TrapdoorProfile, TrapdoorSetup, StdRng) {
    let tp = TrapdoorProfile::small();
    let mut rng = StdRng::seed_from_u64(seed);
    let setup = setup_trapdoor(&tp, &mut rng, |_, _| {}).unwrap();
    (tp, setup, rng)
}

/// H B mod q for H row-major n by n: the matrix that H(id) adds to A1 in F_id.
fn h_times_b(pk: &PublicKey, h: &[u64]) -> Vec<u64> {
    let p = pk.profile();
    let (n, m, q) = (p.n(), p.m(), u128::from(p.q()));
    let mut out = Vec::with_capacity(n * m);
    for i in 0..n {
        let mut row = vec![0u128; m];
        for t in 0..n {
            let coefficient = u128::from(h[i * n + t]);
            for (sum, &x) in row.iter_mut().zip(&pk.b()[t * m..(t + 1) * m]) {
                *sum = (*sum + coefficient * u128::from(x)) % q;
            }
        }
        out.extend(row.into_iter().map(|x| x as u64));
    }
    out
}

/// (A0 X1 + (A1 + H B) X2)[i][j] for keys stored transposed (J by m), with `h` the
/// row-major H(id) (None leaves out H B), as exact checked i128 sums (each term is
/// below 2^94 and m below 2^17), reduced once.
fn key_image(pk: &PublicKey, first_t: &[i32], second_t: &[i32], h: Option<&[u64]>) -> Vec<u64> {
    let p = pk.profile();
    let (n, m, q, jb) = (p.n(), p.m(), p.q(), p.ordinary_bits());
    let hb = h.map(|h| h_times_b(pk, h));
    let term = |a: u64, x: i32| i128::from(a).checked_mul(i128::from(x)).unwrap();
    let mut out = Vec::with_capacity(n * jb);
    for i in 0..n {
        let row = i * m..(i + 1) * m;
        let (a0, a1) = (&pk.a0()[row.clone()], &pk.a1()[row.clone()]);
        let hb = hb.as_ref().map(|hb| &hb[row]);
        for j in 0..jb {
            let (x1, x2) = (&first_t[j * m..(j + 1) * m], &second_t[j * m..(j + 1) * m]);
            let mut total = 0i128;
            for k in 0..m {
                total = total.checked_add(term(a0[k], x1[k])).unwrap();
                total = total.checked_add(term(a1[k], x2[k])).unwrap();
                if let Some(hb) = hb {
                    total = total.checked_add(term(hb[k], x2[k])).unwrap();
                }
            }
            out.push(total.rem_euclid(i128::from(q)) as u64);
        }
    }
    out
}

fn standard_deviation(values: &[i32]) -> f64 {
    let n = values.len() as f64;
    let mean = values.iter().map(|&x| f64::from(x)).sum::<f64>() / n;
    (values
        .iter()
        .map(|&x| (f64::from(x) - mean).powi(2))
        .sum::<f64>()
        / (n - 1.0))
        .sqrt()
}

/// Every hidden message of up to four bits, in counting order. Longer messages: all
/// zeros, all ones and both alternating patterns, so every bit takes both values.
fn hidden_messages(bits: usize) -> Vec<Vec<u8>> {
    if bits <= 4 {
        (0..1usize << bits)
            .map(|v| (0..bits).map(|i| ((v >> i) & 1) as u8).collect())
            .collect()
    } else {
        vec![
            vec![0; bits],
            vec![1; bits],
            (0..bits).map(|i| (i % 2) as u8).collect(),
            (0..bits).map(|i| ((i + 1) % 2) as u8).collect(),
        ]
    }
}

fn random_bits(count: usize, rng: &mut StdRng) -> Vec<u8> {
    (0..count).map(|_| rng.random_range(0..2)).collect()
}

/// The unit identity and two uniformly random ones.
fn identities(pk: &PublicKey, rng: &mut StdRng) -> Vec<Identity> {
    vec![
        pk.unit_identity(),
        common::random_identity(pk, rng),
        common::random_identity(pk, rng),
    ]
}

/// A profile that differs from the small one only in g, for cross-parameter checks.
fn other_profile() -> Profile {
    let p = TrapdoorProfile::small().profile();
    Profile::new(p.n(), p.m(), p.q(), p.ordinary_bits(), 7, p.c_bound()).unwrap()
}

fn pow_mod(base: u64, mut exponent: u64, q: u64) -> u64 {
    let (mut result, q128) = (1u128, u128::from(q));
    let mut b = u128::from(base) % q128;
    while exponent > 0 {
        if exponent & 1 == 1 {
            result = result * b % q128;
        }
        b = b * b % q128;
        exponent >>= 1;
    }
    result as u64
}

fn mul_mod(a: u64, b: u64, q: u64) -> u64 {
    (u128::from(a) * u128::from(b) % u128::from(q)) as u64
}

fn sub_mod(a: u64, b: u64, q: u64) -> u64 {
    ((u128::from(a) + u128::from(q) - u128::from(b)) % u128::from(q)) as u64
}

/// Z_q[X]/(f) for the monic f with lower coefficients `f`, residues lowest first.
struct Quotient<'a> {
    f: &'a [u64],
    q: u64,
}

impl Quotient<'_> {
    fn mul(&self, a: &[u64], b: &[u64]) -> Vec<u64> {
        let (n, q) = (self.f.len(), u128::from(self.q));
        let mut product = vec![0u128; 2 * n - 1];
        for (i, &x) in a.iter().enumerate() {
            for (j, &y) in b.iter().enumerate() {
                product[i + j] = (product[i + j] + u128::from(x) * u128::from(y)) % q;
            }
        }
        // X^k = X^(k-n) X^n with X^n = -(c_0 + ... + c_(n-1) X^(n-1)), highest first.
        for k in (n..2 * n - 1).rev() {
            let top = product[k];
            for (j, &c) in self.f.iter().enumerate() {
                let t = &mut product[k - n + j];
                *t = (*t + q - top * u128::from(c) % q) % q;
            }
        }
        product[..n].iter().map(|&x| x as u64).collect()
    }

    fn pow(&self, base: &[u64], mut exponent: u64) -> Vec<u64> {
        let mut result = vec![0; self.f.len()];
        result[0] = 1;
        let mut base = base.to_vec();
        while exponent > 0 {
            if exponent & 1 == 1 {
                result = self.mul(&result, &base);
            }
            base = self.mul(&base, &base);
            exponent >>= 1;
        }
        result
    }
}

fn trim(mut a: Vec<u64>) -> Vec<u64> {
    while a.last() == Some(&0) {
        a.pop();
    }
    a
}

/// gcd(a, b) is a nonzero constant (plain Euclid over Z_q).
fn coprime(a: Vec<u64>, b: Vec<u64>, q: u64) -> bool {
    let (mut a, mut b) = (trim(a), trim(b));
    while !b.is_empty() {
        let inverse = pow_mod(*b.last().unwrap(), q - 2, q);
        while a.len() >= b.len() {
            let shift = a.len() - b.len();
            let factor = mul_mod(*a.last().unwrap(), inverse, q);
            for (j, &c) in b.iter().enumerate() {
                a[shift + j] = sub_mod(a[shift + j], mul_mod(factor, c, q), q);
            }
            a = trim(a);
        }
        std::mem::swap(&mut a, &mut b);
    }
    a.len() == 1
}

fn prime_divisors(mut n: usize) -> Vec<usize> {
    let mut primes = Vec::new();
    let mut r = 2;
    while r * r <= n {
        if n.is_multiple_of(r) {
            primes.push(r);
            while n.is_multiple_of(r) {
                n /= r;
            }
        }
        r += 1;
    }
    if n > 1 {
        primes.push(n);
    }
    primes
}

/// Rabin's test: the monic f of degree n >= 2 is irreducible over Z_q iff
/// X^(q^n) = X mod f and gcd(X^(q^(n/r)) - X, f) = 1 for every prime r dividing n.
fn rabin_irreducible(f: &[u64], q: u64) -> bool {
    let n = f.len();
    let ring = Quotient { f, q };
    let mut x = vec![0u64; n];
    x[1] = 1;
    // frobenius[k] = X^(q^k) mod f.
    let mut frobenius = vec![x.clone()];
    for _ in 0..n {
        let next = ring.pow(frobenius.last().unwrap(), q);
        frobenius.push(next);
    }
    if frobenius[n] != x {
        return false;
    }
    let mut monic = f.to_vec();
    monic.push(1);
    prime_divisors(n).into_iter().all(|r| {
        let mut difference = frobenius[n / r].clone();
        difference[1] = sub_mod(difference[1], 1, q);
        coprime(difference, monic.clone(), q)
    })
}

/// The rank of a row-major matrix over Z_q, by Gaussian elimination.
fn rank_mod(mut a: Vec<u64>, rows: usize, cols: usize, q: u64) -> usize {
    let mut rank = 0;
    for col in 0..cols {
        let Some(pivot) = (rank..rows).find(|&r| a[r * cols + col] != 0) else {
            continue;
        };
        for c in 0..cols {
            a.swap(pivot * cols + c, rank * cols + c);
        }
        let inverse = pow_mod(a[rank * cols + col], q - 2, q);
        for r in 0..rows {
            let value = a[r * cols + col];
            if r != rank && value != 0 {
                let factor = mul_mod(value, inverse, q);
                for c in col..cols {
                    let x = mul_mod(factor, a[rank * cols + c], q);
                    a[r * cols + c] = sub_mod(a[r * cols + c], x, q);
                }
            }
        }
        rank += 1;
    }
    rank
}

#[test]
fn identity_matrices_are_rot_f_of_the_identity_and_the_unit_gives_i() {
    let p = TrapdoorProfile::small().profile();
    let (n, q) = (p.n(), p.q());
    let encoding = IdentityEncoding::new(p).unwrap();
    // f depends only on (n, q): every party derives the same encoding.
    assert_eq!(IdentityEncoding::new(p).unwrap(), encoding);
    assert_eq!(encoding.modulus().len(), n);
    let mut eye = vec![0u64; n * n];
    for i in 0..n {
        eye[i * n + i] = 1;
    }
    let unit = encoding.unit();
    assert_eq!(unit.vector()[0], 1);
    assert!(unit.vector()[1..].iter().all(|&x| x == 0));
    assert_eq!(unit.matrix(), eye);
    assert_eq!(common::rot_f(encoding.modulus(), unit.vector(), q), eye);
    let mut rng = StdRng::seed_from_u64(31);
    for _ in 0..8 {
        let coordinates: Vec<u64> = (0..n).map(|_| rng.random_range(0..q)).collect();
        let id = encoding.identity(&coordinates).unwrap();
        assert_eq!(id.vector(), coordinates.as_slice());
        let h = common::rot_f(encoding.modulus(), &coordinates, q);
        assert_eq!(id.matrix(), h);
        // H(id)^T s and H(id) X agree with the matrix.
        let s = uniform(n, q, &mut rng).unwrap();
        let expected: Vec<u64> = (0..n)
            .map(|c| {
                let total: u128 = (0..n)
                    .map(|r| u128::from(h[r * n + c]) * u128::from(s[r]))
                    .sum();
                (total % u128::from(q)) as u64
            })
            .collect();
        assert_eq!(id.transpose_times(&s).unwrap(), expected);
        let count = 3;
        let x = uniform(n * count, q, &mut rng).unwrap();
        let expected: Vec<u64> = (0..n * count)
            .map(|e| {
                let (i, c) = (e / count, e % count);
                let total: u128 = (0..n)
                    .map(|t| u128::from(h[i * n + t]) * u128::from(x[t * count + c]))
                    .sum();
                (total % u128::from(q)) as u64
            })
            .collect();
        assert_eq!(id.times(&x, count).unwrap(), expected);
    }
    // Malformed identities and operands are refused.
    assert!(encoding.identity(&vec![0; n - 1]).is_err());
    assert!(encoding.identity(&vec![0; n + 1]).is_err());
    let mut bad = vec![0; n];
    bad[3] = q;
    assert!(encoding.identity(&bad).is_err());
    assert!(unit.transpose_times(&vec![0; n + 1]).is_err());
    assert!(unit.transpose_times(&bad).is_err());
    assert!(unit.times(&[], 0).is_err());
    assert!(unit.times(&vec![0; 2 * n - 1], 2).is_err());
}

#[test]
fn identity_moduli_are_irreducible_so_h_has_full_rank_differences() {
    // Rabin's test itself: X^2 + 1 is irreducible over Z_7 and splits over Z_5,
    // X^3 + 2 is irreducible over Z_7, X^4 + 1 splits over Z_17, and
    // (X^2 + 1)(X^2 + X + 3) over Z_7 passes the first condition but not the second.
    assert!(rabin_irreducible(&[1, 0], 7));
    assert!(!rabin_irreducible(&[1, 0], 5));
    assert!(rabin_irreducible(&[2, 0, 0], 7));
    assert!(!rabin_irreducible(&[1, 0, 0, 0], 17));
    assert!(!rabin_irreducible(&[3, 1, 4, 1], 7));
    // The encodings' moduli, one prime power or composed sums of several.
    for (n, q) in [
        (7, 65_537),
        (12, 65_537),
        (16, 39_130_261),
        (30, 691_947_401),
        (36, 1_000_003),
    ] {
        let p = Profile::new(n, n + 1, q, 1, 1, 1).unwrap();
        let encoding = IdentityEncoding::new(p).unwrap();
        assert_eq!(encoding.modulus().len(), n);
        assert!(
            rabin_irreducible(encoding.modulus(), q),
            "f is reducible for n = {n}, q = {q}"
        );
    }
    // Eq. (19) directly on the small profile: H is linear and H(a) - H(b) has full
    // rank, for random pairs and for pairs that differ in one coordinate.
    let p = TrapdoorProfile::small().profile();
    let (n, q) = (p.n(), p.q());
    let encoding = IdentityEncoding::new(p).unwrap();
    let f = encoding.modulus();
    let mut rng = StdRng::seed_from_u64(32);
    for trial in 0..16 {
        let a: Vec<u64> = (0..n).map(|_| rng.random_range(0..q)).collect();
        let mut b: Vec<u64> = (0..n).map(|_| rng.random_range(0..q)).collect();
        if trial % 2 == 1 {
            b = a.clone();
            let k = trial % n;
            b[k] = (b[k] + rng.random_range(1..q)) % q;
        }
        let (ha, hb) = (common::rot_f(f, &a, q), common::rot_f(f, &b, q));
        let sum: Vec<u64> = a.iter().zip(&b).map(|(&x, &y)| (x + y) % q).collect();
        let added: Vec<u64> = ha.iter().zip(&hb).map(|(&x, &y)| (x + y) % q).collect();
        assert_eq!(common::rot_f(f, &sum, q), added, "H is additive");
        let c = rng.random_range(1..q);
        let scaled: Vec<u64> = a.iter().map(|&x| mul_mod(c, x, q)).collect();
        let times: Vec<u64> = ha.iter().map(|&x| mul_mod(c, x, q)).collect();
        assert_eq!(common::rot_f(f, &scaled, q), times, "H is homogeneous");
        let difference: Vec<u64> = ha
            .iter()
            .zip(&hb)
            .map(|(&x, &y)| sub_mod(x, y, q))
            .collect();
        assert_eq!(rank_mod(difference, n, n, q), n, "H(a) - H(b) is singular");
    }
}

#[test]
fn master_shortcut_key_satisfies_epoque_section_5_2_1_for_every_identity() {
    let (tp, setup, mut rng) = small_setup(1);
    let (pk, key) = (&setup.setup.public, &setup.setup.shortcut);
    let p = pk.profile();
    let (n, m, q, jb) = (p.n(), p.m(), p.q(), p.ordinary_bits());
    let (d1t, d2t) = (key.d1_transposed(), key.d2_transposed());
    // B d2_j = 0 (d2_j in Lambda^perp(B)) on every entry.
    for i in 0..n {
        for j in 0..jb {
            let mut kernel = Integer::new();
            for k in 0..m {
                kernel += Integer::from(pk.b()[i * m + k]) * d2t[j * m + k];
            }
            assert_eq!(residue(&kernel, q), 0, "B D2 at ({i},{j})");
        }
    }
    // A0 d1_j + A1 d2_j = u_j, hence F_id d_j = [A0 | A1 + H(id) B] d_j = u_j for
    // every identity (eq. 29), checked for the unit and two random identities.
    assert_eq!(key_image(pk, d1t, d2t, None), pk.u());
    for id in identities(pk, &mut rng) {
        let h = common::identity_matrix(pk, &id);
        assert_eq!(key_image(pk, d1t, d2t, Some(&h)), pk.u());
    }
    // Width-s Gaussian keys, within the profile's noise bound C.
    let expected = tp.width() / (2.0 * std::f64::consts::PI).sqrt();
    for (name, keys) in [("D1", d1t), ("D2", d2t)] {
        let sd = standard_deviation(keys);
        assert!(
            (sd / expected - 1.0).abs() < 0.1,
            "{name}: sd {sd}, expected {expected}"
        );
        assert!(
            keys.iter().any(|x| x.abs() > 127),
            "{name} should need the wide product"
        );
    }
    for j in 0..jb {
        let bound = tp.noise_bound(&d1t[j * m..(j + 1) * m], &d2t[j * m..(j + 1) * m]);
        assert!(bound <= p.c_bound() as f64, "column {j}: {bound} > C");
    }
    assert!(setup.trapgen_attempts.iter().all(|&a| a >= 1));
}

#[test]
fn trapdoor_keys_decrypt_every_identity_and_every_recipient_reads_the_hidden_message() {
    let (_, setup, mut rng) = small_setup(2);
    let (pk, key) = (&setup.setup.public, &setup.setup.shortcut);
    let jb = pk.profile().ordinary_bits();
    let ids = identities(pk, &mut rng);
    let id_keys: Vec<IdentityKey> = ids
        .iter()
        .map(|id| setup.master.extract(pk, id, &mut rng).unwrap())
        .collect();
    for (shape, &(size, bits)) in [(1, 1), (2, 2), (4, 2), (8, 1), (2, 4), (1, 8)]
        .iter()
        .enumerate()
    {
        // aKGen for N = 2|S| users; S holds the even-numbered ones.
        let (users, group) = common::users_and_group(size);
        let (apk, keys) = covert_keygen(pk, users, bits, &mut rng).unwrap();
        let plan = prepare_hidden(pk, &apk, &group).unwrap();
        assert!(
            plan.sampler().full_rank(),
            "H_S is rank deficient for {size}x{bits}"
        );
        // The identities take turns, so each meets several group shapes.
        let (id, id_key) = (&ids[shape % ids.len()], &id_keys[shape % ids.len()]);
        assert_eq!(id_key.identity(), id);
        // At least eight trials, and every hidden message at least once.
        let messages = hidden_messages(bits);
        for trial in 0..messages.len().max(8) {
            let hidden = &messages[trial % messages.len()];
            let ordinary = random_bits(jb, &mut rng);
            let ct = encrypt_ordinary(pk, id, &ordinary, &mut rng).unwrap();
            assert_eq!(decrypt_ordinary(key, &ct).unwrap(), ordinary);
            assert_eq!(decrypt_identity(id_key, &ct).unwrap(), ordinary);
            let ct = plan.encrypt(id, &ordinary, hidden, &mut rng).unwrap();
            assert_eq!(decrypt_ordinary(key, &ct).unwrap(), ordinary);
            assert_eq!(decrypt_identity(id_key, &ct).unwrap(), ordinary);
            common::assert_every_recipient_decrypts(&keys, &group, &ct, hidden, "small trapdoor");
        }
    }
}

#[test]
fn same_errors_give_identical_ordinary_residuals_in_both_modes() {
    // Construction 3, eq. (9): both ordinary decryption paths cancel s, so an
    // ordinary and an anamorphic ciphertext built from the same errors (y, x and
    // R^T y) have identical residuals before rounding, with the shortcut key and
    // with the identity key.
    let (_, setup, mut rng) = small_setup(9);
    let (pk, key) = (&setup.setup.public, &setup.setup.shortcut);
    let p = pk.profile();
    let (n, q, jb) = (p.n(), p.q(), p.ordinary_bits());
    let id = common::random_identity(pk, &mut rng);
    let id_key = setup.master.extract(pk, &id, &mut rng).unwrap();
    let (users, group) = common::users_and_group(4);
    let (apk, keys) = covert_keygen(pk, users, 2, &mut rng).unwrap();
    let plan = prepare_hidden(pk, &apk, &group).unwrap();
    assert!(plan.sampler().full_rank());
    for hidden in hidden_messages(plan.bits()) {
        let ordinary = random_bits(jb, &mut rng);
        let rhs: Vec<u64> = (0..group.len())
            .flat_map(|_| hidden.iter().map(|&bit| u64::from(bit) * (q / 2)))
            .collect();
        let errors = sample_errors(p, &mut rng).unwrap();
        let constrained = plan.sampler().sample(&rhs, &mut rng).unwrap();
        let free = uniform(n, q, &mut rng).unwrap();
        let anamorphic = encrypt_with_coins(pk, &id, &ordinary, &constrained, &errors).unwrap();
        let plain = encrypt_with_coins(pk, &id, &ordinary, &free, &errors).unwrap();
        assert_ne!(anamorphic.c1(), plain.c1());
        assert_eq!(
            ordinary_residual(key, &anamorphic).unwrap(),
            ordinary_residual(key, &plain).unwrap()
        );
        assert_eq!(
            identity_residual(&id_key, &anamorphic).unwrap(),
            identity_residual(&id_key, &plain).unwrap()
        );
        assert_eq!(decrypt_ordinary(key, &anamorphic).unwrap(), ordinary);
        common::assert_every_recipient_decrypts(&keys, &group, &anamorphic, &hidden, "same errors");
    }
}

#[test]
fn decryption_noise_stays_within_the_bound_from_c() {
    // C bounds |d1 + R d2| for the shortcut keys, so the noise of
    // c - D1^T v - D2^T w has standard deviation at most q/65, whatever the identity.
    let (_, setup, mut rng) = small_setup(3);
    let (pk, key) = (&setup.setup.public, &setup.setup.shortcut);
    let p = pk.profile();
    let (q, jb) = (p.q(), p.ordinary_bits());
    let mut noise = Vec::new();
    for _ in 0..40 {
        let id = common::random_identity(pk, &mut rng);
        let ordinary = random_bits(jb, &mut rng);
        let ct = encrypt_ordinary(pk, &id, &ordinary, &mut rng).unwrap();
        for (z, &bit) in ordinary_residual(key, &ct)
            .unwrap()
            .into_iter()
            .zip(&ordinary)
        {
            let centered =
                (i128::from(z) - i128::from(bit) * i128::from(q / 2)).rem_euclid(i128::from(q));
            let centered = if centered > i128::from(q / 2) {
                centered - i128::from(q)
            } else {
                centered
            };
            noise.push(centered as f64);
        }
    }
    let sd = (noise.iter().map(|x| x * x).sum::<f64>() / noise.len() as f64).sqrt();
    assert!(
        sd < q as f64 / 65.0,
        "noise sd {sd} vs q/65 = {}",
        q as f64 / 65.0
    );
    assert!(noise.iter().all(|x| x.abs() < q as f64 / 4.0));
}

#[test]
fn extracted_identity_keys_satisfy_eq_23_and_decrypt_only_their_identity() {
    let (tp, setup, mut rng) = small_setup(4);
    let pk = &setup.setup.public;
    let p = pk.profile();
    let (m, jb) = (p.m(), p.ordinary_bits());
    let expected = tp.width() / (2.0 * std::f64::consts::PI).sqrt();
    let ids = identities(pk, &mut rng);
    let mut id_keys = Vec::new();
    for id in &ids {
        let id_key = setup.master.extract(pk, id, &mut rng).unwrap();
        assert_eq!(id_key.identity(), id);
        // [A0 | A1 + H(id) B] (e1_j; e2_j) = u_j on every entry (eq. 23).
        let h = common::identity_matrix(pk, id);
        assert_eq!(
            key_image(pk, id_key.e1_transposed(), id_key.e2_transposed(), Some(&h)),
            pk.u()
        );
        // e2 follows D_{Z^m,s}, and the identity key is a different short vector.
        let sd = standard_deviation(id_key.e2_transposed());
        assert!((sd / expected - 1.0).abs() < 0.1, "e2 sd {sd}");
        assert_ne!(id_key.e1_transposed(), setup.setup.shortcut.d1_transposed());
        for j in 0..jb {
            let bound = tp.noise_bound(
                &id_key.e1_transposed()[j * m..(j + 1) * m],
                &id_key.e2_transposed()[j * m..(j + 1) * m],
            );
            assert!(bound <= p.c_bound() as f64);
        }
        id_keys.push(id_key);
    }
    let (users, group) = common::users_and_group(4);
    let (apk, keys) = covert_keygen(pk, users, 2, &mut rng).unwrap();
    let plan = prepare_hidden(pk, &apk, &group).unwrap();
    assert!(plan.sampler().full_rank());
    // A key for identity a on a ciphertext for identity b != a leaves
    // ((H(a) - H(b)) B e2)^T s in the residual, uniform for uniform s: its bits agree
    // with the message only by chance. The master shortcut key decrypts every identity.
    let (mut agree, mut total) = (0usize, 0usize);
    for trial in 0..32 {
        let (a, b) = (trial % ids.len(), (trial + 1) % ids.len());
        let ordinary = random_bits(jb, &mut rng);
        let hidden = [(trial & 1) as u8, ((trial >> 1) & 1) as u8];
        let ct = if trial % 2 == 0 {
            encrypt_ordinary(pk, &ids[a], &ordinary, &mut rng).unwrap()
        } else {
            plan.encrypt(&ids[a], &ordinary, &hidden, &mut rng).unwrap()
        };
        assert_eq!(decrypt_identity(&id_keys[a], &ct).unwrap(), ordinary);
        assert_eq!(
            decrypt_ordinary(&setup.setup.shortcut, &ct).unwrap(),
            ordinary
        );
        if trial % 2 == 1 {
            common::assert_every_recipient_decrypts(&keys, &group, &ct, &hidden, "identity key");
        }
        let other = decrypt_identity(&id_keys[b], &ct).unwrap();
        agree += other.iter().zip(&ordinary).filter(|(x, y)| x == y).count();
        total += jb;
    }
    let rate = agree as f64 / total as f64;
    assert!(
        (rate - 0.5).abs() < 6.0 * 0.5 / (total as f64).sqrt(),
        "a key of another identity agrees at rate {rate}"
    );
    // A trapdoor of another setup, or an identity of other parameters, is refused.
    let (_, other, _) = small_setup(5);
    assert!(other.master.extract(pk, &ids[1], &mut rng).is_err());
    let foreign = IdentityEncoding::new(other_profile()).unwrap().unit();
    assert!(setup.master.extract(pk, &foreign, &mut rng).is_err());
}

#[test]
fn akgen_gives_binary_keys_and_their_images_and_aenc_uses_only_public_inputs() {
    let (_, setup, mut rng) = small_setup(10);
    let pk = &setup.setup.public;
    let p = pk.profile();
    let (n, m, q, jb) = (p.n(), p.m(), p.q(), p.ordinary_bits());
    let (users, bits) = (6, 2);
    // aKGen is N independent ask_i, then apk = (A0 k̂_(i,a)), coin for coin.
    let mut first = StdRng::seed_from_u64(77);
    let mut second = StdRng::seed_from_u64(77);
    let (apk, keys) = covert_keygen(pk, users, bits, &mut first).unwrap();
    let separate: Vec<CovertSecretKey> = (0..users)
        .map(|_| covert_secret_keygen(p, bits, &mut second).unwrap())
        .collect();
    assert_eq!(keys, separate);
    assert_eq!(
        CovertPublicKey::from_secret_keys(pk, &separate).unwrap(),
        apk
    );
    assert_eq!((apk.users(), apk.bits(), apk.profile()), (users, bits, p));
    let mut ones = 0usize;
    for (i, key) in keys.iter().enumerate() {
        assert_eq!((key.bits(), key.profile()), (bits, p));
        let kt = key.transposed_keys();
        assert_eq!(kt.len(), bits * m);
        assert!(kt.iter().all(|&x| x == 0 || x == 1));
        ones += kt.iter().filter(|&&x| x == 1).count();
        for a in 0..bits {
            let block = &kt[a * m..(a + 1) * m];
            let image: Vec<u64> = (0..n)
                .map(|r| {
                    let mut total = Integer::new();
                    for (&a0, &k) in pk.a0()[r * m..(r + 1) * m].iter().zip(block) {
                        total += Integer::from(a0) * k;
                    }
                    residue(&total, q)
                })
                .collect();
            assert_eq!(apk.vector(i, a), image.as_slice(), "b̂_({i},{a}) = A0 k̂");
        }
    }
    let count = (users * bits * m) as f64;
    assert!(
        (ones as f64 - count / 2.0).abs() < 6.0 * (count / 4.0).sqrt(),
        "anamorphic key bits are not balanced"
    );
    // aEnc for S = {1, 2, 4} from (pk, apk) alone: column (j - 1) L + a of H_S is
    // a1 - b̂_(i_j, a), with a1 the first column of A0.
    let group = [1, 2, 4];
    let plan = prepare_hidden(pk, &apk, &group).unwrap();
    assert_eq!((plan.group(), plan.bits()), (&group[..], bits));
    assert!(plan.sampler().full_rank());
    assert_eq!(plan.constraints().len(), group.len() * bits);
    for (j, &i) in group.iter().enumerate() {
        for a in 0..bits {
            let column: Vec<u64> = (0..n)
                .map(|r| sub_mod(pk.a0()[r * m], apk.vector(i, a)[r], q))
                .collect();
            assert_eq!(plan.constraints()[j * bits + a], column);
        }
    }
    let id = common::random_identity(pk, &mut rng);
    for hidden in hidden_messages(bits) {
        let ordinary = random_bits(jb, &mut rng);
        let ct = plan.encrypt(&id, &ordinary, &hidden, &mut rng).unwrap();
        assert_eq!(
            decrypt_ordinary(&setup.setup.shortcut, &ct).unwrap(),
            ordinary
        );
        common::assert_every_recipient_decrypts(&keys, &group, &ct, &hidden, "S = {1, 2, 4}");
    }
    // Malformed keys, users and groups are refused.
    let g = p.maximum_constraints();
    let mut key = vec![0; m * bits];
    assert!(CovertSecretKey::new(p, bits, &key).is_ok());
    key[5] = 2;
    assert!(CovertSecretKey::new(p, bits, &key).is_err());
    key[5] = -1;
    assert!(CovertSecretKey::new(p, bits, &key).is_err());
    assert!(CovertSecretKey::new(p, bits, &vec![0; m * bits - 1]).is_err());
    assert!(CovertSecretKey::new(p, 0, &[]).is_err());
    assert!(CovertSecretKey::new(p, g + 1, &vec![0; m * (g + 1)]).is_err());
    assert!(covert_secret_keygen(p, 0, &mut rng).is_err());
    assert!(covert_secret_keygen(p, g + 1, &mut rng).is_err());
    assert!(covert_keygen(pk, 0, bits, &mut rng).is_err());
    assert!(CovertPublicKey::from_secret_keys(pk, &[]).is_err());
    let single = covert_secret_keygen(p, 1, &mut rng).unwrap();
    assert!(CovertPublicKey::from_secret_keys(pk, &[keys[0].clone(), single]).is_err());
    let bad: [&[usize]; 5] = [&[], &[2, 1], &[1, 1], &[0, users], &[5, 3]];
    for group in bad {
        assert!(prepare_hidden(pk, &apk, group).is_err(), "group {group:?}");
    }
    // |S| L must not exceed g = 8: four users of two bits fit, five do not.
    assert!(prepare_hidden(pk, &apk, &[0, 1, 2, 3]).is_ok());
    assert!(prepare_hidden(pk, &apk, &[0, 1, 2, 3, 4]).is_err());
    // Keys and apk of other parameters are refused.
    let other = other_profile();
    let zeros = PublicKey::new(
        other,
        vec![0; n * m],
        vec![0; n * m],
        vec![0; n * m],
        vec![0; n * jb],
    )
    .unwrap();
    let foreign_key = covert_secret_keygen(other, 1, &mut rng).unwrap();
    assert!(CovertPublicKey::from_secret_keys(pk, std::slice::from_ref(&foreign_key)).is_err());
    let foreign_apk = CovertPublicKey::from_secret_keys(&zeros, &[foreign_key]).unwrap();
    assert!(prepare_hidden(pk, &foreign_apk, &[0]).is_err());
}

#[test]
fn rank_deficient_groups_fall_back_to_ordinary_encryption_with_the_same_coins() {
    let (_, setup, mut rng) = small_setup(11);
    let pk = &setup.setup.public;
    let p = pk.profile();
    let (q, jb) = (p.q(), p.ordinary_bits());
    let id = common::random_identity(pk, &mut rng);
    // Two users with the same key: H_S has two equal columns, so Construction 1
    // encrypts ordinarily.
    let key = covert_secret_keygen(p, 1, &mut rng).unwrap();
    let apk = CovertPublicKey::from_secret_keys(pk, &[key.clone(), key]).unwrap();
    let plan = prepare_hidden(pk, &apk, &[0, 1]).unwrap();
    assert!(!plan.sampler().full_rank());
    let ordinary: Vec<u8> = (0..jb).map(|j| (j % 2) as u8).collect();
    let mut a = StdRng::seed_from_u64(541);
    let mut b = StdRng::seed_from_u64(541);
    let fallback = plan.encrypt(&id, &ordinary, &[1], &mut a).unwrap();
    assert_eq!(
        fallback,
        encrypt_ordinary(pk, &id, &ordinary, &mut b).unwrap()
    );
    assert_eq!(a.next_u64(), b.next_u64());
    assert_eq!(
        decrypt_ordinary(&setup.setup.shortcut, &fallback).unwrap(),
        ordinary
    );
    // s is uniform: the holder of T_A0 does not see floor(q/2) am in H_S^T s
    // (except with probability 2/q).
    let s = setup.master.recover_secret(pk, &fallback).unwrap();
    let h = &plan.constraints()[0];
    let value = (h
        .iter()
        .zip(&s)
        .map(|(&x, &y)| u128::from(x) * u128::from(y))
        .sum::<u128>()
        % u128::from(q)) as u64;
    assert!(value != 0 && value != q / 2);
    // Invalid requests are refused before any coin is drawn.
    let foreign = IdentityEncoding::new(other_profile()).unwrap().unit();
    let mut a = StdRng::seed_from_u64(62);
    let mut b = StdRng::seed_from_u64(62);
    let mut not_bits = ordinary.clone();
    not_bits[0] = 2;
    assert!(plan.encrypt(&id, &ordinary[1..], &[1], &mut a).is_err());
    assert!(plan.encrypt(&id, &not_bits, &[1], &mut a).is_err());
    assert!(plan.encrypt(&id, &ordinary, &[1, 0], &mut a).is_err());
    assert!(plan.encrypt(&id, &ordinary, &[2], &mut a).is_err());
    assert!(plan.encrypt(&foreign, &ordinary, &[1], &mut a).is_err());
    assert!(encrypt_ordinary(pk, &foreign, &ordinary, &mut a).is_err());
    assert!(encrypt_ordinary(pk, &id, &not_bits, &mut a).is_err());
    assert_eq!(a.next_u64(), b.next_u64());
}

#[test]
fn master_trapdoor_recovers_s_and_reads_the_hidden_message() {
    // Security boundary: T_A0 inverts c1[1..m] = A0^T s + y. Its holder recomputes
    // H_S^T s from public data and sees floor(q/2) am exactly, while an ordinary
    // ciphertext gives values spread over Z_q. The same ciphertexts decrypt to the
    // ordinary message with both keys, and every recipient reads the hidden message.
    let (_, setup, mut rng) = small_setup(6);
    let pk = &setup.setup.public;
    let p = pk.profile();
    let (n, q, jb) = (p.n(), p.q(), p.ordinary_bits());
    let id = common::random_identity(pk, &mut rng);
    let id_key = setup.master.extract(pk, &id, &mut rng).unwrap();
    let (users, group) = common::users_and_group(4);
    let (apk, keys) = covert_keygen(pk, users, 2, &mut rng).unwrap();
    let plan = prepare_hidden(pk, &apk, &group).unwrap();
    assert!(plan.sampler().full_rank());
    let constraint_values = |s: &[u64]| -> Vec<u64> {
        plan.constraints()
            .iter()
            .map(|h| {
                let mut total = Integer::new();
                for i in 0..n {
                    total += Integer::from(h[i]) * s[i];
                }
                residue(&total, q)
            })
            .collect()
    };
    let ordinary: Vec<u8> = (0..jb).map(|j| (j % 2) as u8).collect();
    for hidden in hidden_messages(plan.bits()) {
        let rhs: Vec<u64> = (0..group.len())
            .flat_map(|_| hidden.iter().map(|&bit| u64::from(bit) * (q / 2)))
            .collect();
        let s = plan.sampler().sample(&rhs, &mut rng).unwrap();
        let errors = sample_errors(p, &mut rng).unwrap();
        let ct = encrypt_with_coins(pk, &id, &ordinary, &s, &errors).unwrap();
        assert_eq!(
            decrypt_ordinary(&setup.setup.shortcut, &ct).unwrap(),
            ordinary
        );
        assert_eq!(decrypt_identity(&id_key, &ct).unwrap(), ordinary);
        common::assert_every_recipient_decrypts(&keys, &group, &ct, &hidden, "T_A0");
        let recovered = setup.master.recover_secret(pk, &ct).unwrap();
        assert_eq!(recovered, s);
        assert_eq!(constraint_values(&recovered), rhs);
    }
    let s = uniform(n, q, &mut rng).unwrap();
    let errors = sample_errors(p, &mut rng).unwrap();
    let ct = encrypt_with_coins(pk, &id, &ordinary, &s, &errors).unwrap();
    assert_eq!(
        decrypt_ordinary(&setup.setup.shortcut, &ct).unwrap(),
        ordinary
    );
    assert_eq!(decrypt_identity(&id_key, &ct).unwrap(), ordinary);
    let recovered = setup.master.recover_secret(pk, &ct).unwrap();
    assert_eq!(recovered, s);
    assert!(
        constraint_values(&recovered)
            .iter()
            .any(|&x| x != 0 && x != q / 2)
    );
}

#[test]
fn trapdoor_ciphertexts_round_trip_through_the_wire_format() {
    let (_, setup, mut rng) = small_setup(7);
    let pk = &setup.setup.public;
    let p = pk.profile();
    let id = common::random_identity(pk, &mut rng);
    let ordinary: Vec<u8> = (0..p.ordinary_bits()).map(|j| (j % 3 == 0) as u8).collect();
    let ct = encrypt_ordinary(pk, &id, &ordinary, &mut rng).unwrap();
    let bytes = encode(p, &ct).unwrap();
    assert_eq!(bytes.len(), p.ciphertext_bytes());
    let decoded = decode(p, &bytes).unwrap();
    assert_eq!(decoded, ct);
    assert_eq!(
        decrypt_ordinary(&setup.setup.shortcut, &decoded).unwrap(),
        ordinary
    );
}

#[test]
#[ignore = "full profile (n = 1360): about 2 GB and minutes; run with --release -- --ignored"]
fn full_profile_end_to_end() {
    let tp = TrapdoorProfile::full();
    let p = tp.profile();
    let mut rng = StdRng::seed_from_u64(20261102);
    let setup =
        setup_trapdoor(&tp, &mut rng, |phase, count| eprintln!("{phase}: {count}")).unwrap();
    let pk = &setup.setup.public;
    let id = common::random_identity(pk, &mut rng);
    let id_key = setup.master.extract(pk, &id, &mut rng).unwrap();
    let (users, group) = common::users_and_group(4);
    let (apk, keys) = covert_keygen(pk, users, 2, &mut rng).unwrap();
    let plan = prepare_hidden(pk, &apk, &group).unwrap();
    assert!(plan.sampler().full_rank());
    let ordinary: Vec<u8> = (0..p.ordinary_bits()).map(|j| (j % 2) as u8).collect();
    let ct = encrypt_ordinary(pk, &id, &ordinary, &mut rng).unwrap();
    assert_eq!(
        decrypt_ordinary(&setup.setup.shortcut, &ct).unwrap(),
        ordinary
    );
    assert_eq!(decrypt_identity(&id_key, &ct).unwrap(), ordinary);
    // One anamorphic ciphertext per hidden message, each checked with both keys and
    // by every recipient.
    for hidden in hidden_messages(plan.bits()) {
        let ct = plan.encrypt(&id, &ordinary, &hidden, &mut rng).unwrap();
        assert_eq!(
            decrypt_ordinary(&setup.setup.shortcut, &ct).unwrap(),
            ordinary
        );
        assert_eq!(decrypt_identity(&id_key, &ct).unwrap(), ordinary);
        common::assert_every_recipient_decrypts(&keys, &group, &ct, &hidden, "full profile");
    }
}
