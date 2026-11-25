//! Value audit of Epoque's trapdoor setup with the anamorphic broadcast inside, on
//! real ciphertexts for a random identity.
//!
//! Every quantity must look like what the construction prescribes. Nothing may
//! be zero, constant or repeated unless the protocol says so. The protocol's
//! zeros are B D2 = 0, the hidden target h^T s = 0 for a zero bit, about half of
//! the entries of the binary anamorphic keys, about a 1/s fraction of the entries
//! of the width-s Gaussian keys D1, D2, and errors that are zero with their
//! probability under the rounded normal law. A rank failure falls back to
//! ordinary encryption with the same coins.
//!
//! Each check is a hard bound at about six standard deviations, so it fails
//! only if something is wrong. Run with `--nocapture` to see the statistics.
//! The full profile allocates about 4 GB and takes minutes, so it is opt-in:
//! `cargo test --release --test sanity -- --ignored --nocapture`.
mod common;

use epoque_shortcut::arithmetic::uniform;
use epoque_shortcut::trapdoor::{TrapdoorProfile, TrapdoorSetup, setup_trapdoor};
use epoque_shortcut::*;
use rand::{Rng, RngExt, SeedableRng, TryRng, rngs::StdRng};
use rug::Integer;
use std::collections::BTreeSet;
use std::convert::Infallible;

/// Column block of the sign matrix R in `sample_errors`.
const SIGN_BLOCK: usize = 2048;

fn centred(x: u64, q: u64) -> i128 {
    if x > q / 2 {
        i128::from(x) - i128::from(q)
    } else {
        i128::from(x)
    }
}

fn residue(x: i128, q: u64) -> u64 {
    x.rem_euclid(i128::from(q)) as u64
}

fn small(values: &[Integer]) -> Vec<i128> {
    values
        .iter()
        .map(|x| i128::from(x.to_i64().expect("error fits in i64")))
        .collect()
}

/// a^T t modulo q for row-major a with `cols` columns.
fn transpose_times(a: &[u64], cols: usize, t: &[u64], q: u64) -> Vec<u64> {
    let mut acc = vec![0u128; cols];
    for (row, &x) in a.chunks_exact(cols).zip(t) {
        for (sum, &y) in acc.iter_mut().zip(row) {
            *sum += u128::from(y) * u128::from(x);
        }
    }
    acc.into_iter()
        .map(|x| (x % u128::from(q)) as u64)
        .collect()
}

/// D^T y over the integers for D stored transposed (`dt[j * len + k]`).
fn small_transpose_times(dt: &[i32], len: usize, y: &[i128]) -> Vec<i128> {
    dt.chunks_exact(len)
        .map(|column| column.iter().zip(y).map(|(&d, &x)| i128::from(d) * x).sum())
        .collect()
}

/// erf by its Taylor series. The arguments here are below 1, where 40 terms
/// are far beyond f64 precision.
fn erf(x: f64) -> f64 {
    assert!(x.abs() < 1.0);
    let (mut term, mut sum) = (x, x);
    for k in 1..40 {
        term *= -x * x / f64::from(k);
        sum += term / f64::from(2 * k + 1);
    }
    sum * 2.0 / std::f64::consts::PI.sqrt()
}

struct Stats {
    count: usize,
    zeros: usize,
    mean: f64,
    sd: f64,
}

fn stats(values: impl Iterator<Item = f64>) -> Stats {
    let (mut count, mut zeros, mut sum, mut sum2) = (0usize, 0usize, 0f64, 0f64);
    for v in values {
        count += 1;
        zeros += usize::from(v == 0.0);
        sum += v;
        sum2 += v * v;
    }
    let mean = sum / count as f64;
    Stats {
        count,
        zeros,
        mean,
        sd: (sum2 / count as f64 - mean * mean).max(0.0).sqrt(),
    }
}

/// Uniform residues modulo q: mean, spread, sixteen buckets and zeros.
fn check_uniform(label: &str, values: &[u64], q: u64) {
    assert!(values.iter().all(|&v| v < q), "{label}: noncanonical");
    let s = stats(values.iter().map(|&v| v as f64));
    let (n, qf) = (s.count as f64, q as f64);
    let expected_zeros = n / qf;
    eprintln!(
        "{label}: {} values, mean/q {:.5}, sd/q {:.5}, zeros {}",
        s.count,
        s.mean / qf,
        s.sd / qf,
        s.zeros
    );
    assert!(
        (s.mean - (qf - 1.0) / 2.0).abs() < 6.0 * qf / (12.0 * n).sqrt() + 1.0,
        "{label}: mean"
    );
    if s.count >= 1000 {
        assert!(
            (s.sd / qf - 12f64.sqrt().recip()).abs() < 0.02,
            "{label}: spread"
        );
    }
    if s.count >= 1600 {
        let mut buckets = [0usize; 16];
        for &v in values {
            buckets[(u128::from(v) * 16 / u128::from(q)) as usize] += 1;
        }
        let (mean, sd) = (n / 16.0, (n * 15.0 / 256.0).sqrt());
        for (i, &count) in buckets.iter().enumerate() {
            assert!(
                (count as f64 - mean).abs() < 6.0 * sd + 1.0,
                "{label}: bucket {i} holds {count}"
            );
        }
    }
    assert!(
        (s.zeros as f64) < expected_zeros + 6.0 * expected_zeros.sqrt() + 3.0,
        "{label}: {} zero entries",
        s.zeros
    );
    if s.count > 1 {
        assert!(values.windows(2).any(|w| w[0] != w[1]), "{label}: constant");
    }
}

/// Errors floor(N(0, sigma^2) + 1/2): sd sqrt(sigma^2 + 1/12), P(0) = erf(1/(2 sqrt(2) sigma)).
fn check_rounded_normal(label: &str, values: &[i128], sigma: f64) {
    let s = stats(values.iter().map(|&e| e as f64));
    let n = s.count as f64;
    let sd = (sigma * sigma + 1.0 / 12.0).sqrt();
    let p0 = erf((2.0 * 2f64.sqrt() * sigma).recip());
    let largest = values.iter().map(|e| e.unsigned_abs()).max().unwrap();
    eprintln!(
        "{label}: {} errors, mean {:.3}, sd {:.4} (expected {sd:.4}), zeros {} (expected {:.1}), max |e| {largest}",
        s.count,
        s.mean,
        s.sd,
        s.zeros,
        n * p0
    );
    assert!(s.mean.abs() < 6.0 * sd / n.sqrt(), "{label}: mean");
    assert!(
        (s.sd / sd - 1.0).abs() < 6.0 / (2.0 * n).sqrt() + 1e-3,
        "{label}: spread"
    );
    assert!(
        (s.zeros as f64 - n * p0).abs() < 6.0 * (n * p0 * (1.0 - p0)).sqrt() + 3.0,
        "{label}: zeros"
    );
    assert!(
        (largest as f64) < 10.0 * sigma + 1.0,
        "{label}: implausible tail"
    );
}

/// Width-s Gaussian keys (MP12 preimages): mean 0, standard deviation s/sqrt(2 pi)
/// in every coordinate, a fraction of about 1/s zeros, and no implausible tail.
fn check_key_gaussian(label: &str, values: &[i32], width: f64) {
    let n = values.len() as f64;
    let s = stats(values.iter().map(|&x| f64::from(x)));
    let expected = width / (2.0 * std::f64::consts::PI).sqrt();
    let p0 = width.recip();
    let largest = values.iter().map(|x| x.unsigned_abs()).max().unwrap();
    eprintln!(
        "{label}: {} entries, mean {:.3}, sd {:.2} (expected {expected:.2}), zeros {} (expected {:.1}), max |x| {largest}",
        s.count,
        s.mean,
        s.sd,
        s.zeros,
        n * p0
    );
    assert!(s.mean.abs() < 6.0 * expected / n.sqrt(), "{label}: mean");
    assert!(
        (s.sd / expected - 1.0).abs() < 6.0 / (2.0 * n).sqrt() + 0.01,
        "{label}: spread"
    );
    assert!(
        (s.zeros as f64 - n * p0).abs() < 6.0 * (n * p0).sqrt() + 3.0,
        "{label}: zeros"
    );
    assert!(
        f64::from(largest) < 8.0 * width,
        "{label}: implausible tail"
    );
}

fn balanced(ones: u64, bits: u64) -> bool {
    let bits = bits as f64;
    (ones as f64 - bits / 2.0).abs() < 6.0 * (bits / 4.0).sqrt() + 1.0
}

/// Each coordinate takes at least two values over the trials.
fn check_varies(label: &str, trials: &[Vec<u64>]) {
    if trials.len() < 2 {
        return;
    }
    for i in 0..trials[0].len() {
        assert!(
            trials.iter().any(|t| t[i] != trials[0][i]),
            "{label}: coordinate {i} is the same in all {} trials",
            trials.len()
        );
    }
}

/// Pass-through generator. In `sample_errors` the normals use next_u64 and
/// every fill_bytes request is sign material of R, so this sees the whole sign
/// matrix: it counts requests, bytes and one bits, and keeps the first request
/// (the first column block of R).
struct Recorder<'a> {
    inner: &'a mut StdRng,
    signs: Signs,
}

#[derive(Default)]
struct Signs {
    calls: usize,
    bytes: usize,
    ones: u64,
    first: Vec<u8>,
}

impl<'a> Recorder<'a> {
    fn new(inner: &'a mut StdRng) -> Self {
        Self {
            inner,
            signs: Signs::default(),
        }
    }
}

impl TryRng for Recorder<'_> {
    type Error = Infallible;
    fn try_next_u32(&mut self) -> std::result::Result<u32, Infallible> {
        Ok(self.inner.next_u32())
    }
    fn try_next_u64(&mut self) -> std::result::Result<u64, Infallible> {
        Ok(self.inner.next_u64())
    }
    fn try_fill_bytes(&mut self, dst: &mut [u8]) -> std::result::Result<(), Infallible> {
        self.inner.fill_bytes(dst);
        if self.signs.calls == 0 {
            self.signs.first = dst.to_vec();
        }
        self.signs.calls += 1;
        self.signs.bytes += dst.len();
        self.signs.ones += dst.iter().map(|b| u64::from(b.count_ones())).sum::<u64>();
        Ok(())
    }
}

/// Error coins drawn through the recorder, so R can be checked as well.
fn recorded_errors(p: Profile, rng: &mut StdRng) -> (ErrorCoins, Signs) {
    let mut recorder = Recorder::new(rng);
    let errors = sample_errors(p, &mut recorder).unwrap();
    (errors, recorder.signs)
}

#[derive(Default)]
struct Totals {
    errors: Vec<i128>,
    correlated_square: f64,
    expected_square: f64,
    correlated_count: usize,
    sign_ones: u64,
    sign_bits: u64,
    first_blocks: BTreeSet<Vec<u8>>,
    s_hidden: Vec<u64>,
    s_ordinary: Vec<u64>,
    c1_top: Vec<u64>,
    c1_bottom: Vec<u64>,
    c0: Vec<u64>,
    ordinary_noise: Vec<i128>,
    ordinary_zero_expected: f64,
    covert_noise: Vec<i128>,
    covert_zero_expected: f64,
    outside: Vec<f64>,
    largest_noise: u128,
}

/// The host keys and H(id) of the audited identity, shared by every ciphertext check.
struct Host<'a> {
    setup: &'a Setup,
    h: &'a [u64],
}

/// One ciphertext against its coins: the sign matrix R, the ciphertext equations
/// for c0 and both halves of c1, the ordinary residual and the wire format.
fn check_ciphertext(
    t: &mut Totals,
    host: &Host,
    s: &[u64],
    errors: &ErrorCoins,
    signs: Signs,
    ct: &Ciphertext,
    ordinary: &[u8],
) {
    let (setup, h) = (host.setup, host.h);
    let pk = &setup.public;
    let p = pk.profile();
    let (n, m, q, jb) = (p.n(), p.m(), p.q(), p.ordinary_bits());
    let delta = q / 2;
    let sigma = q as f64 / p.denominator() as f64;
    let rounded_sd = (sigma * sigma + 1.0 / 12.0).sqrt();
    let (y, x, correlated) = (
        small(&errors.y),
        small(&errors.x),
        small(&errors.correlated),
    );
    assert_eq!((y.len(), x.len(), correlated.len()), (m, jb, m));

    // R is a fresh dense m x m sign matrix: every block requested, half ones.
    let groups = m.div_ceil(8);
    assert_eq!(signs.calls, m.div_ceil(SIGN_BLOCK), "sign blocks");
    assert_eq!(signs.bytes, groups * m, "sign bytes");
    assert!(
        balanced(signs.ones, 8 * signs.bytes as u64),
        "sign bits are not balanced"
    );
    t.sign_ones += signs.ones;
    t.sign_bits += 8 * signs.bytes as u64;
    let width = SIGN_BLOCK.min(m);
    assert_eq!(signs.first.len(), groups * width);
    assert!(
        t.first_blocks
            .insert(signs.first[..64usize.min(signs.first.len())].to_vec()),
        "a sign block was reused"
    );
    // First block exactly: correlated = R^T y with the recorded signs (bit 1 is +1).
    for (j, &value) in correlated.iter().take(width).enumerate() {
        let sum: i128 = y
            .iter()
            .enumerate()
            .map(|(i, &yi)| {
                if (signs.first[(i / 8) * width + j] >> (i % 8)) & 1 == 1 {
                    yi
                } else {
                    -yi
                }
            })
            .sum();
        assert_eq!(
            sum, value,
            "R^T y differs from the recorded signs in column {j}"
        );
    }
    // Every column: a sum of +-y_i has the parity of sum y_i (no sign is zero).
    let parity = y.iter().sum::<i128>().rem_euclid(2);
    assert!(
        correlated.iter().all(|z| z.rem_euclid(2) == parity),
        "a column of R^T y has a zero sign"
    );
    t.correlated_square += correlated.iter().map(|&z| (z as f64).powi(2)).sum::<f64>();
    t.expected_square += m as f64 * y.iter().map(|&z| (z as f64).powi(2)).sum::<f64>();
    t.correlated_count += m;
    t.errors.extend(&y);
    t.errors.extend(&x);

    // c1 = F_id^T s + [y; R^T y] and c0 = U^T s + x + floor(q/2) mu (BHM21 (25), (26)),
    // with (H(id) B)^T s = B^T (H(id)^T s).
    let (a0s, a1s, bs) = (
        transpose_times(pk.a0(), m, s, q),
        transpose_times(pk.a1(), m, s, q),
        transpose_times(pk.b(), m, &transpose_times(h, n, s, q), q),
    );
    let us = transpose_times(pk.u(), jb, s, q);
    let (top, bottom) = ct.c1().split_at(m);
    for j in 0..m {
        assert_eq!(top[j], residue(i128::from(a0s[j]) + y[j], q), "c1[{j}]");
        assert_eq!(
            bottom[j],
            residue(i128::from(a1s[j]) + i128::from(bs[j]) + correlated[j], q),
            "c1[m + {j}]"
        );
    }
    for j in 0..jb {
        let message = i128::from(ordinary[j]) * i128::from(delta);
        assert_eq!(
            ct.c0()[j],
            residue(i128::from(us[j]) + x[j] + message, q),
            "c0[{j}]"
        );
    }
    t.c1_top.extend(top);
    t.c1_bottom.extend(bottom);
    t.c0.extend(ct.c0());

    // Ordinary residual: floor(q/2) m + x - D1^T y - D2^T R^T y, s cancels.
    let d1t = setup.shortcut.d1_transposed();
    let d2t = setup.shortcut.d2_transposed();
    let d1y = small_transpose_times(d1t, m, &y);
    let d2r = small_transpose_times(d2t, m, &correlated);
    let residual = ordinary_residual(&setup.shortcut, ct).unwrap();
    for j in 0..jb {
        let noise = x[j] - d1y[j] - d2r[j];
        let message = i128::from(ordinary[j]) * i128::from(delta);
        assert_eq!(
            residual[j],
            residue(noise + message, q),
            "ordinary residual {j}"
        );
        assert!(
            noise.unsigned_abs() < u128::from(q / 4),
            "ordinary noise beyond q/4"
        );
        t.largest_noise = t.largest_noise.max(noise.unsigned_abs());
        t.ordinary_noise.push(noise);
        // Normal approximation with the variance averaged over R.
        let d1 = &d1t[j * m..(j + 1) * m];
        let d2 = &d2t[j * m..(j + 1) * m];
        let square = |d: &[i32]| d.iter().map(|&z| f64::from(z * z)).sum::<f64>();
        let sd = rounded_sd * (1.0 + square(d1) + m as f64 * square(d2)).sqrt();
        t.ordinary_zero_expected += 1.0 / (sd * (2.0 * std::f64::consts::PI).sqrt());
    }
    assert_eq!(decrypt_ordinary(&setup.shortcut, ct).unwrap(), ordinary);

    let wire = encode(p, ct).unwrap();
    assert_eq!(wire.len(), p.ciphertext_bytes());
    assert_eq!(&decode(p, &wire).unwrap(), ct);
}

/// A recipient: c1[0] - k^T c1[..m] = floor(q/2) bit + y[0] - k^T y for each key block.
fn check_covert(
    t: &mut Totals,
    key: &CovertSecretKey,
    ct: &Ciphertext,
    y: &[i128],
    hidden: &[u8],
    q: u64,
    sigma: f64,
) {
    let m = y.len();
    let rounded_sd = (sigma * sigma + 1.0 / 12.0).sqrt();
    let residual = covert_residual(key, ct).unwrap();
    for (j, block) in key.transposed_keys().chunks_exact(m).enumerate() {
        let noise = y[0]
            - block
                .iter()
                .zip(y)
                .map(|(&k, &e)| i128::from(k) * e)
                .sum::<i128>();
        let message = i128::from(hidden[j]) * i128::from(q / 2);
        assert_eq!(
            residual[j],
            residue(noise + message, q),
            "covert residual {j}"
        );
        assert!(
            noise.unsigned_abs() < u128::from(q / 4),
            "covert noise beyond q/4"
        );
        t.largest_noise = t.largest_noise.max(noise.unsigned_abs());
        t.covert_noise.push(noise);
        let weight =
            f64::from(1 - block[0]).powi(2) + block[1..].iter().map(|&k| f64::from(k)).sum::<f64>();
        let sd = rounded_sd * weight.sqrt();
        t.covert_zero_expected += 1.0 / (sd * (2.0 * std::f64::consts::PI).sqrt());
    }
    assert_eq!(decrypt_covert(key, ct).unwrap(), hidden);
}

/// Keys that should see nothing: residuals spread over Z_q.
fn spread(t: &mut Totals, key: &CovertSecretKey, ct: &Ciphertext, q: u64) {
    for z in covert_residual(key, ct).unwrap() {
        let to_zero = centred(z, q).unsigned_abs();
        let to_delta = centred((z + q - q / 2) % q, q).unsigned_abs();
        t.outside.push(to_zero.min(to_delta) as f64 / q as f64);
    }
}

fn check_noise(label: &str, noise: &[i128], expected_zeros: f64) {
    let zeros = noise.iter().filter(|&&z| z == 0).count();
    let s = stats(noise.iter().map(|&z| z as f64));
    eprintln!(
        "{label}: {} residuals, mean {:.2}, sd {:.2}, zeros {zeros} (expected about {expected_zeros:.2})",
        s.count, s.mean, s.sd
    );
    assert!(
        (zeros as f64) < expected_zeros + 6.0 * expected_zeros.sqrt() + 3.0,
        "{label}: too many zero residuals"
    );
    assert!(noise.windows(2).any(|w| w[0] != w[1]), "{label}: constant");
}

fn audit(tp: TrapdoorProfile, seed: u64, shapes: &[(usize, usize)], trials: u64, every_row: bool) {
    let p = tp.profile();
    let (n, m, q, jb) = (p.n(), p.m(), p.q(), p.ordinary_bits());
    let delta = q / 2;
    let sigma = q as f64 / p.denominator() as f64;
    eprintln!(
        "=== n {n}, m {m}, q {q}, J {jb}, key width {}, error width {sigma:.4}",
        tp.width()
    );
    let mut rng = StdRng::seed_from_u64(seed);
    let TrapdoorSetup {
        setup,
        master,
        trapgen_attempts,
        key_resamples,
    } = setup_trapdoor(&tp, &mut rng, |_, _| {}).unwrap();
    // The audit needs no identity keys: T_A0 is dropped at once.
    drop(master);
    eprintln!("TrapGen draws {trapgen_attempts:?}, shortcut columns redrawn {key_resamples}");
    let pk = &setup.public;
    let (a0, a1, b, u) = (pk.a0(), pk.a1(), pk.b(), pk.u());
    check_uniform("A0 (TrapGen)", a0, q);
    check_uniform(
        "first column of A0",
        &(0..n).map(|i| a0[i * m]).collect::<Vec<_>>(),
        q,
    );
    check_uniform("A1", a1, q);
    check_uniform("B (TrapGen)", b, q);
    check_uniform("U", u, q);
    let id = common::random_identity(pk, &mut rng);
    let h = common::identity_matrix(pk, &id);
    check_uniform("H(id) of a random identity", &h, q);
    let d1t = setup.shortcut.d1_transposed();
    let d2t = setup.shortcut.d2_transposed();
    check_key_gaussian("D1", d1t, tp.width());
    check_key_gaussian("D2", d2t, tp.width());
    assert_ne!(d1t, d2t, "D1 and D2 coincide");

    // Exactly on rows: B D2 = 0 and U = A0 D1 + A1 D2. B D1 is a negative control.
    let rows: Vec<usize> = if every_row {
        (0..n).collect()
    } else {
        let mut rows = vec![0, 1, n / 2, n - 1];
        rows.extend((0..12).map(|_| rng.random_range(0..n)));
        rows
    };
    let mut control = Vec::new();
    for &r in &rows {
        for j in 0..jb {
            let (mut kernel, mut image, mut other) = (0i128, 0i128, 0i128);
            for k in 0..m {
                let (d1, d2) = (i128::from(d1t[j * m + k]), i128::from(d2t[j * m + k]));
                kernel += i128::from(b[r * m + k]) * d2;
                image += i128::from(a0[r * m + k]) * d1 + i128::from(a1[r * m + k]) * d2;
                other += i128::from(b[r * m + k]) * d1;
            }
            assert_eq!(residue(kernel, q), 0, "B D2 is not zero at ({r}, {j})");
            assert_eq!(
                residue(image, q),
                u[r * jb + j],
                "U is not A0 D1 + A1 D2 at ({r}, {j})"
            );
            control.push(residue(other, q));
        }
    }
    eprintln!("B D2 = 0 and U = A0 D1 + A1 D2 hold on {} rows", rows.len());
    check_uniform("negative control B D1 on those rows", &control, q);
    // Whole matrices through a random vector r: D2^T B^T r = 0 and
    // U^T r = D1^T A0^T r + D2^T (A1 + H(id) B)^T r, so F_id D = U (eq. 29).
    let r = uniform(n, q, &mut rng).unwrap();
    let lift = |v: Vec<u64>| v.into_iter().map(i128::from).collect::<Vec<_>>();
    let kernel = small_transpose_times(d2t, m, &lift(transpose_times(b, m, &r, q)));
    assert!(
        kernel.iter().all(|&z| residue(z, q) == 0),
        "B D2 is not zero"
    );
    let first = small_transpose_times(d1t, m, &lift(transpose_times(a0, m, &r, q)));
    let mut right = transpose_times(a1, m, &r, q);
    let hb = transpose_times(b, m, &transpose_times(&h, n, &r, q), q);
    for (x, &y) in right.iter_mut().zip(&hb) {
        *x = (*x + y) % q;
    }
    let second = small_transpose_times(d2t, m, &lift(right));
    let image = transpose_times(u, jb, &r, q);
    for j in 0..jb {
        assert_eq!(
            residue(first[j] + second[j], q),
            image[j],
            "U is not F_id D"
        );
    }

    let host = Host {
        setup: &setup,
        h: &h,
    };
    let mut t = Totals::default();
    for &(size, bits) in shapes {
        let g = size * bits;
        let (users, group) = common::users_and_group(size);
        eprintln!("--- {size} recipient(s) of {users} users, {bits} hidden bit(s)");
        let (apk, keys) = covert_keygen(pk, users, bits, &mut rng).unwrap();
        let all: Vec<i32> = keys.iter().flat_map(|key| key.transposed_keys()).collect();
        assert!(all.iter().all(|&x| x == 0 || x == 1));
        let ones = all.iter().filter(|&&x| x == 1).count() as u64;
        assert!(
            balanced(ones, all.len() as u64),
            "anamorphic key bits are not balanced"
        );
        assert_eq!(
            all.chunks(m).collect::<BTreeSet<_>>().len(),
            users * bits,
            "repeated anamorphic key block"
        );
        let plan = prepare_hidden(pk, &apk, &group).unwrap();
        assert!(plan.sampler().full_rank(), "H_S is rank deficient");
        for (j, column) in plan.constraints().iter().enumerate() {
            let (i, bit) = (group[j / bits], j % bits);
            let key = &keys[i].transposed_keys()[bit * m..(bit + 1) * m];
            for row in 0..n {
                let a = &a0[row * m..(row + 1) * m];
                let image: u128 = a
                    .iter()
                    .zip(key)
                    .filter(|&(_, &k)| k == 1)
                    .map(|(&x, _)| u128::from(x))
                    .sum();
                let expect =
                    (u128::from(a[0]) + u128::from(q) - image % u128::from(q)) % u128::from(q);
                assert_eq!(
                    u128::from(column[row]),
                    expect,
                    "h_{j} is not A0[:,0] - A0 k at row {row}"
                );
                assert_eq!(
                    u128::from(apk.vector(i, bit)[row]),
                    image % u128::from(q),
                    "b̂_({i},{bit}) is not A0 k at row {row}"
                );
            }
        }
        check_uniform("H_S (h = A0[:,0] - A0 k)", &plan.constraints().concat(), q);
        let (mut secrets, mut headers, mut plain_secrets) = (Vec::new(), Vec::new(), Vec::new());
        for trial in 0..trials {
            let ordinary: Vec<u8> = (0..jb).map(|j| ((j as u64 + trial) % 2) as u8).collect();
            let hidden: Vec<u8> = (0..bits as u64)
                .map(|a| {
                    (match trial % 4 {
                        0 => a % 2,
                        1 => 1,
                        2 => (a + 1) % 2,
                        _ => 0,
                    }) as u8
                })
                .collect();
            let rhs: Vec<u64> = (0..size)
                .flat_map(|_| hidden.iter().map(|&x| u64::from(x) * delta))
                .collect();
            let coins = seed ^ (trial << 32) ^ ((g as u64) << 16) ^ size as u64;

            // Anamorphic encryption, step by step, then the library call on the same coins.
            let mut stream = StdRng::seed_from_u64(coins);
            let s = plan.sampler().sample(&rhs, &mut stream).unwrap();
            for (column, &target) in plan.constraints().iter().zip(&rhs) {
                let dot = column
                    .iter()
                    .zip(&s)
                    .map(|(&x, &z)| u128::from(x) * u128::from(z))
                    .sum::<u128>();
                assert_eq!(
                    (dot % u128::from(q)) as u64,
                    target,
                    "h^T s is not floor(q/2) bit"
                );
            }
            let (errors, signs) = recorded_errors(p, &mut stream);
            let ct = encrypt_with_coins(pk, &id, &ordinary, &s, &errors).unwrap();
            assert_eq!(
                plan.encrypt(&id, &ordinary, &hidden, &mut StdRng::seed_from_u64(coins))
                    .unwrap(),
                ct,
                "HiddenPlan::encrypt is not the audited computation"
            );
            check_ciphertext(&mut t, &host, &s, &errors, signs, &ct, &ordinary);
            let y = small(&errors.y);
            for (i, key) in keys.iter().enumerate() {
                if group.contains(&i) {
                    check_covert(&mut t, key, &ct, &y, &hidden, q, sigma);
                } else {
                    spread(&mut t, key, &ct, q);
                }
            }
            t.s_hidden.extend(&s);
            secrets.push(s);
            headers.push(ct.c1()[..m].to_vec());

            // Ordinary encryption: uniform s, same errors. Members see nothing.
            let mut stream = StdRng::seed_from_u64(coins + 1);
            let s = uniform(n, q, &mut stream).unwrap();
            let (errors, signs) = recorded_errors(p, &mut stream);
            let ct = encrypt_with_coins(pk, &id, &ordinary, &s, &errors).unwrap();
            assert_eq!(
                encrypt_ordinary(pk, &id, &ordinary, &mut StdRng::seed_from_u64(coins + 1))
                    .unwrap(),
                ct,
                "encrypt_ordinary is not the audited computation"
            );
            check_ciphertext(&mut t, &host, &s, &errors, signs, &ct, &ordinary);
            for &i in &group {
                spread(&mut t, &keys[i], &ct, q);
            }
            t.s_ordinary.extend(&s);
            plain_secrets.push(s);
        }
        check_varies("anamorphic s", &secrets);
        check_varies("ordinary s", &plain_secrets);
        check_varies("c1[..m]", &headers);
    }

    // Rank failure: two users with equal keys make H_S deficient, and aEnc returns
    // the ordinary encryption with the same coins.
    let twin = covert_secret_keygen(p, 1, &mut rng).unwrap();
    let apk = CovertPublicKey::from_secret_keys(pk, &[twin.clone(), twin]).unwrap();
    let deficient = prepare_hidden(pk, &apk, &[0, 1]).unwrap();
    assert!(!deficient.sampler().full_rank());
    let ordinary = vec![1u8; jb];
    let fallback = deficient
        .encrypt(&id, &ordinary, &[1], &mut StdRng::seed_from_u64(seed + 99))
        .unwrap();
    let plain =
        encrypt_ordinary(pk, &id, &ordinary, &mut StdRng::seed_from_u64(seed + 99)).unwrap();
    assert_eq!(fallback, plain);
    assert_eq!(
        decrypt_ordinary(&setup.shortcut, &fallback).unwrap(),
        ordinary
    );

    eprintln!("=== totals over {} ciphertexts", t.correlated_count / m);
    check_rounded_normal("errors y and x", &t.errors, sigma);
    let ratio = t.correlated_square / t.expected_square;
    eprintln!(
        "R^T y: mean square / (sum of y^2) {ratio:.4} over {} columns, sign ones {:.6}",
        t.correlated_count,
        t.sign_ones as f64 / t.sign_bits as f64
    );
    assert!(
        (ratio - 1.0).abs() < 6.0 * (2.0 / t.correlated_count as f64).sqrt(),
        "R^T y spread"
    );
    assert!(
        balanced(t.sign_ones, t.sign_bits),
        "sign bits are not balanced"
    );
    check_uniform("anamorphic s", &t.s_hidden, q);
    check_uniform("ordinary s", &t.s_ordinary, q);
    check_uniform("c1[..m]", &t.c1_top, q);
    check_uniform("c1[m..]", &t.c1_bottom, q);
    check_uniform("c0", &t.c0, q);
    check_noise(
        "ordinary noise",
        &t.ordinary_noise,
        t.ordinary_zero_expected,
    );
    check_noise("covert noise", &t.covert_noise, t.covert_zero_expected);
    let mean = t.outside.iter().sum::<f64>() / t.outside.len() as f64;
    eprintln!(
        "keys outside the audience: {} residuals, mean distance to 0 or floor(q/2) {mean:.4} q (uniform 0.125 q)",
        t.outside.len()
    );
    assert!((mean - 0.125).abs() < 6.0 * 0.0722 / (t.outside.len() as f64).sqrt());
    eprintln!(
        "largest decryption noise {:.3e} of q/4",
        t.largest_noise as f64 / (q / 4) as f64
    );
}

#[test]
fn small_profile_values_are_not_degenerate() {
    audit(
        TrapdoorProfile::small(),
        0x5a17,
        &[(1, 1), (2, 2), (4, 2), (1, 8)],
        12,
        true,
    );
}

#[test]
#[ignore = "full profile: about 2 GB of memory and minutes, run with --release -- --ignored"]
fn full_profile_values_are_not_degenerate() {
    audit(TrapdoorProfile::full(), 0xf011, &[(4, 2)], 2, false);
}
