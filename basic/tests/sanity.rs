//! Value audit: every quantity of a real run must look like what Construction 1
//! prescribes. Nothing may be zero, constant, repeated or missing unless the
//! protocol says so (the target Delta * am is zero for a zero bit, the rank
//! fallback ignores the payload, a single error can be zero with its tiny
//! probability under the rounded normal distribution).
//!
//! Each check is a hard bound at about six standard deviations, so it fails
//! only if something is wrong. Run with `--nocapture` to see the statistics.

use basic_anamorphic::*;
use rand::{Rng, RngExt, SeedableRng, rngs::StdRng};

/// Centered representative of a residue in (-q/2, q/2].
fn center(x: u32, q: u32) -> i64 {
    if x > q / 2 {
        i64::from(x) - i64::from(q)
    } else {
        i64::from(x)
    }
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

/// Uniform residues mod q: mean, spread and the number of zeros.
fn check_uniform(label: &str, values: &[u32], q: u32) {
    let s = stats(values.iter().map(|&v| f64::from(v)));
    let qf = f64::from(q);
    let mean_sd = qf / (12.0 * s.count as f64).sqrt();
    let expected_zeros = s.count as f64 / qf;
    eprintln!(
        "{label}: {} values, mean/q {:.5}, sd/q {:.5}, zeros {}",
        s.count,
        s.mean / qf,
        s.sd / qf,
        s.zeros
    );
    assert!(
        (s.mean - (qf - 1.0) / 2.0).abs() < 6.0 * mean_sd + 1.0,
        "{label}: mean"
    );
    if s.count >= 1000 {
        assert!(
            (s.sd / qf - 12f64.sqrt().recip()).abs() < 0.02,
            "{label}: spread"
        );
    }
    assert!(
        (s.zeros as f64) < expected_zeros + 6.0 * expected_zeros.sqrt() + 3.0,
        "{label}: {} zero entries",
        s.zeros
    );
}

/// Rounded normal errors with standard deviation `sd`.
fn check_errors(label: &str, errors: &[i64], sd: f64, bound: i64) {
    let s = stats(errors.iter().map(|&e| e as f64));
    let n = s.count as f64;
    let expected_zero = 1.0 / (sd * (2.0 * std::f64::consts::PI).sqrt());
    eprintln!(
        "{label}: {} errors, mean {:.1}, sd {:.1} (expected {:.1}), zeros {}, max |e| {}",
        s.count,
        s.mean,
        s.sd,
        sd,
        s.zeros,
        errors.iter().map(|e| e.abs()).max().unwrap()
    );
    assert!(s.mean.abs() < 6.0 * sd / n.sqrt(), "{label}: mean");
    assert!(
        (s.sd / sd - 1.0).abs() < 6.0 / (2.0 * n).sqrt() + 1e-3,
        "{label}: spread"
    );
    assert!(
        (s.zeros as f64) < n * expected_zero + 6.0 * (n * expected_zero).sqrt() + 3.0,
        "{label}: zeros"
    );
    assert!(
        errors.iter().all(|e| e.abs() <= bound),
        "{label}: beyond the table bound"
    );
    assert!(errors.windows(2).any(|w| w[0] != w[1]), "{label}: constant");
}

fn ones_fraction_ok(bits: &[u8]) -> bool {
    let ones = bits.iter().filter(|&&b| b == 1).count() as f64;
    let n = bits.len() as f64;
    bits.iter().all(|&b| b <= 1) && (ones - n / 2.0).abs() < 6.0 * (n / 4.0).sqrt() + 1.0
}

fn audit(n: usize, m: usize, q: u32, n_min: usize, rows: &[(usize, usize)], mode: Mode) {
    // (q sigma)^2 = n (m + 1), used exactly by the noise table.
    let variance = (n * (m + 1)) as u64;
    let sd = (variance as f64).sqrt();
    let population = rows.iter().map(|r| r.0).max().unwrap();
    // Decryption noise: exact, below q/4, and zero only at the rate of its law.
    let (mut residual_count, mut zero_noise, mut largest) = (0usize, 0usize, 0u64);
    for &(t, bits) in rows {
        eprintln!("--- n {n}, m {m}, q {q}, |S| {t}, L {bits}, {mode:?}");
        let p = Parameters::with_noise_variance(n, m, q, variance, n_min, bits).unwrap();
        assert_eq!(noise_table(&p).unwrap().variance(), Some(variance));
        let bound = noise_table(&p).unwrap().bound() as i64;
        let mut rng = StdRng::seed_from_u64(0xA0D1 + (t * 100 + bits) as u64);
        let pair = keygen_with(p, &mut rng, mode).unwrap();
        let (a, b) = (pair.public.matrix(), pair.public.vector());
        check_uniform("A", a, q);
        check_uniform(
            "first column a_1",
            &(0..n).map(|i| a[i * m]).collect::<Vec<_>>(),
            q,
        );
        assert!(ones_fraction_ok(&pair.secret), "ordinary key k");
        check_uniform("b = A k", b, q);
        for compressed in [true, false] {
            let keys = anamorphic_keygen_with(&pair.public, population, compressed, &mut rng, mode)
                .unwrap();
            let mut seen = std::collections::BTreeSet::new();
            for (i, key) in keys.user_keys.iter().enumerate() {
                for (a_index, block) in user_blocks(key, &p).unwrap().iter().enumerate() {
                    assert!(ones_fraction_ok(block), "user {i} block {a_index}");
                    assert!(seen.insert(block.clone()), "repeated key block");
                }
            }
            let all: Vec<u32> = keys.directory.iter().flatten().flatten().copied().collect();
            check_uniform("directory b_{i,a}", &all, q);
            let audience: Vec<usize> = (0..t).rev().collect();
            let plan = prepare_hidden(&pair.public, &keys.directory, &audience).unwrap();
            assert!(plan.fiber().full_rank(), "H_S is rank deficient");
            for column in plan.fiber().matrix() {
                assert!(column.iter().any(|&x| x != 0), "zero column of H_S");
            }
            let h: Vec<u32> = plan.fiber().matrix().iter().flatten().copied().collect();
            check_uniform("H_S", &h, q);
            let k_blocks: Vec<Vec<Vec<u8>>> = keys
                .user_keys
                .iter()
                .map(|key| user_blocks(key, &p).unwrap())
                .collect();
            let mut all_errors = Vec::new();
            let mut outside = Vec::new();
            let mut noise_seeds = std::collections::BTreeSet::new();
            for trial in 0..3u64 {
                let bit = (trial % 2) as u8;
                let payload: Vec<u8> = (0..bits)
                    .map(|x| ((trial as usize + x) % 2) as u8)
                    .collect();
                // Anamorphic encryption with its coins recomputed from the same stream.
                // Every row, trial and key representation gets its own coins, and
                // bit 0 marks the ordinary encryption below.
                let seed =
                    (((t * 100 + bits) as u64) << 8) | (trial << 2) | (u64::from(!compressed) << 1);
                let ct = plan
                    .encrypt_with(bit, &payload, &mut StdRng::seed_from_u64(seed), mode)
                    .unwrap();
                let mut coins = StdRng::seed_from_u64(seed);
                let s = plan.sample_secret(&payload, &mut coins).unwrap();
                let mut noise_seed = [0u8; 32];
                coins.fill_bytes(&mut noise_seed);
                assert!(noise_seeds.insert(noise_seed), "noise seed repeated");
                check_uniform("s", &s, q);
                // H_S^T s = Delta * am for every member: zero exactly where am is zero.
                for (j, column) in plan.fiber().matrix().iter().enumerate() {
                    let dot = column
                        .iter()
                        .zip(&s)
                        .map(|(&x, &y)| u128::from(x) * u128::from(y))
                        .sum::<u128>()
                        % u128::from(q);
                    assert_eq!(dot as u32, p.delta() * u32::from(payload[j % bits]));
                }
                let (errors, y) = noise_from_seed(&p, &noise_seed).unwrap();
                // The ciphertext really contains A^T s + e and b^T s + y + Delta m.
                let header = ct.header();
                check_uniform("header r", header, q);
                for (j, (&r, &e)) in header.iter().zip(&errors).enumerate() {
                    let ats = (0..n)
                        .map(|i| u128::from(a[i * m + j]) * u128::from(s[i]))
                        .sum::<u128>()
                        % u128::from(q);
                    let found = center(
                        ((u128::from(r) + u128::from(q) - ats) % u128::from(q)) as u32,
                        q,
                    );
                    assert_eq!(found, e, "header coordinate {j} does not carry its error");
                    if j > 64 {
                        break; // the full equation is checked by tests/optimized.rs
                    }
                }
                let bts = b
                    .iter()
                    .zip(&s)
                    .map(|(&x, &z)| u128::from(x) * u128::from(z))
                    .sum::<u128>();
                let scalar_noise = center(
                    ((u128::from(ct.scalar()) + u128::from(q) * 2
                        - bts % u128::from(q)
                        - u128::from(p.delta()) * u128::from(bit))
                        % u128::from(q)) as u32,
                    q,
                );
                assert_eq!(scalar_noise, y);
                // Decryption residuals: message plus small noise.
                let ordinary = center(
                    (ordinary_residual(&ct, &pair.secret, q).unwrap() + q
                        - p.delta() * u32::from(bit))
                        % q,
                    q,
                );
                let kte: i64 = pair
                    .secret
                    .iter()
                    .zip(&errors)
                    .map(|(&k, &e)| i64::from(k) * e)
                    .sum();
                assert_eq!(ordinary, y - kte, "ordinary residual noise");
                assert!(ordinary.unsigned_abs() < u64::from(q / 4));
                residual_count += 1;
                zero_noise += usize::from(ordinary == 0);
                largest = largest.max(ordinary.unsigned_abs());
                assert_eq!(ordinary_decrypt(&ct, &pair.secret, q).unwrap(), bit);
                for &i in plan.audience() {
                    let residuals = hidden_residuals(&ct, &keys.user_keys[i], &p).unwrap();
                    for (a_index, &z) in residuals.iter().enumerate() {
                        let noise =
                            center((z + q - p.delta() * u32::from(payload[a_index])) % q, q);
                        let kte: i64 = k_blocks[i][a_index]
                            .iter()
                            .zip(&errors)
                            .map(|(&k, &e)| i64::from(k) * e)
                            .sum();
                        assert_eq!(noise, errors[0] - kte, "hidden residual noise");
                        assert!(noise.unsigned_abs() < u64::from(q / 4));
                        residual_count += 1;
                        zero_noise += usize::from(noise == 0);
                        largest = largest.max(noise.unsigned_abs());
                    }
                    assert_eq!(
                        hidden_decrypt(&ct, &keys.user_keys[i], &p).unwrap(),
                        payload
                    );
                }
                // Users outside S get residuals spread over Z_q, not near 0 or Delta.
                for i in (0..population).filter(|i| !plan.audience().contains(i)) {
                    for z in hidden_residuals(&ct, &keys.user_keys[i], &p).unwrap() {
                        let to_zero = center(z, q).unsigned_abs();
                        let to_delta = center((z + q - p.delta()) % q, q).unsigned_abs();
                        outside.push(to_zero.min(to_delta) as f64 / f64::from(q));
                    }
                }
                all_errors.extend(errors.iter().copied());
                all_errors.push(y);
                // Each ChaCha12 stream gives different errors (stream ids work).
                let chunks: Vec<&[i64]> = errors.chunks(NOISE_CHUNK).collect();
                for x in 0..chunks.len() {
                    for z in x + 1..chunks.len() {
                        let len = chunks[x].len().min(chunks[z].len());
                        assert_ne!(
                            chunks[x][..len],
                            chunks[z][..len],
                            "streams {x} and {z} repeat"
                        );
                    }
                }
                // Ordinary encryption: uniform s, same error distribution.
                let ct = encrypt_with(
                    &pair.public,
                    bit,
                    &mut StdRng::seed_from_u64(seed | 1),
                    mode,
                )
                .unwrap();
                let mut coins = StdRng::seed_from_u64(seed | 1);
                let s: Vec<u32> = (0..n).map(|_| coins.random_range(0..q)).collect();
                check_uniform("ordinary s", &s, q);
                assert_eq!(ordinary_decrypt(&ct, &pair.secret, q).unwrap(), bit);
                // Members of S see nothing in an ordinary ciphertext.
                for &i in plan.audience() {
                    for z in hidden_residuals(&ct, &keys.user_keys[i], &p).unwrap() {
                        let to_zero = center(z, q).unsigned_abs();
                        let to_delta = center((z + q - p.delta()) % q, q).unsigned_abs();
                        outside.push(to_zero.min(to_delta) as f64 / f64::from(q));
                    }
                }
            }
            check_errors("errors e and y", &all_errors, sd, bound);
            {
                // Uniform residues: distance to {0, Delta} is uniform on [0, q/4].
                let mean = outside.iter().sum::<f64>() / outside.len() as f64;
                eprintln!(
                    "keys that see no payload (outsiders, members on ordinary ciphertexts): {} residuals, mean distance to 0 or Delta {mean:.4} q (uniform 0.125 q)",
                    outside.len()
                );
                assert!((mean - 0.125).abs() < 6.0 * 0.0722 / (outside.len() as f64).sqrt());
            }
            // Rank fallback: a repeated public vector makes H_S deficient, and aEnc
            // returns an ordinary encryption with the same coins.
            if t >= 2 {
                let mut repeated = keys.directory.clone();
                repeated[1] = repeated[0].clone();
                let deficient = prepare_hidden(&pair.public, &repeated, &[0, 1]).unwrap();
                assert!(!deficient.fiber().full_rank());
                let payload = vec![1; bits];
                let x = deficient
                    .encrypt_with(1, &payload, &mut StdRng::seed_from_u64(9), mode)
                    .unwrap();
                let y = encrypt_with(&pair.public, 1, &mut StdRng::seed_from_u64(9), mode).unwrap();
                assert_eq!(x, y);
                assert_eq!(ordinary_decrypt(&x, &pair.secret, q).unwrap(), 1);
            }
            let _ = rng.random::<u64>();
        }
    }
    // The noise has standard deviation above 500 here, so a zero is rare.
    eprintln!(
        "{residual_count} decryption residuals, largest noise {:.4} of q/4, zero noise {zero_noise}",
        largest as f64 / f64::from(q / 4)
    );
    assert!(zero_noise <= 3, "{zero_noise} zero decryption noises");
}

#[test]
fn small_profile_values_are_not_degenerate() {
    for mode in [Mode::Single, Mode::Multi] {
        audit(96, 2900, 1_000_003, 64, &[(1, 2), (4, 8)], mode);
    }
}

#[test]
#[cfg_attr(debug_assertions, ignore = "production profile: run with --release")]
fn production_profile_values_are_not_degenerate() {
    audit(
        1152,
        30357,
        67537879,
        1120,
        &[(1, 2), (4, 8), (8, 4)],
        Mode::Single,
    );
    audit(1152, 30357, 67537879, 1120, &[(4, 2)], Mode::Multi);
}
