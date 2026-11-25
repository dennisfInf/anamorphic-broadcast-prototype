//! The optimized implementation against the frozen reference in
//! `tests/legacy/basic_v1.rs` (the code before the kernels, modes and noise
//! table were added), and the two execution modes against each other.
//!
//! Everything that does not involve the error sampler must be bit-identical
//! to the reference: keys, directories, eliminations, affine solutions,
//! ciphertexts for explicit coins, residuals and the random-number
//! consumption of every step that draws from the caller's RNG.

#[path = "legacy/basic_v1.rs"]
mod legacy;

use basic_anamorphic::*;
use rand::{Rng, RngExt, SeedableRng, rngs::StdRng};

const MODES: [Mode; 2] = [Mode::Single, Mode::Multi];

fn both(
    n: usize,
    m: usize,
    q: u32,
    sd: f64,
    n_min: usize,
    l: usize,
) -> (Parameters, legacy::Parameters) {
    (
        Parameters::new(n, m, q, sd, n_min, l).unwrap(),
        legacy::Parameters::new(n, m, q, sd, n_min, l).unwrap(),
    )
}

fn next_words(rng: &mut StdRng) -> [u64; 4] {
    std::array::from_fn(|_| rng.next_u64())
}

/// Small, medium, odd-sized and near-2^32 profiles (n, m, q, L).
const PROFILES: [(usize, usize, u32, usize); 6] = [
    (6, 28, 257, 2),
    (9, 13, 3, 1),
    (24, 161, 12289, 3),
    (40, 1003, 67537879, 4),
    (8, 77, 4294967291, 2),
    (70, 2049, 2147483647, 5),
];

fn legacy_key(key: &UserKey) -> legacy::UserKey {
    match key {
        UserKey::Seed(seed) => legacy::UserKey::Seed(*seed),
        UserKey::Expanded(blocks) => legacy::UserKey::Expanded(blocks.clone()),
    }
}

#[test]
fn keys_and_directories_match_the_reference_in_both_modes() {
    for (index, &(n, m, q, l)) in PROFILES.iter().enumerate() {
        let (p, lp) = both(n, m, q, 2.5, n / 2, l);
        let seed = 1000 + index as u64;
        let mut reference_rng = StdRng::seed_from_u64(seed);
        let reference = legacy::keygen(lp, &mut reference_rng).unwrap();
        let reference_words = next_words(&mut reference_rng);
        for mode in MODES {
            let mut rng = StdRng::seed_from_u64(seed);
            let pair = keygen_with(p, &mut rng, mode).unwrap();
            assert_eq!(pair.public.matrix(), reference.public.matrix());
            assert_eq!(pair.public.vector(), reference.public.vector());
            assert_eq!(pair.secret, reference.secret);
            assert_eq!(next_words(&mut rng), reference_words);
            for compressed in [false, true] {
                let mut rng = StdRng::seed_from_u64(seed + 50);
                let mut reference_rng = StdRng::seed_from_u64(seed + 50);
                let keys =
                    anamorphic_keygen_with(&pair.public, 5, compressed, &mut rng, mode).unwrap();
                let expected =
                    legacy::anamorphic_keygen(&reference.public, 5, compressed, &mut reference_rng)
                        .unwrap();
                assert_eq!(
                    keys.directory, expected.directory,
                    "profile {index} {mode:?}"
                );
                assert_eq!(keys.user_keys.len(), expected.user_keys.len());
                for (a, b) in keys.user_keys.iter().zip(&expected.user_keys) {
                    assert!(legacy_key(a) == *b);
                    assert_eq!(
                        user_blocks(a, &p).unwrap(),
                        legacy::user_blocks(b, &lp).unwrap()
                    );
                }
                assert_eq!(next_words(&mut rng), next_words(&mut reference_rng));
            }
        }
    }
}

#[test]
fn eliminations_and_affine_solutions_match_the_reference() {
    let mut rng = StdRng::seed_from_u64(77);
    let (mut full, mut deficient) = (0, 0);
    for &(n, m, q, l) in &PROFILES {
        let (p, lp) = both(n, m, q, 2.5, n / 2, l);
        let mut setup = StdRng::seed_from_u64(78);
        let reference = legacy::keygen(lp, &mut StdRng::seed_from_u64(78)).unwrap();
        let pair = keygen(p, &mut setup).unwrap();
        let keys = anamorphic_keygen(&pair.public, 6, rng.random(), &mut rng).unwrap();
        // Random audiences, plus rank-deficient directories (repeated vectors and
        // vectors equal to the first column of A).
        let mut directories = vec![keys.directory.clone()];
        let mut repeated = keys.directory.clone();
        repeated[1] = repeated[0].clone();
        directories.push(repeated);
        let mut degenerate = keys.directory.clone();
        degenerate[2][0] = (0..n).map(|j| pair.public.matrix()[j * m]).collect();
        directories.push(degenerate);
        let capacity = (n - n / 2) / l;
        for directory in &directories {
            for size in 1..=capacity.min(4) {
                for _ in 0..3 {
                    let mut audience: Vec<usize> = (0..6).collect();
                    for i in (1..audience.len()).rev() {
                        audience.swap(i, rng.random_range(0..=i));
                    }
                    audience.truncate(size);
                    let plan = prepare_hidden(&pair.public, directory, &audience).unwrap();
                    let expected =
                        legacy::prepare_hidden(&reference.public, directory, &audience).unwrap();
                    let (f, g) = (plan.fiber(), expected.fiber());
                    assert_eq!(plan.audience(), expected.audience());
                    assert_eq!(f.matrix(), g.matrix());
                    assert_eq!(f.reduced(), g.reduced());
                    assert_eq!(f.transform(), g.transform());
                    assert_eq!(f.pivots(), g.pivots());
                    assert_eq!(f.free(), g.free());
                    assert_eq!(f.full_rank(), g.full_rank());
                    if f.full_rank() {
                        full += 1;
                    } else {
                        deficient += 1;
                    }
                    let d = f.matrix().len();
                    let target: Vec<u32> = (0..d).map(|_| rng.random_range(0..q)).collect();
                    let free: Vec<u32> = (0..f.free().len()).map(|_| rng.random()).collect();
                    assert_eq!(
                        f.from_free(&target, &free).ok(),
                        g.from_free(&target, &free).ok()
                    );
                    let mut a = StdRng::seed_from_u64(5);
                    let mut b = StdRng::seed_from_u64(5);
                    assert_eq!(
                        f.sample(&target, &mut a).ok(),
                        g.sample(&target, &mut b).ok()
                    );
                    assert_eq!(next_words(&mut a), next_words(&mut b));
                    // The broadcast step equals the reference sampler on the
                    // target Delta * am repeated for every member.
                    let payload: Vec<u8> = (0..l).map(|_| rng.random_range(0..2)).collect();
                    let target: Vec<u32> = plan
                        .audience()
                        .iter()
                        .flat_map(|_| payload.iter().map(|&b| p.delta() * u32::from(b)))
                        .collect();
                    let mut a = StdRng::seed_from_u64(6);
                    let mut b = StdRng::seed_from_u64(6);
                    if f.full_rank() {
                        assert_eq!(
                            plan.sample_secret(&payload, &mut a).unwrap(),
                            g.sample(&target, &mut b).unwrap()
                        );
                    } else {
                        assert!(plan.sample_secret(&payload, &mut a).is_err());
                    }
                    assert_eq!(next_words(&mut a), next_words(&mut b));
                }
            }
        }
    }
    assert!(
        full > 20 && deficient > 10,
        "{full} full-rank and {deficient} deficient plans"
    );
}

#[test]
fn ciphertexts_and_residuals_match_the_reference_for_explicit_coins() {
    let mut rng = StdRng::seed_from_u64(91);
    for &(n, m, q, l) in &PROFILES {
        let (p, lp) = both(n, m, q, 2.5, n / 2, l);
        let mut setup = StdRng::seed_from_u64(92);
        let reference = legacy::keygen(lp, &mut StdRng::seed_from_u64(92)).unwrap();
        let pair = keygen(p, &mut setup).unwrap();
        let keys = anamorphic_keygen(&pair.public, 3, true, &mut rng).unwrap();
        for trial in 0..6 {
            let secret: Vec<u32> = (0..n).map(|_| rng.random_range(0..q)).collect();
            let errors: Vec<i64> = (0..m)
                .map(|j| match (trial + j) % 7 {
                    0 => i64::MIN,
                    1 => i64::MAX,
                    2 => rng.random_range(-5000..5000),
                    _ => rng.random(),
                })
                .collect();
            let scalar_error = if trial % 2 == 0 {
                i64::MIN
            } else {
                rng.random()
            };
            let bit = (trial % 2) as u8;
            let expected =
                legacy::native_from_coins(&reference.public, bit, &secret, &errors, scalar_error)
                    .unwrap();
            for mode in MODES {
                let ct =
                    native_from_coins_with(&pair.public, bit, &secret, &errors, scalar_error, mode)
                        .unwrap();
                assert_eq!(ct.header(), expected.header());
                assert_eq!(ct.scalar(), expected.scalar());
            }
            let ct = native_from_coins(&pair.public, bit, &secret, &errors, scalar_error).unwrap();
            assert_eq!(
                ordinary_residual(&ct, &pair.secret, q).unwrap(),
                legacy::ordinary_residual(&expected, &reference.secret, q).unwrap()
            );
            for key in &keys.user_keys {
                let expanded = UserKey::Expanded(user_blocks(key, &p).unwrap());
                for key in [key, &expanded] {
                    assert_eq!(
                        hidden_residuals(&ct, key, &p).unwrap(),
                        legacy::hidden_residuals(&expected, &legacy_key(key), &lp).unwrap()
                    );
                }
            }
        }
    }
}

#[test]
fn encryption_is_identical_in_both_modes_and_uses_the_stated_coins() {
    let (p, _) = both(40, 1003, 67537879, 1000.0, 20, 4);
    let mut setup = StdRng::seed_from_u64(5);
    let pair = keygen(p, &mut setup).unwrap();
    let keys = anamorphic_keygen(&pair.public, 4, true, &mut setup).unwrap();
    let plan = prepare_hidden(&pair.public, &keys.directory, &[3, 1]).unwrap();
    assert!(plan.fiber().full_rank());
    for trial in 0..4u64 {
        let bit = (trial % 2) as u8;
        let payload = [1, 0, (trial % 2) as u8, 1];
        // Ordinary encryption draws s (n field elements) and then a 32-byte seed.
        let mut a = StdRng::seed_from_u64(trial);
        let mut b = StdRng::seed_from_u64(trial);
        let single = encrypt_with(&pair.public, bit, &mut a, Mode::Single).unwrap();
        let multi = encrypt_with(&pair.public, bit, &mut b, Mode::Multi).unwrap();
        assert_eq!(single, multi);
        let mut c = StdRng::seed_from_u64(trial);
        let secret: Vec<u32> = (0..p.n()).map(|_| c.random_range(0..p.q())).collect();
        let mut seed = [0u8; 32];
        c.fill_bytes(&mut seed);
        let words = next_words(&mut c);
        assert_eq!(next_words(&mut a), words);
        assert_eq!(next_words(&mut b), words);
        let (errors, scalar) = noise_from_seed(&p, &seed).unwrap();
        assert_eq!(
            single,
            native_from_coins(&pair.public, bit, &secret, &errors, scalar).unwrap()
        );
        assert_eq!(
            single,
            encrypt_from_seed(&pair.public, bit, &secret, &seed, Mode::Multi).unwrap()
        );
        // Anamorphic encryption draws the free coordinates, then the seed.
        let mut a = StdRng::seed_from_u64(100 + trial);
        let mut b = StdRng::seed_from_u64(100 + trial);
        let single = plan
            .encrypt_with(bit, &payload, &mut a, Mode::Single)
            .unwrap();
        let multi = plan
            .encrypt_with(bit, &payload, &mut b, Mode::Multi)
            .unwrap();
        assert_eq!(single, multi);
        let mut c = StdRng::seed_from_u64(100 + trial);
        let secret = plan.sample_secret(&payload, &mut c).unwrap();
        let mut seed = [0u8; 32];
        c.fill_bytes(&mut seed);
        let words = next_words(&mut c);
        assert_eq!(next_words(&mut a), words);
        assert_eq!(next_words(&mut b), words);
        assert_eq!(
            single,
            encrypt_from_seed(&pair.public, bit, &secret, &seed, Mode::Single).unwrap()
        );
        assert_eq!(ordinary_decrypt(&single, &pair.secret, p.q()).unwrap(), bit);
        for &i in plan.audience() {
            assert_eq!(
                hidden_decrypt(&single, &keys.user_keys[i], &p).unwrap(),
                payload
            );
        }
    }
}

#[test]
fn noise_seeds_give_fixed_streams_per_chunk() {
    let (p, _) = both(16, 3000, 67537879, 77.5, 8, 1);
    let seed = [7u8; 32];
    let (errors, scalar) = noise_from_seed(&p, &seed).unwrap();
    assert_eq!(
        noise_from_seed(&p, &seed).unwrap(),
        (errors.clone(), scalar)
    );
    // A longer header with the same seed starts with the same streams.
    let (longer, _) = both(16, 5000, 67537879, 77.5, 8, 1);
    let (more, _) = noise_from_seed(&longer, &seed).unwrap();
    assert_eq!(&more[..2 * NOISE_CHUNK], &errors[..2 * NOISE_CHUNK]);
    let other = noise_from_seed(&p, &[8u8; 32]).unwrap();
    assert_ne!(other.0, errors);
    let bound = noise_table(&p).unwrap().bound() as i64;
    assert!(errors.iter().chain([&scalar]).all(|e| e.abs() <= bound));
}

#[test]
#[cfg_attr(debug_assertions, ignore = "production profile: run with --release")]
fn production_profile_matches_the_reference() {
    let (n, m, q) = (1152, 30357, 67537879);
    let sd = ((n * (m + 1)) as f64).sqrt();
    let (p, lp) = both(n, m, q, sd, 1120, 8);
    let mut setup = StdRng::seed_from_u64(2026);
    let reference = legacy::keygen(lp, &mut StdRng::seed_from_u64(2026)).unwrap();
    let pair = keygen_with(p, &mut setup, Mode::Multi).unwrap();
    assert_eq!(pair.public.vector(), reference.public.vector());
    let mut a = StdRng::seed_from_u64(3);
    let mut b = StdRng::seed_from_u64(3);
    let keys = anamorphic_keygen_with(&pair.public, 4, true, &mut a, Mode::Multi).unwrap();
    let expected = legacy::anamorphic_keygen(&reference.public, 4, true, &mut b).unwrap();
    assert_eq!(keys.directory, expected.directory);
    let plan = prepare_hidden(&pair.public, &keys.directory, &[0, 1, 2, 3]).unwrap();
    let reference_plan =
        legacy::prepare_hidden(&reference.public, &expected.directory, &[0, 1, 2, 3]).unwrap();
    assert_eq!(plan.fiber().transform(), reference_plan.fiber().transform());
    assert_eq!(plan.fiber().reduced(), reference_plan.fiber().reduced());
    let payload = [1, 0, 1, 1, 0, 0, 1, 0];
    let mut a = StdRng::seed_from_u64(4);
    let mut b = StdRng::seed_from_u64(4);
    let secret = plan.sample_secret(&payload, &mut a).unwrap();
    let target: Vec<u32> = (0..4)
        .flat_map(|_| payload.iter().map(|&x| p.delta() * u32::from(x)))
        .collect();
    assert_eq!(
        secret,
        reference_plan.fiber().sample(&target, &mut b).unwrap()
    );
    let mut seed = [0u8; 32];
    a.fill_bytes(&mut seed);
    let (errors, scalar) = noise_from_seed(&p, &seed).unwrap();
    let expected_ct =
        legacy::native_from_coins(&reference.public, 1, &secret, &errors, scalar).unwrap();
    for mode in MODES {
        let ct = encrypt_from_seed(&pair.public, 1, &secret, &seed, mode).unwrap();
        assert_eq!(ct.header(), expected_ct.header());
        assert_eq!(ct.scalar(), expected_ct.scalar());
        assert_eq!(ordinary_decrypt(&ct, &pair.secret, q).unwrap(), 1);
        for i in 0..4 {
            assert_eq!(
                hidden_residuals(&ct, &keys.user_keys[i], &p).unwrap(),
                legacy::hidden_residuals(&expected_ct, &legacy_key(&keys.user_keys[i]), &lp)
                    .unwrap()
            );
            assert_eq!(
                hidden_decrypt(&ct, &keys.user_keys[i], &p).unwrap(),
                payload
            );
        }
    }
}

#[test]
fn multi_mode_results_do_not_depend_on_the_thread_count() {
    let (p, _) = both(48, 5000, 67537879, 300.0, 24, 3);
    let run = || {
        let mut rng = StdRng::seed_from_u64(42);
        let pair = keygen_with(p, &mut rng, Mode::Multi).unwrap();
        let keys = anamorphic_keygen_with(&pair.public, 5, true, &mut rng, Mode::Multi).unwrap();
        let plan = prepare_hidden(&pair.public, &keys.directory, &[4, 0, 2]).unwrap();
        let ordinary = encrypt_with(&pair.public, 1, &mut rng, Mode::Multi).unwrap();
        let hidden = plan
            .encrypt_with(0, &[1, 1, 0], &mut rng, Mode::Multi)
            .unwrap();
        (
            pair.public.vector().to_vec(),
            keys.directory,
            ordinary,
            hidden,
        )
    };
    let reference = run();
    for threads in [1, 2, 3, 7] {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .unwrap();
        assert!(pool.install(run) == reference, "{threads} threads");
    }
}
