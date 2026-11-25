//! Independent Python explicit-coin vectors and finite algebraic regression checks.
//! These tests do not assert equivalence of the two numerical Gaussian samplers.

use basic_anamorphic::*;
use rand::{Rng, RngExt, SeedableRng, rngs::StdRng};
use serde::de::DeserializeOwned;
use serde_json::Value;
use std::collections::BTreeSet;

fn fixture() -> Value {
    serde_json::from_str(include_str!("fixtures/python.json")).unwrap()
}

fn value<T: DeserializeOwned>(v: &Value) -> T {
    serde_json::from_value(v.clone()).unwrap()
}

fn parameters(v: &Value) -> Parameters {
    Parameters::new(
        value(&v["n"]),
        value(&v["m"]),
        value(&v["q"]),
        value(&v["noise_sd"]),
        value(&v["n_min"]),
        value(&v["L"]),
    )
    .unwrap()
}

fn public_key(f: &Value) -> PublicKey {
    PublicKey::new(
        parameters(&f["params"]),
        value(&f["matrix"]),
        value(&f["vector"]),
    )
    .unwrap()
}

fn keys(f: &Value, source: &str) -> Vec<UserKey> {
    f["key_sets"][source]["user_keys"]
        .as_array()
        .unwrap()
        .iter()
        .map(|k| {
            if source == "seeded" {
                UserKey::Seed(value::<Vec<u8>>(k).try_into().unwrap())
            } else {
                UserKey::Expanded(value(k))
            }
        })
        .collect()
}

fn directory(f: &Value, source: &str) -> Vec<Vec<Vec<u32>>> {
    value(&f["key_sets"][source]["directory"])
}

fn next_words(rng: &mut StdRng) -> [u64; 4] {
    std::array::from_fn(|_| rng.next_u64())
}

#[test]
fn python_seed_vectors_include_non_byte_aligned_blocks() {
    let f = fixture();
    for case in f["seed_vectors"].as_array().unwrap() {
        let p = parameters(&case["params"]);
        let seed: [u8; 32] = value::<Vec<u8>>(&case["seed"]).try_into().unwrap();
        let expected: Vec<Vec<u8>> = value(&case["blocks"]);
        let blocks = expand_seed(&seed, &p);
        assert_eq!(blocks, expected);
        // Pack independently: block boundaries must not restart at a byte boundary.
        let mut packed = vec![0u8; (p.m() * p.payload_bits()).div_ceil(8)];
        for (i, bit) in blocks.iter().flatten().enumerate() {
            packed[i / 8] |= bit << (i % 8);
        }
        let mut expected_bytes: Vec<u8> = value(&case["shake_bytes"]);
        let last_bits = p.m() * p.payload_bits() % 8;
        if last_bits != 0 {
            *expected_bytes.last_mut().unwrap() &= (1u8 << last_bits) - 1;
        }
        assert_eq!(packed, expected_bytes);
    }
}

#[test]
fn python_explicit_coins_match_affine_solutions_ciphertexts_and_residuals() {
    let f = fixture();
    let pk = public_key(&f);
    let p = pk.params();
    let ordinary_key: Vec<u8> = value(&f["ordinary_key"]);
    for case in f["cases"].as_array().unwrap() {
        let secret: Vec<u32> = value(&case["secret"]);
        let errors: Vec<i64> = value(&case["errors"]);
        let bit: u8 = value(&case["bit"]);
        let ct =
            native_from_coins(&pk, bit, &secret, &errors, value(&case["scalar_error"])).unwrap();
        let mut fields = ct.header().to_vec();
        fields.push(ct.scalar());
        assert_eq!(
            fields,
            value::<Vec<u32>>(&case["fields"]),
            "{}",
            case["name"]
        );
        let bytes: Vec<u8> = value(&case["bytes"]);
        assert_eq!(serialize(&ct, p).unwrap(), bytes);
        let decoded = deserialize(&bytes, p).unwrap();
        assert_eq!(decoded.header(), ct.header());
        assert_eq!(decoded.scalar(), ct.scalar());
        assert_eq!(
            ordinary_residual(&ct, &ordinary_key, p.q()).unwrap(),
            value::<u32>(&case["ordinary_residual"])
        );
        assert_eq!(ordinary_decrypt(&ct, &ordinary_key, p.q()).unwrap(), bit);
        for source in ["expanded", "seeded"] {
            for (i, key) in keys(&f, source).iter().enumerate() {
                assert_eq!(
                    user_blocks(key, p).unwrap(),
                    value::<Vec<Vec<u8>>>(&f["key_sets"][source]["blocks"][i])
                );
                assert_eq!(
                    hidden_residuals(&ct, key, p).unwrap(),
                    value::<Vec<u32>>(&case["hidden_residuals"][source][i])
                );
            }
        }
        if let Some(source) = case["source"].as_str() {
            let audience: Vec<usize> = value(&case["audience_input"]);
            let dir = directory(&f, source);
            let plan = prepare_hidden(&pk, &dir, &audience).unwrap();
            assert_eq!(plan.audience(), value::<Vec<usize>>(&case["audience"]));
            assert_eq!(
                plan.fiber().matrix(),
                value::<Vec<Vec<u32>>>(&case["constraints"])
            );
            assert_eq!(plan.fiber().pivots(), value::<Vec<usize>>(&case["pivots"]));
            assert_eq!(
                plan.fiber().free(),
                value::<Vec<usize>>(&case["free_columns"])
            );
            assert_eq!(
                plan.fiber().reduced(),
                value::<Vec<Vec<u32>>>(&case["reduced"])
            );
            assert_eq!(
                plan.fiber().transform(),
                value::<Vec<Vec<u32>>>(&case["transform"])
            );
            assert_eq!(
                plan.fiber()
                    .from_free(
                        &value::<Vec<u32>>(&case["target"]),
                        &value::<Vec<u32>>(&case["free_values"])
                    )
                    .unwrap(),
                secret
            );
            let payload: Vec<u8> = value(&case["payload"]);
            for recipient in plan.audience() {
                assert_eq!(
                    hidden_decrypt(&ct, &keys(&f, source)[*recipient], p).unwrap(),
                    payload
                );
            }
        }
    }
}

#[test]
fn all_payloads_and_audiences_recover_in_both_key_representations() {
    let f = fixture();
    let pk = public_key(&f);
    let sk: Vec<u8> = value(&f["ordinary_key"]);
    let mut rng = StdRng::seed_from_u64(173);
    for source in ["expanded", "seeded"] {
        let dir = directory(&f, source);
        let user_keys = keys(&f, source);
        for audience in [
            vec![0],
            vec![1],
            vec![2],
            vec![0, 1],
            vec![0, 2],
            vec![1, 2],
        ] {
            let plan = prepare_hidden(&pk, &dir, &audience).unwrap();
            assert!(plan.fiber().full_rank());
            for code in 0..4 {
                let payload = [(code & 1) as u8, (code >> 1) as u8];
                for bit in 0..2 {
                    let ct = plan.encrypt(bit, &payload, &mut rng).unwrap();
                    assert_eq!(ordinary_decrypt(&ct, &sk, pk.params().q()).unwrap(), bit);
                    for i in &audience {
                        assert_eq!(
                            hidden_decrypt(&ct, &user_keys[*i], pk.params()).unwrap(),
                            payload
                        );
                    }
                }
            }
        }
    }
}

#[test]
fn exhaustive_affine_fibers_are_bijections() {
    // Every 2 x 3 matrix over F_3, all targets and all affine free values.
    let mut full_rank_count = 0;
    for code in 0..729 {
        let mut rest = code;
        let mut rows = vec![vec![0; 3]; 2];
        for x in rows.iter_mut().flatten() {
            *x = rest % 3;
            rest /= 3;
        }
        let fiber = AffineFiber::new(&rows, 3, 3).unwrap();
        if !fiber.full_rank() {
            let mut rng = StdRng::seed_from_u64(51);
            let mut control = StdRng::seed_from_u64(51);
            assert!(fiber.sample(&[0, 0], &mut rng).is_err());
            assert_eq!(next_words(&mut rng), next_words(&mut control));
            continue;
        }
        full_rank_count += 1;
        for t0 in 0..3 {
            for t1 in 0..3 {
                let target = [t0, t1];
                let actual: BTreeSet<_> = (0..3)
                    .map(|x| fiber.from_free(&target, &[x]).unwrap())
                    .collect();
                let mut expected = BTreeSet::new();
                for x0 in 0..3 {
                    for x1 in 0..3 {
                        for x2 in 0..3 {
                            let point = vec![x0, x1, x2];
                            if rows
                                .iter()
                                .map(|row| {
                                    row.iter().zip(&point).map(|(a, b)| a * b).sum::<u32>() % 3
                                })
                                .eq(target)
                            {
                                expected.insert(point);
                            }
                        }
                    }
                }
                assert_eq!(actual.len(), 3);
                assert_eq!(actual, expected);
            }
        }
    }
    assert_eq!(full_rank_count, (27 - 1) * (27 - 3));
}

#[test]
fn rank_failure_uses_exact_ordinary_rng_path() {
    let p = Parameters::new(6, 28, 257, 0.25, 2, 2).unwrap();
    let pk = PublicKey::new(p, vec![0; p.n() * p.m()], vec![0; p.n()]).unwrap();
    let hidden = anamorphic_keygen(&pk, 3, false, &mut StdRng::seed_from_u64(71)).unwrap();
    let plan = prepare_hidden(&pk, &hidden.directory, &[0, 1]).unwrap();
    assert_eq!(plan.fiber().rank(), 0);
    for code in 0..4 {
        let payload = [(code & 1) as u8, (code >> 1) as u8];
        for bit in 0..2 {
            let mut normal_rng = StdRng::seed_from_u64(83);
            let mut fallback_rng = StdRng::seed_from_u64(83);
            let normal = encrypt(&pk, bit, &mut normal_rng).unwrap();
            let fallback = plan.encrypt(bit, &payload, &mut fallback_rng).unwrap();
            assert_eq!(
                serialize(&normal, &p).unwrap(),
                serialize(&fallback, &p).unwrap()
            );
            assert_eq!(next_words(&mut normal_rng), next_words(&mut fallback_rng));
        }
    }
}

#[test]
fn setup_uses_unconditioned_uniform_matrix_and_binary_key() {
    let p = Parameters::new(6, 28, 257, 0.25, 2, 2).unwrap();
    let mut rng = StdRng::seed_from_u64(241);
    let mut control = StdRng::seed_from_u64(241);
    let kp = keygen(p, &mut rng).unwrap();
    let matrix: Vec<u32> = (0..p.n() * p.m())
        .map(|_| control.random_range(0..p.q()))
        .collect();
    let secret: Vec<u8> = (0..p.m()).map(|_| control.random_range(0..2u8)).collect();
    assert_eq!(kp.public.matrix(), matrix);
    assert_eq!(kp.secret, secret);
    let vector: Vec<u32> = matrix
        .chunks(p.m())
        .map(|row| {
            row.iter()
                .zip(&secret)
                .map(|(a, b)| *a as u64 * *b as u64)
                .sum::<u64>() as u32
                % p.q()
        })
        .collect();
    assert_eq!(kp.public.vector(), vector);
    assert_eq!(next_words(&mut rng), next_words(&mut control));
}

#[test]
fn invalid_parameters_keys_and_coins_are_rejected() {
    for args in [
        (1, 28, 257, 0.25, 1, 1),
        (6, 0, 257, 0.25, 2, 2),
        (6, 28, 9, 0.25, 2, 2),
        (6, 28, 2, 0.25, 2, 2),
        (6, 28, 257, 0.0, 2, 2),
        (6, 28, 257, f64::NAN, 2, 2),
        (6, 28, 257, f64::INFINITY, 2, 2),
        (6, 28, 257, 0.25, 0, 2),
        (6, 28, 257, 0.25, 6, 2),
        (6, 28, 257, 0.25, 2, 0),
        (6, 28, 257, 0.25, 2, 5),
    ] {
        assert!(Parameters::new(args.0, args.1, args.2, args.3, args.4, args.5).is_err());
    }
    let f = fixture();
    let pk = public_key(&f);
    let p = pk.params();
    assert!(PublicKey::new(*p, vec![0; p.n() * p.m() - 1], vec![0; p.n()]).is_err());
    assert!(PublicKey::new(*p, vec![p.q(); p.n() * p.m()], vec![0; p.n()]).is_err());
    assert!(PublicKey::new(*p, vec![0; p.n() * p.m()], vec![0; p.n() - 1]).is_err());
    assert!(native_from_coins(&pk, 2, &[0; 6], &[0; 28], 0).is_err());
    assert!(native_from_coins(&pk, 0, &[0; 5], &[0; 28], 0).is_err());
    assert!(native_from_coins(&pk, 0, &[0; 6], &[0; 27], 0).is_err());
    assert!(native_from_coins(&pk, 0, &[257; 6], &[0; 28], 0).is_err());
    for key in [
        UserKey::Expanded(vec![vec![0; 28]]),
        UserKey::Expanded(vec![vec![0; 27]; 2]),
        UserKey::Expanded(vec![vec![2; 28]; 2]),
    ] {
        assert!(user_blocks(&key, p).is_err());
    }
}

#[test]
fn invalid_calls_do_not_consume_randomness() {
    let f = fixture();
    let pk = public_key(&f);
    let dir = directory(&f, "expanded");
    for ids in [vec![], vec![0, 0], vec![3], vec![0, 1, 2]] {
        assert!(prepare_hidden(&pk, &dir, &ids).is_err());
    }
    let mut bad_dir = dir.clone();
    bad_dir[0][0].pop();
    assert!(prepare_hidden(&pk, &bad_dir, &[0]).is_err());
    let mut bad_dir = dir.clone();
    bad_dir[0].pop();
    assert!(prepare_hidden(&pk, &bad_dir, &[0]).is_err());
    let plan = prepare_hidden(&pk, &dir, &[2, 0]).unwrap();
    assert_eq!(plan.audience(), [0, 2]);
    for payload in [vec![], vec![0], vec![0, 1, 0], vec![0, 2]] {
        let mut rng = StdRng::seed_from_u64(3);
        let mut control = StdRng::seed_from_u64(3);
        assert!(plan.encrypt(0, &payload, &mut rng).is_err());
        assert_eq!(next_words(&mut rng), next_words(&mut control));
    }
    let mut rng = StdRng::seed_from_u64(3);
    let mut control = StdRng::seed_from_u64(3);
    assert!(encrypt(&pk, 2, &mut rng).is_err());
    assert_eq!(next_words(&mut rng), next_words(&mut control));
    let mut rng = StdRng::seed_from_u64(3);
    let mut control = StdRng::seed_from_u64(3);
    assert!(plan.fiber().sample(&[0], &mut rng).is_err());
    assert_eq!(next_words(&mut rng), next_words(&mut control));
}

#[test]
fn codec_rejects_wrong_lengths_and_noncanonical_fields() {
    let p = Parameters::new(6, 28, 257, 0.25, 2, 2).unwrap();
    let ct = Ciphertext::new(vec![0; 28], 1, &p).unwrap();
    let bytes = serialize(&ct, &p).unwrap();
    assert_eq!(bytes.len(), 58);
    for bad in [
        vec![],
        bytes[..57].to_vec(),
        [bytes.clone(), vec![0]].concat(),
    ] {
        assert!(deserialize(&bad, &p).is_err());
    }
    for i in 0..29 {
        let mut bad = bytes.clone();
        bad[2 * i..2 * i + 2].copy_from_slice(&257u16.to_le_bytes());
        assert!(deserialize(&bad, &p).is_err());
    }
    assert!(Ciphertext::new(vec![0; 27], 0, &p).is_err());
    assert!(Ciphertext::new(vec![0; 29], 0, &p).is_err());
    assert!(Ciphertext::new(vec![0; 28], 257, &p).is_err());
    assert!(Ciphertext::new(vec![257; 28], 0, &p).is_err());
    let explicit = Ciphertext::new((0..28).collect(), 256, &p).unwrap();
    assert_eq!(&serialize(&explicit, &p).unwrap()[56..], &[0, 1]);
    let current = Parameters::new(1152, 30357, 67537879, 5913.748050094796, 1120, 2).unwrap();
    assert_eq!(current.field_bytes(), 4);
    assert_eq!(current.ciphertext_bytes(), 121432);
}

#[test]
fn round_bit_matches_independent_circular_distances() {
    for q in [3, 5, 7, 11, 17, 257] {
        for z in 0..q {
            let distance = |encoding: u32| {
                (-1..=1)
                    .map(|lift: i64| (z as i64 - encoding as i64 - lift * q as i64).abs())
                    .min()
                    .unwrap()
            };
            assert_eq!(
                round_bit(z, q).unwrap(),
                u8::from(distance(q / 2) < distance(0))
            );
        }
    }
}

#[test]
fn rounded_noise_has_python_ties_and_rejects_unrepresentable_values() {
    for (x, expected) in [
        (-3.5, -4),
        (-2.5, -2),
        (-1.5, -2),
        (-0.5, 0),
        (0.5, 0),
        (1.5, 2),
        (2.5, 2),
        (3.5, 4),
    ] {
        assert_eq!(round_noise(x).unwrap(), expected);
    }
    for bad in [
        f64::NAN,
        f64::INFINITY,
        f64::NEG_INFINITY,
        2f64.powi(63),
        -2f64.powi(64),
    ] {
        assert!(round_noise(bad).is_err());
    }
}

#[test]
fn large_modulus_dot_products_and_signed_errors_do_not_overflow() {
    let q = 4_294_967_291;
    let p = Parameters::new(4, 2, q, 0.25, 2, 1).unwrap();
    let pk = PublicKey::new(p, vec![q - 1; 8], vec![q - 2; 4]).unwrap();
    let ct = native_from_coins(&pk, 1, &[q - 1; 4], &[i64::MIN, i64::MAX], i64::MIN).unwrap();
    // The unreduced dot product is 73_786_976_088_679_776_400 (> u64::MAX).
    // Values below were checked by Python's arbitrary-precision arithmetic.
    assert_eq!(ct.header(), [2_147_483_637, 2_147_483_661]);
    assert_eq!(ct.scalar(), 4_294_967_286);
    assert_eq!(serialize(&ct, &p).unwrap().len(), 12);
    let decoded = deserialize(&serialize(&ct, &p).unwrap(), &p).unwrap();
    assert_eq!(decoded.header(), ct.header());
    assert_eq!(decoded.scalar(), ct.scalar());
    assert_eq!(round_noise(-2f64.powi(63)).unwrap(), i64::MIN);
    let below_limit = f64::from_bits(2f64.powi(63).to_bits() - 1);
    assert_eq!(round_noise(below_limit).unwrap(), 9_223_372_036_854_774_784);
    assert!(round_noise(f64::from_bits((-2f64.powi(63)).to_bits() + 1)).is_err());
}
