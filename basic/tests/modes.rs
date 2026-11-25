//! The multi-thread variants of the group setup (`prepare_hidden_with`), the
//! broadcast step of aEnc (`HiddenPlan::sample_secret_with`, `encrypt_with`) and
//! hidden decryption (`hidden_decrypt_with`, `expand_seed_with`) give the same
//! plans, coins, secrets, ciphertexts and bits as `Mode::Single`, for several
//! sizes of rayon's pool. The profile is large enough that every pivot step of
//! the elimination is split: 128 constraints on n = 600 coordinates.
use basic_anamorphic::*;
use rand::{Rng, RngExt, SeedableRng, rngs::StdRng};

fn pool<T: Send>(threads: usize, f: impl FnOnce() -> T + Send) -> T {
    rayon::ThreadPoolBuilder::new()
        .num_threads(threads)
        .build()
        .unwrap()
        .install(f)
}

#[test]
#[cfg_attr(debug_assertions, ignore = "aKGen for 16 users of 8 bits: run with --release")]
fn group_setup_broadcast_step_and_hidden_decryption_agree_between_modes() {
    // 16 users of 8 bits use all n - n_min = 128 constraints.
    let (n, m, q, n_min, bits) = (600, 4000, 67_537_879, 472, 8);
    let params =
        Parameters::with_noise_variance(n, m, q, (n * (m + 1)) as u64, n_min, bits).unwrap();
    let mut rng = StdRng::seed_from_u64(21);
    let ordinary = keygen(params, &mut rng).unwrap();
    let pk = &ordinary.public;
    let keys = anamorphic_keygen(pk, 16, true, &mut rng).unwrap();
    let group: Vec<usize> = (0..16).collect();
    let single = prepare_hidden(pk, &keys.directory, &group).unwrap();
    assert!(single.fiber().full_rank());
    let payload: Vec<u8> = (0..bits).map(|a| (a % 3 == 0) as u8).collect();
    let mut coins = StdRng::seed_from_u64(22);
    let secret = single.sample_secret(&payload, &mut coins).unwrap();
    let after = coins.next_u64();
    let ct = single
        .encrypt_with(1, &payload, &mut StdRng::seed_from_u64(23), Mode::Single)
        .unwrap();
    let expanded: Vec<UserKey> = keys
        .user_keys
        .iter()
        .map(|key| UserKey::Expanded(user_blocks(key, &params).unwrap()))
        .collect();
    for threads in [1, 2, 3, 7] {
        pool(threads, || {
            let multi = prepare_hidden_with(pk, &keys.directory, &group, Mode::Multi).unwrap();
            assert_eq!(multi.fiber().pivots(), single.fiber().pivots());
            assert_eq!(multi.fiber().reduced(), single.fiber().reduced());
            assert_eq!(multi.fiber().transform(), single.fiber().transform());
            let mut coins = StdRng::seed_from_u64(22);
            let multi_secret = multi
                .sample_secret_with(&payload, &mut coins, Mode::Multi)
                .unwrap();
            assert_eq!(multi_secret, secret, "{threads} threads: s");
            assert_eq!(coins.next_u64(), after, "{threads} threads: coins");
            let multi_ct = multi
                .encrypt_with(1, &payload, &mut StdRng::seed_from_u64(23), Mode::Multi)
                .unwrap();
            assert_eq!(multi_ct, ct, "{threads} threads: ciphertext");
            for (i, key) in keys.user_keys.iter().enumerate() {
                let blocks = user_blocks(key, &params).unwrap();
                if let UserKey::Seed(seed) = key {
                    assert_eq!(expand_seed_with(seed, &params, Mode::Multi), blocks);
                }
                for key in [key, &expanded[i]] {
                    assert_eq!(
                        hidden_residuals_with(&ct, key, &params, Mode::Multi).unwrap(),
                        hidden_residuals(&ct, key, &params).unwrap(),
                        "{threads} threads: residuals of user {i}"
                    );
                    assert_eq!(
                        hidden_decrypt_with(&ct, key, &params, Mode::Multi).unwrap(),
                        payload
                    );
                }
            }
        });
    }
    // A rank-deficient group (one user twice is refused, so use a dependent set of
    // keys): the same fallback in both modes.
    let mut directory = keys.directory.clone();
    directory[1] = directory[0].clone();
    let a = prepare_hidden(pk, &directory, &[0, 1]).unwrap();
    let b = pool(4, || prepare_hidden_with(pk, &directory, &[0, 1], Mode::Multi).unwrap());
    assert!(!a.fiber().full_rank());
    assert_eq!(a.fiber().pivots(), b.fiber().pivots());
    let bit = rng.random_range(0..2u8);
    assert_eq!(
        a.encrypt_with(bit, &payload, &mut StdRng::seed_from_u64(24), Mode::Single)
            .unwrap(),
        pool(4, || b
            .encrypt_with(bit, &payload, &mut StdRng::seed_from_u64(24), Mode::Multi)
            .unwrap())
    );
}
