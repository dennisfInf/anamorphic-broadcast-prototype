//! Correctness of Construction 1 through the library's own aEnc
//! (`HiddenPlan::encrypt`): encrypt anamorphic messages to a subset S of the
//! users and check that every member of S decrypts exactly that message, and
//! that the ordinary receiver still decrypts its bit (Definition 6).
//!
//! Unlike tests/paper_conformance.rs, nothing is rebuilt by hand here: s comes
//! from `HiddenPlan::sample_secret` and the errors from the noise table, as in
//! real use.
use basic_anamorphic::*;
use rand::{RngExt, SeedableRng, rngs::StdRng};

/// Every message in {0,1}^bits: bit a of the number v is am_a.
fn all_messages(bits: usize) -> Vec<Vec<u8>> {
    (0..1usize << bits)
        .map(|v| (0..bits).map(|a| ((v >> a) & 1) as u8).collect())
        .collect()
}

/// KGen, aKGen, aEnc and aDec: encrypts each message in `messages` to `group`
/// (with a random ordinary bit) and checks every member of the group, for
/// expanded keys and for seed keys.
fn check_members_decrypt(
    params: Parameters,
    population: usize,
    group: &[usize],
    messages: &[Vec<u8>],
    seed: u64,
) {
    // Public seed: reproducible, for tests only.
    let mut rng = StdRng::seed_from_u64(seed);
    let q = params.q();
    // KGen: A uniform, k binary, b = A k.
    let ordinary = keygen(params, &mut rng).unwrap();
    let pk = &ordinary.public;
    for compressed in [false, true] {
        // aKGen: L binary keys k_{i,a} per user and b_{i,a} = A k_{i,a}.
        let keys = anamorphic_keygen(pk, population, compressed, &mut rng).unwrap();
        // aEnc steps 1-4, once per group: sort S, build H_S, eliminate.
        let plan = prepare_hidden(pk, &keys.directory, group).unwrap();
        // With deficient rank aEnc falls back to Enc and nobody could decrypt.
        assert!(
            plan.fiber().full_rank(),
            "H_S is rank deficient for group {group:?}"
        );
        for message in messages {
            let bit: u8 = rng.random_range(0..2);
            // aEnc steps 5-6: s uniform with H_S^T s = Delta * am, then
            // r = A^T s + e and c = b^T s + y + Delta * bit.
            let ct = plan.encrypt(bit, message, &mut rng).unwrap();
            // The ordinary receiver still gets its bit.
            assert_eq!(ordinary_decrypt(&ct, &ordinary.secret, q).unwrap(), bit);
            // aDec(ask_i, C) = am for every member i of S.
            for &i in group {
                let decrypted = hidden_decrypt(&ct, &keys.user_keys[i], &params).unwrap();
                assert_eq!(
                    &decrypted, message,
                    "member {i} of {group:?} (seed keys: {compressed}) got a wrong message"
                );
            }
        }
    }
}

#[test]
fn every_member_decrypts_the_hidden_message_small_profile() {
    // n - n_min = 32 constraints, so with L = 4 bits a group has at most 8 users.
    let (n, m, q, n_min, bits) = (96, 2900, 1_000_003, 64, 4);
    let params =
        Parameters::with_noise_variance(n, m, q, (n * (m + 1)) as u64, n_min, bits).unwrap();
    // All 16 messages of 4 bits.
    let messages = all_messages(bits);
    // Three of 10 users, listed out of order on purpose.
    check_members_decrypt(params, 10, &[7, 2, 5], &messages, 1);
    // A full group: 8 users x 4 bits = 32 = n - n_min constraints.
    check_members_decrypt(params, 10, &[9, 0, 4, 6, 1, 8, 3, 7], &messages, 2);
}

#[test]
#[cfg_attr(debug_assertions, ignore = "production profile: run with --release")]
fn every_member_decrypts_the_hidden_message_production_profile() {
    let (n, m, q, n_min, bits) = (1152, 30357, 67_537_879, 1120, 8);
    let params =
        Parameters::with_noise_variance(n, m, q, (n * (m + 1)) as u64, n_min, bits).unwrap();
    // All zeros, all ones and six random messages.
    let mut rng = StdRng::seed_from_u64(3);
    let mut messages = vec![vec![0u8; bits], vec![1u8; bits]];
    for _ in 0..6 {
        messages.push((0..bits).map(|_| rng.random_range(0..2u8)).collect());
    }
    // 4 users x 8 bits = 32 = n - n_min constraints, the largest group for L = 8.
    check_members_decrypt(params, 8, &[6, 1, 4, 3], &messages, 4);
}