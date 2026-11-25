//! Results and coin use do not depend on the number of worker threads
//! (`arithmetic::threads`, the rayon pool: RAYON_NUM_THREADS).
//!
//! The profile is large enough that every parallel step of encryption is split:
//! n m = 2^22 entries per matrix, 2048 sign groups per column block of R and
//! m + J normals. It has no security; the public key is uniform, so no trapdoor
//! setup is needed. Ordinary decryption runs the products checked in the unit
//! tests of `arithmetic`.
use epoque_shortcut::arithmetic::{sign_product, with_threads};
use epoque_shortcut::sampling::ErrorSampler;
use epoque_shortcut::*;
use rand::{Rng, RngExt, SeedableRng, rngs::StdRng};

fn profile() -> Profile {
    Profile::new(256, 1 << 14, 220_301_035_189, 64, 16, 112_566_231).unwrap()
}

/// The errors with the earlier one-by-one sampling: y, then x, then R block by block.
fn one_by_one(p: Profile, rng: &mut StdRng) -> ErrorCoins {
    let mut sampler = ErrorSampler::new(p.q(), p.denominator()).unwrap();
    let y: Vec<_> = (0..p.m()).map(|_| sampler.sample(rng).unwrap()).collect();
    let x: Vec<_> = (0..p.ordinary_bits())
        .map(|_| sampler.sample(rng).unwrap())
        .collect();
    let correlated = with_threads(1, || sign_product(&y, p.m(), 2048, rng)).unwrap();
    ErrorCoins { y, x, correlated }
}

#[test]
fn errors_and_ciphertexts_do_not_depend_on_the_thread_count() {
    let p = profile();
    let (n, m, q, jb) = (p.n(), p.m(), p.q(), p.ordinary_bits());
    let mut rng = StdRng::seed_from_u64(0x7d);
    let mut matrix = |len: usize| -> Vec<u64> { (0..len).map(|_| rng.random_range(0..q)).collect() };
    let pk = PublicKey::new(p, matrix(n * m), matrix(n * m), matrix(n * m), matrix(n * jb)).unwrap();
    let id = pk.unit_identity();

    // The same coins as one-by-one sampling, in the same order, for every thread count.
    let mut reference_rng = StdRng::seed_from_u64(1);
    let reference = one_by_one(p, &mut reference_rng);
    let next = reference_rng.next_u64();
    for threads in [1, 2, 5] {
        let mut rng = StdRng::seed_from_u64(1);
        let errors = with_threads(threads, || sample_errors(p, &mut rng)).unwrap();
        assert_eq!(errors.y, reference.y, "y, {threads} threads");
        assert_eq!(errors.x, reference.x, "x, {threads} threads");
        assert_eq!(errors.correlated, reference.correlated, "R^T y, {threads} threads");
        assert_eq!(rng.next_u64(), next, "coins used, {threads} threads");
    }

    // Ordinary and anamorphic ciphertexts (4 recipients of 2 bits), hidden decryption.
    let ordinary: Vec<u8> = (0..jb).map(|i| (i % 3 == 0) as u8).collect();
    let hidden = [1u8, 0];
    let mut key_rng = StdRng::seed_from_u64(3);
    let keys: Vec<_> = (0..4)
        .map(|_| covert_secret_keygen(p, 2, &mut key_rng).unwrap())
        .collect();
    let mut apk = CovertPublicKey::with_capacity(&pk, 2, 4).unwrap();
    apk.append(&pk, &keys).unwrap();
    let plan = prepare_hidden(&pk, &apk, &[0, 1, 2, 3]).unwrap();
    let run = |threads| {
        with_threads(threads, || {
            let mut rng = StdRng::seed_from_u64(2);
            let plain = encrypt_ordinary(&pk, &id, &ordinary, &mut rng).unwrap();
            let covert = plan.encrypt(&id, &ordinary, &hidden, &mut rng).unwrap();
            let received: Vec<_> = keys
                .iter()
                .map(|key| decrypt_covert(key, &covert).unwrap())
                .collect();
            (plain, covert, received, rng.next_u64())
        })
    };
    let one = run(1);
    assert!(one.2.iter().all(|bits| bits == &hidden));
    for threads in [2, 5] {
        assert!(run(threads) == one, "{threads} threads");
    }
}
