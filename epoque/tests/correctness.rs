//! End-to-end correctness of Epoque's trapdoor setup with the anamorphic broadcast
//! of Construction 1 embedded through A0.
//!
//! Ordinary Epoque: every message encrypted to an identity decrypts to itself with
//! the master shortcut key and with the key extracted for that identity; the small
//! profile tries all 2^J ordinary messages. Two identities take turns: the unit
//! identity (H = I) and a uniformly random one. Anamorphic broadcast: aKGen for
//! N = 2|S| users, of which the even-numbered ones form S. For several group shapes
//! and every hidden message of up to 4 bits, every recipient in S recovers the
//! intended hidden message holding only its own key ask_i, the same ciphertext
//! decrypts to the ordinary message with both ordinary keys, and the users outside
//! S (keys from the same aKGen) agree with the hidden bits only by chance. Each
//! check also records the largest decoding noise relative to the margin q/4.
mod common;

use epoque_shortcut::trapdoor::{
    IdentityKey, TrapdoorProfile, decrypt_identity, identity_residual, setup_trapdoor,
};
use epoque_shortcut::*;
use rand::{RngExt, SeedableRng, rngs::StdRng};

struct Instance {
    name: String,
    public: PublicKey,
    shortcut: ShortcutKey,
    /// The unit identity and a random one, each with its extracted key.
    identities: Vec<(Identity, IdentityKey)>,
}

/// The trapdoor setup and two identity keys. T_A0 is dropped after extraction.
fn trapdoor(tp: TrapdoorProfile, seed: u64) -> (Instance, StdRng) {
    let mut rng = StdRng::seed_from_u64(seed);
    let s = setup_trapdoor(&tp, &mut rng, |_, _| {}).unwrap();
    let pk = &s.setup.public;
    let identities = [pk.unit_identity(), common::random_identity(pk, &mut rng)]
        .into_iter()
        .map(|id| {
            let key = s.master.extract(pk, &id, &mut rng).unwrap();
            (id, key)
        })
        .collect();
    let instance = Instance {
        name: format!("trapdoor n{}", tp.profile().n()),
        public: s.setup.public,
        shortcut: s.setup.shortcut,
        identities,
    };
    (instance, rng)
}

/// The decoding noise: the residual minus bit * floor(q/2), centered in (-q/2, q/2].
fn noise(z: u64, bit: u8, q: u64) -> u64 {
    let x = (i128::from(z) - i128::from(bit) * i128::from(q / 2)).rem_euclid(i128::from(q));
    let x = if x > i128::from(q / 2) {
        i128::from(q) - x
    } else {
        x
    };
    x as u64
}

/// Largest decoding noise seen, as a fraction of the margin q/4.
#[derive(Default)]
struct Margins {
    ordinary: u64,
    identity: u64,
    hidden: u64,
    decoded_bits: usize,
}

impl Margins {
    /// Both ordinary keys decrypt `ct`, encrypted to identity number `which`.
    fn ordinary(&mut self, inst: &Instance, which: usize, ct: &Ciphertext, message: &[u8]) {
        let q = inst.public.profile().q();
        assert_eq!(
            decrypt_ordinary(&inst.shortcut, ct).unwrap(),
            message,
            "{}: shortcut-key decryption",
            inst.name
        );
        for (z, &bit) in ordinary_residual(&inst.shortcut, ct)
            .unwrap()
            .into_iter()
            .zip(message)
        {
            self.ordinary = self.ordinary.max(noise(z, bit, q));
        }
        let key = &inst.identities[which].1;
        assert_eq!(
            decrypt_identity(key, ct).unwrap(),
            message,
            "{}: identity-key decryption",
            inst.name
        );
        for (z, &bit) in identity_residual(key, ct).unwrap().into_iter().zip(message) {
            self.identity = self.identity.max(noise(z, bit, q));
        }
        self.decoded_bits += message.len();
    }

    fn report(&self, inst: &Instance) {
        let q = inst.public.profile().q();
        let quarter = (q / 4) as f64;
        eprintln!(
            "{}: {} ordinary bits decoded; largest noise / (q/4): shortcut {:.2e}, identity {:.2e}, hidden {:.2e}",
            inst.name,
            self.decoded_bits,
            self.ordinary as f64 / quarter,
            self.identity as f64 / quarter,
            self.hidden as f64 / quarter
        );
        assert!(self.ordinary < q / 4);
        assert!(self.identity < q / 4);
        assert!(self.hidden < q / 4);
    }
}

fn messages(bits: usize) -> Vec<Vec<u8>> {
    (0..1usize << bits)
        .map(|v| (0..bits).map(|i| ((v >> i) & 1) as u8).collect())
        .collect()
}

/// Ordinary encryption of `count` messages: all 2^J when `count` is None, else the
/// all-zero, all-one and alternating messages followed by random ones. The
/// identities take turns.
fn check_ordinary(inst: &Instance, rng: &mut StdRng, count: Option<usize>, margins: &mut Margins) {
    let jb = inst.public.profile().ordinary_bits();
    let list: Vec<Vec<u8>> = match count {
        None => messages(jb),
        Some(count) => {
            let mut list = vec![
                vec![0; jb],
                vec![1; jb],
                (0..jb).map(|i| (i % 2) as u8).collect(),
            ];
            while list.len() < count {
                list.push((0..jb).map(|_| rng.random_range(0..2)).collect());
            }
            list
        }
    };
    for (index, message) in list.iter().enumerate() {
        let which = index % inst.identities.len();
        let ct = encrypt_ordinary(&inst.public, &inst.identities[which].0, message, rng).unwrap();
        margins.ordinary(inst, which, &ct, message);
    }
}

/// For each shape (|S|, L): every hidden message, `repeats` times, with fresh
/// ordinary messages. On each anamorphic ciphertext the ordinary message must
/// decrypt with both ordinary keys, and every recipient in S must decode the hidden
/// message with its own key. The users outside S must agree with it only about half
/// of the time.
fn check_broadcast(
    inst: &Instance,
    rng: &mut StdRng,
    shapes: &[(usize, usize)],
    repeats: usize,
    margins: &mut Margins,
) {
    let p = inst.public.profile();
    let (q, jb) = (p.q(), p.ordinary_bits());
    let (mut agree, mut total, mut sent) = (0usize, 0usize, 0usize);
    for &(size, bits) in shapes {
        let (users, group) = common::users_and_group(size);
        let (apk, keys) = covert_keygen(&inst.public, users, bits, rng).unwrap();
        let plan = prepare_hidden(&inst.public, &apk, &group).unwrap();
        assert!(
            plan.sampler().full_rank(),
            "{}: H_S rank deficient for {size}x{bits}",
            inst.name
        );
        for _ in 0..repeats {
            for hidden in messages(bits) {
                let which = sent % inst.identities.len();
                sent += 1;
                let ordinary: Vec<u8> = (0..jb).map(|_| rng.random_range(0..2)).collect();
                let ct = plan
                    .encrypt(&inst.identities[which].0, &ordinary, &hidden, rng)
                    .unwrap();
                margins.ordinary(inst, which, &ct, &ordinary);
                common::assert_every_recipient_decrypts(&keys, &group, &ct, &hidden, &inst.name);
                for (i, key) in keys.iter().enumerate() {
                    if group.contains(&i) {
                        let residual = covert_residual(key, &ct).unwrap();
                        for (z, &bit) in residual.into_iter().zip(&hidden) {
                            margins.hidden = margins.hidden.max(noise(z, bit, q));
                        }
                    } else {
                        let decoded = decrypt_covert(key, &ct).unwrap();
                        agree += decoded.iter().zip(&hidden).filter(|(a, b)| a == b).count();
                        total += bits;
                    }
                }
            }
        }
    }
    let rate = agree as f64 / total as f64;
    eprintln!(
        "{}: users outside S agree with the hidden bits at rate {rate:.3} over {total} bits",
        inst.name
    );
    assert!(
        (rate - 0.5).abs() < 6.0 * 0.5 / (total as f64).sqrt(),
        "{}: outsider rate {rate}",
        inst.name
    );
}

const SHAPES: [(usize, usize); 6] = [(1, 1), (1, 4), (2, 4), (4, 2), (8, 1), (2, 3)];

#[test]
fn trapdoor_small_profile_is_correct_in_both_modes() {
    let (inst, mut rng) = trapdoor(TrapdoorProfile::small(), 2);
    let mut margins = Margins::default();
    check_ordinary(&inst, &mut rng, None, &mut margins);
    check_broadcast(&inst, &mut rng, &SHAPES, 3, &mut margins);
    margins.report(&inst);
}

#[test]
#[ignore = "full profile (n = 1360): about 2 GB and minutes; run with --release -- --ignored"]
fn trapdoor_full_profile_is_correct_in_both_modes() {
    let (inst, mut rng) = trapdoor(TrapdoorProfile::full(), 5);
    let mut margins = Margins::default();
    check_ordinary(&inst, &mut rng, Some(4), &mut margins);
    check_broadcast(&inst, &mut rng, &[(4, 2), (8, 1), (2, 4)], 1, &mut margins);
    margins.report(&inst);
}
