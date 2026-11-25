//! Independent Python/Arb fixtures exercise equations, bytes and retained bits.
//!
//! The fixture uses the unit identity (H = I), two users with L = 2 bits each and
//! S = {0, 1}: its m by 4 key matrix holds user 0's blocks in columns 0 and 1 and
//! user 1's in columns 2 and 3. Its public key and keys come from explicit coins
//! (they test algebra, not a key generator).
use epoque_shortcut::arithmetic::{AffineSampler, sign_product_from_bytes};
use epoque_shortcut::sampling::ErrorSampler;
use epoque_shortcut::*;
use rand::{Rng, SeedableRng, TryRng, rngs::StdRng};
use rug::Integer;
use serde_json::Value;
use std::convert::Infallible;

const USERS: usize = 2;
const BITS: usize = 2;
const GROUP: [usize; 2] = [0, 1];

fn fixture() -> Value {
    serde_json::from_str(include_str!("fixtures/python.json")).unwrap()
}
fn values(v: &Value) -> Vec<u64> {
    v.as_array()
        .unwrap()
        .iter()
        .map(|x| x.as_u64().unwrap())
        .collect()
}
fn bytes(v: &Value) -> Vec<u8> {
    values(v)
        .into_iter()
        .map(|x| u8::try_from(x).unwrap())
        .collect()
}
fn integers(v: &Value) -> Vec<Integer> {
    v.as_array()
        .unwrap()
        .iter()
        .map(|x| Integer::from_str_radix(x.as_str().unwrap(), 10).unwrap())
        .collect()
}
fn rows(v: &Value) -> Vec<Vec<u64>> {
    v.as_array().unwrap().iter().map(values).collect()
}
fn flat(v: &Value) -> Vec<u64> {
    rows(v).into_iter().flatten().collect()
}
fn small_flat(v: &Value) -> Vec<i32> {
    v.as_array()
        .unwrap()
        .iter()
        .flat_map(|row| row.as_array().unwrap().iter())
        .map(|x| i32::try_from(x.as_i64().unwrap()).unwrap())
        .collect()
}
fn hex(v: &Value) -> Vec<u8> {
    hex::decode(v.as_str().unwrap()).unwrap()
}
fn profile(d: &Value) -> Profile {
    let p = &d["parameters"];
    Profile::new(
        p["n"].as_u64().unwrap() as usize,
        p["m"].as_u64().unwrap() as usize,
        p["q"].as_u64().unwrap(),
        p["J"].as_u64().unwrap() as usize,
        p["g"].as_u64().unwrap() as usize,
        p["C"].as_u64().unwrap(),
    )
    .unwrap()
}
fn public(d: &Value) -> PublicKey {
    let m = &d["matrices"];
    PublicKey::new(
        profile(d),
        flat(&m["A0"]),
        flat(&m["A1"]),
        flat(&m["B"]),
        flat(&m["U"]),
    )
    .unwrap()
}
fn shortcut(d: &Value) -> ShortcutKey {
    ShortcutKey::new(
        profile(d),
        &small_flat(&d["matrices"]["D1"]),
        &small_flat(&d["matrices"]["D2"]),
    )
    .unwrap()
}
/// ask_0 and ask_1: columns a and 2 + a of the fixture's m by 4 key matrix.
fn covert_keys(d: &Value) -> Vec<CovertSecretKey> {
    let p = profile(d);
    let all = small_flat(&d["matrices"]["covert_keys"]);
    let width = USERS * BITS;
    (0..USERS)
        .map(|user| {
            let own: Vec<i32> = (0..p.m() * BITS)
                .map(|e| all[(e / BITS) * width + user * BITS + e % BITS])
                .collect();
            CovertSecretKey::new(p, BITS, &own).unwrap()
        })
        .collect()
}
fn coins(v: &Value) -> ErrorCoins {
    ErrorCoins {
        y: integers(&v["y"]),
        x: integers(&v["x"]),
        correlated: integers(&v["correlated"]),
    }
}

struct WordStream {
    words: Vec<u64>,
    used: usize,
}
impl WordStream {
    fn new(v: &Value) -> Self {
        Self {
            words: v
                .as_array()
                .unwrap()
                .iter()
                .map(|x| x.as_str().unwrap().parse().unwrap())
                .collect(),
            used: 0,
        }
    }
}
impl TryRng for WordStream {
    type Error = Infallible;
    fn try_next_u64(&mut self) -> std::result::Result<u64, Self::Error> {
        let word = *self
            .words
            .get(self.used)
            .expect("fixture exhausted before certified termination");
        self.used += 1;
        Ok(word)
    }
    fn try_next_u32(&mut self) -> std::result::Result<u32, Self::Error> {
        Ok(self.try_next_u64()? as u32)
    }
    fn try_fill_bytes(&mut self, dst: &mut [u8]) -> std::result::Result<(), Self::Error> {
        for chunk in dst.chunks_mut(8) {
            chunk.copy_from_slice(&self.try_next_u64()?.to_le_bytes()[..chunk.len()]);
        }
        Ok(())
    }
}

#[test]
fn python_affine_plans_and_all_payloads_match() {
    let d = fixture();
    let pk = public(&d);
    let keys = covert_keys(&d);
    let apk = CovertPublicKey::from_secret_keys(&pk, &keys).unwrap();
    let plan = prepare_hidden(&pk, &apk, &GROUP).unwrap();
    assert!(plan.sampler().full_rank());
    assert_eq!(plan.constraints(), rows(&d["covert"]["constraints"]));
    assert_eq!(
        plan.sampler().pivots(),
        values(&d["covert"]["pivots"])
            .into_iter()
            .map(|x| x as usize)
            .collect::<Vec<_>>()
    );
    assert_eq!(
        plan.sampler().free(),
        values(&d["covert"]["free"])
            .into_iter()
            .map(|x| x as usize)
            .collect::<Vec<_>>()
    );
    let mut checked = 0;
    for c in d["cases"].as_array().unwrap() {
        if !c["target"].is_null() {
            let secret = plan
                .sampler()
                .from_free(&values(&c["target"]), &values(&c["free_values"]))
                .unwrap();
            assert_eq!(secret, values(&c["secret"]));
            checked += 1;
        }
    }
    assert_eq!(checked, 4);
    let late = &d["late_pivot"];
    let a = AffineSampler::new(rows(&late["constraints"]), 6, 257).unwrap();
    assert_eq!(a.pivots(), [2, 4]);
    assert_eq!(
        a.from_free(&values(&late["target"]), &values(&late["free_values"]))
            .unwrap(),
        values(&late["secret"])
    );
}

#[test]
fn python_ciphertext_residuals_and_wire_format_match_including_huge_errors() {
    let d = fixture();
    let p = profile(&d);
    let pk = public(&d);
    let id = pk.unit_identity();
    let sk = shortcut(&d);
    let keys = covert_keys(&d);
    for c in d["cases"].as_array().unwrap() {
        let ct = encrypt_with_coins(
            &pk,
            &id,
            &bytes(&c["ordinary_bits"]),
            &values(&c["secret"]),
            &coins(&c["errors"]),
        )
        .unwrap();
        assert_eq!(ct.c0(), values(&c["ciphertext"]["c0"]), "{}", c["name"]);
        assert_eq!(ct.c1(), values(&c["ciphertext"]["c1"]), "{}", c["name"]);
        assert_eq!(
            ordinary_residual(&sk, &ct).unwrap(),
            values(&c["ordinary_residual"])
        );
        let residuals: Vec<u64> = keys
            .iter()
            .flat_map(|key| covert_residual(key, &ct).unwrap())
            .collect();
        assert_eq!(residuals, values(&c["covert_residual"]));
        assert_eq!(
            decrypt_ordinary(&sk, &ct).unwrap(),
            bytes(&c["ordinary_decoded"])
        );
        let expected: Vec<Vec<u8>> = c["covert_decoded"]
            .as_array()
            .unwrap()
            .iter()
            .map(bytes)
            .collect();
        let decoded: Vec<Vec<u8>> = keys
            .iter()
            .map(|key| decrypt_covert(key, &ct).unwrap())
            .collect();
        assert_eq!(decoded, expected);
        let wire = encode(p, &ct).unwrap();
        assert_eq!(wire, hex(&c["codec_hex"]));
        assert_eq!(decode(p, &wire).unwrap(), ct);
    }
}

#[test]
fn python_sign_layout_matches_with_partial_byte_group_and_large_fallback() {
    let d = fixture();
    for c in d["sign_products"].as_array().unwrap() {
        assert_eq!(
            sign_product_from_bytes(
                &integers(&c["y"]),
                &bytes(&c["raw"]),
                c["columns"].as_u64().unwrap() as usize
            )
            .unwrap(),
            integers(&c["expected"])
        );
    }
}

#[test]
fn python_arb_rounded_normals_match_with_retained_prefixes_and_unbounded_outputs() {
    let d = fixture();
    let mut checked = 0;
    let mut beyond_i64 = 0;
    let mut refined = 0;
    for case in d["rounded_normal"].as_array().unwrap() {
        let numerator =
            Integer::from_str_radix(case["variance_numerator"].as_str().unwrap(), 10).unwrap();
        let denominator =
            Integer::from_str_radix(case["variance_denominator"].as_str().unwrap(), 10).unwrap();
        let num_root = numerator.clone().sqrt();
        let den_root = denominator.clone().sqrt();
        // The carrier API takes a rational standard deviation q/d. Nonsquare
        // reference-only variances exercise Python but are outside that API.
        if Integer::from(&num_root * &num_root) != numerator
            || Integer::from(&den_root * &den_root) != denominator
        {
            continue;
        }
        let (Some(num), Some(den)) = (num_root.to_u64(), den_root.to_u64()) else {
            continue;
        };
        let mut sampler = ErrorSampler::new(num, den).unwrap();
        let mut rng = WordStream::new(&case["words"]);
        let expected = Integer::from_str_radix(case["expected"].as_str().unwrap(), 10).unwrap();
        let actual = sampler.sample(&mut rng).unwrap();
        assert_eq!(actual, expected, "normal fixture {case}");
        if expected.to_i64().is_none() {
            beyond_i64 += 1;
        }
        if rng.used > 1 {
            refined += 1;
        }
        checked += 1;
    }
    assert_eq!(checked, 13);
    assert_eq!(beyond_i64, 3);
    assert!(refined >= 4);
}

#[test]
fn sampled_anamorphic_roundtrips_preserve_both_messages_for_every_payload() {
    let d = fixture();
    let p = profile(&d);
    let pk = public(&d);
    let id = pk.unit_identity();
    let sk = shortcut(&d);
    let keys = covert_keys(&d);
    let apk = CovertPublicKey::from_secret_keys(&pk, &keys).unwrap();
    let plan = prepare_hidden(&pk, &apk, &GROUP).unwrap();
    let mut rng = StdRng::seed_from_u64(74012);
    for payload in [[0, 0], [0, 1], [1, 0], [1, 1]] {
        for ordinary in [[0, 1], [1, 0]] {
            let ct = plan.encrypt(&id, &ordinary, &payload, &mut rng).unwrap();
            let received = decode(p, &encode(p, &ct).unwrap()).unwrap();
            assert_eq!(decrypt_ordinary(&sk, &received).unwrap(), ordinary);
            for key in &keys {
                assert_eq!(decrypt_covert(key, &received).unwrap(), payload);
            }
        }
    }
}

#[test]
fn rank_deficiency_uses_exact_ordinary_fallback_and_invalid_requests_consume_no_coins() {
    let d = fixture();
    let p = profile(&d);
    let pk = PublicKey::new(
        p,
        vec![0; p.n() * p.m()],
        vec![0; p.n() * p.m()],
        vec![0; p.n() * p.m()],
        vec![0; p.n() * p.ordinary_bits()],
    )
    .unwrap();
    let id = pk.unit_identity();
    let zero = CovertSecretKey::new(p, BITS, &vec![0; p.m() * BITS]).unwrap();
    let apk = CovertPublicKey::from_secret_keys(&pk, &[zero.clone(), zero]).unwrap();
    let plan = prepare_hidden(&pk, &apk, &GROUP).unwrap();
    assert!(!plan.sampler().full_rank());
    let mut a = StdRng::seed_from_u64(541);
    let mut b = StdRng::seed_from_u64(541);
    assert_eq!(
        plan.encrypt(&id, &[1, 0], &[1, 0], &mut a).unwrap(),
        encrypt_ordinary(&pk, &id, &[1, 0], &mut b).unwrap()
    );
    assert_eq!(a.next_u64(), b.next_u64());
    let mut a = StdRng::seed_from_u64(62);
    let mut b = StdRng::seed_from_u64(62);
    assert!(plan.encrypt(&id, &[1], &[1, 0], &mut a).is_err());
    assert!(plan.encrypt(&id, &[1, 0], &[1], &mut a).is_err());
    assert!(plan.encrypt(&id, &[1, 2], &[1, 0], &mut a).is_err());
    assert!(plan.encrypt(&id, &[1, 0], &[1, 2], &mut a).is_err());
    assert_eq!(a.next_u64(), b.next_u64());
}

#[test]
fn malformed_keys_coins_and_ciphertexts_are_rejected() {
    let d = fixture();
    let p = profile(&d);
    let pk = public(&d);
    let id = pk.unit_identity();
    let c = &d["cases"][0];
    let ct = encrypt_with_coins(
        &pk,
        &id,
        &bytes(&c["ordinary_bits"]),
        &values(&c["secret"]),
        &coins(&c["errors"]),
    )
    .unwrap();
    let wire = encode(p, &ct).unwrap();
    assert!(decode(p, &wire[..wire.len() - 1]).is_err());
    let mut longer = wire.clone();
    longer.push(0);
    assert!(decode(p, &longer).is_err());
    for field in 0..2 * p.m() + p.ordinary_bits() {
        let mut bad = wire.clone();
        let first = field * p.field_bytes();
        bad[first..first + p.field_bytes()]
            .copy_from_slice(&p.q().to_le_bytes()[..p.field_bytes()]);
        assert!(decode(p, &bad).is_err());
    }
    let mut a0 = pk.a0().to_vec();
    a0[0] = p.q();
    assert!(PublicKey::new(p, a0, pk.a1().to_vec(), pk.b().to_vec(), pk.u().to_vec()).is_err());
    // Shortcut keys take any integers, in m by J.
    let d1 = small_flat(&d["matrices"]["D1"]);
    let d2 = small_flat(&d["matrices"]["D2"]);
    assert!(ShortcutKey::new(p, &d1[1..], &d2[1..]).is_err());
    assert!(ShortcutKey::new(p, &d1, &d2[1..]).is_err());
    // Anamorphic keys are binary, with 1 <= L <= g.
    let mut key = vec![0; p.m() * BITS];
    key[3] = 1;
    assert!(CovertSecretKey::new(p, BITS, &key).is_ok());
    key[0] = 2;
    assert!(CovertSecretKey::new(p, BITS, &key).is_err());
    key[0] = -1;
    assert!(CovertSecretKey::new(p, BITS, &key).is_err());
    assert!(CovertSecretKey::new(p, 0, &[]).is_err());
    let g = p.maximum_constraints();
    assert!(CovertSecretKey::new(p, g + 1, &vec![0; p.m() * (g + 1)]).is_err());
    // Identities are n canonical residues.
    assert!(pk.identity(&vec![0; p.n() - 1]).is_err());
    let mut coordinates = vec![0; p.n()];
    coordinates[0] = p.q();
    assert!(pk.identity(&coordinates).is_err());
    // Coins must match the profile.
    let mut s = values(&c["secret"]);
    s[0] = p.q();
    assert!(encrypt_with_coins(&pk, &id, &[0, 1], &s, &coins(&c["errors"])).is_err());
    let mut error = coins(&c["errors"]);
    error.y.pop();
    assert!(encrypt_with_coins(&pk, &id, &[0, 1], &values(&c["secret"]), &error).is_err());
    // Profiles: q prime, odd and above n; n < m; J < m; 1 <= g < n; 65 C fits.
    assert!(Profile::new(2, 3, 2, 1, 1, 1).is_err());
    assert!(Profile::new(2, 3, 9, 1, 1, 1).is_err());
    assert!(Profile::new(3, 3, 17, 1, 1, 1).is_err());
    assert!(Profile::new(2, 3, 17, 1, 2, 1).is_err());
    assert!(Profile::new(2, 3, 17, 1, 1, u64::MAX).is_err());
    assert!(Profile::new(3, 5, 3, 1, 1, 1).is_err());
    assert!(Profile::new(2, 5, 3, 1, 1, 1).is_ok());
}

#[test]
fn ordinary_and_covert_decoders_keep_their_distinct_reference_rules() {
    for q in [3, 5, 7, 11, 17, 257] {
        let p = Profile::new(2, 3, q, 1, 1, 1).unwrap();
        let sk = ShortcutKey::new(p, &[0; 3], &[0; 3]).unwrap();
        let key = CovertSecretKey::new(p, 1, &[0; 3]).unwrap();
        for z in 0..q {
            let ct = Ciphertext::new(p, vec![z], vec![z, 0, 0, 0, 0, 0]).unwrap();
            assert_eq!(
                decrypt_ordinary(&sk, &ct).unwrap(),
                [u8::from(z.abs_diff(q / 2) < q / 4)]
            );
            let dist0 = (z as i128).min((q - z) as i128);
            let delta = (z as i128 - q as i128 / 2).abs();
            let dist1 = delta.min(q as i128 - delta);
            assert_eq!(
                decrypt_covert(&key, &ct).unwrap(),
                [u8::from(dist1 < dist0)]
            );
        }
    }
}
