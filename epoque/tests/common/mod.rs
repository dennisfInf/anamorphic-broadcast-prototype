//! Helpers shared by the integration tests: random identities, H(id) rebuilt from
//! the public modulus f without the library's code, and the per-recipient check of
//! Construction 1's aDec. Not every test crate uses every helper.
#![allow(dead_code)]

use epoque_shortcut::{Ciphertext, CovertSecretKey, Identity, PublicKey, decrypt_covert};
use rand::{Rng, RngExt};

/// A uniformly random identity id in Z_q^n.
pub fn random_identity<R: Rng + ?Sized>(pk: &PublicKey, rng: &mut R) -> Identity {
    let p = pk.profile();
    let coordinates: Vec<u64> = (0..p.n()).map(|_| rng.random_range(0..p.q())).collect();
    pk.identity(&coordinates).unwrap()
}

/// rot_f(g) of Agrawal, Boneh and Boyen (Epoque eq. 19), built here with plain u128
/// arithmetic: row i holds the coefficients of X^i g(X) mod f(X), for the monic
/// f(X) = X^n + c_(n-1) X^(n-1) + ... + c_0 with `modulus` = (c_0, ..., c_(n-1)).
/// Row-major n by n.
pub fn rot_f(modulus: &[u64], g: &[u64], q: u64) -> Vec<u64> {
    let n = modulus.len();
    assert_eq!(g.len(), n, "g has n coefficients");
    let q128 = u128::from(q);
    let mut out = Vec::with_capacity(n * n);
    let mut row = g.to_vec();
    for _ in 0..n {
        out.extend_from_slice(&row);
        // X * row: the coefficient of X^n folds back as -top (c_0 + ... + c_(n-1) X^(n-1)).
        let top = u128::from(row[n - 1]);
        let next: Vec<u64> = (0..n)
            .map(|j| {
                let shifted = if j == 0 { 0 } else { u128::from(row[j - 1]) };
                let fold = top * u128::from(modulus[j]) % q128;
                ((shifted + q128 - fold) % q128) as u64
            })
            .collect();
        row = next;
    }
    out
}

/// H(id) for an identity of `pk`, rebuilt from the public modulus f.
pub fn identity_matrix(pk: &PublicKey, id: &Identity) -> Vec<u64> {
    rot_f(
        pk.identity_encoding().modulus(),
        id.vector(),
        pk.profile().q(),
    )
}

/// Every recipient i in `group` recovers `hidden` from `ct` holding only its own
/// key ask_i (aDec of Construction 1).
pub fn assert_every_recipient_decrypts(
    keys: &[CovertSecretKey],
    group: &[usize],
    ct: &Ciphertext,
    hidden: &[u8],
    context: &str,
) {
    assert!(!group.is_empty(), "{context}: empty group");
    for &i in group {
        assert_eq!(
            decrypt_covert(&keys[i], ct).unwrap(),
            hidden,
            "{context}: recipient {i} of {group:?}"
        );
    }
}

/// The users of aKGen and the group S for a shape (|S|, L): N = 2|S| users, of
/// which the even-numbered ones are the recipients and the odd-numbered ones are
/// outside S.
pub fn users_and_group(size: usize) -> (usize, Vec<usize>) {
    (2 * size, (0..size).map(|j| 2 * j).collect())
}
