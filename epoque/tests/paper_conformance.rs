//! Epoque IBE (ePrint 2021/304, Section 5) with the one-ciphertext broadcast of
//! Construction 1 embedded through A0 (Construction 3), checked equation by
//! equation with arbitrary integers on the small trapdoor profile, for a random
//! identity id whose H(id) is rebuilt from the public modulus f:
//!   KeyGen: B D2 = 0 and A0 D1 + A1 D2 = U, so F_id D = U for every id (eq. 29)
//!   Extr:   F_id E = U with F_id = [A0 | A1 + H(id) B] (eq. 23)
//!   Enc:    c0 = U^T s + x + floor(q/2) mu,  c1 = F_id^T s + [y; R^T y]  (eqs. 25, 26)
//!   aKGen:  binary k̂_(i,a) and b̂_(i,a) = A0 k̂_(i,a)
//!   aEnc:   column (j-1)L + a of H_S is a1 - b̂_(i_j,a), and H_S^T s = floor(q/2) (am, .., am)
//!   Dec:    c0 - D^T c1 = floor(q/2) mu + x - D1^T y - D2^T R^T y  (s cancels, eq. 27),
//!           and the same with the identity key E = (E1; E2)
//!   aDec:   c1[1] - k̂_(i,a)^T c1[1..m] = floor(q/2) am_a + y[1] - k̂_(i,a)^T y
mod common;

use epoque_shortcut::trapdoor::{
    TrapdoorProfile, decrypt_identity, identity_residual, setup_trapdoor,
};
use epoque_shortcut::*;
use rand::{SeedableRng, rngs::StdRng};
use rug::Integer;

fn m_of(x: &Integer, q: u64) -> u64 {
    integer_mod(x, q)
}

#[test]
fn trapdoor_setup_encryption_and_every_decoder_satisfy_the_paper_equations() {
    let tp = TrapdoorProfile::small();
    let p = tp.profile();
    let (n, m, q, jb) = (p.n(), p.m(), p.q(), p.ordinary_bits());
    let mut rng = StdRng::seed_from_u64(0xe90e);
    let setup = setup_trapdoor(&tp, &mut rng, |_, _| {}).unwrap();
    let pk = &setup.setup.public;
    let (a0, a1, b, u) = (pk.a0(), pk.a1(), pk.b(), pk.u());
    let id = common::random_identity(pk, &mut rng);
    let h = common::identity_matrix(pk, &id);
    // The right block of F_id, A1 + H(id) B, over the integers.
    let right: Vec<Integer> = (0..n * m)
        .map(|e| {
            let (r, k) = (e / m, e % m);
            let mut x = Integer::from(a1[e]);
            for t in 0..n {
                x += Integer::from(h[r * n + t]) * b[t * m + k];
            }
            x
        })
        .collect();
    let d1t = setup.setup.shortcut.d1_transposed();
    let d2t = setup.setup.shortcut.d2_transposed();
    let id_key = setup.master.extract(pk, &id, &mut rng).unwrap();
    let (e1t, e2t) = (id_key.e1_transposed(), id_key.e2_transposed());
    // Key equations on every row and column.
    for row in 0..n {
        for col in 0..jb {
            let (mut kernel, mut shortcut) = (Integer::new(), Integer::new());
            let (mut fid_d, mut fid_e) = (Integer::new(), Integer::new());
            for k in 0..m {
                let e = row * m + k;
                let (d1, d2) = (d1t[col * m + k], d2t[col * m + k]);
                let (e1, e2) = (e1t[col * m + k], e2t[col * m + k]);
                kernel += Integer::from(b[e]) * d2;
                shortcut += Integer::from(a0[e]) * d1;
                shortcut += Integer::from(a1[e]) * d2;
                fid_d += Integer::from(a0[e]) * d1;
                fid_d += right[e].clone() * d2;
                fid_e += Integer::from(a0[e]) * e1;
                fid_e += right[e].clone() * e2;
            }
            let target = u[row * jb + col];
            assert_eq!(m_of(&kernel, q), 0, "B D2 = 0");
            assert_eq!(m_of(&shortcut, q), target, "A0 D1 + A1 D2 = U");
            assert_eq!(m_of(&fid_d, q), target, "F_id D = U (eq. 29)");
            assert_eq!(m_of(&fid_e, q), target, "F_id E = U (eq. 23)");
        }
    }
    // aKGen for N = 4 users of L = 2 bits; aEnc for S = {0, 1, 3}.
    let (users, bits, group) = (4, 2, [0usize, 1, 3]);
    let (apk, keys) = covert_keygen(pk, users, bits, &mut rng).unwrap();
    for (i, key) in keys.iter().enumerate() {
        let kt = key.transposed_keys();
        assert!(kt.iter().all(|&x| x == 0 || x == 1), "binary k̂");
        for a in 0..bits {
            for row in 0..n {
                let mut image = Integer::new();
                for k in 0..m {
                    image += Integer::from(a0[row * m + k]) * kt[a * m + k];
                }
                assert_eq!(apk.vector(i, a)[row], m_of(&image, q), "b̂ = A0 k̂");
            }
        }
    }
    let plan = prepare_hidden(pk, &apk, &group).unwrap();
    assert!(plan.sampler().full_rank());
    for (j, &i) in group.iter().enumerate() {
        let kt = keys[i].transposed_keys();
        for a in 0..bits {
            let column = &plan.constraints()[j * bits + a];
            for row in 0..n {
                let mut value = Integer::from(a0[row * m]);
                for k in 0..m {
                    value -= Integer::from(a0[row * m + k]) * kt[a * m + k];
                }
                assert_eq!(column[row], m_of(&value, q), "H_S column = a1 - A0 k̂");
            }
        }
    }
    let delta = q / 2;
    for trial in 0..4u64 {
        let ordinary: Vec<u8> = (0..jb).map(|j| ((j as u64 + trial) % 2) as u8).collect();
        let hidden: Vec<u8> = (0..bits).map(|a| ((trial >> a) & 1) as u8).collect();
        let rhs: Vec<u64> = (0..group.len())
            .flat_map(|_| hidden.iter().map(|&x| u64::from(x) * delta))
            .collect();
        let s = plan.sampler().sample(&rhs, &mut rng).unwrap();
        for (column, &target) in plan.constraints().iter().zip(&rhs) {
            let mut dot = Integer::new();
            for (&x, &y) in column.iter().zip(&s) {
                dot += Integer::from(x) * y;
            }
            assert_eq!(m_of(&dot, q), target, "h^T s = floor(q/2) am_a");
        }
        let errors = sample_errors(p, &mut rng).unwrap();
        let ct = encrypt_with_coins(pk, &id, &ordinary, &s, &errors).unwrap();
        for j in 0..m {
            let (mut v, mut w) = (errors.y[j].clone(), errors.correlated[j].clone());
            for r in 0..n {
                v += Integer::from(a0[r * m + j]) * s[r];
                w += right[r * m + j].clone() * s[r];
            }
            assert_eq!(ct.c1()[j], m_of(&v, q), "top of c1 = A0^T s + y");
            assert_eq!(
                ct.c1()[m + j],
                m_of(&w, q),
                "bottom of c1 = (A1 + H(id) B)^T s + R^T y"
            );
        }
        for j in 0..jb {
            let mut c = errors.x[j].clone();
            c += u64::from(ordinary[j]) * delta;
            for r in 0..n {
                c += Integer::from(u[r * jb + j]) * s[r];
            }
            assert_eq!(ct.c0()[j], m_of(&c, q), "c0 = U^T s + x + floor(q/2) mu");
        }
        // Dec with the master shortcut key and with the identity key: s cancels.
        let decoders = [
            (
                "shortcut",
                ordinary_residual(&setup.setup.shortcut, &ct).unwrap(),
                d1t,
                d2t,
            ),
            (
                "identity",
                identity_residual(&id_key, &ct).unwrap(),
                e1t,
                e2t,
            ),
        ];
        for (name, residual, first, second) in decoders {
            for j in 0..jb {
                let mut z = errors.x[j].clone();
                z += u64::from(ordinary[j]) * delta;
                for k in 0..m {
                    z -= errors.y[k].clone() * first[j * m + k];
                    z -= errors.correlated[k].clone() * second[j * m + k];
                }
                assert_eq!(residual[j], m_of(&z, q), "{name} residual: s cancels");
            }
        }
        assert_eq!(
            decrypt_ordinary(&setup.setup.shortcut, &ct).unwrap(),
            ordinary
        );
        assert_eq!(decrypt_identity(&id_key, &ct).unwrap(), ordinary);
        // aDec for every recipient in S, bit by bit.
        for &i in &group {
            let kt = keys[i].transposed_keys();
            let residual = covert_residual(&keys[i], &ct).unwrap();
            for a in 0..bits {
                let mut e = errors.y[0].clone();
                e += u64::from(hidden[a]) * delta;
                for k in 0..m {
                    e -= errors.y[k].clone() * kt[a * m + k];
                }
                assert_eq!(residual[a], m_of(&e, q), "c1[1] - k̂^T c1[1..m]");
            }
            assert_eq!(decrypt_covert(&keys[i], &ct).unwrap(), hidden);
        }
    }
}
