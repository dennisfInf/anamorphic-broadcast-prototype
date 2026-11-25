//! Construction 1 ("One-ciphertext anamorphic broadcast") checked equation by
//! equation against the paper, independently of the Python fixtures:
//!   aKGen:  b_{i,a} = A k_{i,a} with binary k_{i,a}
//!   aEnc:   column (j-1)L+a of H_S is a_1 - b_{i_j,a};  H_S^T s = Delta * am_bar;
//!           r = A^T s + e,  c = b^T s + y + Delta * m;  rank deficiency -> Enc(pk, m)
//!   aDec:   Round(r[1] - k_{i,a}^T r), residual = Delta*am_a + e[1] - k_{i,a}^T e
//!   Dec:    Round(c - k^T r), unaffected by the constraint on s.
use basic_anamorphic::*;
use rand::{RngExt, SeedableRng, rngs::StdRng};

fn modq(x: i128, q: u32) -> u32 {
    x.rem_euclid(i128::from(q)) as u32
}

fn small_params(bits: usize) -> Parameters {
    // q prime, n - n_min = 12 constraints of capacity.
    Parameters::new(24, 160, 12289, 3.0, 12, bits).unwrap()
}

#[test]
fn construction_1_equations_hold_for_every_group_member() {
    let mut rng = StdRng::seed_from_u64(0x5ec7_1011);
    for (bits, group) in [
        (1usize, vec![2usize]),
        (2, vec![0, 3, 1]),
        (3, vec![4, 2, 0, 1]),
    ] {
        let p = small_params(bits);
        let q = p.q();
        let delta = p.delta();
        let ordinary = keygen(p, &mut rng).unwrap();
        let pk = &ordinary.public;
        let a = pk.matrix();
        let (n, m) = (p.n(), p.m());
        for compressed in [false, true] {
            let keys = anamorphic_keygen(pk, 5, compressed, &mut rng).unwrap();
            // aKGen: b_{i,a} = A k_{i,a}, k binary.
            for (i, key) in keys.user_keys.iter().enumerate() {
                let blocks = user_blocks(key, &p).unwrap();
                assert_eq!(blocks.len(), bits);
                for (ab, block) in blocks.iter().enumerate() {
                    assert!(block.iter().all(|&x| x <= 1));
                    for row in 0..n {
                        let expect: u64 = (0..m)
                            .map(|j| u64::from(a[row * m + j]) * u64::from(block[j]))
                            .sum();
                        assert_eq!(keys.directory[i][ab][row], (expect % u64::from(q)) as u32);
                    }
                }
            }
            let plan = prepare_hidden(pk, &keys.directory, &group).unwrap();
            // S is processed in increasing order: column (j-1)L + a  <->  (i_j, a).
            let mut sorted = group.clone();
            sorted.sort_unstable();
            assert_eq!(plan.audience(), sorted.as_slice());
            assert!(plan.fiber().full_rank());
            for (j, &i) in sorted.iter().enumerate() {
                for ab in 0..bits {
                    let column = &plan.fiber().matrix()[j * bits + ab];
                    for row in 0..n {
                        let h = modq(
                            i128::from(a[row * m]) - i128::from(keys.directory[i][ab][row]),
                            q,
                        );
                        assert_eq!(column[row], h, "H_S column (j-1)L+a mismatch");
                    }
                }
            }
            for trial in 0..8 {
                let bit = (trial % 2) as u8;
                let am: Vec<u8> = (0..bits).map(|x| ((trial >> x) & 1) as u8).collect();
                let target: Vec<u32> = sorted
                    .iter()
                    .flat_map(|_| am.iter().map(|&b| delta * u32::from(b)))
                    .collect();
                // Step 5: s uniform subject to H_S^T s = Delta * am_bar.
                let s = plan.fiber().sample(&target, &mut rng).unwrap();
                for (col, &t) in plan.fiber().matrix().iter().zip(&target) {
                    let dot: u64 = col
                        .iter()
                        .zip(&s)
                        .map(|(&h, &x)| u64::from(h) * u64::from(x))
                        .sum();
                    assert_eq!((dot % u64::from(q)) as u32, t);
                }
                // Step 6 with explicit coins.
                let e: Vec<i64> = (0..m).map(|_| rng.random_range(-9..=9)).collect();
                let y: i64 = rng.random_range(-9..=9);
                let ct = native_from_coins(pk, bit, &s, &e, y).unwrap();
                for j in 0..m {
                    let ats: i128 = (0..n)
                        .map(|r| i128::from(a[r * m + j]) * i128::from(s[r]))
                        .sum();
                    assert_eq!(ct.header()[j], modq(ats + i128::from(e[j]), q));
                }
                let bts: i128 = (0..n)
                    .map(|r| i128::from(pk.vector()[r]) * i128::from(s[r]))
                    .sum();
                assert_eq!(
                    ct.scalar(),
                    modq(bts + i128::from(y) + i128::from(delta) * i128::from(bit), q)
                );
                // Dec recovers m; its residual is Delta*m + y - k^T e, independent of s.
                let kte: i128 = (0..m)
                    .map(|j| i128::from(ordinary.secret[j]) * i128::from(e[j]))
                    .sum();
                assert_eq!(
                    ordinary_residual(&ct, &ordinary.secret, q).unwrap(),
                    modq(i128::from(delta) * i128::from(bit) + i128::from(y) - kte, q)
                );
                assert_eq!(ordinary_decrypt(&ct, &ordinary.secret, q).unwrap(), bit);
                // aDec residual: Delta*am_a + e[1] - k^T e for every i in S and a in [L].
                for &i in &sorted {
                    let blocks = user_blocks(&keys.user_keys[i], &p).unwrap();
                    let res = hidden_residuals(&ct, &keys.user_keys[i], &p).unwrap();
                    for ab in 0..bits {
                        let kte: i128 = (0..m)
                            .map(|j| i128::from(blocks[ab][j]) * i128::from(e[j]))
                            .sum();
                        let expect = modq(
                            i128::from(delta) * i128::from(am[ab]) + i128::from(e[0]) - kte,
                            q,
                        );
                        assert_eq!(res[ab], expect);
                    }
                    assert_eq!(hidden_decrypt(&ct, &keys.user_keys[i], &p).unwrap(), am);
                }
            }
        }
    }
}

#[test]
fn group_size_bound_and_rank_fallback_follow_construction_1() {
    let mut rng = StdRng::seed_from_u64(7);
    let p = small_params(4);
    let ordinary = keygen(p, &mut rng).unwrap();
    let keys = anamorphic_keygen(&ordinary.public, 4, false, &mut rng).unwrap();
    // s_max = min{N, floor((n - n_min)/L)} = min{4, 3} = 3.
    assert!(prepare_hidden(&ordinary.public, &keys.directory, &[0, 1, 2]).is_ok());
    assert!(prepare_hidden(&ordinary.public, &keys.directory, &[0, 1, 2, 3]).is_err());
    // Two identical public vectors make H_S rank deficient: aEnc returns Enc(pk, m)
    // with exactly the ordinary coin consumption.
    let mut dir = keys.directory.clone();
    dir[1] = dir[0].clone();
    let plan = prepare_hidden(&ordinary.public, &dir, &[0, 1]).unwrap();
    assert!(!plan.fiber().full_rank());
    let mut r1 = StdRng::seed_from_u64(99);
    let mut r2 = StdRng::seed_from_u64(99);
    assert_eq!(
        plan.encrypt(1, &[1, 0, 1, 1], &mut r1).unwrap(),
        encrypt(&ordinary.public, 1, &mut r2).unwrap()
    );
}
