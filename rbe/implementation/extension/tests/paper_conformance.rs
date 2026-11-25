//! The RBE embedding of Construction 1, checked against the host's equations.
//! In the module setting the anamorphic scalar is the first inactive root column:
//!   carrier  v^T = r_0^T C + e_C              (host encryption, normalised products)
//!   apk      b = C k, k with coefficients in [-2^18, 2^18)
//!   aEnc     (c_1 - b)^T r_0 = Delta * am     in every CRT slot (eq. (3))
//!   aDec     v[0] - v^T k = Delta * am + e_C[0] - e_C^T k  (+ codec rounding, eq. (4))
//! Run serially in a disposable working directory (clears registry_db).
use rbe_anamorphic_review::*;
use scalable_rbe::{
    mat::Mat,
    ntt::MatNTT,
    rbe::{D, N, Q, RBE},
    utils::{decompress, drop_db},
};

#[test]
fn rank_budget_matches_the_profile() {
    assert_eq!(N - MAX_CONSTRAINTS, MIN_ROOT_RANK);
    match N {
        // Sec. 7.2: rank 16 with g <= 4 leaves root rank at least 12.
        16 => {
            assert_eq!(MIN_ROOT_RANK, 12);
            assert_eq!(MAX_CONSTRAINTS, 4);
        }
        // The authors' rank-7 host, up to six constraints for cost comparisons.
        7 => assert_eq!(MAX_CONSTRAINTS, 6),
        // Ranks 8 to 15 keep the authors' rank 7 free.
        _ => assert_eq!(MIN_ROOT_RANK, 7),
    }
}

#[test]
fn carrier_constraint_and_residual_identities_hold() {
    // The group below needs four constraints.
    if MAX_CONSTRAINTS < 4 {
        return;
    }
    drop_db();
    let host = RBE::init(false);
    let (pp, _st0, reg) = host.setup();
    let depth = 8;
    let branch = 1;
    let id = format!("{branch}{}", &"012".repeat(3)[..depth - 1]);
    let (pk, sk) = host.keygen(&pp);
    let st = host.update(&reg, &pp, 0, &id, &pk);
    let wit = host.witness_gen(&reg, 0, &id).unwrap();
    let users = 2;
    let bits = 300; // two payload polynomials per recipient, g = 4
    let family = Family::generate(&host, &pp, branch, users + 1, bits, COVERT_KEY_WIDTH).unwrap();
    let group = [2usize, 0];
    let context = family.context(&group).unwrap();
    assert_eq!(context.constraint_count(), 4);
    assert!(context.full_rank());
    // One more recipient (three of two polynomials) exceeds the rank-16 budget of four
    // constraints and still fits the six of the rank-7 host.
    assert_eq!(family.context(&[0, 1, 2]).is_err(), 3 * 2 > MAX_CONSTRAINTS);
    // apk: b = C k, with C the inactive columns of the explicit host matrix A.
    let a = &pp.0;
    for (col, &j) in family.carrier_columns.iter().enumerate() {
        assert_ne!(
            j / host.m_a,
            branch,
            "carrier must avoid the active branch block"
        );
        assert_eq!(family.carrier.get_entry(0, col), a.get_entry(0, j));
    }
    for rec in &family.recipients {
        for (k, b) in rec.secrets.iter().zip(&rec.public) {
            assert_eq!(&family.carrier.mul(k, &host.ntt_plan), b);
            let mut t = k.clone();
            t.normalise(&host.ntt_plan);
            let coeffs = t.inv_ntt(&host.ntt_plan);
            assert!(
                coeffs
                    .matrix
                    .iter()
                    .all(|&x| centred_abs(x) <= COVERT_KEY_WIDTH / 2)
            );
        }
    }
    let ordinary = common_ordinary(1);
    for trial in 0..3 {
        let message: Vec<u8> = (0..bits).map(|j| ((j + 3 * trial) % 5 < 2) as u8).collect();
        let mut coins = Coins::sample(&host, depth);
        assert!(context.apply(&host, &id, &mut coins, &message).unwrap());
        assert!(context.check(&host, &coins.r[0], &message));
        let ct = coins.encrypt(&host, &pp, &st, &id, ordinary.clone());
        // Carrier equation: decompressed root columns = r_0^T C + e_C up to codec rounding.
        let mut product = coins.r[0].mul(&family.carrier, &host.ntt_plan);
        product.normalise(&host.ntt_plan);
        let expect = product.inv_ntt(&host.ntt_plan);
        let mut root = ct[0].clone();
        decompress(&mut root.matrix, host.root_prune_by, Q);
        for (col, &j) in family.carrier_columns.iter().enumerate() {
            for z in 0..D {
                let e = coins.e[0].get_entry(0, j)[z];
                let want = add(expect.get_entry(0, col)[z], e);
                let got = root.get_entry(0, j)[z];
                // Rounding error at most 2^(p-1): q = 1 mod 2^13 leaves no offset.
                assert!(centred_abs(sub(got, want)) <= 1 << (host.root_prune_by - 1));
            }
        }
        // Every intended recipient decodes the common payload; the ordinary receiver its bits.
        for &i in &group {
            assert_eq!(family.decode(&host, i, &ct), message);
        }
        assert_eq!(host.dec(&sk, 0, &wit, &ct).matrix, ordinary.get_entry(0, 0));
        // The residual with the uncompressed carrier is exactly Delta*am + e_C[0] - e_C^T k.
        let mut v = Mat {
            matrix: Vec::new(),
            d: D,
            nr_rows: 1,
            nr_columns: family.carrier_columns.len(),
        };
        let mut ev = v.clone();
        for (col, &j) in family.carrier_columns.iter().enumerate() {
            let e = coins.e[0].get_entry(0, j);
            v.matrix.extend(
                expect
                    .get_entry(0, col)
                    .iter()
                    .zip(e)
                    .map(|(&x, &y)| add(x, y)),
            );
            ev.matrix.extend_from_slice(e);
        }
        let v_ntt = MatNTT::ntt(v.clone(), &host.ntt_plan);
        let e_ntt = MatNTT::ntt(ev.clone(), &host.ntt_plan);
        for (block, k) in family.recipients[group[0]].secrets.iter().enumerate() {
            let mut vk = v_ntt.mul(k, &host.ntt_plan);
            vk.normalise(&host.ntt_plan);
            let vk = vk.inv_ntt(&host.ntt_plan);
            let mut ek = e_ntt.mul(k, &host.ntt_plan);
            ek.normalise(&host.ntt_plan);
            let ek = ek.inv_ntt(&host.ntt_plan);
            for z in 0..D {
                let residual = sub(v.get_entry(0, 0)[z], vk.matrix[z]);
                let bit = message.get(block * D + z).copied().unwrap_or(0) as u64;
                let noise = sub(ev.get_entry(0, 0)[z], ek.matrix[z]);
                assert_eq!(residual, add(mul(Q / 2, bit), noise));
            }
        }
    }
    drop(reg);
    drop_db();
}
