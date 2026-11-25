use rbe_anamorphic_review::*;
use scalable_rbe::{
    mat::Mat,
    ntt::MatNTT,
    rbe::{B, D, N, Q, RBE},
    utils::{drop_db, unpack},
};

#[test]
fn setup_supports_encryption_before_registration() {
    drop_db();
    let host = RBE::init(false);
    let (pp, st, reg) = host.setup();
    assert_eq!((st.d, st.nr_rows, st.nr_columns), (D, N, B));
    assert_eq!(st.matrix.len(), D * N * B);
    for t in 0..B {
        let stored = reg
            .open_tree(format!("{t}"))
            .unwrap()
            .get("")
            .unwrap()
            .unwrap();
        assert_eq!(
            &st.matrix[t * N * D..(t + 1) * N * D],
            unpack(&stored, Q, N * D)
        );
    }
    let family = Family::generate(&host, &pp, 0, 1, 256, COVERT_KEY_WIDTH).unwrap();
    let context = family.context(&[0]).unwrap();
    let message: Vec<_> = (0..256).map(|j| (j % 2) as u8).collect();
    let mut coins = Coins::sample(&host, 1);
    let ordinary = common_ordinary(0);
    let baseline = coins.encrypt(&host, &pp, &st, "0", ordinary.clone());
    let (baseline, baseline_bytes) = roundtrip(&host, &baseline);
    assert_eq!(baseline.last().unwrap().nr_columns, B);
    assert!(context.apply(&host, "0", &mut coins, &message).unwrap());
    let (hidden, hidden_bytes) = roundtrip(
        &host,
        &coins.encrypt(&host, &pp, &st, "0", ordinary.clone()),
    );
    assert_eq!(baseline_bytes, hidden_bytes);
    assert_eq!(family.decode(&host, 0, &hidden), message);

    let (pk, sk) = host.keygen(&pp);
    let updated = host.update(&reg, &pp, 0, "0", &pk);
    assert_eq!(&updated.matrix[N * D..], &st.matrix[N * D..]);
    let witness = host.witness_gen(&reg, 0, "0").unwrap();
    let ct = coins.encrypt(&host, &pp, &updated, "0", ordinary.clone());
    assert_eq!(
        host.dec(&sk, 0, &witness, &ct).matrix,
        ordinary.get_entry(0, 0)
    );
    drop(reg);
    drop_db();
}

#[test]
fn anamorphic_coins_skip_the_gaussian_root() {
    // aEnc draws every coin of RBE::enc except the Gaussian r_0, which the broadcast step
    // computes instead; with a Gaussian r_0 in its place the coins are ordinary again.
    drop_db();
    let host = RBE::init(false);
    let (pp, _, reg) = host.setup();
    let id = format!("0{}", &"012".repeat(54)[..161]);
    let (pk, sk) = host.keygen(&pp);
    let st = host.update(&reg, &pp, 0, &id, &pk);
    let witness = host.witness_gen(&reg, 0, &id).unwrap();
    let family = Family::generate(&host, &pp, 0, 2, 256, COVERT_KEY_WIDTH).unwrap();
    let context = family.context(&[0, 1]).unwrap();
    let ordinary = common_ordinary(1);
    let message: Vec<_> = (0..256).map(|j| (j % 3 == 0) as u8).collect();
    let mut coins = Coins::sample_without_root(&host, 162);
    assert_eq!((coins.r.len(), coins.e.len()), (163, 162));
    assert!(coins.r[0].matrix.iter().all(|&x| x == 0));
    assert!(
        coins.r[1..]
            .iter()
            .all(|r| r.matrix.iter().any(|&x| x != 0))
    );
    assert!(context.apply(&host, &id, &mut coins, &message).unwrap());
    assert!(context.check(&host, &coins.r[0], &message));
    let ct = coins.encrypt(&host, &pp, &st, &id, ordinary.clone());
    let (ct, bytes) = roundtrip(&host, &ct);
    let (_, ordinary_bytes) = roundtrip(&host, &host.enc(&pp, &st, &id, ordinary.clone()));
    assert_eq!(bytes, ordinary_bytes);
    for recipient in 0..2 {
        assert_eq!(family.decode(&host, recipient, &ct), message);
    }
    assert_eq!(
        host.dec(&sk, 0, &witness, &ct).matrix,
        ordinary.get_entry(0, 0)
    );
    let mut coins = Coins::sample_without_root(&host, 162);
    coins.r[0] = Coins::sample_root(&host);
    let ct = coins.encrypt(&host, &pp, &st, &id, ordinary.clone());
    assert_eq!(
        host.dec(&sk, 0, &witness, &ct).matrix,
        ordinary.get_entry(0, 0)
    );
    drop(reg);
    drop_db();
}

#[test]
fn authors_core_and_anamorphic_integration() {
    // This test must run in a dedicated working directory with --test-threads=1.
    drop_db();
    let mut host = RBE::init(false);
    let (pp, mut st, reg) = host.setup();
    assert!(Family::generate(&host, &pp, 0, 1, 256, 1 << 18).is_err());
    // One recipient more than the host's constraint budget is rejected.
    let too_many =
        Family::generate(&host, &pp, 0, MAX_CONSTRAINTS + 1, 256, COVERT_KEY_WIDTH).unwrap();
    let everyone: Vec<usize> = (0..=MAX_CONSTRAINTS).collect();
    assert!(too_many.context(&everyone).is_err());
    let ids: Vec<_> = (0..3)
        .map(|h| format!("{h}{}", &"012".repeat(54)[..161]))
        .collect();
    let mut ordinary_keys = Vec::new();
    for (h, id) in ids.iter().enumerate() {
        let (pk, sk) = host.keygen(&pp);
        st = host.update(&reg, &pp, h, id, &pk);
        ordinary_keys.push(sk);
    }
    let witnesses: Vec<_> = (0..3)
        .map(|h| host.witness_gen(&reg, h, &ids[h]).unwrap())
        .collect();
    // Verify the inverse-NTT normalization required by paper-model encryption.
    let mut polynomial = vec![0; D];
    polynomial[0] = 1;
    let fwd = MatNTT::ntt(
        Mat {
            matrix: polynomial,
            d: D,
            nr_rows: 1,
            nr_columns: 1,
        },
        &host.ntt_plan,
    );
    assert_eq!(fwd.clone().inv_ntt(&host.ntt_plan).matrix[0], D as u64);
    let mut corrected = fwd;
    corrected.normalise(&host.ntt_plan);
    assert_eq!(corrected.inv_ntt(&host.ntt_plan).matrix[0], 1);
    println!(
        "kind,branch,group,L,g,root_drop,trial,ordinary_errors,covert_errors,bytes,max_covert_error"
    );
    let cases = [(1, 1), (2, 256), (2, 257), (3, 256), (4, 256)];
    let mut total = 0;
    for h in 0..3 {
        for &(users, bits) in &cases {
            let family = Family::generate(&host, &pp, h, users + 1, bits, 1 << 19).unwrap();
            let group: Vec<_> = (0..users).collect();
            let context = family.context(&group).unwrap();
            assert!(context.full_rank());
            assert!(family.context(&[]).is_err());
            assert!(family.context(&[0, 0]).is_err());
            for prune in [8, 13] {
                host.root_prune_by = prune;
                for trial in 0..2 {
                    let message: Vec<_> =
                        (0..bits).map(|j| ((j * 7 + trial + h) % 2) as u8).collect();
                    let ordinary = common_ordinary(trial);
                    let mut coins = Coins::sample(&host, 162);
                    let baseline = coins.encrypt(&host, &pp, &st, &ids[h], ordinary.clone());
                    let baseline_dec = host.dec(&ordinary_keys[h], h, &witnesses[h], &baseline);
                    assert_eq!(baseline_dec.matrix, ordinary.get_entry(0, h));
                    let root = context.sample(&host, &message).unwrap().unwrap();
                    assert!(context.check(&host, &root, &message));
                    coins.r[0] = root;
                    assert!(context.apply(&host, &ids[h], &mut coins, &message).unwrap());
                    let wrong_branch = (h + 1) % 3;
                    assert!(
                        context
                            .apply(&host, &ids[wrong_branch], &mut coins, &message)
                            .is_err()
                    );
                    // Full registry root relation, using normalized ring products.
                    let a = &pp.0;
                    let lhs = coins.r[0]
                        .mul(a, &host.ntt_plan)
                        .mul(&witnesses[h][0], &host.ntt_plan);
                    let rhs = coins.r[0].mul(&st, &host.ntt_plan);
                    assert_eq!(lhs.get_entry(0, 0), rhs.get_entry(0, h));
                    let ct = coins.encrypt(&host, &pp, &st, &ids[h], ordinary.clone());
                    let (ct, bytes) = roundtrip(&host, &ct);
                    let wire = serialize(&host, &ct);
                    assert!(deserialize(&host, 162, &wire[..wire.len() - 1]).is_err());
                    let mut trailing = wire;
                    trailing.push(0);
                    assert!(deserialize(&host, 162, &trailing).is_err());
                    let ordinary_out = host.dec(&ordinary_keys[h], h, &witnesses[h], &ct);
                    let ordinary_errors = ordinary_out
                        .matrix
                        .iter()
                        .zip(ordinary.get_entry(0, h))
                        .filter(|(x, y)| x != y)
                        .count();
                    let mut covert_errors = 0;
                    let mut max_error = 0;
                    for i in 0..users {
                        covert_errors += bit_errors(&family.decode(&host, i, &ct), &message);
                        max_error = max_error.max(family.max_covert_error(&host, i, &ct, &message));
                    }
                    println!(
                        "functional,{h},{users},{bits},{},{prune},{trial},{ordinary_errors},{covert_errors},{bytes},{max_error}",
                        context.constraint_count()
                    );
                    assert_eq!(ordinary_errors, 0);
                    assert_eq!(covert_errors, 0);
                    total += 1;
                }
            }
        }
    }
    // Force a genuinely rank-deficient matrix by duplicating a public recipient key.
    let mut family = Family::generate(&host, &pp, 0, 2, 256, 1 << 19).unwrap();
    family.recipients[1] = family.recipients[0].clone();
    let context = family.context(&[0, 1]).unwrap();
    assert!(!context.full_rank());
    let message = vec![1; 256];
    assert!(context.sample(&host, &message).unwrap().is_none());
    let coins = Coins::sample(&host, 162);
    let ordinary = common_ordinary(0);
    let expected = coins.encrypt(&host, &pp, &st, &ids[0], ordinary.clone());
    let mut fallback = coins.clone();
    if let Some(root) = context.sample(&host, &message).unwrap() {
        fallback.r[0] = root;
    }
    assert_eq!(
        expected,
        fallback.encrypt(&host, &pp, &st, &ids[0], ordinary)
    );
    // Payloads that exceed the constraint budget are rejected before encryption.
    let family = Family::generate(&host, &pp, 0, 2, 1792, 1 << 19).unwrap();
    assert!(family.context(&[0, 1]).is_err());
    println!(
        "summary,functional_ciphertexts={total},full_path_depth=162,branches=3,rank_failure_fallback=pass,ntt_scale=1"
    );
    drop(reg);
    drop_db();
}

#[test]
fn rejects_noncanonical_root_coefficient() {
    let host = RBE::init(false);
    let blocks: Vec<_> = [3 * host.m_a, N, B]
        .into_iter()
        .map(|columns| Mat {
            matrix: vec![0; D * columns],
            d: D,
            nr_rows: 1,
            nr_columns: columns,
        })
        .collect();
    let mut wire = serialize(&host, &blocks);
    assert!(deserialize(&host, 1, &wire).is_ok());
    // The first coefficient uses 35 bits, but the all-ones value is outside its bound.
    wire[..4].fill(0xff);
    wire[4] |= 0x07;
    assert_eq!(
        deserialize(&host, 1, &wire).unwrap_err(),
        "noncanonical ciphertext coefficient"
    );
}
