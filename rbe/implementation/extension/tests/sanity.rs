//! Value audit of the RBE channel on real ciphertexts: nothing may be zero,
//! constant or repeated unless the protocol says so (a zero payload bit gives a
//! zero target, the last payload polynomial is zero-padded, a rank failure
//! keeps the host's own coins). Bounds are about six standard deviations.
//! Run serially in a disposable working directory (clears registry_db), with
//! `--nocapture` to see the statistics.
use rbe_anamorphic_review::*;
use scalable_rbe::{
    mat::Mat,
    ntt::MatNTT,
    rbe::{D, K, N, Q, RBE},
    utils::{decompress, drop_db},
};

fn centred(x: u64) -> i128 {
    if x > Q / 2 {
        i128::from(x) - i128::from(Q)
    } else {
        i128::from(x)
    }
}

/// Residues that should be uniform modulo Q.
fn check_uniform(label: &str, values: &[u64]) {
    let n = values.len() as f64;
    let q = Q as f64;
    let mean = values.iter().map(|&v| v as f64).sum::<f64>() / n;
    let sd = (values
        .iter()
        .map(|&v| (v as f64 - mean).powi(2))
        .sum::<f64>()
        / n)
        .sqrt();
    let zeros = values.iter().filter(|&&v| v == 0).count();
    eprintln!(
        "{label}: {} values, mean/q {:.4}, sd/q {:.4}, zeros {zeros}",
        values.len(),
        mean / q,
        sd / q
    );
    assert!(values.iter().all(|&v| v < Q), "{label}: noncanonical");
    assert!(
        (mean / q - 0.5).abs() < 6.0 * 0.2887 / n.sqrt(),
        "{label}: mean"
    );
    assert!(
        (sd / q - 0.2887).abs() < 0.02 + 6.0 * 0.2 / n.sqrt(),
        "{label}: spread"
    );
    assert!(zeros <= 2, "{label}: {zeros} zeros");
}

/// Small signed values (centred residues) with an expected standard deviation.
fn check_small(label: &str, values: &[u64], expected_sd: f64) {
    let v: Vec<f64> = values.iter().map(|&x| centred(x) as f64).collect();
    let n = v.len() as f64;
    let mean = v.iter().sum::<f64>() / n;
    let sd = (v.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n).sqrt();
    let zeros = v.iter().filter(|&&x| x == 0.0).count();
    eprintln!(
        "{label}: {} values, mean {mean:.2}, sd {sd:.2} (expected {expected_sd:.2}), zeros {zeros}",
        v.len()
    );
    assert!(
        mean.abs() < 6.0 * expected_sd / n.sqrt() + 1e-9,
        "{label}: mean"
    );
    assert!((sd / expected_sd - 1.0).abs() < 0.05, "{label}: spread");
    assert!(v.windows(2).any(|w| w[0] != w[1]), "{label}: constant");
}

fn coefficients(host: &Host, x: &MatNTT) -> Vec<u64> {
    let mut t = x.clone();
    t.normalise(&host.ntt_plan);
    t.inv_ntt(&host.ntt_plan).matrix
}

#[test]
fn rbe_channel_values_are_not_degenerate() {
    drop_db();
    let host = RBE::init(false);
    let (pp, _st0, reg) = host.setup();
    let depth = 162;
    let (pk, sk) = host.keygen(&pp);
    let users = 5;
    let mut checked_ciphertexts = 0;
    let mut worst = 0u64;
    for branch in 0..K {
        let id = format!("{branch}{}", &"012".repeat(depth)[..depth - 1]);
        let st = host.update(&reg, &pp, 0, &id, &pk);
        let wit = host.witness_gen(&reg, 0, &id).unwrap();
        for (group, bits) in [
            (vec![3usize], 256usize),
            (vec![0, 4], 300),
            (vec![4, 1, 2, 0], 200),
        ] {
            eprintln!("--- branch {branch}, group {group:?}, {bits} payload bits");
            let family =
                Family::generate(&host, &pp, branch, users, bits, COVERT_KEY_WIDTH).unwrap();
            check_uniform("carrier C (NTT values)", &family.carrier.matrix);
            let mut keys = Vec::new();
            let mut publics = Vec::new();
            for rec in &family.recipients {
                for (k, b) in rec.secrets.iter().zip(&rec.public) {
                    keys.extend(coefficients(&host, k));
                    publics.extend_from_slice(&b.matrix);
                }
            }
            // Covert keys: uniform in [-2^18, 2^18), sd 2^19/sqrt(12).
            check_small(
                "covert keys k",
                &keys,
                COVERT_KEY_WIDTH as f64 / 12f64.sqrt(),
            );
            check_uniform("covert public keys C k (NTT values)", &publics);
            let context = family.context(&group).unwrap();
            assert!(context.full_rank(), "constraint systems rank deficient");
            let mut roots = Vec::new();
            let mut outsiders = (0usize, 0usize);
            for trial in 0..3 {
                let message: Vec<u8> = (0..bits)
                    .map(|j| ((j * 7 + trial) % 3 == 0) as u8)
                    .collect();
                let ordinary = common_ordinary(trial);
                let mut coins = Coins::sample(&host, depth);
                // Host coins: small nonzero Gaussian vectors.
                let original_root = coins.r[0].clone();
                assert!(coins.r.iter().all(|r| r.matrix.iter().any(|&x| x != 0)));
                assert!(coins.e.iter().all(|e| e.matrix.iter().any(|&x| x != 0)));
                assert!(coins.e_leaf.matrix.iter().any(|&x| x != 0));
                assert!(coins.e_message.matrix.iter().any(|&x| x != 0));
                assert!(context.apply(&host, &id, &mut coins, &message).unwrap());
                assert!(context.check(&host, &coins.r[0], &message));
                assert_ne!(coins.r[0], original_root, "r_0 was not replaced");
                roots.extend_from_slice(&coins.r[0].matrix);
                let ct = coins.encrypt(&host, &pp, &st, &id, ordinary.clone());
                let (ct, _) = roundtrip(&host, &ct);
                assert_eq!(ct.len(), depth + 2);
                // Every block is a full-width pseudorandom vector.
                for (j, block) in ct.iter().enumerate() {
                    let sample = &block.matrix[..block.matrix.len().min(4096)];
                    let distinct: std::collections::BTreeSet<_> = sample.iter().collect();
                    assert!(
                        distinct.len() * 10 > sample.len() * 9,
                        "block {j} looks degenerate: {} distinct of {}",
                        distinct.len(),
                        sample.len()
                    );
                }
                // Carrier: decompressed root = r_0^T C + e_C + codec rounding, nonzero noise.
                let mut product = coins.r[0].mul(&family.carrier, &host.ntt_plan);
                product.normalise(&host.ntt_plan);
                let expect = product.inv_ntt(&host.ntt_plan);
                let mut root = ct[0].clone();
                decompress(&mut root.matrix, host.root_prune_by, Q);
                let mut noise = Vec::new();
                for (col, &j) in family.carrier_columns.iter().enumerate() {
                    for z in 0..D {
                        noise.push(sub(root.get_entry(0, j)[z], expect.get_entry(0, col)[z]));
                    }
                }
                // Rounding to multiples of 2^13 is about uniform on [-2^12, 2^12).
                check_small(
                    "carrier noise (codec rounding plus e_C)",
                    &noise,
                    (1u64 << host.root_prune_by) as f64 / 12f64.sqrt(),
                );
                for &i in &group {
                    assert_eq!(family.decode(&host, i, &ct), message);
                    worst = worst.max(family.max_covert_error(&host, i, &ct, &message));
                }
                for i in (0..users).filter(|i| !group.contains(i)) {
                    let decoded = family.decode(&host, i, &ct);
                    outsiders.0 += bits - bit_errors(&decoded, &message);
                    outsiders.1 += bits;
                }
                assert_eq!(host.dec(&sk, 0, &wit, &ct).matrix, ordinary.get_entry(0, 0));
                checked_ciphertexts += 1;
            }
            // The constrained roots are uniform in every CRT slot.
            check_uniform("anamorphic r_0 (NTT values)", &roots);
            let rate = outsiders.0 as f64 / outsiders.1 as f64;
            eprintln!(
                "non-recipients agree with the payload on {rate:.3} of {} bits (uniform 0.5)",
                outsiders.1
            );
            assert!((rate - 0.5).abs() < 6.0 * 0.5 / (outsiders.1 as f64).sqrt());
            // Rank failure keeps the host's coins: two identical recipients.
            let mut twin = Family::generate(&host, &pp, branch, 2, 256, COVERT_KEY_WIDTH).unwrap();
            twin.recipients[1] = twin.recipients[0].clone();
            let deficient = twin.context(&[0, 1]).unwrap();
            assert!(!deficient.full_rank());
            let mut coins = Coins::sample(&host, depth);
            let before = coins.r[0].clone();
            assert!(
                !deficient
                    .apply(&host, &id, &mut coins, &vec![1; 256])
                    .unwrap()
            );
            assert_eq!(coins.r[0], before);
        }
    }
    eprintln!(
        "{checked_ciphertexts} anamorphic ciphertexts checked, largest covert residual {:.4} of q/4",
        worst as f64 / (Q / 4) as f64
    );
    assert!(worst < Q / 4);
    let _ = (
        N,
        Mat::sample_small::<rand::rngs::StdRng, rand_distr::Uniform<u64>>,
    );
    drop(reg);
    drop_db();
}
