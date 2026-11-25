//! Audit regressions; run serially in a disposable working directory.
//! The ignored residual test records diagnostics and checks recovery.
use rbe_anamorphic_review::*;
use scalable_rbe::{
    mat::Mat,
    ntt::MatNTT,
    rbe::{B, D, K, M_B, PRUNE_CT_BY, Q},
    utils::{compress, decompress, drop_db},
};

fn centred(v: u64) -> u64 {
    centred_abs(v % Q)
}

/// The pre-follow-up decompression, kept here only for comparison.
fn old_decompress(v: u64, p: usize, q: u64) -> u64 {
    if v > (q >> p) / 2 {
        q - (((q >> p) - v) << p)
    } else {
        v << p
    }
}

#[test]
fn decompression_is_canonical_and_preserves_residues() {
    // Exhaustive on small moduli with q = 1 mod 2^(p+1), every valid symbol.
    for (q, p) in [(257u64, 4usize), (12289, 10), (65537, 8), (65537, 13)] {
        let symbols: Vec<u64> = (0..=(q >> p)).collect();
        let mut new = symbols.clone();
        decompress(&mut new, p, q);
        for (&c, &n) in symbols.iter().zip(&new) {
            let o = old_decompress(c, p, q);
            assert!(n < q && o <= q);
            assert_eq!(n, o % q, "q={q} p={p} c={c}");
        }
    }
    // Profile modulus: every symbol near both ends and the middle.
    for p in [8usize, 13] {
        let top = Q >> p;
        let mut symbols: Vec<u64> = (0..5000).collect();
        symbols.extend(top - 5000..=top);
        symbols.extend(top / 2 - 5000..top / 2 + 5000);
        let mut new = symbols.clone();
        decompress(&mut new, p, Q);
        let mut moved = 0;
        for (&c, &n) in symbols.iter().zip(&new) {
            let o = old_decompress(c, p, Q);
            assert!(n < Q);
            assert_eq!(n, o % Q);
            if o == Q {
                moved += 1;
            }
        }
        assert_eq!(moved, 1, "only the top symbol changes representative");
        // Compression of every original value in the top bucket now round-trips to 0.
        let mut x: Vec<u64> = (Q - (1 << (p - 1))..Q).collect();
        compress(&mut x, p, Q);
        assert!(x.iter().all(|&c| c == top));
        decompress(&mut x, p, Q);
        assert!(x.iter().all(|&v| v == 0));
    }
}

/// Force the top root symbol into a real ciphertext: wire round trip, ordinary
/// decryption and covert decoding must equal those with the zero symbol.
#[test]
fn top_symbol_is_wire_valid_and_decrypts_like_zero() {
    drop_db();
    let mut host = Host::init(false);
    host.root_prune_by = 13;
    let (pp, _st0, reg) = host.setup();
    let depth = 16;
    let id = format!("2{}", &"012".repeat(6)[..depth - 1]);
    let (pk, sk) = host.keygen(&pp);
    let st = host.update(&reg, &pp, 0, &id, &pk);
    let wit = host.witness_gen(&reg, 0, &id).unwrap();
    let family = Family::generate(&host, &pp, 2, 2, 256, COVERT_KEY_WIDTH).unwrap();
    let context = family.context(&[0]).unwrap();
    let ordinary = common_ordinary(0);
    let message: Vec<u8> = (0..256).map(|j| (j % 3 == 0) as u8).collect();
    let mut coins = Coins::sample(&host, depth);
    assert!(context.apply(&host, &id, &mut coins, &message).unwrap());
    let ct = coins.encrypt(&host, &pp, &st, &id, ordinary.clone());
    let top = Q >> 13;
    // Carrier column 0 coefficient 0 and a non-carrier root coefficient.
    let active_column = family.branch * host.m_a;
    assert!(!family.carrier_columns.contains(&active_column));
    let positions = [family.carrier_columns[0] * D, active_column * D + 7];
    let mut a = ct.clone();
    let mut b = ct.clone();
    for &i in &positions {
        a[0].matrix[i] = top;
        b[0].matrix[i] = 0;
    }
    let (a, _) = roundtrip(&host, &a);
    let (b, _) = roundtrip(&host, &b);
    assert_eq!(
        host.dec(&sk, 0, &wit, &a).matrix,
        host.dec(&sk, 0, &wit, &b).matrix
    );
    assert_eq!(family.decode(&host, 0, &a), family.decode(&host, 0, &b));
    drop(reg);
    drop_db();
}

#[test]
fn unpruned_wire_accepts_q_minus_one_and_rejects_q() {
    let mut host = Host::init(false);
    host.root_prune_by = 0;
    let mut blocks: Vec<_> = [K * host.m_a, M_B, B]
        .into_iter()
        .map(|columns| Mat {
            matrix: vec![0; D * columns],
            d: D,
            nr_rows: 1,
            nr_columns: columns,
        })
        .collect();
    blocks[0].matrix[0] = Q - 1;
    let mut wire = serialize(&host, &blocks);
    assert_eq!(deserialize(&host, 1, &wire).unwrap(), blocks);
    // The unpruned modulus uses 48 wire bits; inject the forbidden value q.
    wire[..6].copy_from_slice(&Q.to_le_bytes()[..6]);
    assert_eq!(
        deserialize(&host, 1, &wire).unwrap_err(),
        "noncanonical ciphertext coefficient"
    );
}

/// Record the ordinary-key residual in both modes; RMS is descriptive evidence.
#[test]
#[ignore = "manual residual diagnostic; run with --ignored --nocapture"]
fn exposed_key_residual_diagnostic() {
    drop_db();
    let mut host = Host::init(false);
    host.root_prune_by = 13;
    let (pp, _st0, reg) = host.setup();
    let depth = 16;
    let id = format!("1{}", &"012".repeat(6)[..depth - 1]);
    let (pk, sk) = host.keygen(&pp);
    let st = host.update(&reg, &pp, 0, &id, &pk);
    let wit = host.witness_gen(&reg, 0, &id).unwrap();
    let family = Family::generate(&host, &pp, 1, 3, 512, COVERT_KEY_WIDTH).unwrap();
    let context = family.context(&[0, 1]).unwrap();
    let ordinary = common_ordinary(1);
    let mut stats = Vec::new();
    for mode in ["ordinary", "anamorphic"] {
        let (mut worst, mut sumsq, mut count, mut r0_max) = (0u64, 0f64, 0f64, 0u64);
        for trial in 0..6 {
            let message: Vec<u8> = (0..512).map(|j| ((j * 3 + trial) % 2) as u8).collect();
            let mut coins = Coins::sample(&host, depth);
            if mode == "anamorphic" {
                assert!(context.apply(&host, &id, &mut coins, &message).unwrap());
            }
            let mut r0 = coins.r[0].clone();
            r0.normalise(&host.ntt_plan);
            let r0 = r0.inv_ntt(&host.ntt_plan);
            r0_max = r0_max.max(r0.matrix.iter().map(|&v| centred(v)).max().unwrap());
            let ct = coins.encrypt(&host, &pp, &st, &id, ordinary.clone());
            let (ct, _) = roundtrip(&host, &ct);
            let mut mu = Mat {
                matrix: ct.last().unwrap().get_entry(0, 0).to_vec(),
                d: D,
                nr_rows: 1,
                nr_columns: 1,
            };
            decompress(&mut mu.matrix, PRUNE_CT_BY, Q);
            let mut acc = MatNTT::ntt(mu, &host.ntt_plan);
            for (j, w) in wit.iter().enumerate() {
                let mut c = ct[j].clone();
                decompress(
                    &mut c.matrix,
                    if j == 0 {
                        host.root_prune_by
                    } else {
                        PRUNE_CT_BY
                    },
                    Q,
                );
                acc.sub_assign(&MatNTT::ntt(c, &host.ntt_plan).mul(w, &host.ntt_plan), Q);
            }
            let mut cl = ct[wit.len()].clone();
            decompress(&mut cl.matrix, PRUNE_CT_BY, Q);
            acc.sub_assign(&MatNTT::ntt(cl, &host.ntt_plan).mul(&sk, &host.ntt_plan), Q);
            acc.normalise(&host.ntt_plan);
            let res = acc.inv_ntt(&host.ntt_plan);
            for (z, &v) in res.matrix.iter().enumerate() {
                let noise = centred(v + Q - (Q / 2) * ordinary.get_entry(0, 0)[z]);
                worst = worst.max(noise);
                sumsq += (noise as f64).powi(2);
                count += 1.0;
            }
            assert_eq!(host.dec(&sk, 0, &wit, &ct).matrix, ordinary.get_entry(0, 0));
            if mode == "anamorphic" {
                for i in 0..2 {
                    assert_eq!(family.decode(&host, i, &ct), message);
                }
            }
        }
        // Full-depth profile bound, reproduced by security/check_parameters.py.
        assert!(worst <= 60_043_642_798_208);
        let rms = (sumsq / count).sqrt();
        println!("{mode}: max|r0|={r0_max} residual max={worst} rms={rms:.3e}");
        stats.push((r0_max, worst, rms));
    }
    let ratio = stats[1].2 / stats[0].2;
    println!("anamorphic/ordinary residual RMS ratio: {ratio:.3}");
    drop(reg);
    drop_db();
}
