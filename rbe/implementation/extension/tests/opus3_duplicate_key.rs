// Regression for the low-level BLE interface: repeated leaf keys expose slot
// differences. A common ordinary message cancels in this statistic.
// Run serially from a disposable copy (clears registry_db):
//   cargo test --locked --release --test opus3_duplicate_key -- --test-threads=1 --nocapture
use rbe_anamorphic_review::*;
use scalable_rbe::{
    mat::Mat,
    ntt::MatNTT,
    rbe::{D, PRUNE_CT_BY, Q},
    utils::{compress, decompress, drop_db},
};

/// R_t = d_t - sum_{j<l} c_j^T u_{t,j}, computed from public data only (no secret key).
fn public_residual(host: &Host, ct: &[Mat], wit: &[MatNTT], t: usize) -> Vec<u64> {
    let mut d = Mat {
        matrix: ct.last().unwrap().get_entry(0, t).to_vec(),
        d: D,
        nr_rows: 1,
        nr_columns: 1,
    };
    decompress(&mut d.matrix, PRUNE_CT_BY, Q);
    let mut acc = MatNTT::ntt(d, &host.ntt_plan);
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
    acc.normalise(&host.ntt_plan);
    acc.inv_ntt(&host.ntt_plan).matrix
}

fn max_centred_diff(a: &[u64], b: &[u64]) -> u64 {
    a.iter()
        .zip(b)
        .map(|(&x, &y)| centred_abs(sub(x, y)))
        .max()
        .unwrap()
}

#[test]
fn duplicate_leaf_key_exposes_slot_difference() {
    drop_db();
    let host = Host::init(false);
    let (pp, _st0, reg) = host.setup();
    let depth = 16;
    let id = format!("0{}", &"012".repeat(6)[..depth - 1]);
    // Same honest key registered for the same identity in subregistries 0 and 1.
    let (pk, _sk) = host.keygen(&pp);
    let _ = host.update(&reg, &pp, 0, &id, &pk);
    // A different honest key for the same identity in subregistry 2 (control).
    let (pk2, _sk2) = host.keygen(&pp);
    let _ = host.update(&reg, &pp, 2, &id, &pk2);
    let st = host.update(&reg, &pp, 1, &id, &pk);
    let w: Vec<_> = (0..3)
        .map(|t| host.witness_gen(&reg, t, &id).unwrap())
        .collect();
    let mut dup = Vec::new();
    let mut distinct = Vec::new();
    let mut uniform = Vec::new();
    for trial in 0..5 {
        let coins = Coins::sample(&host, depth);
        let ct = coins.encrypt(&host, &pp, &st, &id, common_ordinary(trial));
        let (ct, _) = roundtrip(&host, &ct);
        let r: Vec<_> = (0..3)
            .map(|t| public_residual(&host, &ct, &w[t], t))
            .collect();
        dup.push(max_centred_diff(&r[0], &r[1]));
        distinct.push(max_centred_diff(&r[0], &r[2]));
        // Match the proof endpoint: compress uniform uncompressed coefficients.
        let mut rnd = ct.clone();
        let mut rng = rand::rng();
        use rand::RngExt;
        for x in rnd.last_mut().unwrap().matrix.iter_mut() {
            *x = rng.random_range(0..Q);
        }
        compress(&mut rnd.last_mut().unwrap().matrix, PRUNE_CT_BY, Q);
        let ru: Vec<_> = (0..2)
            .map(|t| public_residual(&host, &rnd, &w[t], t))
            .collect();
        uniform.push(max_centred_diff(&ru[0], &ru[1]));
    }
    // BLE-level per-slot messages: slot 0 carries m, slot 1 carries m XOR 1 in coefficient 0..127.
    let coins = Coins::sample(&host, depth);
    let mut msg = common_ordinary(0);
    for z in 0..128 {
        msg.matrix[D + z] ^= 1;
    }
    let ct = coins.encrypt(&host, &pp, &st, &id, msg.clone());
    let r0 = public_residual(&host, &ct, &w[0], 0);
    let r1 = public_residual(&host, &ct, &w[1], 1);
    let mut recovered = 0;
    for z in 0..D {
        let diff = centred_abs(sub(r0[z], r1[z]));
        let guess = u64::from(diff > Q / 4);
        if guess == (msg.matrix[z] ^ msg.matrix[D + z]) {
            recovered += 1;
        }
    }
    println!(
        "per-slot message difference recovered from public data: {recovered}/{D} coefficients"
    );
    assert_eq!(recovered, D);
    println!("q/4 = {}", Q / 4);
    println!("same key in t=0,1   : max |R_0 - R_1| per trial = {dup:?}");
    println!("distinct keys t=0,2 : max |R_0 - R_2| per trial = {distinct:?}");
    println!("uniform d (IND$ end): max |R_0 - R_1| per trial = {uniform:?}");
    // Use the attack's decision radius, not an unsupported tighter threshold.
    assert!(dup.iter().all(|&m| m < Q / 4));
    assert!(distinct.iter().all(|&m| m > Q / 4));
    assert!(uniform.iter().all(|&m| m > Q / 4));
    drop(reg);
    drop_db();
}
