//! Encryption benchmark: the authors' RBE alone versus RBE with the anamorphic
//! broadcast inside, plus anamorphic key generation, group setup and their sizes.
//!
//! Run in a disposable working directory (setup creates and this driver clears
//! `registry_db`), single-threaded for stable component timings:
//!   RAYON_NUM_THREADS=1 cargo run --locked --release -- --reps 50 --output rbe.json
//! Every timed ciphertext is decrypted afterwards (ordinary and every intended
//! hidden recipient); a single failure aborts the run.
use rand::Rng;
use rbe_anamorphic_review::*;
use scalable_rbe::{
    mat::Mat,
    rbe::{B, D, K, M_B, N, PRUNE_CT_BY, Q, RBE},
    utils::drop_db,
};
use serde_json::{Value, json};
use std::{fs, time::Instant};

fn ms(start: Instant) -> f64 {
    start.elapsed().as_secs_f64() * 1e3
}

fn stats(xs: &[f64]) -> Value {
    let mut s = xs.to_vec();
    s.sort_by(f64::total_cmp);
    let q = |p: f64| {
        let i = p * (s.len() - 1) as f64;
        let (lo, hi) = (i.floor() as usize, i.ceil() as usize);
        s[lo] + (s[hi] - s[lo]) * (i - lo as f64)
    };
    let mean = s.iter().sum::<f64>() / s.len() as f64;
    let sd = if s.len() > 1 {
        (s.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (s.len() - 1) as f64).sqrt()
    } else {
        0.0
    };
    json!({"n": s.len(), "median_ms": q(0.5), "mean_ms": mean, "sd_ms": sd, "q1_ms": q(0.25),
           "q3_ms": q(0.75), "min_ms": s[0], "max_ms": s[s.len() - 1], "samples_ms": xs})
}

struct Args {
    reps: usize,
    depth: usize,
    workloads: Vec<(usize, usize)>,
    output: String,
}

fn parse() -> Args {
    let mut a = Args {
        reps: 50,
        depth: 162,
        workloads: vec![(1, 256), (2, 256), (4, 256), (2, 512), (1, 1024)],
        output: "rbe-ab-benchmark.json".into(),
    };
    let mut it = std::env::args().skip(1);
    while let Some(flag) = it.next() {
        let v = it.next().expect("missing option value");
        match flag.as_str() {
            "--reps" => a.reps = v.parse().unwrap(),
            "--depth" => a.depth = v.parse().unwrap(),
            "--output" => a.output = v,
            "--workloads" => {
                a.workloads = v
                    .split(',')
                    .map(|w| {
                        let (s, l) = w.split_once(':').expect("GROUP:BITS");
                        (s.parse().unwrap(), l.parse().unwrap())
                    })
                    .collect()
            }
            _ => {
                panic!("unknown option {flag}; use --reps --depth --workloads 1:256,4:256 --output")
            }
        }
    }
    assert!(a.reps >= 2 && (1..=162).contains(&a.depth));
    a
}

fn random_ordinary(rng: &mut impl Rng) -> Mat {
    Mat {
        matrix: (0..B * D).map(|_| (rng.next_u32() & 1) as u64).collect(),
        d: D,
        nr_rows: 1,
        nr_columns: B,
    }
}

fn main() {
    if cfg!(debug_assertions) {
        eprintln!("run benchmarks with --release");
        std::process::exit(2);
    }
    let args = parse();
    let field_bits = 64 - (Q - 1).leading_zeros() as usize; // 48
    let key_bits = COVERT_KEY_WIDTH.trailing_zeros() as usize; // 19
    let mut rows = Vec::new();
    let mut root_drop = 0;
    for (w, &(users, bits)) in args.workloads.iter().enumerate() {
        let branch = w % K;
        let id = format!(
            "{branch}{}",
            &"012".repeat(args.depth.div_ceil(3))[..args.depth - 1]
        );
        drop_db();
        let host = RBE::init(false);
        root_drop = host.root_prune_by;
        let t = Instant::now();
        let (pp, _st0, reg) = host.setup();
        let host_setup_ms = ms(t);
        let t = Instant::now();
        let (pk, sk) = host.keygen(&pp);
        let host_keygen_ms = ms(t);
        let t = Instant::now();
        let st = host.update(&reg, &pp, 0, &id, &pk);
        let registration_ms = ms(t);
        let wit = host.witness_gen(&reg, 0, &id).unwrap();

        // Anamorphic key generation (aKGen) for |S| recipients plus one outsider.
        let population = users + 1;
        // aKGen reads A from the explicit public parameters, then samples keys and
        // computes b = C k for each user.
        let mut keygen_ms = Vec::new();
        let mut family = None;
        for _ in 0..args.reps {
            let t = Instant::now();
            let f =
                Family::generate(&host, &pp, branch, population, bits, COVERT_KEY_WIDTH).unwrap();
            keygen_ms.push(ms(t));
            family.get_or_insert(f);
        }
        let family = family.unwrap();
        let group: Vec<_> = (0..users).collect();
        // Anamorphic setup for the fixed group: H_S and its per-CRT-slot elimination.
        let mut setup_ms = Vec::new();
        let mut context = None;
        for _ in 0..args.reps {
            let t = Instant::now();
            let c = family.context(&group).unwrap();
            setup_ms.push(ms(t));
            context.get_or_insert(c);
        }
        let context = context.unwrap();
        assert!(context.full_rank());
        let g = context.constraint_count();
        let blocks = bits.div_ceil(D);
        let (pk_bytes, sk_bytes) = packed_key_sizes(&host, &family, 0);
        let (h_elems, sys_elems) = context.stored_field_elements();

        let mut payload_rng = rand::rng();
        let mut m_rbe = Vec::new();
        let mut m_rbe_coins = Vec::new();
        let (mut m_sample, mut m_ab, mut m_enc, mut m_total) =
            (Vec::new(), Vec::new(), Vec::new(), Vec::new());
        let (mut m_dec, mut m_hdec) = (Vec::new(), Vec::new());
        let mut ct_bytes = None;
        let mut checked = 0usize;
        for trial in 0..=args.reps {
            let ordinary = random_ordinary(&mut payload_rng);
            let message: Vec<u8> = (0..bits)
                .map(|_| (payload_rng.next_u32() & 1) as u8)
                .collect();
            let order: [u8; 3] = match trial % 3 {
                0 => [0, 1, 2],
                1 => [1, 2, 0],
                _ => [2, 0, 1],
            };
            for mode in order {
                let (ct, times) = match mode {
                    // The authors' encryption exactly as published: rbe.enc samples its own coins.
                    0 => {
                        let t = Instant::now();
                        let ct = host.enc(&pp, &st, &id, ordinary.clone());
                        (ct, vec![ms(t)])
                    }
                    // The same encryption through explicit coins (control for the split below).
                    1 => {
                        let t = Instant::now();
                        let coins = Coins::sample(&host, args.depth);
                        let ct = coins.encrypt(&host, &pp, &st, &id, ordinary.clone());
                        (ct, vec![ms(t)])
                    }
                    // RBE with the anamorphic broadcast inside.
                    _ => {
                        let t0 = Instant::now();
                        let mut coins = Coins::sample(&host, args.depth);
                        let t1 = Instant::now();
                        let embedded = context.apply(&host, &id, &mut coins, &message).unwrap();
                        let t2 = Instant::now();
                        let ct = coins.encrypt(&host, &pp, &st, &id, ordinary.clone());
                        let t3 = Instant::now();
                        assert!(embedded);
                        let f = |a: Instant, b: Instant| (b - a).as_secs_f64() * 1e3;
                        (ct, vec![f(t0, t1), f(t1, t2), f(t2, t3), f(t0, t3)])
                    }
                };
                // Checks (untimed for encryption; decryption timings recorded separately).
                let (ct, bytes) = roundtrip(&host, &ct);
                assert_eq!(*ct_bytes.get_or_insert(bytes), bytes);
                let t = Instant::now();
                let out = host.dec(&sk, 0, &wit, &ct);
                let dec_ms = ms(t);
                assert_eq!(
                    out.matrix,
                    ordinary.get_entry(0, 0),
                    "ordinary decryption failed"
                );
                if mode == 2 {
                    for &i in &group {
                        let t = Instant::now();
                        let got = family.decode(&host, i, &ct);
                        let hdec = ms(t);
                        assert_eq!(got, message, "hidden decryption failed");
                        if trial > 0 && i == 0 {
                            m_hdec.push(hdec);
                        }
                    }
                }
                checked += 1;
                if trial == 0 {
                    continue; // warm-up round, not recorded
                }
                match mode {
                    0 => {
                        m_rbe.push(times[0]);
                        m_dec.push(dec_ms);
                    }
                    1 => m_rbe_coins.push(times[0]),
                    _ => {
                        m_sample.push(times[0]);
                        m_ab.push(times[1]);
                        m_enc.push(times[2]);
                        m_total.push(times[3]);
                    }
                }
            }
        }
        let rbe_part: Vec<f64> = m_sample.iter().zip(&m_enc).map(|(a, b)| a + b).collect();
        let row = json!({
            "workload": {"depth": args.depth, "recipients": users, "payload_bits": bits,
                          "payload_polynomials_per_recipient": blocks, "constraints_g": g,
                          "branch": branch, "registered_covert_users": population,
                          "root_rank_left": N - g},
            "encryption": {
                "rbe_alone_authors_enc": stats(&m_rbe),
                "rbe_alone_via_coins_control": stats(&m_rbe_coins),
                "rbe_with_ab_total": stats(&m_total),
                "rbe_with_ab_part_broadcast": stats(&m_ab),
                "rbe_with_ab_part_rbe": stats(&rbe_part),
                "rbe_with_ab_part_rbe_coin_sampling": stats(&m_sample),
                "rbe_with_ab_part_rbe_encryption": stats(&m_enc),
            },
            "broadcast_keygen": {"ms_for_all_users": stats(&keygen_ms),
                                  "users": population},
            "broadcast_setup": {"ms": stats(&setup_ms)},
            "decryption": {"ordinary": stats(&m_dec), "hidden_one_recipient": stats(&m_hdec)},
            "sizes_bits": {
                "apk_per_recipient": blocks * N * D * field_bits,
                "apk_per_recipient_packed_measured": pk_bytes * 8,
                "ask_per_recipient": blocks * family.carrier_columns.len() * D * key_bits,
                "ask_per_recipient_packed_measured": sk_bytes * 8,
                "apk_total_registered_users": population * blocks * N * D * field_bits,
                "ask_total_registered_users": population * blocks * family.carrier_columns.len() * D * key_bits,
                "setup_H_S": h_elems * field_bits,
                "setup_precomputed_rref_and_transforms": sys_elems * field_bits,
                "setup_pivot_indices": context.stored_pivots(),
                "host_public_parameters": (K * host.m_a + M_B) * N * D * field_bits,
                "host_state": B * N * D * field_bits,
                "ordinary_public_key": N * D * field_bits,
                "ordinary_secret_key": M_B * D * 16,
                "ciphertext_both_modes": ct_bytes.unwrap() * 8,
            },
            "one_time_ms": {"host_setup": host_setup_ms, "host_keygen": host_keygen_ms,
                             "host_registration": registration_ms},
            "checked_ciphertexts": checked,
        });
        eprintln!(
            "depth {} |S|={users} L={bits} g={g}: RBE {:.1} ms | RBE+AB {:.1} ms (AB {:.3} ms, RBE {:.1} ms) | aKGen {:.2} ms | setup {:.3} ms",
            args.depth,
            row["encryption"]["rbe_alone_authors_enc"]["median_ms"]
                .as_f64()
                .unwrap(),
            row["encryption"]["rbe_with_ab_total"]["median_ms"]
                .as_f64()
                .unwrap(),
            row["encryption"]["rbe_with_ab_part_broadcast"]["median_ms"]
                .as_f64()
                .unwrap(),
            row["encryption"]["rbe_with_ab_part_rbe"]["median_ms"]
                .as_f64()
                .unwrap(),
            row["broadcast_keygen"]["ms_for_all_users"]["median_ms"]
                .as_f64()
                .unwrap(),
            row["broadcast_setup"]["ms"]["median_ms"].as_f64().unwrap(),
        );
        rows.push(row);
        drop(reg);
        drop_db();
    }
    let report = json!({
        "host": {"crate": "scalable-rbe 0.1.1 at 5f1e7416893de7950d231f11b1207dd5bc02924d with the rank-16 joint profile (rbe/implementation/upstream, n16-joint-profile.patch)",
                  "q": Q, "ring_degree": D, "module_rank_n": N, "branches_k": K, "subregistries_b": B,
                  "dropped_bits_root": root_drop, "dropped_bits_other_blocks": PRUNE_CT_BY,
                  "field_bits": field_bits},
        "channel": {"covert_key_width": COVERT_KEY_WIDTH, "max_constraints": MAX_CONSTRAINTS,
                    "min_root_rank": MIN_ROOT_RANK},
        "rayon_threads": rayon::current_num_threads(),
        "reps": args.reps, "warmup_rounds": 1,
        "timing": "wall clock per call; mode order rotates every round; fresh coins for every ciphertext; checks untimed",
        "rows": rows,
    });
    fs::write(&args.output, serde_json::to_vec_pretty(&report).unwrap()).unwrap();
    eprintln!("wrote {}", args.output);
}
