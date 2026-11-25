//! Encryption and decryption benchmark: Epoque (trapdoor KeyGen, one identity) alone
//! versus Epoque with the anamorphic broadcast of Construction 1 inside, plus aKGen,
//! aEnc's group preparation and their sizes.
//!
//!   cargo run --locked --release --bin bench_ab -- --profile full --reps 10 --output epoque.json
//!
//! The split encryption below performs exactly the steps of `HiddenPlan::encrypt` and
//! `encrypt_ordinary` through the public API; a control re-times the library calls.
//! Every timed ciphertext is decrypted with the master shortcut key and with the
//! identity key, and every anamorphic one by each recipient in S with its own key;
//! a single failure aborts the run.
use clap::Parser;
use epoque_shortcut::arithmetic::uniform;
use epoque_shortcut::trapdoor::{TrapdoorProfile, TrapdoorSetup, decrypt_identity, setup_trapdoor};
use epoque_shortcut::*;
use rand::RngExt;
use rand_core::UnwrapErr;
use serde_json::{Value, json};
use std::{fs, time::Instant};

fn ms(t: Instant) -> f64 {
    t.elapsed().as_secs_f64() * 1e3
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

/// Times Epoque encryption and decryption alone and with the anamorphic broadcast inside.
#[derive(Parser)]
struct Args {
    /// Parameter profile.
    #[arg(long, value_parser = ["small", "full"], default_value = "full")]
    profile: String,
    /// Identity to encrypt to: uniformly random in Z_q^n, or the unit identity (H = I).
    #[arg(long, value_parser = ["random", "unit"], default_value = "random")]
    identity: String,
    /// Measured repetitions per workload.
    #[arg(long, default_value_t = 10)]
    reps: usize,
    /// Repetitions of the control, which re-times the library calls.
    #[arg(long, default_value_t = 2)]
    control: usize,
    /// Result file.
    #[arg(long, default_value = "epoque-ab-benchmark.json")]
    output: String,
    /// Workloads as RECIPIENTS:BITS, separated by commas. aKGen makes one key per
    /// recipient, so N = |S|.
    #[arg(long, value_delimiter = ',', value_parser = parse_workload, default_value = "1:1,2:2,4:2")]
    workloads: Vec<(usize, usize)>,
}

fn parse_workload(workload: &str) -> std::result::Result<(usize, usize), String> {
    let (recipients, bits) = workload
        .split_once(':')
        .ok_or("workloads use RECIPIENTS:BITS")?;
    let number = |x: &str| x.parse::<usize>().map_err(|e| format!("{workload}: {e}"));
    Ok((number(recipients)?, number(bits)?))
}

fn main() -> std::result::Result<(), Box<dyn std::error::Error>> {
    let Args {
        profile,
        identity,
        reps,
        control,
        output,
        workloads,
    } = Args::parse();
    if cfg!(debug_assertions) {
        return Err("run benchmarks with --release".into());
    }
    let tp = match profile.as_str() {
        "small" => TrapdoorProfile::small(),
        "full" => TrapdoorProfile::full(),
        _ => return Err("profile must be small or full".into()),
    };
    let p = tp.profile();
    let (n, m, q, jb) = (p.n(), p.m(), p.q(), p.ordinary_bits());
    let field_bits = (64 - (q - 1).leading_zeros()) as usize;
    let mut rng = UnwrapErr(OsBits::default());

    let t = Instant::now();
    let generated = setup_trapdoor(&tp, &mut rng, |_, _| {})?;
    let host_setup_ms = ms(t);
    eprintln!(
        "host setup (trapdoors, shortcut keys, public matrices): {:.1} s",
        host_setup_ms / 1e3
    );
    let id = if identity == "unit" {
        generated.setup.public.unit_identity()
    } else {
        let coordinates: Vec<u64> = (0..n).map(|_| rng.random_range(0..q)).collect();
        generated.setup.public.identity(&coordinates)?
    };
    let t = Instant::now();
    let id_key = generated
        .master
        .extract(&generated.setup.public, &id, &mut rng)?;
    let extraction_ms = ms(t);
    // T_A0 is not needed after extraction: whoever holds it can read every hidden message.
    let TrapdoorSetup { setup, master, .. } = generated;
    drop(master);
    let pk = &setup.public;

    let mut rows = Vec::new();
    for &(size, bits) in &workloads {
        let g = size * bits;
        let group: Vec<usize> = (0..size).collect();
        // aKGen (binary ask_i, then apk = A0 k̂) and aEnc's preparation for S.
        let (mut kg_secret, mut kg_public, mut setup_ms) = (Vec::new(), Vec::new(), Vec::new());
        let mut keep = None;
        for _ in 0..reps.max(1) {
            let t0 = Instant::now();
            let keys = (0..size)
                .map(|_| covert_secret_keygen(p, bits, &mut rng))
                .collect::<Result<Vec<_>>>()?;
            let t1 = Instant::now();
            let apk = CovertPublicKey::from_secret_keys(pk, &keys)?;
            let t2 = Instant::now();
            let plan = prepare_hidden(pk, &apk, &group)?;
            let t3 = Instant::now();
            kg_secret.push((t1 - t0).as_secs_f64() * 1e3);
            kg_public.push((t2 - t1).as_secs_f64() * 1e3);
            setup_ms.push((t3 - t2).as_secs_f64() * 1e3);
            assert!(plan.sampler().full_rank(), "H_S is rank deficient");
            keep.get_or_insert((keys, apk));
        }
        let (keys, apk) = keep.unwrap();
        let plan = prepare_hidden(pk, &apk, &group)?;
        let sampler = plan.sampler();

        let (mut e_s, mut e_err, mut e_prod, mut e_tot) = (vec![], vec![], vec![], vec![]);
        let (mut a_s, mut a_err, mut a_prod, mut a_tot) = (vec![], vec![], vec![], vec![]);
        let (mut dec_shortcut, mut dec_identity, mut hdec_all) = (vec![], vec![], vec![]);
        let mut checked = 0;
        for trial in 0..=reps {
            let ordinary: Vec<u8> = (0..jb).map(|_| rng.random_range(0..2u8)).collect();
            let hidden: Vec<u8> = (0..bits).map(|_| rng.random_range(0..2u8)).collect();
            let modes = if trial % 2 == 0 {
                [false, true]
            } else {
                [true, false]
            };
            for anamorphic in modes {
                let t0 = Instant::now();
                let s = if anamorphic {
                    // The broadcast: s uniform subject to H_S^T s = floor(q/2) (am, ..., am).
                    let rhs: Vec<u64> = (0..size)
                        .flat_map(|_| hidden.iter().map(|&b| u64::from(b) * (q / 2)))
                        .collect();
                    sampler.sample(&rhs, &mut rng)?
                } else {
                    uniform(n, q, &mut rng)?
                };
                let t1 = Instant::now();
                let errors = sample_errors(p, &mut rng)?;
                let t2 = Instant::now();
                let ct = encrypt_with_coins(pk, &id, &ordinary, &s, &errors)?;
                let t3 = Instant::now();
                let f = |a: Instant, b: Instant| (b - a).as_secs_f64() * 1e3;
                let t = Instant::now();
                assert_eq!(decrypt_ordinary(&setup.shortcut, &ct)?, ordinary);
                let d_shortcut = ms(t);
                let t = Instant::now();
                assert_eq!(decrypt_identity(&id_key, &ct)?, ordinary);
                let d_identity = ms(t);
                if anamorphic {
                    let t = Instant::now();
                    let received = group
                        .iter()
                        .map(|&i| decrypt_covert(&keys[i], &ct))
                        .collect::<Result<Vec<_>>>()?;
                    let h = ms(t);
                    assert!(
                        received.iter().all(|bits| bits == &hidden),
                        "hidden decryption failed"
                    );
                    if trial > 0 {
                        hdec_all.push(h);
                    }
                }
                checked += 1;
                if trial == 0 {
                    continue;
                }
                let (vs, ve, vp, vt) = if anamorphic {
                    (&mut a_s, &mut a_err, &mut a_prod, &mut a_tot)
                } else {
                    dec_shortcut.push(d_shortcut);
                    dec_identity.push(d_identity);
                    (&mut e_s, &mut e_err, &mut e_prod, &mut e_tot)
                };
                vs.push(f(t0, t1));
                ve.push(f(t1, t2));
                vp.push(f(t2, t3));
                vt.push(f(t0, t3));
            }
        }
        // Control: the library entry points themselves.
        let (mut c_ord, mut c_ana) = (vec![], vec![]);
        for _ in 0..control {
            let ordinary: Vec<u8> = (0..jb).map(|_| rng.random_range(0..2u8)).collect();
            let hidden: Vec<u8> = (0..bits).map(|_| rng.random_range(0..2u8)).collect();
            let t = Instant::now();
            let ct = encrypt_ordinary(pk, &id, &ordinary, &mut rng)?;
            c_ord.push(ms(t));
            assert_eq!(decrypt_ordinary(&setup.shortcut, &ct)?, ordinary);
            let t = Instant::now();
            let ct = plan.encrypt(&id, &ordinary, &hidden, &mut rng)?;
            c_ana.push(ms(t));
            for &i in &group {
                assert_eq!(decrypt_covert(&keys[i], &ct)?, hidden);
            }
            checked += 2;
        }
        let epoque_part: Vec<f64> = a_err.iter().zip(&a_prod).map(|(x, y)| x + y).collect();
        let kg_total: Vec<f64> = kg_secret
            .iter()
            .zip(&kg_public)
            .map(|(x, y)| x + y)
            .collect();
        let hdec_each: Vec<f64> = hdec_all.iter().map(|x| x / size as f64).collect();
        let row = json!({
            "workload": {"recipients": size, "payload_bits": bits, "constraints_g": g,
                          "lwe_dimension_left": n - g},
            "encryption": {
                "epoque_alone": stats(&e_tot),
                "epoque_alone_part_uniform_s": stats(&e_s),
                "epoque_alone_part_errors_and_signs": stats(&e_err),
                "epoque_alone_part_matrix_products": stats(&e_prod),
                "epoque_with_ab_total": stats(&a_tot),
                "epoque_with_ab_part_broadcast": stats(&a_s),
                "epoque_with_ab_part_epoque": stats(&epoque_part),
                "epoque_with_ab_part_epoque_errors_and_signs": stats(&a_err),
                "epoque_with_ab_part_epoque_matrix_products": stats(&a_prod),
                "control_library_encrypt_ordinary": if control > 0 { stats(&c_ord) } else { json!(null) },
                "control_library_plan_encrypt": if control > 0 { stats(&c_ana) } else { json!(null) },
            },
            "broadcast_keygen": {"ms_total": stats(&kg_total),
                                  "ms_secret_keys": stats(&kg_secret),
                                  "ms_public_vectors_A0k": stats(&kg_public),
                                  "users": size},
            "broadcast_setup": {"ms": stats(&setup_ms)},
            "decryption": {"ordinary_shortcut_key": stats(&dec_shortcut),
                           "ordinary_identity_key": stats(&dec_identity),
                           "hidden_all_recipients": stats(&hdec_all),
                           "hidden_per_recipient": stats(&hdec_each)},
            "sizes_bits": {
                "ask_per_recipient": bits * m,
                "apk_per_recipient": bits * n * field_bits,
                "ask_total": g * m,
                "apk_total": g * n * field_bits,
                "setup_H_S": g * n * field_bits,
                "setup_precomputed_rref_and_transform": (g * n + g * g) * field_bits,
                "setup_pivot_and_free_indices": n,
                "ciphertext_both_modes": p.ciphertext_bytes() * 8,
                "host_public_key": (3 * n * m + n * jb) * field_bits,
                "host_identity": n * field_bits,
                "host_shortcut_key_entries": 2 * m * jb,
                "host_identity_key_entries": 2 * m * jb,
            },
            "checked_ciphertexts": checked,
        });
        eprintln!(
            "|S|={size} L={bits} g={g}: Epoque {:.0} ms | Epoque+AB {:.0} ms (AB {:.3} ms, Epoque {:.0} ms) | aKGen {:.1} ms | setup {:.2} ms",
            row["encryption"]["epoque_alone"]["median_ms"]
                .as_f64()
                .unwrap(),
            row["encryption"]["epoque_with_ab_total"]["median_ms"]
                .as_f64()
                .unwrap(),
            row["encryption"]["epoque_with_ab_part_broadcast"]["median_ms"]
                .as_f64()
                .unwrap(),
            row["encryption"]["epoque_with_ab_part_epoque"]["median_ms"]
                .as_f64()
                .unwrap(),
            row["broadcast_keygen"]["ms_total"]["median_ms"]
                .as_f64()
                .unwrap(),
            row["broadcast_setup"]["ms"]["median_ms"].as_f64().unwrap(),
        );
        rows.push(row);
        fs::write(
            &output,
            serde_json::to_vec_pretty(&json!({"partial": true, "rows": rows}))?,
        )?;
    }
    let report = json!({
        "profile": {"name": profile, "n": n, "m": m, "q": q.to_string(), "J": jb,
                     "max_constraints": p.maximum_constraints(), "C": p.c_bound(),
                     "key_width_s": tp.width(), "gadget_base": tp.base(),
                     "identity": {"kind": identity, "first_coordinates": id.vector()[..n.min(8)]},
                     "field_bits": field_bits},
        "host_setup_ms_single_run": host_setup_ms,
        "identity_extraction_ms_single_run": extraction_ms,
        "reps": reps, "warmup_rounds": 1, "control_runs": control,
        "timing": "wall clock; mode order alternates; fresh coins per ciphertext; OS randomness; checks untimed",
        "rows": rows,
    });
    fs::write(&output, serde_json::to_vec_pretty(&report)?)?;
    eprintln!("wrote {output}");
    Ok(())
}
