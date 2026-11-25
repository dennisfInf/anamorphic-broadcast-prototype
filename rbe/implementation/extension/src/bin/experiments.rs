//! Experiments: the average cost of the authors' RBE algorithms and of the anamorphic
//! broadcast inside, over repeated runs, for several groups of recipients.
//!
//! Run in a disposable copy of this package (setup creates and this driver clears
//! `registry_db`), single-threaded as the other RBE benchmarks:
//!   RAYON_NUM_THREADS=1 cargo run --locked --release --bin experiments -- \
//!       --reps 50 --output rbe-experiments.json
//!
//! Ordinary algorithms: setup is the host's Setup, KeyGen and the registration (Upd)
//! of the receiver, encryption is the authors' `RBE::enc` and decryption `RBE::dec`.
//! Their cost is fixed by the host's parameters (B = 30 subregistries for at most 2^30
//! users, identity depth 162). Witness generation is reported alongside.
//!
//! The host's module rank N is set by RBE_RANK at build time (default 16, the joint
//! profile). `RBE_RANK=7 + |S|` keeps the authors' rank 7 free for |S| recipients, as
//! `experiments/run_all.sh` does for one to six. At RBE_RANK=7, the authors' host itself,
//! every constraint takes one of its seven ranks, so those runs measure cost only.
//!
//! Anamorphic algorithms, for several group sizes |S| (--groups, default MAX_CONSTRAINTS
//! down to 1, e.g. 4,3,2,1 at rank 16), each recipient with a payload of --bits bits
//! (default 256, one payload polynomial; 128 bits cost the same, since they also take one
//! ring constraint): aKGen for exactly the |S| recipients, the group setup, aEnc, and aDec
//! by one fixed recipient (the first of S). Every ciphertext is checked untimed: ordinary
//! decryption and hidden decryption by every recipient in S. Every result row records
//! |S|, L and the equations they use.
//!
//! Enc and aEnc are timed in the same rounds: after --warmup untimed rounds, each round
//! encrypts fresh messages once with `RBE::enc` and once with aEnc, Enc first in even
//! rounds and aEnc first in odd ones, so that drift of the machine (clock, temperature,
//! other load) falls on both alike. aEnc draws every coin `RBE::enc` draws except the
//! Gaussian r_0, which its broadcast step (`Context::apply`) computes instead. The step in
//! which the two differ is also timed alone, in --step-reps paired iterations of the
//! Gaussian r_0 and the constrained r_0 (`step`); their difference is aEnc's overhead,
//! and it may be negative. Both are single-threaded, so with RAYON_NUM_THREADS=1 it is
//! also the wall-clock overhead. Each row records Enc of its own rounds and aEnc minus Enc
//! round by round (`encryption_difference`) as a check.
//!
//! Loops of fast operations (aKGen, the group setup, the encryption rounds, decryption)
//! run at least --keygen-reps or --reps times and then until --min-time seconds have
//! passed, at most MAX_RUNS times. Dec and aDec are timed in a loop of their own on the
//! last rounds' ciphertexts. The host's ordinary encryption and decryption are those of
//! all trades.
//!
//! Sizes (`sizes_bytes`): ordinary public key y = B x + e* and secret key x, public
//! parameters (A, B), registry state, ciphertexts of both modes as serialized, the packed
//! anamorphic public and secret keys of one user, and the group matrix H_S.
use rand::Rng;
use rbe_anamorphic_review::*;
use scalable_rbe::{
    mat::Mat,
    rbe::{B, D, K, M_B, N, Q, RBE},
    utils::drop_db,
};
use serde_json::{Value, json};
use std::{fs, hint::black_box, time::Instant};

/// Most runs of one timed loop that --min-time extends.
const MAX_RUNS: usize = 20_000;
/// Ciphertext pairs of the last rounds that the decryption loop cycles through.
const POOL: usize = 2;

fn ms(t: Instant) -> f64 {
    t.elapsed().as_secs_f64() * 1e3
}

/// Milliseconds from `start` to `end`.
fn span(start: Instant, end: Instant) -> f64 {
    (end - start).as_secs_f64() * 1e3
}

fn mean(xs: &[f64]) -> f64 {
    xs.iter().sum::<f64>() / xs.len() as f64
}

fn median(xs: &[f64]) -> f64 {
    let mut s = xs.to_vec();
    s.sort_by(f64::total_cmp);
    (s[(s.len() - 1) / 2] + s[s.len() / 2]) / 2.0
}

/// Whether a timed loop goes on after `done` runs: until `reps` runs, then while less
/// than `min_time` seconds have passed since `start`, at most MAX_RUNS runs.
fn more(done: usize, reps: usize, start: Instant, min_time: f64) -> bool {
    done < reps || (done < MAX_RUNS && start.elapsed().as_secs_f64() < min_time)
}

/// Mean, sample standard deviation, median, extremes and all samples, in milliseconds.
fn stats(xs: &[f64]) -> Value {
    let mut s = xs.to_vec();
    s.sort_by(f64::total_cmp);
    let mean = s.iter().sum::<f64>() / s.len() as f64;
    let sd = if s.len() > 1 {
        (s.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (s.len() - 1) as f64).sqrt()
    } else {
        0.0
    };
    let mid = s.len() / 2;
    let median = if s.len() % 2 == 1 {
        s[mid]
    } else {
        (s[mid - 1] + s[mid]) / 2.0
    };
    json!({"runs": s.len(), "mean_ms": mean, "sd_ms": sd, "median_ms": median,
           "min_ms": s[0], "max_ms": s[s.len() - 1], "samples_ms": xs})
}

/// The sum of series run by run (as many runs as the shortest series).
fn paired_sum(series: &[&[f64]]) -> Vec<f64> {
    let runs = series.iter().map(|s| s.len()).min().unwrap_or(0);
    (0..runs)
        .map(|i| series.iter().map(|s| s[i]).sum())
        .collect()
}

/// `a - b` run by run.
fn paired_difference(a: &[f64], b: &[f64]) -> Vec<f64> {
    a.iter().zip(b).map(|(x, y)| x - y).collect()
}

struct Args {
    reps: usize,
    warmup: usize,
    keygen_reps: Option<usize>,
    step_reps: usize,
    min_time: f64,
    groups: Vec<usize>,
    bits: Vec<usize>,
    depth: usize,
    output: String,
}

fn parse() -> Result<Args, String> {
    let mut a = Args {
        reps: 50,
        warmup: 2,
        keygen_reps: None,
        step_reps: 2000,
        min_time: 10.0,
        groups: (1..=MAX_CONSTRAINTS).rev().collect(),
        bits: vec![D],
        depth: 162,
        output: "rbe-experiments.json".into(),
    };
    let mut it = std::env::args().skip(1);
    while let Some(flag) = it.next() {
        let value = it.next().ok_or(format!("missing value for {flag}"))?;
        let number = |v: &str| v.parse::<usize>().map_err(|e| format!("{flag}: {e}"));
        let list = |v: &str| v.split(',').map(number).collect::<Result<Vec<_>, _>>();
        match flag.as_str() {
            "--reps" => a.reps = number(&value)?,
            "--warmup" => a.warmup = number(&value)?,
            "--keygen-reps" => a.keygen_reps = Some(number(&value)?),
            "--step-reps" => a.step_reps = number(&value)?,
            "--min-time" => {
                a.min_time = value.parse::<f64>().map_err(|e| format!("{flag}: {e}"))?
            }
            "--groups" => a.groups = list(&value)?,
            "--bits" => a.bits = list(&value)?,
            "--depth" => a.depth = number(&value)?,
            "--output" => a.output = value,
            _ => {
                return Err(format!(
                    "unknown option {flag}; use --reps --warmup --keygen-reps --step-reps --min-time --groups 4,3,2,1 --bits 256 --depth --output"
                ));
            }
        }
    }
    if a.reps == 0
        || a.keygen_reps == Some(0)
        || a.step_reps == 0
        || !(a.min_time.is_finite() && a.min_time >= 0.0)
        || a.groups.is_empty()
        || a.groups.iter().any(|&s| s == 0 || s > MAX_CONSTRAINTS)
        || a.bits.is_empty()
        || a.bits.contains(&0)
        || !(1..=162).contains(&a.depth)
    {
        return Err(format!(
            "need reps, keygen reps and step reps >= 1, min time >= 0, group sizes in 1..={MAX_CONSTRAINTS}, bits >= 1 and depth in 1..=162"
        ));
    }
    if let Some((size, bits)) = a
        .groups
        .iter()
        .flat_map(|&size| a.bits.iter().map(move |&bits| (size, bits)))
        .find(|&(size, bits)| size * bits.div_ceil(D) > MAX_CONSTRAINTS)
    {
        return Err(format!(
            "|S| = {size} with L = {bits} needs more than the {MAX_CONSTRAINTS} ring constraints of this host"
        ));
    }
    Ok(a)
}

fn random_ordinary(rng: &mut impl Rng) -> Mat {
    Mat {
        matrix: (0..B * D).map(|_| (rng.next_u32() & 1) as u64).collect(),
        d: D,
        nr_rows: 1,
        nr_columns: B,
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    if cfg!(debug_assertions) {
        return Err("run experiments with --release".into());
    }
    let args = parse()?;
    let (reps, warmup, depth) = (args.reps, args.warmup, args.depth);
    let (step_reps, min_time) = (args.step_reps, args.min_time);
    let keygen_reps = args.keygen_reps.unwrap_or(reps);
    // Every recipient gets a payload of each length in --bits; up to 256 bits take one
    // payload polynomial, one of the MAX_CONSTRAINTS ring equations.
    let configs: Vec<(&str, usize, usize)> = args
        .groups
        .iter()
        .flat_map(|&size| {
            args.bits.iter().map(move |&bits| {
                let length = match bits {
                    D => "full",
                    b if 2 * b == D => "half",
                    _ => "part",
                };
                (length, size, bits)
            })
        })
        .collect();
    let branch = 0;
    let id = format!("{branch}{}", &"012".repeat(depth.div_ceil(3))[..depth - 1]);
    let field_bits = 64 - (Q - 1).leading_zeros() as usize;
    let mut report = json!({
        "scheme": "rbe",
        "complete": false,
        "config": {"reps": reps, "keygen_reps": keygen_reps,
                   "trades": configs.iter().map(|c| json!({"recipients": c.1, "bits": c.2, "length": c.0})).collect::<Vec<_>>(),
                   "identity_depth": depth, "branch": branch, "rank": N,
                   "max_constraints": MAX_CONSTRAINTS,
                   "warmup_rounds": warmup, "step_reps": step_reps,
                   "min_time_s": min_time, "max_runs": MAX_RUNS,
                   "timing": "per trade, --warmup untimed rounds, then at least --reps measured rounds and until --min-time seconds; every round encrypts fresh messages with Enc and with aEnc, Enc first in even rounds and aEnc first in odd ones; the differing step alone in --step-reps paired iterations, alternating which goes first; wall clock; checks untimed"},
        "environment": {"os": std::env::consts::OS, "arch": std::env::consts::ARCH,
                        "rayon_threads": rayon::current_num_threads()},
        "parameters": {"q": Q, "ring_degree": D, "module_rank_n": N, "branches_k": K,
                       "subregistries_b": B, "max_registered_users": 1u64 << B,
                       "covert_key_width": COVERT_KEY_WIDTH, "field_bits": field_bits},
        "sizes_bytes": {"ordinary_public_key": N * D * field_bits / 8,
                        "ordinary_secret_key": M_B * D * 2,
                        "state": B * N * D * field_bits / 8},
        "definitions": {
            "ordinary.setup": "RBE::setup + RBE::keygen + RBE::update (registration of the receiver)",
            "ordinary.encryption": "RBE::enc (the authors' encryption, sampling its own coins), in the encryption rounds of all trades",
            "ordinary.decryption": "RBE::dec with the receiver's witness, repeated on the last rounds' ordinary ciphertexts",
            "anamorphic.keygen": "aKGen for the |S| recipients only: covert keys and public columns b = C k (A read from the explicit public parameters)",
            "anamorphic.group_setup": "Family::context for the group S (reads only S's public columns)",
            "anamorphic.setup": "keygen + group_setup, run by run (as many runs as the shorter series)",
            "anamorphic.encryption": "Coins::sample_without_root + Context::apply + host encryption: every coin of RBE::enc except the Gaussian r_0, which the broadcast step computes instead",
            "anamorphic.ordinary_encryption": "RBE::enc in the same rounds as aEnc",
            "anamorphic.encryption_difference": "aEnc minus Enc of the same round",
            "anamorphic.broadcast_step": "Context::apply inside aEnc: the target and the CRT-slot solves for r_0",
            "anamorphic.step": "the differing step alone in --step-reps paired iterations: replaced = the Gaussian r_0 of Enc (Coins::sample_root), broadcast = the constrained r_0 of aEnc (Context::sample), overhead = broadcast minus replaced; both single-threaded, so with RAYON_NUM_THREADS=1 the overhead is aEnc's wall-clock overhead, while with more threads Enc draws r_0 in parallel with its other coins",
            "anamorphic.decryption": "Family::decode by one fixed recipient of S (the first), repeated on the last rounds' anamorphic ciphertexts",
            "anamorphic.ordinary_decryption": "RBE::dec of the trade's last ordinary ciphertexts, repeated"},
    });
    let save = |report: &Value| -> std::io::Result<()> {
        fs::write(
            &args.output,
            serde_json::to_vec_pretty(report).expect("json"),
        )
    };
    save(&report)?;
    let mut payload_rng = rand::rng();

    // Ordinary setup: a fresh registry per run, and the last one is kept.
    let host = RBE::init(false);
    report["sizes_bytes"]["public_parameters"] =
        json!((K * host.m_a + M_B) * N * D * field_bits / 8);
    let (mut setup_ms, mut keygen_ms, mut update_ms, mut witness_ms) =
        (Vec::new(), Vec::new(), Vec::new(), Vec::new());
    let mut kept = None;
    for _ in 0..reps {
        drop(kept.take());
        drop_db();
        let t = Instant::now();
        let (pp, _st0, reg) = host.setup();
        setup_ms.push(ms(t));
        let t = Instant::now();
        let (pk, sk) = host.keygen(&pp);
        keygen_ms.push(ms(t));
        let t = Instant::now();
        let st = host.update(&reg, &pp, 0, &id, &pk);
        update_ms.push(ms(t));
        let t = Instant::now();
        let wit = host
            .witness_gen(&reg, 0, &id)
            .ok_or("no witness for the receiver")?;
        witness_ms.push(ms(t));
        kept = Some((pp, reg, sk, st, wit));
    }
    let (pp, reg, sk, st, wit) = kept.expect("at least one setup");
    let setup_total = paired_sum(&[&setup_ms, &keygen_ms, &update_ms]);
    // Ordinary encryption and decryption are timed with aEnc and aDec below, and added
    // here after every trade.
    report["ordinary"] = json!({
        "setup": stats(&setup_total),
        "host_setup": stats(&setup_ms),
        "host_keygen": stats(&keygen_ms),
        "registration": stats(&update_ms),
        "witness_generation": stats(&witness_ms),
    });
    save(&report)?;
    eprintln!("ordinary: setup {:.1} ms", mean(&setup_total));

    // For every group and length: aKGen for exactly the recipients, the group setup, the
    // encryption rounds (Enc and aEnc), the differing step alone, and decryption.
    let a = &pp.0;
    let mut rows = Vec::new();
    let (mut all_enc_ms, mut all_dec_ms) = (Vec::new(), Vec::new());
    let mut checked = 0usize;
    for &(length, size, bits) in &configs {
        let mut akgen_ms = Vec::new();
        let mut kept: Option<Family> = None;
        let start = Instant::now();
        while more(akgen_ms.len(), keygen_reps, start, min_time) {
            drop(kept.take());
            let t = Instant::now();
            let family =
                Family::generate_with_matrix(&host, a, branch, size, bits, COVERT_KEY_WIDTH)?;
            akgen_ms.push(ms(t));
            kept = Some(family);
        }
        let family = kept.expect("at least one aKGen run");
        let (apk_row, _) = packed_key_sizes(&host, &family, 0);
        if rows.is_empty() {
            let (apk_bytes, ask_bytes) = packed_key_sizes(&host, &family, 0);
            report["sizes_bytes"]["anamorphic_public_key_per_user"] = json!(apk_bytes);
            report["sizes_bytes"]["anamorphic_secret_key_per_user"] = json!(ask_bytes);
        }
        let group: Vec<usize> = (0..size).collect();
        let recipient = group[0];
        let mut group_ms = Vec::new();
        let start = Instant::now();
        while more(group_ms.len(), reps, start, min_time) {
            let t = Instant::now();
            let context = family.context(&group)?;
            group_ms.push(ms(t));
            if !context.full_rank() {
                return Err("the group's constraints are rank deficient".into());
            }
        }
        let context = family.context(&group)?;
        let (h_elements, _) = context.stored_field_elements();
        if rows.is_empty() {
            report["sizes_bytes"]["anamorphic_group_matrix"] = json!(h_elements * field_bits / 8);
        }
        // Every round encrypts fresh messages once with Enc and once with aEnc, Enc first
        // in even rounds and aEnc first in odd ones, so that drift of the machine falls on
        // both alike. Timing them in separate blocks let that drift decide the sign of
        // aEnc - Enc, which is far below the run-to-run spread of one encryption.
        let (mut enc_ms, mut aenc_ms, mut step_ms) = (Vec::new(), Vec::new(), Vec::new());
        let mut pool = Vec::new();
        let start = Instant::now();
        let mut round = 0;
        while round < warmup || more(round - warmup, reps, start, min_time) {
            let ordinary = random_ordinary(&mut payload_rng);
            let message: Vec<u8> = (0..bits)
                .map(|_| (payload_rng.next_u32() & 1) as u8)
                .collect();
            let (mut ordinary_ct, mut anamorphic_ct) = (None, None);
            let (mut enc, mut aenc, mut step, mut embedded) = (0.0, 0.0, 0.0, false);
            let order = if round % 2 == 0 {
                [false, true]
            } else {
                [true, false]
            };
            for anamorphic in order {
                if anamorphic {
                    // aEnc: the host's coins as RBE::enc samples them, except r_0, which the
                    // broadcast step sets to a solution of the group's constraints, then the
                    // host's encryption with these coins.
                    let t0 = Instant::now();
                    let mut coins = Coins::sample_without_root(&host, depth);
                    let t1 = Instant::now();
                    embedded = context.apply(&host, &id, &mut coins, &message)?;
                    let t2 = Instant::now();
                    let ct = coins.encrypt(&host, &pp, &st, &id, ordinary.clone());
                    let t3 = Instant::now();
                    aenc = span(t0, t3);
                    step = span(t1, t2);
                    anamorphic_ct = Some(ct);
                } else {
                    let t0 = Instant::now();
                    let ct = host.enc(&pp, &st, &id, ordinary.clone());
                    enc = span(t0, Instant::now());
                    ordinary_ct = Some(ct);
                }
            }
            let ordinary_ct = ordinary_ct.expect("Enc ran in this round");
            let anamorphic_ct = anamorphic_ct.expect("aEnc ran in this round");
            if host.dec(&sk, 0, &wit, &ordinary_ct).matrix != ordinary.get_entry(0, 0)
                || !embedded
                || group
                    .iter()
                    .any(|&i| family.decode(&host, i, &anamorphic_ct) != message)
                || host.dec(&sk, 0, &wit, &anamorphic_ct).matrix != ordinary.get_entry(0, 0)
            {
                return Err("ordinary or anamorphic decryption failed".into());
            }
            if round == 0 {
                report["sizes_bytes"]["ordinary_ciphertext"] =
                    json!(roundtrip(&host, &ordinary_ct).1);
                report["sizes_bytes"]["anamorphic_ciphertext"] =
                    json!(roundtrip(&host, &anamorphic_ct).1);
            }
            checked += 2;
            if round >= warmup {
                enc_ms.push(enc);
                aenc_ms.push(aenc);
                step_ms.push(step);
            }
            pool.push((ordinary_ct, anamorphic_ct));
            if pool.len() > POOL {
                pool.remove(0);
            }
            round += 1;
        }
        let difference = paired_difference(&aenc_ms, &enc_ms);
        // Dec by the receiver and aDec by one fixed recipient, many times on the last
        // rounds' ciphertexts (all of them were checked above).
        let (mut dec_ms, mut adec_ms) = (Vec::new(), Vec::new());
        let start = Instant::now();
        while more(dec_ms.len(), reps, start, min_time) {
            let (ordinary_ct, anamorphic_ct) = &pool[dec_ms.len() % pool.len()];
            let t = Instant::now();
            black_box(host.dec(&sk, 0, &wit, ordinary_ct));
            dec_ms.push(ms(t));
            let t = Instant::now();
            black_box(family.decode(&host, recipient, anamorphic_ct));
            adec_ms.push(ms(t));
        }
        // The step in which Enc and aEnc differ, alone: the Gaussian r_0 that Enc draws and
        // the constrained r_0 that aEnc computes instead, alternating which goes first.
        let (mut replaced_alone, mut broadcast_alone) = (Vec::new(), Vec::new());
        for i in 0..warmup + step_reps {
            let message: Vec<u8> = (0..bits)
                .map(|_| (payload_rng.next_u32() & 1) as u8)
                .collect();
            let order = if i % 2 == 0 {
                [false, true]
            } else {
                [true, false]
            };
            for broadcast in order {
                let t = Instant::now();
                if broadcast {
                    black_box(context.sample(&host, &message)?);
                    if i >= warmup {
                        broadcast_alone.push(ms(t));
                    }
                } else {
                    black_box(Coins::sample_root(&host));
                    if i >= warmup {
                        replaced_alone.push(ms(t));
                    }
                }
            }
        }
        let step_overhead = paired_difference(&broadcast_alone, &replaced_alone);
        all_enc_ms.extend_from_slice(&enc_ms);
        all_dec_ms.extend_from_slice(&dec_ms);
        report["ordinary"]["encryption"] = stats(&all_enc_ms);
        report["ordinary"]["decryption"] = stats(&all_dec_ms);
        let row = json!({
            "length": length,
            "role": "sweep",
            "recipients": size,
            "bits": bits,
            "constraints": size * bits.div_ceil(D),
            "rank": N,
            "root_rank_left": N - size * bits.div_ceil(D),
            "decryption_recipient": recipient,
            "apk_bytes_per_recipient": apk_row,
            "group_matrix_bytes": h_elements * field_bits / 8,
            "keygen": stats(&akgen_ms),
            "group_setup": stats(&group_ms),
            "setup": stats(&paired_sum(&[&akgen_ms, &group_ms])),
            "encryption": stats(&aenc_ms),
            "ordinary_encryption": stats(&enc_ms),
            "encryption_difference": stats(&difference),
            "broadcast_step": stats(&step_ms),
            "step": {"iterations": step_reps,
                     "replaced": stats(&replaced_alone),
                     "broadcast": stats(&broadcast_alone),
                     "overhead": stats(&step_overhead)},
            "decryption": stats(&adec_ms),
            "ordinary_decryption": stats(&dec_ms),
        });
        eprintln!(
            "rank {N}, |S| = {size}, L = {bits}: aKGen {:.2} ms, group setup {:.3} ms, Enc {:.1} ms, aEnc {:.1} ms ({} rounds, median aEnc - Enc {:+.1} ms); step alone (medians): Gaussian r_0 {:.1} us, constrained r_0 {:.1} us, overhead {:+.1} us; Dec {:.2} ms, aDec {:.3} ms",
            mean(&akgen_ms),
            mean(&group_ms),
            mean(&enc_ms),
            mean(&aenc_ms),
            enc_ms.len(),
            median(&difference),
            1e3 * median(&replaced_alone),
            1e3 * median(&broadcast_alone),
            1e3 * median(&step_overhead),
            mean(&dec_ms),
            mean(&adec_ms)
        );
        rows.push(row);
        report["anamorphic"] = json!(rows);
        report["checks"] = json!({"checked_ciphertexts": checked});
        save(&report)?;
    }
    report["complete"] = json!(true);
    save(&report)?;
    drop(reg);
    drop_db();
    eprintln!("wrote {}", args.output);
    Ok(())
}
