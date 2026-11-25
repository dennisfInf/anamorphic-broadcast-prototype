//! Experiments: the average cost of Epoque's ordinary algorithms and of the anamorphic
//! broadcast of Construction 1, over repeated runs, across the number of equations
//! d = |S| L that a group's message imposes.
//!
//!   cargo run --locked --release --bin experiments -- --reps 50 \
//!       --output epoque-experiments.json
//!
//! Profiles: `full32` (the default) is n = 1424, m = 53926, q = 220301035189, J = 192
//! (the 192-bit ballots of Epoque's own benchmarks) with g = 64 constraints, so two
//! hidden bits reach 32 recipients (128-bit core-SVP
//! estimate for dimension n - g, derived by tools/trapdoor_parameters.py). `full` is the
//! maintained n = 1360 preset (g = 8, 4 recipients of two bits) and `small` the test
//! profile.
//!
//! Ordinary algorithms (run once per repetition): setup is Epoque's KeyGen with the
//! master shortcut key (`setup_trapdoor`), encryption is `encrypt_ordinary` for one
//! identity and decryption uses the master shortcut key. Identity-key extraction and
//! decryption with the identity key are reported alongside.
//!
//! Anamorphic algorithms, for the trades |S|:L of --trades and --checks: aKGen for
//! exactly the |S| recipients, the group setup, aEnc, and aDec by one fixed recipient (the
//! first of S). Every sender-side cost depends on d = |S| L only and aDec on L only, so
//! the default sweep keeps |S| = 2 and doubles L from 1 up to g/2 (d = 2, 4, ..., g), and
//! the default checks repeat d = g with two other splits (g/2 recipients of 2 bits, and
//! |S| close to L). --groups instead runs, for every |S|, the longest L that fits and half
//! of it, as before the sweep. Every ciphertext is checked untimed: ordinary decryption
//! with both keys and hidden decryption by every recipient in S. Every result row records
//! |S|, L, d and whether it belongs to the sweep or the checks.
//!
//! Enc and aEnc are timed in the same rounds: after --warmup untimed rounds, each round
//! encrypts fresh messages once with Enc and once with aEnc, Enc first in even rounds and
//! aEnc first in odd ones, so that drift of the machine (clock, temperature, other load)
//! falls on both alike. Both run the library's steps through its public API, so that the
//! one step in which they differ is timed inside them: Enc as `encrypt_ordinary` (s
//! uniform, then `sample_errors` and `encrypt_with_coins`), aEnc as `HiddenPlan::encrypt`
//! (the broadcast step, s uniform subject to H_S^T s = floor(q/2) am, then the same two
//! calls). That step is also timed alone, in --step-reps paired iterations of the uniform
//! s and the constrained s (`step`); their difference is aEnc's overhead. Each row records
//! Enc of its own rounds and aEnc minus Enc round by round (`encryption_difference`) as a
//! check.
//!
//! Loops of fast operations (aKGen, the group setup, the encryption rounds, decryption)
//! run at least --keygen-reps or --reps times and then until --min-time seconds have
//! passed, at most MAX_RUNS times. The decryptions are timed in a loop of their own on
//! the last rounds' ciphertexts. The ordinary encryption and decryptions are those of all
//! trades.
//!
//! Sizes (`sizes_bytes`): public parameters (A0, A1, B, U), the master shortcut key
//! (D1, D2) at the width of its largest entry, ciphertexts of both modes as encoded,
//! the anamorphic public vector and binary secret key of one hidden bit and of one user
//! of the largest trade, and that trade's group matrix H_S. Field elements count at the
//! byte width the ciphertext encoding uses.
use clap::Parser;
use epoque_shortcut::trapdoor::{TrapdoorProfile, TrapdoorSetup, decrypt_identity, setup_trapdoor};
use epoque_shortcut::*;
use rand::RngExt;
use rand_core::UnwrapErr;
use serde_json::{Value, json};
use std::{fs, hint::black_box, time::Instant};

/// Most runs of one timed loop that --min-time extends.
const MAX_RUNS: usize = 20_000;
/// Ciphertext pairs of the last rounds that the decryption loop cycles through.
const POOL: usize = 8;

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

/// The sum of two series run by run (as many runs as the shorter one).
fn paired_sum(a: &[f64], b: &[f64]) -> Vec<f64> {
    a.iter().zip(b).map(|(x, y)| x + y).collect()
}

/// `a - b` run by run.
fn paired_difference(a: &[f64], b: &[f64]) -> Vec<f64> {
    a.iter().zip(b).map(|(x, y)| x - y).collect()
}

/// A trade RECIPIENTS:BITS.
fn parse_trade(trade: &str) -> std::result::Result<(usize, usize), String> {
    let (recipients, bits) = trade.split_once(':').ok_or("trades use RECIPIENTS:BITS")?;
    let number = |x: &str| x.parse::<usize>().map_err(|e| format!("{trade}: {e}"));
    Ok((number(recipients)?, number(bits)?))
}

/// The default sweep: |S| = 2 and L = 1, 2, 4, ... while d = 2L fits the budget.
fn sweep(budget: usize) -> Vec<(usize, usize)> {
    (0..usize::BITS)
        .map(|k| (2, 1usize << k))
        .take_while(|&(size, bits)| size * bits <= budget)
        .collect()
}

/// The default checks: the sweep's largest d split as d/2 recipients of 2 bits and with
/// |S| close to L, where these differ from the sweep.
fn splits(sweep: &[(usize, usize)]) -> Vec<(usize, usize)> {
    let d = sweep.last().map_or(0, |&(size, bits)| size * bits);
    let mut out = Vec::new();
    for size in [d / 2, 1usize << (d.max(1).ilog2() / 2)] {
        let trade = (size, d / size.max(1));
        if size >= 1 && size * trade.1 == d && !sweep.contains(&trade) && !out.contains(&trade) {
            out.push(trade);
        }
    }
    out
}

/// Times Epoque's ordinary algorithms and the anamorphic broadcast across d = |S| L.
#[derive(Parser)]
struct Args {
    /// Parameter profile.
    #[arg(long, value_parser = ["small", "full", "full32"], default_value = "full32")]
    profile: String,
    /// Identity to encrypt to: uniformly random in Z_q^n, or the unit identity (H = I).
    #[arg(long, value_parser = ["random", "unit"], default_value = "random")]
    identity: String,
    /// Runs of every measured algorithm.
    #[arg(long, default_value_t = 50)]
    reps: usize,
    /// Untimed encryption rounds per trade before the measured ones.
    #[arg(long, default_value_t = 2)]
    warmup: usize,
    /// Runs of the ordinary setup, the costliest step (default: --reps).
    #[arg(long)]
    setup_reps: Option<usize>,
    /// Runs of aKGen for the group's recipients (default: --reps).
    #[arg(long)]
    keygen_reps: Option<usize>,
    /// Paired iterations of the differing step alone, per trade.
    #[arg(long, default_value_t = 2000)]
    step_reps: usize,
    /// Seconds that every loop of fast operations runs at least, after its --reps runs.
    #[arg(long, default_value_t = 10.0)]
    min_time: f64,
    /// Trades RECIPIENTS:BITS of the sweep (default: |S| = 2 and L = 1, 2, 4, ... up to
    /// d = g).
    #[arg(long, value_delimiter = ',', value_parser = parse_trade)]
    trades: Vec<(usize, usize)>,
    /// Trades RECIPIENTS:BITS that check that the costs depend on d only (default with
    /// the default sweep: two other splits of its largest d).
    #[arg(long, value_delimiter = ',', value_parser = parse_trade)]
    checks: Vec<(usize, usize)>,
    /// Group sizes |S|, each with the longest message the profile's g allows and with
    /// half of it, instead of --trades and --checks.
    #[arg(long, value_delimiter = ',')]
    groups: Vec<usize>,
    /// Result file (JSON), rewritten after every finished part.
    #[arg(long, default_value = "epoque-experiments.json")]
    output: String,
}

fn main() -> std::result::Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();
    if cfg!(debug_assertions) {
        return Err("run experiments with --release".into());
    }
    let tp = match args.profile.as_str() {
        "small" => TrapdoorProfile::small(),
        "full" => TrapdoorProfile::full(),
        // 64 constraints: 32 recipients of two bits, and J = 192 ordinary bits as in
        // Epoque's benchmarks (tools/trapdoor_parameters.py, J does not change n, m, q).
        _ => TrapdoorProfile::new(
            Profile::new(1424, 53_926, 220_301_035_189, 192, 64, 112_566_231)?,
            4,
            4810.0,
        )?,
    };
    let p = tp.profile();
    let (n, m, q, jb) = (p.n(), p.m(), p.q(), p.ordinary_bits());
    let (reps, warmup) = (args.reps, args.warmup);
    let (step_reps, min_time) = (args.step_reps, args.min_time);
    let budget = p.maximum_constraints();
    let label = |size: usize, bits: usize| match size * bits {
        d if d == budget => "full",
        d if 2 * d == budget => "half",
        _ => "part",
    };
    // (length, role, |S|, L) of every trade.
    let configs: Vec<(&str, &str, usize, usize)> = if !args.groups.is_empty() {
        args.groups
            .iter()
            .filter(|&&size| size > 0)
            .flat_map(|&size| [(size, budget / size), (size, budget / size / 2)])
            .filter(|&(_, bits)| bits > 0)
            .map(|(size, bits)| (label(size, bits), "sweep", size, bits))
            .collect()
    } else {
        let trades = if args.trades.is_empty() {
            sweep(budget)
        } else {
            args.trades.clone()
        };
        let checks = if args.trades.is_empty() && args.checks.is_empty() {
            splits(&trades)
        } else {
            args.checks.clone()
        };
        trades
            .iter()
            .map(|&(size, bits)| (label(size, bits), "sweep", size, bits))
            .chain(
                checks
                    .iter()
                    .map(|&(size, bits)| (label(size, bits), "check", size, bits)),
            )
            .collect()
    };
    // Sizes per user and of H_S for the largest trade, preferring many recipients.
    let (size, bits) = configs
        .iter()
        .map(|c| (c.2, c.3))
        .max_by_key(|&(size, bits)| (size * bits, size))
        .unwrap_or((0, 0));
    let setup_reps = args.setup_reps.unwrap_or(reps);
    let keygen_reps = args.keygen_reps.unwrap_or(reps);
    if reps == 0
        || setup_reps == 0
        || keygen_reps == 0
        || step_reps == 0
        || !(min_time.is_finite() && min_time >= 0.0)
        || configs.is_empty()
        || configs
            .iter()
            .any(|c| c.2 == 0 || c.3 == 0 || c.2 * c.3 > budget)
    {
        return Err(format!(
            "need reps (all kinds) >= 1, min time >= 0 and trades with |S|, L >= 1 and |S| L <= g = {budget}"
        )
        .into());
    }
    let field_bits = (64 - (q - 1).leading_zeros()) as usize;
    let mut rng = UnwrapErr(OsBits::default());
    let mut report = json!({
        "scheme": "epoque",
        "complete": false,
        "config": {"profile": args.profile, "identity": args.identity, "reps": reps,
                   "setup_reps": setup_reps, "keygen_reps": keygen_reps,
                   "trades": configs.iter().map(|c| json!({"recipients": c.2, "bits": c.3, "length": c.0, "role": c.1})).collect::<Vec<_>>(),
                   "warmup_rounds": warmup, "step_reps": step_reps,
                   "min_time_s": min_time, "max_runs": MAX_RUNS,
                   "timing": "per trade, --warmup untimed rounds, then at least --reps measured rounds and until --min-time seconds; every round encrypts fresh messages with Enc and with aEnc, Enc first in even rounds and aEnc first in odd ones; the differing step alone in --step-reps paired iterations, alternating which goes first; wall clock; checks untimed"},
        "environment": {"os": std::env::consts::OS, "arch": std::env::consts::ARCH,
                        "threads": arithmetic::threads(),
                        "threads_from": "rayon pool: RAYON_NUM_THREADS, else all logical CPUs",
                        "logical_cpus": std::thread::available_parallelism().map_or(1, |x| x.get())},
        "parameters": {"n": n, "m": m, "q": q.to_string(), "J": jb, "g": p.maximum_constraints(),
                       "C": p.c_bound(), "key_width_s": tp.width(), "gadget_base": tp.base(),
                       "field_bits": field_bits},
        "sizes_bytes": {"ordinary_public_key": (3 * n * m + n * jb) * p.field_bytes(),
                        "anamorphic_public_key_per_bit": n * p.field_bytes(),
                        "anamorphic_secret_key_per_bit": m.div_ceil(8),
                        "anamorphic_public_key_per_user": bits * n * p.field_bytes(),
                        "anamorphic_secret_key_per_user": (bits * m).div_ceil(8),
                        "anamorphic_group_matrix": n * size * bits * p.field_bytes()},
        "definitions": {
            "ordinary.setup": "setup_trapdoor: TrapGen for A0 and B, A1 and U uniform, the master shortcut key, exact checks",
            "ordinary.encryption": "encrypt_ordinary's steps for one identity, including all coins: s uniform, sample_errors, encrypt_with_coins; in the encryption rounds of all trades",
            "ordinary.decryption": "decrypt_ordinary with the master shortcut key, repeated on the last rounds' ordinary ciphertexts",
            "anamorphic.keygen": "aKGen for the |S| recipients only: binary keys, then apk = A0 k for every key",
            "anamorphic.group_setup": "prepare_hidden for the group S (reads only S's public vectors)",
            "anamorphic.setup": "keygen + group_setup, run by run (as many runs as the shorter series)",
            "anamorphic.encryption": "HiddenPlan::encrypt's steps (aEnc steps 3 to 7) for the identity: the broadcast step, sample_errors, encrypt_with_coins",
            "anamorphic.ordinary_encryption": "Enc in the same rounds as aEnc",
            "anamorphic.encryption_difference": "aEnc minus Enc of the same round",
            "anamorphic.broadcast_step": "s uniform subject to H_S^T s = floor(q/2) am per recipient (AffineSampler::sample), inside aEnc",
            "anamorphic.replaced_step": "s uniform in Z_q^n inside Enc, the step the broadcast step replaces",
            "anamorphic.step": "the differing step alone in --step-reps paired iterations: replaced = s uniform in Z_q^n (arithmetic::uniform), broadcast = s subject to H_S^T s = floor(q/2) am (AffineSampler::sample), overhead = broadcast minus replaced",
            "anamorphic.decryption": "decrypt_covert by one fixed recipient of S (the first), repeated on the last rounds' anamorphic ciphertexts",
            "anamorphic.ordinary_decryption": "decrypt_ordinary of the trade's last ordinary ciphertexts, repeated",
            "anamorphic.ordinary_decryption_identity_key": "decrypt_identity of the trade's last ordinary ciphertexts, repeated"},
    });
    let save = |report: &Value| -> std::io::Result<()> {
        fs::write(
            &args.output,
            serde_json::to_vec_pretty(report).expect("json"),
        )
    };
    save(&report)?;

    // Ordinary setup. Only the last run's keys are kept, the earlier ones are
    // dropped before the next run starts.
    let mut setup_ms = Vec::new();
    let mut kept = None;
    for run in 0..setup_reps {
        drop(kept.take());
        let t = Instant::now();
        let generated = setup_trapdoor(&tp, &mut rng, |_, _| {})?;
        setup_ms.push(ms(t));
        kept = Some(generated);
        eprintln!(
            "setup {}/{setup_reps}: {:.1} s",
            run + 1,
            setup_ms[run] / 1e3
        );
    }
    let generated = kept.expect("at least one setup");
    let id = if args.identity == "unit" {
        generated.setup.public.unit_identity()
    } else {
        let coordinates: Vec<u64> = (0..n).map(|_| rng.random_range(0..q)).collect();
        generated.setup.public.identity(&coordinates)?
    };
    let mut extraction_ms = Vec::new();
    let mut id_key = None;
    for _ in 0..reps {
        let t = Instant::now();
        let key = generated
            .master
            .extract(&generated.setup.public, &id, &mut rng)?;
        extraction_ms.push(ms(t));
        id_key = Some(key);
    }
    let id_key = id_key.expect("at least one extraction");
    // T_A0 is not needed after extraction: whoever holds it can read every hidden message.
    let TrapdoorSetup { setup, master, .. } = generated;
    drop(master);
    let pk = &setup.public;
    // The master shortcut key (D1, D2): 2 J m entries at the width of its largest entry.
    let widest = setup
        .shortcut
        .d1_transposed()
        .iter()
        .chain(setup.shortcut.d2_transposed())
        .map(|x| x.unsigned_abs())
        .max()
        .unwrap_or(0);
    let entry_bits = 1 + (32 - widest.leading_zeros()) as usize;
    report["sizes_bytes"]["ordinary_secret_key"] = json!((2 * jb * m * entry_bits).div_ceil(8));

    // Ordinary encryption and decryption are timed with aEnc and aDec below, and added
    // here after every trade.
    report["ordinary"] = json!({
        "setup": stats(&setup_ms),
        "identity_extraction": stats(&extraction_ms),
    });
    save(&report)?;
    eprintln!(
        "ordinary: setup {:.1} s, identity extraction {:.1} s",
        mean(&setup_ms) / 1e3,
        mean(&extraction_ms) / 1e3
    );

    // For every trade: aKGen for exactly the recipients, the preparation, the encryption
    // rounds (Enc and aEnc), the differing step alone, and decryption.
    let mut rows = Vec::new();
    let (mut all_enc_ms, mut all_dec_ms, mut all_dec_id_ms) = (Vec::new(), Vec::new(), Vec::new());
    let mut checked = 0usize;
    for &(length, role, size, bits) in &configs {
        let mut keygen_ms = Vec::new();
        let mut kept = None;
        let start = Instant::now();
        while more(keygen_ms.len(), keygen_reps, start, min_time) {
            drop(kept.take());
            let t = Instant::now();
            let mut apk = CovertPublicKey::with_capacity(pk, bits, size)?;
            let keys = (0..size)
                .map(|_| covert_secret_keygen(p, bits, &mut rng))
                .collect::<Result<Vec<_>>>()?;
            apk.append(pk, &keys)?;
            keygen_ms.push(ms(t));
            kept = Some((apk, keys));
        }
        let (apk, keys) = kept.expect("at least one aKGen run");
        let group: Vec<usize> = (0..size).collect();
        let recipient = group[0];
        let mut group_ms = Vec::new();
        let start = Instant::now();
        while more(group_ms.len(), reps, start, min_time) {
            let t = Instant::now();
            let plan = prepare_hidden(pk, &apk, &group)?;
            group_ms.push(ms(t));
            if !plan.sampler().full_rank() {
                return Err("H_S is rank deficient".into());
            }
        }
        let plan = prepare_hidden(pk, &apk, &group)?;
        let sampler = plan.sampler();
        // Every round encrypts fresh messages once with Enc and once with aEnc, Enc first
        // in even rounds and aEnc first in odd ones, so that drift of the machine falls on
        // both alike. Timing them in separate blocks let that drift decide the sign of
        // aEnc - Enc, which is far below the run-to-run spread of one encryption.
        let (mut enc_ms, mut aenc_ms) = (Vec::new(), Vec::new());
        let (mut step_ms, mut replaced_ms) = (Vec::new(), Vec::new());
        let mut pool = Vec::new();
        let start = Instant::now();
        let mut round = 0;
        while round < warmup || more(round - warmup, reps, start, min_time) {
            let ordinary: Vec<u8> = (0..jb).map(|_| rng.random_range(0..2u8)).collect();
            let hidden: Vec<u8> = (0..bits).map(|_| rng.random_range(0..2u8)).collect();
            let (mut ordinary_ct, mut anamorphic_ct) = (None, None);
            let (mut enc, mut aenc, mut step, mut replaced) = (0.0, 0.0, 0.0, 0.0);
            let order = if round % 2 == 0 {
                [false, true]
            } else {
                [true, false]
            };
            for anamorphic in order {
                // The steps of encrypt_ordinary and of HiddenPlan::encrypt: the secret s,
                // then Epoque's encryption with fresh errors, the same calls in both.
                let t0 = Instant::now();
                let s = if anamorphic {
                    // The broadcast step: s uniform subject to H_S^T s = floor(q/2) am,
                    // with am repeated once per recipient.
                    let rhs: Vec<u64> = (0..size)
                        .flat_map(|_| hidden.iter().map(|&bit| u64::from(bit) * (q / 2)))
                        .collect();
                    sampler.sample(&rhs, &mut rng)?
                } else {
                    arithmetic::uniform(n, q, &mut rng)?
                };
                let t1 = Instant::now();
                let ct = encrypt_with_coins(pk, &id, &ordinary, &s, &sample_errors(p, &mut rng)?)?;
                let t2 = Instant::now();
                if anamorphic {
                    aenc = span(t0, t2);
                    step = span(t0, t1);
                    anamorphic_ct = Some(ct);
                } else {
                    enc = span(t0, t2);
                    replaced = span(t0, t1);
                    ordinary_ct = Some(ct);
                }
            }
            let ordinary_ct = ordinary_ct.expect("Enc ran in this round");
            let anamorphic_ct = anamorphic_ct.expect("aEnc ran in this round");
            let all = group
                .iter()
                .map(|&i| decrypt_covert(&keys[i], &anamorphic_ct))
                .collect::<Result<Vec<_>>>()?;
            if decrypt_ordinary(&setup.shortcut, &ordinary_ct)? != ordinary
                || decrypt_identity(&id_key, &ordinary_ct)? != ordinary
                || all.iter().any(|bits| bits != &hidden)
                || decrypt_ordinary(&setup.shortcut, &anamorphic_ct)? != ordinary
                || decrypt_identity(&id_key, &anamorphic_ct)? != ordinary
            {
                return Err("ordinary or anamorphic decryption failed".into());
            }
            if round == 0 {
                report["sizes_bytes"]["ordinary_ciphertext"] =
                    json!(encode(p, &ordinary_ct)?.len());
                report["sizes_bytes"]["anamorphic_ciphertext"] =
                    json!(encode(p, &anamorphic_ct)?.len());
            }
            checked += 2;
            if round >= warmup {
                enc_ms.push(enc);
                aenc_ms.push(aenc);
                step_ms.push(step);
                replaced_ms.push(replaced);
            }
            pool.push((ordinary_ct, anamorphic_ct));
            if pool.len() > POOL {
                pool.remove(0);
            }
            round += 1;
        }
        let difference = paired_difference(&aenc_ms, &enc_ms);
        // Dec with the shortcut key and with the identity key, and aDec by one fixed
        // recipient, many times on the last rounds' ciphertexts (all checked above).
        let (mut dec_ms, mut dec_id_ms, mut adec_ms) = (Vec::new(), Vec::new(), Vec::new());
        let start = Instant::now();
        while more(dec_ms.len(), reps, start, min_time) {
            let (ordinary_ct, anamorphic_ct) = &pool[dec_ms.len() % pool.len()];
            let t = Instant::now();
            black_box(decrypt_ordinary(&setup.shortcut, ordinary_ct)?);
            dec_ms.push(ms(t));
            let t = Instant::now();
            black_box(decrypt_identity(&id_key, ordinary_ct)?);
            dec_id_ms.push(ms(t));
            let t = Instant::now();
            black_box(decrypt_covert(&keys[recipient], anamorphic_ct)?);
            adec_ms.push(ms(t));
        }
        // The step in which Enc and aEnc differ, alone: the uniform s of Enc and the
        // constrained s of aEnc, alternating which goes first.
        let (mut replaced_alone, mut broadcast_alone) = (Vec::new(), Vec::new());
        for i in 0..warmup + step_reps {
            let hidden: Vec<u8> = (0..bits).map(|_| rng.random_range(0..2u8)).collect();
            let order = if i % 2 == 0 {
                [false, true]
            } else {
                [true, false]
            };
            for broadcast in order {
                let t = Instant::now();
                if broadcast {
                    let rhs: Vec<u64> = (0..size)
                        .flat_map(|_| hidden.iter().map(|&bit| u64::from(bit) * (q / 2)))
                        .collect();
                    black_box(sampler.sample(&rhs, &mut rng)?);
                    if i >= warmup {
                        broadcast_alone.push(ms(t));
                    }
                } else {
                    black_box(arithmetic::uniform(n, q, &mut rng)?);
                    if i >= warmup {
                        replaced_alone.push(ms(t));
                    }
                }
            }
        }
        let step_overhead = paired_difference(&broadcast_alone, &replaced_alone);
        all_enc_ms.extend_from_slice(&enc_ms);
        all_dec_ms.extend_from_slice(&dec_ms);
        all_dec_id_ms.extend_from_slice(&dec_id_ms);
        report["ordinary"]["encryption"] = stats(&all_enc_ms);
        report["ordinary"]["decryption"] = stats(&all_dec_ms);
        report["ordinary"]["decryption_identity_key"] = stats(&all_dec_id_ms);
        let row = json!({
            "length": length,
            "role": role,
            "recipients": size,
            "bits": bits,
            "constraints": size * bits,
            "decryption_recipient": recipient,
            "apk_bytes_per_recipient": bits * n * p.field_bytes(),
            "group_matrix_bytes": n * size * bits * p.field_bytes(),
            "keygen": stats(&keygen_ms),
            "group_setup": stats(&group_ms),
            "setup": stats(&paired_sum(&keygen_ms, &group_ms)),
            "encryption": stats(&aenc_ms),
            "ordinary_encryption": stats(&enc_ms),
            "encryption_difference": stats(&difference),
            "broadcast_step": stats(&step_ms),
            "replaced_step": stats(&replaced_ms),
            "step": {"iterations": step_reps,
                     "replaced": stats(&replaced_alone),
                     "broadcast": stats(&broadcast_alone),
                     "overhead": stats(&step_overhead)},
            "decryption": stats(&adec_ms),
            "ordinary_decryption": stats(&dec_ms),
            "ordinary_decryption_identity_key": stats(&dec_id_ms),
        });
        eprintln!(
            "|S| = {size}, L = {bits}, d = {} ({role}): aKGen {:.1} ms, group setup {:.2} ms, Enc {:.1} ms, aEnc {:.1} ms ({} rounds, median aEnc - Enc {:+.1} ms); step alone (medians): uniform s {:.1} us, constrained s {:.1} us, overhead {:+.1} us; aDec {:.3} ms",
            size * bits,
            mean(&keygen_ms),
            mean(&group_ms),
            mean(&enc_ms),
            mean(&aenc_ms),
            enc_ms.len(),
            median(&difference),
            1e3 * median(&replaced_alone),
            1e3 * median(&broadcast_alone),
            1e3 * median(&step_overhead),
            mean(&adec_ms)
        );
        rows.push(row);
        report["anamorphic"] = json!(rows);
        report["checks"] = json!({"checked_ciphertexts": checked});
        save(&report)?;
    }
    report["complete"] = json!(true);
    save(&report)?;
    eprintln!("wrote {}", args.output);
    Ok(())
}
