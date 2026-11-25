//! Experiments: the average cost of the Dual-Regev host's ordinary algorithms and of
//! Construction 1's anamorphic broadcast, over repeated runs, across the number of
//! equations d = |S| L that a group's message imposes.
//!
//!   cargo run --locked --release --bin experiments -- --reps 50 \
//!       --output basic-experiments.json
//!
//! Profiles: `g256` (the default) has n = 1384, m = 36974, q = 90147931 and n_min = 1128,
//! so its 256 constraints carry an 8-bit message to 32 recipients. `n1152` is the
//! maintained profile n = 1152, m = 30357, q = 67537879, n_min = 1120 (32 constraints,
//! 4 recipients of 8 bits).
//!
//! Ordinary algorithms (run once per repetition): setup is KGen (uniform A, binary k,
//! b = A k), encryption is Dual-Regev encryption of one bit and decryption the ordinary
//! rounding. The noise table, built once per process and shared by all encryptions, is
//! timed separately.
//!
//! Anamorphic algorithms, for the trades |S|:L of --trades and --checks: aKGen for
//! exactly the |S| recipients, the group setup, aEnc, and aDec by one fixed recipient (the
//! first of S). The group setup and aEnc depend on d = |S| L only and aDec on L only, so
//! the default sweep keeps |S| = 2 and doubles L from 1 up to (n - n_min)/2 (d = 2, 4,
//! ..., n - n_min), and the default checks repeat the largest d with two other splits
//! (d/2 recipients of 2 bits, and |S| close to L). --groups instead runs, for every |S|,
//! the longest L that the n - n_min equations allow and half of it, as before the sweep.
//! Every ciphertext is checked untimed: ordinary decryption and hidden decryption by
//! every recipient in S. Every result row records |S|, L, d and whether it belongs to the
//! sweep or the checks.
//!
//! Enc and aEnc are timed in the same rounds, with the trade's key pair: after --warmup
//! untimed rounds, each round encrypts fresh messages once with Enc and once with aEnc,
//! Enc first in even rounds and aEnc first in odd ones, so that drift of the machine
//! (clock, temperature, other load) falls on both alike. Both run the library's steps
//! through its public API, so that the one step in which they differ is timed inside
//! them: Enc as `encrypt_with` (s uniform, then the noise seed and `encrypt_from_seed`),
//! aEnc as `HiddenPlan::encrypt_with` (the broadcast step `HiddenPlan::sample_secret_with`,
//! then the same two steps). That step is also timed alone, in --step-reps paired
//! iterations of the uniform s and the constrained s (`step`); their difference is aEnc's
//! overhead. Each row records Enc of its own rounds and aEnc minus Enc round by round
//! (`encryption_difference`) as a check.
//!
//! Loops of fast operations (aKGen, the group setup, the encryption rounds, decryption)
//! run at least --keygen-reps or --reps times and then until --min-time seconds have
//! passed, at most MAX_RUNS times. The decryptions are timed in a loop of their own on
//! the last rounds' ciphertexts. The ordinary encryption and decryption are those of all
//! trades.
//!
//! Sizes (`sizes_bytes`): ordinary public key (A, b) and binary secret key k, ciphertexts
//! of both modes as serialized, the anamorphic public vectors and seed key of one hidden
//! bit and of one user of the largest trade (also expanded), and that trade's group
//! matrix H_S. Field elements count at the 4-byte width the ciphertext serialization
//! uses; binary keys count one bit per entry.
use basic_anamorphic::*;
use rand::{Rng, RngExt};
use serde_json::{Value, json};
use std::{error::Error, fs, hint::black_box, time::Instant};

type BenchResult<T> = std::result::Result<T, Box<dyn Error>>;

/// Users whose keys one `anamorphic_keygen_with` call makes. Each call streams A once
/// and every row of A meets the masks of all the call's key blocks (121 KB each), so
/// eight users (64 blocks at L = 8, about 8 MB of masks) keep the masks in cache.
const BATCH: usize = 8;
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

/// The 32-byte noise seed that encryption draws after s.
fn noise_seed<R: Rng + ?Sized>(rng: &mut R) -> [u8; 32] {
    let mut seed = [0u8; 32];
    rng.fill_bytes(&mut seed);
    seed
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

struct Args {
    reps: usize,
    warmup: usize,
    keygen_reps: Option<usize>,
    step_reps: usize,
    min_time: f64,
    profile: String,
    trades: Vec<(usize, usize)>,
    checks: Vec<(usize, usize)>,
    groups: Vec<usize>,
    mode: Mode,
    output: String,
}

fn parse() -> std::result::Result<Args, String> {
    let mut a = Args {
        reps: 50,
        warmup: 2,
        keygen_reps: None,
        step_reps: 2000,
        min_time: 10.0,
        profile: "g256".into(),
        trades: Vec::new(),
        checks: Vec::new(),
        groups: Vec::new(),
        mode: Mode::Multi,
        output: "basic-experiments.json".into(),
    };
    let mut it = std::env::args().skip(1);
    while let Some(flag) = it.next() {
        let value = it.next().ok_or(format!("missing value for {flag}"))?;
        let number = |v: &str| v.parse::<usize>().map_err(|e| format!("{flag}: {e}"));
        let trades = |v: &str| {
            v.split(',')
                .map(|trade| {
                    let (size, bits) = trade
                        .split_once(':')
                        .ok_or(format!("{flag}: trades use RECIPIENTS:BITS"))?;
                    Ok((number(size)?, number(bits)?))
                })
                .collect::<std::result::Result<Vec<_>, String>>()
        };
        match flag.as_str() {
            "--reps" => a.reps = number(&value)?,
            "--warmup" => a.warmup = number(&value)?,
            "--keygen-reps" => a.keygen_reps = Some(number(&value)?),
            "--step-reps" => a.step_reps = number(&value)?,
            "--min-time" => a.min_time = value.parse().map_err(|e| format!("{flag}: {e}"))?,
            "--profile" => a.profile = value,
            "--trades" => a.trades = trades(&value)?,
            "--checks" => a.checks = trades(&value)?,
            "--groups" => {
                a.groups = value
                    .split(',')
                    .map(&number)
                    .collect::<std::result::Result<Vec<_>, _>>()?
            }
            "--mode" => {
                a.mode = match value.as_str() {
                    "single" => Mode::Single,
                    "multi" => Mode::Multi,
                    other => return Err(format!("unknown mode {other}")),
                }
            }
            "--output" => a.output = value,
            _ => {
                return Err(format!(
                    "unknown option {flag}; use --reps --warmup --keygen-reps --step-reps --min-time SECONDS --profile g256|n1152 --trades 2:1,2:2 --checks 128:2 --groups 128,64 --mode single|multi --output"
                ));
            }
        }
    }
    if a.reps == 0
        || a.keygen_reps == Some(0)
        || a.step_reps == 0
        || !(a.min_time.is_finite() && a.min_time >= 0.0)
        || a.groups.contains(&0)
        || !matches!(a.profile.as_str(), "g256" | "n1152")
    {
        return Err(
            "need reps (all kinds) >= 1, min time >= 0, group sizes >= 1 and profile g256 or n1152"
                .into(),
        );
    }
    Ok(a)
}

fn main() -> BenchResult<()> {
    if cfg!(debug_assertions) {
        return Err("run experiments with --release".into());
    }
    let args = parse()?;
    let (reps, warmup, mode) = (args.reps, args.warmup, args.mode);
    let (step_reps, min_time) = (args.step_reps, args.min_time);
    let keygen_reps = args.keygen_reps.unwrap_or(reps);
    let (n, m, q, n_min) = if args.profile == "n1152" {
        (1152, 30357, 67537879, 1120)
    } else {
        (1384, 36974, 90147931, 1128)
    };
    // (q sigma)^2 = n (m + 1), used exactly by the noise table.
    let variance = (n * (m + 1)) as u64;
    // The ordinary algorithms do not depend on L, so they run with L = 1.
    let params = Parameters::with_noise_variance(n, m, q, variance, n_min, 1)?;
    // The n - n_min equations that a group's message may impose.
    let budget = n - n_min;
    let label = |size: usize, bits: usize| match size * bits {
        d if d == budget => "full",
        d if 2 * d == budget => "half",
        _ => "part",
    };
    // (length, role, |S|, L) of every trade.
    let configs: Vec<(&str, &str, usize, usize)> = if !args.groups.is_empty() {
        args.groups
            .iter()
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
    if configs.is_empty()
        || configs
            .iter()
            .any(|c| c.2 == 0 || c.3 == 0 || c.2 * c.3 > budget)
    {
        return Err(
            format!("every trade needs |S|, L >= 1 and |S| L <= n - n_min = {budget}").into(),
        );
    }
    // Sizes per user and of H_S for the largest trade, preferring many recipients.
    let (size, bits) = configs
        .iter()
        .map(|c| (c.2, c.3))
        .max_by_key(|&(size, bits)| (size * bits, size))
        .expect("at least one trade");
    let field_bits = 32 - q.leading_zeros() as usize;
    let mut rng = rand::rng();
    let mode_name = if mode == Mode::Multi {
        "multi"
    } else {
        "single"
    };
    let mut report = json!({
        "scheme": "basic",
        "complete": false,
        "config": {"profile": args.profile, "reps": reps, "keygen_reps": keygen_reps,
                   "trades": configs.iter().map(|c| json!({"recipients": c.2, "bits": c.3, "length": c.0, "role": c.1})).collect::<Vec<_>>(),
                   "mode": mode_name,
                   "threads": mode.threads(), "anamorphic_keys": "seed (32 bytes, SHAKE256-expanded)",
                   "akgen_batch_users": BATCH,
                   "warmup_rounds": warmup, "step_reps": step_reps,
                   "min_time_s": min_time, "max_runs": MAX_RUNS,
                   "timing": "per trade with the trade's key pair, --warmup untimed rounds, then at least --reps measured rounds and until --min-time seconds; every round encrypts fresh messages with Enc and with aEnc, Enc first in even rounds and aEnc first in odd ones; the differing step alone in --step-reps paired iterations, alternating which goes first; wall clock; checks untimed"},
        "environment": {"os": std::env::consts::OS, "arch": std::env::consts::ARCH,
                        "logical_cpus": std::thread::available_parallelism().map_or(1, |x| x.get())},
        "parameters": {"n": n, "m": m, "q": q, "n_min": n_min, "noise_variance": variance,
                       "field_bits": field_bits},
        "sizes_bytes": {"ordinary_public_key": (n * m + n) * params.field_bytes(),
                        "ordinary_secret_key": m.div_ceil(8),
                        "anamorphic_public_key_per_bit": n * params.field_bytes(),
                        "anamorphic_secret_key_per_bit_expanded": m.div_ceil(8),
                        "anamorphic_public_key_per_user": bits * n * params.field_bytes(),
                        "anamorphic_secret_key_per_user": SEED_BYTES,
                        "anamorphic_secret_key_per_user_expanded": (bits * m).div_ceil(8),
                        "anamorphic_group_matrix": n * size * bits * params.field_bytes()},
        "definitions": {
            "ordinary.setup": "keygen_with: uniform A, binary k, b = A k",
            "ordinary.encryption": "encrypt_with's steps: uniform s, the noise seed and encrypt_from_seed (errors from the shared noise table); in the encryption rounds of all trades",
            "ordinary.decryption": "ordinary_decrypt: c - k^T r, rounded, repeated on the last rounds' ordinary ciphertexts of all trades",
            "anamorphic.keygen": "aKGen for the |S| recipients only: seeds, SHAKE256 expansion, b = A k for every key block",
            "anamorphic.group_setup": "prepare_hidden_with (the run's mode) for the group S (reads only S's directory vectors)",
            "anamorphic.setup": "keygen + group_setup, run by run (as many runs as the shorter series)",
            "anamorphic.encryption": "HiddenPlan::encrypt_with's steps: the broadcast step, the noise seed and encrypt_from_seed",
            "anamorphic.ordinary_encryption": "Enc in the same rounds as aEnc, with the same key pair",
            "anamorphic.broadcast_step": "HiddenPlan::sample_secret_with inside aEnc: s uniform subject to H_S^T s = Delta am",
            "anamorphic.replaced_step": "s uniform in Z_q^n inside Enc, the step the broadcast step replaces",
            "anamorphic.encryption_difference": "aEnc minus Enc of the same round",
            "anamorphic.step": "the differing step alone in --step-reps paired iterations: replaced = s uniform in Z_q^n, broadcast = HiddenPlan::sample_secret_with (the run's mode), overhead = broadcast minus replaced",
            "anamorphic.decryption": "hidden_decrypt_with (the run's mode) by one fixed recipient of S (the first) from its seed, repeated on the last rounds' anamorphic ciphertexts",
            "anamorphic.decryption_expanded_key": "the same from the recipient's expanded key blocks",
            "anamorphic.ordinary_decryption": "ordinary_decrypt of the trade's last ordinary ciphertexts, repeated"},
    });
    let save = |report: &Value| -> std::io::Result<()> {
        fs::write(
            &args.output,
            serde_json::to_vec_pretty(report).expect("json"),
        )
    };
    save(&report)?;

    // The noise table, built once per process for all encryptions and timed apart.
    let tables = reps.min(10);
    let mut table_ms = Vec::new();
    for _ in 0..tables {
        let t = Instant::now();
        let table = black_box(NoiseTable::with_variance(variance, mode)?);
        table_ms.push(ms(t));
        drop(table);
    }
    noise_table_with(&params, mode)?;

    // Ordinary setup. Ordinary encryption and decryption are timed below, with each
    // trade's key pair, together with aEnc and aDec, and added here after every trade.
    let mut setup_ms = Vec::new();
    let mut kept = None;
    for _ in 0..reps {
        drop(kept.take());
        let t = Instant::now();
        let pair = keygen_with(params, &mut rng, mode)?;
        setup_ms.push(ms(t));
        kept = Some(pair);
    }
    drop(kept);
    report["ordinary"] = json!({
        "setup": stats(&setup_ms),
        "noise_table": stats(&table_ms),
    });
    save(&report)?;
    eprintln!(
        "ordinary: setup {:.1} ms, noise table {:.1} ms",
        mean(&setup_ms),
        mean(&table_ms)
    );

    // For every trade: aKGen for exactly the recipients, the group setup, the encryption
    // rounds (Enc and aEnc), the differing step alone, and decryption.
    let mut rows = Vec::new();
    let (mut all_enc_ms, mut all_dec_ms) = (Vec::new(), Vec::new());
    let mut checked = 0usize;
    for &(length, role, size, bits) in &configs {
        // L is part of the key's parameters: a key pair for this L, untimed.
        let params = Parameters::with_noise_variance(n, m, q, variance, n_min, bits)?;
        let pair = keygen_with(params, &mut rng, mode)?;
        let pk = &pair.public;
        let mut keygen_ms = Vec::new();
        let mut kept = None;
        let start = Instant::now();
        while more(keygen_ms.len(), keygen_reps, start, min_time) {
            drop(kept.take());
            let mut directory: Directory = Vec::with_capacity(size);
            let mut user_keys: Vec<UserKey> = Vec::with_capacity(size);
            let t = Instant::now();
            while user_keys.len() < size {
                let batch = (size - user_keys.len()).min(BATCH);
                let part = anamorphic_keygen_with(pk, batch, true, &mut rng, mode)?;
                directory.extend(part.directory);
                user_keys.extend(part.user_keys);
            }
            keygen_ms.push(ms(t));
            kept = Some((directory, user_keys));
        }
        let (directory, user_keys) = kept.expect("at least one aKGen run");
        let audience: Vec<usize> = (0..size).collect();
        let recipient = audience[0];
        let mut group_ms = Vec::new();
        let start = Instant::now();
        while more(group_ms.len(), reps, start, min_time) {
            let t = Instant::now();
            let plan = prepare_hidden_with(pk, &directory, &audience, mode)?;
            group_ms.push(ms(t));
            if !plan.fiber().full_rank() {
                return Err("H_S is rank deficient".into());
            }
        }
        let plan = prepare_hidden_with(pk, &directory, &audience, mode)?;
        let expanded = user_blocks(&user_keys[recipient], &params).map(UserKey::Expanded)?;
        // Every round encrypts fresh messages once with Enc and once with aEnc, Enc first
        // in even rounds and aEnc first in odd ones, so that drift of the machine falls on
        // both alike. Timing them in separate blocks let that drift enter aEnc - Enc.
        let (mut enc_ms, mut aenc_ms) = (Vec::new(), Vec::new());
        let (mut step_ms, mut replaced_ms) = (Vec::new(), Vec::new());
        let mut pool = Vec::new();
        let start = Instant::now();
        let mut round = 0;
        while round < warmup || more(round - warmup, reps, start, min_time) {
            // Messages change every other round, so that both orders see every pattern.
            let bit = ((round / 2) % 2) as u8;
            let payload: Vec<u8> = (0..bits).map(|a| ((round / 2 + a) % 2) as u8).collect();
            let (mut ordinary_ct, mut anamorphic_ct) = (None, None);
            let (mut enc, mut aenc, mut step, mut replaced) = (0.0, 0.0, 0.0, 0.0);
            let order = if round % 2 == 0 {
                [false, true]
            } else {
                [true, false]
            };
            for anamorphic in order {
                // The steps of encrypt_with and of HiddenPlan::encrypt_with: the secret s,
                // then the noise seed and the Dual-Regev encryption, the same in both.
                let t0 = Instant::now();
                let secret = if anamorphic {
                    // The broadcast step: s uniform subject to H_S^T s = Delta am.
                    plan.sample_secret_with(&payload, &mut rng, mode)?
                } else {
                    (0..n).map(|_| rng.random_range(0..q)).collect::<Vec<u32>>()
                };
                let t1 = Instant::now();
                let ct = encrypt_from_seed(pk, bit, &secret, &noise_seed(&mut rng), mode)?;
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
            let mut all_ok = ordinary_decrypt(&ordinary_ct, &pair.secret, q)? == bit
                && ordinary_decrypt(&anamorphic_ct, &pair.secret, q)? == bit
                && hidden_decrypt_with(&anamorphic_ct, &user_keys[recipient], &params, mode)?
                    == payload
                && hidden_decrypt_with(&anamorphic_ct, &expanded, &params, mode)? == payload;
            for &i in &audience {
                all_ok &= hidden_decrypt(&anamorphic_ct, &user_keys[i], &params)? == payload;
            }
            if !all_ok {
                return Err("ordinary or anamorphic decryption failed".into());
            }
            if round == 0 {
                report["sizes_bytes"]["ordinary_ciphertext"] =
                    json!(serialize(&ordinary_ct, &params)?.len());
                report["sizes_bytes"]["anamorphic_ciphertext"] =
                    json!(serialize(&anamorphic_ct, &params)?.len());
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
        // Dec, and aDec by one fixed recipient from its seed and from its expanded
        // blocks, many times on the last rounds' ciphertexts (all checked above).
        let (mut dec_ms, mut adec_ms, mut adec_expanded_ms) = (Vec::new(), Vec::new(), Vec::new());
        let start = Instant::now();
        while more(dec_ms.len(), reps, start, min_time) {
            let (ordinary_ct, anamorphic_ct) = &pool[dec_ms.len() % pool.len()];
            let t = Instant::now();
            black_box(ordinary_decrypt(ordinary_ct, &pair.secret, q)?);
            dec_ms.push(ms(t));
            let t = Instant::now();
            black_box(hidden_decrypt_with(
                anamorphic_ct,
                &user_keys[recipient],
                &params,
                mode,
            )?);
            adec_ms.push(ms(t));
            let t = Instant::now();
            black_box(hidden_decrypt_with(
                anamorphic_ct,
                &expanded,
                &params,
                mode,
            )?);
            adec_expanded_ms.push(ms(t));
        }
        // The step in which Enc and aEnc differ, alone: the uniform s of Enc and the
        // constrained s of aEnc, alternating which goes first.
        let (mut replaced_alone, mut broadcast_alone) = (Vec::new(), Vec::new());
        for i in 0..warmup + step_reps {
            let payload: Vec<u8> = (0..bits).map(|_| rng.random_range(0..2u8)).collect();
            let order = if i % 2 == 0 {
                [false, true]
            } else {
                [true, false]
            };
            for broadcast in order {
                let t = Instant::now();
                if broadcast {
                    black_box(plan.sample_secret_with(&payload, &mut rng, mode)?);
                    if i >= warmup {
                        broadcast_alone.push(ms(t));
                    }
                } else {
                    black_box((0..n).map(|_| rng.random_range(0..q)).collect::<Vec<u32>>());
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
            "role": role,
            "recipients": size,
            "bits": bits,
            "constraints": size * bits,
            "decryption_recipient": recipient,
            "apk_bytes_per_recipient": bits * n * params.field_bytes(),
            "group_matrix_bytes": n * size * bits * params.field_bytes(),
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
            "decryption_expanded_key": stats(&adec_expanded_ms),
            "ordinary_decryption": stats(&dec_ms),
        });
        eprintln!(
            "|S| = {size}, L = {bits}, d = {} ({role}): aKGen {:.1} ms, group setup {:.3} ms, Enc {:.2} ms, aEnc {:.2} ms ({} rounds, median aEnc - Enc {:+.3} ms); step alone (medians): uniform s {:.1} us, constrained s {:.1} us, overhead {:+.1} us; aDec {:.3} ms",
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
