//! Reproducible research benchmark; all seeds are public and are not deployment keys.
//!
//! Each row (recipients, payload bits) is measured in both execution modes, with
//! the modes interleaved iteration by iteration. Both modes use the same seeds,
//! so every ciphertext must agree between the modes, which is checked. Per mode
//! the benchmark times key generation, anamorphic key generation, the noise
//! table, ordinary encryption and anamorphic encryption, the latter in total and
//! split into the broadcast step (sampling `s` with `H_S^T s = Delta am`) and the
//! Dual-Regev part (`r = A^T s + e`, `c = b^T s + y + Delta m`). Group setup and
//! decryption run on the calling thread in both modes and are timed once.
use basic_anamorphic::*;
use clap::Parser;
use rand::{Rng, RngExt, SeedableRng, rngs::StdRng};
use serde_json::{Value, json};
use std::{
    collections::BTreeMap,
    error::Error,
    fs,
    hint::black_box,
    io::{Seek, Write},
    path::{Path, PathBuf},
    time::Instant,
};

type BenchResult<T> = std::result::Result<T, Box<dyn Error>>;

/// Reproducible research benchmark of the fixed reviewed candidate n=1152, m=30357,
/// q=67537879, n_min=1120. All seeds are public and are not deployment keys.
#[derive(Parser)]
#[command(
    after_help = "Rows are RECIPIENTS:BITS with RECIPIENTS <= population and product <= 32.\nMulti mode uses rayon's global pool, set RAYON_NUM_THREADS to choose its size."
)]
struct Args {
    /// Measured repetitions per row (at least 4).
    #[arg(long, default_value_t = 21)]
    repetitions: usize,
    /// Unmeasured repetitions before the measured ones.
    #[arg(long, default_value_t = 3)]
    warmups: usize,
    /// Repetitions of key generation, anamorphic key generation and the noise table (at least 1).
    #[arg(long, default_value_t = 3)]
    one_time: usize,
    /// Registered hidden users (1 to 64).
    #[arg(long, default_value_t = 8)]
    population: usize,
    /// Rows to measure, as RECIPIENTS:BITS separated by commas.
    #[arg(long, value_delimiter = ',', value_parser = parse_row, default_value = "1:2,4:2")]
    rows: Vec<(usize, usize)>,
    /// Execution modes, separated by commas.
    #[arg(long, value_delimiter = ',', value_parser = parse_mode, default_value = "single,multi")]
    modes: Vec<Mode>,
    /// Result file. It must not exist yet.
    #[arg(long, default_value = "results/benchmarks.json")]
    output: PathBuf,
}

fn mode_name(mode: Mode) -> &'static str {
    match mode {
        Mode::Single => "single",
        Mode::Multi => "multi",
    }
}

fn parse_mode(name: &str) -> std::result::Result<Mode, String> {
    match name {
        "single" => Ok(Mode::Single),
        "multi" => Ok(Mode::Multi),
        other => Err(format!("unknown mode: {other}")),
    }
}

fn parse_row(row: &str) -> std::result::Result<(usize, usize), String> {
    let (t, bits) = row.split_once(':').ok_or("rows use RECIPIENTS:BITS")?;
    let number = |x: &str| x.parse::<usize>().map_err(|e| format!("{row}: {e}"));
    Ok((number(t)?, number(bits)?))
}

fn parse_args() -> BenchResult<Args> {
    let args = Args::parse();
    if args.repetitions < 4
        || args.one_time == 0
        || args.repetitions.checked_add(args.warmups).is_none()
        || args.rows.is_empty()
        || args.modes.is_empty()
        || !(1..=64).contains(&args.population)
        || args.rows.iter().any(|&(t, bits)| {
            !(1..=args.population).contains(&t)
                || bits == 0
                || t.checked_mul(bits).is_none_or(|d| d > 32)
        })
    {
        return Err("need >=4 repetitions, >=1 one-time repetition, 1<=population<=64, 1<=recipients<=population, bits>=1, recipients*bits<=32 and at least one mode".into());
    }
    Ok(args)
}

fn timed<T>(f: impl FnOnce() -> T) -> (T, f64) {
    let start = Instant::now();
    let out = black_box(f());
    (out, start.elapsed().as_secs_f64() * 1000.0)
}

fn summary(samples: &[f64]) -> Value {
    let mut sorted = samples.to_vec();
    sorted.sort_by(f64::total_cmp);
    let quantile = |p: f64| {
        let index = p * (sorted.len() - 1) as f64;
        let low = index.floor() as usize;
        let high = index.ceil() as usize;
        sorted[low] + (sorted[high] - sorted[low]) * (index - low as f64)
    };
    json!({"median_ms":quantile(0.5),"q1_ms":quantile(0.25),"q3_ms":quantile(0.75),
           "min_ms":sorted[0],"max_ms":sorted[sorted.len()-1],"samples_ms":samples})
}

type Metrics = BTreeMap<&'static str, Vec<f64>>;

fn push(
    metrics: &mut BTreeMap<&'static str, Metrics>,
    mode: &'static str,
    name: &'static str,
    ms: f64,
) {
    metrics
        .entry(mode)
        .or_default()
        .entry(name)
        .or_default()
        .push(ms);
}

fn summaries(metrics: &BTreeMap<&'static str, Metrics>) -> Value {
    metrics
        .iter()
        .map(|(mode, values)| {
            let inner: serde_json::Map<String, Value> = values
                .iter()
                .map(|(name, samples)| (name.to_string(), summary(samples)))
                .collect();
            (mode.to_string(), Value::Object(inner))
        })
        .collect::<serde_json::Map<_, _>>()
        .into()
}

fn median(metrics: &BTreeMap<&'static str, Metrics>, mode: &str, name: &str) -> f64 {
    summary(&metrics[mode][name])["median_ms"].as_f64().unwrap()
}

struct Streams {
    ordinary: StdRng,
    hidden: StdRng,
    split: StdRng,
}

fn run_row(args: &Args, t: usize, bits: usize, seed: u64) -> BenchResult<Value> {
    let (n, m, q, n_min) = (1152, 30357, 67537879, 1120);
    // (q sigma)^2 = n (m + 1), used exactly by the noise table.
    let variance = (n * (m + 1)) as u64;
    let params = Parameters::with_noise_variance(n, m, q, variance, n_min, bits)?;
    let sd = params.noise_sd();
    let population = args.population;
    let mut one_time: BTreeMap<&'static str, Metrics> = BTreeMap::new();
    let mut per_ciphertext: BTreeMap<&'static str, Metrics> = BTreeMap::new();

    // One-time costs. The modes alternate, and every repetition starts from the
    // same seed, so all modes must produce the same keys.
    let mut table_bytes = 0;
    let mut table_distance = 0.0;
    let mut reference: Option<(OrdinaryKeyPair, AnamorphicKeySet)> = None;
    for rep in 0..args.one_time {
        let order: Vec<Mode> = if rep % 2 == 0 {
            args.modes.clone()
        } else {
            args.modes.iter().rev().copied().collect()
        };
        for &mode in &order {
            let name = mode_name(mode);
            let (table, ms) = timed(|| NoiseTable::with_variance(variance, mode));
            let table = table?;
            table_bytes = table.bytes();
            table_distance = table.distance_log2();
            push(&mut one_time, name, "noise_table_build", ms);
            let (pair, ms) = timed(|| keygen_with(params, &mut StdRng::seed_from_u64(seed), mode));
            push(&mut one_time, name, "keygen", ms);
            let pair = pair?;
            let (keys, ms) = timed(|| {
                anamorphic_keygen_with(
                    &pair.public,
                    population,
                    true,
                    &mut StdRng::seed_from_u64(seed + 9),
                    mode,
                )
            });
            push(&mut one_time, name, "anamorphic_keygen", ms);
            let keys = keys?;
            match &reference {
                None => reference = Some((pair, keys)),
                Some((first, first_keys)) => {
                    if pair.public.vector() != first.public.vector()
                        || pair.public.matrix() != first.public.matrix()
                        || pair.secret != first.secret
                        || keys.directory != first_keys.directory
                    {
                        return Err("key generation differs between repetitions or modes".into());
                    }
                }
            }
        }
    }
    let (ordinary, keys) = reference.ok_or("no key generation ran")?;
    // The shared table used by encryption, built before any timed encryption.
    noise_table(&params)?;
    let pk = &ordinary.public;
    let audience: Vec<_> = (0..t).collect();
    let plan = prepare_hidden(pk, &keys.directory, &audience)?;
    if !plan.fiber().full_rank() {
        return Err("benchmark sampled rank-deficient setup; no retries were made".into());
    }
    assert_eq!(plan.fiber().rank(), t * bits);
    for i in 0..args.warmups + args.repetitions {
        let (fresh_plan, ms) = timed(|| prepare_hidden(pk, &keys.directory, &audience));
        assert_eq!(fresh_plan?.fiber().rank(), t * bits);
        if i >= args.warmups {
            push(&mut one_time, "mode_independent", "group_setup", ms);
        }
    }
    let expanded: Vec<_> = keys
        .user_keys
        .iter()
        .map(|key| user_blocks(key, &params).map(UserKey::Expanded))
        .collect::<std::result::Result<_, _>>()?;

    // Per-ciphertext costs. Every mode gets the same RNG streams.
    let mut messages = StdRng::seed_from_u64(seed + 1);
    let mut streams: Vec<Streams> = args
        .modes
        .iter()
        .map(|_| Streams {
            ordinary: StdRng::seed_from_u64(seed + 2),
            hidden: StdRng::seed_from_u64(seed + 3),
            split: StdRng::seed_from_u64(seed + 4),
        })
        .collect();
    let (mut ordinary_checks, mut hidden_bits, mut roundtrips, mut mode_matches) = (0, 0, 0, 0);
    for i in 0..args.warmups + args.repetitions {
        let measured = i >= args.warmups;
        let bit = messages.random_range(0..2);
        let payload: Vec<u8> = (0..bits).map(|_| messages.random_range(0..2)).collect();
        let order: Vec<usize> = if i % 2 == 0 {
            (0..args.modes.len()).collect()
        } else {
            (0..args.modes.len()).rev().collect()
        };
        let mut produced: Vec<Option<[Ciphertext; 3]>> = vec![None; args.modes.len()];
        for &index in &order {
            let mode = args.modes[index];
            let name = mode_name(mode);
            let s = &mut streams[index];
            let ((regular, ordinary_ms), (hidden, hidden_ms)) = if i % 2 == 0 {
                let regular = timed(|| encrypt_with(pk, bit, &mut s.ordinary, mode));
                (
                    regular,
                    timed(|| plan.encrypt_with(bit, &payload, &mut s.hidden, mode)),
                )
            } else {
                let hidden = timed(|| plan.encrypt_with(bit, &payload, &mut s.hidden, mode));
                (
                    timed(|| encrypt_with(pk, bit, &mut s.ordinary, mode)),
                    hidden,
                )
            };
            let (secret, broadcast_ms) = timed(|| plan.sample_secret(&payload, &mut s.split));
            let secret = secret?;
            let (split, dual_regev_ms) = timed(|| {
                let mut noise_seed = [0u8; 32];
                s.split.fill_bytes(&mut noise_seed);
                encrypt_from_seed(pk, bit, &secret, &noise_seed, mode)
            });
            if measured {
                push(
                    &mut per_ciphertext,
                    name,
                    "ordinary_encryption",
                    ordinary_ms,
                );
                push(
                    &mut per_ciphertext,
                    name,
                    "anamorphic_encryption",
                    hidden_ms,
                );
                push(&mut per_ciphertext, name, "broadcast_step", broadcast_ms);
                push(&mut per_ciphertext, name, "dual_regev_part", dual_regev_ms);
            }
            produced[index] = Some([regular?, hidden?, split?]);
        }
        let produced: Vec<[Ciphertext; 3]> = produced.into_iter().map(Option::unwrap).collect();
        for other in &produced[1..] {
            if other != &produced[0] {
                return Err("the execution modes produced different ciphertexts".into());
            }
            mode_matches += 1;
        }
        let [regular, hidden, split] = &produced[0];
        for ct in [regular, hidden, split] {
            assert_eq!(ordinary_decrypt(ct, &ordinary.secret, q)?, bit);
            ordinary_checks += 1;
            let wire = serialize(ct, &params)?;
            assert_eq!(wire.len(), params.ciphertext_bytes());
            assert_eq!(&deserialize(&wire, &params)?, ct);
            roundtrips += 1;
        }
        for &user in &audience {
            for key in [&keys.user_keys[user], &expanded[user]] {
                for ct in [hidden, split] {
                    assert_eq!(hidden_decrypt(ct, key, &params)?, payload);
                    hidden_bits += bits;
                }
            }
        }
        let (decoded, ms) = timed(|| ordinary_decrypt(hidden, &ordinary.secret, q));
        assert_eq!(decoded?, bit);
        ordinary_checks += 1;
        if measured {
            push(
                &mut per_ciphertext,
                "mode_independent",
                "ordinary_decryption",
                ms,
            );
        }
        let kinds = if i % 2 == 0 {
            ["expanded", "seed"]
        } else {
            ["seed", "expanded"]
        };
        for kind in kinds {
            let key = if kind == "expanded" {
                &expanded[0]
            } else {
                &keys.user_keys[0]
            };
            let (decoded, ms) = timed(|| hidden_decrypt(hidden, key, &params));
            assert_eq!(decoded?, payload);
            hidden_bits += bits;
            if measured {
                let metric = if kind == "expanded" {
                    "hidden_decryption_expanded"
                } else {
                    "hidden_decryption_seed"
                };
                push(&mut per_ciphertext, "mode_independent", metric, ms);
            }
        }
    }
    for &mode in &args.modes {
        let name = mode_name(mode);
        eprintln!(
            "t={t}, L={bits}, {name} ({} threads): Enc {:.3} ms, aEnc {:.3} ms (broadcast {:.4} + Dual-Regev {:.3}), aKGen {:.1} ms, KGen {:.1} ms",
            mode.threads(),
            median(&per_ciphertext, name, "ordinary_encryption"),
            median(&per_ciphertext, name, "anamorphic_encryption"),
            median(&per_ciphertext, name, "broadcast_step"),
            median(&per_ciphertext, name, "dual_regev_part"),
            median(&one_time, name, "anamorphic_keygen"),
            median(&one_time, name, "keygen"),
        );
    }
    eprintln!(
        "t={t}, L={bits}: setup {:.3} ms, Dec {:.4} ms, aDec seed {:.4} ms, expanded {:.4} ms",
        median(&one_time, "mode_independent", "group_setup"),
        median(&per_ciphertext, "mode_independent", "ordinary_decryption"),
        median(
            &per_ciphertext,
            "mode_independent",
            "hidden_decryption_seed"
        ),
        median(
            &per_ciphertext,
            "mode_independent",
            "hidden_decryption_expanded"
        ),
    );
    let field_bits = 32 - q.leading_zeros() as usize;
    let codec_bits = params.field_bytes() * 8;
    let directory_elements = population * bits * n;
    let setup_elements = plan.stored_field_elements();
    let threads: BTreeMap<&str, usize> = args
        .modes
        .iter()
        .map(|&mode| (mode_name(mode), mode.threads()))
        .collect();
    Ok(json!({
        "t":t,"L":bits,"d":t*bits,"remaining_dimension":n-t*bits,"seed":seed,"constraint_rank":plan.fiber().rank(),
        "parameters":{"n":n,"m":m,"q":q,"n_min":n_min,"L":bits,"noise_sd":sd,"noise_variance":variance,"population":population},
        "threads":threads,
        "timings":summaries(&per_ciphertext),
        "one_time_ms":summaries(&one_time),
        "checks":{"ordinary_decryptions":ordinary_checks,"hidden_bit_decryptions":hidden_bits,"ciphertext_roundtrips":roundtrips,"cross_mode_identical_ciphertext_triples":mode_matches},
        "sizes":{
            "field_bits":field_bits,"codec_bits_per_field_element":codec_bits,
            "ciphertext_fields":m+1,"ciphertext_bytes":params.ciphertext_bytes(),
            "ordinary_public_key_fields":n*m+n,"ordinary_public_key_bits_codec":(n*m+n)*codec_bits,"ordinary_public_key_bits_packed":(n*m+n)*field_bits,
            "anamorphic_public_key_fields_per_user":bits*n,"anamorphic_public_key_bits_per_user_codec":bits*n*codec_bits,"anamorphic_public_key_bits_per_user_packed":bits*n*field_bits,
            "anamorphic_public_key_bits_all_users_codec":directory_elements*codec_bits,
            "anamorphic_secret_key_bits_expanded":bits*m,"anamorphic_secret_key_bits_seed":SEED_BYTES*8,
            "group_setup_fields":setup_elements,"group_setup_bits_codec":setup_elements*codec_bits,"group_setup_bits_packed":setup_elements*field_bits,
            "group_setup_index_entries":n,
            "noise_table_bytes":table_bytes,"noise_table_distance_log2_per_sample":table_distance
        },
        "scope":"Heuristically screened candidate; table-based noise with the stated per-sample distance bound; variable time; no deployment-security claim"
    }))
}

fn write_checkpoint(output: &mut fs::File, report: &Value) -> BenchResult<()> {
    let bytes = serde_json::to_vec_pretty(report)?;
    output.rewind()?;
    output.write_all(&bytes)?;
    output.set_len(bytes.len() as u64)?;
    output.sync_data()?;
    Ok(())
}

fn create_report(path: &Path, report: &Value) -> BenchResult<fs::File> {
    let mut output = fs::File::create_new(path)?;
    write_checkpoint(&mut output, report)?;
    Ok(output)
}

fn main() -> BenchResult<()> {
    let args = parse_args()?;
    if cfg!(debug_assertions) {
        return Err("run benchmarks with --release".into());
    }
    if let Some(parent) = args.output.parent().filter(|p| !p.as_os_str().is_empty()) {
        fs::create_dir_all(parent)?;
    }
    let mut report = json!({
        "complete":false,"environment":{"os":std::env::consts::OS,"architecture":std::env::consts::ARCH,"logical_cpus":std::thread::available_parallelism()?.get(),"rayon_threads":rayon_threads(),"kernels":"portable Rust, auto-vectorized for the build target"},
        "contract":{"repetitions":args.repetitions,"warmups":args.warmups,"one_time_repetitions":args.one_time,"N":args.population,"lambda":192,"maximum_constraints":32,
            "modes":args.modes.iter().map(|&m| mode_name(m)).collect::<Vec<_>>(),
            "arithmetic":"packed u32 field values; exact u64/u128 integer sums, reduced and combined modulo q by crypto-bigint; noise table built with MPFR (rug); Multi mode splits matrix work over rayon",
            "randomness":"StdRng public deterministic u64 seeds (benchmark only); unbiased random_range for s; noise from a 32-byte seed through ChaCha12 streams and the 256-bit base table",
            "timing":"modes interleaved per iteration with alternating order; Enc/aEnc order alternates; inclusive quartiles; encryption includes coin sampling; the split runs sample_secret, then the seed draw and encrypt_from_seed",
            "excludes":["serialization","verification","network","CPU isolation"]},
        "rows":[]
    });
    let mut output = create_report(&args.output, &report)?;
    for (j, &(t, bits)) in args.rows.iter().enumerate() {
        let row = run_row(&args, t, bits, 2026101200 + j as u64 * 10)?;
        report["rows"].as_array_mut().unwrap().push(row);
        write_checkpoint(&mut output, &report)?;
    }
    report["complete"] = json!(true);
    write_checkpoint(&mut output, &report)?;
    println!(
        "All measured and warmup checks passed. Results: {}",
        args.output.display()
    );
    Ok(())
}

fn rayon_threads() -> usize {
    Mode::Multi.threads()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn existing_output_is_preserved_and_checkpoints_are_complete_json() -> BenchResult<()> {
        let path = std::env::temp_dir().join(format!(
            "basic-benchmark-output-{}-{}.json",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)?
                .as_nanos()
        ));
        let initial = json!({"complete":false,"rows":["unfinished checkpoint"]});
        let mut output = create_report(&path, &initial)?;
        let preserved = fs::read(&path)?;
        assert_eq!(serde_json::from_slice::<Value>(&preserved)?, initial);
        assert!(create_report(&path, &json!({"replacement":true})).is_err());
        assert_eq!(fs::read(&path)?, preserved);

        let completed = json!({"complete":true,"rows":[]});
        write_checkpoint(&mut output, &completed)?;
        assert_eq!(
            serde_json::from_slice::<Value>(&fs::read(&path)?)?,
            completed
        );
        drop(output);
        fs::remove_file(path)?;
        Ok(())
    }
}
