//! Large-sample statistics of the noise that encryption actually adds, for
//! `tools/noise_stats.py`. Every error comes from `noise_from_seed`, the path
//! that `encrypt_from_seed` uses: one 32-byte seed per ciphertext, ChaCha12
//! stream `c` for header coordinates `[1024 c, 1024 c + 1024)` and the next
//! stream for the scalar error.
//!
//! ```sh
//! cargo run --release --example noise_stats -- --ciphertexts 330000 --output stats.json
//! python3 tools/noise_stats.py stats.json
//! ```
//!
//! Options: `--variance V` (default 1152 * 30358, used exactly) or `--sd S`,
//! `--m M` (header length, default 30357), `--threads T` (default 2),
//! `--seed X` (distinguishes runs). Seed `i` of a run is the little-endian
//! encoding of `(i, X)` followed by zeros.

use basic_anamorphic::{NoiseTable, Parameters, noise_from_seed, noise_table};
use clap::Parser;
use serde_json::json;
use std::{error::Error, path::PathBuf, time::Instant};

/// Draws the errors of many encryptions through `noise_from_seed` and writes
/// their statistics for `tools/noise_stats.py`.
#[derive(Parser)]
struct Args {
    /// Integer variance V, used exactly (default 1152 * 30358).
    #[arg(long, conflicts_with = "sd")]
    variance: Option<u64>,
    /// Standard deviation S, used as the exact double.
    #[arg(long)]
    sd: Option<f64>,
    /// Header length.
    #[arg(long, default_value_t = 30357)]
    m: usize,
    /// Worker threads.
    #[arg(long, default_value_t = 2)]
    threads: usize,
    /// Number of seeds, one per ciphertext.
    #[arg(long, default_value_t = 1000)]
    ciphertexts: u64,
    /// Distinguishes runs: seed i is LE64(i) || LE64(X) || zeros.
    #[arg(long = "seed", default_value_t = 1)]
    run_seed: u64,
    /// Result file.
    #[arg(long)]
    output: PathBuf,
}

/// Largest lag of the within-stream correlations.
const LAGS: usize = 4;

struct Acc {
    /// `pos[k]` and `neg[k]` count `e = k` and `e = -k`; `pos[0]` counts zeros.
    pos: Vec<u64>,
    neg: Vec<u64>,
    moments: [i128; 5],
    lag: [i128; LAGS],
    lag_pairs: [u64; LAGS],
    /// Last header error of stream `c` against the first of stream `c + 1`.
    boundary: (i128, u64),
    /// Header error `j` of stream `c` against error `j` of stream `c + 1`.
    aligned: (i128, u64),
    /// First header error of consecutive seeds.
    seeds: (i128, u64),
    /// Header error 0 against the scalar error of the same seed.
    header_scalar: (i128, u64),
    /// Scalar errors only: count, zeros, sum, sum of squares.
    scalar: (u64, u64, i128, i128),
    /// Consecutive pairs within a stream, by sign (-, 0, +) and by quartile class of `|e|`.
    sign_pairs: [[u64; 3]; 3],
    class_pairs: [[u64; 4]; 4],
    /// Per position inside a header stream: count, zeros, sum, sum of squares.
    position: Vec<(u64, u64, i128, i128)>,
}

impl Acc {
    fn new(bound: usize, chunk: usize) -> Self {
        Self {
            pos: vec![0; bound + 1],
            neg: vec![0; bound + 1],
            moments: [0; 5],
            lag: [0; LAGS],
            lag_pairs: [0; LAGS],
            boundary: (0, 0),
            aligned: (0, 0),
            seeds: (0, 0),
            header_scalar: (0, 0),
            scalar: (0, 0, 0, 0),
            sign_pairs: [[0; 3]; 3],
            class_pairs: [[0; 4]; 4],
            position: vec![(0, 0, 0, 0); chunk],
        }
    }

    fn merge(&mut self, other: &Acc) {
        for (a, b) in self.pos.iter_mut().zip(&other.pos) {
            *a += b;
        }
        for (a, b) in self.neg.iter_mut().zip(&other.neg) {
            *a += b;
        }
        for i in 0..5 {
            self.moments[i] += other.moments[i];
        }
        for i in 0..LAGS {
            self.lag[i] += other.lag[i];
            self.lag_pairs[i] += other.lag_pairs[i];
        }
        for (a, b) in [
            (&mut self.boundary, other.boundary),
            (&mut self.aligned, other.aligned),
            (&mut self.seeds, other.seeds),
            (&mut self.header_scalar, other.header_scalar),
        ] {
            a.0 += b.0;
            a.1 += b.1;
        }
        self.scalar.0 += other.scalar.0;
        self.scalar.1 += other.scalar.1;
        self.scalar.2 += other.scalar.2;
        self.scalar.3 += other.scalar.3;
        for i in 0..3 {
            for j in 0..3 {
                self.sign_pairs[i][j] += other.sign_pairs[i][j];
            }
        }
        for i in 0..4 {
            for j in 0..4 {
                self.class_pairs[i][j] += other.class_pairs[i][j];
            }
        }
        for (a, b) in self.position.iter_mut().zip(&other.position) {
            a.0 += b.0;
            a.1 += b.1;
            a.2 += b.2;
            a.3 += b.3;
        }
    }
}

/// Upper ends of the quartile classes of `|e|`: the smallest `k` whose cumulative
/// table value reaches 1/4, 1/2 and 3/4.
fn quartiles(table: &NoiseTable) -> [usize; 3] {
    std::array::from_fn(|i| {
        let target = ((i as u64) + 1) << 62;
        (0..table.entries())
            .find(|&k| table.entry(k)[0] >= target)
            .unwrap_or(table.bound())
    })
}

fn class(e: i64, quartiles: &[usize; 3]) -> usize {
    let a = e.unsigned_abs() as usize;
    quartiles.iter().take_while(|&&q| a > q).count()
}

fn sign(e: i64) -> usize {
    (e.signum() + 1) as usize
}

fn run(
    params: &Parameters,
    first: u64,
    count: u64,
    run_seed: u64,
    quartiles: &[usize; 3],
) -> Result<Acc, String> {
    let table = noise_table(params).map_err(|e| e.to_string())?;
    let chunk = basic_anamorphic::NOISE_CHUNK;
    let mut acc = Acc::new(table.bound(), chunk);
    let mut previous_first: Option<i64> = None;
    for i in first..first + count {
        let mut seed = [0u8; 32];
        seed[..8].copy_from_slice(&i.to_le_bytes());
        seed[8..16].copy_from_slice(&run_seed.to_le_bytes());
        let (errors, scalar) = noise_from_seed(params, &seed).map_err(|e| e.to_string())?;
        for &e in errors.iter().chain([&scalar]) {
            let a = e.unsigned_abs() as usize;
            if e < 0 {
                acc.neg[a] += 1;
            } else {
                acc.pos[a] += 1;
            }
            let w = i128::from(e);
            let mut power = 1i128;
            for m in acc.moments.iter_mut() {
                *m += power;
                power *= w;
            }
        }
        let streams: Vec<&[i64]> = errors.chunks(chunk).collect();
        for (c, stream) in streams.iter().enumerate() {
            for (j, &e) in stream.iter().enumerate() {
                let w = i128::from(e);
                let p = &mut acc.position[j];
                p.0 += 1;
                p.1 += u64::from(e == 0);
                p.2 += w;
                p.3 += w * w;
                for lag in 1..=LAGS {
                    if let Some(&f) = stream.get(j + lag) {
                        acc.lag[lag - 1] += w * i128::from(f);
                        acc.lag_pairs[lag - 1] += 1;
                    }
                }
                if let Some(&f) = stream.get(j + 1) {
                    acc.sign_pairs[sign(e)][sign(f)] += 1;
                    acc.class_pairs[class(e, quartiles)][class(f, quartiles)] += 1;
                }
                if let Some(&f) = streams.get(c + 1).and_then(|next| next.get(j)) {
                    acc.aligned.0 += w * i128::from(f);
                    acc.aligned.1 += 1;
                }
            }
            if let (Some(&last), Some(next)) = (stream.last(), streams.get(c + 1)) {
                acc.boundary.0 += i128::from(last) * i128::from(next[0]);
                acc.boundary.1 += 1;
            }
        }
        let s = i128::from(scalar);
        acc.scalar.0 += 1;
        acc.scalar.1 += u64::from(scalar == 0);
        acc.scalar.2 += s;
        acc.scalar.3 += s * s;
        acc.header_scalar.0 += i128::from(errors[0]) * s;
        acc.header_scalar.1 += 1;
        if let Some(f) = previous_first {
            acc.seeds.0 += i128::from(f) * i128::from(errors[0]);
            acc.seeds.1 += 1;
        }
        previous_first = Some(errors[0]);
    }
    Ok(acc)
}

fn main() -> Result<(), Box<dyn Error>> {
    let Args {
        variance,
        sd,
        m,
        threads,
        ciphertexts,
        run_seed,
        output,
    } = Args::parse();
    if threads == 0 {
        return Err("--threads must be at least 1".into());
    }
    let variance = match sd {
        Some(_) => None,
        None => Some(variance.unwrap_or(1152u64 * 30358)),
    };
    // Only the error width and m matter here, the rest are placeholders.
    let params = match (variance, sd) {
        (Some(v), _) => Parameters::with_noise_variance(4, m, 67537879, v, 2, 1)?,
        (None, Some(s)) => Parameters::new(4, m, 67537879, s, 2, 1)?,
        (None, None) => unreachable!("one width is always set"),
    };
    let table = noise_table(&params)?;
    let quartiles = quartiles(&table);
    let start = Instant::now();
    let per_thread = ciphertexts.div_ceil(threads as u64);
    let parts: Vec<Result<Acc, String>> = std::thread::scope(|scope| {
        let handles: Vec<_> = (0..threads as u64)
            .map(|t| {
                let first = t * per_thread;
                let count = per_thread.min(ciphertexts.saturating_sub(first));
                let (params, quartiles) = (&params, &quartiles);
                scope.spawn(move || run(params, first, count, run_seed, quartiles))
            })
            .collect();
        handles
            .into_iter()
            .map(|h| h.join().expect("worker"))
            .collect()
    });
    let mut total = Acc::new(table.bound(), basic_anamorphic::NOISE_CHUNK);
    for part in parts {
        total.merge(&part?);
    }
    let seconds = start.elapsed().as_secs_f64();
    let samples = total.moments[0];
    eprintln!("{samples} errors from {ciphertexts} seeds in {seconds:.1} s");
    let pairs = |p: (i128, u64)| json!({"sum": p.0.to_string(), "pairs": p.1});
    let report = json!({
        "variance": params.noise_variance(),
        "sd": params.noise_sd(),
        "m": m,
        "chunk": basic_anamorphic::NOISE_CHUNK,
        "ciphertexts": ciphertexts,
        "run_seed": run_seed,
        "threads": threads,
        "seconds": seconds,
        "bound": table.bound(),
        "quartile_ends": quartiles,
        "pos": total.pos,
        "neg": total.neg,
        "moments": total.moments.iter().map(|x| x.to_string()).collect::<Vec<_>>(),
        "lag": (0..LAGS).map(|i| pairs((total.lag[i], total.lag_pairs[i]))).collect::<Vec<_>>(),
        "boundary": pairs(total.boundary),
        "aligned": pairs(total.aligned),
        "seeds": pairs(total.seeds),
        "header_scalar": pairs(total.header_scalar),
        "scalar": {"count": total.scalar.0, "zeros": total.scalar.1,
                   "sum": total.scalar.2.to_string(), "sum2": total.scalar.3.to_string()},
        "sign_pairs": total.sign_pairs,
        "class_pairs": total.class_pairs,
        "position": total.position.iter().map(|p| json!([p.0, p.1, p.2.to_string(), p.3.to_string()])).collect::<Vec<_>>(),
    });
    std::fs::write(&output, serde_json::to_string(&report)?)?;
    eprintln!("wrote {}", output.display());
    Ok(())
}
