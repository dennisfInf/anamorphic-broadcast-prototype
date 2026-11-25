//! Builds the noise base table for an error width and prints its size, bound
//! and statistical-distance bound. With `--dump PATH` it also writes every
//! 256-bit entry as 64 hexadecimal digits, one per line, for comparison with the
//! independent MPFR computation in `tools/noise_table_mpfr.c`.
use basic_anamorphic::{Mode, Parameters, noise_table_with};
use clap::Parser;
use std::{error::Error, fmt::Write as _, fs, path::PathBuf, time::Instant};

/// Builds the noise base table for an error width. Without --sd or --variance it
/// builds the maintained profile, the exact variance 1152 * 30358.
#[derive(Parser)]
struct Args {
    /// Standard deviation S, used as the exact double.
    #[arg(long, conflicts_with = "variance")]
    sd: Option<f64>,
    /// Integer variance V, with standard deviation exactly sqrt(V).
    #[arg(long)]
    variance: Option<u64>,
    /// Build in the multi-thread mode.
    #[arg(long)]
    multi: bool,
    /// Write every entry as 64 hexadecimal digits per line to this file.
    #[arg(long)]
    dump: Option<PathBuf>,
}

fn main() -> Result<(), Box<dyn Error>> {
    let args = Args::parse();
    let (sd, variance) = match (args.sd, args.variance) {
        (Some(sd), _) => (Some(sd), None),
        (None, variance) => (None, Some(variance.unwrap_or(1152u64 * 30358))),
    };
    let mode = if args.multi {
        Mode::Multi
    } else {
        Mode::Single
    };
    let dump = args.dump;
    // Only the error width matters for the table, the rest are placeholders.
    let params = match (sd, variance) {
        (Some(sd), _) => Parameters::new(4, 4, 257, sd, 2, 1)?,
        (None, Some(v)) => Parameters::with_noise_variance(4, 4, 257, v, 2, 1)?,
        (None, None) => unreachable!("one width is always set"),
    };
    let label = match params.noise_variance() {
        Some(v) => format!("variance {v} (sd {:?})", params.noise_sd()),
        None => format!("sd {:?}", params.noise_sd()),
    };
    let start = Instant::now();
    let table = noise_table_with(&params, mode)?;
    let ms = start.elapsed().as_secs_f64() * 1e3;
    println!(
        "{label}: bound {} ({} entries), {} bytes, statistical distance per sample <= 2^{:.2} (tail 2^{:.2}), built in {ms:.1} ms ({mode:?})",
        table.bound(),
        table.entries(),
        table.bytes(),
        table.distance_log2(),
        table.tail_log2(),
    );
    if let Some(path) = dump {
        let mut text = String::with_capacity(table.entries() * 65);
        for k in 0..table.entries() {
            for limb in table.entry(k) {
                write!(text, "{limb:016x}")?;
            }
            text.push('\n');
        }
        fs::write(&path, text)?;
        println!("wrote {} entries to {}", table.entries(), path.display());
    }
    Ok(())
}
