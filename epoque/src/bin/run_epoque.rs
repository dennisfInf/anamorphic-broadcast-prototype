//! One run of Epoque's trapdoor KeyGen and Extr with the anamorphic broadcast of
//! Construction 1 inside, recording phase timings. An ordinary and an anamorphic
//! ciphertext for one identity are written, read back and decrypted with the
//! master shortcut key, with the identity key and (anamorphic) by every recipient.
use clap::Parser;
use epoque_shortcut::trapdoor::{self, IdentityKey, TrapdoorProfile, TrapdoorSetup};
use epoque_shortcut::*;
use rand::RngExt;
use rand_core::UnwrapErr;
use rug::Integer;
use serde_json::{Value, json};
use std::{
    error::Error,
    fs,
    path::{Path, PathBuf},
    time::Instant,
};
type RunResult<T> = std::result::Result<T, Box<dyn Error>>;

/// Users of aKGen, hidden bits per user, and the recipients S.
const USERS: usize = 4;
const BITS: usize = 2;
const GROUP: [usize; 4] = [0, 1, 2, 3];

fn event(
    report: &mut Value,
    path: &Path,
    name: &str,
    seconds: Option<f64>,
    details: Value,
) -> RunResult<()> {
    let mut value = details;
    value["phase"] = json!(name);
    if let Some(seconds) = seconds {
        value["seconds"] = json!(seconds);
    }
    println!("{value}");
    report["events"].as_array_mut().unwrap().push(value);
    fs::write(path, serde_json::to_vec_pretty(report)?)?;
    Ok(())
}

/// KeyGen on three entries with arbitrary integers: B D2 = 0 and A0 D1 + A1 D2 = U,
/// so F_id D = U for every identity.
fn check_setup(setup: &Setup) -> RunResult<()> {
    let p = setup.public.profile();
    for (row, col) in [
        (0, 0),
        (p.n() / 2, p.ordinary_bits() / 2),
        (p.n() - 1, p.ordinary_bits() - 1),
    ] {
        let mut kernel = Integer::new();
        let mut image = Integer::new();
        for k in 0..p.m() {
            let d1 = setup.shortcut.d1_transposed()[col * p.m() + k];
            let d2 = setup.shortcut.d2_transposed()[col * p.m() + k];
            kernel += Integer::from(setup.public.b()[row * p.m() + k]) * d2;
            image += Integer::from(setup.public.a0()[row * p.m() + k]) * d1;
            image += Integer::from(setup.public.a1()[row * p.m() + k]) * d2;
        }
        if integer_mod(&kernel, p.q()) != 0
            || integer_mod(&image, p.q()) != setup.public.u()[row * p.ordinary_bits() + col]
        {
            return Err("independent setup identity check failed".into());
        }
    }
    Ok(())
}

/// Extr on three entries: A0 e1 + A1 e2 + H(id) (B e2) = U, with exact i128 sums
/// (each term is below 2^94 and m below 2^17) and H(id) from `Identity::matrix`.
fn check_identity_key(pk: &PublicKey, key: &IdentityKey) -> RunResult<()> {
    let p = pk.profile();
    let (n, m, q, jb) = (p.n(), p.m(), p.q(), p.ordinary_bits());
    let h = key.identity().matrix();
    let reduce = |x: i128| x.rem_euclid(i128::from(q)) as u64;
    for (row, col) in [(0, 0), (n / 2, jb / 2), (n - 1, jb - 1)] {
        let e1 = &key.e1_transposed()[col * m..(col + 1) * m];
        let e2 = &key.e2_transposed()[col * m..(col + 1) * m];
        let dot = |a: &[u64], x: &[i32]| -> i128 {
            a.iter()
                .zip(x)
                .map(|(&a, &x)| i128::from(a) * i128::from(x))
                .sum()
        };
        let be2: Vec<u64> = (0..n)
            .map(|t| reduce(dot(&pk.b()[t * m..(t + 1) * m], e2)))
            .collect();
        let hbe2 = h[row * n..(row + 1) * n]
            .iter()
            .zip(&be2)
            .fold(0u128, |acc, (&x, &y)| {
                (acc + u128::from(x) * u128::from(y)) % u128::from(q)
            }) as u64;
        let left = reduce(
            dot(&pk.a0()[row * m..(row + 1) * m], e1)
                + dot(&pk.a1()[row * m..(row + 1) * m], e2)
                + i128::from(hbe2),
        );
        if left != pk.u()[row * jb + col] {
            return Err("independent identity-key check failed".into());
        }
    }
    Ok(())
}

/// aKGen for four users, aEnc's preparation for S = all four, then one ordinary and
/// one anamorphic ciphertext for `id`. Each is written, read back and decrypted with
/// the shortcut key, with the identity key and, for the anamorphic one, by every
/// recipient with its own key.
fn exercise(
    setup: &Setup,
    id: &Identity,
    id_key: &IdentityKey,
    output: &Path,
    report: &mut Value,
    report_path: &Path,
    rng: &mut UnwrapErr<OsBits>,
) -> RunResult<()> {
    let pk = &setup.public;
    let p = pk.profile();
    let before = Instant::now();
    let (apk, keys) = covert_keygen(pk, USERS, BITS, rng)?;
    event(
        report,
        report_path,
        "anamorphic_keygen",
        Some(before.elapsed().as_secs_f64()),
        json!({"users":USERS,"bits_per_user":BITS,"ask_bits_per_user":BITS*p.m(),"apk_vectors":USERS*BITS}),
    )?;
    let before = Instant::now();
    let plan = prepare_hidden(pk, &apk, &GROUP)?;
    event(
        report,
        report_path,
        "anamorphic_setup",
        Some(before.elapsed().as_secs_f64()),
        json!({"full_rank":plan.sampler().full_rank(),"recipients":GROUP.len(),"bits":BITS,"constraints":GROUP.len()*BITS}),
    )?;
    let ordinary: Vec<u8> = (0..p.ordinary_bits()).map(|j| (j % 2) as u8).collect();
    let hidden = [1, 0];
    for mode in ["ordinary", "anamorphic"] {
        let before = Instant::now();
        let ct = if mode == "ordinary" {
            encrypt_ordinary(pk, id, &ordinary, rng)?
        } else {
            plan.encrypt(id, &ordinary, &hidden, rng)?
        };
        event(
            report,
            report_path,
            &format!("{mode}_encryption"),
            Some(before.elapsed().as_secs_f64()),
            json!({"includes":"secret, exact rounded errors, independent signs, arithmetic","rank_fallback":mode=="anamorphic"&&!plan.sampler().full_rank()}),
        )?;
        let before = Instant::now();
        let wire = encode(p, &ct)?;
        fs::write(output.join(format!("{mode}.bin")), &wire)?;
        let decoded = decode(p, &fs::read(output.join(format!("{mode}.bin")))?)?;
        if decoded != ct {
            return Err("ciphertext wire mismatch".into());
        }
        event(
            report,
            report_path,
            &format!("{mode}_wire"),
            Some(before.elapsed().as_secs_f64()),
            json!({"bytes":wire.len(),"passed":true}),
        )?;
        let before = Instant::now();
        if decrypt_ordinary(&setup.shortcut, &decoded)? != ordinary {
            return Err("ordinary decryption failed".into());
        }
        event(
            report,
            report_path,
            &format!("{mode}_decryption"),
            Some(before.elapsed().as_secs_f64()),
            json!({"key":"master shortcut key","ordinary_bits_checked":p.ordinary_bits(),"ordinary_bits_match":true}),
        )?;
        let before = Instant::now();
        if trapdoor::decrypt_identity(id_key, &decoded)? != ordinary {
            return Err("identity-key decryption failed".into());
        }
        event(
            report,
            report_path,
            &format!("{mode}_identity_decryption"),
            Some(before.elapsed().as_secs_f64()),
            json!({"key":"identity key","ordinary_bits_checked":p.ordinary_bits(),"ordinary_bits_match":true}),
        )?;
        if mode == "anamorphic" {
            let before = Instant::now();
            let received = GROUP
                .iter()
                .map(|&i| decrypt_covert(&keys[i], &decoded))
                .collect::<Result<Vec<_>>>()?;
            if received.iter().any(|bits| bits != &hidden) || !plan.sampler().full_rank() {
                return Err("hidden decryption or requested full-rank delivery failed".into());
            }
            event(
                report,
                report_path,
                "anamorphic_hidden_decryption",
                Some(before.elapsed().as_secs_f64()),
                json!({"recipients":GROUP,"hidden_bits":received,"hidden_bits_match":true}),
            )?;
        }
    }
    Ok(())
}

/// Epoque's trapdoor KeyGen with the master shortcut key, then Extr for `id_kind`.
fn execute(
    tp: TrapdoorProfile,
    id_kind: &str,
    output: &Path,
    report: &mut Value,
    report_path: &Path,
) -> RunResult<()> {
    let started = Instant::now();
    let mut rng = UnwrapErr(OsBits::default());
    let mut last = Instant::now();
    let mut progress_error = None;
    // Checkpoint errors are propagated after the callback; cryptographic code does not write files.
    let generated = trapdoor::setup_trapdoor(&tp, &mut rng, |phase, count| {
        if progress_error.is_some() {
            return;
        }
        let seconds = last.elapsed().as_secs_f64();
        last = Instant::now();
        if let Err(error) = event(
            report,
            report_path,
            phase,
            Some(seconds),
            json!({"count":count}),
        ) {
            progress_error = Some(error);
        }
    })?;
    if let Some(error) = progress_error {
        return Err(error);
    }
    let p = tp.profile();
    event(
        report,
        report_path,
        "setup_complete",
        Some(started.elapsed().as_secs_f64()),
        json!({"trapgen_attempts":generated.trapgen_attempts,"key_resamples":generated.key_resamples,"dense_matrix_bytes":3*p.n()*p.m()*8,"master_trapdoor_bytes":tp.trapdoor_rows()*tp.gadget_columns()+tp.trapdoor_rows()*tp.trapdoor_rows()*8,"shortcut_key_memory_bytes":2*p.m()*p.ordinary_bits()*4}),
    )?;
    let before = Instant::now();
    check_setup(&generated.setup)?;
    event(
        report,
        report_path,
        "independent_setup_checks",
        Some(before.elapsed().as_secs_f64()),
        json!({"entries":3,"arithmetic":"rug arbitrary integers","passed":true}),
    )?;
    let pk = &generated.setup.public;
    let id = if id_kind == "unit" {
        pk.unit_identity()
    } else {
        let coordinates: Vec<u64> = (0..p.n()).map(|_| rng.random_range(0..p.q())).collect();
        pk.identity(&coordinates)?
    };
    report["profile"]["identity"] =
        json!({"kind":id_kind,"first_coordinates":id.vector()[..p.n().min(8)]});
    let before = Instant::now();
    let id_key = generated.master.extract(pk, &id, &mut rng)?;
    event(
        report,
        report_path,
        "identity_extraction",
        Some(before.elapsed().as_secs_f64()),
        json!({"identity":id_kind,"key_resamples":id_key.resamples}),
    )?;
    let before = Instant::now();
    check_identity_key(pk, &id_key)?;
    event(
        report,
        report_path,
        "independent_identity_key_checks",
        Some(before.elapsed().as_secs_f64()),
        json!({"entries":3,"arithmetic":"exact i128 sums reduced modulo q","passed":true}),
    )?;
    let TrapdoorSetup { setup, master, .. } = generated;
    // T_A0 is not needed after extraction: whoever holds it can read every hidden message.
    drop(master);
    exercise(&setup, &id, &id_key, output, report, report_path, &mut rng)?;
    drop(id_key);
    drop(setup);
    report["success"] = json!(true);
    report["elapsed_seconds"] = json!(started.elapsed().as_secs_f64());
    report["dense_matrices"] = json!("heap allocations released; no dense files created");
    Ok(())
}

/// Runs Epoque's trapdoor KeyGen, Extr and both encryption modes once and records
/// their phase timings.
#[derive(Parser)]
#[command(
    after_help = "Small: n16, m671, J8, g8 (test-sized, no security). Full: n1360, m51325, J128, g8.\nFour recipients of two hidden bits each. Uses buffered OS randomness, f64 lattice samplers\nfor the keys and certified samplers for the encryption errors. The full run allocates\nabout 1.7 GB of public matrices; use --release."
)]
struct Args {
    /// Parameter profile.
    #[arg(long, value_parser = ["small", "full"], default_value = "small")]
    profile: String,
    /// Identity to encrypt to: uniformly random in Z_q^n, or the unit identity (H = I).
    #[arg(long, value_parser = ["random", "unit"], default_value = "random")]
    identity: String,
    /// New directory for the results; previous evidence is never overwritten.
    #[arg(long)]
    output: PathBuf,
}

fn main() -> RunResult<()> {
    let Args {
        profile,
        identity,
        output,
    } = Args::parse();
    let tp = match profile.as_str() {
        "small" => TrapdoorProfile::small(),
        "full" => TrapdoorProfile::full(),
        _ => return Err("profile must be small or full".into()),
    };
    let p = tp.profile();
    if profile == "full" && cfg!(debug_assertions) {
        return Err("full profile requires --release".into());
    }
    fs::create_dir(&output)?;
    let report_path = output.join("timings.json");
    let mut report = json!({"success":false,"profile":{"name":profile,"n":p.n(),"m":p.m(),"q":p.q().to_string(),"J":p.ordinary_bits(),"g":p.maximum_constraints(),"C":p.c_bound(),"key_width_s":tp.width(),"gadget_base":tp.base(),"rounding_width":tp.rounding_width(),"gadget_width":tp.gadget_width(),"identity":{"kind":identity}},"anamorphic":{"users":USERS,"recipients":GROUP,"bits_per_recipient":BITS,"keys":"binary (Construction 1)"},"native_error_standard_deviation":{"numerator":p.q().to_string(),"denominator":p.denominator().to_string()},"scope":"Epoque trapdoor KeyGen (MP12 gadget trapdoors for A0 and B), master shortcut key (Section 5.2.1), Extr for one identity in Z_q^n, and Construction 1 embedded through A0; no trustees or voting","random_source":"buffered OS bytes; no seeded matrix/sign PRG","arithmetic":"exact integer sums reduced by crypto-bigint; GMP/MPFR through rug for primality and unbounded values; f64 lattice Gaussian sampling with exact preimage checks","environment":{"os":std::env::consts::OS,"arch":std::env::consts::ARCH},"events":[]});
    fs::write(&report_path, serde_json::to_vec_pretty(&report)?)?;
    let result = execute(tp, &identity, &output, &mut report, &report_path);
    if let Err(error) = &result {
        report["error"] = json!(error.to_string());
    }
    fs::write(&report_path, serde_json::to_vec_pretty(&report)?)?;
    println!(
        "{}",
        json!({"phase":"finished","success":report["success"],"seconds":report["elapsed_seconds"],"report":report_path})
    );
    result
}
