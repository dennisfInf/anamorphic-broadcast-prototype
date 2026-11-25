//! The noise base table against an independent MPFR evaluation, and the
//! sampler against its own table.
//!
//! `tools/noise_table_mpfr.c` computes `floor(2^256 erf((k+1/2)/(s sqrt 2)) /
//! erf((B+1/2)/(s sqrt 2)))` at 1024-bit precision. The digests below are
//! SHA3-256 of its output (64 hex digits per entry, one line each), which must
//! equal the same rendering of the Rust table. `s` is either a double, taken
//! exactly, or the exact square root of an integer variance (`var:V`).

use basic_anamorphic::*;
use sha3::{Digest, Sha3_256};

fn table(sd: f64) -> std::sync::Arc<NoiseTable> {
    noise_table(&Parameters::new(4, 4, 257, sd, 2, 1).unwrap()).unwrap()
}

fn exact_table(variance: u64) -> std::sync::Arc<NoiseTable> {
    noise_table(&Parameters::with_noise_variance(4, 4, 257, variance, 2, 1).unwrap()).unwrap()
}

fn digest(table: &NoiseTable) -> String {
    let mut hasher = Sha3_256::new();
    for k in 0..table.entries() {
        let line: String = table
            .entry(k)
            .iter()
            .map(|limb| format!("{limb:016x}"))
            .collect();
        hasher.update(line.as_bytes());
        hasher.update(b"\n");
    }
    hasher
        .finalize()
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect()
}

#[test]
fn small_tables_equal_the_mpfr_reference() {
    let t = table(0.25);
    assert_eq!(t.bound(), 5);
    let expected = [
        "f45a183e9b13d03318bb3e531cc1a69b33bf93449a56854e7720dd48980f4869",
        "fffffff7867887fb695be0f0556c1cbf56ba927a1e738aeacc40d493e03e8a38",
        "ffffffffffffffffffed938a17b04ef294c6bdc38ceafb5110344e8625fbfdaf",
        "ffffffffffffffffffffffffffffffffffffa70377632069c34de3a3d60d8a5b",
        "fffffffffffffffffffffffffffffffffffffffffffffffffffffffffffc8ecf",
    ];
    for (k, hex) in expected.iter().enumerate() {
        let got: String = t.entry(k).iter().map(|l| format!("{l:016x}")).collect();
        assert_eq!(&got, hex, "entry {k}");
    }
    let t = table(3.0);
    assert_eq!((t.bound(), t.entries()), (56, 56));
    assert_eq!(
        digest(&t),
        "46d80c63e97e9b258e85ff29119a164d26ded1861a01b7e30543b2715e34dead"
    );
    let t = table(77.5);
    assert_eq!(t.bound(), 1458);
    assert_eq!(
        digest(&t),
        "3c8bd553fbf077f1e89ea46ae33efc4b1eba37cdc7c559195eb637a661ad1a26"
    );
}

#[test]
#[cfg_attr(debug_assertions, ignore = "large tables: run with --release")]
fn large_tables_equal_the_mpfr_reference() {
    let t = table(1000.0);
    assert_eq!(t.bound(), 18818);
    assert_eq!(
        digest(&t),
        "a4b7f97d545dc03085d74b553d9c415e2c69e3c692b66283a8b710e17bea010c"
    );
    // The maintained profile, s = sqrt(1152 * 30358) exactly (var:34972416).
    let t = exact_table(1152 * 30358);
    assert_eq!((t.bound(), t.variance()), (111_285, Some(34_972_416)));
    assert_eq!(
        digest(&t),
        "bd264f728cc90484e4bb26f2fa99f3cffce1557dd778cf4eb770ec2e13f9fbd0"
    );
    assert!(t.distance_log2() < -239.2);
    assert!(t.tail_log2() <= -260.0);
    // The same width rounded to the nearest double, 2^-57.7 larger.
    let t = table((1152.0f64 * 30358.0).sqrt());
    assert_eq!(t.bound(), 111_285);
    assert_eq!(
        digest(&t),
        "595cc57cae245e39f6ff38db8da5245340177f31747c9d5c54f52120586b803c"
    );
}

#[test]
fn samples_follow_the_table() {
    let sd = 77.5;
    let p = Parameters::new(4, 50_000, 67537879, sd, 2, 1).unwrap();
    let t = noise_table(&p).unwrap();
    let mut counts = vec![0u64; t.bound() + 1];
    let (mut negative, mut nonzero, mut square) = (0u64, 0u64, 0f64);
    let seeds = if cfg!(debug_assertions) { 8 } else { 40 };
    for s in 0..seeds {
        let mut seed = [0u8; 32];
        seed[..8].copy_from_slice(&(s as u64).to_le_bytes());
        let (errors, scalar) = noise_from_seed(&p, &seed).unwrap();
        for &e in errors.iter().chain([&scalar]) {
            counts[e.unsigned_abs() as usize] += 1;
            square += (e * e) as f64;
            if e != 0 {
                nonzero += 1;
                negative += u64::from(e < 0);
            }
        }
    }
    let total: u64 = counts.iter().sum();
    // Probabilities of |e| = k from the leading words of the cumulative entries.
    let cumulative = |k: usize| -> f64 {
        if k >= t.entries() {
            1.0
        } else {
            t.entry(k)[0] as f64 / 2f64.powi(64)
        }
    };
    let (mut chi2, mut bins, mut start) = (0.0, 0usize, 0usize);
    while start <= t.bound() {
        let mut stop = start;
        while stop < t.bound()
            && (cumulative(stop)
                - if start == 0 {
                    0.0
                } else {
                    cumulative(start - 1)
                })
                * (total as f64)
                < 200.0
        {
            stop += 1;
        }
        let low = if start == 0 {
            0.0
        } else {
            cumulative(start - 1)
        };
        let expected = (cumulative(stop) - low) * total as f64;
        let observed: u64 = counts[start..=stop].iter().sum();
        if expected > 0.0 {
            chi2 += (observed as f64 - expected).powi(2) / expected;
            bins += 1;
        }
        start = stop + 1;
    }
    let dof = (bins - 1) as f64;
    assert!(
        chi2 < dof + 6.0 * (2.0 * dof).sqrt(),
        "chi2 {chi2} with {dof} degrees of freedom"
    );
    let half = nonzero as f64 / 2.0;
    assert!((negative as f64 - half).abs() < 6.0 * half.sqrt() / std::f64::consts::SQRT_2);
    // Var(round(N(0, s^2))) = s^2 + 1/12 up to terms of order exp(-2 pi^2 s^2).
    let variance = square / total as f64;
    let expected_variance = sd * sd + 1.0 / 12.0;
    let tolerance = 6.0 * expected_variance * (2.0 / total as f64).sqrt();
    assert!(
        (variance - expected_variance).abs() < tolerance,
        "variance {variance}"
    );
}
