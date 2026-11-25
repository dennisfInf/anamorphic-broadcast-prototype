//! Sets the host's module rank from RBE_RANK at build time (default 16, the joint
//! profile). Rank 7 restores the authors' benchmark parameters. Ranks 8 to 15 use the
//! joint profile's pruning and noise width, which it checks at the larger rank 16.
use std::{env, fs, path::Path};

fn main() {
    println!("cargo:rerun-if-env-changed=RBE_RANK");
    let rank: usize = match env::var("RBE_RANK") {
        Ok(value) => value.trim().parse().expect("RBE_RANK must be an integer"),
        Err(_) => 16,
    };
    assert!((7..=16).contains(&rank), "RBE_RANK must lie between 7 and 16");
    // The authors' PRUNE_CT_BY and SIGMA_TILDE (33_519_950_000, rounded to the power of
    // two 2^35 that the joint sampler needs) at rank 7, the joint profile's otherwise.
    let (prune, sigma_tilde) = if rank == 7 {
        (16, "34_359_738_368.0")
    } else {
        (8, "68_719_476_736.0")
    };
    let code = format!(
        "/// Defines the matrix height of `A` and `B` among other vectors (RBE_RANK at build\n\
         /// time: 16 for the joint profile, 7 for the authors' benchmark host).\n\
         pub const N: usize = {rank};\n\
         /// Bits dropped from each non-root ciphertext coefficient.\n\
         pub const PRUNE_CT_BY: usize = {prune};\n\
         /// Width of the message-error law in the same Gaussian convention.\n\
         pub const SIGMA_TILDE: f64 = {sigma_tilde};\n"
    );
    let path = Path::new(&env::var("OUT_DIR").expect("OUT_DIR")).join("profile.rs");
    fs::write(path, code).expect("write the profile constants");
}
