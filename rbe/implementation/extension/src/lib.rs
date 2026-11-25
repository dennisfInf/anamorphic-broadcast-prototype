// Copyright 2026 Anamorphic Broadcast research contributors
// SPDX-License-Identifier: MPL-2.0
//! Affine anamorphic channel on the authors' scalable-RBE crate (rev 5f1e741) with the
//! rank-16 joint profile of the paper's Sec. 7.2 applied, vendored in `../upstream`
//! (`../n16-joint-profile.patch` recreates it from the upstream revision).
//!
//! For a group and first identity branch fixed before setup, generate a [`Family`]
//! and prepare [`Family::context`]. Apply that [`Context`] to fresh [`Coins`]
//! before ordinary encryption. Each recipient recovers the hidden payload with
//! [`Family::decode`].
//!
//! Facts about the host that this layer relies on (all from its source):
//! * `PP` holds the explicit uniform matrices `(A, B)` in NTT form.
//! * Encryption normalises every product once before the inverse NTT, so the root
//!   block is `c_0 = r_0^T A + r_1^T (i (x) G) + e_0`, matching eq. (10), and the
//!   hidden target is `Delta*am` as in eq. (3) (see [`Context::sample`]).
//! * The root block drops `host.root_prune_by` (13) low bits, every other block
//!   [`PRUNE_CT_BY`] (8, or the authors' 16 at rank 7); `decompress` returns canonical
//!   residues.
//! * `setup` already publishes every subregistry's terminator, so the state is
//!   complete before the first registration.
//! * Module rank is the host's `N` (16 by default, set by RBE_RANK at build time);
//!   `g` constraints leave root rank `N - g`, and [`MIN_ROOT_RANK`] keeps it at 12 or
//!   more at rank 16, at the authors' 7 or more at ranks 8 to 15.
use rand::{Rng, distr::Distribution, rng};
use rand_distr::Uniform;
use rayon::prelude::*;
use scalable_rbe::{
    joint_dgs::JointGaussian,
    mat::Mat,
    ntt::MatNTT,
    rbe::{B, D, K, M_B, N, PRUNE_CT_BY, PRUNE_LAST_CT_VECTOR_BY, Q, RBE},
    utils::{decompress, pack, unpack},
};

/// Low-level BLE host of the joint profile; the caller chooses subregistries.
/// Ordinary privacy requires distinct honest public-key values at the
/// occupied selected leaves for each challenge identity. This API does not enforce it.
pub type Host = RBE<JointGaussian, JointGaussian>;
/// The host's public parameters: the explicit matrices `(A, B)` in NTT form.
pub type PublicParameters = scalable_rbe::rbe::PP;
/// Root rank that must stay free after the constraints. The channel's security rests
/// on module-LWE in the `N - g` free root coordinates; Sec. 7.2 of the paper requires
/// `n - g >= 12`, which its Core-SVP estimate (219.0 classical, 198.75 quantum bits)
/// assumes.
///
/// Other ranks come from RBE_RANK at build time. Ranks 8 to 15 keep the authors' rank 7
/// free, so up to `N - 7` recipients keep the security level of the authors' benchmark
/// host. At rank 7, the authors' host itself, up to six constraints are allowed for cost
/// comparisons only: each then takes one of its seven ranks.
pub const MIN_ROOT_RANK: usize = match N {
    16 => 12,
    7 => 1,
    _ => 7,
};
const _: () = assert!(
    N > MIN_ROOT_RANK,
    "the host's module rank leaves no root rank for anamorphic constraints"
);
/// Each recipient and each 256-bit payload block consumes one constraint.
/// Derived from the host's rank, so a smaller host cannot silently weaken the channel:
/// four at rank 16, `N - 7` at ranks 8 to 15, six at rank 7.
pub const MAX_CONSTRAINTS: usize = N - MIN_ROOT_RANK;
/// Covert coefficients are uniform in `[-width/2, width/2)`.
pub const COVERT_KEY_WIDTH: u64 = 1 << 19;
/// Arithmetic helpers take canonical residues modulo Q.
pub fn add(a: u64, b: u64) -> u64 {
    (a + b) % Q
}
pub fn sub(a: u64, b: u64) -> u64 {
    (a + Q - b) % Q
}
pub fn mul(a: u64, b: u64) -> u64 {
    ((a as u128 * b as u128) % Q as u128) as u64
}
fn pow(mut a: u64, mut e: u64) -> u64 {
    let mut r = 1;
    while e > 0 {
        if e & 1 == 1 {
            r = mul(r, a);
        }
        a = mul(a, a);
        e >>= 1;
    }
    r
}
pub fn centred_abs(a: u64) -> u64 {
    a.min(Q - a)
}

/// Ordinary encryption randomness; the channel replaces only the root vector `r[0]`.
/// Public fields and `Clone` support controlled experiments, not safe coin reuse.
/// Publishing encryptions of different messages with the same coins leaks their difference.
#[derive(Clone)]
pub struct Coins {
    pub r: Vec<MatNTT>,
    pub e: Vec<Mat>,
    pub e_leaf: Mat,
    pub e_message: Mat,
}
impl Coins {
    /// The host's own Gaussian samplers, drawn exactly as `RBE::enc` draws its coins.
    /// Sampling includes `r[0]` even when successful embedding later replaces it.
    pub fn sample(host: &Host, depth: usize) -> Self {
        let r = (0..=depth)
            .into_par_iter()
            .map_init(rng, |rng, _| {
                MatNTT::ntt(
                    Mat::sample_disc_gauss(D, 1, N, &host.dgs_sigma, rng),
                    &host.ntt_plan,
                )
            })
            .collect();
        let e = (0..depth)
            .into_par_iter()
            .map_init(rng, |rng, _| {
                Mat::sample_disc_gauss(D, 1, K * host.m_a, &host.dgs_sigma, rng)
            })
            .collect();
        let e_leaf = Mat::sample_disc_gauss(D, 1, M_B, &host.dgs_sigma, &mut rng());
        let e_message = Mat::sample_disc_gauss(D, 1, B, &host.dgs_e_tilde, &mut rng());
        Self {
            r,
            e,
            e_leaf,
            e_message,
        }
    }
    /// The Gaussian root vector `r[0]`, drawn exactly as [`Coins::sample`] and `RBE::enc`
    /// draw it.
    pub fn sample_root(host: &Host) -> MatNTT {
        MatNTT::ntt(
            Mat::sample_disc_gauss(D, 1, N, &host.dgs_sigma, &mut rng()),
            &host.ntt_plan,
        )
    }
    /// aEnc's coins: [`Coins::sample`] without the Gaussian `r[0]`, which stays zero
    /// because the broadcast step ([`Context::apply`]) computes it. Every other coin is
    /// drawn exactly as in [`Coins::sample`]. If `apply` reports a rank failure, set
    /// `r[0]` to [`Coins::sample_root`] before encrypting, which gives ordinary coins.
    pub fn sample_without_root(host: &Host, depth: usize) -> Self {
        let r = (0..=depth)
            .into_par_iter()
            .map_init(rng, |rng, j| {
                if j == 0 {
                    MatNTT {
                        matrix: vec![0; N * D],
                        d: D,
                        nr_rows: 1,
                        nr_columns: N,
                    }
                } else {
                    MatNTT::ntt(
                        Mat::sample_disc_gauss(D, 1, N, &host.dgs_sigma, rng),
                        &host.ntt_plan,
                    )
                }
            })
            .collect();
        let e = (0..depth)
            .into_par_iter()
            .map_init(rng, |rng, _| {
                Mat::sample_disc_gauss(D, 1, K * host.m_a, &host.dgs_sigma, rng)
            })
            .collect();
        let e_leaf = Mat::sample_disc_gauss(D, 1, M_B, &host.dgs_sigma, &mut rng());
        let e_message = Mat::sample_disc_gauss(D, 1, B, &host.dgs_e_tilde, &mut rng());
        Self {
            r,
            e,
            e_leaf,
            e_message,
        }
    }
    /// Encrypt with these coins; sample fresh coins for each new ciphertext.
    pub fn encrypt(
        &self,
        host: &Host,
        pp: &PublicParameters,
        st: &MatNTT,
        id: &str,
        ordinary: Mat,
    ) -> Vec<Mat> {
        host.enc_wo_randomness(
            pp,
            st,
            id,
            ordinary,
            &self.r,
            &self.e,
            &self.e_leaf,
            &self.e_message,
        )
    }
}

#[derive(Clone)]
pub struct Recipient {
    /// One m_C-by-1 secret column per payload block, stored in NTT form.
    pub secrets: Vec<MatNTT>,
    /// Corresponding n-by-1 columns C * secret, in the same NTT representation.
    pub public: Vec<MatNTT>,
}
/// Keys and inactive root columns for one first identity branch and payload length.
pub struct Family {
    pub branch: usize,
    pub bits: usize,
    pub width: u64,
    pub carrier_columns: Vec<usize>,
    pub carrier: MatNTT,
    pub recipients: Vec<Recipient>,
}
impl Family {
    /// Generate recipient keys; capacity is checked when selecting a group.
    /// The proved channel fixes its branch, target group and payload length before host setup.
    /// It covers one family; exposures of keys from additional families are outside its game.
    pub fn generate(
        host: &Host,
        pp: &PublicParameters,
        branch: usize,
        users: usize,
        bits: usize,
        width: u64,
    ) -> Result<Self, String> {
        Self::generate_with_matrix(host, &pp.0, branch, users, bits, width)
    }
    /// [`Family::generate`] for the host matrix `A` given directly (the first matrix of `pp`).
    pub fn generate_with_matrix(
        host: &Host,
        a: &MatNTT,
        branch: usize,
        users: usize,
        bits: usize,
        width: u64,
    ) -> Result<Self, String> {
        if branch >= K || users == 0 || bits == 0 || width != COVERT_KEY_WIDTH {
            return Err("invalid family parameters".into());
        }
        let columns: Vec<_> = (0..K * host.m_a)
            .filter(|j| j / host.m_a != branch)
            .collect();
        let mut c = MatNTT {
            matrix: Vec::new(),
            d: D,
            nr_rows: N,
            nr_columns: columns.len(),
        };
        for &j in &columns {
            c.matrix
                .extend_from_slice(&a.matrix[j * N * D..(j + 1) * N * D]);
        }
        let mut recipients = Vec::new();
        for _ in 0..users {
            let mut secrets = Vec::new();
            let mut public = Vec::new();
            for _ in 0..bits.div_ceil(D) {
                let key = Mat::sample_small(
                    D,
                    columns.len(),
                    1,
                    &Uniform::new(0, width).unwrap(),
                    width / 2,
                    Q,
                    &mut rng(),
                );
                let key = MatNTT::ntt(key, &host.ntt_plan);
                // mul returns raw CRT products; public is in the same CRT basis as the explicit C.
                let pk = c.mul(&key, &host.ntt_plan);
                secrets.push(key);
                public.push(pk);
            }
            recipients.push(Recipient { secrets, public });
        }
        Ok(Self {
            branch,
            bits,
            width,
            carrier_columns: columns,
            carrier: c,
            recipients,
        })
    }
    /// Prepare the fixed group's constraints, retaining ordinary fallback on rank failure.
    /// Multiple families or contexts can be exercised, but the theorem covers one family
    /// with a group and branch fixed before host setup. Keys of additional families are
    /// outside that game; this method does not enforce its timing restriction.
    pub fn context(&self, group: &[usize]) -> Result<Context, String> {
        if group.is_empty() || group.iter().any(|&i| i >= self.recipients.len()) {
            return Err("invalid group".into());
        }
        let mut unique = group.to_vec();
        unique.sort_unstable();
        unique.dedup();
        if unique.len() != group.len() {
            return Err("duplicate recipient".into());
        }
        let nu = self.bits.div_ceil(D);
        let g = group.len() * nu;
        if g > MAX_CONSTRAINTS {
            return Err(format!(
                "at most {MAX_CONSTRAINTS} recipient/payload-polynomial constraints keep root rank {MIN_ROOT_RANK}"
            ));
        }
        // The host's encryption normalises every product before the inverse NTT, so its
        // carrier is C^T*r0 + e and the target is Delta*am (see `Context::targets`).
        // H is indexed by CRT coordinate, then recipient/payload block, then root coordinate.
        let mut h = vec![0; D * g * N];
        for (s, &i) in group.iter().enumerate() {
            for a in 0..nu {
                for row in 0..N {
                    for z in 0..D {
                        h[(z * g + s * nu + a) * N + row] = sub(
                            self.carrier.get_entry(row, 0)[z],
                            self.recipients[i].public[a].get_entry(row, 0)[z],
                        );
                    }
                }
            }
        }
        let mut systems = Vec::new();
        for z in 0..D {
            match System::new(h[z * g * N..(z + 1) * g * N].to_vec(), g) {
                Some(s) => systems.push(s),
                None => {
                    return Ok(Context {
                        branch: self.branch,
                        group: group.to_vec(),
                        bits: self.bits,
                        h,
                        systems: None,
                    });
                }
            }
        }
        Ok(Context {
            branch: self.branch,
            group: group.to_vec(),
            bits: self.bits,
            h,
            systems: Some(systems),
        })
    }
    fn decompress_carrier(&self, host: &Host, ct: &[Mat]) -> (Vec<u64>, MatNTT) {
        let mut root = ct[0].clone();
        decompress(&mut root.matrix, host.root_prune_by, Q);
        let mut v = Mat {
            matrix: Vec::new(),
            d: D,
            nr_rows: 1,
            nr_columns: self.carrier_columns.len(),
        };
        for &j in &self.carrier_columns {
            v.matrix.extend_from_slice(root.get_entry(0, j));
        }
        let reference = v.get_entry(0, 0).to_vec();
        (reference, MatNTT::ntt(v, &host.ntt_plan))
    }

    /// Decode the payload using this recipient's key and the decompressed root block.
    /// Returns bits even for ordinary ciphertexts or other groups; it neither detects
    /// a covert payload nor authenticates it. Any payload framing consumes part of `bits`.
    pub fn decode(&self, host: &Host, recipient: usize, ct: &[Mat]) -> Vec<u8> {
        assert!(recipient < self.recipients.len());
        let (reference, v_ntt) = self.decompress_carrier(host, ct);
        let mut out = Vec::new();
        for k in &self.recipients[recipient].secrets {
            let mut p = v_ntt.mul(k, &host.ntt_plan);
            p.normalise(&host.ntt_plan);
            let p = p.inv_ntt(&host.ntt_plan);
            for (&a, &b) in reference.iter().zip(p.matrix.iter()) {
                let z = sub(a, b);
                let d0 = centred_abs(z);
                let d1 = centred_abs(sub(z, Q / 2));
                out.push(u8::from(d1 < d0));
            }
        }
        out.truncate(self.bits);
        out
    }
    /// Measure the largest residual for a known test payload, outside decoding timings.
    pub fn max_covert_error(
        &self,
        host: &Host,
        recipient: usize,
        ct: &[Mat],
        message: &[u8],
    ) -> u64 {
        let (reference, v) = self.decompress_carrier(host, ct);
        let mut worst = 0;
        for (a, k) in self.recipients[recipient].secrets.iter().enumerate() {
            let mut p = v.mul(k, &host.ntt_plan);
            p.normalise(&host.ntt_plan);
            let p = p.inv_ntt(&host.ntt_plan);
            for z in 0..D {
                if a * D + z < self.bits {
                    worst = worst.max(centred_abs(sub(
                        sub(reference[z], p.matrix[z]),
                        (Q / 2) * message[a * D + z] as u64,
                    )));
                }
            }
        }
        worst
    }
}
/// One CRT coordinate's linear system: rref = transform * original H.
/// Pivot columns are stored in ascending order.
struct System {
    rref: Vec<u64>,
    transform: Vec<u64>,
    pivots: Vec<usize>,
    g: usize,
}
impl System {
    fn new(mut a: Vec<u64>, g: usize) -> Option<Self> {
        let mut transform = vec![0; g * g];
        for j in 0..g {
            transform[j * g + j] = 1;
        }
        let mut pivots = Vec::new();
        for col in 0..N {
            let row = pivots.len();
            if row == g {
                break;
            }
            let pivot = (row..g).find(|&j| a[j * N + col] != 0);
            if pivot.is_none() {
                continue;
            }
            let p = pivot.unwrap();
            for c in 0..N {
                a.swap(row * N + c, p * N + c);
            }
            for c in 0..g {
                transform.swap(row * g + c, p * g + c);
            }
            let inverse = pow(a[row * N + col], Q - 2);
            for c in 0..N {
                a[row * N + c] = mul(a[row * N + c], inverse);
            }
            for c in 0..g {
                transform[row * g + c] = mul(transform[row * g + c], inverse);
            }
            for j in 0..g {
                if j != row {
                    let factor = a[j * N + col];
                    for c in 0..N {
                        a[j * N + c] = sub(a[j * N + c], mul(factor, a[row * N + c]));
                    }
                    for c in 0..g {
                        transform[j * g + c] =
                            sub(transform[j * g + c], mul(factor, transform[row * g + c]));
                    }
                }
            }
            pivots.push(col);
        }
        if pivots.len() != g {
            None
        } else {
            Some(Self {
                rref: a,
                transform,
                pivots,
                g,
            })
        }
    }
    // Uniform free coordinates determine a uniform solution; pivot values follow exactly.
    fn solve(&self, rhs: &[u64], rng: &mut impl Rng) -> Vec<u64> {
        let uniform = Uniform::new(0, Q).unwrap();
        let mut x = vec![0; N];
        for (col, value) in x.iter_mut().enumerate() {
            if !self.pivots.contains(&col) {
                *value = uniform.sample(rng);
            }
        }
        for row in 0..self.g {
            let mut y = 0;
            for (j, &value) in rhs[..self.g].iter().enumerate() {
                y = add(y, mul(self.transform[row * self.g + j], value));
            }
            for (col, &value) in x.iter().enumerate() {
                if !self.pivots.contains(&col) {
                    y = sub(y, mul(self.rref[row * N + col], value));
                }
            }
            x[self.pivots[row]] = y;
        }
        x
    }
}
/// Precomputed constraints for a fixed recipient group and first identity branch.
pub struct Context {
    pub branch: usize,
    pub group: Vec<usize>,
    pub bits: usize,
    h: Vec<u64>,
    systems: Option<Vec<System>>,
}
impl Context {
    /// Replace `r[0]` on success; return false and leave all coins unchanged on rank failure.
    pub fn apply(
        &self,
        host: &Host,
        id: &str,
        coins: &mut Coins,
        message: &[u8],
    ) -> Result<bool, String> {
        if id.is_empty()
            || !id.bytes().all(|b| b >= b'0' && b < b'0' + K as u8)
            || (id.as_bytes()[0] - b'0') as usize != self.branch
            || coins.r.len() != id.len() + 1
        {
            return Err(
                "identity does not match the branch-specific family or sampled path".into(),
            );
        }
        if let Some(r) = self.sample(host, message)? {
            coins.r[0] = r;
            Ok(true)
        } else {
            Ok(false)
        }
    }
    /// Whether every CRT coordinate admits the requested number of independent constraints.
    pub fn full_rank(&self) -> bool {
        self.systems.is_some()
    }
    /// Number of requested constraints g, including when the system is rank deficient.
    pub fn constraint_count(&self) -> usize {
        self.group.len() * self.bits.div_ceil(D)
    }
    /// Field elements held by this precomputation: (`H` itself, per-slot RREF plus transform).
    pub fn stored_field_elements(&self) -> (usize, usize) {
        let systems = self.systems.as_ref().map_or(0, |s| {
            s.iter().map(|x| x.rref.len() + x.transform.len()).sum()
        });
        (self.h.len(), systems)
    }
    /// Pivot indices held by this precomputation (one per constraint and CRT slot).
    pub fn stored_pivots(&self) -> usize {
        self.systems
            .as_ref()
            .map_or(0, |s| s.iter().map(|x| x.pivots.len()).sum())
    }
    fn targets(&self, host: &Host, message: &[u8]) -> Result<MatNTT, String> {
        if message.len() != self.bits || message.iter().any(|&x| x > 1) {
            return Err("invalid covert message".into());
        }
        let nu = self.bits.div_ceil(D);
        // Encryption normalises its products, so the target is Delta*am exactly as in
        // eq. (3); the carrier residual then decodes to Delta*am.
        let mut p = vec![0; nu * D];
        for (i, &b) in message.iter().enumerate() {
            p[i] = (Q / 2) * b as u64;
        }
        Ok(MatNTT::ntt(
            Mat {
                matrix: p,
                d: D,
                nr_rows: 1,
                nr_columns: nu,
            },
            &host.ntt_plan,
        ))
    }
    /// Sample an affine root, or return None for the setup's ordinary-fallback case.
    pub fn sample(&self, host: &Host, message: &[u8]) -> Result<Option<MatNTT>, String> {
        let target = self.targets(host, message)?;
        let Some(systems) = &self.systems else {
            return Ok(None);
        };
        let g = self.constraint_count();
        let nu = self.bits.div_ceil(D);
        let mut r = MatNTT {
            matrix: vec![0; N * D],
            d: D,
            nr_rows: 1,
            nr_columns: N,
        };
        for (z, system) in systems[..D].iter().enumerate() {
            let rhs: Vec<_> = (0..g).map(|i| target.matrix[(i % nu) * D + z]).collect();
            let x = system.solve(&rhs, &mut rng());
            for (j, &value) in x.iter().enumerate() {
                r.matrix[j * D + z] = value;
            }
        }
        Ok(Some(r))
    }
    /// Check H*r against a valid known payload; used by the integration tests.
    pub fn check(&self, host: &Host, r: &MatNTT, message: &[u8]) -> bool {
        let target = self.targets(host, message).unwrap();
        let nu = self.bits.div_ceil(D);
        let g = self.constraint_count();
        for z in 0..D {
            for row in 0..g {
                let mut v = 0;
                for j in 0..N {
                    v = add(v, mul(self.h[(z * g + row) * N + j], r.matrix[j * D + z]));
                }
                if v != target.matrix[(row % nu) * D + z] {
                    return false;
                }
            }
        }
        true
    }
}

/// Per-block byte serialization with the same public profile for both modes.
/// Dimensions follow the known identity length and source constants; no mode flag is written.
pub fn serialize(host: &Host, ct: &[Mat]) -> Vec<u8> {
    let mut out = Vec::new();
    for (j, block) in ct.iter().enumerate() {
        let prune = if j == 0 {
            host.root_prune_by
        } else if j == ct.len() - 1 && host.prune_aggressively {
            PRUNE_LAST_CT_VECTOR_BY
        } else {
            PRUNE_CT_BY
        };
        let bound = if prune == 0 { Q } else { (Q >> prune) + 1 };
        out.extend(pack(&block.matrix, bound));
    }
    out
}
/// Read the profile's blocks for a known identity depth, rejecting malformed coefficients.
pub fn deserialize(host: &Host, depth: usize, input: &[u8]) -> Result<Vec<Mat>, String> {
    if depth == 0 {
        return Err("empty identity".into());
    }
    let mut offset = 0;
    let mut out = Vec::new();
    for j in 0..depth + 2 {
        let cols = if j < depth {
            K * host.m_a
        } else if j == depth {
            M_B
        } else {
            B
        };
        let prune = if j == 0 {
            host.root_prune_by
        } else if j == depth + 1 && host.prune_aggressively {
            PRUNE_LAST_CT_VECTOR_BY
        } else {
            PRUNE_CT_BY
        };
        let bound = if prune == 0 { Q } else { (Q >> prune) + 1 };
        let bits = (64 - (bound - 1).leading_zeros()) as usize;
        let length = (cols * D * bits).div_ceil(8);
        if offset + length > input.len() {
            return Err("truncated ciphertext".into());
        }
        let matrix = unpack(&input[offset..offset + length], bound, cols * D);
        if matrix.iter().any(|&x| x >= bound) {
            return Err("noncanonical ciphertext coefficient".into());
        }
        out.push(Mat {
            matrix,
            d: D,
            nr_rows: 1,
            nr_columns: cols,
        });
        offset += length;
    }
    if offset != input.len() {
        return Err("trailing ciphertext data".into());
    }
    Ok(out)
}
/// Assert a test ciphertext's wire roundtrip and report its serialized length.
pub fn roundtrip(host: &Host, ct: &[Mat]) -> (Vec<Mat>, usize) {
    let bytes = serialize(host, ct);
    let out = deserialize(host, ct.len() - 2, &bytes).unwrap();
    assert_eq!(out, ct);
    (out, bytes.len())
}
/// Actual packed public and secret key roundtrips; secret keys use coefficient-box encoding.
pub fn packed_key_sizes(host: &Host, family: &Family, recipient: usize) -> (usize, usize) {
    let mut public_bytes = 0;
    let mut secret_bytes = 0;
    for public in &family.recipients[recipient].public {
        let bytes = pack(&public.matrix, Q);
        assert_eq!(public.matrix, unpack(&bytes, Q, public.matrix.len()));
        public_bytes += bytes.len();
    }
    for secret in &family.recipients[recipient].secrets {
        let mut t = secret.clone();
        t.normalise(&host.ntt_plan);
        let mut coeff = t.inv_ntt(&host.ntt_plan);
        for x in &mut coeff.matrix {
            *x = add(*x, family.width / 2);
            assert!(*x < family.width);
        }
        let bytes = pack(&coeff.matrix, family.width);
        coeff.matrix = unpack(&bytes, family.width, coeff.matrix.len());
        for x in &mut coeff.matrix {
            *x = sub(*x, family.width / 2);
        }
        assert_eq!(&MatNTT::ntt(coeff, &host.ntt_plan), secret);
        secret_bytes += bytes.len();
    }
    (public_bytes, secret_bytes)
}
/// Alternating-bit fixture, repeated across all ordinary subregistries.
pub fn common_ordinary(bit_pattern: usize) -> Mat {
    let p: Vec<_> = (0..D).map(|i| ((i + bit_pattern) % 2) as u64).collect();
    Mat {
        matrix: p.repeat(B),
        d: D,
        nr_rows: 1,
        nr_columns: B,
    }
}
/// Count disagreements between equal-length test payloads.
pub fn bit_errors(actual: &[u8], expected: &[u8]) -> usize {
    actual.iter().zip(expected).filter(|(a, b)| a != b).count()
}
