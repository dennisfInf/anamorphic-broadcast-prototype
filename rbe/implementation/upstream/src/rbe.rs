// Copyright 2026 Jan Niklas Siemer
//
// This file is part of scalable_rbe.
//
// scalable_rbe is free software: you can redistribute it and/or modify it under
// the terms of the Mozilla Public License Version 2.0 as published by the
// Mozilla Foundation. See <https://mozilla.org/en-US/MPL/2.0/>.

//! Implementation of the Registration-Based Encryption scheme.

use crate::{
    dgs::DGS,
    joint_dgs::JointGaussian,
    mat::Mat,
    ntt::MatNTT,
    utils::{compress, decompress, pack, unpack},
};
use concrete_ntt::prime64::Plan;
use rand::rng;
use rand_distr::Uniform;
use rayon::iter::{IntoParallelIterator, ParallelIterator};
use sled::{Batch, Db, Tree};
use std::{mem::MaybeUninit, vec};

// Parameters
/// Degree of modulus-polynomial (X^d + 1) mod [`Q`]
pub const D: usize = 256;
/// Modulus applied to each coefficient
pub const Q: u64 = 281_474_976_694_273;
// N, PRUNE_CT_BY and SIGMA_TILDE come from build.rs (RBE_RANK, default 16).
include!(concat!(env!("OUT_DIR"), "/profile.rs"));
/// Matrix width of B
pub const M_B: usize = N;

// Max. Integer hidden per coefficient
// pub const P: u64 = 2; // P = 2 is now fixed for this implementation
/// Identity arity
pub const K: usize = 3;
/// Upper bound for the number of dictionaries in `reg`
pub const B: usize = 30;
/// Base of the pruned gadget matrix `G~`
pub const GADGET_BASE: u64 = 65_536;
/// Number of entries each gadget vector is pruned by
pub const PRUNE_GADGET_BY: usize = 1;
/// Number of lower-order bits to remove from each coefficient of the last vector of ciphertext
/// Retained from the source prototype; aggressive pruning is disabled in this profile.
pub const PRUNE_LAST_CT_VECTOR_BY: usize = 43;

/// Inclusive sampling maximum; subtracting 32767 gives U(R_b)=(-b/2,b/2].
pub const SMALL_RND_B: u64 = 65_535;
/// Width of the small-noise law exp(-pi*x^2/sigma^2), not its standard deviation.
pub const SIGMA: f64 = 16.0;
// For q = 1 mod 4, nearest-center decoding with ties to zero.
fn decode_bit(v: u64) -> u64 {
    u64::from(v > Q / 4 && v <= 3 * Q / 4)
}

/// Implements the Registration-Based-Encryption scheme.
///
/// Holds the selected profile, transform plan and precomputed samplers.
/// [`Self::setup`] returns the public matrices and initial registry state.
///
/// Attributes:
/// - `prune_aggressively`: Determines whether the last vector of each ciphertext is pruned more aggressively
/// - `ntt_plan`: Contains values to perform NTT transformation and vice versa
/// - `m_a`: Width of each branch block of `A`; the full width is `K * m_a`
/// - `log_b_q`: Contains `ceil(log_{GADGET_BASE}(q))`
/// - `mat_g`: Gadget matrix in NTT form
/// - `dgs_sigma`: Discrete Gaussian sampler for vectors `r` and `e`
/// - `dgs_e_tilde`: Discrete Gaussian sampler for vector `e_tilde`
///
/// # Example
/// Run in a disposable working directory: this example uses and clears `registry_db`.
/// ```
/// use scalable_rbe::{rbe::{RBE, B, D, Q}, mat::Mat};
/// use rand::rng;
/// use rand_distr::Uniform;
/// # use scalable_rbe::utils::drop_db;
/// # drop_db();
///
/// let id = "01210";
/// let t = rand::random_range(0..B);
///
/// let drbe = RBE::init(false);
/// let (pp, _st, reg) = drbe.setup();
/// let (pk, sk) = drbe.keygen(&pp);
/// let st = drbe.update(&reg, &pp, t, id, &pk);
/// let wit = drbe.witness_gen(&reg, t, id).unwrap();
///
/// let msg_dist = Uniform::new(0, 2).unwrap();
/// let vec_m_t = Mat::sample_small(D, 1, B, &msg_dist, 0, Q, &mut rng());
///
/// let ct = drbe.enc(&pp, &st, id, vec_m_t.clone());
///
/// let dec = drbe.dec(&sk, t, &wit, &ct);
///
/// assert_eq!(vec_m_t.get_entry(0, t), dec.matrix);
/// # drop(reg);
/// # drop_db();
/// ```
pub struct RBE<S: DGS, T: DGS> {
    pub prune_aggressively: bool,
    /// Bits dropped from each root coefficient in both encryption modes.
    pub root_prune_by: usize,
    pub ntt_plan: Plan,
    pub m_a: usize,
    pub log_b_q: usize,
    pub mat_g: MatNTT,
    pub dgs_sigma: S,
    pub dgs_e_tilde: T,
}

// Type definitions
/// Uniform public matrices `(A, B)` in NTT form; no setup seed is exposed.
pub type PP = (MatNTT, MatNTT);
type ST = MatNTT;
#[allow(clippy::upper_case_acronyms)]
type REG = Db;
type PK = MatNTT;
type SK = MatNTT;
type CT = Vec<Mat>;
#[allow(clippy::upper_case_acronyms)]
type WIT = Vec<MatNTT>;

impl RBE<JointGaussian, JointGaussian> {
    /// Initialises the [`RBE`] scheme with a negacyclic NTT-[`Plan`] `X^D + 1 mod Q`
    /// and `m_a := n * (ceil(log_{GADGET_BASE}(q)) - PRUNE_GADGET_BY)`.
    /// Furthermore, it generates and stores the gadget matrix in NTT form and precomputes
    /// any required tables for discrete Gaussian samplers.
    ///
    /// This joint profile uses certified Gaussian approximations and disables
    /// the source prototype's aggressive final-component compression.
    pub fn init(prune_aggressively: bool) -> Self {
        assert!(
            !prune_aggressively,
            "aggressive pruning is outside the joint profile"
        );
        let ntt_plan = Plan::try_new(D, Q).unwrap();

        // ceil(log_b(q))
        let log_b_q = (Q as f64).log(GADGET_BASE as f64).ceil() as usize;

        // m_A = (ceil(log_base q) - prune_gadget_by) * n
        let m_a = (log_b_q - PRUNE_GADGET_BY) * N;

        // Pre-generate (i_j^t ⊗ G˜) for all j \in [K] and store it in struct
        let mat_g = Mat::gen_pruned_gadget(log_b_q, GADGET_BASE, PRUNE_GADGET_BY, N, D);
        let mat_g = MatNTT::ntt(mat_g, &ntt_plan);

        let dgs_sigma = JointGaussian::init(SIGMA, Q);
        let dgs_e_tilde = JointGaussian::init(SIGMA_TILDE, Q);

        Self {
            prune_aggressively,
            root_prune_by: 13,
            ntt_plan,
            m_a,
            log_b_q,
            mat_g,
            dgs_sigma,
            dgs_e_tilde,
        }
    }

    /// Samples explicit uniform matrices `A ∈ R_q^{n × K m_A}` and `B ∈ R_q^{n × m_B}`.
    /// The returned state contains every empty-tree terminator, so encryption can
    /// run before registration. The registry stores these roots and their
    /// negative gadget decompositions in NTT form.
    ///
    /// # Panics
    /// Requires a fresh or empty `registry_db` in the working directory. An
    /// existing nonempty registry is rejected before any stored record is changed.
    /// Opening or writing the database can also fail.
    pub fn setup(&self) -> (PP, ST, REG) {
        let mat_a = MatNTT::sample_uniform(D, N, K * self.m_a, Q, &mut rng());
        let mat_b = MatNTT::sample_uniform(D, N, M_B, Q, &mut rng());

        // pp = (A, B)
        // The initial public state contains all empty-tree terminators.
        let mut st = MatNTT {
            matrix: Vec::with_capacity(B * N * D),
            d: D,
            nr_rows: N,
            nr_columns: B,
        };
        // reg = empty dictionary
        let reg = sled::Config::default()
            .path("registry_db")
            .flush_every_ms(Some(1_000_000))
            .mode(sled::Mode::HighThroughput)
            .use_compression(false)
            .open()
            .expect("Failed to open database.");
        assert!(
            reg.tree_names().iter().all(|name| reg
                .open_tree(name)
                .expect("Failed to inspect registry.")
                .is_empty()),
            "setup requires a fresh registry; existing records were not changed"
        );

        for t in 0..B {
            // y_t^{term} ← R_q^{n}
            let vec_t_term = Mat::sample_uniform(D, N, 1, &Uniform::new(0, Q).unwrap(), &mut rng());
            let ntt_vec_t_term = MatNTT::ntt(vec_t_term.clone(), &self.ntt_plan);
            st.matrix.extend_from_slice(&ntt_vec_t_term.matrix);
            let mut decomp_vec_t_term =
                vec_t_term.pruned_binary_decomp(self.log_b_q, GADGET_BASE, PRUNE_GADGET_BY, Q);
            decomp_vec_t_term.negate(Q);
            let ntt_decomp_vec_t_term = MatNTT::ntt(decomp_vec_t_term, &self.ntt_plan);

            let tree_t = reg.open_tree(format!("{t}")).unwrap();
            // reg[t, ∈] = y_t^{term}
            tree_t.insert("", pack(&ntt_vec_t_term.matrix, Q)).unwrap();
            // reg[t, term] = -G^{-1}(y_t^{term})
            tree_t
                .insert(b"term", pack(&ntt_decomp_vec_t_term.matrix, Q))
                .unwrap();
        }

        ((mat_a, mat_b), st, reg)
    }

    /// Generates the key-pair `(pk, sk)` with short secret key `x ← U(R_b^{m_B})`
    /// and LWE sample `y := B * x + e^*` with `e^* ← U(R_b^n)` as its public key.
    ///
    /// Returns the public encoding `-G^{-1}(y)` in NTT form, as expected by
    /// [`Self::update`]; generating a key does not register it.
    ///
    /// Parameters:
    /// - `pp`: Explicit public matrices `A` and `B`
    pub fn keygen(&self, pp: &PP) -> (PK, SK) {
        let mat_b = &pp.1;

        // x ← U(R_b^{m_B}), with coefficient support (-b/2,b/2].
        let vec_x = Mat::sample_small(
            D,
            M_B,
            1,
            &Uniform::new_inclusive(0, SMALL_RND_B).unwrap(),
            SMALL_RND_B / 2,
            Q,
            &mut rng(),
        );
        // e^* ← U(R_b^n), with the same coefficient support.
        let vec_e_star = Mat::sample_small(
            D,
            N,
            1,
            &Uniform::new_inclusive(0, SMALL_RND_B).unwrap(),
            SMALL_RND_B / 2,
            Q,
            &mut rng(),
        );

        let ntt_vec_x = MatNTT::ntt(vec_x, &self.ntt_plan);
        let mut ntt_res = mat_b.mul(&ntt_vec_x, &self.ntt_plan);
        ntt_res.normalise(&self.ntt_plan);
        let mut res = ntt_res.inv_ntt(&self.ntt_plan);
        // y := B * x + e^*
        res.add_assign(&vec_e_star, Q);

        // Output -G^{-1}(y) as pk
        let mut decomp_res =
            res.pruned_binary_decomp(self.log_b_q, GADGET_BASE, PRUNE_GADGET_BY, Q);
        decomp_res.negate(Q);

        let ntt_decomp_res = MatNTT::ntt(decomp_res, &self.ntt_plan);

        // pk is the negative gadget decomposition of y; sk = x.
        (ntt_decomp_res, ntt_vec_x)
    }

    /// Encrypts a message `vec_m_t ∈ R_{0,1}^{1 x B}` with respect to the identity `id`.
    /// Drops `root_prune_by` bits from root coefficients and [`PRUNE_CT_BY`]
    /// bits from all other coefficients.
    ///
    /// Parameters:
    /// - `pp`: Explicit public matrices `A` and `B`
    /// - `st`: The state containing the digests from each SIS-hash-tree
    /// - `id`: Nonempty identity of at most 162 digits in radix `K`
    /// - `vec_m_t`: Transposed message vector of size `R_{0,1}^{1 x B}`
    ///
    /// # Panics
    /// Rejects unsupported compression or identity depth. Invalid identity digits
    /// or malformed matrix inputs may also panic; this is not a validating API.
    pub fn enc(&self, pp: &PP, st: &ST, id: &str, vec_m_t: Mat) -> CT {
        // ℓ := |id|
        let l = id.len();

        // r_j ← χ_s^n ∀j ∈ {0,1,...,ℓ}
        let vecs_r_j_t: Vec<MatNTT> = (0..=l)
            .into_par_iter()
            .map_init(
                || {
                    // this part runs once per thread, not per iteration
                    rng()
                },
                |rng, _| {
                    let vec_r_j_t = Mat::sample_disc_gauss(D, 1, N, &self.dgs_sigma, rng);
                    MatNTT::ntt(vec_r_j_t, &self.ntt_plan)
                },
            )
            .collect();

        // e_j ← χ_e^{k * m_A} ∀j ∈ [ℓ]
        let vecs_e_j_t: Vec<Mat> = (0..l)
            .into_par_iter()
            .map_init(
                || {
                    // this part runs once per thread, not per iteration
                    rng()
                },
                |rng, _| Mat::sample_disc_gauss(D, 1, K * self.m_a, &self.dgs_sigma, rng),
            )
            .collect();

        // e_ℓ ← χ_e^{m_B}
        let vec_e_l_t = Mat::sample_disc_gauss(D, 1, M_B, &self.dgs_sigma, &mut rng());

        // e¯ ←$ χ¯^T
        // Certified approximation to D_{R,sigma_tilde} for the joint profile.
        let vec_e_tilde_t =
            Mat::sample_disc_gauss(D, 1, st.nr_columns, &self.dgs_e_tilde, &mut rng());

        self.enc_wo_randomness(
            pp,
            st,
            id,
            vec_m_t,
            &vecs_r_j_t,
            &vecs_e_j_t,
            &vec_e_l_t,
            &vec_e_tilde_t,
        )
    }

    /// Enables encryption of a message `vec_m_t ∈ R_{0,1}^{1 x B}` with respect to the identity `id`
    /// with supplied randomness and the same codec as [`Self::enc`].
    /// All NTT inputs must use the ordinary forward-transform representation;
    /// the three output paths normalize exactly once before inverse transformation.
    ///
    /// Parameters:
    /// - `pp`: Explicit public matrices `A` and `B`
    /// - `st`: The state containing the digests from each SIS-hash-tree
    /// - `id`: Nonempty identity of at most 162 digits in radix `K`
    /// - `vec_m_t`: Transposed message vector of size `R_{0,1}^{1 x B}`
    /// - `vecs_r_j_t`: `|id| + 1` LWE secrets of size `R_q^{1 x N}` in NTT form
    /// - `vecs_e_j_t`: `|id|` LWE errors of size `R_q^{1 x K m_A}`
    /// - `vec_e_l_t`: LWE-error for leaf of path to `id` of size `R_q^{1 x m_B}`
    /// - `vec_e_tilde_t`: LWE-error to hide message of size `R_q^{1 x B}`
    ///
    /// # Panics
    /// Rejects unsupported compression or identity depth. Invalid identity digits
    /// or malformed matrix inputs may also panic; this is not a validating API.
    #[allow(clippy::too_many_arguments)]
    pub fn enc_wo_randomness(
        &self,
        pp: &PP,
        st: &ST,
        id: &str,
        mut vec_m_t: Mat,
        vecs_r_j_t: &[MatNTT],
        vecs_e_j_t: &[Mat],
        vec_e_l_t: &Mat,
        vec_e_tilde_t: &Mat,
    ) -> CT {
        assert!(
            !self.prune_aggressively && self.root_prune_by <= 13,
            "compression is outside the joint profile"
        );
        assert!(
            !id.is_empty() && id.len() <= 162,
            "identity depth is outside the joint profile"
        );
        let ntt_plan = self.ntt_plan.clone();
        let mat_a = &pp.0;
        let mat_b = &pp.1;

        // Take string of `id` and map it to numbers w.r.t. `k`, i.e. id arity, which should be close to `e`
        let id: Vec<u32> = id.chars().map(|c| c.to_digit(K as u32).unwrap()).collect();

        // ℓ := |id|
        let l = id.len();

        // Assemble Y_T = [y_ϵ^t]_{i \in T}
        let mat_y_t = st;

        // r_j ← χ_s^n ∀j ∈ {0,1,...,ℓ} given by parameter vecs_r_j_t

        let mat_g = &self.mat_g;
        let mut result: Vec<Mat> = (0..l)
            .into_par_iter()
            .map(|j| {
                // e_j ← χ_e^{k * m_A} given by parameter vecs_e_j_t

                // B_j = [A^t | (i_id_j^t ⊗ G˜)^t]^t
                // c_j^t = (r_j^T , r_{j+1}^T) * B_j + e_j^t <=> r_j^T * A + r_{j+1}^T * (i_id_j^t ⊗ G˜) + e_j^t
                // r_j^T * A
                let mut ntt_vec_c_j_t = vecs_r_j_t[j].mul(&mat_a, &self.ntt_plan);

                let mut ntt_vec_r_j_times_g = MatNTT {
                    matrix: Vec::with_capacity(N * D * K * self.m_a),
                    d: D,
                    nr_rows: 1,
                    nr_columns: K * self.m_a,
                };
                // (i_id_j^t ⊗ G˜) + e_j^t
                for i in 0..K as u32 {
                    if i == id[j] {
                        ntt_vec_r_j_times_g
                            .matrix
                            .append(&mut vecs_r_j_t[j + 1].mul(mat_g, &self.ntt_plan).matrix);
                    } else {
                        ntt_vec_r_j_times_g
                            .matrix
                            .append(&mut vec![0; D * self.m_a]);
                    }
                }

                // c_j^t = r_j^T * A + r_{j+1}^T * (i_id_j^t ⊗ G˜) + e_j^t
                ntt_vec_c_j_t.add_assign(&ntt_vec_r_j_times_g);
                // Inverse NTT is unnormalized; apply its factor exactly once.
                ntt_vec_c_j_t.normalise(&self.ntt_plan);
                let mut vec_c_j_t = ntt_vec_c_j_t.inv_ntt(&self.ntt_plan);
                vec_c_j_t.add_assign(&vecs_e_j_t[j], Q);

                // Drop lower-order bits of ciphertext
                compress(
                    &mut vec_c_j_t.matrix,
                    if j == 0 {
                        self.root_prune_by
                    } else {
                        PRUNE_CT_BY
                    },
                    Q,
                );
                vec_c_j_t
            })
            .collect();

        // e_ℓ ← χ_e^{m_B} given by parameter vec_e_l_t

        // c_l^t = r_l^t * B + e_l^t
        let mut ntt_c_l_t = vecs_r_j_t[l].mul(&mat_b, &self.ntt_plan);
        ntt_c_l_t.normalise(&self.ntt_plan);
        let mut c_l_t = ntt_c_l_t.inv_ntt(&ntt_plan);
        c_l_t.add_assign(vec_e_l_t, Q);

        compress(&mut c_l_t.matrix, PRUNE_CT_BY, Q);
        result.push(c_l_t);

        // Encode each ordinary bit at 0 or floor(q/2).
        vec_m_t.mul_scalar(Q / 2);

        // d^t = r_0^t * Y_T + e¯^t + floor(q/p) * m^t
        // let vec_d_t =
        //     &vecs_r_j_t[0] * mat_y_t + vec_e_tilde_t + self.modulus.get_q().div_floor(P) * vec_m_t;
        let mut ntt_d_t = vecs_r_j_t[0].mul(mat_y_t, &self.ntt_plan);
        ntt_d_t.normalise(&self.ntt_plan);
        let mut vec_d_t = ntt_d_t.inv_ntt(&ntt_plan);
        vec_d_t.add_assign(vec_e_tilde_t, Q);
        vec_d_t.add_assign(&vec_m_t, Q);

        // Use the same message-block codec in both modes.
        compress(
            &mut vec_d_t.matrix,
            if self.prune_aggressively {
                PRUNE_LAST_CT_VECTOR_BY
            } else {
                PRUNE_CT_BY
            },
            Q,
        );
        result.push(vec_d_t);

        result
    }

    /// Decrypts the ordinary message for subregistry `t`.
    /// Rounding always returns bits; it does not authenticate or reject a ciphertext.
    ///
    /// Parameters:
    /// - `sk`: The short secret key, also denoted by `vec_x`
    /// - `t`: Denotes the entry of the ciphertext that should be decrypted, i.e.
    ///   `t` Defines the tree `t` that the identity is currently in
    /// - `wit`: denotes the witness containing `|id|`-many vectors
    /// - `ct`: `|id|` path blocks, one leaf block and one message block
    ///
    /// # Panics
    /// Rejects unsupported compression or witness depth. An invalid subregistry
    /// index or malformed ciphertext may also panic.
    pub fn dec(&self, sk: &SK, t: usize, wit: &WIT, ct: &CT) -> Mat {
        assert!(
            !self.prune_aggressively && self.root_prune_by <= 13,
            "compression is outside the joint profile"
        );
        assert!(
            !wit.is_empty() && wit.len() <= 162,
            "witness depth is outside the joint profile"
        );
        let mut mu_bar = Mat {
            matrix: ct.last().unwrap().get_entry(0, t).to_vec(),
            d: D,
            nr_rows: 1,
            nr_columns: 1,
        };
        decompress(
            &mut mu_bar.matrix,
            if self.prune_aggressively {
                PRUNE_LAST_CT_VECTOR_BY
            } else {
                PRUNE_CT_BY
            },
            Q,
        );

        // µ¯ = d_t - ∑j∈[ℓ] c_j^t * u_j - c_l^t * x
        let mut ntt_mu_bar = MatNTT::ntt(mu_bar, &self.ntt_plan);
        for j in 0..wit.len() {
            let mut ct_j = ct[j].clone();
            decompress(
                &mut ct_j.matrix,
                if j == 0 {
                    self.root_prune_by
                } else {
                    PRUNE_CT_BY
                },
                Q,
            );

            let tmp = MatNTT::ntt(ct_j, &self.ntt_plan).mul(&wit[j], &self.ntt_plan);
            ntt_mu_bar.sub_assign(&tmp, Q); // order of ct and wit might differ
        }
        let mut ct_l = ct[wit.len()].clone();
        decompress(&mut ct_l.matrix, PRUNE_CT_BY, Q);

        let tmp = MatNTT::ntt(ct_l, &self.ntt_plan).mul(sk, &self.ntt_plan);
        ntt_mu_bar.sub_assign(&tmp, Q);
        ntt_mu_bar.normalise(&self.ntt_plan);

        let mu_bar = ntt_mu_bar.inv_ntt(&self.ntt_plan);

        let mat_vec = mu_bar.matrix.iter().map(|&v| decode_bit(v)).collect();
        Mat {
            matrix: mat_vec,
            d: D,
            nr_rows: 1,
            nr_columns: 1,
        }

        // mu_bar.mul_scalar_f64(P as f64 / Q as f64, Q);
        // mu_bar.reduce(P);
        // mu_bar
    }

    /// Updates an existing public key or inserts a new one into the [`K`]-ary tree identified by `t`.
    /// It ensures that any registered identity is always capable of generating a witness to decrypt
    /// ciphertexts addressed to it.
    /// Callers supply prefix-free, nonempty [`K`]-ary identities of length at
    /// most 162 and public encodings from [`Self::keygen`]. This low-level API
    /// does not validate arbitrary registration encodings or identity prefixes.
    ///
    /// Parameters:
    /// - `reg`: Registry containing all [`K`]-ary trees with all public keys to generate the state and witnesses
    /// - `pp`: Explicit public matrices `A` and `B`
    /// - `t`: Index defining the tree to update / insert `pk` in
    /// - `id`: Identity to update in tree `t`
    /// - `pk`: New public key
    pub fn update(&self, reg: &REG, pp: &PP, t: usize, id: &str, pk: &PK) -> ST {
        let tree_t = reg.open_tree(format!("{t}")).unwrap();

        let mut batch = Batch::default();

        // if pk = ⊥ - we assume that pk = ⊥ is denoted by any dimension-defficient vector
        // In an RBE, a user can't remove its key - otherwise we couldn't keep our witness update requirement below log(N)

        // Callers ensure prefix-free identities; the source uses fixed-length hashes.

        // reg[t, id] := pk

        batch.insert(id, pack(&pk.matrix, Q));

        self.tree_update(&tree_t, pp, id, pk, &mut batch);

        tree_t.apply_batch(batch).unwrap();

        // Assemble `new_st`
        let mut new_st: Vec<MaybeUninit<u64>> = Vec::with_capacity(B * D * N);
        unsafe {
            new_st.set_len(B * D * N);
        }
        for t in 0..B {
            let vec_y_t_epsilon = reg
                .open_tree(format!("{t}"))
                .unwrap()
                .get("")
                .unwrap()
                .unwrap();
            let unpacked_vec_y_t_epsilon = unpack(&vec_y_t_epsilon, Q, D * N);
            for j in 0..D * N {
                new_st[t * N * D + j] = MaybeUninit::new(unpacked_vec_y_t_epsilon[j]);
            }
        }
        let new_st =
            unsafe { std::mem::transmute::<Vec<std::mem::MaybeUninit<u64>>, Vec<u64>>(new_st) };
        MatNTT {
            matrix: new_st,
            d: D,
            nr_rows: N,
            nr_columns: B,
        }
    }

    /// Handles the internal logic of keeping the following invariants for tree `t`.
    /// - Any leaf has either a public key as its value or the terminating symbol of that tree.
    /// - Any inner node has exactly `K` children.
    ///
    /// We optimise the data layout slightly compared to the pseudocode given in Figure 4 by
    /// storing `-G^{-1}(u_{t, str})` for any string `str` other than `ϵ` in NTT form. This removes
    /// constant re-computation of `G^{-1}(u_{t, str})` in [`Self::witness_gen`].
    ///
    /// Parameters:
    /// - `reg`: Registry containing all [`K`]-ary trees with all public keys to generate the state and witnesses
    /// - `tree_t`: Pointer to the tree `t` to update
    /// - `pp`: Explicit public matrices `A` and `B`
    /// - `id`: Identity that is updated in the tree `t`
    /// - `pk`: The public key of `id`
    /// - `batch`: A [`Batch`] to collect all database writes of `reg`
    fn tree_update(&self, tree_t: &Tree, pp: &PP, id: &str, pk: &PK, batch: &mut Batch) {
        let mat_a = &pp.0;

        let term_ivec = tree_t.get(b"term").unwrap().unwrap();
        let term_vec = unpack(&term_ivec, Q, self.m_a * D);

        let mut last_vec = pk.clone();

        for j in (0..id.len()).rev() {
            // par := parent(id)
            let par = &id[0..j];

            // if reg[t, par∥i] ∈ {⊥, reg[t, term]} ∀i ∈ [k]
            // In an RBE, a user is expected to keep its key - it can't update or remove it.
            // Otherwise, we couldn't keep the number of witness updates in log(N). So, we only deal with case 2

            // Case 2: at least one of the children is assigned

            // u := (− G˜^{−1}(y_{par∥i}))_{i∈[k]}
            // u is build up in the following loop
            let mut ntt_vec_u = MatNTT {
                matrix: Vec::with_capacity(self.m_a * K * D),
                d: D,
                nr_rows: K * self.m_a,
                nr_columns: 1,
            };

            // for i ∈ [k]
            for i in 0..K {
                if format!("{}", i) == id[j..j + 1] {
                    ntt_vec_u.matrix.append(&mut last_vec.matrix);
                } else {
                    let child = format!("{}{}", par, i);
                    // if reg[t, par∥i] = ⊥
                    let c_node = tree_t.get(&child).unwrap();
                    if c_node.is_none() {
                        // reg[t, par∥i] := reg[t, term] // update siblings
                        // Store string "term" instead of value of reg[t, term] multiple times
                        batch.insert(child.as_str(), b"term");
                        ntt_vec_u.matrix.append(&mut term_vec.clone());
                    } else if c_node.is_some_and(|v| v == b"term") {
                        // If returned value is empty string, then it was reg[t, term]
                        ntt_vec_u.matrix.append(&mut term_vec.clone());
                    } else {
                        let mut ntt_vec_y_par_i =
                            unpack(&tree_t.get(&child).unwrap().unwrap(), Q, self.m_a * D);

                        // u := (− G˜^{−1}(y_{par∥i}))_{i∈[k]}
                        ntt_vec_u.matrix.append(&mut ntt_vec_y_par_i);
                    }
                }
            }

            // reg[t, par] := y_{par} := A · u mod q
            let mut ntt_vec_y_par = mat_a.mul(&ntt_vec_u, &self.ntt_plan);

            // If par != ϵ, then store -G^{-1}(y_{par}). If par = ϵ, then we never need the -G^{-1} value, but need A*u directly for `st`
            if !par.is_empty() {
                ntt_vec_y_par.normalise(&self.ntt_plan);
                let vec_y_par = ntt_vec_y_par.inv_ntt(&self.ntt_plan);
                let mut decomp_vec_y_par =
                    vec_y_par.pruned_binary_decomp(self.log_b_q, GADGET_BASE, PRUNE_GADGET_BY, Q);
                decomp_vec_y_par.negate(Q);
                ntt_vec_y_par = MatNTT::ntt(decomp_vec_y_par, &self.ntt_plan);

                last_vec = ntt_vec_y_par.clone();
            }

            // update parent (if it already exists, otherwise insert it)
            batch.insert(par, pack(&ntt_vec_y_par.matrix, Q));
        }

        // if id = ϵ then
        // String "" is equivalent to ϵ for us
        // stop recursion when reaching root

        // return TreeUpdate^{reg}(pp, st, t, par)
    }

    /// Generates the witness for tree `t` and identity `id`.
    /// Returns `None` if the `id` is not available in tree `t`.
    ///
    /// Parameters:
    /// - `reg`: Registry containing all `K`-ary trees with all public keys to generate the state and witnesses
    /// - `t`: Index defining the tree to generate a witness in
    /// - `id`: Identity in tree `t` that a witness should be generated for
    pub fn witness_gen(&self, reg: &REG, t: usize, id: &str) -> Option<WIT> {
        let mut wit_t = vec![];
        let tree_t = reg.open_tree(format!("{t}")).unwrap();

        let term_ivec = tree_t.get(b"term").unwrap().unwrap();
        let term_vec = unpack(&term_ivec, Q, self.m_a * D);

        // for j ∈ [ℓ]
        for i in 0..id.len() {
            let string = &id[..i];
            // u^t_id[0:j] := (− G˜^{−1}(y^t_{id[0:j] | i}))_{i ∈ [k]}
            let mut ntt_vec_u_t_str = MatNTT {
                matrix: Vec::with_capacity(K * self.m_a * D),
                d: D,
                nr_rows: K * self.m_a,
                nr_columns: 1,
            };
            // m_A = (ceil(log_base q) - prune_gadget_by) * n must hold
            for j in 0..K {
                let string = format!("{string}{j}");

                // y^t_str := reg[t, str]
                // u^t_id[0:j] := (− G˜^{−1}(y^t_{id[0:j] | i }))
                let c_node = tree_t.get(&string).unwrap();
                if c_node.is_none() {
                    return None;
                } else if c_node.is_some_and(|v| v == b"term") {
                    // if value is "term", clone reg[t, term]
                    ntt_vec_u_t_str.matrix.append(&mut term_vec.clone());
                } else {
                    ntt_vec_u_t_str.matrix.append(&mut unpack(
                        &tree_t.get(&string).unwrap().unwrap(),
                        Q,
                        D * self.m_a,
                    ));
                }
            }

            // wit_t := (u^t_id[0:j])_{j ∈ [ℓ]}
            wit_t.push(ntt_vec_u_t_str);
        }

        // return wit := (t, wit_t)
        Some(wit_t)
    }
}

// Run registry tests with --test-threads=1 from a disposable working directory.
#[cfg(test)]
mod test {
    use super::{B, D, Q, decode_bit};
    use crate::{mat::Mat, ntt::MatNTT, rbe::RBE, utils::drop_db};
    use rand::{distr::Uniform, rng, seq::IndexedRandom};

    static IDS: &[&str] = &["012", "120", "000", "111", "2"];

    #[test]
    fn ordinary_decoder_matches_nearest_centers_at_boundaries() {
        assert_eq!(Q % 4, 1);
        let k = Q / 4;
        for (residue, expected) in [
            (0, 0),
            (k - 1, 0),
            (k, 0),
            (k + 1, 1),
            (Q / 2, 1),
            (3 * k - 1, 1),
            (3 * k, 1),
            (3 * k + 1, 0),
            (Q - 1, 0),
        ] {
            let d0 = residue.min(Q - residue);
            let shifted = (residue + Q - Q / 2) % Q;
            let d1 = shifted.min(Q - shifted);
            assert_eq!(expected, u64::from(d1 < d0));
            assert_eq!(decode_bit(residue), expected, "residue={residue}");
        }
    }

    #[test]
    fn a_setup_rejects_existing_registry_without_overwriting_it() {
        drop_db();
        let host = RBE::init(false);
        let (_pp, _st, reg) = host.setup();
        let tree = reg.open_tree("0").unwrap();
        let original_root = tree.get("").unwrap().unwrap();
        tree.insert("sentinel-child", b"preserve this existing record")
            .unwrap();
        reg.flush().unwrap();
        drop(tree);
        drop(reg);

        let rejected = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| host.setup()));
        let reason = rejected.expect_err("fresh setup must reject existing records");
        let reason = reason
            .downcast_ref::<String>()
            .map(String::as_str)
            .or_else(|| reason.downcast_ref::<&str>().copied())
            .unwrap_or("");
        assert!(reason.contains("setup requires a fresh registry"));
        let reg = sled::open("registry_db").unwrap();
        let tree = reg.open_tree("0").unwrap();
        assert_eq!(tree.get("").unwrap().unwrap(), original_root);
        assert_eq!(
            tree.get("sentinel-child").unwrap().unwrap().as_ref(),
            b"preserve this existing record"
        );
        drop(tree);
        drop(reg);
        drop_db();
    }

    /// Enters a public key into the registry, encrypts and decrypts w.r.t. this public key
    /// and checks whether decryption works.
    #[test]
    fn single_pk() {
        drop_db();
        let id = IDS.choose(&mut rand::rng()).unwrap();
        let t = rand::random_range(0..B);

        let drbe = RBE::init(false);
        let (pp, _st, reg) = drbe.setup();
        let (pk, sk) = drbe.keygen(&pp);
        let st = drbe.update(&reg, &pp, t, id, &pk);
        let wit_t = drbe.witness_gen(&reg, t, id).unwrap();

        let msg_dist = Uniform::new(0, 2).unwrap();
        let vec_m_t = Mat::sample_small(D, 1, B, &msg_dist, 0, Q, &mut rng());

        let ct = drbe.enc(&pp, &st, id, vec_m_t.clone());

        let dec = drbe.dec(&sk, t, &wit_t, &ct);

        assert_eq!(
            vec_m_t.matrix[vec_m_t.d * t..vec_m_t.d * (t + 1)],
            dec.matrix
        );

        drop(reg);
        drop_db();
    }

    /// Enters multiple entries into the registry and encrypts and decrypts a message w.r.t. a (pk, sk)-pair
    /// and checks whether decryption works.
    #[test]
    fn multi_pk() {
        drop_db();
        let drbe = RBE::init(false);
        let (pp, mut st, reg) = drbe.setup();
        let id = IDS.choose(&mut rand::rng()).unwrap();
        let t = rand::random_range(0..B);

        let mut sk = MatNTT {
            matrix: vec![],
            d: 0,
            nr_rows: 0,
            nr_columns: 0,
        };
        for id_i in IDS {
            let (pk_i, sk_i) = drbe.keygen(&pp);
            if id_i == id {
                sk = sk_i;
            }
            st = drbe.update(&reg, &pp, t, id_i, &pk_i);
        }

        let wit_t = drbe.witness_gen(&reg, t, id).unwrap();

        let msg_dist = Uniform::new(0, 2).unwrap();

        let vec_m_t = Mat::sample_small(D, 1, B, &msg_dist, 0, Q, &mut rng());

        let ct = drbe.enc(&pp, &st, id, vec_m_t.clone());

        let dec = drbe.dec(&sk, t, &wit_t, &ct);

        assert_eq!(
            vec_m_t.matrix[vec_m_t.d * t..vec_m_t.d * (t + 1)],
            dec.matrix
        );

        drop(reg);
        drop_db();
    }

    /// Enters a full-length public key into the registry, encrypts and decrypts w.r.t. this public key
    /// and checks whether decryption works.
    #[test]
    fn full_length_id() {
        drop_db();
        let id = "0".repeat(162);
        let t = 0;

        let drbe = RBE::init(false);
        let (pp, _st, reg) = drbe.setup();
        let (pk, sk) = drbe.keygen(&pp);
        let st = drbe.update(&reg, &pp, t, &id, &pk);
        let wit_t = drbe.witness_gen(&reg, t, &id).unwrap();

        let msg_dist = Uniform::new(0, 2).unwrap();
        let vec_m_t = Mat::sample_small(D, 1, B, &msg_dist, 0, Q, &mut rng());

        let ct = drbe.enc(&pp, &st, &id, vec_m_t.clone());

        let dec = drbe.dec(&sk, t, &wit_t, &ct);

        assert_eq!(
            vec_m_t.matrix[vec_m_t.d * t..vec_m_t.d * (t + 1)],
            dec.matrix
        );

        drop(reg);
        drop_db();
    }

    /// Checks multiple full-length encryptions on correctness.
    #[ignore = "Expensive"]
    #[test]
    fn repeated_full_length_id() {
        drop_db();
        let id = "0".repeat(162);
        let nr_iterations = 1_000;

        let drbe = RBE::init(false);
        let (pp, mut st, reg) = drbe.setup();
        let (pk, sk) = drbe.keygen(&pp);
        let mut wits = vec![];
        for t in 0..B {
            st = drbe.update(&reg, &pp, t, &id, &pk);
            wits.push(drbe.witness_gen(&reg, t, &id).unwrap());
        }

        let msg_dist = Uniform::new(0, 2).unwrap();
        let vec_m_t = Mat::sample_small(D, 1, B, &msg_dist, 0, Q, &mut rng());

        let mut successful = 0;
        for _ in 0..nr_iterations {
            let ct = drbe.enc(&pp, &st, &id, vec_m_t.clone());

            for t in 0..B {
                let dec = drbe.dec(&sk, t, &wits[t], &ct);

                if vec_m_t.matrix[vec_m_t.d * t..vec_m_t.d * (t + 1)] == dec.matrix {
                    successful += 1;
                }
            }
        }

        println!("{successful} out of {} were successful", B * nr_iterations);

        drop(reg);
        drop_db();
    }
}
