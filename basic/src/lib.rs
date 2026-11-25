//! Research implementation of the one-ciphertext Dual-Regev channel.
//!
//! The rounded normal errors come from a 256-bit cumulative distribution table
//! (see [`NoiseTable`]) whose statistical distance to the ideal distribution is
//! bounded per table. The code is variable time. The caller supplies
//! randomness, and reproducible seeded RNGs are intended only for tests and
//! benchmarks. Ciphertexts contain no mode, audience, seed, dimensions, or
//! sender coins.
//!
//! Every operation that touches the public matrix exists in two execution
//! modes, selected by [`Mode`]. `Mode::Single` runs on the calling thread and
//! `Mode::Multi` splits the matrix work over the current rayon thread pool (the
//! global pool unless the caller installs another). Both modes return
//! identical results for identical inputs and randomness, whatever the number
//! of threads. The functions without a mode argument run in `Mode::Single`.
//!
//! The arithmetic comes from established crates: every operation modulo q is a
//! `crypto-bigint` operation (see `field.rs`), primality and the multiprecision
//! floats of the noise table are GMP and MPFR through `rug`, key bits are read
//! with `bitvec`, and uniform residues come from `rand`. The matrix products are
//! exact integer sums, reduced once per output coordinate (see `kernels.rs`).

mod field;
mod kernels;
mod noise;

use bitvec::{order::Lsb0, view::BitView};
use field::Field;
pub use noise::NoiseTable;
use rand::{Rng, RngExt};
use rayon::prelude::*;
use rug::{Integer, integer::IsPrime};
use sha3::{
    Shake256,
    digest::{ExtendableOutput, Update, XofReader},
};
use std::{borrow::Cow, fmt, sync::Arc};

pub const SEED_BYTES: usize = 32;
pub const SEED_DOMAIN: &[u8] = b"one-ciphertext-anamorphic-broadcast/seed-expansion/v1\0";
pub type Directory = Vec<Vec<Vec<u32>>>;
pub type Result<T> = std::result::Result<T, Error>;

/// Error coordinates drawn from one ChaCha12 stream of a noise seed. Stream `c`
/// covers header coordinates `[c * NOISE_CHUNK, (c + 1) * NOISE_CHUNK)` and the
/// stream after the last header stream gives the scalar error.
pub const NOISE_CHUNK: usize = noise::NOISE_CHUNK;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Error(&'static str);
impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.0)
    }
}
impl std::error::Error for Error {}

/// Where the matrix work runs. Results never depend on the mode.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
pub enum Mode {
    /// Everything runs on the calling thread.
    #[default]
    Single,
    /// Matrix products, binary products and table construction are split over
    /// the current rayon pool. The global pool takes its size from
    /// `RAYON_NUM_THREADS` or the CPU count.
    Multi,
}

impl Mode {
    /// Threads the mode uses for parallel work.
    pub fn threads(self) -> usize {
        match self {
            Mode::Single => 1,
            Mode::Multi => rayon::current_num_threads(),
        }
    }
}

/// Whether q is an odd prime, by GMP's primality test (through `rug`). GMP runs
/// trial division and a Baillie-PSW test, and reports the answer as certain
/// (`IsPrime::Yes`) for every q below 2^51, so for all `u32` moduli.
fn valid_prime(q: u32) -> bool {
    q >= 3 && !q.is_multiple_of(2) && Integer::from(q).is_probably_prime(30) == IsPrime::Yes
}

fn checked_size(count: usize, bytes: usize) -> Result<usize> {
    count
        .checked_mul(bytes)
        .filter(|&n| n <= isize::MAX as usize)
        .ok_or(Error("array size exceeds supported address space"))
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Parameters {
    n: usize,
    m: usize,
    q: u32,
    noise_sd: f64,
    noise_variance: Option<u64>,
    n_min: usize,
    payload_bits: usize,
    field: Field,
}
impl Parameters {
    /// Parameters whose error width is the exact value of the double `noise_sd`.
    pub fn new(
        n: usize,
        m: usize,
        q: u32,
        noise_sd: f64,
        n_min: usize,
        payload_bits: usize,
    ) -> Result<Self> {
        if n < 2
            || m == 0
            || n_min == 0
            || n_min >= n
            || payload_bits == 0
            || payload_bits > n - n_min
            || !valid_prime(q)
            || !noise_sd.is_finite()
            || noise_sd <= 0.0
        {
            return Err(Error("invalid research parameters"));
        }
        let matrix_len = n
            .checked_mul(m)
            .ok_or(Error("matrix dimensions overflow"))?;
        checked_size(matrix_len, size_of::<u32>())?;
        checked_size(m, size_of::<u128>())?;
        checked_size(
            payload_bits
                .checked_mul(m)
                .ok_or(Error("key dimensions overflow"))?,
            1,
        )?;
        checked_size(
            m.checked_add(1)
                .ok_or(Error("ciphertext dimensions overflow"))?,
            4,
        )?;
        Ok(Self {
            n,
            m,
            q,
            noise_sd,
            noise_variance: None,
            n_min,
            payload_bits,
            field: Field::new(q),
        })
    }
    /// Parameters whose error width is exactly `sqrt(noise_variance)`. The
    /// manuscript sets `(q sigma)^2 = n (m + 1)`, an integer, and a double can only
    /// approximate its square root (for the maintained profile the nearest double
    /// is `2^-57.7` too large, about `2^-58.7` of statistical distance per error).
    /// The noise table is built from the exact variance.
    pub fn with_noise_variance(
        n: usize,
        m: usize,
        q: u32,
        noise_variance: u64,
        n_min: usize,
        payload_bits: usize,
    ) -> Result<Self> {
        if noise_variance == 0 {
            return Err(Error("invalid research parameters"));
        }
        let mut params = Self::new(n, m, q, (noise_variance as f64).sqrt(), n_min, payload_bits)?;
        params.noise_variance = Some(noise_variance);
        Ok(params)
    }
    pub fn n(&self) -> usize {
        self.n
    }
    pub fn m(&self) -> usize {
        self.m
    }
    pub fn q(&self) -> u32 {
        self.q
    }
    /// The error standard deviation, or the double nearest it when the
    /// parameters carry an exact variance.
    pub fn noise_sd(&self) -> f64 {
        self.noise_sd
    }
    /// The exact error variance, if the parameters were built with one.
    pub fn noise_variance(&self) -> Option<u64> {
        self.noise_variance
    }
    fn noise_width(&self) -> noise::Width {
        match self.noise_variance {
            Some(v) => noise::Width::Variance(v),
            None => noise::Width::Sd(self.noise_sd),
        }
    }
    pub fn n_min(&self) -> usize {
        self.n_min
    }
    pub fn payload_bits(&self) -> usize {
        self.payload_bits
    }
    pub fn delta(&self) -> u32 {
        self.q / 2
    }
    pub fn field_bytes(&self) -> usize {
        (32 - self.q.leading_zeros()).div_ceil(8) as usize
    }
    pub fn ciphertext_bytes(&self) -> usize {
        (self.m + 1) * self.field_bytes()
    }
}

/// An immutable matrix in row-major order, and its ordinary public vector.
pub struct PublicKey {
    params: Parameters,
    matrix: Vec<u32>,
    vector: Vec<u32>,
}
impl PublicKey {
    pub fn new(params: Parameters, matrix: Vec<u32>, vector: Vec<u32>) -> Result<Self> {
        if matrix.len() != params.n * params.m
            || vector.len() != params.n
            || matrix.iter().chain(&vector).any(|&x| x >= params.q)
        {
            return Err(Error("invalid public-key dimensions or field values"));
        }
        Ok(Self {
            params,
            matrix,
            vector,
        })
    }
    pub fn params(&self) -> &Parameters {
        &self.params
    }
    pub fn matrix(&self) -> &[u32] {
        &self.matrix
    }
    pub fn vector(&self) -> &[u32] {
        &self.vector
    }
}

pub struct OrdinaryKeyPair {
    pub public: PublicKey,
    pub secret: Vec<u8>,
}
#[derive(Clone, PartialEq, Eq)]
pub enum UserKey {
    Seed([u8; SEED_BYTES]),
    Expanded(Vec<Vec<u8>>),
}
pub struct AnamorphicKeySet {
    pub directory: Directory,
    pub user_keys: Vec<UserKey>,
    pub compressed: bool,
}

/// Binary products `out[k][i] = sum_j rows[i][j] key_k[j] mod q` of a row-major
/// matrix with rows of length `len` and keys with entries 0 or 1.
fn binary_products(
    field: &Field,
    rows: &[u32],
    len: usize,
    keys: &[&[u8]],
    mode: Mode,
) -> Vec<Vec<u32>> {
    let count = keys.len();
    let row_count = rows.len().checked_div(len).unwrap_or(0);
    if count == 0 || row_count == 0 {
        return vec![vec![0; row_count]; count];
    }
    let masks: Vec<Vec<u32>> = keys.iter().map(|key| kernels::key_mask(key)).collect();
    let keys: Vec<&[u32]> = masks.iter().map(Vec::as_slice).collect();
    let keys = keys.as_slice();
    let mut flat = vec![0u32; row_count * count];
    match mode {
        Mode::Single => kernels::binary_rows(field, rows, len, keys, &mut flat),
        Mode::Multi => {
            let per_task = row_count
                .div_ceil(8 * rayon::current_num_threads())
                .clamp(1, 64);
            flat.par_chunks_mut(per_task * count)
                .zip(rows.par_chunks(per_task * len))
                .for_each(|(out, rows)| kernels::binary_rows(field, rows, len, keys, out));
        }
    }
    (0..count)
        .map(|k| flat.iter().skip(k).step_by(count).copied().collect())
        .collect()
}

/// Sample A once uniformly and the ordinary key as independent bits. No retries.
pub fn keygen<R: Rng + ?Sized>(params: Parameters, rng: &mut R) -> Result<OrdinaryKeyPair> {
    keygen_with(params, rng, Mode::Single)
}

/// [`keygen`] with an execution mode. The matrix and key are drawn from `rng` in
/// the same order in both modes, only the product `b = A k` is parallel.
pub fn keygen_with<R: Rng + ?Sized>(
    params: Parameters,
    rng: &mut R,
    mode: Mode,
) -> Result<OrdinaryKeyPair> {
    let matrix: Vec<_> = (0..params.n * params.m)
        .map(|_| rng.random_range(0..params.q))
        .collect();
    let secret: Vec<_> = (0..params.m).map(|_| rng.random_range(0..2_u8)).collect();
    let vector = binary_products(&params.field, &matrix, params.m, &[&secret], mode)
        .pop()
        .expect("one key");
    // Entries come from random_range(0..q), so they are already canonical.
    Ok(OrdinaryKeyPair {
        public: PublicKey {
            params,
            matrix,
            vector,
        },
        secret,
    })
}

/// The SHAKE256 stream behind a seed: `L m` bits, least significant bit first.
fn seed_stream(seed: &[u8; SEED_BYTES], params: &Parameters) -> Vec<u8> {
    let count = params.payload_bits * params.m;
    let mut state = Shake256::default();
    state.update(SEED_DOMAIN);
    for value in [params.n, params.m, params.payload_bits] {
        state.update(&(value as u64).to_le_bytes());
    }
    state.update(seed);
    let mut stream = vec![0; count.div_ceil(8)];
    state.finalize_xof().read(&mut stream);
    stream
}

/// SHAKE256(domain || LE64(n) || LE64(m) || LE64(L) || seed), LSB-first bits.
/// Block `a` holds stream bits `[a m, (a + 1) m)`, read through `bitvec`.
pub fn expand_seed(seed: &[u8; SEED_BYTES], params: &Parameters) -> Vec<Vec<u8>> {
    let stream = seed_stream(seed, params);
    stream
        .view_bits::<Lsb0>()
        .chunks_exact(params.m)
        .take(params.payload_bits)
        .map(|block| block.iter().by_vals().map(u8::from).collect())
        .collect()
}

/// [`expand_seed`] with an execution mode: `Mode::Multi` reads the L blocks of
/// the stream on the rayon pool. Bit `i` of the stream is bit `i % 8` of byte
/// `i / 8` (least significant first, as `bitvec`'s `Lsb0`), so the blocks are
/// the same in both modes.
pub fn expand_seed_with(seed: &[u8; SEED_BYTES], params: &Parameters, mode: Mode) -> Vec<Vec<u8>> {
    if mode == Mode::Single {
        return expand_seed(seed, params);
    }
    let stream = seed_stream(seed, params);
    let m = params.m;
    (0..params.payload_bits)
        .into_par_iter()
        .map(|a| {
            (a * m..(a + 1) * m)
                .map(|i| (stream[i / 8] >> (i % 8)) & 1)
                .collect()
        })
        .collect()
}

/// The L key blocks of a user (entries 0 or 1), expanded from a seed or
/// borrowed from an expanded key after validation.
fn key_blocks<'a>(key: &'a UserKey, params: &Parameters) -> Result<Cow<'a, [Vec<u8>]>> {
    match key {
        UserKey::Seed(seed) => Ok(Cow::Owned(expand_seed(seed, params))),
        UserKey::Expanded(blocks) => {
            validate_blocks(blocks, params)?;
            Ok(Cow::Borrowed(blocks))
        }
    }
}

fn validate_blocks(blocks: &[Vec<u8>], params: &Parameters) -> Result<()> {
    if blocks.len() != params.payload_bits
        || blocks
            .iter()
            .any(|block| block.len() != params.m || block.iter().any(|&b| b > 1))
    {
        return Err(Error("user key must contain exactly L binary m-bit blocks"));
    }
    Ok(())
}

pub fn anamorphic_keygen<R: Rng + ?Sized>(
    pk: &PublicKey,
    population: usize,
    compressed: bool,
    rng: &mut R,
) -> Result<AnamorphicKeySet> {
    anamorphic_keygen_with(pk, population, compressed, rng, Mode::Single)
}

/// [`anamorphic_keygen`] with an execution mode. All keys are drawn first, in
/// the same order as before, then every directory vector `b_{i,a} = A k_{i,a}`
/// is computed in one pass over A.
pub fn anamorphic_keygen_with<R: Rng + ?Sized>(
    pk: &PublicKey,
    population: usize,
    compressed: bool,
    rng: &mut R,
    mode: Mode,
) -> Result<AnamorphicKeySet> {
    if population == 0 {
        return Err(Error("population must be positive"));
    }
    checked_size(population, size_of::<UserKey>())?;
    let p = pk.params();
    let mut user_keys = Vec::with_capacity(population);
    for _ in 0..population {
        user_keys.push(if compressed {
            let mut seed = [0; SEED_BYTES];
            rng.fill_bytes(&mut seed);
            UserKey::Seed(seed)
        } else {
            UserKey::Expanded(
                (0..p.payload_bits)
                    .map(|_| (0..p.m).map(|_| rng.random_range(0..2_u8)).collect())
                    .collect(),
            )
        });
    }
    let blocks = user_keys
        .iter()
        .map(|key| key_blocks(key, p))
        .collect::<Result<Vec<_>>>()?;
    let keys: Vec<&[u8]> = blocks
        .iter()
        .flat_map(|user| user.iter().map(Vec::as_slice))
        .collect();
    let vectors = binary_products(&p.field, &pk.matrix, p.m, &keys, mode);
    drop(keys);
    drop(blocks);
    let directory = vectors
        .chunks(p.payload_bits)
        .map(<[Vec<u32>]>::to_vec)
        .collect();
    Ok(AnamorphicKeySet {
        directory,
        user_keys,
        compressed,
    })
}

/// RREF with its invertible row transformation. Full rank gives a bijective sampler.
pub struct AffineFiber {
    n: usize,
    field: Field,
    matrix: Vec<Vec<u32>>,
    reduced: Vec<Vec<u32>>,
    transform: Vec<Vec<u32>>,
    pivots: Vec<usize>,
    free: Vec<usize>,
    /// Row `i` holds `reduced[i][free[k]]` for every free column, for `i < rank`.
    free_coefficients: Vec<u32>,
}
impl AffineFiber {
    pub fn new(rows: &[Vec<u32>], n: usize, q: u32) -> Result<Self> {
        Self::new_with(rows, n, q, Mode::Single)
    }
    /// [`AffineFiber::new`] with an execution mode for the elimination.
    pub fn new_with(rows: &[Vec<u32>], n: usize, q: u32, mode: Mode) -> Result<Self> {
        if n == 0 || !valid_prime(q) || rows.iter().any(|row| row.len() != n) {
            return Err(Error("invalid affine field or constraint dimensions"));
        }
        let d = rows.len();
        let width = n
            .checked_add(d)
            .ok_or(Error("affine dimensions overflow"))?;
        checked_size(
            d.checked_mul(width)
                .ok_or(Error("affine dimensions overflow"))?,
            4,
        )?;
        checked_size(n, size_of::<usize>())?;
        let field = Field::new(q);
        let matrix: Vec<Vec<_>> = rows
            .iter()
            .map(|row| row.iter().map(|&x| x % q).collect())
            .collect();
        let (augmented, pivots) = kernels::gauss_jordan_with(&field, &matrix, n, mode);
        let free: Vec<usize> = (0..n)
            .filter(|j| pivots.binary_search(j).is_err())
            .collect();
        let reduced: Vec<Vec<u32>> = augmented.iter().map(|row| row[..n].to_vec()).collect();
        let transform = augmented.iter().map(|row| row[n..].to_vec()).collect();
        let free_coefficients = reduced[..pivots.len()]
            .iter()
            .flat_map(|row| free.iter().map(move |&k| row[k]))
            .collect();
        Ok(Self {
            n,
            field,
            matrix,
            reduced,
            transform,
            pivots,
            free,
            free_coefficients,
        })
    }
    pub fn rank(&self) -> usize {
        self.pivots.len()
    }
    pub fn full_rank(&self) -> bool {
        self.rank() == self.matrix.len()
    }
    pub fn pivots(&self) -> &[usize] {
        &self.pivots
    }
    pub fn free(&self) -> &[usize] {
        &self.free
    }
    pub fn reduced(&self) -> &[Vec<u32>] {
        &self.reduced
    }
    pub fn transform(&self) -> &[Vec<u32>] {
        &self.transform
    }
    pub fn matrix(&self) -> &[Vec<u32>] {
        &self.matrix
    }
    /// Field elements stored by the setup: the reduced constraint rows, the
    /// row transformation and the packed free-column coefficients.
    pub fn stored_field_elements(&self) -> usize {
        self.reduced.iter().map(Vec::len).sum::<usize>()
            + self.transform.iter().map(Vec::len).sum::<usize>()
            + self.free_coefficients.len()
    }
    /// The solution with the given free coordinates (reduced) and pivot
    /// right-hand sides `rhs[i] = (T target)[i]`.
    /// `Mode::Multi` computes the pivot coordinates on the rayon pool.
    fn solve(&self, rhs: &[u32], free_values: &[u32], mode: Mode) -> Vec<u32> {
        let f = self.free.len();
        let mut out = vec![0; self.n];
        for (&j, &value) in self.free.iter().zip(free_values) {
            out[j] = value;
        }
        let pivot_value = |i: usize| {
            let sum = kernels::dot(
                &self.field,
                &self.free_coefficients[i * f..(i + 1) * f],
                free_values,
            );
            self.field.sub(rhs[i], sum)
        };
        let values: Vec<u32> = match mode {
            Mode::Multi => (0..self.pivots.len()).into_par_iter().map(pivot_value).collect(),
            Mode::Single => (0..self.pivots.len()).map(pivot_value).collect(),
        };
        for (&j, value) in self.pivots.iter().zip(values) {
            out[j] = value;
        }
        out
    }
    pub fn from_free(&self, target: &[u32], free_values: &[u32]) -> Result<Vec<u32>> {
        if !self.full_rank() {
            return Err(Error("deficient constraints require ordinary fallback"));
        }
        if target.len() != self.matrix.len() || free_values.len() != self.free.len() {
            return Err(Error("affine sampler input dimensions differ"));
        }
        let q = self.field.q();
        let target: Vec<u32> = target.iter().map(|&t| t % q).collect();
        let free_values: Vec<u32> = free_values.iter().map(|&v| v % q).collect();
        let rhs: Vec<u32> = self.transform[..self.rank()]
            .iter()
            .map(|row| kernels::dot(&self.field, row, &target))
            .collect();
        Ok(self.solve(&rhs, &free_values, Mode::Single))
    }
    pub fn sample<R: Rng + ?Sized>(&self, target: &[u32], rng: &mut R) -> Result<Vec<u32>> {
        if !self.full_rank() || target.len() != self.matrix.len() {
            return Err(Error("invalid or deficient affine sample request"));
        }
        let values: Vec<_> = self.free_values(rng);
        self.from_free(target, &values)
    }
    fn free_values<R: Rng + ?Sized>(&self, rng: &mut R) -> Vec<u32> {
        let q = self.field.q();
        self.free.iter().map(|_| rng.random_range(0..q)).collect()
    }
}

pub struct HiddenPlan<'a> {
    public_key: &'a PublicKey,
    audience: Vec<usize>,
    fiber: AffineFiber,
    /// `offsets[a][i] = Delta * sum_j T[i][j L + a] mod q`: the pivot right-hand
    /// side contributed by payload bit `a`, precomputed for full-rank plans.
    offsets: Vec<Vec<u32>>,
}
impl HiddenPlan<'_> {
    pub fn public_key(&self) -> &PublicKey {
        self.public_key
    }
    pub fn audience(&self) -> &[usize] {
        &self.audience
    }
    pub fn fiber(&self) -> &AffineFiber {
        &self.fiber
    }
    /// Field elements of the precomputed setup, including the payload offsets.
    pub fn stored_field_elements(&self) -> usize {
        self.fiber.stored_field_elements() + self.offsets.iter().map(Vec::len).sum::<usize>()
    }
    fn check_inputs(&self, bit: u8, payload: &[u8]) -> Result<()> {
        let p = self.public_key.params();
        if bit > 1 || payload.len() != p.payload_bits || payload.iter().any(|&b| b > 1) {
            return Err(Error("invalid ordinary bit or anamorphic payload"));
        }
        Ok(())
    }
    /// The broadcast step of aEnc alone: a uniform `s` with
    /// `H_S^T s = Delta * am` for every group member. Draws the free coordinates
    /// from `rng` exactly as [`AffineFiber::sample`] does.
    pub fn sample_secret<R: Rng + ?Sized>(&self, payload: &[u8], rng: &mut R) -> Result<Vec<u32>> {
        self.sample_secret_with(payload, rng, Mode::Single)
    }
    /// [`HiddenPlan::sample_secret`] with an execution mode (the same coins and s).
    pub fn sample_secret_with<R: Rng + ?Sized>(
        &self,
        payload: &[u8],
        rng: &mut R,
        mode: Mode,
    ) -> Result<Vec<u32>> {
        self.check_inputs(0, payload)?;
        if !self.fiber.full_rank() {
            return Err(Error("deficient constraints require ordinary fallback"));
        }
        let field = &self.fiber.field;
        let rank = self.fiber.rank();
        let mut rhs = vec![0u32; rank];
        for (offsets, &bit) in self.offsets.iter().zip(payload) {
            if bit == 1 {
                for (r, &o) in rhs.iter_mut().zip(offsets) {
                    *r = field.add(*r, o);
                }
            }
        }
        let values = self.fiber.free_values(rng);
        Ok(self.fiber.solve(&rhs, &values, mode))
    }
    pub fn encrypt<R: Rng + ?Sized>(
        &self,
        bit: u8,
        payload: &[u8],
        rng: &mut R,
    ) -> Result<Ciphertext> {
        self.encrypt_with(bit, payload, rng, Mode::Single)
    }
    /// [`HiddenPlan::encrypt`] with an execution mode.
    pub fn encrypt_with<R: Rng + ?Sized>(
        &self,
        bit: u8,
        payload: &[u8],
        rng: &mut R,
        mode: Mode,
    ) -> Result<Ciphertext> {
        self.check_inputs(bit, payload)?;
        if !self.fiber.full_rank() {
            return encrypt_with(self.public_key, bit, rng, mode);
        }
        let secret = self.sample_secret_with(payload, rng, mode)?;
        encrypt_with_secret(self.public_key, bit, &secret, rng, mode)
    }
}

pub fn prepare_hidden<'a>(
    pk: &'a PublicKey,
    directory: &[Vec<Vec<u32>>],
    audience: &[usize],
) -> Result<HiddenPlan<'a>> {
    prepare_hidden_with(pk, directory, audience, Mode::Single)
}

/// [`prepare_hidden`] with an execution mode: `Mode::Multi` runs the elimination
/// of H_S on the rayon pool. The plan is the same in both modes.
pub fn prepare_hidden_with<'a>(
    pk: &'a PublicKey,
    directory: &[Vec<Vec<u32>>],
    audience: &[usize],
    mode: Mode,
) -> Result<HiddenPlan<'a>> {
    let p = pk.params();
    let mut audience = audience.to_vec();
    audience.sort_unstable();
    if audience.is_empty()
        || audience.iter().any(|&i| i >= directory.len())
        || audience.windows(2).any(|w| w[0] == w[1])
    {
        return Err(Error(
            "audience must be nonempty, in range, and duplicate-free",
        ));
    }
    if audience.len() > (p.n - p.n_min) / p.payload_bits {
        return Err(Error("audience uses too many constraints for n_min"));
    }
    if directory.iter().any(|user| user.len() != p.payload_bits)
        || audience.iter().any(|&i| {
            directory[i]
                .iter()
                .any(|vector| vector.len() != p.n || vector.iter().any(|&x| x >= p.q))
        })
    {
        return Err(Error(
            "invalid selected directory vector or user block count",
        ));
    }
    let field = &p.field;
    let mut rows = Vec::with_capacity(audience.len() * p.payload_bits);
    for &i in &audience {
        for vector in &directory[i] {
            rows.push(
                vector
                    .iter()
                    .enumerate()
                    .map(|(j, &v)| field.sub(pk.matrix[j * p.m], v))
                    .collect(),
            );
        }
    }
    let fiber = AffineFiber::new_with(&rows, p.n, p.q, mode)?;
    let offsets = if fiber.full_rank() {
        (0..p.payload_bits)
            .map(|a| {
                fiber.transform[..fiber.rank()]
                    .iter()
                    .map(|row| {
                        let sum = (0..audience.len())
                            .fold(0, |acc, j| field.add(acc, row[j * p.payload_bits + a]));
                        field.mul(p.delta(), sum)
                    })
                    .collect()
            })
            .collect()
    } else {
        Vec::new()
    };
    Ok(HiddenPlan {
        public_key: pk,
        audience,
        fiber,
        offsets,
    })
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Ciphertext {
    header: Vec<u32>,
    scalar: u32,
}
impl Ciphertext {
    pub fn new(header: Vec<u32>, scalar: u32, params: &Parameters) -> Result<Self> {
        let value = Self { header, scalar };
        value.validate(params)?;
        Ok(value)
    }
    pub fn header(&self) -> &[u32] {
        &self.header
    }
    pub fn scalar(&self) -> u32 {
        self.scalar
    }
    fn validate(&self, params: &Parameters) -> Result<()> {
        if self.header.len() != params.m
            || self.scalar >= params.q
            || self.header.iter().any(|&v| v >= params.q)
        {
            return Err(Error(
                "invalid ciphertext dimensions or noncanonical field value",
            ));
        }
        Ok(())
    }
}

/// Reject exceptional finite-precision outcomes instead of saturating or resampling.
pub fn round_noise(value: f64) -> Result<i64> {
    let rounded = value.round_ties_even();
    // i64::MAX rounds upward when cast to f64, so the upper comparison is strict.
    let limit = (1_u64 << 63) as f64;
    if !rounded.is_finite() || !(-limit..limit).contains(&rounded) {
        return Err(Error("normal sample is outside the supported signed range"));
    }
    Ok(rounded as i64)
}

/// One rounded normal error with the error width of `params`, drawn from the
/// base table with `rng`.
pub fn rounded_noise<R: Rng + ?Sized>(params: &Parameters, rng: &mut R) -> Result<i64> {
    let table = noise_table(params)?;
    Ok(table.sample(rng, &mut noise::SignBits::default()))
}

/// The base table for the error width of `params` (its exact variance if it
/// has one, otherwise `params.noise_sd()`), built on first use and shared.
pub fn noise_table(params: &Parameters) -> Result<Arc<NoiseTable>> {
    noise::table(params.noise_width(), false)
}

/// [`noise_table`], built in the given mode on first use (the table itself
/// does not depend on the mode).
pub fn noise_table_with(params: &Parameters, mode: Mode) -> Result<Arc<NoiseTable>> {
    noise::table(params.noise_width(), mode == Mode::Multi)
}

/// The errors that [`encrypt_from_seed`] adds for a noise seed: the `m` header
/// errors and the scalar error.
pub fn noise_from_seed(params: &Parameters, seed: &[u8; 32]) -> Result<(Vec<i64>, i64)> {
    let table = noise_table(params)?;
    let mut errors = vec![0i64; params.m];
    for (stream, chunk) in errors.chunks_mut(NOISE_CHUNK).enumerate() {
        table.fill_stream(seed, stream as u64, chunk);
    }
    let mut scalar = [0i64];
    table.fill_stream(seed, params.m.div_ceil(NOISE_CHUNK) as u64, &mut scalar);
    Ok((errors, scalar[0]))
}

/// Error terms of a ciphertext, given explicitly or derived from a seed.
enum Noise<'a> {
    Explicit {
        errors: &'a [i64],
        scalar: i64,
    },
    Seeded {
        table: &'a NoiseTable,
        seed: &'a [u8; 32],
    },
}

/// Header columns per task, a multiple of [`NOISE_CHUNK`]. One task streams
/// whole rows best on a single thread, two tasks per thread balance the pool.
fn block_columns(m: usize, mode: Mode) -> usize {
    let target = match mode {
        Mode::Single => m,
        Mode::Multi => m.div_ceil(2 * rayon::current_num_threads()),
    };
    target.max(1).div_ceil(NOISE_CHUNK) * NOISE_CHUNK
}

/// `r = A^T s + e` and `c = b^T s + y + Delta bit` with checked inputs.
fn assemble(
    pk: &PublicKey,
    bit: u8,
    secret: &[u32],
    noise: Noise<'_>,
    mode: Mode,
) -> Result<Ciphertext> {
    let p = pk.params();
    let field = &p.field;
    let m = p.m;
    let block = block_columns(m, mode);
    let fill = |index: usize, out: &mut [u32]| {
        let c0 = index * block;
        // out = (A^T s) on the columns of this block, then the errors are added.
        kernels::transposed_block(field, &pk.matrix, m, secret, c0, out);
        match &noise {
            Noise::Explicit { errors, .. } => {
                for (o, &e) in out.iter_mut().zip(&errors[c0..]) {
                    *o = field.add(*o, field.residue(e));
                }
            }
            Noise::Seeded { table, seed } => {
                let mut buffer = [0i64; NOISE_CHUNK];
                for (t, o) in out.chunks_mut(NOISE_CHUNK).enumerate() {
                    let errors = &mut buffer[..o.len()];
                    table.fill_stream(seed, (c0 / NOISE_CHUNK + t) as u64, errors);
                    for (o, &e) in o.iter_mut().zip(errors.iter()) {
                        *o = field.add(*o, field.residue(e));
                    }
                }
            }
        }
    };
    let mut header = vec![0u32; m];
    match mode {
        Mode::Single => header
            .chunks_mut(block)
            .enumerate()
            .for_each(|(i, out)| fill(i, out)),
        Mode::Multi => header
            .par_chunks_mut(block)
            .enumerate()
            .for_each(|(i, out)| fill(i, out)),
    }
    let scalar_error = match noise {
        Noise::Explicit { scalar, .. } => scalar,
        Noise::Seeded { table, seed } => {
            let mut y = [0i64];
            table.fill_stream(seed, m.div_ceil(NOISE_CHUNK) as u64, &mut y);
            y[0]
        }
    };
    let scalar = field.add(
        field.add(
            kernels::dot(field, &pk.vector, secret),
            field.residue(scalar_error),
        ),
        field.mul(p.delta(), u32::from(bit)),
    );
    Ciphertext::new(header, scalar, p)
}

fn check_coins(pk: &PublicKey, bit: u8, secret: &[u32]) -> Result<()> {
    let p = pk.params();
    if bit > 1 || secret.len() != p.n || secret.iter().any(|&s| s >= p.q) {
        return Err(Error("invalid plaintext or encryption coins"));
    }
    Ok(())
}

pub fn native_from_coins(
    pk: &PublicKey,
    bit: u8,
    secret: &[u32],
    errors: &[i64],
    scalar_error: i64,
) -> Result<Ciphertext> {
    native_from_coins_with(pk, bit, secret, errors, scalar_error, Mode::Single)
}

/// [`native_from_coins`] with an execution mode.
pub fn native_from_coins_with(
    pk: &PublicKey,
    bit: u8,
    secret: &[u32],
    errors: &[i64],
    scalar_error: i64,
    mode: Mode,
) -> Result<Ciphertext> {
    check_coins(pk, bit, secret)?;
    if errors.len() != pk.params.m {
        return Err(Error("invalid plaintext or encryption coins"));
    }
    assemble(
        pk,
        bit,
        secret,
        Noise::Explicit {
            errors,
            scalar: scalar_error,
        },
        mode,
    )
}

/// The Dual-Regev part of encryption with explicit coins: `secret` is `s` and
/// the errors are those of [`noise_from_seed`] for `seed`.
pub fn encrypt_from_seed(
    pk: &PublicKey,
    bit: u8,
    secret: &[u32],
    seed: &[u8; 32],
    mode: Mode,
) -> Result<Ciphertext> {
    check_coins(pk, bit, secret)?;
    let table = noise_table_with(pk.params(), mode)?;
    assemble(
        pk,
        bit,
        secret,
        Noise::Seeded {
            table: &table,
            seed,
        },
        mode,
    )
}

/// Draws the 32-byte noise seed after `s`, then encrypts.
fn encrypt_with_secret<R: Rng + ?Sized>(
    pk: &PublicKey,
    bit: u8,
    secret: &[u32],
    rng: &mut R,
    mode: Mode,
) -> Result<Ciphertext> {
    let mut seed = [0u8; 32];
    rng.fill_bytes(&mut seed);
    encrypt_from_seed(pk, bit, secret, &seed, mode)
}

pub fn encrypt<R: Rng + ?Sized>(pk: &PublicKey, bit: u8, rng: &mut R) -> Result<Ciphertext> {
    encrypt_with(pk, bit, rng, Mode::Single)
}

/// [`encrypt`] with an execution mode. Draws `s` uniformly (n field elements)
/// and then a 32-byte noise seed from `rng`.
pub fn encrypt_with<R: Rng + ?Sized>(
    pk: &PublicKey,
    bit: u8,
    rng: &mut R,
    mode: Mode,
) -> Result<Ciphertext> {
    if bit > 1 {
        return Err(Error("ordinary plaintext must be one bit"));
    }
    let secret: Vec<_> = (0..pk.params.n)
        .map(|_| rng.random_range(0..pk.params.q))
        .collect();
    encrypt_with_secret(pk, bit, &secret, rng, mode)
}

pub fn round_bit(z: u32, q: u32) -> Result<u8> {
    if q < 3 || q.is_multiple_of(2) {
        return Err(Error("invalid odd modulus"));
    }
    let field = Field::new(q);
    let z = z % q;
    let zero = z.min(q - z);
    let difference = field.sub(z, q / 2);
    Ok(u8::from(difference.min(q - difference) < zero))
}
pub fn ordinary_residual(ct: &Ciphertext, secret: &[u8], q: u32) -> Result<u32> {
    if q < 3
        || q.is_multiple_of(2)
        || ct.header.is_empty()
        || secret.len() != ct.header.len()
        || secret.iter().any(|&b| b > 1)
        || ct.scalar >= q
        || ct.header.iter().any(|&v| v >= q)
    {
        return Err(Error("invalid ordinary key, ciphertext, or modulus"));
    }
    let field = Field::new(q);
    let sums = binary_products(&field, &ct.header, ct.header.len(), &[secret], Mode::Single);
    Ok(field.sub(ct.scalar, sums[0][0]))
}
pub fn ordinary_decrypt(ct: &Ciphertext, secret: &[u8], q: u32) -> Result<u8> {
    round_bit(ordinary_residual(ct, secret, q)?, q)
}
pub fn user_blocks(key: &UserKey, params: &Parameters) -> Result<Vec<Vec<u8>>> {
    let blocks = match key {
        UserKey::Seed(seed) => expand_seed(seed, params),
        UserKey::Expanded(blocks) => blocks.clone(),
    };
    validate_blocks(&blocks, params)?;
    Ok(blocks)
}
pub fn hidden_residuals(ct: &Ciphertext, key: &UserKey, params: &Parameters) -> Result<Vec<u32>> {
    hidden_residuals_with(ct, key, params, Mode::Single)
}
/// [`hidden_residuals`] with an execution mode: `Mode::Multi` expands a seed key
/// and forms the L products with the header on the rayon pool, one key block per
/// task. The residuals are the same in both modes.
pub fn hidden_residuals_with(
    ct: &Ciphertext,
    key: &UserKey,
    params: &Parameters,
    mode: Mode,
) -> Result<Vec<u32>> {
    ct.validate(params)?;
    let blocks = match key {
        UserKey::Seed(seed) => Cow::Owned(expand_seed_with(seed, params, mode)),
        UserKey::Expanded(_) => key_blocks(key, params)?,
    };
    let keys: Vec<&[u8]> = blocks.iter().map(Vec::as_slice).collect();
    let field = &params.field;
    let sums: Vec<u32> = match mode {
        Mode::Single => binary_products(field, &ct.header, params.m, &keys, Mode::Single)
            .into_iter()
            .map(|sum| sum[0])
            .collect(),
        Mode::Multi => keys
            .par_iter()
            .map(|&key| binary_products(field, &ct.header, params.m, &[key], Mode::Single)[0][0])
            .collect(),
    };
    Ok(sums
        .into_iter()
        .map(|sum| field.sub(ct.header[0], sum))
        .collect())
}
pub fn hidden_decrypt(ct: &Ciphertext, key: &UserKey, params: &Parameters) -> Result<Vec<u8>> {
    hidden_decrypt_with(ct, key, params, Mode::Single)
}
/// [`hidden_decrypt`] with an execution mode (see [`hidden_residuals_with`]).
pub fn hidden_decrypt_with(
    ct: &Ciphertext,
    key: &UserKey,
    params: &Parameters,
    mode: Mode,
) -> Result<Vec<u8>> {
    hidden_residuals_with(ct, key, params, mode)?
        .into_iter()
        .map(|z| round_bit(z, params.q))
        .collect()
}
pub fn serialize(ct: &Ciphertext, params: &Parameters) -> Result<Vec<u8>> {
    ct.validate(params)?;
    let mut bytes = Vec::with_capacity(params.ciphertext_bytes());
    for &value in ct.header.iter().chain(std::iter::once(&ct.scalar)) {
        bytes.extend_from_slice(&value.to_le_bytes()[..params.field_bytes()]);
    }
    Ok(bytes)
}
pub fn deserialize(data: &[u8], params: &Parameters) -> Result<Ciphertext> {
    if data.len() != params.ciphertext_bytes() {
        return Err(Error("invalid native ciphertext length"));
    }
    let mut fields = Vec::with_capacity(params.m + 1);
    for chunk in data.chunks_exact(params.field_bytes()) {
        let mut word = [0; 4];
        word[..chunk.len()].copy_from_slice(chunk);
        let value = u32::from_le_bytes(word);
        if value >= params.q {
            return Err(Error("noncanonical native encoding"));
        }
        fields.push(value);
    }
    let scalar = fields.pop().ok_or(Error("missing scalar"))?;
    Ciphertext::new(fields, scalar, params)
}
