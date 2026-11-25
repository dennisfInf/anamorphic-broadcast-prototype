//! Frozen reference: `basic/src/lib.rs` as of the initial commit, before the
//! optimized kernels, the execution modes and the table-based noise sampler.
//! Only the tests use it, to check that the new code computes exactly the same
//! keys, directories, eliminations, affine solutions, ciphertexts (for explicit
//! coins) and residuals. Do not edit.
#![allow(dead_code, clippy::all)]
//! Research implementation of the one-ciphertext Dual-Regev channel.
//!
//! This variable-time code uses a numerical normal sampler, not a certified
//! exact Gaussian. The caller supplies randomness; reproducible seeded RNGs
//! are intended only for tests and benchmarks. Ciphertexts contain no mode,
//! audience, seed, dimensions, or sender coins.

use rand::{Rng, RngExt};
use rand_distr::{Distribution, Normal};
use sha3::{
    Shake256,
    digest::{ExtendableOutput, Update, XofReader},
};
use std::fmt;

pub const SEED_BYTES: usize = 32;
pub const SEED_DOMAIN: &[u8] = b"one-ciphertext-anamorphic-broadcast/seed-expansion/v1\0";
pub type Directory = Vec<Vec<Vec<u32>>>;
pub type Result<T> = std::result::Result<T, Error>;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Error(&'static str);
impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.0)
    }
}
impl std::error::Error for Error {}

fn valid_prime(q: u32) -> bool {
    if q < 3 || q.is_multiple_of(2) {
        return false;
    }
    let mut d = 3_u64;
    while d * d <= u64::from(q) {
        if u64::from(q).is_multiple_of(d) {
            return false;
        }
        d += 2;
    }
    true
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
    n_min: usize,
    payload_bits: usize,
}
impl Parameters {
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
            n_min,
            payload_bits,
        })
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
    pub fn noise_sd(&self) -> f64 {
        self.noise_sd
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

fn binary_product(row: &[u32], bits: &[u8], q: u32) -> u32 {
    (row.iter()
        .zip(bits)
        .map(|(&x, &b)| u128::from(x) * u128::from(b))
        .sum::<u128>()
        % u128::from(q)) as u32
}

/// Sample A once uniformly and the ordinary key as independent bits. No retries.
pub fn keygen<R: Rng + ?Sized>(params: Parameters, rng: &mut R) -> Result<OrdinaryKeyPair> {
    let matrix: Vec<_> = (0..params.n * params.m)
        .map(|_| rng.random_range(0..params.q))
        .collect();
    let secret: Vec<_> = (0..params.m).map(|_| rng.random_range(0..2_u8)).collect();
    let vector = matrix
        .chunks_exact(params.m)
        .map(|row| binary_product(row, &secret, params.q))
        .collect();
    Ok(OrdinaryKeyPair {
        public: PublicKey::new(params, matrix, vector)?,
        secret,
    })
}

/// SHAKE256(domain || LE64(n) || LE64(m) || LE64(L) || seed), LSB-first bits.
pub fn expand_seed(seed: &[u8; SEED_BYTES], params: &Parameters) -> Vec<Vec<u8>> {
    let count = params.payload_bits * params.m;
    let mut state = Shake256::default();
    state.update(SEED_DOMAIN);
    for value in [params.n, params.m, params.payload_bits] {
        state.update(&(value as u64).to_le_bytes());
    }
    state.update(seed);
    let mut stream = vec![0; count.div_ceil(8)];
    state.finalize_xof().read(&mut stream);
    (0..params.payload_bits)
        .map(|a| {
            (0..params.m)
                .map(|j| {
                    let bit = a * params.m + j;
                    (stream[bit / 8] >> (bit % 8)) & 1
                })
                .collect()
        })
        .collect()
}

pub fn anamorphic_keygen<R: Rng + ?Sized>(
    pk: &PublicKey,
    population: usize,
    compressed: bool,
    rng: &mut R,
) -> Result<AnamorphicKeySet> {
    if population == 0 {
        return Err(Error("population must be positive"));
    }
    checked_size(population, size_of::<UserKey>())?;
    let p = pk.params();
    let mut user_keys = Vec::with_capacity(population);
    let mut directory = Vec::with_capacity(population);
    for _ in 0..population {
        let key = if compressed {
            let mut seed = [0; SEED_BYTES];
            rng.fill_bytes(&mut seed);
            UserKey::Seed(seed)
        } else {
            UserKey::Expanded(
                (0..p.payload_bits)
                    .map(|_| (0..p.m).map(|_| rng.random_range(0..2_u8)).collect())
                    .collect(),
            )
        };
        let blocks = user_blocks(&key, p)?;
        let vectors = blocks
            .iter()
            .map(|block| {
                pk.matrix
                    .chunks_exact(p.m)
                    .map(|row| binary_product(row, block, p.q))
                    .collect()
            })
            .collect();
        user_keys.push(key);
        directory.push(vectors);
    }
    Ok(AnamorphicKeySet {
        directory,
        user_keys,
        compressed,
    })
}

fn dot_mod(a: &[u32], b: &[u32], q: u32) -> u32 {
    (a.iter()
        .zip(b)
        .map(|(&x, &y)| u128::from(x) * u128::from(y))
        .sum::<u128>()
        % u128::from(q)) as u32
}
fn subtract_mod(a: u32, b: u32, q: u32) -> u32 {
    ((u64::from(a) + u64::from(q) - u64::from(b)) % u64::from(q)) as u32
}
fn multiply_mod(a: u32, b: u32, q: u32) -> u32 {
    (u64::from(a) * u64::from(b) % u64::from(q)) as u32
}
fn inverse_mod(mut a: u32, q: u32) -> u32 {
    let mut exponent = q - 2;
    let mut result = 1;
    while exponent > 0 {
        if exponent & 1 == 1 {
            result = multiply_mod(result, a, q);
        }
        a = multiply_mod(a, a, q);
        exponent >>= 1;
    }
    result
}

/// RREF with its invertible row transformation. Full rank gives a bijective sampler.
pub struct AffineFiber {
    n: usize,
    q: u32,
    matrix: Vec<Vec<u32>>,
    reduced: Vec<Vec<u32>>,
    transform: Vec<Vec<u32>>,
    pivots: Vec<usize>,
    free: Vec<usize>,
}
impl AffineFiber {
    pub fn new(rows: &[Vec<u32>], n: usize, q: u32) -> Result<Self> {
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
        let matrix: Vec<Vec<_>> = rows
            .iter()
            .map(|row| row.iter().map(|&x| x % q).collect())
            .collect();
        let mut augmented: Vec<Vec<_>> = matrix
            .iter()
            .enumerate()
            .map(|(i, row)| {
                let mut row = row.clone();
                row.extend((0..d).map(|j| u32::from(i == j)));
                row
            })
            .collect();
        let mut pivots = Vec::new();
        for column in 0..n {
            let rank = pivots.len();
            let Some(source) = (rank..d).find(|&i| augmented[i][column] != 0) else {
                continue;
            };
            augmented.swap(rank, source);
            let inverse = inverse_mod(augmented[rank][column], q);
            for value in &mut augmented[rank] {
                *value = multiply_mod(*value, inverse, q);
            }
            let pivot = augmented[rank].clone();
            for (i, row) in augmented.iter_mut().enumerate() {
                if i == rank {
                    continue;
                }
                let coefficient = row[column];
                if coefficient != 0 {
                    for (value, &pivot_value) in row.iter_mut().zip(&pivot) {
                        *value = subtract_mod(*value, multiply_mod(coefficient, pivot_value, q), q);
                    }
                }
            }
            pivots.push(column);
            if pivots.len() == d {
                break;
            }
        }
        let free = (0..n)
            .filter(|j| pivots.binary_search(j).is_err())
            .collect();
        let reduced = augmented.iter().map(|row| row[..n].to_vec()).collect();
        let transform = augmented.iter().map(|row| row[n..].to_vec()).collect();
        Ok(Self {
            n,
            q,
            matrix,
            reduced,
            transform,
            pivots,
            free,
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
    pub fn from_free(&self, target: &[u32], free_values: &[u32]) -> Result<Vec<u32>> {
        if !self.full_rank() {
            return Err(Error("deficient constraints require ordinary fallback"));
        }
        if target.len() != self.matrix.len() || free_values.len() != self.free.len() {
            return Err(Error("affine sampler input dimensions differ"));
        }
        let mut out = vec![0; self.n];
        for (&j, &value) in self.free.iter().zip(free_values) {
            out[j] = value % self.q;
        }
        for (i, &j) in self.pivots.iter().enumerate() {
            let rhs = dot_mod(&self.transform[i], target, self.q);
            let sum = (self
                .free
                .iter()
                .map(|&k| u128::from(self.reduced[i][k]) * u128::from(out[k]))
                .sum::<u128>()
                % u128::from(self.q)) as u32;
            out[j] = subtract_mod(rhs, sum, self.q);
        }
        Ok(out)
    }
    pub fn sample<R: Rng + ?Sized>(&self, target: &[u32], rng: &mut R) -> Result<Vec<u32>> {
        if !self.full_rank() || target.len() != self.matrix.len() {
            return Err(Error("invalid or deficient affine sample request"));
        }
        let values: Vec<_> = self
            .free
            .iter()
            .map(|_| rng.random_range(0..self.q))
            .collect();
        self.from_free(target, &values)
    }
}

pub struct HiddenPlan<'a> {
    public_key: &'a PublicKey,
    audience: Vec<usize>,
    fiber: AffineFiber,
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
    pub fn encrypt<R: Rng + ?Sized>(
        &self,
        bit: u8,
        payload: &[u8],
        rng: &mut R,
    ) -> Result<Ciphertext> {
        let p = self.public_key.params();
        if bit > 1 || payload.len() != p.payload_bits || payload.iter().any(|&b| b > 1) {
            return Err(Error("invalid ordinary bit or anamorphic payload"));
        }
        if !self.fiber.full_rank() {
            return encrypt(self.public_key, bit, rng);
        }
        let target: Vec<_> = self
            .audience
            .iter()
            .flat_map(|_| payload.iter().map(|&b| p.delta() * u32::from(b)))
            .collect();
        let secret = self.fiber.sample(&target, rng)?;
        encrypt_with_secret(self.public_key, bit, &secret, rng)
    }
}

pub fn prepare_hidden<'a>(
    pk: &'a PublicKey,
    directory: &[Vec<Vec<u32>>],
    audience: &[usize],
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
    let mut rows = Vec::with_capacity(audience.len() * p.payload_bits);
    for &i in &audience {
        for vector in &directory[i] {
            rows.push(
                vector
                    .iter()
                    .enumerate()
                    .map(|(j, &v)| subtract_mod(pk.matrix[j * p.m], v, p.q))
                    .collect(),
            );
        }
    }
    Ok(HiddenPlan {
        public_key: pk,
        audience,
        fiber: AffineFiber::new(&rows, p.n, p.q)?,
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
pub fn rounded_noise<R: Rng + ?Sized>(params: &Parameters, rng: &mut R) -> Result<i64> {
    let normal =
        Normal::new(0.0, params.noise_sd).map_err(|_| Error("invalid normal distribution"))?;
    round_noise(normal.sample(rng))
}

pub fn native_from_coins(
    pk: &PublicKey,
    bit: u8,
    secret: &[u32],
    errors: &[i64],
    scalar_error: i64,
) -> Result<Ciphertext> {
    let p = pk.params();
    if bit > 1 || secret.len() != p.n || errors.len() != p.m || secret.iter().any(|&s| s >= p.q) {
        return Err(Error("invalid plaintext or encryption coins"));
    }
    let mut sums = vec![0_u128; p.m];
    for (row, &s) in pk.matrix.chunks_exact(p.m).zip(secret) {
        for (sum, &a) in sums.iter_mut().zip(row) {
            *sum += u128::from(a) * u128::from(s);
        }
    }
    let header = sums
        .into_iter()
        .zip(errors)
        .map(|(sum, &error)| {
            ((sum % u128::from(p.q) + i128::from(error).rem_euclid(i128::from(p.q)) as u128)
                % u128::from(p.q)) as u32
        })
        .collect();
    let scalar = ((u128::from(dot_mod(&pk.vector, secret, p.q))
        + i128::from(scalar_error).rem_euclid(i128::from(p.q)) as u128
        + u128::from(p.delta()) * u128::from(bit))
        % u128::from(p.q)) as u32;
    Ciphertext::new(header, scalar, p)
}

fn encrypt_with_secret<R: Rng + ?Sized>(
    pk: &PublicKey,
    bit: u8,
    secret: &[u32],
    rng: &mut R,
) -> Result<Ciphertext> {
    let normal =
        Normal::new(0.0, pk.params.noise_sd).map_err(|_| Error("invalid normal distribution"))?;
    let errors: Result<Vec<_>> = (0..pk.params.m)
        .map(|_| round_noise(normal.sample(rng)))
        .collect();
    let errors = errors?;
    let scalar_error = round_noise(normal.sample(rng))?;
    native_from_coins(pk, bit, secret, &errors, scalar_error)
}

pub fn encrypt<R: Rng + ?Sized>(pk: &PublicKey, bit: u8, rng: &mut R) -> Result<Ciphertext> {
    if bit > 1 {
        return Err(Error("ordinary plaintext must be one bit"));
    }
    let secret: Vec<_> = (0..pk.params.n)
        .map(|_| rng.random_range(0..pk.params.q))
        .collect();
    encrypt_with_secret(pk, bit, &secret, rng)
}

pub fn round_bit(z: u32, q: u32) -> Result<u8> {
    if q < 3 || q.is_multiple_of(2) {
        return Err(Error("invalid odd modulus"));
    }
    let z = z % q;
    let zero = z.min(q - z);
    let difference = subtract_mod(z, q / 2, q);
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
    Ok(subtract_mod(
        ct.scalar,
        binary_product(&ct.header, secret, q),
        q,
    ))
}
pub fn ordinary_decrypt(ct: &Ciphertext, secret: &[u8], q: u32) -> Result<u8> {
    round_bit(ordinary_residual(ct, secret, q)?, q)
}
pub fn user_blocks(key: &UserKey, params: &Parameters) -> Result<Vec<Vec<u8>>> {
    let blocks = match key {
        UserKey::Seed(seed) => expand_seed(seed, params),
        UserKey::Expanded(blocks) => blocks.clone(),
    };
    if blocks.len() != params.payload_bits
        || blocks
            .iter()
            .any(|block| block.len() != params.m || block.iter().any(|&b| b > 1))
    {
        return Err(Error("user key must contain exactly L binary m-bit blocks"));
    }
    Ok(blocks)
}
pub fn hidden_residuals(ct: &Ciphertext, key: &UserKey, params: &Parameters) -> Result<Vec<u32>> {
    ct.validate(params)?;
    Ok(user_blocks(key, params)?
        .iter()
        .map(|block| {
            subtract_mod(
                ct.header[0],
                binary_product(&ct.header, block, params.q),
                params.q,
            )
        })
        .collect())
}
pub fn hidden_decrypt(ct: &Ciphertext, key: &UserKey, params: &Parameters) -> Result<Vec<u8>> {
    hidden_residuals(ct, key, params)?
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
