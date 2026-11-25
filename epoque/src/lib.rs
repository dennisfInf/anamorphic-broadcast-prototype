//! Independent research implementation of Epoque's identity-based encryption
//! (Boyen, Haines and Mueller, EuroS&P 2021, Section 5) with the one-ciphertext
//! anamorphic broadcast of Construction 1 embedded through A0, as in Construction 3
//! of "Anamorphic Broadcast Encryption, Revisited".
//!
//! Keys come from Epoque's trapdoor KeyGen (`trapdoor::setup_trapdoor`): the master
//! shortcut key decrypts for every identity, and Extr gives identity keys. Any
//! identity in Z_q^n is supported through the full-rank-difference encoding of
//! `identity`. The anamorphic algorithms follow Construction 1: `covert_keygen` is
//! aKGen, `prepare_hidden` with `HiddenPlan::encrypt` is aEnc (public inputs only),
//! and `decrypt_covert` is aDec for one user. No trustees or ballots.
pub mod arithmetic;
pub mod identity;
pub mod sampling;
pub mod trapdoor;

pub use identity::{Identity, IdentityEncoding};

use arithmetic::*;
use rand::{Rng, TryRng, rngs::SysRng};
use rayon::prelude::*;
use rug::Integer;
use sampling::ErrorSampler;
use std::fmt;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Error(pub(crate) &'static str);
impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.0)
    }
}
impl std::error::Error for Error {}
pub type Result<T> = std::result::Result<T, Error>;

/// Buffered OS bytes, without replacing a uniform matrix by a seeded expansion.
/// Requests of at least 4 MiB (the column blocks of the sign matrix R at the full
/// profiles) take the buffered bytes first and the rest straight from the OS, in
/// one contiguous piece per worker thread (`arithmetic::threads()`).
pub struct OsBits {
    bytes: Vec<u8>,
    offset: usize,
}

/// Smallest request that `OsBits` fills on several threads.
const OS_PARALLEL_FILL: usize = 1 << 22;

/// Fill `dst` from the OS, in at most `threads()` contiguous pieces of at least
/// 1 MiB, on the rayon pool.
fn fill_from_os(dst: &mut [u8]) -> std::result::Result<(), <SysRng as TryRng>::Error> {
    let piece = dst.len().div_ceil(threads().max(1)).max(1 << 20);
    if threads() < 2 || dst.len() <= piece {
        return SysRng.try_fill_bytes(dst);
    }
    dst.par_chunks_mut(piece)
        .try_for_each(|chunk| SysRng.try_fill_bytes(chunk))
}
impl Default for OsBits {
    fn default() -> Self {
        Self {
            bytes: vec![0; 1 << 20],
            offset: 1 << 20,
        }
    }
}
impl TryRng for OsBits {
    type Error = <SysRng as TryRng>::Error;
    fn try_next_u32(&mut self) -> std::result::Result<u32, Self::Error> {
        let mut bytes = [0; 4];
        self.try_fill_bytes(&mut bytes)?;
        Ok(u32::from_le_bytes(bytes))
    }
    fn try_next_u64(&mut self) -> std::result::Result<u64, Self::Error> {
        let mut bytes = [0; 8];
        self.try_fill_bytes(&mut bytes)?;
        Ok(u64::from_le_bytes(bytes))
    }
    fn try_fill_bytes(&mut self, mut dst: &mut [u8]) -> std::result::Result<(), Self::Error> {
        if dst.len() >= OS_PARALLEL_FILL {
            let buffered = (self.bytes.len() - self.offset).min(dst.len());
            dst[..buffered].copy_from_slice(&self.bytes[self.offset..self.offset + buffered]);
            self.offset += buffered;
            return fill_from_os(&mut dst[buffered..]);
        }
        while !dst.is_empty() {
            if self.offset == self.bytes.len() {
                SysRng.try_fill_bytes(&mut self.bytes)?;
                self.offset = 0;
            }
            let count = dst.len().min(self.bytes.len() - self.offset);
            dst[..count].copy_from_slice(&self.bytes[self.offset..self.offset + count]);
            self.offset += count;
            dst = &mut dst[count..];
        }
        Ok(())
    }
}

/// n, m, the prime q, J ordinary bits, g = the largest |S| L, and the noise bound C.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Profile {
    n: usize,
    m: usize,
    q: u64,
    ordinary_bits: usize,
    maximum_constraints: usize,
    c_bound: u64,
    /// Arithmetic modulo q (`crypto-bigint`).
    field: Field,
}
impl Profile {
    pub fn new(n: usize, m: usize, q: u64, j: usize, g: usize, c: u64) -> Result<Self> {
        if n == 0
            || n >= m
            || j == 0
            || j >= m
            || g == 0
            || g >= n
            || c == 0
            || c.checked_mul(65).is_none()
            || q < 3
            || q.is_multiple_of(2)
            || q >= 1u64 << 63
            || u128::from(q) <= n as u128
            || !is_prime(q)
        {
            return Err(Error("invalid profile"));
        }
        let field = Field::new(q)?;
        for (a, b, bytes) in [(n, m, 8usize), (m, j, 4), (n, j, 8), (m, g, 4)] {
            if a.checked_mul(b)
                .and_then(|x| x.checked_mul(bytes))
                .is_none_or(|x| x > isize::MAX as usize)
            {
                return Err(Error("matrix dimensions overflow"));
            }
        }
        if m.checked_mul(2)
            .and_then(|x| x.checked_add(j))
            .and_then(|x| x.checked_mul(8))
            .is_none_or(|x| x > isize::MAX as usize)
        {
            return Err(Error("ciphertext dimensions overflow"));
        }
        Ok(Self {
            n,
            m,
            q,
            ordinary_bits: j,
            maximum_constraints: g,
            c_bound: c,
            field,
        })
    }
    pub fn n(&self) -> usize {
        self.n
    }
    pub fn m(&self) -> usize {
        self.m
    }
    pub fn q(&self) -> u64 {
        self.q
    }
    pub fn ordinary_bits(&self) -> usize {
        self.ordinary_bits
    }
    pub fn maximum_constraints(&self) -> usize {
        self.maximum_constraints
    }
    pub fn c_bound(&self) -> u64 {
        self.c_bound
    }
    pub fn denominator(&self) -> u64 {
        65 * self.c_bound
    }
    pub fn field_bytes(&self) -> usize {
        (64 - self.q.leading_zeros()).div_ceil(8) as usize
    }
    pub fn ciphertext_bytes(&self) -> usize {
        (2 * self.m + self.ordinary_bits) * self.field_bytes()
    }
}

/// Epoque's public parameters (A0, A1, B, U) and the identity encoding H.
pub struct PublicKey {
    profile: Profile,
    a0: Vec<u64>,
    a1: Vec<u64>,
    b: Vec<u64>,
    u: Vec<u64>,
    encoding: IdentityEncoding,
}
impl PublicKey {
    pub fn new(p: Profile, a0: Vec<u64>, a1: Vec<u64>, b: Vec<u64>, u: Vec<u64>) -> Result<Self> {
        if [a0.len(), a1.len(), b.len()] != [p.n * p.m; 3]
            || u.len() != p.n * p.ordinary_bits
            || a0.iter().chain(&a1).chain(&b).chain(&u).any(|&x| x >= p.q)
        {
            return Err(Error("invalid public-key dimensions or field entries"));
        }
        Ok(Self {
            profile: p,
            a0,
            a1,
            b,
            u,
            encoding: IdentityEncoding::new(p)?,
        })
    }
    pub fn profile(&self) -> Profile {
        self.profile
    }
    pub fn a0(&self) -> &[u64] {
        &self.a0
    }
    pub fn a1(&self) -> &[u64] {
        &self.a1
    }
    pub fn b(&self) -> &[u64] {
        &self.b
    }
    pub fn u(&self) -> &[u64] {
        &self.u
    }
    pub fn identity_encoding(&self) -> &IdentityEncoding {
        &self.encoding
    }
    /// The identity id in Z_q^n, with H(id) from this key's encoding.
    pub fn identity(&self, id: &[u64]) -> Result<Identity> {
        self.encoding.identity(id)
    }
    /// The unit identity (1, 0, ..., 0), whose matrix is H = I.
    pub fn unit_identity(&self) -> Identity {
        self.encoding.unit()
    }
}

/// Epoque's master shortcut key (D1, D2) of Section 5.2.1, stored transposed (J by m).
pub struct ShortcutKey {
    profile: Profile,
    d1t: Vec<i32>,
    d2t: Vec<i32>,
    /// Entries outside [-127,127] use the wide exact product.
    wide: bool,
}
impl ShortcutKey {
    /// Keys row-major m by J with arbitrary `i32` entries; storage transposes them once.
    pub fn new(p: Profile, d1: &[i32], d2: &[i32]) -> Result<Self> {
        if p.m.checked_mul(p.ordinary_bits) != Some(d1.len()) || d2.len() != d1.len() {
            return Err(Error("invalid shortcut key dimensions"));
        }
        Self::from_transposed(
            p,
            transpose_small(d1, p.m, p.ordinary_bits),
            transpose_small(d2, p.m, p.ordinary_bits),
        )
    }
    /// Keys already stored transposed (J by m), with arbitrary `i32` entries.
    pub(crate) fn from_transposed(p: Profile, d1t: Vec<i32>, d2t: Vec<i32>) -> Result<Self> {
        if d1t.len() != p.m * p.ordinary_bits || d2t.len() != d1t.len() {
            return Err(Error("invalid shortcut key dimensions"));
        }
        let wide = d1t.iter().chain(&d2t).any(|&x| !(-127..=127).contains(&x));
        Ok(Self {
            profile: p,
            d1t,
            d2t,
            wide,
        })
    }
    pub fn d1_transposed(&self) -> &[i32] {
        &self.d1t
    }
    pub fn d2_transposed(&self) -> &[i32] {
        &self.d2t
    }
}

/// The public parameters and the master shortcut key.
pub struct Setup {
    pub public: PublicKey,
    pub shortcut: ShortcutKey,
}

fn transpose_small(values: &[i32], rows: usize, cols: usize) -> Vec<i32> {
    let mut out = vec![0; values.len()];
    for i in 0..rows {
        for j in 0..cols {
            out[j * rows + i] = values[i * cols + j];
        }
    }
    out
}
/// The residue of an arbitrary integer modulo q, in `[0, q)`, by GMP (`rug`).
pub fn integer_mod(value: &Integer, q: u64) -> u64 {
    assert!(q > 0);
    let mut reduced = value.clone() % q;
    if reduced < 0 {
        reduced += q;
    }
    reduced.to_u64().expect("canonical u64 residue")
}

/// ask_i of Construction 1: one user's L binary key blocks k̂_(i,1), ..., k̂_(i,L),
/// stored as L m bits (block a in words [a w, (a + 1) w), w = `packed_words(m)`).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CovertSecretKey {
    profile: Profile,
    bits: usize,
    packed: Vec<u64>,
}
impl CovertSecretKey {
    /// Keys row-major m by L (column a is k̂_(i,a)), every entry 0 or 1.
    pub fn new(p: Profile, bits: usize, keys: &[i32]) -> Result<Self> {
        if bits == 0
            || bits > p.maximum_constraints
            || p.m.checked_mul(bits) != Some(keys.len())
            || keys.iter().any(|&x| x != 0 && x != 1)
        {
            return Err(Error("invalid anamorphic secret key"));
        }
        let words = packed_words(p.m);
        let mut packed = vec![0u64; words * bits];
        for (j, row) in keys.chunks_exact(bits).enumerate() {
            for (a, &bit) in row.iter().enumerate() {
                packed[a * words + j / 64] |= u64::from(bit == 1) << (j % 64);
            }
        }
        Ok(Self {
            profile: p,
            bits,
            packed,
        })
    }
    pub fn profile(&self) -> Profile {
        self.profile
    }
    pub fn bits(&self) -> usize {
        self.bits
    }
    /// The key blocks as L rows of length m, entries 0 or 1.
    pub fn transposed_keys(&self) -> Vec<i32> {
        let m = self.profile.m;
        self.packed
            .chunks_exact(packed_words(m))
            .flat_map(|block| (0..m).map(move |j| i32::from((block[j / 64] >> (j % 64)) & 1 == 1)))
            .collect()
    }
    /// The key blocks as packed bits (see the type's description).
    pub fn packed_keys(&self) -> &[u64] {
        &self.packed
    }
}

/// Users whose key blocks share one pass over A0 in aKGen.
const AKGEN_BATCH: usize = 512;

/// apk of Construction 1: b̂_(i,a) = A0 k̂_(i,a) for every user i and bit a.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CovertPublicKey {
    profile: Profile,
    users: usize,
    bits: usize,
    /// Row i L + a, of length n, is b̂_(i,a).
    vectors: Vec<u64>,
}
impl CovertPublicKey {
    /// The public vectors of the given users' keys, as aKGen computes them.
    pub fn from_secret_keys(pk: &PublicKey, keys: &[CovertSecretKey]) -> Result<Self> {
        let bits = keys.first().ok_or(Error("no anamorphic users"))?.bits;
        let mut apk = Self::with_capacity(pk, bits, keys.len())?;
        apk.append(pk, keys)?;
        Ok(apk)
    }
    /// An anamorphic public key without users, with room for `users` of them.
    /// `append` adds users, so aKGen can run in batches.
    pub fn with_capacity(pk: &PublicKey, bits: usize, users: usize) -> Result<Self> {
        let p = pk.profile;
        if bits == 0 || bits > p.maximum_constraints {
            return Err(Error("invalid anamorphic message length"));
        }
        let mut vectors = Vec::new();
        users
            .checked_mul(bits * p.n)
            .and_then(|len| vectors.try_reserve_exact(len).ok())
            .ok_or(Error("allocation failed"))?;
        Ok(Self {
            profile: p,
            users: 0,
            bits,
            vectors,
        })
    }
    /// Appends b̂_(i,a) = A0 k̂_(i,a) for the users of `keys`, numbered after the
    /// existing ones. Batches of users share one exact pass over A0.
    pub fn append(&mut self, pk: &PublicKey, keys: &[CovertSecretKey]) -> Result<()> {
        let p = pk.profile;
        if p != self.profile
            || keys
                .iter()
                .any(|key| key.profile != p || key.bits != self.bits)
        {
            return Err(Error("anamorphic keys differ in parameters or length"));
        }
        for batch in keys.chunks(AKGEN_BATCH) {
            let count = batch.len() * self.bits;
            let packed: Vec<u64> = batch
                .iter()
                .flat_map(|key| key.packed.iter().copied())
                .collect();
            let image = binary_right_product_in(&p.field, &pk.a0, p.n, p.m, &packed, count)?;
            self.vectors
                .try_reserve(count * p.n)
                .map_err(|_| Error("allocation failed"))?;
            for k in 0..count {
                self.vectors.extend((0..p.n).map(|r| image[r * count + k]));
            }
        }
        self.users += keys.len();
        Ok(())
    }
    pub fn profile(&self) -> Profile {
        self.profile
    }
    pub fn users(&self) -> usize {
        self.users
    }
    pub fn bits(&self) -> usize {
        self.bits
    }
    /// b̂_(user, bit), of length n.
    pub fn vector(&self, user: usize, bit: usize) -> &[u64] {
        assert!(
            user < self.users && bit < self.bits,
            "no such public vector"
        );
        let n = self.profile.n;
        let start = (user * self.bits + bit) * n;
        &self.vectors[start..start + n]
    }
}

/// One user's ask_i: L independent uniform binary key blocks (aKGen, step 1).
pub fn covert_secret_keygen<R: Rng + ?Sized>(
    p: Profile,
    bits: usize,
    rng: &mut R,
) -> Result<CovertSecretKey> {
    if bits == 0 || bits > p.maximum_constraints {
        return Err(Error("invalid anamorphic message length"));
    }
    let count = p.m * bits;
    let mut bytes = vec![0; count.div_ceil(8)];
    rng.fill_bytes(&mut bytes);
    // Bit a m + j of the stream is entry j of block a.
    let words = packed_words(p.m);
    let mut packed = vec![0u64; words * bits];
    for (a, block) in packed.chunks_exact_mut(words).enumerate() {
        for j in 0..p.m {
            let i = a * p.m + j;
            block[j >> 6] |= u64::from((bytes[i >> 3] >> (i & 7)) & 1) << (j & 63);
        }
    }
    Ok(CovertSecretKey {
        profile: p,
        bits,
        packed,
    })
}

/// aKGen(pk, N, L): the secret keys of N users and the batched public key apk.
pub fn covert_keygen<R: Rng + ?Sized>(
    pk: &PublicKey,
    users: usize,
    bits: usize,
    rng: &mut R,
) -> Result<(CovertPublicKey, Vec<CovertSecretKey>)> {
    if users == 0 {
        return Err(Error("aKGen needs at least one user"));
    }
    let keys = (0..users)
        .map(|_| covert_secret_keygen(pk.profile, bits, rng))
        .collect::<Result<Vec<_>>>()?;
    let apk = CovertPublicKey::from_secret_keys(pk, &keys)?;
    Ok((apk, keys))
}

/// aEnc for a fixed group S, prepared once (Construction 1, steps 1, 2 and 4).
pub struct HiddenPlan<'a> {
    public: &'a PublicKey,
    group: Vec<usize>,
    bits: usize,
    constraints: Vec<Vec<u64>>,
    sampler: AffineSampler,
}
impl HiddenPlan<'_> {
    /// S = {i_1 < ... < i_t}.
    pub fn group(&self) -> &[usize] {
        &self.group
    }
    pub fn bits(&self) -> usize {
        self.bits
    }
    pub fn sampler(&self) -> &AffineSampler {
        &self.sampler
    }
    /// The columns of H_S, in the order of Construction 1.
    pub fn constraints(&self) -> &[Vec<u64>] {
        &self.constraints
    }
    /// aEnc(pk, apk, S, m, am) for identity `id` (steps 3 to 7): s uniform subject to
    /// H_S^T s = floor(q/2) am repeated once per recipient, then Epoque's encryption
    /// with fresh errors. A rank-deficient H_S falls back to ordinary encryption.
    pub fn encrypt<R: Rng + ?Sized>(
        &self,
        id: &Identity,
        ordinary: &[u8],
        hidden: &[u8],
        rng: &mut R,
    ) -> Result<Ciphertext> {
        check_bits(ordinary, self.public.profile.ordinary_bits)?;
        check_bits(hidden, self.bits)?;
        check_identity(self.public, id)?;
        if !self.sampler.full_rank() {
            return encrypt_ordinary(self.public, id, ordinary, rng);
        }
        let delta = self.public.profile.q / 2;
        let rhs: Vec<u64> = (0..self.group.len())
            .flat_map(|_| hidden.iter().map(|&bit| u64::from(bit) * delta))
            .collect();
        let s = self.sampler.sample(&rhs, rng)?;
        encrypt_with_coins(
            self.public,
            id,
            ordinary,
            &s,
            &sample_errors(self.public.profile, rng)?,
        )
    }
}

/// aEnc's preparation for a fixed group S from public inputs only: H_S has column
/// (j - 1)L + a equal to a1 - b̂_(i_j, a), with a1 the first column of A0.
pub fn prepare_hidden<'a>(
    pk: &'a PublicKey,
    apk: &CovertPublicKey,
    group: &[usize],
) -> Result<HiddenPlan<'a>> {
    let p = pk.profile;
    if apk.profile != p {
        return Err(Error(
            "anamorphic public key belongs to different parameters",
        ));
    }
    if group.is_empty()
        || group.windows(2).any(|w| w[0] >= w[1])
        || group.last().is_some_and(|&i| i >= apk.users)
    {
        return Err(Error("the group must be increasing indices of users"));
    }
    if group
        .len()
        .checked_mul(apk.bits)
        .is_none_or(|d| d > p.maximum_constraints)
    {
        return Err(Error("|S| L exceeds the profile's g"));
    }
    let constraints: Vec<Vec<u64>> = group
        .iter()
        .flat_map(|&i| (0..apk.bits).map(move |a| (i, a)))
        .map(|(i, a)| {
            let b = apk.vector(i, a);
            (0..p.n)
                .map(|r| p.field.sub(pk.a0[r * p.m], b[r]))
                .collect()
        })
        .collect();
    let sampler = AffineSampler::new(constraints.clone(), p.n, p.q)?;
    Ok(HiddenPlan {
        public: pk,
        group: group.to_vec(),
        bits: apk.bits,
        constraints,
        sampler,
    })
}

pub struct ErrorCoins {
    pub y: Vec<Integer>,
    pub x: Vec<Integer>,
    pub correlated: Vec<Integer>,
}
/// The encryption errors y (m entries) and x (J entries), rounded normals of
/// standard deviation q/(65 C), and R^T y for a fresh dense m by m sign matrix R,
/// all from `rng` and in this order: one 64-bit word per normal (further words for
/// the rare normal that needs them, `ErrorSampler::sample_many`), then R in column
/// blocks of 2048, one `fill_bytes` request each. The normals are certified and
/// the sums of R^T y formed on `arithmetic::threads()` threads; the coins and the
/// result do not depend on the thread count.
pub fn sample_errors<R: Rng + ?Sized>(p: Profile, rng: &mut R) -> Result<ErrorCoins> {
    let mut sampler = ErrorSampler::new(p.q, p.denominator())?;
    let mut y = sampler.sample_many(p.m + p.ordinary_bits, rng)?;
    let x = y.split_off(p.m);
    let correlated = sign_product(&y, p.m, 2048, rng)?;
    Ok(ErrorCoins { y, x, correlated })
}
/// Epoque's ciphertext CT = (c0, c1), BHM21 (25) and (26): c0 in Z_q^J has one entry
/// per ordinary bit (BHM21's c0, c0', ... for the vectors u, u', ...), and
/// c1 = F_id^T s + [y; R^T y] in Z_q^(2m) with F_id = [A0 | A1 + H(id) B].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Ciphertext {
    c0: Vec<u64>,
    c1: Vec<u64>,
}
impl Ciphertext {
    pub fn new(p: Profile, c0: Vec<u64>, c1: Vec<u64>) -> Result<Self> {
        let ct = Self { c0, c1 };
        ct.validate(p)?;
        Ok(ct)
    }
    fn validate(&self, p: Profile) -> Result<()> {
        if self.c0.len() != p.ordinary_bits
            || self.c1.len() != 2 * p.m
            || noncanonical(&self.c0, p.q)
            || noncanonical(&self.c1, p.q)
        {
            return Err(Error(
                "invalid ciphertext dimensions or noncanonical residue",
            ));
        }
        Ok(())
    }
    /// c0 = U^T s + x + floor(q/2) mu.
    pub fn c0(&self) -> &[u64] {
        &self.c0
    }
    /// c1 = F_id^T s + [y; R^T y]. Its first m entries are A0^T s + y, the part that
    /// aDec and the trapdoor holder read.
    pub fn c1(&self) -> &[u64] {
        &self.c1
    }
}
fn check_bits(bits: &[u8], count: usize) -> Result<()> {
    if bits.len() != count || bits.iter().any(|&x| x > 1) {
        Err(Error("invalid message length or bit"))
    } else {
        Ok(())
    }
}
fn check_identity(pk: &PublicKey, id: &Identity) -> Result<()> {
    if id.profile() != pk.profile {
        Err(Error("identity belongs to different parameters"))
    } else {
        Ok(())
    }
}
/// Epoque's encryption for identity `id` with the given coins, BHM21 (25) and (26):
/// c0 = U^T s + x + floor(q/2) mu and c1 = F_id^T s + [y; R^T y].
pub fn encrypt_with_coins(
    pk: &PublicKey,
    id: &Identity,
    bits: &[u8],
    s: &[u64],
    errors: &ErrorCoins,
) -> Result<Ciphertext> {
    let p = pk.profile;
    check_bits(bits, p.ordinary_bits)?;
    check_identity(pk, id)?;
    if s.len() != p.n
        || s.iter().any(|&x| x >= p.q)
        || errors.y.len() != p.m
        || errors.correlated.len() != p.m
        || errors.x.len() != p.ordinary_bits
    {
        return Err(Error("invalid encryption coin dimensions or residues"));
    }
    // F_id^T s = [A0^T s; A1^T s + B^T (H(id)^T s)], as (H(id) B)^T s = B^T (H(id)^T s).
    let mut c1 = Vec::with_capacity(2 * p.m);
    c1.extend(transpose_matvec_in(&p.field, &pk.a0, p.n, p.m, s)?);
    let a1s = transpose_matvec_in(&p.field, &pk.a1, p.n, p.m, s)?;
    let bhs = transpose_matvec_in(&p.field, &pk.b, p.n, p.m, &id.transpose_times(s)?)?;
    let mut c0 = transpose_matvec_in(&p.field, &pk.u, p.n, p.ordinary_bits, s)?;
    let f = &p.field;
    // Plus the noise [y; R^T y].
    for (x, y) in c1.iter_mut().zip(&errors.y) {
        *x = f.add(*x, integer_mod(y, p.q));
    }
    c1.extend(
        a1s.iter()
            .zip(&bhs)
            .zip(&errors.correlated)
            .map(|((&a, &b), z)| f.add(f.add(a, b), integer_mod(z, p.q))),
    );
    for j in 0..p.ordinary_bits {
        c0[j] = f.add(
            f.add(c0[j], integer_mod(&errors.x[j], p.q)),
            u64::from(bits[j]) * (p.q / 2),
        );
    }
    Ciphertext::new(p, c0, c1)
}
/// Epoque's encryption for identity `id`: s uniform, fresh errors.
pub fn encrypt_ordinary<R: Rng + ?Sized>(
    pk: &PublicKey,
    id: &Identity,
    bits: &[u8],
    rng: &mut R,
) -> Result<Ciphertext> {
    check_bits(bits, pk.profile.ordinary_bits)?;
    check_identity(pk, id)?;
    let secret = uniform(pk.profile.n, pk.profile.q, rng)?;
    encrypt_with_coins(pk, id, bits, &secret, &sample_errors(pk.profile, rng)?)
}
/// `c0 - d^T c1` for d = (first; second), each half stored transposed (J by m).
pub(crate) fn key_residual(
    p: Profile,
    first_t: &[i32],
    second_t: &[i32],
    wide: bool,
    ct: &Ciphertext,
) -> Result<Vec<u64>> {
    ct.validate(p)?;
    let product = if wide {
        wide_right_product_in
    } else {
        small_right_product_in
    };
    let (top, bottom) = ct.c1.split_at(p.m);
    let d1 = product(&p.field, top, 1, p.m, first_t, p.ordinary_bits)?;
    let d2 = product(&p.field, bottom, 1, p.m, second_t, p.ordinary_bits)?;
    Ok((0..p.ordinary_bits)
        .map(|j| p.field.sub(p.field.sub(ct.c0[j], d1[j]), d2[j]))
        .collect())
}
pub(crate) fn decode_residual(p: Profile, residual: Vec<u64>) -> Vec<u8> {
    residual
        .into_iter()
        .map(|z| u8::from(z.abs_diff(p.q / 2) < p.q / 4))
        .collect()
}
pub fn ordinary_residual(key: &ShortcutKey, ct: &Ciphertext) -> Result<Vec<u64>> {
    key_residual(key.profile, &key.d1t, &key.d2t, key.wide, ct)
}
/// Epoque's Dec (eq. 27) with the master shortcut key, for every identity.
pub fn decrypt_ordinary(key: &ShortcutKey, ct: &Ciphertext) -> Result<Vec<u8>> {
    Ok(decode_residual(key.profile, ordinary_residual(key, ct)?))
}
/// `c1[1] - k̂_(i,a)^T c1[1..m]` (1-based) for each of one user's L key blocks.
pub fn covert_residual(key: &CovertSecretKey, ct: &Ciphertext) -> Result<Vec<u64>> {
    let p = key.profile;
    ct.validate(p)?;
    let top = &ct.c1[..p.m];
    let image = binary_right_product_in(&p.field, top, 1, p.m, &key.packed, key.bits)?;
    Ok(image.into_iter().map(|x| p.field.sub(top[0], x)).collect())
}
/// aDec(ask_i, C): each residual rounded to the nearer of 0 and floor(q/2) in
/// circular distance, ties to 0. Returns the user's L hidden bits.
pub fn decrypt_covert(key: &CovertSecretKey, ct: &Ciphertext) -> Result<Vec<u8>> {
    let p = key.profile;
    Ok(covert_residual(key, ct)?
        .into_iter()
        .map(|z| {
            let d = p.field.sub(z, p.q / 2);
            u8::from(d.min(p.q - d) < z.min(p.q - z))
        })
        .collect())
}
pub fn encode(p: Profile, ct: &Ciphertext) -> Result<Vec<u8>> {
    ct.validate(p)?;
    let mut bytes = Vec::with_capacity(p.ciphertext_bytes());
    for &x in ct.c0.iter().chain(&ct.c1) {
        bytes.extend_from_slice(&x.to_le_bytes()[..p.field_bytes()]);
    }
    Ok(bytes)
}
pub fn decode(p: Profile, bytes: &[u8]) -> Result<Ciphertext> {
    if bytes.len() != p.ciphertext_bytes() {
        return Err(Error("wrong ciphertext byte length"));
    }
    let mut fields = Vec::with_capacity(2 * p.m + p.ordinary_bits);
    for chunk in bytes.chunks_exact(p.field_bytes()) {
        let mut word = [0; 8];
        word[..chunk.len()].copy_from_slice(chunk);
        fields.push(u64::from_le_bytes(word));
    }
    let c1 = fields.split_off(p.ordinary_bits);
    Ciphertext::new(p, fields, c1)
}
