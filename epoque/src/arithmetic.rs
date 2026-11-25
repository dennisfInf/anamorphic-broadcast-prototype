//! Exact arithmetic for the Epoque shortcut experiment.
//!
//! Matrices are row-major except `dt` in `small_right_product` (its rows are the
//! columns of the right factor) and packed binary keys. No operation clips errors.
//!
//! All modular arithmetic is delegated to established crates: primality to
//! GMP (through `rug`), uniform residues to `rand`, and reductions, modular
//! sums, products and inverses to `crypto-bigint` (see `Field`). Matrix
//! products are exact integer sums in `u128` or `i128`, with overflow bounds
//! checked before the loops, reduced once per output coordinate. Parallel steps
//! run on rayon's thread pool (`threads`, `with_threads`).

use crate::Error;
use crypto_bigint::{NonZero, Odd, U64, U128};
use rand::{Rng, RngExt};
use rayon::prelude::*;
use rug::{Integer, integer::IsPrime};
use std::ops::Range;
use std::sync::atomic::{AtomicBool, Ordering};

type Result<T> = std::result::Result<T, Error>;

/// Worker threads of the parallel steps: the size of the current rayon pool. The
/// global pool takes it from `RAYON_NUM_THREADS`, else every logical CPU, as in
/// basic and RBE; inside `with_threads` it is that pool's size.
pub fn threads() -> usize {
    rayon::current_num_threads()
}

/// Run `f` on a pool of `count` worker threads (at least one): every parallel step
/// it starts runs there. No result and no coin depends on the count.
pub fn with_threads<T: Send>(count: usize, f: impl FnOnce() -> T + Send) -> T {
    rayon::ThreadPoolBuilder::new()
        .num_threads(count.max(1))
        .build()
        .expect("a rayon thread pool")
        .install(f)
}

/// Jobs below this many elementary steps (`items * cost_per_item`) stay on the
/// calling thread, where waking the pool would cost more than it saves.
const PARALLEL_WORK: usize = 1 << 16;

/// Pieces per thread when a job is split, so that the pool can balance them.
const PIECES_PER_THREAD: usize = 4;

/// Elementary steps counted for one modular product and sum through `crypto-bigint`.
const FIELD_STEPS: usize = 16;

/// Whether a job of `items` items of `cost_per_item` steps each is split.
fn split(items: usize, cost_per_item: usize) -> bool {
    items >= 2 && threads() >= 2 && items.saturating_mul(cost_per_item) >= PARALLEL_WORK
}

/// Split `0..count` into contiguous ranges, run `work` on each on the rayon pool and
/// return the results in range order. Small jobs run as one range on the calling
/// thread.
pub(crate) fn map_ranges<T: Send>(
    count: usize,
    cost_per_item: usize,
    work: impl Fn(Range<usize>) -> T + Sync,
) -> Vec<T> {
    if !split(count, cost_per_item) {
        return vec![work(0..count)];
    }
    let per_range = count.div_ceil((PIECES_PER_THREAD * threads()).min(count));
    (0..count.div_ceil(per_range))
        .into_par_iter()
        .map(|k| work(k * per_range..((k + 1) * per_range).min(count)))
        .collect()
}

/// Primality of a u64 by GMP's test (trial division, then Baillie-PSW and
/// Miller-Rabin rounds with random bases). Baillie-PSW has no pseudoprimes
/// below 2^64, and GMP reports primes below about 2^51 as certain.
pub fn is_prime(q: u64) -> bool {
    q >= 2 && Integer::from(q).is_probably_prime(30) != IsPrime::No
}

pub fn validate_modulus(q: u64) -> Result<()> {
    if !(3..1 << 63).contains(&q) || q & 1 == 0 {
        return Err(Error("require an odd prime modulus below 2^63"));
    }
    if !is_prime(q) {
        return Err(Error("modulus must be prime"));
    }
    Ok(())
}

/// Arithmetic modulo an odd prime q < 2^63. Every operation is a
/// `crypto-bigint` operation on canonical residues (constant time in the
/// operands): `add_mod`, `sub_mod`, `neg_mod`, `mul_mod`, `invert_odd_mod`, and
/// remainders of double-width integers.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Field {
    q: u64,
    modulus: NonZero<U64>,
    odd: Odd<U64>,
}

impl Field {
    pub(crate) fn new(q: u64) -> Result<Self> {
        validate_modulus(q)?;
        Ok(Self {
            q,
            modulus: NonZero::new(U64::from_u64(q)).expect("validated modulus is nonzero"),
            odd: Odd::new(U64::from_u64(q)).expect("validated modulus is odd"),
        })
    }

    fn element(&self, a: u64) -> U64 {
        debug_assert!(a < self.q, "field elements are canonical");
        U64::from_u64(a)
    }

    /// `x mod q`.
    pub(crate) fn reduce(&self, x: u128) -> u64 {
        u64::from(U128::from_u128(x).rem(&self.modulus))
    }

    /// `x mod q` for a signed integer, in `[0, q)`.
    pub(crate) fn reduce_signed(&self, x: i128) -> u64 {
        let magnitude = self.reduce(x.unsigned_abs());
        if x < 0 {
            self.neg(magnitude)
        } else {
            magnitude
        }
    }

    pub(crate) fn add(&self, a: u64, b: u64) -> u64 {
        u64::from(self.element(a).add_mod(&self.element(b), &self.modulus))
    }

    pub(crate) fn sub(&self, a: u64, b: u64) -> u64 {
        u64::from(self.element(a).sub_mod(&self.element(b), &self.modulus))
    }

    pub(crate) fn neg(&self, a: u64) -> u64 {
        u64::from(self.element(a).neg_mod(&self.modulus))
    }

    pub(crate) fn mul(&self, a: u64, b: u64) -> u64 {
        u64::from(self.element(a).mul_mod(&self.element(b), &self.modulus))
    }

    /// The inverse of a nonzero element.
    pub(crate) fn inverse(&self, a: u64) -> u64 {
        u64::from(
            self.element(a)
                .invert_odd_mod(&self.odd)
                .expect("nonzero elements of a prime field are invertible"),
        )
    }
}

fn size(rows: usize, columns: usize) -> Result<usize> {
    rows.checked_mul(columns)
        .ok_or(Error("matrix size overflow"))
}

fn zeroed<T: Clone>(count: usize, value: T) -> Result<Vec<T>> {
    let mut out = Vec::new();
    out.try_reserve_exact(count)
        .map_err(|_| Error("allocation failed"))?;
    out.resize(count, value);
    Ok(out)
}

/// Independent uniform residues, from `rand`'s unbiased `random_range`.
pub fn uniform<R: Rng + ?Sized>(len: usize, q: u64, rng: &mut R) -> Result<Vec<u64>> {
    validate_modulus(q)?;
    let mut out = Vec::new();
    out.try_reserve_exact(len)
        .map_err(|_| Error("allocation failed"))?;
    out.extend((0..len).map(|_| rng.random_range(0..q)));
    Ok(out)
}

/// Whether some entry is not below q, on `threads()` threads for long slices.
pub(crate) fn noncanonical(values: &[u64], q: u64) -> bool {
    map_ranges(values.len(), 1, |range| values[range].iter().any(|&x| x >= q))
        .into_iter()
        .any(|found| found)
}

/// The largest magnitude of the entries, on `threads()` threads for long slices.
fn largest_magnitude(values: &[i32]) -> u32 {
    map_ranges(values.len(), 1, |range| {
        values[range].iter().map(|x| x.unsigned_abs()).max().unwrap_or(0)
    })
    .into_iter()
    .max()
    .unwrap_or(0)
}

/// Compute A^T s modulo q. Each coordinate is an exact `u128` sum, reduced once.
/// Column blocks run in parallel, each reading every row of A once and checking
/// its entries as it reads them, before adding any; the result and the errors do
/// not depend on the thread count.
pub fn transpose_matvec(
    a: &[u64],
    rows: usize,
    cols: usize,
    s: &[u64],
    q: u64,
) -> Result<Vec<u64>> {
    transpose_matvec_in(&Field::new(q)?, a, rows, cols, s)
}

/// [`transpose_matvec`] in a field already validated (a profile's).
pub(crate) fn transpose_matvec_in(
    field: &Field,
    a: &[u64],
    rows: usize,
    cols: usize,
    s: &[u64],
) -> Result<Vec<u64>> {
    let q = field.q;
    if rows == 0 || cols == 0 || a.len() != size(rows, cols)? || s.len() != rows {
        return Err(Error("matrix-vector dimensions do not match"));
    }
    let bound = u128::from(q - 1)
        .checked_mul(u128::from(q - 1))
        .and_then(|x| x.checked_mul(rows as u128));
    if s.iter().any(|&x| x >= q) || (bound.is_none() && noncanonical(a, q)) {
        return Err(Error("noncanonical field element"));
    }
    bound.ok_or(Error("u128 accumulator bound exceeded"))?;
    let found = AtomicBool::new(false);
    let mut acc = zeroed(cols, 0u128)?;
    for_each_row_block(&mut acc, 1, rows, |first, block| {
        for (row, &coefficient) in a.chunks_exact(cols).zip(s) {
            let row = &row[first..first + block.len()];
            if row.iter().fold(false, |bad, &x| bad | (x >= q)) {
                found.store(true, Ordering::Relaxed);
                return;
            }
            for (sum, &x) in block.iter_mut().zip(row) {
                *sum += u128::from(x) * u128::from(coefficient);
            }
        }
    });
    if found.into_inner() {
        return Err(Error("noncanonical field element"));
    }
    Ok(acc.into_iter().map(|x| field.reduce(x)).collect())
}

/// Compute A D modulo q. `dt` is D transposed: `cols` rows of length `inner`.
/// Each entry is an exact integer sum, reduced once (the kernel of
/// `wide_right_product`). Entries are computed in parallel.
pub fn small_right_product(
    a: &[u64],
    rows: usize,
    inner: usize,
    dt: &[i32],
    cols: usize,
    q: u64,
) -> Result<Vec<u64>> {
    small_right_product_in(&Field::new(q)?, a, rows, inner, dt, cols)
}

/// [`small_right_product`] in a field already validated (a profile's).
pub(crate) fn small_right_product_in(
    field: &Field,
    a: &[u64],
    rows: usize,
    inner: usize,
    dt: &[i32],
    cols: usize,
) -> Result<Vec<u64>> {
    let q = field.q;
    if rows == 0
        || inner == 0
        || cols == 0
        || a.len() != size(rows, inner)?
        || dt.len() != size(cols, inner)?
    {
        return Err(Error("matrix product dimensions do not match"));
    }
    if noncanonical(a, q) {
        return Err(Error("noncanonical field element"));
    }
    let maximum = largest_magnitude(dt);
    if maximum > 127 {
        return Err(Error("right matrix entries must lie in [-127,127]"));
    }
    let whole_bound = (inner as u128)
        .checked_mul(u128::from(q - 1))
        .and_then(|x| x.checked_mul(u128::from(maximum)));
    if whole_bound.is_none_or(|x| x >= 1 << 126) {
        return Err(Error("small-product accumulator bound exceeded"));
    }
    signed_product(field, a, rows, inner, dt, cols, maximum)
}

/// Run `work(first_row, block)` on disjoint blocks of whole rows of `out` on the
/// rayon pool. Deterministic callers produce results that do not depend on the
/// thread count or the blocks. Small jobs (`rows * cost_per_row` below 2^16) stay
/// on the calling thread.
pub(crate) fn for_each_row_block<T: Send>(
    out: &mut [T],
    row_len: usize,
    cost_per_row: usize,
    work: impl Fn(usize, &mut [T]) + Sync,
) {
    let rows = out.len().checked_div(row_len).unwrap_or(0);
    if !split(rows, cost_per_row) {
        work(0, out);
        return;
    }
    let per_block = rows.div_ceil((PIECES_PER_THREAD * threads()).min(rows));
    out.par_chunks_mut(per_block * row_len)
        .enumerate()
        .for_each(|(index, block)| work(index * per_block, block));
}

/// The entries of A D modulo q, `dt` = D transposed, in parallel over the entries
/// (so a single row, as in decryption, is split as well). Each entry is an exact
/// integer sum: runs of terms are added in `i64` only while their magnitude is
/// provably below 2^63, and the run totals in `i128`, reduced once. The callers
/// have checked the dimensions, the residues and that every total is below 2^126,
/// and pass the largest magnitude in `dt`.
fn signed_product(
    field: &Field,
    a: &[u64],
    rows: usize,
    inner: usize,
    dt: &[i32],
    cols: usize,
    maximum: u32,
) -> Result<Vec<u64>> {
    let term = u128::from(field.q - 1) * u128::from(maximum);
    // Largest run whose i64 sum cannot overflow; zero means single terms need i128.
    let run = (i64::MAX as u128)
        .checked_div(term)
        .map_or(inner, |x| x.min(inner as u128) as usize);
    let mut out = zeroed(size(rows, cols)?, 0u64)?;
    for_each_row_block(&mut out, 1, inner, |first, block| {
        for (index, value) in (first..).zip(block.iter_mut()) {
            let a_row = &a[index / cols * inner..][..inner];
            let d_row = &dt[index % cols * inner..][..inner];
            let total: i128 = if run == 0 {
                a_row
                    .iter()
                    .zip(d_row)
                    .map(|(&x, &d)| i128::from(x) * i128::from(d))
                    .sum()
            } else {
                a_row
                    .chunks(run)
                    .zip(d_row.chunks(run))
                    .map(|(xs, ds)| {
                        i128::from(
                            xs.iter()
                                .zip(ds)
                                .map(|(&x, &d)| x as i64 * i64::from(d))
                                .sum::<i64>(),
                        )
                    })
                    .sum()
            };
            *value = field.reduce_signed(total);
        }
    });
    Ok(out)
}

/// Compute A D modulo q for an integer matrix D with arbitrary `i32` entries;
/// `dt` is D transposed (`cols` rows of length `inner`). Each entry is an exact
/// integer sum: runs of terms are added in `i64` only while their magnitude is
/// provably below 2^63, and the run totals in `i128`, reduced once. Entries are
/// computed in parallel; the result does not depend on the thread count.
pub fn wide_right_product(
    a: &[u64],
    rows: usize,
    inner: usize,
    dt: &[i32],
    cols: usize,
    q: u64,
) -> Result<Vec<u64>> {
    wide_right_product_in(&Field::new(q)?, a, rows, inner, dt, cols)
}

/// [`wide_right_product`] in a field already validated (a profile's).
pub(crate) fn wide_right_product_in(
    field: &Field,
    a: &[u64],
    rows: usize,
    inner: usize,
    dt: &[i32],
    cols: usize,
) -> Result<Vec<u64>> {
    let q = field.q;
    if rows == 0
        || inner == 0
        || cols == 0
        || a.len() != size(rows, inner)?
        || dt.len() != size(cols, inner)?
    {
        return Err(Error("matrix product dimensions do not match"));
    }
    if noncanonical(a, q) {
        return Err(Error("noncanonical field element"));
    }
    let maximum = largest_magnitude(dt);
    let term = u128::from(q - 1) * u128::from(maximum);
    if (inner as u128)
        .checked_mul(term)
        .is_none_or(|x| x >= 1 << 126)
    {
        return Err(Error("wide-product accumulator bound exceeded"));
    }
    signed_product(field, a, rows, inner, dt, cols, maximum)
}

enum SignTables {
    Small(Vec<[i64; 256]>),
    Large(Vec<Vec<Integer>>),
}

impl SignTables {
    fn new(y: &[Integer]) -> Result<Self> {
        if y.is_empty() {
            return Err(Error("sign product needs a nonempty error vector"));
        }
        let small: Option<Vec<i64>> = y.iter().map(Integer::to_i64).collect();
        if let Some(small) = small {
            let absolute_sum = small.iter().try_fold(0u128, |sum, x| {
                sum.checked_add(u128::from(x.unsigned_abs()))
            });
            if absolute_sum.is_some_and(|x| x < 1 << 62) {
                let mut tables = zeroed(y.len().div_ceil(8), [0i64; 256])?;
                for (weights, table) in small.chunks(8).zip(&mut tables) {
                    table[0] = -weights.iter().sum::<i64>();
                    for bits in 1usize..256 {
                        let index = bits.trailing_zeros() as usize;
                        let weight = weights.get(index).copied().unwrap_or(0);
                        table[bits] = table[bits & (bits - 1)] + 2 * weight;
                    }
                }
                return Ok(Self::Small(tables));
            }
        }
        // Exceptional large errors retain all their bits. Tables use exactly
        // the same signs as the fast path; no rejection or truncation occurs.
        let mut tables = Vec::new();
        tables
            .try_reserve_exact(y.len().div_ceil(8))
            .map_err(|_| Error("allocation failed"))?;
        for weights in y.chunks(8) {
            let mut table = zeroed(256, Integer::new())?;
            for weight in weights {
                table[0] -= weight;
            }
            for bits in 1usize..256 {
                let index = bits.trailing_zeros() as usize;
                let mut value = table[bits & (bits - 1)].clone();
                if let Some(weight) = weights.get(index) {
                    value += weight;
                    value += weight;
                }
                table[bits] = value;
            }
            tables.push(table);
        }
        Ok(Self::Large(tables))
    }

    fn groups(&self) -> usize {
        match self {
            Self::Small(tables) => tables.len(),
            Self::Large(tables) => tables.len(),
        }
    }

    /// R^T y for one block of `columns` columns, `raw` holding `columns` sign bytes
    /// per group. The groups are split over the threads: each part sums its groups
    /// for every column, and the parts are added in order. All sums are exact (in
    /// `i64`, every partial sum is at most sum |y_i| < 2^62 in magnitude), so the
    /// result does not depend on the thread count.
    fn product(&self, raw: &[u8], columns: usize) -> Result<Vec<Integer>> {
        if columns == 0 || raw.len() != size(self.groups(), columns)? {
            return Err(Error("sign dimensions or coin length do not match"));
        }
        match self {
            Self::Small(tables) => {
                let parts = map_ranges(tables.len(), columns, |groups| {
                    let mut part = vec![0i64; columns];
                    let rows = raw[groups.start * columns..groups.end * columns].chunks_exact(columns);
                    for (table, row) in tables[groups].iter().zip(rows) {
                        for (value, &bits) in part.iter_mut().zip(row) {
                            *value += table[usize::from(bits)];
                        }
                    }
                    part
                });
                let mut out = zeroed(columns, 0i64)?;
                for part in parts {
                    for (value, x) in out.iter_mut().zip(part) {
                        *value += x;
                    }
                }
                Ok(out.into_iter().map(Integer::from).collect())
            }
            Self::Large(tables) => {
                let parts = map_ranges(tables.len(), columns, |groups| {
                    let mut part = vec![Integer::new(); columns];
                    let rows = raw[groups.start * columns..groups.end * columns].chunks_exact(columns);
                    for (table, row) in tables[groups].iter().zip(rows) {
                        for (value, &bits) in part.iter_mut().zip(row) {
                            *value += &table[usize::from(bits)];
                        }
                    }
                    part
                });
                let mut out = zeroed(columns, Integer::new())?;
                for part in parts {
                    for (value, x) in out.iter_mut().zip(part) {
                        *value += x;
                    }
                }
                Ok(out)
            }
        }
    }
}

/// `raw[group * columns + j]` contains eight signs for output coordinate j.
/// The low bit applies to `y[8*group]`; one means +1 and zero means -1.
pub fn sign_product_from_bytes(y: &[Integer], raw: &[u8], cols: usize) -> Result<Vec<Integer>> {
    SignTables::new(y)?.product(raw, cols)
}

/// Sample independent signs in column blocks, preserving the reference layout
/// within each block. This is a dense random sign matrix, not a seeded matrix:
/// every block is one `fill_bytes` request to the caller's generator, made in
/// order on the calling thread. Only the sums of a block run in parallel.
pub fn sign_product<R: Rng + ?Sized>(
    y: &[Integer],
    cols: usize,
    block_cols: usize,
    rng: &mut R,
) -> Result<Vec<Integer>> {
    if cols == 0 || block_cols == 0 {
        return Err(Error("sign dimensions must be positive"));
    }
    let tables = SignTables::new(y)?;
    let mut out = Vec::new();
    out.try_reserve_exact(cols)
        .map_err(|_| Error("allocation failed"))?;
    let mut buffer = zeroed(size(tables.groups(), block_cols.min(cols))?, 0u8)?;
    while out.len() < cols {
        let width = block_cols.min(cols - out.len());
        let raw = &mut buffer[..size(tables.groups(), width)?];
        rng.fill_bytes(raw);
        out.extend(tables.product(raw, width)?);
    }
    Ok(out)
}

/// Uniform affine-fiber sampler over Z_q. Deficient systems are reported, so
/// the caller can use the specified ordinary-encryption fallback.
#[derive(Clone, Debug)]
pub struct AffineSampler {
    field: Field,
    n: usize,
    rows: Vec<Vec<u64>>,
    transform: Vec<Vec<u64>>,
    pivots: Vec<usize>,
    free: Vec<usize>,
}

impl AffineSampler {
    pub fn new(mut rows: Vec<Vec<u64>>, n: usize, q: u64) -> Result<Self> {
        let field = Field::new(q)?;
        let g = rows.len();
        if g == 0 || g >= n || rows.iter().any(|row| row.len() != n) {
            return Err(Error(
                "require 0 < constraints < coordinates and matching row widths",
            ));
        }
        if rows.iter().flatten().any(|&x| x >= q) {
            return Err(Error("noncanonical affine coefficient"));
        }
        let mut transform = vec![vec![0u64; g]; g];
        for (i, row) in transform.iter_mut().enumerate() {
            row[i] = 1;
        }
        let mut pivots = Vec::new();
        for col in 0..n {
            let row = pivots.len();
            let Some(pick) = (row..g).find(|&i| rows[i][col] != 0) else {
                continue;
            };
            rows.swap(row, pick);
            transform.swap(row, pick);
            let inverse = field.inverse(rows[row][col]);
            for x in &mut rows[row] {
                *x = field.mul(*x, inverse);
            }
            for x in &mut transform[row] {
                *x = field.mul(*x, inverse);
            }
            let normalized = rows[row].clone();
            let transformed = transform[row].clone();
            // Clear the column in every other row; the rows are independent, so
            // they are updated in parallel when there are enough of them.
            let update = |(i, (r, t)): (usize, (&mut Vec<u64>, &mut Vec<u64>))| {
                if i == row || r[col] == 0 {
                    return;
                }
                let coefficient = r[col];
                for (x, &y) in r.iter_mut().zip(&normalized) {
                    *x = field.sub(*x, field.mul(coefficient, y));
                }
                for (x, &y) in t.iter_mut().zip(&transformed) {
                    *x = field.sub(*x, field.mul(coefficient, y));
                }
            };
            if split(g, (n + g) * FIELD_STEPS) {
                rows.par_iter_mut()
                    .zip(transform.par_iter_mut())
                    .enumerate()
                    .for_each(update);
            } else {
                rows.iter_mut()
                    .zip(transform.iter_mut())
                    .enumerate()
                    .for_each(update);
            }
            pivots.push(col);
            if pivots.len() == g {
                break;
            }
        }
        let free = (0..n)
            .filter(|col| pivots.binary_search(col).is_err())
            .collect();
        Ok(Self {
            field,
            n,
            rows,
            transform,
            pivots,
            free,
        })
    }

    pub fn full_rank(&self) -> bool {
        self.pivots.len() == self.rows.len()
    }
    pub fn pivots(&self) -> &[usize] {
        &self.pivots
    }
    pub fn free(&self) -> &[usize] {
        &self.free
    }

    pub fn from_free(&self, rhs: &[u64], free_values: &[u64]) -> Result<Vec<u64>> {
        if !self.full_rank() {
            return Err(Error("rank-deficient affine constraints"));
        }
        if rhs.len() != self.rows.len() || free_values.len() != self.free.len() {
            return Err(Error("affine vector dimensions do not match"));
        }
        let field = &self.field;
        if rhs.iter().chain(free_values).any(|&x| x >= field.q) {
            return Err(Error("noncanonical affine value"));
        }
        let mut out = vec![0; self.n];
        for (&col, &value) in self.free.iter().zip(free_values) {
            out[col] = value;
        }
        // The pivot coordinates are independent sums, in parallel when large.
        let pivot_value = |i: usize| {
            let mut value = 0;
            for (&x, &r) in self.transform[i].iter().zip(rhs) {
                value = field.add(value, field.mul(x, r));
            }
            for (&free_col, &free_value) in self.free.iter().zip(free_values) {
                value = field.sub(value, field.mul(self.rows[i][free_col], free_value));
            }
            value
        };
        let rank = self.pivots.len();
        let values: Vec<u64> = if split(rank, self.n * FIELD_STEPS) {
            (0..rank).into_par_iter().map(pivot_value).collect()
        } else {
            (0..rank).map(pivot_value).collect()
        };
        for (&col, value) in self.pivots.iter().zip(values) {
            out[col] = value;
        }
        Ok(out)
    }

    pub fn sample<R: Rng + ?Sized>(&self, rhs: &[u64], rng: &mut R) -> Result<Vec<u64>> {
        // Validate before consuming coins, including the deficient case.
        if !self.full_rank()
            || rhs.len() != self.rows.len()
            || rhs.iter().any(|&x| x >= self.field.q)
        {
            return Err(Error("invalid affine target or rank-deficient constraints"));
        }
        self.from_free(rhs, &uniform(self.free.len(), self.field.q, rng)?)
    }
}

/// Words of a packed binary vector of length `len`: bit j is bit j % 64 of word j / 64.
pub fn packed_words(len: usize) -> usize {
    len.div_ceil(64)
}

/// Keys from which `binary_right_product` uses subset-sum tables of 4 entries,
/// and from which it uses tables of 8 entries.
const TABLE_KEYS: [usize; 2] = [16, 256];

/// Compute A K modulo q for `count` binary keys of length `inner`, packed as bits:
/// key k occupies words `[k w, (k + 1) w)` of `keys`, w = `packed_words(inner)`.
/// The result is row-major, `rows` by `count`. Each entry is an exact sum of the
/// selected row entries, in `u64` when `inner (q - 1)` fits (every Epoque profile)
/// and in `u128` otherwise, reduced once. For many keys, each block of 64 row
/// entries first gets the sums of all subsets of each 4 or 8 entries (the method of
/// the four Russians), so a key needs 16 or 8 table lookups per block instead of 64
/// masked additions. A must hold canonical residues, as every `PublicKey` matrix does. It
/// is not rechecked here, because aKGen calls this product once per batch of users.
/// Rows are computed in parallel; a single row (hidden decryption) splits its
/// 64-entry blocks over the threads instead. The result does not depend on the
/// thread count.
pub fn binary_right_product(
    a: &[u64],
    rows: usize,
    inner: usize,
    keys: &[u64],
    count: usize,
    q: u64,
) -> Result<Vec<u64>> {
    binary_right_product_in(&Field::new(q)?, a, rows, inner, keys, count)
}

/// [`binary_right_product`] in a field already validated (a profile's).
pub(crate) fn binary_right_product_in(
    field: &Field,
    a: &[u64],
    rows: usize,
    inner: usize,
    keys: &[u64],
    count: usize,
) -> Result<Vec<u64>> {
    let q = field.q;
    let words = packed_words(inner);
    if rows == 0
        || inner == 0
        || count == 0
        || a.len() != size(rows, inner)?
        || keys.len() != size(count, words)?
    {
        return Err(Error("matrix product dimensions do not match"));
    }
    let tail = inner % 64;
    if tail != 0
        && keys
            .chunks_exact(words)
            .any(|key| key[words - 1] >> tail != 0)
    {
        return Err(Error("binary key has bits beyond its length"));
    }
    debug_assert!(a.iter().all(|&x| x < q), "canonical matrix");
    // A whole row of selected entries stays below 2^64.
    let narrow = (inner as u128) * u128::from(q - 1) <= u128::from(u64::MAX);
    // Word b of key k at b * count + k, so one row block meets its key words in order.
    let mut by_block = zeroed(size(words, count)?, 0u64)?;
    for (k, key) in keys.chunks_exact(words).enumerate() {
        for (b, &word) in key.iter().enumerate() {
            by_block[b * count + k] = word;
        }
    }
    // Subset-sum tables over groups of `width` entries (0: masked additions).
    let width = match count {
        _ if !narrow => 0,
        c if c >= TABLE_KEYS[1] => 8,
        c if c >= TABLE_KEYS[0] => 4,
        _ => 0,
    };
    // The selected entries of `row`, added to `acc` and `narrow_acc`; block b of
    // `row` (64 entries) meets the key words `words[b * count..(b + 1) * count]`.
    let add_row = |row: &[u64], words: &[u64], acc: &mut [u128], narrow_acc: &mut [u64]| {
        match width {
            8 => add_selected_8(row, words, narrow_acc),
            4 => add_selected_4(row, words, narrow_acc),
            _ => {
                for (b, chunk) in row.chunks(64).enumerate() {
                    let words = &words[b * count..(b + 1) * count];
                    for (total, &word) in acc.iter_mut().zip(words) {
                        if word == 0 {
                            continue;
                        }
                        *total += if narrow {
                            let mut part = 0u64;
                            for (j, &x) in chunk.iter().enumerate() {
                                part += x & 0u64.wrapping_sub((word >> j) & 1);
                            }
                            u128::from(part)
                        } else {
                            chunk
                                .iter()
                                .enumerate()
                                .map(|(j, &x)| u128::from(x & 0u64.wrapping_sub((word >> j) & 1)))
                                .sum()
                        };
                    }
                }
            }
        }
    };
    let mut out = zeroed(size(rows, count)?, 0u64)?;
    if rows == 1 {
        // One row, as in hidden decryption: its 64-entry blocks are split over the
        // threads, and the exact partial sums are added in order.
        let blocks = inner.div_ceil(64);
        let parts = map_ranges(blocks, 64 * count, |range| {
            let mut acc = vec![0u128; count];
            let mut narrow_acc = vec![0u64; count];
            let row = &a[range.start * 64..(range.end * 64).min(inner)];
            let words = &by_block[range.start * count..range.end * count];
            add_row(row, words, &mut acc, &mut narrow_acc);
            (acc, narrow_acc)
        });
        let mut totals = vec![0u128; count];
        for (acc, narrow_acc) in parts {
            for ((total, x), y) in totals.iter_mut().zip(acc).zip(narrow_acc) {
                *total += x + u128::from(y);
            }
        }
        for (value, total) in out.iter_mut().zip(totals) {
            *value = field.reduce(total);
        }
        return Ok(out);
    }
    for_each_row_block(
        &mut out,
        count,
        inner.saturating_mul(count),
        |first, block| {
            let mut acc = vec![0u128; count];
            let mut narrow_acc = vec![0u64; count];
            for (r, out_row) in block.chunks_exact_mut(count).enumerate() {
                let row = &a[(first + r) * inner..(first + r + 1) * inner];
                acc.fill(0);
                narrow_acc.fill(0);
                add_row(row, &by_block, &mut acc, &mut narrow_acc);
                for ((value, &total), &narrow_total) in
                    out_row.iter_mut().zip(&acc).zip(&narrow_acc)
                {
                    *value = field.reduce(total + u128::from(narrow_total));
                }
            }
        },
    );
    Ok(out)
}

/// `add_selected_W(row, by_block, acc)` adds to `acc[k]` the entries of `row`
/// selected by key k (word b of key k at `by_block[b * count + k]`), through tables
/// of the sums of all subsets of each W consecutive entries of a 64-entry block.
/// The caller ensures that no sum exceeds 2^64. Fixed-size tables let the compiler
/// drop every bounds check of the inner loops.
macro_rules! add_selected {
    ($name:ident, $width:literal) => {
        fn $name(row: &[u64], by_block: &[u64], acc: &mut [u64]) {
            const W: usize = $width;
            const MASK: u64 = (1 << W) - 1;
            let count = acc.len();
            let mut sums = [[0u64; 1 << W]; 64 / W];
            let mut padded = [0u64; 64];
            for (b, chunk) in row.chunks(64).enumerate() {
                // The last block is zero-padded: keys have no bits beyond their length.
                let x: &[u64] = if chunk.len() == 64 {
                    chunk
                } else {
                    padded.fill(0);
                    padded[..chunk.len()].copy_from_slice(chunk);
                    &padded
                };
                // Table g, entry v: the sum of the entries W g + j with bit j of v set.
                for (g, table) in sums.iter_mut().enumerate() {
                    for v in 1..1 << W {
                        table[v] = table[v & (v - 1)] + x[W * g + v.trailing_zeros() as usize];
                    }
                }
                for (total, &word) in acc.iter_mut().zip(&by_block[b * count..(b + 1) * count]) {
                    let mut part = 0u64;
                    for (g, table) in sums.iter().enumerate() {
                        part += table[((word >> (W * g)) & MASK) as usize];
                    }
                    *total += part;
                }
            }
        }
    };
}
add_selected!(add_selected_4, 4);
add_selected!(add_selected_8, 8);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn binary_products_match_small_products() {
        use rand::{SeedableRng, rngs::StdRng};
        let mut rng = StdRng::seed_from_u64(13);
        // Twenty keys use subset-sum tables (the second with a padded last block). The
        // two largest moduli take the u128 path, as a whole row could exceed 2^64.
        for (q, rows, inner, count) in [
            (17u64, 3, 1, 1),
            (411_508_781_897, 5, 130, 3),
            (441_207_115_057, 64, 1100, 9),
            (441_207_115_057, 7, 1000, 20),
            (178_282_599_284_737, 3, 300, 20),
            (2_305_843_009_213_693_951, 4, 200, 5),
        ] {
            let a = uniform(rows * inner, q, &mut rng).unwrap();
            let bits: Vec<i32> = (0..count * inner).map(|_| rng.random_range(0..2)).collect();
            let words = packed_words(inner);
            let mut packed = vec![0u64; count * words];
            for (i, &bit) in bits.iter().enumerate() {
                let (k, j) = (i / inner, i % inner);
                packed[k * words + j / 64] |= (bit as u64) << (j % 64);
            }
            assert_eq!(
                binary_right_product(&a, rows, inner, &packed, count, q).unwrap(),
                small_right_product(&a, rows, inner, &bits, count, q).unwrap()
            );
            if inner % 64 != 0 {
                packed[words - 1] |= 1 << 63;
                assert!(binary_right_product(&a, rows, inner, &packed, count, q).is_err());
            }
        }
        // A large job runs on several threads with the same result.
        let (q, rows, inner, count) = (441_207_115_057u64, 300, 2000, 16);
        let a = uniform(rows * inner, q, &mut rng).unwrap();
        let bits: Vec<i32> = (0..count * inner).map(|_| rng.random_range(0..2)).collect();
        let words = packed_words(inner);
        let mut packed = vec![0u64; count * words];
        for (i, &bit) in bits.iter().enumerate() {
            let (k, j) = (i / inner, i % inner);
            packed[k * words + j / 64] |= (bit as u64) << (j % 64);
        }
        assert_eq!(
            binary_right_product(&a, rows, inner, &packed, count, q).unwrap(),
            small_right_product(&a, rows, inner, &bits, count, q).unwrap()
        );
        assert!(binary_right_product(&a, rows, inner, &packed[1..], count, q).is_err());
    }

    #[test]
    fn word_moduli_and_rejections() {
        for q in [
            3,
            17,
            65_537,
            178_282_599_284_737,
            2_305_843_009_213_693_951,
        ] {
            validate_modulus(q).unwrap();
        }
        for q in [0, 1, 2, 9, 15, 341, 3_215_031_751, (1 << 63) + 29] {
            assert!(validate_modulus(q).is_err());
        }
        // Strong pseudoprimes to several bases, a Carmichael number and squares.
        for q in [
            561u64,
            25_326_001,
            3_825_123_056_546_413_051,
            4_294_967_297,
            2_305_843_009_213_693_951 * 3,
            65_537 * 65_537,
        ] {
            assert!(!is_prime(q), "{q}");
        }
        assert!(transpose_matvec(&[17], 1, 1, &[1], 17).is_err());
        assert!(small_right_product(&[1], 1, 1, &[128], 1, 17).is_err());
        assert!(transpose_matvec(&[0; 65], 65, 1, &[0; 65], 2_305_843_009_213_693_951).is_err());
    }

    #[test]
    fn field_operations_match_wide_arithmetic() {
        use rand::{SeedableRng, rngs::StdRng};
        let mut rng = StdRng::seed_from_u64(5);
        for q in [3u64, 65_537, 178_282_599_284_737, 2_305_843_009_213_693_951] {
            let field = Field::new(q).unwrap();
            let wide = |x: i128| x.rem_euclid(i128::from(q)) as u64;
            for _ in 0..5000 {
                let (a, b) = (rng.random_range(0..q), rng.random_range(0..q));
                let (wa, wb) = (i128::from(a), i128::from(b));
                assert_eq!(field.add(a, b), wide(wa + wb));
                assert_eq!(field.sub(a, b), wide(wa - wb));
                assert_eq!(field.neg(a), wide(-wa));
                assert_eq!(
                    field.mul(a, b),
                    (u128::from(a) * u128::from(b) % u128::from(q)) as u64
                );
                if a != 0 {
                    assert_eq!(field.mul(a, field.inverse(a)), 1);
                }
                let x: u128 = rng.random();
                assert_eq!(u128::from(field.reduce(x)), x % u128::from(q));
                let s: i128 = rng.random();
                assert_eq!(field.reduce_signed(s), wide(s));
            }
            for x in [i128::MIN, i128::MIN + 1, -1, 0, 1, i128::MAX] {
                assert_eq!(
                    i128::from(field.reduce_signed(x)),
                    x.rem_euclid(i128::from(q))
                );
            }
        }
    }

    #[test]
    fn exact_matrix_products_for_word_moduli() {
        for q in [17, 178_282_599_284_737, 2_305_843_009_213_693_951] {
            let a = [q - 1, q / 2, 3, 4, q - 7, 9];
            let dt = [-127, 126, -2, 3, 1, 0];
            let actual = small_right_product(&a, 2, 3, &dt, 2, q).unwrap();
            for i in 0..2 {
                for j in 0..2 {
                    let expected: i128 = (0..3)
                        .map(|k| i128::from(a[i * 3 + k]) * i128::from(dt[j * 3 + k]))
                        .sum();
                    assert_eq!(actual[i * 2 + j], expected.rem_euclid(i128::from(q)) as u64);
                }
            }
            let s = [q - 1, q - 2];
            let actual = transpose_matvec(&a, 2, 3, &s, q).unwrap();
            for k in 0..3 {
                let expected = (u128::from(a[k]) * u128::from(s[0])
                    + u128::from(a[3 + k]) * u128::from(s[1]))
                    % u128::from(q);
                assert_eq!(actual[k], expected as u64);
            }
        }
    }

    #[test]
    fn wide_products_are_exact_for_large_keys() {
        use rand::{SeedableRng, rngs::StdRng};
        let mut rng = StdRng::seed_from_u64(11);
        for q in [17, 411_508_781_897, 2_305_843_009_213_693_951] {
            let (rows, inner, cols) = (3, 70, 4);
            let a = uniform(rows * inner, q, &mut rng).unwrap();
            let mut dt: Vec<i32> = (0..cols * inner)
                .map(|_| rng.random_range(-200_000..=200_000))
                .collect();
            dt[5] = i32::MIN + 1;
            dt[6] = i32::MAX;
            let actual = wide_right_product(&a, rows, inner, &dt, cols, q).unwrap();
            for i in 0..rows {
                for j in 0..cols {
                    let expected: i128 = (0..inner)
                        .map(|k| i128::from(a[i * inner + k]) * i128::from(dt[j * inner + k]))
                        .sum();
                    assert_eq!(
                        actual[i * cols + j],
                        expected.rem_euclid(i128::from(q)) as u64
                    );
                }
            }
            // Keys in [-127, 127] give the same result as the small product.
            let small: Vec<i32> = dt.iter().map(|&x| x % 128).collect();
            assert_eq!(
                wide_right_product(&a, rows, inner, &small, cols, q).unwrap(),
                small_right_product(&a, rows, inner, &small, cols, q).unwrap()
            );
        }
        assert!(wide_right_product(&[1], 1, 1, &[1], 2, 17).is_err());
        assert!(wide_right_product(&[17], 1, 1, &[1], 1, 17).is_err());
    }

    #[test]
    fn uniform_residues_are_canonical_and_cover_small_fields() {
        use rand::{SeedableRng, rngs::StdRng};
        let mut rng = StdRng::seed_from_u64(9);
        let values = uniform(20_000, 17, &mut rng).unwrap();
        let mut counts = [0usize; 17];
        for &v in &values {
            counts[v as usize] += 1;
        }
        // Each residue has probability 1/17; 6 standard deviations of 20000/17.
        assert!(
            counts
                .iter()
                .all(|&c| (c as f64 - 20_000.0 / 17.0).abs() < 200.0)
        );
        assert!(uniform(4, 15, &mut rng).is_err());
        assert!(uniform(0, 17, &mut rng).unwrap().is_empty());
    }

    #[test]
    fn sign_layout_and_unbounded_fallback() {
        let small: Vec<Integer> = (-4..=5).map(Integer::from).collect();
        let raw = [0x12, 0x5a, 0xff, 0x00, 0x03, 0x02];
        for (variant, mut y) in [small.clone(), small].into_iter().enumerate() {
            if variant == 1 {
                y[0] <<= 100;
            }
            let actual = sign_product_from_bytes(&y, &raw, 3).unwrap();
            for col in 0..3 {
                let mut expected = Integer::new();
                for (i, value) in y.iter().enumerate() {
                    if raw[(i / 8) * 3 + col] & (1 << (i % 8)) != 0 {
                        expected += value;
                    } else {
                        expected -= value;
                    }
                }
                assert_eq!(actual[col], expected);
            }
        }
        assert!(sign_product_from_bytes(&[], &[], 1).is_err());
        assert!(sign_product_from_bytes(&[Integer::from(1)], &[0, 1], 1).is_err());
    }

    #[test]
    fn affine_bijection_and_rank_deficiency() {
        let rows = vec![vec![1, 2, 3], vec![2, 4, 2]];
        let sampler = AffineSampler::new(rows.clone(), 3, 5).unwrap();
        assert!(sampler.full_rank());
        let mut outputs = std::collections::BTreeSet::new();
        for free in 0..5 {
            let s = sampler.from_free(&[1, 2], &[free]).unwrap();
            for (row, expected) in rows.iter().zip([1, 2]) {
                assert_eq!(
                    row.iter().zip(&s).map(|(a, b)| a * b).sum::<u64>() % 5,
                    expected
                );
            }
            assert!(outputs.insert(s));
        }
        let deficient = AffineSampler::new(vec![vec![1, 0, 0], vec![2, 0, 0]], 3, 17).unwrap();
        assert!(!deficient.full_rank());
        assert!(deficient.from_free(&[0, 0], &[0]).is_err());
    }

    #[test]
    fn ranges_cover_every_item_once_in_order() {
        for threads in [1, 2, 3, 7, 64] {
            for count in [0, 1, 2, 5, 1000] {
                let ranges = with_threads(threads, || map_ranges(count, 1 << 22, |r| r));
                assert!(ranges.len() <= PIECES_PER_THREAD * threads);
                assert!(threads > 1 || ranges.len() == 1);
                assert_eq!(ranges.first().map_or(0, |r| r.start), 0);
                assert_eq!(ranges.last().map_or(0, |r| r.end), count);
                assert!(ranges.windows(2).all(|w| w[0].end == w[1].start && w[0].start < w[0].end));
            }
        }
        // Steps started inside a worker run on the same pool.
        let nested = with_threads(4, || map_ranges(8, 1 << 22, |_| threads()));
        assert!(nested.len() > 1 && nested.iter().all(|&t| t == 4));
        assert_eq!(with_threads(3, threads), 3);
    }

    #[test]
    fn parallel_products_do_not_depend_on_the_thread_count() {
        use rand::{SeedableRng, rngs::StdRng};
        let q = 220_301_035_189;
        let mut rng = StdRng::seed_from_u64(77);
        // Every job here is at least 2^22 steps, so it is split over the threads.
        // A^T s for A of 64 rows and 2^16 columns, as in encryption.
        let (rows, cols) = (64, 1 << 16);
        let a = uniform(rows * cols, q, &mut rng).unwrap();
        let s = uniform(rows, q, &mut rng).unwrap();
        // One row times D (2^16 by 64), as in decryption, with key-sized and small entries.
        let (inner, count) = (1 << 16, 64);
        let top = uniform(inner, q, &mut rng).unwrap();
        let wide: Vec<i32> = (0..inner * count)
            .map(|_| rng.random_range(-60_000..=60_000))
            .collect();
        let small: Vec<i32> = (0..inner * count)
            .map(|_| rng.random_range(-127..=127))
            .collect();
        // R^T y for 2^14 errors (2048 sign groups) in blocks of 2048 columns, with
        // ordinary errors (i64 tables) and huge ones (Integer tables).
        let errors = 1 << 14;
        let ordinary: Vec<Integer> = (0..errors)
            .map(|_| Integer::from(rng.random_range(-200i64..=200)))
            .collect();
        let huge: Vec<Integer> = (0..errors)
            .map(|_| Integer::from(rng.random_range(-(1i64 << 50)..=1 << 50) << 12))
            .collect();
        assert!(matches!(SignTables::new(&huge).unwrap(), SignTables::Large(_)));
        assert!(matches!(SignTables::new(&ordinary).unwrap(), SignTables::Small(_)));
        let run = |threads| {
            with_threads(threads, || {
                (
                    transpose_matvec(&a, rows, cols, &s, q).unwrap(),
                    wide_right_product(&top, 1, inner, &wide, count, q).unwrap(),
                    small_right_product(&top, 1, inner, &small, count, q).unwrap(),
                    sign_product(&ordinary, 3000, 2048, &mut StdRng::seed_from_u64(5)).unwrap(),
                    sign_product(&huge, 3000, 2048, &mut StdRng::seed_from_u64(6)).unwrap(),
                )
            })
        };
        let one = run(1);
        // Spot checks against direct sums.
        for j in [0, 1, 4095, cols - 1] {
            let sum: u128 = (0..rows)
                .map(|i| u128::from(a[i * cols + j]) * u128::from(s[i]))
                .sum();
            assert_eq!(one.0[j], (sum % u128::from(q)) as u64);
        }
        for (d, product) in [(&wide, &one.1), (&small, &one.2)] {
            for c in [0, 31, count - 1] {
                let sum: i128 = (0..inner)
                    .map(|k| i128::from(top[k]) * i128::from(d[c * inner + k]))
                    .sum();
                assert_eq!(product[c], sum.rem_euclid(i128::from(q)) as u64);
            }
        }
        for (y, product, seed) in [(&ordinary, &one.3, 5), (&huge, &one.4, 6)] {
            let mut raw = vec![0u8; (errors / 8) * 2048];
            StdRng::seed_from_u64(seed).fill_bytes(&mut raw);
            for j in [0, 777, 2047] {
                let mut expected = Integer::new();
                for (i, value) in y.iter().enumerate() {
                    if raw[(i / 8) * 2048 + j] & (1 << (i % 8)) != 0 {
                        expected += value;
                    } else {
                        expected -= value;
                    }
                }
                assert_eq!(product[j], expected);
            }
        }
        for threads in [2, 3, 8] {
            assert!(run(threads) == one, "{threads} threads");
        }
        // A noncanonical entry late in A is found while A is read, on any thread,
        // and the errors keep their order (residues before the accumulator bound).
        let noncanonical = Err(Error("noncanonical field element"));
        let mut bad = a.clone();
        bad[rows * cols - 5] = q;
        for threads in [1, 3] {
            let result = with_threads(threads, || transpose_matvec(&bad, rows, cols, &s, q));
            assert_eq!(result, noncanonical, "{threads} threads");
        }
        let (big, many) = (2_305_843_009_213_693_951, 65);
        assert_eq!(
            transpose_matvec(&[big; 65], many, 1, &[0; 65], big),
            noncanonical
        );
        assert_eq!(
            transpose_matvec(&[0; 65], many, 1, &[0; 65], big),
            Err(Error("u128 accumulator bound exceeded"))
        );
        let mut wide_bad = wide.clone();
        wide_bad[inner * count - 1] = i32::MIN;
        assert!(wide_right_product(&top, 1, inner, &wide_bad, count, q).is_ok());
        assert_eq!(
            with_threads(3, || small_right_product(&top, 1, inner, &wide_bad, count, q)),
            Err(Error("right matrix entries must lie in [-127,127]"))
        );
    }

    #[test]
    fn one_row_binary_products_split_over_threads_are_exact() {
        use rand::{SeedableRng, rngs::StdRng};
        let mut rng = StdRng::seed_from_u64(78);
        // Hidden decryption multiplies one row by L binary keys. Masked sums (2
        // keys), 4-entry tables (16) and 8-entry tables (300), with u64 sums and,
        // for the Mersenne prime 2^61 - 1, u128 sums; a padded last block.
        for (q, inner, count) in [
            (220_301_035_189u64, (1 << 16) + 37, 2),
            (220_301_035_189, 1 << 16, 16),
            (220_301_035_189, 9_000, 300),
            (2_305_843_009_213_693_951, (1 << 16) + 5, 3),
        ] {
            let row = uniform(inner, q, &mut rng).unwrap();
            let bits: Vec<i32> = (0..count * inner).map(|_| rng.random_range(0..2)).collect();
            let words = packed_words(inner);
            let mut packed = vec![0u64; words * count];
            for (k, key) in bits.chunks_exact(inner).enumerate() {
                for (j, &bit) in key.iter().enumerate() {
                    packed[k * words + j / 64] |= (bit as u64) << (j % 64);
                }
            }
            let expected: Vec<u64> = bits
                .chunks_exact(inner)
                .map(|key| {
                    let sum: u128 = row
                        .iter()
                        .zip(key)
                        .map(|(&x, &bit)| u128::from(x) * bit as u128)
                        .sum();
                    (sum % u128::from(q)) as u64
                })
                .collect();
            for threads in [1, 3, 8] {
                let actual =
                    with_threads(threads, || binary_right_product(&row, 1, inner, &packed, count, q));
                assert_eq!(actual.unwrap(), expected, "q {q}, {count} keys, {threads} threads");
            }
        }
    }

    #[test]
    fn affine_elimination_and_solutions_do_not_depend_on_the_thread_count() {
        use rand::{SeedableRng, rngs::StdRng};
        let q = 220_301_035_189;
        let mut rng = StdRng::seed_from_u64(79);
        // 64 constraints on 1424 coordinates as in full32, and a rank-deficient set.
        let (g, n) = (64, 1424);
        let mut rows: Vec<Vec<u64>> = (0..g).map(|_| uniform(n, q, &mut rng).unwrap()).collect();
        let mut deficient = rows.clone();
        deficient[40] = deficient[3].clone();
        let rhs = uniform(g, q, &mut rng).unwrap();
        let free_values = uniform(n - g, q, &mut rng).unwrap();
        let run = |rows: &Vec<Vec<u64>>, threads| {
            with_threads(threads, || {
                let sampler = AffineSampler::new(rows.clone(), n, q).unwrap();
                let solution = sampler.full_rank().then(|| sampler.from_free(&rhs, &free_values).unwrap());
                (sampler.rows.clone(), sampler.transform.clone(), sampler.pivots.clone(), solution)
            })
        };
        let one = run(&rows, 1);
        let solution = one.3.clone().unwrap();
        for (row, &target) in rows.iter().zip(&rhs) {
            let sum: u128 = row.iter().zip(&solution).map(|(&a, &x)| u128::from(a) * u128::from(x) % u128::from(q)).sum();
            assert_eq!((sum % u128::from(q)) as u64, target);
        }
        let fixed = run(&deficient, 1);
        assert!(fixed.3.is_none() && fixed.2.len() == g - 1);
        for threads in [2, 3, 8] {
            assert!(run(&rows, threads) == one, "{threads} threads");
            assert!(run(&deficient, threads) == fixed, "deficient, {threads} threads");
        }
        rows[0][0] = q;
        assert!(AffineSampler::new(rows, n, q).is_err());
    }
}
