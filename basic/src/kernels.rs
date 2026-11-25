//! Exact products with the public matrix.
//!
//! Every output coordinate is an exact integer sum of products of canonical
//! values. It is accumulated in `u64` when [`Field`] confirms that the sum
//! cannot overflow, otherwise in `u128`, and then reduced once by
//! [`Field::reduce`] or [`Field::reduce_wide`] (both `crypto-bigint`). There is
//! no modular arithmetic of our own in this module, only plain integer sums.
//! The loops are written so that the compiler vectorizes them.

use crate::Mode;
use crate::field::Field;
use rayon::prelude::*;
use std::ops::{Add, Mul};

/// Field updates per pivot step from which `Mode::Multi` splits an elimination.
const PARALLEL_ELIMINATION: usize = 1 << 14;

/// Rows of the matrix that one pass of [`transposed_block`] combines. Adding
/// several rows per pass reads and writes the accumulators less often.
const ROWS: usize = 8;

/// `out[j] = sum_i a[i m + c0 + j] s[i] mod q` for every `j < out.len()`, where
/// `a` is row-major with `s.len()` rows of length `m` and all inputs are canonical.
pub(crate) fn transposed_block(
    field: &Field,
    a: &[u32],
    m: usize,
    s: &[u32],
    c0: usize,
    out: &mut [u32],
) {
    assert!(
        c0 + out.len() <= m && a.len() >= s.len() * m,
        "block outside the matrix"
    );
    if field.products_fit_u64(s.len()) {
        let mut acc = vec![0u64; out.len()];
        transposed_sums(a, m, s, c0, &mut acc);
        for (o, &x) in out.iter_mut().zip(&acc) {
            *o = field.reduce(x);
        }
    } else {
        let mut acc = vec![0u128; out.len()];
        transposed_sums(a, m, s, c0, &mut acc);
        for (o, &x) in out.iter_mut().zip(&acc) {
            *o = field.reduce_wide(x);
        }
    }
}

/// `acc[j] = sum_i a[i m + c0 + j] s[i]` over the integers. The caller chooses
/// an accumulator type in which these sums cannot overflow.
fn transposed_sums<T>(a: &[u32], m: usize, s: &[u32], c0: usize, acc: &mut [T])
where
    T: Copy + Default + From<u32> + Add<Output = T> + Mul<Output = T>,
{
    let w = acc.len();
    acc.fill(T::default());
    let row = |i: usize| &a[i * m + c0..i * m + c0 + w];
    let mut blocks = s.chunks_exact(ROWS);
    let mut i = 0;
    for block in &mut blocks {
        let (r0, r1, r2, r3) = (row(i), row(i + 1), row(i + 2), row(i + 3));
        let (r4, r5, r6, r7) = (row(i + 4), row(i + 5), row(i + 6), row(i + 7));
        let c = |k: usize| T::from(block[k]);
        let (s0, s1, s2, s3) = (c(0), c(1), c(2), c(3));
        let (s4, s5, s6, s7) = (c(4), c(5), c(6), c(7));
        for (j, x) in acc.iter_mut().enumerate() {
            *x = *x
                + T::from(r0[j]) * s0
                + T::from(r1[j]) * s1
                + T::from(r2[j]) * s2
                + T::from(r3[j]) * s3
                + T::from(r4[j]) * s4
                + T::from(r5[j]) * s5
                + T::from(r6[j]) * s6
                + T::from(r7[j]) * s7;
        }
        i += ROWS;
    }
    for &si in blocks.remainder() {
        let si = T::from(si);
        for (x, &value) in acc.iter_mut().zip(row(i)) {
            *x = *x + T::from(value) * si;
        }
        i += 1;
    }
}

/// The mask of a binary key: `u32::MAX` where the key bit is 1, 0 where it is
/// 0, so that `x & mask[j] = x key[j]`. Keys must have entries 0 or 1.
pub(crate) fn key_mask(key: &[u8]) -> Vec<u32> {
    debug_assert!(key.iter().all(|&b| b <= 1), "binary key");
    key.iter().map(|&b| u32::from(b) * u32::MAX).collect()
}

/// Binary products. For row `r` of the row-major `rows` (rows of length `len`)
/// and key `k`, `out[r * masks.len() + k] = sum_j rows[r][j] key_k[j] mod q`,
/// where `masks[k]` is the [`key_mask`] of `key_k`.
pub(crate) fn binary_rows(
    field: &Field,
    rows: &[u32],
    len: usize,
    masks: &[&[u32]],
    out: &mut [u32],
) {
    let count = masks.len();
    if len == 0 || count == 0 {
        out.fill(0);
        return;
    }
    assert!(rows.len().is_multiple_of(len) && out.len() == rows.len() / len * count);
    assert!(masks.iter().all(|mask| mask.len() == len));
    // A sum of `len` canonical values decides the accumulator width.
    let narrow = field.values_fit_u64(len);
    for (row, out) in rows.chunks_exact(len).zip(out.chunks_exact_mut(count)) {
        for (mask, slot) in masks.iter().zip(out.iter_mut()) {
            let selected = row.iter().zip(*mask).map(|(&x, &m)| x & m);
            *slot = if narrow {
                field.reduce(selected.map(u64::from).sum())
            } else {
                field.reduce_wide(selected.map(u128::from).sum())
            };
        }
    }
}

/// `sum_j a[j] b[j] mod q` for canonical inputs of equal length.
pub(crate) fn dot(field: &Field, a: &[u32], b: &[u32]) -> u32 {
    assert_eq!(a.len(), b.len());
    if field.products_fit_u64(a.len()) {
        field.reduce(
            a.iter()
                .zip(b)
                .map(|(&x, &y)| u64::from(x) * u64::from(y))
                .sum(),
        )
    } else {
        field.reduce_wide(
            a.iter()
                .zip(b)
                .map(|(&x, &y)| u128::from(x) * u128::from(y))
                .sum(),
        )
    }
}

/// Gauss-Jordan elimination of `matrix` (canonical rows of length `n`)
/// augmented by the identity, with exactly the pivot rule of the reference
/// code: for each column in order, the first row at or below the current rank
/// with a nonzero entry is swapped up, scaled to a unit pivot and used to clear
/// the column in every other row. Returns the reduced augmented rows (length
/// `n + d`) and the pivot columns. Every operation is a [`Field`] operation;
/// a row update `x - c p` is computed as `(q - c) p + x`, exact in `u64`.
///
/// `Mode::Multi` clears the pivot column in the other rows on the rayon pool.
/// The rows are independent and every entry gets the same field operations, so
/// the result is the same in both modes.
pub(crate) fn gauss_jordan_with(
    field: &Field,
    matrix: &[Vec<u32>],
    n: usize,
    mode: Mode,
) -> (Vec<Vec<u32>>, Vec<usize>) {
    let d = matrix.len();
    let mut rows: Vec<Vec<u32>> = matrix
        .iter()
        .enumerate()
        .map(|(i, row)| {
            let mut augmented = row.clone();
            augmented.extend((0..d).map(|j| u32::from(i == j)));
            augmented
        })
        .collect();
    let mut pivots = Vec::with_capacity(d);
    for column in 0..n {
        let rank = pivots.len();
        let Some(source) = (rank..d).find(|&i| rows[i][column] != 0) else {
            continue;
        };
        rows.swap(rank, source);
        let inverse = field.inverse(rows[rank][column]);
        for x in rows[rank].iter_mut() {
            *x = field.mul(*x, inverse);
        }
        // Columns left of `column` are zero in the pivot row (they are earlier
        // pivot columns or columns that were already zero below the rank).
        let pivot = rows[rank][column..].to_vec();
        let clear = |(i, row): (usize, &mut Vec<u32>)| {
            let coefficient = row[column];
            if i == rank || coefficient == 0 {
                return;
            }
            let minus = field.neg(coefficient);
            for (x, &p) in row[column..].iter_mut().zip(&pivot) {
                *x = field.mul_add(minus, p, *x);
            }
        };
        // Small eliminations stay on the calling thread in both modes.
        if mode == Mode::Multi && d.saturating_mul(pivot.len()) >= PARALLEL_ELIMINATION {
            rows.par_iter_mut().enumerate().for_each(clear);
        } else {
            rows.iter_mut().enumerate().for_each(clear);
        }
        pivots.push(column);
        if pivots.len() == d {
            break;
        }
    }
    (rows, pivots)
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::{RngExt, SeedableRng, rngs::StdRng};

    const MODULI: [u32; 5] = [3, 257, 67537879, 2147483647, 4294967291];

    fn naive_transposed(a: &[u32], n: usize, m: usize, s: &[u32], q: u32) -> Vec<u32> {
        (0..m)
            .map(|j| {
                ((0..n)
                    .map(|i| u128::from(a[i * m + j]) * u128::from(s[i]))
                    .sum::<u128>()
                    % u128::from(q)) as u32
            })
            .collect()
    }

    #[test]
    fn transposed_block_matches_naive_in_both_accumulators() {
        let mut rng = StdRng::seed_from_u64(1);
        for q in MODULI {
            let field = Field::new(q);
            for (n, m) in [(1, 1), (7, 5), (8, 9), (17, 33), (40, 70)] {
                let a: Vec<u32> = (0..n * m).map(|_| rng.random_range(0..q)).collect();
                let s: Vec<u32> = (0..n).map(|_| rng.random_range(0..q)).collect();
                let expected = naive_transposed(&a, n, m, &s, q);
                for width in [1, 3, m] {
                    let mut c0 = 0;
                    while c0 < m {
                        let w = width.min(m - c0);
                        let mut out = vec![7u32; w];
                        transposed_block(&field, &a, m, &s, c0, &mut out);
                        assert_eq!(out, expected[c0..c0 + w]);
                        c0 += w;
                    }
                }
            }
        }
        // Extreme values at and beyond the u64 bound of each modulus.
        for (q, n) in [(67537879u32, 4044), (67537879, 4045), (4294967291, 5)] {
            let field = Field::new(q);
            let m = 3;
            let a = vec![q - 1; n * m];
            let s = vec![q - 1; n];
            let mut out = vec![0u32; m];
            transposed_block(&field, &a, m, &s, 0, &mut out);
            assert_eq!(out, naive_transposed(&a, n, m, &s, q));
        }
    }

    #[test]
    fn binary_rows_match_naive() {
        let mut rng = StdRng::seed_from_u64(2);
        for q in MODULI {
            let field = Field::new(q);
            for (rows, len) in [(1, 1), (2, 7), (3, 8), (2, 9), (4, 4100), (1, 33000)] {
                let a: Vec<u32> = (0..rows * len).map(|_| rng.random_range(0..q)).collect();
                let bits: Vec<Vec<u8>> = (0..5)
                    .map(|_| (0..len).map(|_| rng.random_range(0..2u8)).collect())
                    .collect();
                let masks: Vec<Vec<u32>> = bits.iter().map(|b| key_mask(b)).collect();
                let keys: Vec<&[u32]> = masks.iter().map(Vec::as_slice).collect();
                let mut out = vec![0u32; rows * keys.len()];
                binary_rows(&field, &a, len, &keys, &mut out);
                for r in 0..rows {
                    for (k, key) in bits.iter().enumerate() {
                        let expected = (0..len)
                            .map(|j| u128::from(a[r * len + j]) * u128::from(key[j]))
                            .sum::<u128>()
                            % u128::from(q);
                        assert_eq!(u128::from(out[r * keys.len() + k]), expected);
                    }
                }
            }
            // All-ones keys over maximal entries.
            let len = 70000;
            let a = vec![q - 1; len];
            let key = key_mask(&vec![1u8; len]);
            assert!(key.iter().all(|&m| m == u32::MAX));
            let mut out = [0u32];
            binary_rows(&field, &a, len, &[&key], &mut out);
            let expected = (u128::from(q - 1) * len as u128 % u128::from(q)) as u32;
            assert_eq!(out[0], expected);
        }
    }

    #[test]
    fn dot_products_match_naive() {
        let mut rng = StdRng::seed_from_u64(3);
        for q in MODULI {
            let field = Field::new(q);
            for len in [0, 1, 5, 1000, 9000] {
                let a: Vec<u32> = (0..len).map(|_| rng.random_range(0..q)).collect();
                let b: Vec<u32> = (0..len).map(|_| rng.random_range(0..q)).collect();
                let expected = a
                    .iter()
                    .zip(&b)
                    .map(|(&x, &y)| u128::from(x) * u128::from(y))
                    .sum::<u128>()
                    % u128::from(q);
                assert_eq!(u128::from(dot(&field, &a, &b)), expected);
            }
        }
    }

    #[test]
    fn gauss_jordan_gives_the_reduced_form_and_its_transformation() {
        let mut rng = StdRng::seed_from_u64(4);
        for q in MODULI {
            let field = Field::new(q);
            for (d, n) in [(1, 1), (2, 5), (3, 3), (4, 9), (6, 20)] {
                let mut matrix: Vec<Vec<u32>> = (0..d)
                    .map(|_| (0..n).map(|_| rng.random_range(0..q)).collect())
                    .collect();
                if d > 2 {
                    // A dependent row and a zero column exercise rank deficiency.
                    matrix[2] = matrix[0].clone();
                    for row in &mut matrix {
                        row[0] = 0;
                    }
                }
                let (rows, pivots) = gauss_jordan_with(&field, &matrix, n, Mode::Single);
                assert_eq!(
                    gauss_jordan_with(&field, &matrix, n, Mode::Multi),
                    (rows.clone(), pivots.clone())
                );
                // T M = R, where T and R are the right and left blocks.
                for row in &rows {
                    for j in 0..n {
                        let combined = (0..d).fold(0, |acc, k| {
                            field.add(acc, field.mul(row[n + k], matrix[k][j]))
                        });
                        assert_eq!(combined, row[j], "q {q} d {d} n {n}");
                    }
                }
                // Unit pivots, cleared pivot columns and zero rows below the rank.
                for (i, &p) in pivots.iter().enumerate() {
                    for (k, row) in rows.iter().enumerate() {
                        assert_eq!(row[p], u32::from(i == k));
                    }
                }
                for row in &rows[pivots.len()..] {
                    assert!(row[..n].iter().all(|&x| x == 0));
                }
            }
        }
    }
}
