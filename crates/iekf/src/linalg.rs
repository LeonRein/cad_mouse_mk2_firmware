//! Fixed-size dense linear algebra, sized at compile time.
//!
//! Every matrix is a plain row-major `[[f32; COLS]; ROWS]` on the stack, so
//! there is no allocation, no shape check at run time, and nothing to fail
//! except the factorisation itself. The sizes this crate is built for are
//! small -- a handful of states against a handful of measurements -- which is
//! the regime where a dedicated matrix library costs more in code size and
//! compile time than it saves.

/// Row-major matrix.
pub type Mat<const ROWS: usize, const COLS: usize> = [[f32; COLS]; ROWS];

/// Column vector.
pub type Vec<const N: usize> = [f32; N];

/// Identity.
#[inline(always)]
pub fn eye<const N: usize>() -> Mat<N, N> {
    let mut out = [[0.0; N]; N];
    for (i, row) in out.iter_mut().enumerate() {
        row[i] = 1.0;
    }
    out
}

/// `a * b`.
///
/// The loop order is `i, j, k` with a *register* accumulator, not the more
/// natural `i, k, j` accumulating into `out`. On a single-issue FPU that is
/// the whole ballgame: `i, k, j` turns every multiply-accumulate into a
/// load-fma-store against the stack, so a 9x6 by 6x6 costs 324 round trips to
/// memory instead of 324 register FMAs and 54 stores. Measured on target, the
/// filter step was doing 2.5 memory accesses per arithmetic operation before
/// this was changed.
///
/// It also drops a `if aik == 0.0 { continue; }` guard that used to sit in the
/// inner loop. Skipping a zero looks free and is not: reading an FPU compare
/// back into the core flags (`vcmp` then `vmrs APSR_nzcv`) stalls the pipeline,
/// and neither the covariance nor the measurement Jacobian is sparse, so the
/// branch never actually fired.
#[inline(always)]
pub fn matmul<const A: usize, const B: usize, const C: usize>(
    a: &Mat<A, B>,
    b: &Mat<B, C>,
) -> Mat<A, C> {
    // `from_fn` rather than zero-then-fill: every element is written exactly
    // once here, so the `[[0.0; C]; A]` that used to open this function was
    // dead -- but LLVM could not prove it and emitted a call to
    // `__aeabi_memclr4`, which lives in flash, for each one. Fifteen of those
    // per filter step, from code deliberately relocated to SRAM.
    core::array::from_fn(|i| {
        core::array::from_fn(|j| {
            let mut acc = 0.0;
            for k in 0..B {
                acc += a[i][k] * b[k][j];
            }
            acc
        })
    })
}

/// `a * b^T`, where the result is known to be symmetric.
///
/// Computes the lower triangle and mirrors it. For `S = (J P) J^T` that is 45
/// entries instead of 81, and it makes the result symmetric *exactly* rather
/// than to within rounding -- so the [`symmetrise`] pass that used to follow
/// it is not merely cheaper, it is unnecessary.
///
/// # Correctness
///
/// The caller is asserting the symmetry, not the function: `a * b^T` is
/// symmetric only for particular `a` and `b`. It holds for `J P` against `J`
/// because `P` is symmetric. Passing anything else silently discards the upper
/// triangle.
#[inline(always)]
pub fn matmul_transpose_symmetric<const A: usize, const B: usize>(
    a: &Mat<A, B>,
    b: &Mat<A, B>,
) -> Mat<A, A> {
    let mut out = [[0.0f32; A]; A];
    for i in 0..A {
        for j in 0..=i {
            let mut acc = 0.0;
            for k in 0..B {
                acc += a[i][k] * b[j][k];
            }
            out[i][j] = acc;
            out[j][i] = acc;
        }
    }
    out
}

/// `a * b^T`.
#[inline(always)]
pub fn matmul_transpose<const A: usize, const B: usize, const C: usize>(
    a: &Mat<A, B>,
    b: &Mat<C, B>,
) -> Mat<A, C> {
    core::array::from_fn(|i| {
        core::array::from_fn(|j| {
            let mut acc = 0.0;
            for k in 0..B {
                acc += a[i][k] * b[j][k];
            }
            acc
        })
    })
}

/// `a * v`.
#[inline(always)]
pub fn matvec<const A: usize, const B: usize>(a: &Mat<A, B>, v: &Vec<B>) -> Vec<A> {
    core::array::from_fn(|i| {
        let mut acc = 0.0;
        for k in 0..B {
            acc += a[i][k] * v[k];
        }
        acc
    })
}

/// `a^T`.
#[inline(always)]
pub fn transpose<const A: usize, const B: usize>(a: &Mat<A, B>) -> Mat<B, A> {
    core::array::from_fn(|j| core::array::from_fn(|i| a[i][j]))
}

/// `a + diag(d)`, in place.
#[inline(always)]
pub fn add_diagonal<const N: usize>(a: &mut Mat<N, N>, d: &Vec<N>) {
    for i in 0..N {
        a[i][i] += d[i];
    }
}

/// Force exact symmetry by averaging with the transpose.
///
/// The covariance is symmetric in exact arithmetic and drifts in `f32`. Left
/// alone the asymmetry compounds through the Joseph update and eventually
/// breaks the Cholesky, which is a very confusing way to discover the problem.
#[inline(always)]
pub fn symmetrise<const N: usize>(a: &mut Mat<N, N>) {
    for i in 0..N {
        for j in (i + 1)..N {
            let mean = 0.5 * (a[i][j] + a[j][i]);
            a[i][j] = mean;
            a[j][i] = mean;
        }
    }
}

/// Lower-triangular Cholesky factor `L` with `a == L L^T`.
///
/// Returns `None` if `a` is not positive definite, which for a covariance
/// means something upstream has already gone wrong -- a negative process
/// noise, a measurement noise of zero, or a divergence. Callers are expected
/// to treat that as a reset condition rather than to paper over it.
#[inline(always)]
pub fn cholesky<const N: usize>(a: &Mat<N, N>) -> Option<Mat<N, N>> {
    let mut l = [[0.0f32; N]; N];
    // Column by column rather than row by row, purely so that the reciprocal
    // of the pivot is formed once per column instead of once per entry below
    // it. A divide is an order of magnitude dearer than a multiply on this
    // FPU and there were 36 of them per factorisation.
    for j in 0..N {
        let mut acc = a[j][j];
        for k in 0..j {
            acc -= l[j][k] * l[j][k];
        }
        if !(acc > 0.0) {
            return None;
        }
        let d = libm::sqrtf(acc);
        l[j][j] = d;
        let inv = 1.0 / d;
        for i in (j + 1)..N {
            let mut acc = a[i][j];
            for k in 0..j {
                acc -= l[i][k] * l[j][k];
            }
            l[i][j] = acc * inv;
        }
    }
    Some(l)
}

/// Reciprocals of a Cholesky factor's diagonal, for reuse across right-hand
/// sides.
///
/// Both substitutions divide by `l[i][i]`, twice per element per right-hand
/// side. With six of them that was 108 divides per solve, all by the same nine
/// numbers.
#[inline(always)]
fn inverse_diagonal<const N: usize>(l: &Mat<N, N>) -> Vec<N> {
    core::array::from_fn(|i| 1.0 / l[i][i])
}

/// Solve `L L^T x = b` given the factor's precomputed inverse diagonal.
#[inline(always)]
fn solve_with_inverse_diagonal<const N: usize>(
    l: &Mat<N, N>,
    inv_diag: &Vec<N>,
    b: &Vec<N>,
) -> Vec<N> {
    let mut y = [0.0f32; N];
    for i in 0..N {
        let mut acc = b[i];
        for k in 0..i {
            acc -= l[i][k] * y[k];
        }
        y[i] = acc * inv_diag[i];
    }
    let mut x = [0.0f32; N];
    for i in (0..N).rev() {
        let mut acc = y[i];
        for k in (i + 1)..N {
            acc -= l[k][i] * x[k];
        }
        x[i] = acc * inv_diag[i];
    }
    x
}

/// Solve `L L^T x = b` for one right-hand side.
#[inline(always)]
pub fn cholesky_solve_vec<const N: usize>(l: &Mat<N, N>, b: &Vec<N>) -> Vec<N> {
    solve_with_inverse_diagonal(l, &inverse_diagonal(l), b)
}

/// Solve `L L^T X = B` for `COLS` right-hand sides, returning `X^T`.
///
/// Transposed on the way out because every caller here wants it that way: the
/// Kalman gain is `K = P J^T S^-1`, obtained as `(S^-1 (J P))^T` because `S` is
/// symmetric. Producing `X` and then transposing it costs a full extra pass
/// over the array for nothing -- the scatter below writes each solved column
/// straight into the row it belongs in.
#[inline(always)]
pub fn cholesky_solve_mat_transposed<const N: usize, const COLS: usize>(
    l: &Mat<N, N>,
    b: &Mat<N, COLS>,
) -> Mat<COLS, N> {
    let inv_diag = inverse_diagonal(l);
    core::array::from_fn(|c| {
        let column: Vec<N> = core::array::from_fn(|i| b[i][c]);
        solve_with_inverse_diagonal(l, &inv_diag, &column)
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx<const A: usize, const B: usize>(got: &Mat<A, B>, want: &Mat<A, B>, tol: f32) {
        for i in 0..A {
            for j in 0..B {
                assert!(
                    (got[i][j] - want[i][j]).abs() <= tol,
                    "[{i}][{j}]: {} vs {}",
                    got[i][j],
                    want[i][j]
                );
            }
        }
    }

    #[test]
    fn matmul_matches_by_hand() {
        let a: Mat<2, 3> = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];
        let b: Mat<3, 2> = [[7.0, 8.0], [9.0, 10.0], [11.0, 12.0]];
        approx(&matmul(&a, &b), &[[58.0, 64.0], [139.0, 154.0]], 1e-5);
    }

    #[test]
    fn matmul_transpose_agrees_with_explicit_transpose() {
        let a: Mat<2, 3> = [[1.0, -2.0, 0.5], [4.0, 5.0, -6.0]];
        let b: Mat<4, 3> = [
            [7.0, 8.0, 1.0],
            [9.0, 10.0, -1.0],
            [11.0, 12.0, 0.0],
            [0.5, -0.5, 2.0],
        ];
        approx(&matmul_transpose(&a, &b), &matmul(&a, &transpose(&b)), 1e-4);
    }

    #[test]
    fn cholesky_reproduces_the_matrix() {
        let a: Mat<3, 3> = [[4.0, 1.0, 0.5], [1.0, 3.0, -0.25], [0.5, -0.25, 2.0]];
        let l = cholesky(&a).expect("positive definite");
        approx(&matmul_transpose(&l, &l), &a, 1e-5);
    }

    #[test]
    fn cholesky_solve_inverts() {
        let a: Mat<3, 3> = [[4.0, 1.0, 0.5], [1.0, 3.0, -0.25], [0.5, -0.25, 2.0]];
        let l = cholesky(&a).unwrap();
        let b: Vec<3> = [1.0, -2.0, 3.0];
        let x = cholesky_solve_vec(&l, &b);
        let back = matvec(&a, &x);
        for i in 0..3 {
            assert!((back[i] - b[i]).abs() < 1e-4, "{} vs {}", back[i], b[i]);
        }
    }

    #[test]
    fn cholesky_rejects_indefinite() {
        let a: Mat<2, 2> = [[1.0, 2.0], [2.0, 1.0]];
        assert!(cholesky(&a).is_none());
    }
}
