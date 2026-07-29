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
pub fn eye<const N: usize>() -> Mat<N, N> {
    let mut out = [[0.0; N]; N];
    for (i, row) in out.iter_mut().enumerate() {
        row[i] = 1.0;
    }
    out
}

/// `a * b`.
pub fn matmul<const A: usize, const B: usize, const C: usize>(
    a: &Mat<A, B>,
    b: &Mat<B, C>,
) -> Mat<A, C> {
    let mut out = [[0.0f32; C]; A];
    for i in 0..A {
        for k in 0..B {
            let aik = a[i][k];
            if aik == 0.0 {
                continue;
            }
            for j in 0..C {
                out[i][j] += aik * b[k][j];
            }
        }
    }
    out
}

/// `a * b^T`.
pub fn matmul_transpose<const A: usize, const B: usize, const C: usize>(
    a: &Mat<A, B>,
    b: &Mat<C, B>,
) -> Mat<A, C> {
    let mut out = [[0.0f32; C]; A];
    for i in 0..A {
        for j in 0..C {
            let mut acc = 0.0;
            for k in 0..B {
                acc += a[i][k] * b[j][k];
            }
            out[i][j] = acc;
        }
    }
    out
}

/// `a * v`.
pub fn matvec<const A: usize, const B: usize>(a: &Mat<A, B>, v: &Vec<B>) -> Vec<A> {
    let mut out = [0.0f32; A];
    for i in 0..A {
        let mut acc = 0.0;
        for k in 0..B {
            acc += a[i][k] * v[k];
        }
        out[i] = acc;
    }
    out
}

/// `a^T`.
pub fn transpose<const A: usize, const B: usize>(a: &Mat<A, B>) -> Mat<B, A> {
    let mut out = [[0.0f32; A]; B];
    for i in 0..A {
        for j in 0..B {
            out[j][i] = a[i][j];
        }
    }
    out
}

/// `a + diag(d)`, in place.
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
pub fn cholesky<const N: usize>(a: &Mat<N, N>) -> Option<Mat<N, N>> {
    let mut l = [[0.0f32; N]; N];
    for i in 0..N {
        for j in 0..=i {
            let mut acc = a[i][j];
            for k in 0..j {
                acc -= l[i][k] * l[j][k];
            }
            if i == j {
                if !(acc > 0.0) {
                    return None;
                }
                l[i][j] = libm::sqrtf(acc);
            } else {
                l[i][j] = acc / l[j][j];
            }
        }
    }
    Some(l)
}

/// Solve `L L^T x = b` for one right-hand side.
pub fn cholesky_solve_vec<const N: usize>(l: &Mat<N, N>, b: &Vec<N>) -> Vec<N> {
    let mut y = [0.0f32; N];
    for i in 0..N {
        let mut acc = b[i];
        for k in 0..i {
            acc -= l[i][k] * y[k];
        }
        y[i] = acc / l[i][i];
    }
    let mut x = [0.0f32; N];
    for i in (0..N).rev() {
        let mut acc = y[i];
        for k in (i + 1)..N {
            acc -= l[k][i] * x[k];
        }
        x[i] = acc / l[i][i];
    }
    x
}

/// Solve `L L^T X = B` for `COLS` right-hand sides at once.
pub fn cholesky_solve_mat<const N: usize, const COLS: usize>(
    l: &Mat<N, N>,
    b: &Mat<N, COLS>,
) -> Mat<N, COLS> {
    let mut out = [[0.0f32; COLS]; N];
    for c in 0..COLS {
        let mut column = [0.0f32; N];
        for i in 0..N {
            column[i] = b[i][c];
        }
        let solved = cholesky_solve_vec(l, &column);
        for i in 0..N {
            out[i][c] = solved[i];
        }
    }
    out
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
