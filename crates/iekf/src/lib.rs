//! Iterated extended Kalman filter, allocation-free and `no_std`.
//!
//! Sized entirely at compile time: `N` states against `M` measurements, every
//! matrix a stack array, no allocator and no fallible resize. Built for the
//! case where a filter has to meet a hard deadline on a microcontroller and
//! the sizes are known when the firmware is built.
//!
//! # Why iterated
//!
//! A plain EKF linearises the measurement function once, at the prior mean,
//! and takes a single step from there. The iterated form repeats the update,
//! relinearising about the current estimate while holding the prior fixed --
//! Gauss-Newton on the maximum-a-posteriori estimate. The extra `(prior_x - x)`
//! term in the residual is what makes it *iterated* rather than merely
//! repeated: it keeps every pass anchored to the prior mean while the
//! linearisation point moves.
//!
//! Against an unscented filter it is far cheaper when the measurement function
//! dominates the cost. A UKF evaluates `h` at `2N + 1` sigma points; this
//! evaluates it once per iteration. At `N = 6` that is thirteen evaluations
//! against two, and if `h` is an interpolated physical model rather than a few
//! multiplies, that ratio is the whole compute budget.
//!
//! # What is assumed
//!
//! * **Process model.** A random walk by default -- the state is unchanged and
//!   only its uncertainty grows. [`IteratedEkf::predict_linear`] covers a
//!   general linear transition. A nonlinear one is out of scope; that wants a
//!   different crate.
//! * **Diagonal noise.** Both the process noise and the measurement noise are
//!   given as diagonals. Correlated measurement noise would need a full `R`
//!   and cost an `M x M` multiply per update for a generality that sensor data
//!   sheets rarely support anyway.
//! * **`f32` throughout.** Single-precision, because the targets this exists
//!   for have a single-precision FPU and double is soft-float there. The
//!   Joseph-form covariance update and the explicit symmetrisation are both
//!   there to make `f32` hold up over long runs.
//!
//! # Example
//!
//! ```
//! use iekf::{IteratedEkf, MeasurementModel};
//!
//! // Two states, one measurement: position observed directly.
//! struct Observe;
//! impl MeasurementModel<2, 1> for Observe {
//!     fn predict_and_jacobian(&self, x: &[f32; 2]) -> ([f32; 1], [[f32; 2]; 1]) {
//!         ([x[0]], [[1.0, 0.0]])
//!     }
//! }
//!
//! let mut ekf = IteratedEkf::<2, 1>::new([0.0, 0.0], [1.0, 1.0], [0.04]);
//! ekf.set_process_noise([0.1, 0.1]);
//! ekf.predict(0.01);
//! ekf.update(&Observe, &[1.0]).unwrap();
//! assert!(ekf.state()[0] > 0.9);
//! ```

#![cfg_attr(not(test), no_std)]

pub mod linalg;

use linalg::{Mat, Vec};

/// What the filter needs to know about the thing it is observing.
///
/// The Jacobian must be `d(measurement)/d(state)` in the *same* parametrisation
/// the filter adds its correction to. If the state lives on a manifold and the
/// natural Jacobian is written for a local perturbation, convert it before
/// returning it here -- silently mixing the two conventions produces a filter
/// that works beautifully near zero and degrades with no obvious cause.
pub trait MeasurementModel<const N: usize, const M: usize> {
    /// Predicted measurement and its Jacobian at `state`.
    fn predict_and_jacobian(&self, state: &Vec<N>) -> (Vec<M>, Mat<M, N>);
}

/// Why an update could not be applied.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "defmt", derive(defmt::Format))]
pub enum UpdateError {
    /// The innovation covariance was not positive definite.
    ///
    /// In practice this means the filter has already diverged, or `R` contains
    /// a zero, or `P` has been corrupted. The state and covariance are left
    /// untouched so the caller can reinitialise from something trustworthy
    /// rather than continue from a filter that has lost the plot.
    NotPositiveDefinite,
}

/// An iterated EKF over `N` states and `M` measurements.
pub struct IteratedEkf<const N: usize, const M: usize> {
    x: Vec<N>,
    p: Mat<N, N>,
    q_diag: Vec<N>,
    r_diag: Vec<M>,
    iterations: u8,
    innovation: Vec<M>,
    /// Cholesky factor of the innovation covariance from the *first* pass of
    /// the last update, kept so [`IteratedEkf::nis`] costs a substitution
    /// rather than a factorisation.
    innovation_chol: Option<Mat<M, M>>,
}

impl<const N: usize, const M: usize> IteratedEkf<N, M> {
    /// Start from `state`, with diagonal initial covariance `initial_variance`
    /// and per-channel measurement variance `measurement_variance`.
    ///
    /// A generous initial covariance is not the harmless conservatism it looks
    /// like: it is the width over which the first update linearises, so on a
    /// nonlinear measurement function an initial variance far larger than the
    /// real uncertainty buys a worse first step, not a safer one.
    pub fn new(state: Vec<N>, initial_variance: Vec<N>, measurement_variance: Vec<M>) -> Self {
        let mut p = [[0.0f32; N]; N];
        for i in 0..N {
            p[i][i] = initial_variance[i];
        }
        Self {
            x: state,
            p,
            q_diag: [0.0; N],
            r_diag: measurement_variance,
            iterations: 2,
            innovation: [0.0; M],
            innovation_chol: None,
        }
    }

    /// Process noise as a power spectral density per state: the variance added
    /// over a step is this times `dt`, so the units are `[state]^2` per second.
    pub fn set_process_noise(&mut self, psd: Vec<N>) {
        self.q_diag = psd;
    }

    /// Per-channel measurement variance.
    pub fn set_measurement_variance(&mut self, variance: Vec<M>) {
        self.r_diag = variance;
    }

    /// Relinearisation passes per update, at least one.
    ///
    /// One pass is an ordinary EKF. Two is usually where the returns stop on a
    /// mildly nonlinear problem, and each pass costs a full evaluation of the
    /// measurement model, so this is the knob that trades accuracy during
    /// transients against the deadline.
    pub fn set_iterations(&mut self, iterations: u8) {
        self.iterations = iterations.max(1);
    }

    /// Overwrite the estimate, e.g. after a divergence or a re-zero.
    pub fn reset(&mut self, state: Vec<N>, initial_variance: Vec<N>) {
        self.x = state;
        self.p = [[0.0; N]; N];
        for i in 0..N {
            self.p[i][i] = initial_variance[i];
        }
        self.innovation = [0.0; M];
        self.innovation_chol = None;
    }

    pub fn state(&self) -> &Vec<N> {
        &self.x
    }

    pub fn covariance(&self) -> &Mat<N, N> {
        &self.p
    }

    /// Measurement minus prediction, from *before* the last update was
    /// applied.
    ///
    /// Deliberately the prior quantity. Recording it after the final iteration
    /// instead gives a post-fit residual, which is small by construction and
    /// makes [`nis`](Self::nis) meaningless as a consistency check.
    pub fn innovation(&self) -> &Vec<M> {
        &self.innovation
    }

    /// Normalised innovation squared from the last update.
    ///
    /// The one honest self-check a filter can make without ground truth: if
    /// `Q` and `R` are telling the truth this follows a chi-squared
    /// distribution with `M` degrees of freedom, so its mean should sit at `M`.
    /// Persistently above means the filter is claiming more certainty than it
    /// has.
    pub fn nis(&self) -> f32 {
        let Some(l) = self.innovation_chol.as_ref() else {
            return 0.0;
        };
        let solved = linalg::cholesky_solve_vec(l, &self.innovation);
        let mut acc = 0.0;
        for i in 0..M {
            acc += self.innovation[i] * solved[i];
        }
        acc
    }

    /// Random walk: the state is unchanged, only its uncertainty grows.
    pub fn predict(&mut self, dt: f32) {
        for i in 0..N {
            self.p[i][i] += self.q_diag[i] * dt;
        }
    }

    /// Linear transition `x <- transition * x`, with the same process noise.
    pub fn predict_linear(&mut self, transition: &Mat<N, N>, dt: f32) {
        self.x = linalg::matvec(transition, &self.x);
        let fp = linalg::matmul(transition, &self.p);
        self.p = linalg::matmul_transpose(&fp, transition);
        for i in 0..N {
            self.p[i][i] += self.q_diag[i] * dt;
        }
        linalg::symmetrise(&mut self.p);
    }

    /// Fold in one measurement, relinearising [`set_iterations`] times.
    ///
    /// On [`UpdateError`] the filter is left exactly as it was.
    ///
    /// [`set_iterations`]: Self::set_iterations
    ///
    /// `inline(always)` so that a caller which places itself in RAM takes this
    /// with it. On a part that executes from external flash through a small
    /// cache, that placement is worth more than every other optimisation here
    /// put together -- see `bench_forward`.
    #[inline(always)]
    pub fn update<Model>(&mut self, model: &Model, z: &Vec<M>) -> Result<(), UpdateError>
    where
        Model: MeasurementModel<N, M>,
    {
        let prior_x = self.x;
        let prior_p = self.p;

        let mut x = prior_x;
        let mut gain = [[0.0f32; M]; N];
        let mut jac = [[0.0f32; N]; M];
        let mut first_innovation = [0.0f32; M];
        let mut first_chol = None;

        for iteration in 0..self.iterations {
            let (predicted, j) = model.predict_and_jacobian(&x);
            jac = j;

            // S = J P J^T + R, and its factorisation, reused for both the gain
            // and the NIS.
            let jp = linalg::matmul(&jac, &prior_p); // (M, N)
            let mut s = linalg::matmul_transpose(&jp, &jac); // (M, M)
            linalg::add_diagonal(&mut s, &self.r_diag);
            linalg::symmetrise(&mut s);
            let Some(chol) = linalg::cholesky(&s) else {
                return Err(UpdateError::NotPositiveDefinite);
            };

            // K = P J^T S^-1, obtained as (S^-1 J P)^T since S is symmetric.
            gain = linalg::transpose(&linalg::cholesky_solve_mat(&chol, &jp));

            // The (prior_x - x) term is what makes this iterated rather than
            // merely repeated.
            let offset = {
                let mut d = [0.0f32; N];
                for i in 0..N {
                    d[i] = prior_x[i] - x[i];
                }
                d
            };
            let correction = linalg::matvec(&jac, &offset);

            let mut residual = [0.0f32; M];
            for i in 0..M {
                residual[i] = z[i] - predicted[i] - correction[i];
            }

            if iteration == 0 {
                for i in 0..M {
                    first_innovation[i] = z[i] - predicted[i];
                }
                first_chol = Some(chol);
            }

            let step = linalg::matvec(&gain, &residual);
            for i in 0..N {
                x[i] = prior_x[i] + step[i];
            }
        }

        // Joseph form: stays symmetric and positive definite under f32, which
        // the shorter (I - K H) P does not reliably do.
        let kh = linalg::matmul(&gain, &jac); // (N, N)
        let mut a = linalg::eye::<N>();
        for i in 0..N {
            for j in 0..N {
                a[i][j] -= kh[i][j];
            }
        }
        let ap = linalg::matmul(&a, &prior_p);
        let mut p = linalg::matmul_transpose(&ap, &a);

        // K R K^T, with R diagonal.
        for i in 0..N {
            for j in 0..N {
                let mut acc = 0.0;
                for k in 0..M {
                    acc += gain[i][k] * self.r_diag[k] * gain[j][k];
                }
                p[i][j] += acc;
            }
        }
        linalg::symmetrise(&mut p);

        self.x = x;
        self.p = p;
        self.innovation = first_innovation;
        self.innovation_chol = first_chol;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Position and velocity, position observed.
    struct ObservePosition;

    impl MeasurementModel<2, 1> for ObservePosition {
        fn predict_and_jacobian(&self, x: &[f32; 2]) -> ([f32; 1], [[f32; 2]; 1]) {
            ([x[0]], [[1.0, 0.0]])
        }
    }

    /// A genuinely nonlinear observation, to exercise the iteration.
    struct ObserveSquare;

    impl MeasurementModel<1, 1> for ObserveSquare {
        fn predict_and_jacobian(&self, x: &[f32; 1]) -> ([f32; 1], [[f32; 1]; 1]) {
            ([x[0] * x[0]], [[2.0 * x[0]]])
        }
    }

    #[test]
    fn linear_update_matches_the_textbook_gain() {
        // One state, one measurement: K = P / (P + R), analytically.
        let mut ekf = IteratedEkf::<2, 1>::new([0.0, 0.0], [1.0, 1.0], [1.0]);
        ekf.set_iterations(1);
        ekf.update(&ObservePosition, &[2.0]).unwrap();
        // P = R = 1 gives K = 0.5, so the estimate lands halfway.
        assert!((ekf.state()[0] - 1.0).abs() < 1e-5, "{:?}", ekf.state());
        assert!((ekf.covariance()[0][0] - 0.5).abs() < 1e-5);
    }

    #[test]
    fn repeated_updates_converge_on_the_measurement() {
        let mut ekf = IteratedEkf::<2, 1>::new([0.0, 0.0], [1.0, 1.0], [0.01]);
        for _ in 0..50 {
            ekf.predict(0.001);
            ekf.update(&ObservePosition, &[3.0]).unwrap();
        }
        assert!((ekf.state()[0] - 3.0).abs() < 0.01, "{:?}", ekf.state());
    }

    #[test]
    fn covariance_stays_symmetric_and_positive() {
        let mut ekf = IteratedEkf::<2, 1>::new([0.0, 0.0], [1.0, 1.0], [0.05]);
        ekf.set_process_noise([0.5, 0.5]);
        for k in 0..2000 {
            ekf.predict(0.0005);
            ekf.update(&ObservePosition, &[(k as f32) * 1e-3]).unwrap();
        }
        let p = ekf.covariance();
        assert!((p[0][1] - p[1][0]).abs() < 1e-9);
        assert!(p[0][0] > 0.0 && p[1][1] > 0.0);
        assert!(linalg::cholesky(p).is_some());
    }

    #[test]
    fn iteration_beats_a_single_linearisation_when_h_is_curved() {
        // Truth is x = 3, measured as x^2 = 9, started far enough away that
        // one linearisation undershoots.
        let single = {
            let mut f = IteratedEkf::<1, 1>::new([1.0], [4.0], [0.01]);
            f.set_iterations(1);
            f.update(&ObserveSquare, &[9.0]).unwrap();
            f.state()[0]
        };
        let iterated = {
            let mut f = IteratedEkf::<1, 1>::new([1.0], [4.0], [0.01]);
            f.set_iterations(5);
            f.update(&ObserveSquare, &[9.0]).unwrap();
            f.state()[0]
        };
        assert!(
            (iterated - 3.0).abs() < (single - 3.0).abs(),
            "iterated {iterated} should beat single {single}"
        );
        assert!((iterated - 3.0).abs() < 0.05, "{iterated}");
    }

    #[test]
    fn nis_sits_near_the_channel_count_when_r_is_honest() {
        // A deterministic pseudo-random measurement sequence with exactly the
        // variance R claims; the mean NIS should land near M = 1.
        let mut ekf = IteratedEkf::<2, 1>::new([0.0, 0.0], [0.01, 0.01], [1.0]);
        ekf.set_process_noise([0.0, 0.0]);
        let mut seed = 12345u32;
        let mut total = 0.0;
        let n = 4000;
        for _ in 0..n {
            seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
            // Two uniforms summed, scaled to unit variance: crude but its
            // variance is exact, which is all this test needs.
            let u1 = (seed >> 8) as f32 / 16777216.0 - 0.5;
            seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
            let u2 = (seed >> 8) as f32 / 16777216.0 - 0.5;
            let noise = (u1 + u2) * libm::sqrtf(6.0);
            ekf.predict(0.001);
            ekf.update(&ObservePosition, &[noise]).unwrap();
            total += ekf.nis();
        }
        let mean = total / n as f32;
        assert!(mean > 0.7 && mean < 1.4, "mean NIS {mean}");
    }

    #[test]
    fn a_singular_measurement_covariance_is_reported_not_ignored() {
        let mut ekf = IteratedEkf::<2, 1>::new([0.0, 0.0], [0.0, 0.0], [0.0]);
        assert_eq!(
            ekf.update(&ObservePosition, &[1.0]),
            Err(UpdateError::NotPositiveDefinite)
        );
        // And the state survived the failure untouched.
        assert_eq!(ekf.state(), &[0.0, 0.0]);
    }
}
