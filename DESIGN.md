# KernelLab design

## Exact conditioning, explicit assumptions

An isotropic squared-exponential covariance, fixed supplied constant mean and
iid Gaussian observation noise define the model. Training inputs are rows,
not columns. No feature/target normalization, empirical mean estimation or
hyperparameter fitting. All five parameters are explicit: length_scale,
signal_variance, noise_variance, jitter and prior_mean. Signal/noise parameters
are variances, not standard deviations.

Condition using Cholesky and triangular solves. Production code never inverts
a dense matrix. A separate Gauss-Jordan inverse exists only in tests as an
independent tiny reference. An analytic one-point posterior checks mean,
latent variance, LOO and log marginal likelihood. Symmetry, row order,
multidimensional inputs and noiseless interpolation have direct tests.

## Numerical and artifact boundary

The training covariance is K+(noise+jitter)I. Nonpositive pivots fail; no
adaptive jitter, PSD projection, parameter normalization or silent retry.
Prediction latent covariance subtracts the projected training cross-covariance;
observation covariance adds only specified observation noise on the diagonal.
A query repeated twice shares the same latent function value but has independent
observation noise, so noise is not added off diagonal.

Tiny negative posterior diagonals within max(1e-12,signal_variance*1e-10)
are clamped and counted. Material negatives fail. Off-diagonals are not repaired.
Bounds keep the example small and arithmetic manageable; they do not guarantee
conditioning. Everything uses binary64; large integers can lose distinctions.

Artifacts store version kernellab_v1, raw training rows/targets and parameters.
Prediction validates and refactors, rather than trusting serialized alpha or
factors. An artifact contains training data and must be handled accordingly.
This design favors a small transparent JSON format over fast repeated inference.
No dynamic code or model download.

## Marginal likelihood and LOO

Marginal likelihood uses Cholesky log diagonals, not a determinant product.
It is a training-model density, not an external performance score. Its value
includes the explicit training jitter. No optimization uses it here.

For d_i=(A^-1)_ii, compute each d_i as the squared norm of L^-1 e_i.
The code forms only the diagonal, not a dense inverse. Fixed-parameter LOO mean
is y_i-alpha_i/d_i and effective observation variance is 1/d_i. Because A
includes noise+jitter, this conditional variance includes both. Compare it to
explicit refit observation variance plus jitter, not just latent uncertainty.
The supplied constant mean stays fixed; estimating a mean on all targets first
would violate the target-exclusion claim. Own-target perturbation and explicit
refit tests check that exclusion.

LOO here is a row-level conditional diagnostic with fixed assumptions. It does
not retune parameters, respect groups/time, cure leakage or establish coverage.

## Development fixture and adverse arm

Seed 19: 15 evenly spaced training inputs in [-3,3], truth sin(1.5*x)+.15*x,
Gaussian noise SD .12. The same RNG produces 101 noisy test observations at
inputs [-5,5]. Both arms use signal variance1/noise variance.0144/jitter1e-10/
prior mean0. Fixed length scales .8 and5 were manually chosen, without a grid
search. No test field is accepted by fit; the fixture was inspected in
development and is not an untouched benchmark.

Length .8: observation RMSE .6242366399, mean NLL .1040604253, observed
pointwise 2-SD coverage100/101. Length5: RMSE1.0038088074, NLL12.1309060441,
coverage22/101. Unconditioned mean0/observation variance1.0144 gives RMSE.7879412723,
NLL1.2321062355. The wide-kernel adverse arm is kept visible, not silently tuned
away. The fixture is constructed to need shorter-scale variation; no universal
ranking or causal claim follows.

Length .8 inner/outer mean latent variance .00866/.58142, compared with
.00269/.02796 for length5. Smaller model variance does not mean a more accurate
model. Training LML -7.418945/-210.975199 and LOO mean NLL .069448/15.090203
are reported without selection. Model bands are pointwise observation bands,
not simultaneous or distribution-free intervals. The chart shares axes and
plots means, actual latent truth, observations and +/-2 observation SD.

## Cost and omissions

n,m<=128, width<=8. Dense factor/LOO O(n^3), training storage O(n^2).
Prediction refactors, then O(m*n^2); optional full covariance O(m^2*n) work
and O(m^2) output storage. No sparse/variational model, learned kernel, ARD,
optimizer, sampling, acquisition function or scalable production claim.
The CLI emits either a complete JSON result or an error, never a partial report.
The offline demo checks a singular-model rejection and explicit-jitter recovery,
artifact transfer and covariance semantics. CI regenerates the chart.

## Sources

- https://gaussianprocess.org/gpml/chapters/RW2.pdf
- https://gaussianprocess.org/gpml/chapters/RW5.pdf
- https://www.gregorygundersen.com/blog/2019/09/12/practical-gp-regression/

Standard Gaussian-process formulas, independently implemented. No new theory.
