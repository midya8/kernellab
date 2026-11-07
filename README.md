# KernelLab

Exact small-data Gaussian-process regression from scratch: fixed RBF kernel,
Cholesky conditioning, latent versus observation uncertainty, log marginal
likelihood and row leave-one-out diagnostics. Dependency-free Python 3.10+,
25 tests, train/predict JSON CLI and an offline, reproducible visual demo.
Standard GP theory, not a new method or a scalable GP package.

![Two fixed GP length scales, same observations and shared axes](assets/posterior.svg)

## A useful model and a confidently wrong model

```sh
python -m kernellab demo
bash examples/demo.sh
PYTHONPATH=. python examples/render_demo.py
```

The fixture is `sin(1.5*x) + 0.15*x` with independent Gaussian observation noise
SD 0.12, seed 19. Fifteen training points cover [-3,3]; 101 report-only noisy
test observations cover [-5,5]. Both arms use signal variance 1, noise variance
0.0144, explicit jitter 1e-10 and prior mean 0. Only the fixed length scale differs.

| Fixed model | Observation RMSE | Mean observation NLL | Observed pointwise +/-2 SD coverage |
|---|---:|---:|---:|
| Unconditioned prior, mean 0 / variance 1.0144 | 0.7879 | 1.2321 | Not reported |
| RBF length scale 0.8 | 0.6242 | 0.1041 | 100/101 (0.9901) |
| RBF length scale 5 | 1.0038 | 12.1309 | 22/101 (0.2178) |

For length 0.8, average latent variance is 0.00866 inside the training domain,
0.58142 outside. Length 5 has smaller corresponding variances, 0.00269/0.02796,
but worse predictions on this fixture. Uncertainty can increase away from data
and still be misleading when the kernel, mean or noise assumptions are wrong.
The plot shows **observation** bands; latent function variance is smaller.

Training log marginal likelihood is -7.4189 versus -210.9752; fixed-parameter
LOO mean negative log density is 0.0694 versus 15.0902. These diagnostics are
reported, not optimized. Two manually chosen length scales on development data,
no parameter search or test-based selection routine. The fixture was inspected
during development; it is not an untouched benchmark. Coverage across correlated
function locations from this one fixture is not calibration evidence, a
simultaneous band, a confidence guarantee, or a universal model ranking.

## Condition, audit, serialize, predict

```python
from kernellab import fit, predict

report = fit({
    "rows": [[0], [1], [2]],
    "targets": [1, -0.5, 0.7],
    "kernel": {
        "length_scale": 0.8,
        "signal_variance": 1.3,
        "noise_variance": 0.2,
        "jitter": 1e-9,
        "prior_mean": 0.3,
    },
})
posterior = predict(report["model"], [[0.3], [1.4]], full_covariance=True)
# means, latent_variances, observation_variances;
# optional latent_covariance and observation_covariance.
```

`fit` accepts exactly rows/targets/kernel, with all five kernel settings explicit.
No target centering, feature scaling, mean fitting, optimization or default
noise inference happens silently. Model artifacts use `kernellab_v1` and store
validated training rows, targets and settings, not executable code or trusted
precomputed factors. `predict` revalidates and refactors the training covariance;
JSON round trips preserve tested predictions. Training data is present in the
artifact: do not publish an artifact containing sensitive data.

```sh
python -m kernellab fit --input examples/train.json
# Output contains model, log_marginal_likelihood, loo, diagnostics, scope.
python -m kernellab predict --input prediction-request.json
# {"model": <report.model>, "rows": [[0.3]], "full_covariance": true}
```

Both commands support `--input -`. Invalid JSON, schema errors and failed
factorizations exit 2 with stderr and **no partial JSON**. The offline demo
checks artifact transfer, latent/observation variance separation, singular
noiseless rejection and explicit-jitter recovery. CI also regenerates the SVG
and requires it to match the committed chart.

## Numerical contract

For `k(a,b)=signal_variance*exp(-0.5*sum(((a-b)/length_scale)^2))`:

- Factor `A = K + (noise_variance + jitter) I = L L^T`.
- Solve `alpha = L^-T L^-1 (y-prior_mean)` with triangular solves, no dense inverse.
- Posterior mean: `prior_mean + k_*^T alpha`.
- Latent covariance: `K_** - (L^-1 K_*)^T (L^-1 K_*)`.
- Observation covariance adds **noise_variance only** to the query diagonal.
- Log marginal likelihood: `-0.5*(y-m)^T alpha - sum(log(diag(L))) - n/2*log(2*pi)`.

Jitter is an explicit training-covariance perturbation. It changes predictions
and marginal likelihood; it is not automatically added to future observation
noise. No hidden adaptive jitter or matrix repair. A nonpositive Cholesky pivot
fails with a request to reconsider noise or caller-supplied jitter.

Negative posterior diagonals within `max(1e-12, signal_variance*1e-10)` are
clamped to zero and counted; larger negatives fail. This is a stated numerical
tolerance, not an assertion of exact arithmetic. Off-diagonal covariance is not
silently projected to a positive-semidefinite matrix. Very ill-conditioned
inputs can fail even inside the schema bounds.

LOO uses `mu_i = y_i - alpha_i / d_i` and `variance_i = 1/d_i`, where
`d_i = ||L^-1 e_i||^2` is the diagonal of `A^-1`, computed without materializing
the dense inverse. These are **effective observation variances including
training jitter**. This differs deliberately from future prediction variances.
The fixed supplied mean and hyperparameters are held unchanged across folds.
Tests compare every LOO value with an explicit leave-one-out refit and show
that an omitted row's target does not determine its own predicted mean.

## Scope and limits

1-128 training rows and query rows, 1-8 numeric features; coordinates, targets
and supplied mean in [-1e6,1e6]. Length scale in [1e-6,1e6], signal variance in
[1e-12,1e6], noise variance in [0,1e6], jitter in [0,0.01]. Booleans, NaN,
Infinity, mismatched shapes and unknown fields are rejected. Inputs become
binary64 floats, so large integer distinctions may be lost.

Dense factorization and the LOO audit cost O(n^3), storage O(n^2). Prediction
refactors the artifact, then uses O(m*n^2) triangular work; optional joint
covariance adds O(m^2*n) work and O(m^2) storage. Small-data educational scope.
No sparse approximations, learned kernels, ARD, gradients, optimizer, weights,
missing/categorical values, multi-output, sampling, acquisition function,
causal claim, automatic deployment or production-performance promise.

The Gaussian iid noise/RBF/supplied-mean assumptions are substantive. Row LOO
does not protect against group/time dependence, feature leakage or deployment
shift. A marginal-likelihood score is not a test score; optimizing it would
require a separate selection protocol. Model uncertainty is not a substitute
for empirical validation. No paid service, model download or network call.

## Verification

```sh
python -m unittest discover -s tests -v
```

25 tests cover an analytic one-point posterior and marginal likelihood,
Cholesky reconstruction/linear solves, an independent tiny Gauss-Jordan inverse
reference, explicit LOO refits, own-target exclusion, duplicate-point failure
and explicit-jitter recovery, noiseless interpolation, far-field prior return,
variance independence from targets, covariance symmetry/diagonal/quadratic
checks, counted tiny-negative clamping and material-negative rejection, row permutation, multidimensional inputs, bounds/schema validation,
no input mutation, artifact round trips, CLI errors and deterministic metrics.
CI runs Python 3.10-3.13 and the offline demo.

## Sources and credit

- [Rasmussen & Williams (2006), GPML Chapter 2, Algorithm 2.1](https://gaussianprocess.org/gpml/chapters/RW2.pdf): Cholesky conditioning, predictive uncertainty and marginal likelihood.
- [GPML Chapter 5, equation 5.12](https://gaussianprocess.org/gpml/chapters/RW5.pdf): fixed-parameter leave-one-out conditional means and variances.
- [Gregory Gundersen, practical GP regression](https://www.gregorygundersen.com/blog/2019/09/12/practical-gp-regression/): explanatory Cholesky treatment.

Independent implementation of standard formulas. No source code copied from a
GP library, synthetic data clearly labeled and no external-performance attribution.
MIT license.
