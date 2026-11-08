# Fixed-kernel target response

KernelLab v0.2 adds a live conditioning inspector to the original two manually supplied GP models. It does not change the original observations, fitted conditioning artifacts, kernel settings, optimization demo or reported results. Original length scales 0.8 and 5 remain separate; the second is the original confidently wrong counterexample, not a deliberately retrained weak baseline.

## A linear response is not causal influence

For fixed observed coordinates X, supplied mean m, RBF settings, iid noise variance and explicit training jitter:

```
A = K(X,X) + (noise_variance + jitter) I
k = K(X,x_query)
A w = k
mu(x) = m + sum_i w_i (y_i - m)
mu_edited(x) - mu(x) = sum_i w_i delta_i
latent_variance = k(x,x) - k^T A^-1 k
observation_variance = latent_variance + noise_variance
```

These weights describe sensitivity to observed target values, not features. They may be negative, exceed one or fail to sum to one. They are not probabilities, normalized attention, Shapley values or causal effects. Altering targets at fixed X/kernel/mean/noise/jitter changes the mean but not the conditional covariance. The band shifts center without changing its width. This statement does not survive arbitrary refitting of noise, mean or hyperparameters on the edited targets.

The native `response` API revalidates the complete model/query through original `predict`, solves the adjoint system using the original Cholesky factor, retains every signed weight and centered-target term, and compares the reconstructed original/edited means with direct original inference. The numerical witness records normalized solve residual, observed roundoff disagreement and original negative-variance tolerance. No inverse is materialized; no adaptive jitter or matrix repair is added. The browser independently uses an unpivoted unit-lower positive-pivot LDL^T factorization. This is an SPD educational solve, not SciPy's pivoted indefinite Bunch-Kaufman algorithm.

The normalized infinity residual divides `||Aw-k||inf` by `||A||inf*||w||inf + ||k||inf` (zero numerator/denominator gives zero); values above 1e-10 refuse. Mean disagreement beyond `1e-9*(1+sum(abs(centered terms))+abs(mean shift))` also refuses natively. Original Cholesky failures and material-negative variance failures still refuse. A small residual is not a forward-error or conditioning certificate. No tolerance was relaxed in the existing predictor. Browser/native agreement is a numerical check, not bitwise or exact-real equivalence.

## Live views

- Both original frozen models, one arbitrary query and one target edit. Every original point stays visible; the edited target is purple. The browser bounds queries to [-5,5] and delta to [-2,2] for the fixed plots. Native schema and resulting target bounds remain those of the original model.
- Original versus edited means with a selected latent-function or future-observation pointwise +/-2 model-SD band. Observation variance adds noise only, not training jitter. The fixed y axis covers both models, every plotted query, all allowed single-target edits and edited observations. The curve is a 101-point illustration, not continuous certification. A band is neither simultaneous coverage nor empirical calibration.
- All 15 signed target-response weights and terms. The bar axis includes zero and both models at the current query, so selecting an arm or target does not change that scale. Changing query may change the weight scale. All weights remain in the expanded table and export.
- Strict invalid inputs clear derived outputs and disable witness export. Clipboard denial, success, actual JSON Blob and complete embedded source model are exercised. Embedded public synthetic data is not a source-authenticity signature.

No edit recomputes the original benchmark scores. They are explicitly labeled original report-only metrics. The length-5 model's low variances, bad observation NLL/coverage and worse RMSE remain visible. This is one inspected development fixture, not an untouched external benchmark, comparative uncertainty ranking, calibrated-confidence claim or new test result.

## Reproduce

```
python -m kernellab.response_export
python -m kernellab response-report
python -m kernellab response --input examples/response.json
python -m unittest discover -s tests -v
node --test tests/*.cjs
python -m pip wheel . --no-deps -w /tmp/wheels
python -m pip install --target /tmp/installed /tmp/wheels/kernellab-0.2.0-py3-none-any.whl
bash examples/verify_wheel.sh /tmp/installed
```

Two arm files store original fit/LOO reports and all 15 direct unit-target-edit posterior curves. Eight bounded chunks retain all 202 query-response witnesses. The meta file retains all original regression/optimization summaries and report-only test/truth data; the exporter reproduces them. Archive reproduction requires exact keys, shapes and non-floating values, with 1e-12 absolute/relative tolerance for binary64 floats across Python/platform math builds. Small normalized residuals can differ in their final bits. A test rejects a 1e-8 perturbation, nonfinite values and changed structure. No original predictor tolerance or test was changed. No original test was removed. JSON CLI duplicate keys now refuse globally instead of silently overwrite. Every original fit/predict/suggest/demo/optimize-demo command remains.

## Independent checks

- 202 native/query witnesses including every weight and centered term.
- 3,030 one-target/unit-edit/query responses checked against direct edited-target conditioning, including unchanged latent and observation variances.
- 48 tiny high-precision 70-digit Decimal Gauss-Jordan solves check native Cholesky-derived weights for 1-6 observations, two length scales and four queries. Decimal starts from represented binary64 kernel coefficients; it is not exact-real RBF verification.
- Multiple simultaneous edits and finite differences at 30 query/target controls; signed-weight and nonbarycentric-sum controls; strict bounds/schema/nonmutation and CLI tests.
- Independent browser LDL inference checks all 202 native witnesses and all 3,030 direct refit responses. Actual Chrome renders every single-target state, matches displayed means, checks all 15 table terms, physical keyboard, band identity, invalid/stale refusal, clipboard/Blob and expanded 320/390/1280 layouts.
- The installed package imports all eight modules, executes all seven main commands and reproduces all native archives outside checkout. Original acquisition behaviors, zero-noise restrictions, equal-budget optimization traces and GP/LOO regressions remain within the unchanged original tests.

These are correctness and reproduction checks, not runtime, causal, calibration, external-performance or robustness benchmarks.

## Sources and differences

- [Rasmussen and Williams, GPML Ch2, equations 2.23-2.26 and Algorithm 2.1](https://gaussianprocess.org/gpml/chapters/RW2.pdf): posterior mean is linear in targets, covariance is target-independent at fixed kernel, Cholesky conditioning and observation-noise addition. Our explicit jitter is an additional training-only perturbation.
- [Gundersen, practical GP regression, 2019](https://www.gregorygundersen.com/blog/2019/09/12/practical-gp-regression/): explanatory triangular solves; no copied library code or inversion routine.
- [scikit-learn GP guide](https://scikit-learn.org/stable/modules/gaussian_process.html): regression scope, hyperparameters and conditioning diagonal. This page does not invoke its learned-kernel optimizer, normalization or implementation.
- [GPflow basic GPR](https://gpflow.github.io/GPflow/develop/notebooks/getting_started/basic_usage.html): separate latent and observation predictions; marginal versus joint covariance. We hold supplied parameters fixed instead of following its optimizer example.
- [SciPy LDL reference](https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.ldl.html): factorization terminology. Our positive-pivot scalar SPD implementation has no pivoting, 2x2 indefinite blocks, LAPACK or SciPy dependency and is not an implementation-equivalence claim.
- [Python math.fsum](https://docs.python.org/3/library/math.html): native accurate sums versus independent browser compensated sums, neither an exact-real guarantee.

Established GP theory, independent educational code, public synthetic data. MIT; attribution is not endorsement.
