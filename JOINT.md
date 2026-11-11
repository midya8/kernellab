# Full joint GP uncertainty for finite linear contrasts

Two pointwise means and variances do not identify the uncertainty of a sum or difference. The cross covariance matters. This module returns the full query posterior, signed linear functionals of that one joint posterior, every weighted pair contribution and a diagonal-only comparison.

The earlier GP solver, fixed-kernel target-response inspector and finite-candidate acquisition code are unchanged. No hyperparameter fitting, sampling or simultaneous-band method is added.

## Input and reproduction

```json
{
  "training": {
    "rows": [[0]],
    "targets": [0],
    "kernel": {
      "length_scale": 1,
      "signal_variance": 1,
      "noise_variance": 0,
      "jitter": 0,
      "prior_mean": 0
    }
  },
  "queries": [[-1], [1]],
  "weights": [[1, -1], [0.5, 0.5]]
}
```

Exactly `training`, `queries` and `weights` are required. `training` uses the existing strict rows/targets/kernel schema; all five kernel settings stay explicit. Each weight row is one linear functional of all query occurrences. No coefficient normalization, query deduplication in the public input, target centering, fitted mean or kernel selection happens silently.

```sh
python -m kernellab.joint input.json > report.json
# Or provide JSON on stdin:
python -m kernellab.joint - < input.json > report.json
PYTHONPATH=. python examples/joint_controls.py
python -m unittest discover -s tests -v
node --test tests/*.cjs
python -m pip wheel . --no-deps -w /tmp/wheels
python -m pip install --target /tmp/installed /tmp/wheels/*.whl
bash examples/verify_wheel.sh /tmp/installed
```

Open `docs/joint.html` locally or use the deployed workbench. The browser accepts the same complete input, independently computes the report and can copy or download it. To replay a downloaded report, extract its `input` for the native command. Replay is not an independent archive-verification certificate. Runtime Python uses only the standard library; Node22+ and actual Chrome are needed for the web tests.

## The joint law

With fixed training rows X, targets y, query rows Q and supplied parameters, the original inference contract is:

```text
A = K(X,X) + (noise_variance + jitter) I
A = L Lᵀ
alpha = solve(A, y - prior_mean)
mu = prior_mean + K(Q,X) alpha
Sigma = K(Q,Q) - K(Q,X) solve(A, K(X,Q))
```

The native code uses the original Cholesky solver, not a dense inverse. The browser uses separate unit-lower LDLᵀ solves. Neither method factors the query covariance, so repeated queries and singular joint query laws are allowed. Singular training covariance still refuses when a positive pivot is unavailable; explicit caller-supplied noise or jitter can change that model.

For F, the vector of latent query values, a matrix W of supplied signed weights defines Z=W F:

```text
E[Z] = W mu
Cov(Z) = W Sigma Wᵀ
Var(Z_a) = sum_i sum_j W_ai Sigma_ij W_aj
```

Each native and browser report keeps every cell `W_ai Sigma_ij W_aj`, the diagonal sum, the off-diagonal sum and their complete direct contraction. Both halves of the off-diagonal matrix count. A negative covariance can increase a difference variance because its two coefficients have opposite signs.

The diagonal-only comparison is `W diag(diag(Sigma)) Wᵀ`. It preserves every query marginal mean and variance but changes the joint law to independent query variables. It is not a second fitted model or a valid way to delete dependence from the original model. Depending on W and the sign of covariance, it can overstate or understate a contrast's variance.

## Same marginals, different difference uncertainty

The main example conditions a zero-mean unit-variance length-one RBF on one noiseless observation at zero. Let `r=exp(-1)`. At either query -1 or +1:

```text
mean = 0
variance = 1-r = 0.6321205588
Cov(f(-1), f(+1)) = exp(-2)-exp(-1) = -0.232544158
Cov(f(-1), f(-1)) = 1-r = +0.6321205588
```

Therefore two query lists can have exactly the same marginal means and variances yet different joint uncertainty:

| Query list | Functional | Full latent variance | Diagonal-only variance |
|---|---|---:|---:|
| [-1,+1] | first minus second | 1.729329434 | 1.264241118 |
| [-1,-1] | first minus second | 0 | 1.264241118 |
| [-1,+1] | half of each, summed | 0.199788200 | 0.316060279 |
| [-1,-1] | half of each, summed | 0.632120559 | 0.316060279 |

A repeated latent query is the same random variable, not a new independent draw. It cancels in its own difference. For two nearby queries `[0.8,1]`, the difference variance is `0.025294097` while diagonal-only gives `1.104828135`. The inspected controls demonstrate the covariance identity; they do not measure prediction quality or establish calibrated uncertainty.

## Future observations and shared noise

Future observation occurrences are `Y_i=F_i+epsilon_i`, with iid zero-mean noises independent of the latent process and variance `noise_variance`:

```text
Cov(Y) = Sigma + noise_variance I
Cov(W Y) = W Sigma Wᵀ + noise_variance W Wᵀ
```

Two occurrences at the same coordinate share the same latent value but have distinct future noise terms. At noise variance `.2`, their difference has latent variance zero and observation variance `.4`. Their average adds noise variance `.1`.

Two linear functionals can reuse one query occurrence. Their future noise is then correlated, even though the original per-occurrence noise is iid. For weight rows `[1,0]`, `[1,-1]` and `[.5,.5]`, the added noise covariance is:

```text
[[.2, .2, .1],
 [.2, .4,  0],
 [.1,  0, .1]]
```

This is not fresh independent noise per contrast. Training `jitter` belongs only in A. It changes the conditioning and marginal likelihood but is not added to future query covariance or contrast noise.

## Grouped latent computation and numerical gates

The input query list remains intact in all displayed query matrices and cell tables. For the separate projected prior-minus-conditioning computation only, exactly equal binary64 query coordinates are grouped. Weights for the same latent value are summed before conditioning:

```text
c = grouped coefficients over unique query coordinates U
B = c K(U,X)
projected_prior = c K(U,U) cᵀ
conditioning_reduction = (L⁻¹ Bᵀ)ᵀ (L⁻¹ Bᵀ)
raw_projected_covariance = projected_prior - conditioning_reduction
```

This lets a repeated-value latent difference cancel algebraically before floating subtraction. It does not merge independent observation-noise occurrences. No coordinate-distance tolerance or near-duplicate grouping is used.

The result is compared to the direct full-query contraction `W Sigma Wᵀ`. For each projected cell (a,b), the comparison gate is:

```text
1e-10 * (1 + abs(projected_prior_ab)
           + abs(conditioning_reduction_ab)
           + sum_ij abs(W_ai Sigma_ij W_bj))
```

A disagreement larger than that gate rejects. A projected diagonal below minus its gate rejects. A permitted small negative diagonal clamps to zero and is counted by index. Raw signed subtractions, both covariance computations, their disagreement and the gate remain in the report. The original query-diagonal clamp and diagnostic stay separate in `query_posterior`.

These are binary64 comparison policies, not interval arithmetic, exact PSD checks, relative-error guarantees or forward-error bounds. In particular, the absolute floor can dominate tiny variances. Off-diagonal entries are not repaired, query or projected matrices are not silently projected to PSD and no eigenvalue/rank/density claim is made. Badly conditioned cases can fail inside the schema bounds. Coalescing exact duplicates does not fix arbitrary cancellation between nearby but unequal queries.

## What the intervals mean

Each selected scalar Gaussian functional has mean and model SD, and the JSON includes its mean±2SD interval. The diagram puts the full and diagonal-only scalar intervals on a shared linear axis. It does not draw sampled functions or a joint ellipsoid.

A scalar interval is not a simultaneous band for all query values, all contrasts or a continuous function. No joint rectangle probability, multiple-comparison correction, posterior sampling, calibration test or coverage guarantee is computed. Degenerate zero variance gives a deterministic linear value under the supplied model, not a measurement of the real function.

## Controls and checks

Twelve synthetic controls cover opposite-side negative covariance, repeated latent/noisy queries, nearby positive covariance, a noiseless training query, zero weights, shared observation noise, explicit training jitter, two-feature signed weights, underflowed training-query covariance, tiny signal scale and a large mean/target. None are selected by a measured accuracy result.

- The current suite has 58 Python tests, retaining the earlier43. Fifteen new tests include an independent Decimal80 implementation with Decimal exponential kernels and pivoted Gauss-Jordan inversion, with no native factorization, grouping, covariance or parser code. It checks all12controls and72fixed-seed generated systems, including every projected mean/covariance and the saved query covariances for the controls.
- Nine Node/actual-Chrome tests retain the earlier four. Full native reports match independent LDL browser reports on all12controls plus100generated systems using `2e-9*(1+abs(reference))` numeric tolerance. Structures, strings, groups and indices match exactly; whole-report binary floating identity is not expected.
- Actual Chrome checks24contrasts in both modes,48mode selections,640displayed covariance/pair/projected cells,36responsive layouts, trusted contrast selection and JSON edits, invalid-state clearing, clipboard success/failure and two completed JSON downloads replayed natively. Displayed numbers use `2e-8*(1+abs(reference))` tolerance. Desktop/mobile pixels are inspected separately from numeric assertions.
- Exact repeated-value cancellation, scalar sign/scaling, query/training permutation, unchanged covariance under target edits, explicit noise cross covariance, every-cell contraction, the full16/8/2/4resource budget, strict shapes/bounds, no mutation and singular-training refusal are checked.
- Installed replay imports nine modules, reproduces earlier seven main CLI commands, all202response witnesses/3030target-response predictions, original regression/optimization results and the new12controls, new CLI and12invalid-weight refusals outside the checkout. Original charts still reproduce without changes.

## Bounds and costs

- Training1-16rows, query1-8rows, matching1-2features. One to four weight rows; each has one coefficient per original query occurrence, finite in `[-10,10]`. Zero rows of weights and repeated query coordinates are valid.
- Existing training bounds remain: coordinates, targets and supplied mean in `[-1e6,1e6]`; length scale `[1e-6,1e6]`, signal variance `[1e-12,1e6]`, noise variance `[0,1e6]`, explicit jitter `[0,.01]`. Booleans, nonfinite values, unknown keys and wrong shapes reject.
- CLI/browser JSON is capped at64KiB; duplicate keys reject. Browser nesting is capped at64. The API accepts structured Python values; the file boundary is where duplicate JSON keys and byte length are checked. No partial JSON on a CLI error.
- Dense original conditioning costs O(n³); queries add O(mn²+m²n), and q linear forms add O(qn²+q²(n+m²)). Full reports retain query/cross/projected matrices and q complete m×m pair tables, bounded storage O(n²+mn+m²+qn+q²+qm²). Native deliberately refactors for its independent projected computation rather than advertising a speed optimization.

No large-scale/runtime promise, learned kernel, inference server, multi-output GP, derivative observations, sampling, noisy/batch acquisition, training-parameter sensitivity, probability-of-all-events estimate, PSD certificate or new theorem.

## Sources and convention

- Rasmussen and Williams, [GPML Chapter2](https://gaussianprocess.org/gpml/chapters/RW2.pdf), equations2.23-2.24 and discussion of joint predictions/noisy targets: full posterior covariance, including possible negative posterior cross covariance.
- [GPML AppendixA](https://gaussianprocess.org/gpml/chapters/RWA.pdf): Gaussian identities and conditioning.
- [Stanford STATS305C Normal Theory](https://web.stanford.edu/class/stats305c/lectures/Normal_Theory.html) and [Duke multivariate normal theory](https://www2.stat.duke.edu/courses/Fall16/sta721/slides/NormalTheory/multnorm.pdf): linear combinations and covariance terms.
- [SciPy multivariate_normal documentation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.multivariate_normal.html): singular Gaussian laws differ from nonsingular-density requirements. This module computes no density and does not claim SciPy's implementation or its validation.
- [scikit-learn Gaussian-process documentation](https://scikit-learn.org/stable/modules/gaussian_process.html): covariance versus marginal-SD outputs; its alpha/WhiteKernel conventions are not silently substituted for this repository's explicit jitter/future-noise contract.

Independent educational implementation of standard conditioning and linear covariance formulas. No GP library code copied, performance attribution, source-author endorsement or claim that numerical checks validate the model assumptions.
