"use strict";
const KernelJoint = (() => {
  const R =
    typeof require === "function" ? require("./inference.js") : KernelResponse;
  const sum = R.sum;
  const dot = (a, b) => sum(a.map((v, i) => v * b[i]));
  const number = (v, lo, hi) => {
    if (typeof v !== "number" || !Number.isFinite(v) || v < lo || v > hi)
      throw Error("Numeric value outside documented finite bounds");
    return v;
  };
  function parse(x) {
    if (
      !x ||
      Array.isArray(x) ||
      Object.keys(x).sort().join(",") !== "queries,training,weights"
    )
      throw Error("Input needs training, queries and weights only");
    if (
      !x.training ||
      Object.keys(x.training).sort().join(",") !== "kernel,rows,targets"
    )
      throw Error("Training schema");
    const model = { version: "kernellab_v1", ...x.training };
    if (
      !Array.isArray(model.rows) ||
      model.rows.length > 16 ||
      !Array.isArray(model.rows[0]) ||
      model.rows[0].length > 2
    )
      throw Error("At most16 training rows and2 features");
    const P = R.prepare(model);
    if (
      !Array.isArray(x.queries) ||
      x.queries.length < 1 ||
      x.queries.length > 8
    )
      throw Error("Supply1-8queries");
    for (const row of x.queries) {
      if (!Array.isArray(row) || row.length !== P.d) throw Error("Query width");
      row.forEach((v) => number(v, -1e6, 1e6));
    }
    if (
      !Array.isArray(x.weights) ||
      x.weights.length < 1 ||
      x.weights.length > 4
    )
      throw Error("Supply1-4weight rows");
    for (const row of x.weights) {
      if (!Array.isArray(row) || row.length !== x.queries.length)
        throw Error("One weight per query");
      row.forEach((v) => number(v, -10, 10));
    }
    return { input: structuredClone(x), P };
  }
  function audit(x) {
    const { input, P } = parse(x),
      { training, queries, weights } = input,
      p = training.kernel;
    const n = training.rows.length,
      m = queries.length,
      q = weights.length;
    const kernel = (a, b) => R.kernel(a, b, p);
    const K = queries.map((a) => queries.map((b) => kernel(a, b)));
    const cross = queries.map((a) => training.rows.map((b) => kernel(a, b)));
    // Independent unit-lower LDL^T solves, not native Cholesky factors.
    const solvedQuery = cross.map((k) => R.solve(P.factor, k));
    const alpha = R.solve(
      P.factor,
      training.targets.map((y) => y - p.prior_mean),
    );
    const mus = cross.map((k) => p.prior_mean + dot(k, alpha));
    const tol = Math.max(1e-12, p.signal_variance * 1e-10);
    let clampedQuery = 0;
    const S = queries.map(() => Array(m).fill(0));
    for (let i = 0; i < m; i++)
      for (let j = 0; j <= i; j++) {
        let v = K[i][j] - dot(cross[i], solvedQuery[j]);
        if (i === j) {
          if (v < -tol) throw Error("Material negative query variance");
          if (v < 0) clampedQuery++;
          v = Math.max(0, v);
        }
        S[i][j] = S[j][i] = v;
      }
    const obs = S.map((row, i) =>
      row.map((v, j) => v + (i === j ? p.noise_variance : 0)),
    );
    const posterior = {
      means: mus,
      latent_variances: S.map((r, i) => r[i]),
      observation_variances: obs.map((r, i) => r[i]),
      diagnostics: {
        clamped_latent_diagonals: clampedQuery,
        negative_variance_tolerance: tol,
      },
      scope:
        "Observation variances add noise_variance, not training-only jitter. Pointwise Gaussian model distributions, not distribution-free confidence/coverage guarantees. Input scaling and hyperparameters are caller-owned.",
      latent_covariance: S,
      observation_covariance: obs,
    };
    const unique = [],
      groups = [];
    queries.forEach((row, i) => {
      let k = unique.findIndex((u) => u.every((v, j) => v === row[j]));
      if (k < 0) {
        k = unique.length;
        unique.push(row);
        groups.push([]);
      }
      groups[k].push(i);
    });
    const grouped = weights.map((w) =>
      groups.map((g) => sum(g.map((i) => w[i]))),
    );
    const Ku = unique.map((a) => unique.map((b) => kernel(a, b)));
    const Xu = unique.map((a) => training.rows.map((b) => kernel(a, b)));
    const projected = grouped.map((w) =>
      Array.from({ length: n }, (_, j) => sum(w.map((v, i) => v * Xu[i][j]))),
    );
    const solved = projected.map((k) => R.solve(P.factor, k));
    const prior = grouped.map((a) =>
      grouped.map((b) =>
        sum(a.flatMap((v, i) => b.map((w, j) => v * Ku[i][j] * w))),
      ),
    );
    const reduction = projected.map((a) => solved.map((b) => dot(a, b)));
    const raw = prior.map((r, a) => r.map((v, b) => v - reduction[a][b]));
    const contraction = weights.map((a) =>
      weights.map((b) =>
        sum(a.flatMap((v, i) => b.map((w, j) => v * S[i][j] * w))),
      ),
    );
    const gates = weights.map((a, i) =>
      weights.map(
        (b, j) =>
          1e-10 *
          (1 +
            Math.abs(prior[i][j]) +
            Math.abs(reduction[i][j]) +
            sum(
              a.flatMap((v, k) => b.map((w, l) => Math.abs(v * S[k][l] * w))),
            )),
      ),
    );
    const latent = raw.map((r) => r.slice()),
      clamped = [];
    for (let a = 0; a < q; a++) {
      for (let b = 0; b < q; b++)
        if (Math.abs(raw[a][b] - contraction[a][b]) > gates[a][b])
          throw Error("Contraction numerical gate failed");
      if (raw[a][a] < -gates[a][a])
        throw Error("Material negative linear variance");
      if (raw[a][a] < 0) {
        latent[a][a] = 0;
        clamped.push(a);
      }
    }
    const noise = weights.map((a) =>
      weights.map((b) => p.noise_variance * dot(a, b)),
    );
    const observation = latent.map((r, a) => r.map((v, b) => v + noise[a][b]));
    const diagonal = weights.map((a) =>
      weights.map((b) => sum(a.map((v, i) => v * S[i][i] * b[i]))),
    );
    const means = weights.map((w) => dot(w, mus));
    const contrasts = weights.map((w, a) => {
      const cells = w.map((v, i) => w.map((z, j) => v * S[i][j] * z)),
        lv = latent[a][a],
        ov = observation[a][a],
        dv = diagonal[a][a],
        ls = Math.sqrt(lv),
        os = Math.sqrt(ov);
      return {
        index: a,
        weights: w,
        grouped_weights: grouped[a],
        mean: means[a],
        latent_variance: lv,
        observation_variance: ov,
        diagonal_only_latent_variance: dv,
        diagonal_only_observation_variance: dv + noise[a][a],
        raw_latent_variance: raw[a][a],
        prior_variance: prior[a][a],
        conditioning_reduction: reduction[a][a],
        future_noise_variance: noise[a][a],
        cell_terms: cells,
        diagonal_sum: sum(cells.map((r, i) => r[i])),
        off_diagonal_sum: sum(
          cells.flatMap((r, i) => r.filter((_, j) => j !== i)),
        ),
        contraction: contraction[a][a],
        signed_contraction_disagreement: raw[a][a] - contraction[a][a],
        numerical_gate: gates[a][a],
        latent_sd: ls,
        observation_sd: os,
        latent_two_sd_interval: [means[a] - 2 * ls, means[a] + 2 * ls],
        observation_two_sd_interval: [means[a] - 2 * os, means[a] + 2 * os],
      };
    });
    const result = {
      version: "kernellab_joint_v1",
      input,
      query_posterior: posterior,
      query_prior_covariance: K,
      query_training_cross: cross,
      identical_query_groups: groups,
      grouped_queries: unique,
      grouped_weights: grouped,
      projected_training_cross: projected,
      projected_prior_covariance: prior,
      conditioning_reduction: reduction,
      raw_latent_covariance: raw,
      contracted_latent_covariance: contraction,
      latent_covariance: latent,
      observation_covariance: observation,
      future_noise_covariance: noise,
      diagonal_only_latent_covariance: diagonal,
      means,
      contrasts,
      clamped_contrast_diagonals: clamped,
      scope:
        "Fixed supplied RBF GP, explicit training jitter and iid future observation noise. Linear functionals of one joint posterior, not independent query marginals. Repeated latent coordinates coalesce; repeated future observations retain independent noise. Binary64 numerical diagnostics, not PSD, calibration, simultaneous coverage or optimum certificates.",
    };
    function finite(v) {
      if (typeof v === "number" && !Number.isFinite(v))
        throw Error("Nonfinite joint report");
      if (v && typeof v === "object") Object.values(v).forEach(finite);
    }
    finite(result);
    return result;
  }
  return { audit, parse };
})();
if (typeof module !== "undefined") module.exports = KernelJoint;
