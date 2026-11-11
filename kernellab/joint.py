"""Full joint GP uncertainty for supplied finite linear contrasts.

Fixed parameters and iid future noise. No sampling, fitting or joint confidence
claim. The earlier inference and target-response surfaces remain unchanged.
"""

import json
import math
from .core import _training, _factor, _forward, _number, _rows, predict, rbf


def parse(inp):
    if not isinstance(inp, dict) or set(inp) != {"training", "queries", "weights"}:
        raise ValueError("input requires training, queries and weights only")
    train, targets, p = _training(inp["training"])
    queries = _rows(inp["queries"], len(train[0]))
    if len(train) > 16 or len(queries) > 8 or len(train[0]) > 2:
        raise ValueError(
            "joint audit requires at most 16 training rows, 8 queries and 2 features"
        )
    weights = inp["weights"]
    if not isinstance(weights, list) or not 1 <= len(weights) <= 4:
        raise ValueError("supply 1-4 linear weight rows")
    ws = []
    for row in weights:
        if not isinstance(row, list) or len(row) != len(queries):
            raise ValueError("one weight per query is required in each row")
        ws.append([_number(x, -10, 10, "linear weight") for x in row])
    return {
        "training": {"rows": train, "targets": targets, "kernel": p},
        "queries": queries,
        "weights": ws,
    }


def audit(inp):
    inp = parse(inp)
    train, targets, p = _training(inp["training"])
    queries, weights = inp["queries"], inp["weights"]
    model = dict(version="kernellab_v1", **inp["training"])
    posterior = predict(model, queries, full_covariance=True)
    lower, _, _ = _factor(train, targets, p)
    n, m, q = len(train), len(queries), len(weights)
    S = posterior["latent_covariance"]
    K = [
        [rbf(a, b, p["length_scale"], p["signal_variance"]) for b in queries]
        for a in queries
    ]
    cross = [
        [rbf(a, b, p["length_scale"], p["signal_variance"]) for b in train]
        for a in queries
    ]
    # Coalesce identical latent coordinates BEFORE prior/correction subtraction.
    # A repeated latent value is one random variable; repeated observations keep
    # separate iid noise even when their latent coordinates are identical.
    unique = []
    groups = []
    for i, row in enumerate(queries):
        if row not in unique:
            unique.append(row)
            groups.append([])
        groups[unique.index(row)].append(i)
    grouped = [[math.fsum(w[i] for i in g) for g in groups] for w in weights]
    Ku = [
        [rbf(a, b, p["length_scale"], p["signal_variance"]) for b in unique]
        for a in unique
    ]
    Xu = [
        [rbf(a, b, p["length_scale"], p["signal_variance"]) for b in train]
        for a in unique
    ]
    projected_cross = [
        [math.fsum(w[i] * Xu[i][j] for i in range(len(unique))) for j in range(n)]
        for w in grouped
    ]
    solved = [_forward(lower, k) for k in projected_cross]
    prior = [
        [
            math.fsum(
                weights_a[i] * Ku[i][j] * weights_b[j]
                for i in range(len(unique))
                for j in range(len(unique))
            )
            for weights_b in grouped
        ]
        for weights_a in grouped
    ]
    correction = [
        [math.fsum(a * b for a, b in zip(x, y)) for y in solved] for x in solved
    ]
    raw = [[prior[a][b] - correction[a][b] for b in range(q)] for a in range(q)]
    contracted = [
        [
            math.fsum(
                weights[a][i] * S[i][j] * weights[b][j]
                for i in range(m)
                for j in range(m)
            )
            for b in range(q)
        ]
        for a in range(q)
    ]
    tolerance = [
        [
            1e-10
            * (
                1
                + abs(prior[a][b])
                + abs(correction[a][b])
                + math.fsum(
                    abs(weights[a][i] * S[i][j] * weights[b][j])
                    for i in range(m)
                    for j in range(m)
                )
            )
            for b in range(q)
        ]
        for a in range(q)
    ]
    latent = [row[:] for row in raw]
    clamped = []
    for a in range(q):
        for b in range(q):
            if abs(raw[a][b] - contracted[a][b]) > tolerance[a][b]:
                raise ValueError(
                    "joint contraction disagrees with grouped conditioning beyond numerical gate"
                )
        if raw[a][a] < -tolerance[a][a]:
            raise ValueError("linear variance negative beyond numerical tolerance")
        if raw[a][a] < 0:
            clamped.append(a)
            latent[a][a] = 0.0
    noise = [
        [p["noise_variance"] * math.fsum(x * y for x, y in zip(a, b)) for b in weights]
        for a in weights
    ]
    observation = [[latent[a][b] + noise[a][b] for b in range(q)] for a in range(q)]
    diagonal_only = [
        [
            math.fsum(weights[a][i] * S[i][i] * weights[b][i] for i in range(m))
            for b in range(q)
        ]
        for a in range(q)
    ]
    means = [
        math.fsum(w * mu for w, mu in zip(row, posterior["means"])) for row in weights
    ]
    contrasts = []
    for a, w in enumerate(weights):
        cells = [[w[i] * S[i][j] * w[j] for j in range(m)] for i in range(m)]
        lv, ov = latent[a][a], observation[a][a]
        dv = diagonal_only[a][a]
        contrasts.append(
            dict(
                index=a,
                weights=w,
                grouped_weights=grouped[a],
                mean=means[a],
                latent_variance=lv,
                observation_variance=ov,
                diagonal_only_latent_variance=dv,
                diagonal_only_observation_variance=dv + noise[a][a],
                raw_latent_variance=raw[a][a],
                prior_variance=prior[a][a],
                conditioning_reduction=correction[a][a],
                future_noise_variance=noise[a][a],
                cell_terms=cells,
                diagonal_sum=math.fsum(cells[i][i] for i in range(m)),
                off_diagonal_sum=math.fsum(
                    cells[i][j] for i in range(m) for j in range(m) if i != j
                ),
                contraction=contracted[a][a],
                signed_contraction_disagreement=raw[a][a] - contracted[a][a],
                numerical_gate=tolerance[a][a],
                latent_sd=math.sqrt(lv),
                observation_sd=math.sqrt(ov),
                latent_two_sd_interval=[
                    means[a] - 2 * math.sqrt(lv),
                    means[a] + 2 * math.sqrt(lv),
                ],
                observation_two_sd_interval=[
                    means[a] - 2 * math.sqrt(ov),
                    means[a] + 2 * math.sqrt(ov),
                ],
            )
        )
    result = dict(
        version="kernellab_joint_v1",
        input=inp,
        query_posterior=posterior,
        query_prior_covariance=K,
        query_training_cross=cross,
        identical_query_groups=groups,
        grouped_queries=unique,
        grouped_weights=grouped,
        projected_training_cross=projected_cross,
        projected_prior_covariance=prior,
        conditioning_reduction=correction,
        raw_latent_covariance=raw,
        contracted_latent_covariance=contracted,
        latent_covariance=latent,
        observation_covariance=observation,
        future_noise_covariance=noise,
        diagonal_only_latent_covariance=diagonal_only,
        means=means,
        contrasts=contrasts,
        clamped_contrast_diagonals=clamped,
        scope="Fixed supplied RBF GP, explicit training jitter and iid future observation noise. Linear functionals of one joint posterior, not independent query marginals. Repeated latent coordinates coalesce; repeated future observations retain independent noise. Binary64 numerical diagnostics, not PSD, calibration, simultaneous coverage or optimum certificates.",
    )

    def finite(x):
        if type(x) is float and not math.isfinite(x):
            raise ValueError("nonfinite joint audit")
        if isinstance(x, dict):
            for v in x.values():
                finite(v)
        if isinstance(x, list):
            for v in x:
                finite(v)

    finite(result)
    return result


def main(argv=None):
    import argparse, sys

    p = argparse.ArgumentParser(description="Joint GP linear-contrast audit")
    p.add_argument("input")
    args = p.parse_args(argv)

    def unique(pairs):
        out = {}
        for k, v in pairs:
            if k in out:
                raise ValueError("duplicate JSON key")
            out[k] = v
        return out

    try:
        if args.input == "-":
            text = sys.stdin.read(65537)
        else:
            with open(args.input, encoding="utf-8") as f:
                text = f.read(65537)
        if len(text.encode()) > 65536:
            raise ValueError("input exceeds 64 KiB")
        out = audit(json.loads(text, object_pairs_hook=unique))
        print(json.dumps(out, allow_nan=False, indent=2, sort_keys=True))
    except (ValueError, TypeError, OverflowError, OSError, RecursionError) as e:
        print("error: " + str(e), file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
