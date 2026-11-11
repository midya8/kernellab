"""Synthetic finite joint-covariance controls; no model or seed selection."""

import copy
import json
from pathlib import Path
from kernellab.joint import audit

ROOT = Path(__file__).resolve().parents[1]


def controls():
    base = dict(
        training=dict(
            rows=[[0.0]],
            targets=[0.0],
            kernel=dict(
                length_scale=1.0,
                signal_variance=1.0,
                noise_variance=0.0,
                jitter=0.0,
                prior_mean=0.0,
            ),
        ),
        queries=[[-1.0], [1.0]],
        weights=[[1.0, -1.0], [0.5, 0.5]],
    )
    out = []

    def add(name, description, **changes):
        x = copy.deepcopy(base)
        for k, v in changes.items():
            x[k] = v
        out.append(dict(name=name, description=description, input=x))

    add(
        "opposite-sides",
        "Same marginal variances as repeated-location control, but negative posterior cross covariance. Difference uncertainty is larger than the diagonal-only estimate.",
    )
    add(
        "repeated-latent",
        "Same marginal variances as opposite-sides control. Two queries of the identical latent value cancel exactly in a difference.",
        queries=[[-1.0], [-1.0]],
    )
    training = copy.deepcopy(base["training"])
    training["kernel"]["noise_variance"] = 0.2
    add(
        "repeated-noisy",
        "Identical latent value cancels, but two independent future observation noises do not. Difference noise variance is twice the noise variance.",
        training=training,
        queries=[[-1.0], [-1.0]],
    )
    add(
        "nearby-positive",
        "Nearby queries on the same side share positive posterior covariance. Diagonal-only uncertainty overstates their difference.",
        queries=[[0.8], [1.0]],
    )
    add(
        "training-query",
        "Noiseless training query is deterministic under fixed assumptions; a separate unobserved query remains random.",
        queries=[[0.0], [1.0]],
    )
    add(
        "zero-weights",
        "The zero linear functional is deterministic, not an optimizer or feasibility certificate.",
        weights=[[0.0, 0.0]],
    )
    add(
        "shared-observation-noise",
        "Two contrasts reusing a future observation share its noise. Contrast covariance adds noise times W times transpose(W).",
        training=training,
        queries=[[-1.0], [-1.0]],
        weights=[[1.0, 0.0], [1.0, -1.0], [0.5, 0.5]],
    )
    tj = copy.deepcopy(base["training"])
    tj["kernel"]["jitter"] = 0.01
    add(
        "training-jitter",
        "Explicit jitter changes conditioning but is not future observation noise. Repeated latent differences still vanish.",
        training=tj,
        queries=[[0.0], [0.0]],
    )
    many = dict(
        rows=[[-1.0, 0.0], [0.0, 1.0], [1.0, -1.0]],
        targets=[0.5, -0.2, 0.8],
        kernel=dict(
            length_scale=0.8,
            signal_variance=1.3,
            noise_variance=0.1,
            jitter=1e-8,
            prior_mean=0.3,
        ),
    )
    add(
        "signed-multidimensional",
        "Signed linear forms over a two-feature query set retain every pair contribution and cross-contrast covariance.",
        training=many,
        queries=[[-0.5, 0.2], [0.4, 0.5], [1.0, -0.8]],
        weights=[[1.0, -2.0, 0.5], [1 / 3, 1 / 3, 1 / 3]],
    )
    far = copy.deepcopy(base["training"])
    far["rows"] = [[100.0]]
    add(
        "far-training",
        "Training-query covariance underflows here; posterior returns to the supplied prior, not a calibrated confidence statement.",
        training=far,
    )
    small = copy.deepcopy(base["training"])
    small["kernel"]["signal_variance"] = 1e-12
    add(
        "tiny-signal",
        "Small signal variance remains explicit. The numerical gate is a diagnostic, not a relative-error guarantee near zero.",
        training=small,
    )
    large = copy.deepcopy(base["training"])
    large["targets"] = [1e6]
    large["kernel"]["prior_mean"] = -1e6
    add(
        "large-mean",
        "Changing target and supplied mean moves linear means but does not change fixed-kernel covariance.",
        training=large,
    )
    return out


def main():
    items = []
    for c in controls():
        r = audit(c["input"])
        (ROOT / "kernellab/artifacts" / ("joint-" + c["name"] + ".json")).write_text(
            json.dumps(dict(c, report=r), allow_nan=False, sort_keys=True, indent=2)
            + "\n"
        )
        items.append(c)
        print(
            c["name"],
            [
                (
                    x["latent_variance"],
                    x["diagonal_only_latent_variance"],
                    x["future_noise_variance"],
                )
                for x in r["contrasts"]
            ],
        )
    (ROOT / "docs/joint-data.js").write_text(
        '"use strict";globalThis.KERNEL_JOINT_CONTROLS='
        + json.dumps(items, allow_nan=False, separators=(",", ":"))
        + ";\n"
    )


if __name__ == "__main__":
    main()
