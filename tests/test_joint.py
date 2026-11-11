import copy
import importlib.util
import json
import math
import random
import subprocess
import sys
import unittest
from decimal import Decimal as D, localcontext
from pathlib import Path
from kernellab.joint import audit

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "controls", ROOT / "examples/joint_controls.py"
)
controls = importlib.util.module_from_spec(spec)
spec.loader.exec_module(controls)


def decimal_reference(x):
    # Decimal80 kernels and pivoted Gauss-Jordan inverse, independent of the
    # native Cholesky, query covariance, grouping and contraction code.
    with localcontext() as ctx:
        ctx.prec = 80
        p = x["training"]["kernel"]
        train = x["training"]["rows"]
        queries = x["queries"]
        n = len(train)

        def dec(v):
            return D(str(v))

        def kernel(a, b):
            return (
                dec(p["signal_variance"])
                * (
                    -sum(
                        ((dec(u) - dec(v)) / dec(p["length_scale"])) ** 2
                        for u, v in zip(a, b)
                    )
                    / 2
                ).exp()
            )

        A = [
            [
                kernel(a, b)
                + (dec(p["noise_variance"]) + dec(p["jitter"]) if i == j else 0)
                for j, b in enumerate(train)
            ]
            + [D(i == j) for j in range(n)]
            for i, a in enumerate(train)
        ]
        for i in range(n):
            k = max(range(i, n), key=lambda k: abs(A[k][i]))
            A[i], A[k] = A[k], A[i]
            pivot = A[i][i]
            A[i] = [v / pivot for v in A[i]]
            for j in range(n):
                if j != i:
                    c = A[j][i]
                    A[j] = [v - c * w for v, w in zip(A[j], A[i])]
        inv = [r[n:] for r in A]
        cross = [[kernel(a, b) for b in train] for a in queries]
        centered = [dec(y) - dec(p["prior_mean"]) for y in x["training"]["targets"]]
        mus = [
            dec(p["prior_mean"])
            + sum(k[i] * inv[i][j] * centered[j] for i in range(n) for j in range(n))
            for k in cross
        ]
        S = [
            [
                kernel(a, b)
                - sum(
                    cross[i][k] * inv[k][l] * cross[j][l]
                    for k in range(n)
                    for l in range(n)
                )
                for j, b in enumerate(queries)
            ]
            for i, a in enumerate(queries)
        ]
        W = [[dec(v) for v in r] for r in x["weights"]]
        m = len(queries)
        means = [sum(w * mu for w, mu in zip(row, mus)) for row in W]
        cov = [
            [sum(a[i] * S[i][j] * b[j] for i in range(m) for j in range(m)) for b in W]
            for a in W
        ]
        return (
            [float(v) for v in means],
            [[float(v) for v in r] for r in cov],
            [[float(v) for v in r] for r in S],
        )


class JointTests(unittest.TestCase):
    def test_all12_controls_saved(self):
        self.assertEqual(len(controls.controls()), 12)
        for c in controls.controls():
            self.assertEqual(
                json.loads(
                    (
                        ROOT / "kernellab/artifacts" / ("joint-" + c["name"] + ".json")
                    ).read_text()
                ),
                dict(c, report=audit(c["input"])),
            )

    def test_decimal80_all12_controls(self):
        for c in controls.controls():
            means, cov, S = decimal_reference(c["input"])
            r = audit(c["input"])
            for a, b in zip(means, r["means"]):
                self.assertAlmostEqual(a, b, places=8)
            for a, b in zip(cov, r["latent_covariance"]):
                for v, w in zip(a, b):
                    self.assertAlmostEqual(v, w, places=10)
            for a, b in zip(S, r["query_posterior"]["latent_covariance"]):
                for v, w in zip(a, b):
                    self.assertAlmostEqual(v, w, places=12)

    def test_72_generated_decimal80_systems(self):
        rng = random.Random(20261111)
        for _ in range(72):
            n = rng.randrange(1, 6)
            m = rng.randrange(1, 6)
            q = rng.randrange(1, 5)
            x = dict(
                training=dict(
                    rows=[[rng.uniform(-2, 2), rng.uniform(-2, 2)] for i in range(n)],
                    targets=[rng.uniform(-1, 1) for i in range(n)],
                    kernel=dict(
                        length_scale=rng.uniform(0.3, 2),
                        signal_variance=1.3,
                        noise_variance=0.12,
                        jitter=1e-8,
                        prior_mean=0.3,
                    ),
                ),
                queries=[[rng.uniform(-2, 2), rng.uniform(-2, 2)] for i in range(m)],
                weights=[[rng.uniform(-2, 2) for i in range(m)] for j in range(q)],
            )
            means, cov, _ = decimal_reference(x)
            r = audit(x)
            for a, b in zip(means, r["means"]):
                self.assertAlmostEqual(a, b, places=10)
            for a, b in zip(cov, r["latent_covariance"]):
                for v, w in zip(a, b):
                    self.assertAlmostEqual(v, w, places=10)

    def test_same_marginals_different_joint_law(self):
        a = audit(controls.controls()[0]["input"])
        b = audit(controls.controls()[1]["input"])
        self.assertEqual(
            a["query_posterior"]["latent_variances"],
            b["query_posterior"]["latent_variances"],
        )
        self.assertEqual(a["query_posterior"]["means"], b["query_posterior"]["means"])
        self.assertGreater(a["contrasts"][0]["latent_variance"], 1.7)
        self.assertEqual(b["contrasts"][0]["latent_variance"], 0)
        self.assertGreater(a["contrasts"][0]["diagonal_only_latent_variance"], 1.2)
        self.assertLess(a["query_posterior"]["latent_covariance"][0][1], 0)
        self.assertGreater(b["query_posterior"]["latent_covariance"][0][1], 0)

    def test_repeated_latent_and_independent_future_noise(self):
        r = audit(controls.controls()[2]["input"])
        self.assertEqual(r["contrasts"][0]["latent_variance"], 0)
        self.assertEqual(r["contrasts"][0]["observation_variance"], 0.4)
        self.assertEqual(r["identical_query_groups"], [[0, 1]])
        self.assertEqual(r["contrasts"][0]["grouped_weights"], [0])
        self.assertAlmostEqual(r["contrasts"][1]["future_noise_variance"], 0.1)

    def test_shared_future_observation_noise(self):
        r = audit(controls.controls()[6]["input"])
        N = r["future_noise_covariance"]
        self.assertEqual(N, [[0.2, 0.2, 0.1], [0.2, 0.4, 0], [0.1, 0, 0.1]])
        self.assertGreater(r["observation_covariance"][0][1], 0)

    def test_zero_weights_and_jitter_not_future_noise(self):
        for i in [5, 7]:
            r = audit(controls.controls()[i]["input"])
            c = r["contrasts"][0]
            self.assertEqual(c["latent_variance"], 0)
            self.assertEqual(c["future_noise_variance"], 0)
            self.assertEqual(c["observation_variance"], 0)
        self.assertGreater(
            audit(controls.controls()[7]["input"])["contrasts"][1]["latent_variance"], 0
        )
        from unittest.mock import patch
        import kernellab.joint as joint

        x = copy.deepcopy(controls.controls()[0]["input"])
        x["queries"] = [[0]]
        x["weights"] = [[1]]
        original = joint._factor
        lower, alpha, lml = original(*joint._training(x["training"]))
        with patch.object(
            joint, "_factor", return_value=([[math.sqrt(1 - 5e-11)]], alpha, lml)
        ):
            r = audit(x)
            self.assertLess(r["contrasts"][0]["raw_latent_variance"], 0)
            self.assertEqual(r["clamped_contrast_diagonals"], [0])
            self.assertEqual(r["contrasts"][0]["latent_variance"], 0)
        with patch.object(
            joint, "_factor", return_value=([[math.sqrt(0.9)]], alpha, lml)
        ):
            with self.assertRaisesRegex(ValueError, "gate"):
                audit(x)

    def test_target_edits_preserve_covariance(self):
        x = copy.deepcopy(controls.controls()[8]["input"])
        a = audit(x)
        x["training"]["targets"] = [v + 0.25 for v in x["training"]["targets"]]
        b = audit(x)
        self.assertEqual(a["latent_covariance"], b["latent_covariance"])
        self.assertNotEqual(a["means"], b["means"])

    def test_pair_terms_and_projected_schur_identity(self):
        for c in controls.controls():
            r = audit(c["input"])
            for v in r["contrasts"]:
                self.assertAlmostEqual(
                    v["contraction"], v["diagonal_sum"] + v["off_diagonal_sum"], 10
                )
                self.assertLessEqual(
                    abs(v["signed_contraction_disagreement"]), v["numerical_gate"]
                )
                self.assertAlmostEqual(
                    v["raw_latent_variance"],
                    v["prior_variance"] - v["conditioning_reduction"],
                    10,
                )

    def test_signed_scaling_and_query_permutation(self):
        x = copy.deepcopy(controls.controls()[8]["input"])
        a = audit(x)
        x["queries"] = x["queries"][::-1]
        x["weights"] = [r[::-1] for r in x["weights"]]
        b = audit(x)
        for v, w in zip(a["means"], b["means"]):
            self.assertAlmostEqual(v, w, 12)
        for v, w in zip(a["latent_covariance"], b["latent_covariance"]):
            for c, d in zip(v, w):
                self.assertAlmostEqual(c, d, 12)
        x["weights"] = [[-2 * v for v in row] for row in x["weights"]]
        c = audit(x)
        for v, w in zip(b["means"], c["means"]):
            self.assertAlmostEqual(-2 * v, w, 12)
        for v, w in zip(b["latent_covariance"], c["latent_covariance"]):
            for d, e in zip(v, w):
                self.assertAlmostEqual(4 * d, e, 12)

    def test_training_permutation(self):
        x = copy.deepcopy(controls.controls()[8]["input"])
        a = audit(x)
        x["training"]["rows"].reverse()
        x["training"]["targets"].reverse()
        b = audit(x)
        for c, d in zip(a["contrasts"], b["contrasts"]):
            self.assertAlmostEqual(c["latent_variance"], d["latent_variance"], 12)

    def test_full_budget(self):
        x = copy.deepcopy(controls.controls()[8]["input"])
        x["training"]["rows"] = [[i / 8, (i % 3) / 4] for i in range(16)]
        x["training"]["targets"] = [math.sin(i) for i in range(16)]
        x["queries"] = [[i / 4, i / 5] for i in range(8)]
        x["weights"] = [[(-1) ** (i + j) / 8 for i in range(8)] for j in range(4)]
        r = audit(x)
        self.assertEqual(len(r["query_posterior"]["latent_covariance"]), 8)
        self.assertEqual(len(r["contrasts"]), 4)

    def test_no_mutation_and_strict_schema(self):
        x = copy.deepcopy(controls.controls()[0]["input"])
        old = copy.deepcopy(x)
        audit(x)
        self.assertEqual(x, old)
        for key, val in [
            ("extra", 1),
            ("queries", []),
            ("queries", [[True], [1]]),
            ("queries", [[0]] * 9),
            ("weights", []),
            ("weights", [[1]]),
            ("weights", [[11, 0]]),
            ("weights", [[False, 0]]),
            ("weights", [[float("nan"), 0]]),
            ("weights", [[10**10000, 0]]),
        ]:
            y = copy.deepcopy(x)
            y[key] = val
            with self.assertRaises(ValueError, msg=key):
                audit(y)
        y = copy.deepcopy(x)
        y["training"]["rows"] = [[0]] * 17
        y["training"]["targets"] = [0] * 17
        with self.assertRaises(ValueError):
            audit(y)

    def test_singular_training_refuses_not_query_covariance(self):
        x = copy.deepcopy(controls.controls()[0]["input"])
        x["training"]["rows"] = [[0], [0]]
        x["training"]["targets"] = [0, 0]
        with self.assertRaisesRegex(ValueError, "pivot"):
            audit(x)
        x["training"]["kernel"]["jitter"] = 1e-6
        audit(x)
        audit(controls.controls()[1]["input"])

    def test_cli_archive_replay_and_refusals(self):
        cmd = [sys.executable, "-m", "kernellab.joint", "-"]
        x = controls.controls()[0]["input"]
        p = subprocess.run(
            cmd, input=json.dumps(x), text=True, capture_output=True, cwd=ROOT
        )
        self.assertEqual(p.returncode, 0, p.stderr)
        self.assertEqual(json.loads(p.stdout), audit(x))
        for text in [
            '{"training":1,"training":2}',
            " " * 65537,
            '{"queries":NaN}',
            '{"training":null,"queries":[],"weights":[]}',
        ]:
            p = subprocess.run(
                cmd, input=text, text=True, capture_output=True, cwd=ROOT
            )
            self.assertEqual(p.returncode, 2)
            self.assertFalse(p.stdout)
            self.assertNotIn("Traceback", p.stderr)


if __name__ == "__main__":
    unittest.main()
