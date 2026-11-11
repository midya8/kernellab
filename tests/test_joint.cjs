"use strict";
const test = require("node:test"),
  assert = require("node:assert/strict"),
  fs = require("node:fs"),
  path = require("node:path"),
  { spawnSync } = require("node:child_process");
const E = require("../docs/joint-engine.js"),
  J = require("../docs/joint-json.js");
const controls = fs
  .readdirSync(path.join(__dirname, "../kernellab/artifacts"))
  .filter((p) => p.startsWith("joint-"))
  .sort()
  .map((p) =>
    JSON.parse(
      fs.readFileSync(path.join(__dirname, "../kernellab/artifacts", p)),
    ),
  );
function same(a, b, p = "report") {
  if (typeof b === "number") {
    assert.equal(typeof a, "number", p);
    assert.ok(
      Number.isFinite(a) && Math.abs(a - b) <= 2e-9 * (1 + Math.abs(b)),
      p + ": " + a + " vs " + b,
    );
  } else if (Array.isArray(b)) {
    assert.ok(Array.isArray(a), p);
    assert.equal(a.length, b.length, p);
    b.forEach((v, i) => same(a[i], v, p + "." + i));
  } else if (b && typeof b === "object") {
    assert.deepEqual(Object.keys(a).sort(), Object.keys(b).sort(), p);
    for (const k of Object.keys(b)) same(a[k], b[k], p + "." + k);
  } else assert.equal(a, b, p);
}
test("12complete native/independentLDL reports", () => {
  assert.equal(controls.length, 12);
  for (const c of controls) same(E.audit(c.input), c.report, c.name);
});
test("same marginals and different joint linear variance", () => {
  const a = E.audit(controls.find((c) => c.name === "opposite-sides").input),
    b = E.audit(controls.find((c) => c.name === "repeated-latent").input);
  assert.deepEqual(
    a.query_posterior.latent_variances,
    b.query_posterior.latent_variances,
  );
  assert.equal(b.contrasts[0].latent_variance, 0);
  assert.ok(a.contrasts[0].latent_variance > 1.7);
});
test("strict JSON and bounded schema", () => {
  for (const text of [
    '{"a":1,"a":2}',
    " ".repeat(65537),
    "{} trailing",
    "[1,]",
    "[".repeat(65),
  ])
    assert.throws(() => J.parse(text));
  for (const [k, v] of [
    ["weights", [[11, 0]]],
    ["queries", []],
    ["weights", [[true, 0]]],
    ["extra", 1],
    ["training", null],
  ]) {
    const x = structuredClone(
      controls.find((c) => c.name === "opposite-sides").input,
    );
    x[k] = v;
    assert.throws(() => E.audit(x));
  }
});
test("100generated complete native/independentLDL reports", () => {
  let seed = 10101;
  const rand = () => {
    seed = (Math.imul(seed, 1664525) + 1013904223) >>> 0;
    return seed / 2 ** 32;
  };
  const cases = Array.from({ length: 100 }, () => {
    const n = 1 + Math.floor(rand() * 6),
      m = 1 + Math.floor(rand() * 8),
      q = 1 + Math.floor(rand() * 4);
    return {
      training: {
        rows: Array.from({ length: n }, () => [rand() * 4 - 2, rand() * 4 - 2]),
        targets: Array.from({ length: n }, () => rand() * 2 - 1),
        kernel: {
          length_scale: 0.3 + rand() * 2,
          signal_variance: 1.3,
          noise_variance: 0.12,
          jitter: 1e-8,
          prior_mean: 0.3,
        },
      },
      queries: Array.from({ length: m }, () => [
        rand() * 4 - 2,
        rand() * 4 - 2,
      ]),
      weights: Array.from({ length: q }, () =>
        Array.from({ length: m }, () => rand() * 4 - 2),
      ),
    };
  });
  const py = spawnSync(
    "python",
    [
      "-c",
      "import sys,json\nfrom kernellab.joint import audit\nprint(json.dumps([audit(x) for x in json.load(sys.stdin)],allow_nan=False))",
    ],
    {
      cwd: path.resolve(__dirname, ".."),
      input: JSON.stringify(cases),
      encoding: "utf8",
      maxBuffer: 32 * 1024 * 1024,
    },
  );
  assert.equal(py.status, 0, py.stderr);
  const refs = JSON.parse(py.stdout);
  cases.forEach((x, i) => same(E.audit(x), refs[i], "case" + i));
});
