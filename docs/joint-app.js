"use strict";
(() => {
  const $ = (id) => document.getElementById(id),
    controls = KERNEL_JOINT_CONTROLS;
  const fmt = (x) =>
    Object.is(x, -0) || x === 0
      ? "0"
      : Math.abs(x) < 1e-5 || Math.abs(x) >= 1e7
        ? x.toExponential(6)
        : x.toPrecision(9);
  let report = null,
    index = 0;
  for (const c of controls) {
    const o = document.createElement("option");
    o.value = c.name;
    o.textContent = c.name.replaceAll("-", " ");
    $("control").append(o);
  }
  function cell(row, text, tag = "td") {
    const e = document.createElement(tag);
    e.textContent = text;
    row.append(e);
    return e;
  }
  function matrix(id, A) {
    const t = $(id);
    t.replaceChildren();
    const h = t.createTHead().insertRow();
    cell(h, "i / j", "th");
    A.forEach((_, i) => cell(h, String(i), "th"));
    const body = t.createTBody(),
      scale = Math.max(0, ...A.flat().map(Math.abs));
    A.forEach((r, i) => {
      const tr = body.insertRow();
      cell(tr, String(i), "th");
      r.forEach((v) => {
        const td = cell(tr, fmt(v));
        const a = scale ? Math.abs(v) / scale : 0;
        td.style.backgroundColor =
          v < 0 ? `rgba(168,77,37,${a * 0.3})` : `rgba(33,100,151,${a * 0.25})`;
      });
    });
  }
  function bands(mean, v, d) {
    const svg = $("bands"),
      mobile = innerWidth < 680,
      W = mobile ? 360 : 900,
      H = 230,
      lo = Math.min(mean - 2 * Math.sqrt(v), mean - 2 * Math.sqrt(d)),
      hi = Math.max(mean + 2 * Math.sqrt(v), mean + 2 * Math.sqrt(d)),
      span = hi - lo || Math.max(1, Math.abs(mean) * 0.1),
      left = lo - span * 0.12,
      right = hi + span * 0.12;
    const X = (x) =>
      (mobile ? 100 : 180) +
      ((mobile ? 230 : 660) * (x - left)) / (right - left);
    svg.setAttribute("viewBox", `0 0 ${W} ${H}`);
    svg.replaceChildren();
    const el = (name, attrs, text) => {
      const e = document.createElementNS("http://www.w3.org/2000/svg", name);
      for (const [k, v] of Object.entries(attrs)) e.setAttribute(k, String(v));
      if (text !== undefined) e.textContent = text;
      svg.append(e);
      return e;
    };
    [
      ["Full joint", v, "#235f9b", 65],
      ["Diagonal only", d, "#a05126", 135],
    ].forEach(([name, variance, color, y]) => {
      el(
        "text",
        { x: 4, y: y + 5, fill: "#182c32", "font-size": mobile ? 14 : 18 },
        name,
      );
      const a = mean - 2 * Math.sqrt(variance),
        b = mean + 2 * Math.sqrt(variance);
      el("line", {
        x1: X(a),
        x2: X(b),
        y1: y,
        y2: y,
        stroke: color,
        "stroke-width": 8,
      });
      el("circle", { cx: X(mean), cy: y, r: 6, fill: "#182c32" });
      for (const x of [a, b])
        el("line", {
          x1: X(x),
          x2: X(x),
          y1: y - 12,
          y2: y + 12,
          stroke: color,
          "stroke-width": 2,
        });
      el(
        "text",
        {
          x: X(a),
          y: y + 30,
          "font-size": 14,
          fill: color,
          "text-anchor": "middle",
        },
        mobile ? a.toPrecision(4) : fmt(a),
      );
      if (b !== a)
        el(
          "text",
          {
            x: X(b),
            y: y + 30,
            "font-size": 14,
            fill: color,
            "text-anchor": "middle",
          },
          mobile ? b.toPrecision(4) : fmt(b),
        );
    });
    el(
      "text",
      { x: mobile ? 4 : 180, y: 215, "font-size": 14, fill: "#4d6268" },
      mobile
        ? "Shared axis · same mean · not simultaneous"
        : "Shared linear axis · center = same scalar mean · not simultaneous intervals",
    );
  }
  function render() {
    if (!report) return;
    const c = report.contrasts[index],
      ob = $("kind").value === "observation",
      C = ob
        ? report.query_posterior.observation_covariance
        : report.query_posterior.latent_covariance,
      v = ob ? c.observation_variance : c.latent_variance,
      d = ob
        ? c.diagonal_only_observation_variance
        : c.diagonal_only_latent_variance;
    $("full").textContent = fmt(v);
    $("diagonal").textContent = fmt(d);
    $("offdiag").textContent = fmt(c.off_diagonal_sum);
    $("equation").textContent =
      `Mean = ${fmt(c.mean)}; diagonal terms = ${fmt(d)}; all off-diagonal terms = ${fmt(c.off_diagonal_sum)}. Reported full variance uses the grouped prior-minus-conditioning computation, checked against the direct pair sum.`;
    bands(c.mean, v, d);
    const body = $("summary").tBodies[0];
    body.replaceChildren();
    for (const [label, value] of [
      ["Linear mean", c.mean],
      ["Latent prior variance", c.prior_variance],
      ["Conditioning reduction", c.conditioning_reduction],
      ["Raw latent variance before any clamp", c.raw_latent_variance],
      ["Future iid-noise contribution", c.future_noise_variance],
      ["Direct latent pair sum", c.contraction],
      [
        "Signed grouped-minus-pair disagreement",
        c.signed_contraction_disagreement,
      ],
      ["Numerical comparison gate, not an error bound", c.numerical_gate],
      ["Selected model standard deviation", Math.sqrt(v)],
    ]) {
      const tr = body.insertRow();
      cell(tr, label);
      cell(tr, fmt(value));
    }
    matrix("covariance", C);
    matrix(
      "pairs",
      C.map((r, i) => r.map((v, j) => c.weights[i] * v * c.weights[j])),
    );
    matrix(
      "projected",
      ob ? report.observation_covariance : report.latent_covariance,
    );
    const wb = $("weights").tBodies[0];
    wb.replaceChildren();
    report.input.queries.forEach((row, i) => {
      const tr = wb.insertRow();
      [
        `${i} / [${row.join(", ")}]`,
        fmt(report.query_posterior.means[i]),
        fmt(report.query_posterior.latent_variances[i]),
        fmt(report.query_posterior.observation_variances[i]),
        fmt(c.weights[i]),
      ].forEach((v) => cell(tr, v));
    });
    $("groups").textContent =
      `Identical query-index groups: ${JSON.stringify(report.identical_query_groups)}. Grouped latent coefficients for this contrast: ${JSON.stringify(c.grouped_weights)}. Future noise still uses every original query occurrence.`;
    $("matrix-note").textContent = ob
      ? "Future observation covariance = latent query covariance + noise_variance on each query diagonal. No query jitter."
      : "Latent function covariance. Repeated coordinates refer to the same latent value, including a possibly singular joint law. We do not factor this query covariance.";
    $("report").textContent = JSON.stringify(report, null, 2);
    globalThis.KERNEL_JOINT_SELECTED = c;
  }
  function invalid(e) {
    report = null;
    $("contrast").replaceChildren();
    globalThis.KERNEL_JOINT_REPORT = null;
    globalThis.KERNEL_JOINT_SELECTED = null;
    $("error").textContent = e.message;
    for (const id of ["full", "diagonal", "offdiag"]) $(id).textContent = "-";
    for (const id of ["covariance", "pairs", "projected"])
      $(id).replaceChildren();
    for (const id of ["summary", "weights"]) $(id).tBodies[0].replaceChildren();
    $("bands").replaceChildren();
    for (const id of ["report", "equation", "groups", "matrix-note", "status"])
      $(id).textContent = "";
    $("copy").disabled = $("download").disabled = true;
  }
  function compute() {
    try {
      const x = JOINT_JSON.parse($("input").value);
      report = KernelJoint.audit(x);
      index = Math.min(index, report.contrasts.length - 1);
      $("contrast").replaceChildren();
      report.contrasts.forEach((c, i) => {
        const o = document.createElement("option");
        o.value = i;
        o.textContent = `Contrast ${i}: [${c.weights.join(", ")}]`;
        $("contrast").append(o);
      });
      $("contrast").value = index;
      $("error").textContent = "";
      $("status").textContent = "";
      $("copy").disabled = $("download").disabled = false;
      globalThis.KERNEL_JOINT_REPORT = report;
      render();
    } catch (e) {
      invalid(e);
    }
  }
  function restore() {
    const c = controls.find((c) => c.name === $("control").value);
    $("description").textContent = c.description;
    $("input").value = JSON.stringify(c.input, null, 2);
    index = 0;
    compute();
  }
  $("control").addEventListener("change", restore);
  $("restore").addEventListener("click", restore);
  $("input").addEventListener("input", compute);
  $("contrast").addEventListener("change", () => {
    index = Number($("contrast").value);
    render();
  });
  $("kind").addEventListener("change", render);
  $("copy").addEventListener("click", async () => {
    if (!report) return;
    try {
      await navigator.clipboard.writeText(JSON.stringify(report, null, 2));
      $("status").textContent = "Complete report copied.";
    } catch {
      $("status").textContent =
        "Copy was blocked. Use Download complete JSON instead.";
    }
  });
  $("download").addEventListener("click", () => {
    if (!report) return;
    const url = URL.createObjectURL(
        new Blob([JSON.stringify(report, null, 2) + "\n"], {
          type: "application/json",
        }),
      ),
      a = document.createElement("a");
    a.href = url;
    a.download = "kernellab-joint-report.json";
    document.body.append(a);
    a.click();
    a.remove();
    setTimeout(() => URL.revokeObjectURL(url), 30000);
    $("status").textContent = "Complete report download requested.";
  });
  window.addEventListener("resize", render);
  globalThis.KERNEL_JOINT_APP = {
    restore,
    compute,
    render,
    getReport: () => report,
  };
  restore();
})();
