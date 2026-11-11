"use strict";
const test = require("node:test"),
  assert = require("node:assert/strict"),
  fs = require("node:fs"),
  os = require("node:os"),
  path = require("node:path"),
  { spawn, spawnSync } = require("node:child_process");
const { waitForDebugger } = require("./chrome_support.js");
const controls = fs
  .readdirSync(path.join(__dirname, "../kernellab/artifacts"))
  .filter((p) => p.startsWith("joint-"))
  .sort()
  .map((p) =>
    JSON.parse(
      fs.readFileSync(path.join(__dirname, "../kernellab/artifacts", p)),
    ),
  );
const delay = (ms) => new Promise((r) => setTimeout(r, ms));
test(
  "real Chrome full joint contrasts, modes, cells, editing, downloads and36layouts",
  { timeout: 120000 },
  async () => {
    const profile = fs.mkdtempSync(path.join(os.tmpdir(), "kernel-joint-")),
      downloads = path.join(profile, "downloads");
    fs.mkdirSync(downloads);
    const browser = spawn(
      process.env.CHROME_BIN || "google-chrome",
      [
        "--headless",
        "--no-sandbox",
        "--disable-gpu",
        "--disable-extensions",
        "--disable-dev-shm-usage",
        "--remote-debugging-port=0",
        `--user-data-dir=${profile}`,
        "about:blank",
      ],
      { stdio: ["ignore", "ignore", "pipe"] },
    );
    let ws,
      error,
      stderr = "";
    browser.on("error", (e) => (error = e));
    browser.stderr.on("data", (d) => (stderr = (stderr + d).slice(-4000)));
    try {
      for (
        let i = 0;
        i < 1200 &&
        !error &&
        browser.exitCode === null &&
        !fs.existsSync(path.join(profile, "DevToolsActivePort"));
        i++
      )
        await delay(25);
      assert.ok(
        fs.existsSync(path.join(profile, "DevToolsActivePort")),
        `${error?.message || stderr}`,
      );
      const port = fs
          .readFileSync(path.join(profile, "DevToolsActivePort"), "utf8")
          .split("\n")[0],
        tab = await waitForDebugger(`http://127.0.0.1:${port}/json/list`, {
          alive: () => browser.exitCode === null && !error,
        });
      ws = new WebSocket(tab.webSocketDebuggerUrl);
      await new Promise((res, rej) => {
        ws.onopen = res;
        ws.onerror = rej;
      });
      let seq = 0;
      const pending = new Map();
      ws.onmessage = (e) => {
        const m = JSON.parse(e.data);
        if (pending.has(m.id)) {
          const { res, rej } = pending.get(m.id);
          pending.delete(m.id);
          m.error ? rej(Error(JSON.stringify(m.error))) : res(m.result);
        }
      };
      const call = (method, params = {}) =>
          new Promise((res, rej) => {
            const id = ++seq;
            pending.set(id, { res, rej });
            ws.send(JSON.stringify({ id, method, params }));
          }),
        js = async (expression) => {
          const r = await call("Runtime.evaluate", {
            expression,
            returnByValue: true,
            awaitPromise: true,
          });
          if (r.exceptionDetails)
            throw Error(JSON.stringify(r.exceptionDetails));
          return r.result.value;
        };
      await call("Page.enable");
      await call("Page.navigate", {
        url:
          process.env.KERNEL_JOINT_URL ||
          "file://" + path.resolve(__dirname, "../docs/joint.html"),
      });
      for (let i = 0; i < 500; i++) {
        if (await js("Boolean(globalThis.KERNEL_JOINT_REPORT)")) break;
        await delay(20);
      }
      assert.ok(await js("Boolean(globalThis.KERNEL_JOINT_REPORT)"));
      const near = (a, b) =>
        assert.ok(
          Number.isFinite(a) && Math.abs(a - b) <= 2e-8 * (1 + Math.abs(b)),
          a + " vs " + b,
        );
      let contrasts = 0,
        modes = 0,
        cells = 0,
        layouts = 0;
      for (const c of controls) {
        await js(
          `document.getElementById('control').value=${JSON.stringify(c.name)};KERNEL_JOINT_APP.restore()`,
        );
        assert.deepEqual(await js("KERNEL_JOINT_REPORT.input"), c.input);
        for (let i = 0; i < c.report.contrasts.length; i++) {
          await js(
            `document.getElementById('contrast').value='${i}';document.getElementById('contrast').dispatchEvent(new Event('change'))`,
          );
          for (const kind of ["latent", "observation"]) {
            await js(
              `document.getElementById('kind').value='${kind}';KERNEL_JOINT_APP.render()`,
            );
            const ref = c.report.contrasts[i],
              v =
                kind === "latent"
                  ? ref.latent_variance
                  : ref.observation_variance,
              d =
                kind === "latent"
                  ? ref.diagonal_only_latent_variance
                  : ref.diagonal_only_observation_variance;
            near(
              Number(await js("document.getElementById('full').textContent")),
              v,
            );
            near(
              Number(
                await js("document.getElementById('diagonal').textContent"),
              ),
              d,
            );
            near(
              Number(
                await js("document.getElementById('offdiag').textContent"),
              ),
              ref.off_diagonal_sum,
            );
            const C = c.report.query_posterior[kind + "_covariance"];
            for (const [id, A] of [
              ["covariance", C],
              [
                "pairs",
                C.map((r, a) =>
                  r.map((v, b) => ref.weights[a] * v * ref.weights[b]),
                ),
              ],
              ["projected", c.report[kind + "_covariance"]],
            ]) {
              const displayed = await js(
                `Array.from(document.querySelectorAll('#${id} tbody tr')).map(tr=>Array.from(tr.querySelectorAll('td')).map(td=>Number(td.textContent)))`,
              );
              assert.equal(displayed.length, A.length);
              for (let a = 0; a < A.length; a++) {
                assert.equal(displayed[a].length, A[a].length);
                for (let b = 0; b < A[a].length; b++) {
                  near(displayed[a][b], A[a][b]);
                  cells++;
                }
              }
            }
            modes++;
          }
          contrasts++;
        }
        for (const width of [320, 390, 1280]) {
          await call("Emulation.setDeviceMetricsOverride", {
            width,
            height: 1100,
            deviceScaleFactor: 1,
            mobile: width < 700,
          });
          assert.ok(
            await js("document.documentElement.scrollWidth<=innerWidth"),
            "overflow " + c.name + width,
          );
          assert.ok(
            await js(
              `Array.from(document.querySelectorAll('button,select,summary,textarea')).every(e=>e.getBoundingClientRect().height===0||e.getBoundingClientRect().height>=44)`,
            ),
            "touch " + c.name + width,
          );
          layouts++;
        }
      }
      await js(
        `document.getElementById('control').value='opposite-sides';KERNEL_JOINT_APP.restore();document.getElementById('contrast').focus()`,
      );
      for (const type of ["keyDown", "keyUp"])
        await call("Input.dispatchKeyEvent", {
          type,
          key: "ArrowDown",
          code: "ArrowDown",
          windowsVirtualKeyCode: 40,
        });
      await call("Input.dispatchKeyEvent", {
        type: "keyDown",
        key: "Enter",
        code: "Enter",
        windowsVirtualKeyCode: 13,
      });
      await call("Input.dispatchKeyEvent", {
        type: "keyUp",
        key: "Enter",
        code: "Enter",
        windowsVirtualKeyCode: 13,
      });
      assert.equal(await js("KERNEL_JOINT_SELECTED.index"), 1);
      await js(
        `document.querySelector('.input-panel details').open=true;document.getElementById('input').focus()`,
      );
      async function replace(text) {
        for (const type of ["keyDown", "keyUp"])
          await call("Input.dispatchKeyEvent", {
            type,
            key: "a",
            code: "KeyA",
            windowsVirtualKeyCode: 65,
            modifiers: 2,
          });
        await call("Input.insertText", { text });
      }
      await replace('{"training":1,"training":2}');
      assert.equal(await js("KERNEL_JOINT_APP.getReport()"), null);
      assert.equal(
        await js("document.getElementById('full').textContent"),
        "-",
      );
      assert.equal(
        await js("document.getElementById('download').disabled"),
        true,
      );
      assert.equal(
        await js("document.querySelectorAll('#pairs td').length"),
        0,
      );
      const edited = structuredClone(
        controls.find((c) => c.name === "signed-multidimensional").input,
      );
      edited.weights = [[0.2, -0.5, 0.7]];
      await replace(JSON.stringify(edited));
      assert.deepEqual(await js("KERNEL_JOINT_REPORT.input"), edited);
      await js(
        `Object.defineProperty(navigator,'clipboard',{value:{writeText:async()=>{throw Error('denied')}},configurable:true});document.getElementById('copy').click()`,
      );
      await delay(10);
      assert.match(
        await js("document.getElementById('status').textContent"),
        /Copy was blocked/,
      );
      await js(
        `Object.defineProperty(navigator,'clipboard',{value:{writeText:async s=>globalThis.copied=s},configurable:true});document.getElementById('copy').click()`,
      );
      await delay(10);
      assert.deepEqual(await js("JSON.parse(copied).input"), edited);
      await call("Browser.setDownloadBehavior", {
        behavior: "allow",
        downloadPath: downloads,
      });
      for (const name of ["opposite-sides", "repeated-noisy"]) {
        await js(
          `document.getElementById('control').value='${name}';KERNEL_JOINT_APP.restore();document.getElementById('download').click()`,
        );
        const f = path.join(downloads, "kernellab-joint-report.json");
        for (let i = 0; i < 200 && !fs.existsSync(f); i++) await delay(25);
        assert.ok(fs.existsSync(f));
        const py = spawnSync(
          "python",
          [
            "-c",
            'import sys,json,math\nfrom kernellab.joint import audit\nr=json.load(open(sys.argv[1]));expected=audit(r["input"])\ndef same(a,b):\n if type(b) in (float,int):assert type(a) in (float,int) and abs(a-b)<=2e-9*(1+abs(b))\n elif isinstance(b,dict):\n  assert a.keys()==b.keys()\n  for k in b:same(a[k],b[k])\n elif isinstance(b,list):\n  assert len(a)==len(b)\n  for x,y in zip(a,b):same(x,y)\n else:assert a==b\nsame(r,expected)',
            f,
          ],
          { cwd: path.resolve(__dirname, ".."), encoding: "utf8" },
        );
        assert.equal(py.status, 0, py.stderr);
        fs.unlinkSync(f);
      }
      console.log(
        JSON.stringify({
          controls: 12,
          contrasts,
          modes,
          cells,
          layouts,
          completedNativeReplayedDownloads: 2,
        }),
      );
      if (process.env.KERNEL_JOINT_SCREENSHOTS) {
        fs.mkdirSync(process.env.KERNEL_JOINT_SCREENSHOTS, { recursive: true });
        for (const name of [
          "opposite-sides",
          "repeated-latent",
          "repeated-noisy",
        ])
          for (const width of [390, 1280]) {
            await js(
              `document.getElementById('control').value='${name}';KERNEL_JOINT_APP.restore();document.getElementById('kind').value='${name === "repeated-noisy" ? "observation" : "latent"}';KERNEL_JOINT_APP.render();document.querySelector('.input-panel details').open=false;document.activeElement.blur()`,
            );
            await call("Emulation.setDeviceMetricsOverride", {
              width,
              height: 1200,
              deviceScaleFactor: 1,
              mobile: width < 700,
            });
            for (const [part, id] of [
              ["top", null],
              ["decision", "full"],
              ["cells", "covariance"],
            ]) {
              await js(
                id
                  ? `document.getElementById('${id}').scrollIntoView({block:'start'});scrollBy(0,-70)`
                  : "scrollTo(0,0)",
              );
              const image = await call("Page.captureScreenshot", {
                format: "png",
              });
              fs.writeFileSync(
                path.join(
                  process.env.KERNEL_JOINT_SCREENSHOTS,
                  name + "-" + width + "-" + part + ".png",
                ),
                Buffer.from(image.data, "base64"),
              );
            }
          }
      }
    } finally {
      if (ws) ws.close();
      if (browser.pid && browser.exitCode === null) {
        browser.kill("SIGTERM");
        await new Promise((res) => browser.once("exit", res));
      }
      await fs.promises.rm(profile, {
        recursive: true,
        force: true,
        maxRetries: 10,
        retryDelay: 200,
      });
    }
  },
);
