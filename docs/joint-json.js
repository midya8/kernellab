"use strict";
(function (root) {
  function strictParse(text) {
    if (new TextEncoder().encode(text).length > 65536)
      throw Error("Input exceeds64KiB");
    let at = 0,
      depth = 0;
    const ws = () => {
      while (/[ \t\r\n]/.test(text[at] || "") && at < text.length) at++;
    };
    function string() {
      const start = at++;
      while (at < text.length) {
        const c = text[at++];
        if (c === "\\") {
          at++;
          continue;
        }
        if (c === '"') return JSON.parse(text.slice(start, at));
      }
      throw Error("Unterminated string");
    }
    function value() {
      ws();
      if (++depth > 64) throw Error("JSON nesting exceeds64");
      let x;
      const c = text[at];
      if (c === '"') x = string();
      else if (c === "[") {
        at++;
        x = [];
        ws();
        if (text[at] !== "]")
          while (true) {
            x.push(value());
            ws();
            if (text[at] !== ",") break;
            at++;
          }
        if (text[at++] !== "]") throw Error("Invalid array");
      } else if (c === "{") {
        at++;
        x = {};
        const seen = new Set();
        ws();
        if (text[at] !== "}")
          while (true) {
            ws();
            if (text[at] !== '"') throw Error("Expected JSON key");
            const k = string();
            if (seen.has(k)) throw Error("Duplicate JSON key");
            seen.add(k);
            ws();
            if (text[at++] !== ":") throw Error("Expected colon");
            Object.defineProperty(x, k, {
              value: value(),
              writable: true,
              enumerable: true,
              configurable: true,
            });
            ws();
            if (text[at] !== ",") break;
            at++;
          }
        if (text[at++] !== "}") throw Error("Invalid object");
      } else {
        const m =
          /^(?:true|false|null|-?(?:0|[1-9][0-9]*)(?:\.[0-9]+)?(?:[eE][+-]?[0-9]+)?)/.exec(
            text.slice(at),
          );
        if (!m) throw Error("Invalid JSON value");
        at += m[0].length;
        x = JSON.parse(m[0]);
      }
      depth--;
      return x;
    }
    const out = value();
    ws();
    if (at !== text.length) throw Error("Trailing JSON data");
    return out;
  }
  const api = { parse: strictParse };
  if (typeof module !== "undefined") module.exports = api;
  root.JOINT_JSON = api;
})(globalThis);
