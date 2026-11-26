// Dumps JavaScript Math/division results on edge values, for the Python IEEE helpers.
// Usage: node ieee_dump.mjs  (pure JavaScript, no upstream dependency)
const values = [0, -0, 1, -1, 0.5, -0.5, 1e308, -1e308, Infinity, -Infinity, NaN];
const text = (x) => (Object.is(x, -0) ? "-0" : String(x));
const out = { div: [], exp: [], log: [], max: [] };
for (const a of values) for (const b of values) {
  out.div.push({ a: text(a), b: text(b), r: text(a / b) });
  out.max.push({ a: text(a), b: text(b), r: text(Math.max(a, b)) });
}
for (const x of values) {
  out.exp.push({ x: text(x), r: text(Math.exp(x)) });
  out.log.push({ x: text(x), r: text(Math.log(x)) });
}
console.log(JSON.stringify(out));
