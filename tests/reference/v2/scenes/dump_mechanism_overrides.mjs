// Apply Python-generated mechanism overrides through the actual upstream constructor.
import { readFileSync } from "node:fs";
import { TectonicWorld } from "../tectonics/world.mjs";

const revive = (value) => {
  if (value === "__OROGEN_POSITIVE_INFINITY__") return Infinity;
  if (value === "__OROGEN_NEGATIVE_INFINITY__") return -Infinity;
  if (Array.isArray(value)) return value.map(revive);
  if (value && typeof value === "object") {
    return Object.fromEntries(Object.entries(value).map(([key, item]) => [key, revive(item)]));
  }
  return value;
};
const jsonSafe = (value) => {
  if (typeof value === "number" && !Number.isFinite(value)) {
    return value > 0 ? "__OROGEN_POSITIVE_INFINITY__" : "__OROGEN_NEGATIVE_INFINITY__";
  }
  if (Array.isArray(value)) return value.map(jsonSafe);
  if (value && typeof value === "object") {
    return Object.fromEntries(Object.entries(value).map(([key, item]) => [key, jsonSafe(item)]));
  }
  return value;
};

const payload = revive(JSON.parse(readFileSync(0, "utf8")));
const mesh = { numCells: 1 };
const output = {};
for (const [helper, groups] of Object.entries(payload)) {
  output[helper] = {};
  for (const [name, overrides] of Object.entries(groups)) {
    const world = new TectonicWorld(mesh, 1, overrides);
    const effective = {};
    for (const key of Object.keys(overrides)) effective[key] = jsonSafe(world.config[key]);
    output[helper][name] = effective;
  }
}
console.log(JSON.stringify(output));
