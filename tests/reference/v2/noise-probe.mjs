import { FastNoise } from "./core/fast-noise.mjs";

let input = "";
for await (const chunk of process.stdin) input += chunk;
const points = JSON.parse(input);
const noise = new FastNoise(9876);
const result = points.map(([x, y, z]) => noise.fbm(x, y, z, 5, 2 / 3));
process.stdout.write(JSON.stringify(result));
