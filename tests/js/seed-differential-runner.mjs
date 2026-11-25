import { readFile } from "node:fs/promises";
import { isWordSeed, seedOf } from "../../external/planet_heightmap_generation/v2/js/core/seed.js";
import { makeRng } from "../../external/planet_heightmap_generation/js/rng.js";

const chunks = [];
for await (const chunk of process.stdin) chunks.push(chunk);
const { texts, rngSeed, randomCount } = JSON.parse(Buffer.concat(chunks).toString("utf8"));
const rng = makeRng(rngSeed);
const seeds = texts.map(text => ({
  text,
  word: isWordSeed(text),
  seed: seedOf(text),
}));
const random = Array.from({ length: randomCount }, () => rng());
process.stdout.write(JSON.stringify({ seeds, random }));
