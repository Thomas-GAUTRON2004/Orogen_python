import { readFileSync } from "node:fs";

import { isWordSeed, seedOf } from
  "../../external/planet_heightmap_generation/v2/js/core/seed.js";
import { makeRng } from
  "../../external/planet_heightmap_generation/js/rng.js";

const request = JSON.parse(readFileSync(0, "utf8"));

const functions = {
  "seed.seedAndRng": (input) => {
    const rng = makeRng(input.rngSeed);
    return {
      seeds: input.texts.map((text) => ({
        text,
        word: isWordSeed(text),
        seed: seedOf(text),
      })),
      random: Array.from(
        { length: input.randomCount },
        () => rng(),
      ),
    };
  },
};

const fn = functions[request.function];
if (typeof fn !== "function") {
  throw new Error(`Unknown differential function: ${request.function}`);
}

const result = await fn(request.input);
process.stdout.write(JSON.stringify(result));
