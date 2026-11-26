import readline from "node:readline";
import { isWordSeed, seedOf } from "./core/seed.mjs";

const targets = {
  seed_of: seedOf,
  is_word_seed: isWordSeed,
};

function encode(value) {
  if (typeof value === "number" && !Number.isFinite(value)) {
    return { __number__: String(value) };
  }
  return value;
}

const input = readline.createInterface({
  input: process.stdin,
  crlfDelay: Infinity,
});

for await (const line of input) {
  try {
    const request = JSON.parse(line);
    const target = targets[request.target];
    if (target === undefined) {
      throw new Error(`Unknown differential target: ${request.target}`);
    }
    const result = encode(target(request.value));
    process.stdout.write(JSON.stringify({ ok: true, result }) + "\n");
  } catch (error) {
    process.stdout.write(
      JSON.stringify({
        ok: false,
        error: error instanceof Error ? error.message : String(error),
      }) + "\n",
    );
  }
}
