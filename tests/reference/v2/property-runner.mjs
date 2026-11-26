import readline from "node:readline";
import * as seed from "./core/seed.mjs";

// Static allowlist of modules, mirroring TARGET_MODULES in test_generated.py.
const modules = { seed };

// Map a snake_case target id to the camelCase export name, e.g. seed_of -> seedOf.
function jsName(snake) {
  return snake.replace(/_([a-z0-9])/g, (_, c) => c.toUpperCase());
}

function resolve(moduleId, targetId) {
  const namespace = modules[moduleId ?? "seed"];
  if (namespace === undefined) {
    throw new Error(`Unknown differential module: ${moduleId}`);
  }
  const target = namespace[jsName(targetId)];
  if (typeof target !== "function") {
    throw new Error(`Unknown differential target: ${moduleId}.${targetId}`);
  }
  return target;
}

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
    const target = resolve(request.module, request.target);
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
