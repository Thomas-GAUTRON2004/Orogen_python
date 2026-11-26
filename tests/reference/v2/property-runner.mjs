import readline from "node:readline";

// Modules are loaded by file name from tests/reference/v2/core. The name is
// validated so that config or generated tests cannot reach other files.
const CORE_URL = new URL("./core/", import.meta.url);
const FILE_PATTERN = /^[A-Za-z0-9_-]+\.mjs$/;
const cache = new Map();

async function loadModule(file) {
  if (!FILE_PATTERN.test(file)) {
    throw new Error(`Invalid differential module: ${file}`);
  }
  if (!cache.has(file)) {
    cache.set(file, import(new URL(file, CORE_URL).href));
  }
  return cache.get(file);
}

// Non-finite numbers cannot be represented in JSON; encode them as Python expects.
function replacer(_key, value) {
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
    const namespace = await loadModule(request.module);
    const target = namespace[request.target];
    if (typeof target !== "function") {
      throw new Error(`Unknown differential target: ${request.module}.${request.target}`);
    }
    const result = target(...(request.args ?? []));
    process.stdout.write(JSON.stringify({ ok: true, result }, replacer) + "\n");
  } catch (error) {
    process.stdout.write(
      JSON.stringify({
        ok: false,
        error: error instanceof Error ? error.message : String(error),
      }) + "\n",
    );
  }
}
