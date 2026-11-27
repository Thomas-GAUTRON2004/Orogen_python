// JSPyBridge loads local modules through require(). Modules are read by path
// from tests/reference/v2, mirroring property-runner.mjs. The
// upstream module graph has no top-level await, so require(ESM) is synchronous.
//
// invoke() takes and returns JSON text so results cross the bridge as plain
// strings, matching the Node runner's wire format.
const path = require("node:path");

const REFERENCE_DIR = path.join(__dirname, "reference", "v2");
const FILE_PATTERN = /^[A-Za-z0-9_-]+(\/[A-Za-z0-9_-]+)*\.mjs$/;
const cache = new Map();

function loadModule(file) {
  if (!FILE_PATTERN.test(file)) {
    throw new Error(`Invalid differential module: ${file}`);
  }
  if (!cache.has(file)) {
    cache.set(file, require(path.join(REFERENCE_DIR, file)));
  }
  return cache.get(file);
}

function replacer(_key, value) {
  if (typeof value === "number" && !Number.isFinite(value)) {
    return { __number__: String(value) };
  }
  return value;
}

exports.invoke = function invoke(file, exportName, argsJson, constructorJson, method) {
  const namespace = loadModule(file);
  const target = namespace[exportName];
  if (typeof target !== "function") {
    throw new Error(`Unknown differential target: ${file}.${exportName}`);
  }
  let result;
  if (constructorJson) {
    // A method call: build the instance with `new`, then call the method on it.
    const instance = new target(...JSON.parse(constructorJson));
    const member = instance[method];
    if (typeof member !== "function") {
      throw new Error(`Unknown differential method: ${exportName}.${method}`);
    }
    result = member.apply(instance, JSON.parse(argsJson));
  } else {
    result = target(...JSON.parse(argsJson));
  }
  return JSON.stringify(result, replacer);
};
