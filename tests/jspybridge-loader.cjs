// JSPyBridge loads local modules through dynamic import(). Keep its entry point
// CommonJS and use Node 22's synchronous require(ESM) bridge for the pinned
// reference modules. The upstream module graph has no top-level await.
//
// invoke() takes and returns JSON text so results cross the bridge as plain
// strings, matching the Node runner's wire format.
const modules = {
  seed: require("./reference/v2/core/seed.mjs"),
  rotation: require("./reference/v2/core/rotation.mjs"),
};

function jsName(snake) {
  return snake.replace(/_([a-z0-9])/g, (_, c) => c.toUpperCase());
}

function replacer(_key, value) {
  if (typeof value === "number" && !Number.isFinite(value)) {
    return { __number__: String(value) };
  }
  return value;
}

exports.invoke = function invoke(moduleId, targetId, argsJson) {
  const namespace = modules[moduleId];
  if (namespace === undefined) {
    throw new Error(`Unknown differential module: ${moduleId}`);
  }
  const target = namespace[jsName(targetId)];
  if (typeof target !== "function") {
    throw new Error(`Unknown differential target: ${moduleId}.${targetId}`);
  }
  const result = target(...JSON.parse(argsJson));
  return JSON.stringify(result, replacer);
};
