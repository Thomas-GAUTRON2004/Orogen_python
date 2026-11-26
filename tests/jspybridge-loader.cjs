// JSPyBridge loads local modules through dynamic import(). Keep its entry point
// CommonJS and use Node 22's synchronous require(ESM) bridge for the pinned
// reference modules. The upstream module graph has no top-level await.
// Each module namespace is exposed under its name so Python can resolve
// targets automatically, mirroring the static allowlist in test_generated.py.
exports.seed = require("./reference/v2/core/seed.mjs");
