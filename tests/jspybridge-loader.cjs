// JSPyBridge loads local modules through dynamic import(). Keep its entry point
// CommonJS and use Node 22's synchronous require(ESM) bridge for the pinned
// reference module. The upstream module graph has no top-level await.
const { seedOf, isWordSeed } = require("./reference/v2/core/seed.mjs");

exports.seedOf = seedOf;
exports.isWordSeed = isWordSeed;
