// Dumps the upstream TECTONICS defaults from the pinned submodule (ed466f9).
// Usage: node dump_config.mjs
import { fileURLToPath, pathToFileURL } from 'node:url';
import path from 'node:path';
const UP = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '../../../../third_party/planet_heightmap_generation');
const { TECTONICS } = await import(pathToFileURL(path.join(UP, 'v2/js/tectonics/config.js')).href);
console.log(JSON.stringify(TECTONICS));
