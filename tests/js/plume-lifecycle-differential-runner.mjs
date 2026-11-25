import { readFileSync } from "node:fs";

import { updateHotspots } from "../../external/planet_heightmap_generation/v2/js/tectonics/step/hotspots.js";
import { EventKind } from "../../external/planet_heightmap_generation/v2/js/tectonics/model/world.js";

const input = JSON.parse(readFileSync(0, "utf8"));
const spec = input.scenario;
const events = [];
const world = {
  config: {
    mantleInsulateMyr: 100,
    mantleCoolMyr: 200,
    plumeRatePerMyr: 0,
  },
  time: spec.time,
  mesh: {
    numCells: spec.xyz.length / 3,
    xyz: new Float64Array(spec.xyz),
    index: { nearest: () => 0 },
  },
  owner: new Int16Array(spec.xyz.length / 3).fill(-1),
  ownerCell: new Int32Array(spec.xyz.length / 3),
  plates: [],
  mantleHeat: new Float32Array(spec.heat),
  hotspotsReady: true,
  hotspots: [],
  pendingPlumeRifts: [],
  plumes: [{
    time: spec.time,
    xyz: spec.plume.xyz,
    head: false,
    rift: false,
    name: spec.plume.name,
    lifeMyr: spec.plume.lifeMyr,
    strength: spec.plume.strength,
  }],
  rngHotspot: () => 0,
  log(kind, text, data) {
    events.push({ kind, text, data });
  },
};
updateHotspots(world, 0);
const summary = {
  born: world.hotspots.length,
  expired: events.filter(event => event.kind === EventKind.PLUME_DEATH).length,
  alive: world.hotspots.filter(plume => plume.alive).length,
};
const hotspots = world.hotspots.map(plume => ({
  id: plume.id,
  name: plume.name,
  xyz: plume.xyz,
  bornAt: plume.bornAt,
  dieAt: plume.dieAt,
  alive: plume.alive,
  lip: plume.lip,
  strength: plume.strength,
  pulseUntil: plume.pulseUntil ?? null,
  bursting: plume.bursting,
  pulseIntensity: plume.pulseIntensity,
}));
console.log(JSON.stringify({
  summary,
  hotspots,
  pendingPlumeRifts: world.pendingPlumeRifts,
  scheduledDone: world.plumes[0].done,
  heat: Array.from(world.mantleHeat),
  events,
}));
