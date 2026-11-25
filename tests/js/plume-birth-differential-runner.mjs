import { readFileSync } from "node:fs";

import { updateHotspots } from "../../external/planet_heightmap_generation/v2/js/tectonics/step/hotspots.js";

const input = JSON.parse(readFileSync(0, "utf8"));

function run(spec) {
  const mesh = {
    numCells: 1,
    xyz: new Float64Array(spec.xyz),
    index: { nearest: () => 0 },
    adjOffset: new Uint32Array([0, 0]),
    adjList: new Int32Array(0),
    cellAreaKm2: spec.cellAreaKm2,
  };
  const plates = spec.plate
    ? [{
        id: 0,
        name: spec.plate.name,
        isBackground: false,
        stats: { continentalCells: spec.plate.continentalCells },
        crust: {
          mask: new Uint8Array(spec.plate.mask),
          type: new Uint8Array(spec.plate.type),
          flags: new Uint16Array([0]),
          volcType: new Uint8Array([0]),
          volcAt: new Float64Array([0]),
          volcKm: new Float32Array([0]),
          volcSource: new Uint16Array([0]),
        },
        cellAt() { return 0; },
        toPlateFrame(x, y, z) { return [x, y, z]; },
      }]
    : [];
  const draws = [...spec.draws];
  let drawIndex = 0;
  const events = [];
  const scheduled = {
    time: spec.time,
    xyz: spec.xyz,
    head: spec.head ?? false,
    rift: spec.rift,
    name: spec.name,
    lifeMyr: spec.lifeMyr ?? 0,
    strength: 1.2,
  };
  const world = {
    config: {
      mantleInsulateMyr: 100,
      mantleCoolMyr: 200,
      plumeRatePerMyr: 0,
      riftMinContinentKm2: spec.riftMinContinentKm2,
      plumeRiftChance: spec.plumeRiftChance,
      riftHeatGain: spec.riftHeatGain,
      riftHeatFloor: spec.riftHeatFloor,
      riftHeatPower: spec.riftHeatPower,
      lipAreaMinMkm2: spec.lipAreaMinMkm2,
      lipAreaMaxMkm2: spec.lipAreaMaxMkm2,
      lipLobes: spec.lipLobes,
      lipEdgeNoise: spec.lipEdgeNoise,
      lipThicknessKm: spec.lipThicknessKm,
      hotspotTrackKm: 1000,
      hotspotEdificeKm: 100,
      hotspotBuildKmPerMyr: 1,
      hotspotMaxKm: 50,
      hotspotQuietMinMyr: 1,
      hotspotQuietMaxMyr: 4,
      hotspotBurstMinMyr: 1,
      hotspotBurstMaxMyr: 3,
    },
    time: spec.time,
    mesh,
    owner: new Int16Array([spec.owner]),
    ownerCell: new Int32Array([0]),
    plates,
    mantleHeat: new Float32Array([spec.heat]),
    provisional: new Uint8Array([0]),
    rngVolcano: () => 0.5,
    hotspotsReady: true,
    hotspots: [],
    pendingPlumeRifts: [],
    plumes: [scheduled],
    rngHotspot: () => draws[drawIndex++] ?? 0,
    log(kind, text, data) {
      events.push({ kind, text, data });
    },
  };
  updateHotspots(world, spec.dt ?? 0);
  return {
    summary: {
      born: world.hotspots.length,
      expired: events.filter((event) => event.kind === "plume-death").length,
      alive: world.hotspots.filter((plume) => plume.alive).length,
    },
    hotspots: world.hotspots.map((plume) => ({
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
    })),
    pendingPlumeRifts: world.pendingPlumeRifts,
    scheduledDone: scheduled.done,
    heat: Array.from(world.mantleHeat),
    crust: world.plates.map((plate) => ({
      volcType: Array.from(plate.crust.volcType),
      volcAt: Array.from(plate.crust.volcAt),
      volcKm: Array.from(plate.crust.volcKm),
      volcSource: Array.from(plate.crust.volcSource),
      flags: Array.from(plate.crust.flags),
    })),
    events,
  };
}

console.log(JSON.stringify(input.cases.map(run)));
