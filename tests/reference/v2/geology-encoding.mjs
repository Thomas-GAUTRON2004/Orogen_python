import * as g from "./geology.mjs";
const values = [-1, 0, 0.125, 0.25, 1, 2, 3.25, 8, 16, 32, 1000];
const times = [[0, 100, false], [90, 100, true], [0, 1275, true], [100, 0, true]];
const facies = [[0,0,0,0], [1,.5,.25,0], [.1,.3,.7,1], [2,-1,.5,.5]];
const output = {
  flags: g.GeoFlag, fields: g.GEO_FRAME_FIELDS, reads: g.GEOLOGY_READS,
  cover: values.map(x => [g.encodeCoverKm(x), g.decodeCoverKm(g.encodeCoverKm(x))]),
  exhum: values.map(x => [g.encodeExhumKm(x), g.decodeExhumKm(g.encodeExhumKm(x))]),
  unroof: values.map(x => [g.encodeUnroofKm(x), g.decodeUnroofKm(g.encodeUnroofKm(x))]),
  flood: values.map(x => [g.encodeFloodMyr(x), g.decodeFloodMyr(g.encodeFloodMyr(x))]),
  arc: values.map(x => [g.encodeArcMyr(x), g.decodeArcMyr(g.encodeArcMyr(x))]),
  age: values.map(x => [g.encodeAge(x), g.decodeAge(g.encodeAge(x))]),
  floodAge: times.map(x => [g.encodeFloodAge(...x), g.decodeFloodAge(g.encodeFloodAge(...x))]),
  facies: facies.map(x => [g.encodeFacies(...x), g.decodeFacies(g.encodeFacies(...x))]),
};
console.log(JSON.stringify(output));
