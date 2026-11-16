import Delaunator from "delaunator";
import { setDelaunator, buildSphere, generateFibonacciSphere, stereographicProjection } from "./js/sphere-mesh.mjs";
import { cellDistanceKm as geomCellDistance, centroidOfCells as geomCentroid } from "./core/sphere-geom.mjs";
import { makeRng } from "./js/rng.mjs";
import { SphereIndex } from "./core/sphere-index.mjs";
import { hopDistance, withinKm } from "./core/mesh-bfs.mjs";
import { spreadKm } from "./core/min-heap.mjs";
import { Plate } from "./tectonics/plate.mjs";
setDelaunator(Delaunator);
const cases=[];
for(const [n,seed] of [[24,1],[48,17],[96,2026]]){
  const xyz0=generateFibonacciSphere(n,.35,makeRng(seed+4242));
  const projected=stereographicProjection(xyz0,n);
  const {mesh, r_xyz}=buildSphere(n,.35,makeRng(seed+4242));
  const index=new SphereIndex(r_xyz,mesh,2);
  const edgeRad=Math.sqrt(8*Math.PI/(Math.sqrt(3)*mesh.numRegions));
  const edgeKm=edgeRad*6371;
  const sim={numCells:mesh.numRegions,xyz:r_xyz,adjOffset:mesh.adjOffset,adjList:mesh.adjList,edgeRad,edgeKm};
  const distances=new Float64Array(mesh.numRegions).fill(Infinity);distances[0]=0;
  spreadKm(sim,distances,{maxKm:edgeKm*3.5});
  const within=withinKm(sim,c=>c===0,edgeKm*2.25);
  const hops=hopDistance(sim,c=>c===0,3);
  const plate=new Plate(0,mesh.numRegions);
  for(const cell of [0,1,2,4,7]) if(cell<mesh.numRegions) plate.crust.create(cell,cell===0?2:0,-1,cell+1);
  plate.refreshExtent(r_xyz);
  const queries=[[1,.2,.3],[-.3,1,.4],[.2,-.4,1],[-1,-.3,-.2],[.2,.3,-1],[1,1,1]];
  const triangleKeys=[];
  for(let i=0;i<mesh.triangles.length;i+=3){
    triangleKeys.push([mesh.triangles[i],mesh.triangles[i+1],mesh.triangles[i+2]].sort((a,b)=>a-b).join(","));
  }
  const neighbours=[];
  for(let c=0;c<mesh.numRegions;c++){
    neighbours.push(Array.from(mesh.adjList.slice(mesh.adjOffset[c],mesh.adjOffset[c+1])).sort((a,b)=>a-b));
  }
  cases.push({
    n,seed,xyz:Array.from(r_xyz),projected:Array.from(projected),
    numRegions:mesh.numRegions,numTriangles:mesh.numTriangles,
    triangleKeys:triangleKeys.sort(),
    adjOffset:Array.from(mesh.adjOffset),neighbours,
    edgeRad,edgeKm,cellDistance:geomCellDistance(r_xyz,0,2),centroid:geomCentroid(r_xyz,[0,2,4]),distance:Array.from(distances,v=>Number.isFinite(v)?v:"Infinity"),within:Array.from(within),hops:Array.from(hops),plate:{stats:plate.stats,capCenter:plate.capCenter,capCos:plate.capCos},
    indexRes:index.res,
    nearest:queries.map(p=>index.nearest(...p))
  });
}
console.log(JSON.stringify(cases));
