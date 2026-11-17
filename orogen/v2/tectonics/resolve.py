"""Tectonic convergence/divergence resolver ported from Orogen V2 step/resolve.js."""
from __future__ import annotations

import math
import numpy as np

from ..core.mesh_rings import rings_of
from .config import CM_PER_YR_TO_RAD_PER_MYR
from .crust import CrustFlag, CrustType
from .gather import clear_world_view, for_each_claim
from .occupancy import bin_index, mark_bin, may_have_crust_near
from .polarity import overrides
from .segments import fracture_coordinate

REACH_RINGS = 2
AHEAD_CONE = 0.5
PROXIMITY_RINGS = 3

def resolve(world) -> dict[str, int]:
    """Settle convergence and divergence and rebuild a complete world ownership view."""
    world.occupancyStamp += 1
    consumed = _gather_and_subduct(world)
    gaps = np.flatnonzero(world.owner < 0).tolist()
    created = _grow_crust_into_gaps(world, gaps) if gaps else 0
    if gaps:
        _assign_gap_owners(world, gaps)
    world.budget["created"] += created
    world.budget["consumed"] += consumed
    collisions = int(np.count_nonzero(world.landClaims > 1))
    for plate in world.plates:
        if plate.alive and plate.dirty:
            plate.refresh_extent(world.mesh.xyz)
        plate.reshaped = False
    return {"created": created, "consumed": consumed, "collisionCells": collisions}


def _gather_and_subduct(world) -> int:
    owner, owner_cell = world.owner, world.ownerCell
    claim_count, land_claims = world.claimCount, world.landClaims
    consumed_at, collider, plates = world.consumedAt, world.collider, world.plates
    consumed = 0
    bins, stamp = bin_index(world.mesh), world.occupancyStamp
    clear_world_view(world)

    def subduct(plate, cell: int, world_cell: int) -> None:
        nonlocal consumed
        plate.crust.remove(cell)
        plate.dirty = True
        consumed_at[world_cell] = 1
        consumed += 1

    def claim(w: int, plate, cell: int) -> None:
        mark_bin(plate, int(bins.bin_of[w]), stamp, bins.count)
        land = int(plate.crust.type[cell]) != CrustType["OCEAN"]
        holder_id = int(owner[w])
        holder = plates[holder_id] if holder_id >= 0 else None
        if holder is None or not holder.crust.mask[int(owner_cell[w])]:
            owner[w], owner_cell[w] = plate.id, cell
            claim_count[w] = 1
            land_claims[w] = int(land)
            return
        wins = overrides(plate, cell, holder, int(owner_cell[w]), int(world.oceanOverrider[w]))
        if wins is None:
            claim_count[w] += 1
            land_claims[w] += 1
            if collider[w] < 0:
                collider[w] = plate.id
        elif wins:
            subduct(holder, int(owner_cell[w]), w)
            owner[w], owner_cell[w] = plate.id, cell
            if land:
                land_claims[w] = 1
        else:
            subduct(plate, cell, w)

    for_each_claim(world, claim)
    for w in range(len(owner)):
        if owner[w] >= 0 and not plates[int(owner[w])].crust.mask[int(owner_cell[w])]:
            owner[w] = -1
            claim_count[w] = 0
            land_claims[w] = 0
    return consumed


def _grow_crust_into_gaps(world, gaps: list[int]) -> int:
    growers = [p for p in world.plates if p.alive and not p.isBackground and p.crust.count > 0]
    everyone = [(p, p.cap_in_world(world.mesh)) for p in world.plates if p.alive and p.crust.count > 0]
    births = []
    for grower in growers:
        for cell in candidate_cells(world, grower, gaps):
            verdict = decide_birth(world, grower, cell, everyone)
            if verdict["born"]:
                births.append((grower, cell, verdict["trenchParent"]))
    bins, point = bin_index(world.mesh), np.zeros(3)
    xyz = np.asarray(world.mesh.xyz).reshape(-1, 3)
    # Birth decisions are complete; initialize each plate's batch in one call
    # before recording per-cell geometry and trench metadata in original order.
    groups = {}
    for plate, cell, _ in births:
        group = groups.setdefault(plate.id, [plate, []])
        group[1].append(cell)
    for plate, cells in groups.values():
        plate.crust.create_many(cells, CrustType["OCEAN"], world.time)
    for plate, cell, trench_parent in births:
        crust = plate.crust
        point[:] = plate.to_world(*xyz[cell])
        crust.fzU[cell] = _fracture_at(world, plate, cell, point)
        mark_bin(plate, bins.at(*map(float, point)), world.occupancyStamp, bins.count)
        if trench_parent >= 0:
            crust.arm_trench(cell, float(crust.slabKm[trench_parent]))
            crust.flags[cell] |= CrustFlag["APRON"]
        plate.dirty = True
    return len(births)


def _fracture_at(world, grower, cell: int, point) -> float:
    mesh, owner, plates = world.mesh, world.owner, world.plates
    xyz = np.asarray(mesh.xyz).reshape(-1, 3)
    start = int(mesh.index.nearest(*map(float, point)))
    partner, most, ring = -1, 0, [start]
    seen, counts = {start}, {}
    offsets, adjacency = mesh.adj_offset, mesh.adj_list
    for _ in range(8):
        if not ring or partner >= 0:
            break
        following = []
        for q in ring:
            for k in range(int(offsets[q]) - 1, int(offsets[q + 1])):
                r = q if k < offsets[q] else int(adjacency[k])
                other = int(owner[r])
                if other >= 0 and other != grower.id:
                    counts[other] = counts.get(other, 0) + 1
                    if counts[other] > most:
                        most, partner = counts[other], other
                elif other < 0 and r not in seen:
                    seen.add(r)
                    following.append(r)
        ring = following
    other_omega = ([0.0, 0.0, 0.0] if partner < 0 or plates[partner].isBackground
                   else getattr(plates[partner], "omegaSlow", plates[partner].omega))
    omega = getattr(grower, "omegaSlow", grower.omega)
    relative = (np.asarray(omega) - np.asarray(other_omega) if partner < 0 or grower.id < partner
                else np.asarray(other_omega) - np.asarray(omega))
    local = grower.to_plate_frame(*map(float, relative))
    return fracture_coordinate(*map(float, xyz[cell]), local)


def candidate_cells(world, grower, gaps) -> list[int]:
    mesh = world.mesh
    xyz = np.asarray(mesh.xyz).reshape(-1, 3)
    cap, mask = grower.cap_in_world(mesh), grower.crust.mask
    bins, stamp = bin_index(mesh), world.occupancyStamp
    marks = np.zeros(mesh.num_cells, dtype=np.uint32)
    order, ring = [], []
    for gap in gaps:
        x, y, z = xyz[gap]
        if x * cap[0] + y * cap[1] + z * cap[2] < cap[3]:
            continue
        if not may_have_crust_near(grower, int(bins.bin_of[gap]), stamp, bins):
            continue
        cell = grower.cell_at(mesh.index, float(x), float(y), float(z))
        if marks[cell] == 0:
            marks[cell] = 1
            order.append(cell)
            ring.append(cell)
    offsets, adjacency = mesh.adj_offset, mesh.adj_list
    for _ in range(REACH_RINGS):
        following = []
        for cell in ring:
            for k in range(int(offsets[cell]), int(offsets[cell + 1])):
                nb = int(adjacency[k])
                if marks[nb] == 0:
                    marks[nb] = 1
                    order.append(nb)
                    following.append(nb)
        ring = following
    return [cell for cell in order if not mask[cell]]


def decide_birth(world, grower, cell: int, everyone) -> dict:
    mesh, config = world.mesh, world.config
    xyz = np.asarray(mesh.xyz).reshape(-1, 3)
    point = grower.to_world(*map(float, xyz[cell]))
    own = _nearest_crust(grower, mesh, point)
    if own["cell"] < 0:
        return {"born": False, "why": "not attached to this plate's crust"}
    bins = bin_index(mesh)
    bin_id, stamp = bins.at(*map(float, point)), world.occupancyStamp
    rival_dot, neighbour, neighbour_hit = -2.0, None, None
    for plate, cap in everyone:
        if plate is grower or np.dot(point, cap[:3]) < cap[3]:
            continue
        if not may_have_crust_near(plate, bin_id, stamp, bins):
            continue
        under = plate.cell_at(mesh.index, *map(float, point))
        if plate.crust.mask[under]:
            return {"born": False, "why": "not a gap: another plate has crust here"}
        hit = _nearest_crust(plate, mesh, point, under)
        if hit["cell"] < 0:
            continue
        if not plate.isBackground and hit["dot"] > rival_dot:
            rival_dot = hit["dot"]
        if neighbour_hit is None or hit["dot"] > neighbour_hit["dot"]:
            neighbour_hit, neighbour = hit, plate
    if rival_dot > -2 and math.acos(min(1.0, rival_dot)) - math.acos(min(1.0, own["dot"])) < mesh.edge_rad:
        return {"born": False, "why": "too close to the mid-line"}
    closing = float(config.get("closingCmYr", config.get("closing_cm_yr", 0.0)))
    if neighbour is not None and _is_advancing_on(grower, neighbour, neighbour_hit["cell"], mesh, point, closing):
        return {"born": False, "why": "closing boundary"}
    flags = int(grower.crust.flags[own["cell"]])
    parent_is_trench = bool(flags & CrustFlag["TRENCH"]) and int(grower.crust.type[own["cell"]]) == CrustType["OCEAN"]
    if parent_is_trench and int(grower.crust.wingKm[own["cell"]]) != 0:
        parent_is_trench = not _is_behind(grower, own["cell"], cell, mesh)
    return {"born": True, "trenchParent": own["cell"] if parent_is_trench else -1}


def _assign_gap_owners(world, gaps) -> None:
    mesh, owner, owner_cell = world.mesh, world.owner, world.ownerCell
    xyz = np.asarray(mesh.xyz).reshape(-1, 3)
    everyone = [p for p in world.plates if p.alive and p.crust.count > 0]
    bins, stamp = bin_index(mesh), world.occupancyStamp
    for gap in gaps:
        point = xyz[gap]
        bin_id = int(bins.bin_of[gap])
        near = [p for p in everyone if may_have_crust_near(p, bin_id, stamp, bins)]
        best = best_hit = foreign = foreign_hit = None
        under = []
        for plate in near:
            cell = plate.cell_at(mesh.index, *map(float, point))
            under.append(cell)
            if plate.crust.mask[cell]:
                best, best_hit = plate, {"dot": 1.0, "cell": cell, "real": True}
                break
        if best is None:
            hits = [_nearest_crust(p, mesh, point, under[i]) for i, p in enumerate(near)]
            for plate, hit in zip(near, hits):
                if hit["cell"] >= 0 and not plate.isBackground and (best_hit is None or hit["dot"] > best_hit["dot"]):
                    best, best_hit = plate, hit
            for plate, hit in zip(near, hits):
                if plate is best:
                    continue
                if hit["cell"] >= 0 and (foreign_hit is None or hit["dot"] > foreign_hit["dot"]):
                    foreign, foreign_hit = plate, hit
        if best is None:
            continue
        owner[gap], owner_cell[gap] = best.id, best_hit["cell"]
        world.claimCount[gap] = 1
        if best_hit.get("real"):
            continue
        closing = float(world.config.get("closingCmYr", world.config.get("closing_cm_yr", 0.0)))
        is_closing = foreign is not None and _is_advancing_on(best, foreign, foreign_hit["cell"], mesh, point, closing)
        world.provisional[gap] = 2 if is_closing else 1
        world.provisionalLike[gap] = best_hit["cell"]


def _is_advancing_on(x, y, foreign_cell: int, mesh, point, closing_cm_yr: float) -> bool:
    ox, oy, oz = (np.asarray(x.omega) - np.asarray(y.omega))
    px, py, pz = map(float, point)
    velocity = np.array([oy * pz - oz * py, oz * px - ox * pz, ox * py - oy * px])
    speed = float(np.linalg.norm(velocity))
    if speed <= closing_cm_yr * CM_PER_YR_TO_RAD_PER_MYR:
        return False
    xyz = np.asarray(mesh.xyz).reshape(-1, 3)
    q = y.to_world(*map(float, xyz[foreign_cell]))
    delta = q - np.asarray(point)
    distance = float(np.linalg.norm(delta))
    if distance < 1e-9:
        return True
    return float(np.dot(delta, velocity)) / (distance * speed) > AHEAD_CONE


def _is_behind(plate, edge_cell: int, cell: int, mesh) -> bool:
    xyz = np.asarray(mesh.xyz).reshape(-1, 3)
    mask = plate.crust.mask
    offsets, adjacency = mesh.adj_offset, mesh.adj_list
    normal = np.zeros(3)
    opened = 0
    for k in range(int(offsets[edge_cell]), int(offsets[edge_cell + 1])):
        nb = int(adjacency[k])
        if mask[nb]:
            continue
        opened += 1
        normal += xyz[nb] - xyz[edge_cell]
    if not opened:
        return False
    delta = xyz[cell] - xyz[edge_cell]
    denominator = float(np.linalg.norm(delta) * np.linalg.norm(normal)) or 1.0
    return float(np.dot(delta, normal)) / denominator < -0.35


def _nearest_crust(plate, mesh, point, under: int = -1) -> dict:
    xyz = np.asarray(mesh.xyz).reshape(-1, 3)
    local = plate.to_plate_frame(*map(float, point))
    rings = rings_of(mesh)
    base_cell = under if under >= 0 else int(mesh.index.nearest(*map(float, local)))
    base = base_cell * rings["stride"]
    best, best_cell = -2.0, -1
    for radius in range(PROXIMITY_RINGS + 1):
        start = int(rings["start"][base + radius])
        end = int(rings["start"][base + radius + 1])
        for pos in range(start, end):
            candidate = int(rings["cells"][pos])
            if plate.crust.mask[candidate]:
                score = float(np.dot(local, xyz[candidate]))
                if score > best:
                    best, best_cell = score, candidate
        if best_cell >= 0 and radius >= 1:
            break
    return {"dot": best, "cell": best_cell}
