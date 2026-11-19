"""Deterministic initial-world construction for the V2 tectonic model.

The upstream V2 generator is a coupled geological layout search. This Python
port keeps its public state contract and seeded stream separation while using
mesh-native spherical caps and graph distances for a reproducible, resolution-
aware construction. It is an explicit numerical adaptation, not a claim of
bit-for-bit geometric parity with the JavaScript layout search.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from ..core.mesh_bfs import hop_distance
from ..core.seed import make_rng
from ..core.sphere_geometry import random_point, tangent_basis
from .crust import CrustFlag, CrustType, OrogenyType
from .world import EventKind, TectonicWorld

StartSeam = {"NONE": 0, "INNER": 1, "RING": 2, "MAJOR": 3}


def _mesh_xyz(sim_mesh: Any) -> np.ndarray:
    xyz = np.asarray(sim_mesh.xyz, dtype=np.float64).reshape((-1, 3))
    if xyz.shape[0] != int(sim_mesh.num_cells):
        raise ValueError("mesh xyz does not match num_cells")
    return xyz


def _distance_hops(sim_mesh: Any, sources: np.ndarray, *, can_enter=None, max_hops=None) -> np.ndarray:
    source_mask = np.asarray(sources, dtype=bool)
    if source_mask.shape != (int(sim_mesh.num_cells),):
        raise ValueError("source mask must contain one value per mesh cell")
    if not np.any(source_mask):
        return np.full(int(sim_mesh.num_cells), -1, dtype=np.int16)
    return hop_distance(
        sim_mesh,
        lambda cell: bool(source_mask[cell]),
        max_hops=int(max_hops if max_hops is not None else sim_mesh.num_cells),
        can_enter=can_enter,
    )


def _land_cells(sim_mesh: Any, centre: np.ndarray, count: int) -> tuple[np.ndarray, float]:
    xyz = _mesh_xyz(sim_mesh)
    count = max(1, min(int(count), int(sim_mesh.num_cells) - 1))
    cosine = xyz @ centre
    cells = np.argpartition(cosine, -count)[-count:]
    cells = cells[np.argsort(-cosine[cells], kind="stable")]
    radius = math.acos(float(np.clip(np.min(cosine[cells]), -1.0, 1.0)))
    return cells.astype(np.int32), radius


def _pick_craton_centres(
    xyz: np.ndarray,
    land_cells: np.ndarray,
    centre: np.ndarray,
    land_radius: float,
    count: int,
    minimum_gap_km: float,
    edge_km: float,
    rng,
) -> np.ndarray:
    """Pick well-separated mesh cells inside the continental core."""
    cosine = xyz[land_cells] @ centre
    core_radius = max(edge_km / 6371.0 * 2.0, land_radius * 0.78)
    core = land_cells[cosine >= math.cos(min(core_radius, math.pi))]
    if core.size < count:
        core = land_cells
    random_order = np.argsort(np.fromiter((rng() for _ in range(core.size)), dtype=np.float64))
    candidates = core[random_order]
    chosen = [int(candidates[0])]
    minimum_angle = max(float(minimum_gap_km), 1.5 * float(edge_km)) / 6371.0
    threshold = minimum_angle
    while len(chosen) < count:
        dots = np.clip(xyz[candidates] @ xyz[np.asarray(chosen)].T, -1.0, 1.0)
        separation = np.min(np.arccos(dots), axis=1)
        separation[np.isin(candidates, chosen)] = -1.0
        best_pos = int(np.argmax(separation))
        best_distance = float(separation[best_pos])
        if best_distance < threshold and threshold > minimum_angle * 0.2:
            threshold *= 0.75
            continue
        chosen.append(int(candidates[best_pos]))
        if best_distance <= 0:
            break
    if len(chosen) < count:
        for cell in candidates:
            if int(cell) not in chosen:
                chosen.append(int(cell))
                if len(chosen) == count:
                    break
    return np.asarray(chosen[:count], dtype=np.int32)


def _assign_cratons(
    xyz: np.ndarray,
    land_cells: np.ndarray,
    centres: np.ndarray,
    requested_share: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return a land-cell terrane map, craton mask and craton ownership."""
    centre_xyz = xyz[centres]
    cosine = xyz[land_cells] @ centre_xyz.T
    owner = np.argmax(cosine, axis=1).astype(np.int32)
    target = int(round(land_cells.size * max(0.0, min(1.0, requested_share))))
    target = max(min(len(centres), land_cells.size), min(target, land_cells.size))
    nearest = np.max(cosine, axis=1)
    order = np.argsort(-nearest, kind="stable")
    selected = np.zeros(land_cells.size, dtype=bool)
    selected[order[:target]] = True
    cell_to_local = {int(cell): pos for pos, cell in enumerate(land_cells)}
    for centre in centres:
        selected[cell_to_local[int(centre)]] = True
    return owner + 1, selected, owner


def _seams(
    sim_mesh: Any,
    land_mask: np.ndarray,
    land_cells: np.ndarray,
    terrane_owner: np.ndarray,
    centre: np.ndarray,
    land_radius: float,
    group_for_terrane: np.ndarray,
) -> np.ndarray:
    """Mark narrow terrane-boundary seams and an accreted outer-ring seam."""
    n = int(sim_mesh.num_cells)
    result = np.zeros(n, dtype=np.uint8)
    offsets, adjacency = sim_mesh.adj_offset, sim_mesh.adj_list
    owner = np.full(n, -1, dtype=np.int32)
    owner[land_cells] = terrane_owner
    groups = np.full(n, -1, dtype=np.int32)
    groups[land_cells] = group_for_terrane[terrane_owner - 1]
    for cell in land_cells:
        lo, hi = int(offsets[cell]), int(offsets[cell + 1])
        for neighbour in adjacency[lo:hi]:
            other = int(neighbour)
            if not land_mask[other] or owner[other] == owner[cell]:
                continue
            kind = StartSeam["MAJOR"] if groups[other] != groups[cell] else StartSeam["INNER"]
            if kind > result[cell]:
                result[cell] = kind
    xyz = _mesh_xyz(sim_mesh)
    radial = np.arccos(np.clip(xyz[land_cells] @ centre, -1.0, 1.0))
    ring_radius = land_radius * 0.72
    ring_width = max(float(sim_mesh.edge_rad) * 1.25, land_radius * 0.025)
    ring_cells = land_cells[np.abs(radial - ring_radius) <= ring_width]
    unset = result[ring_cells] == StartSeam["NONE"]
    result[ring_cells[unset]] = StartSeam["RING"]
    return result


def _paint_old_ranges(world: TectonicWorld, plate, land_mask: np.ndarray, seams: np.ndarray) -> None:
    """Seed worn pre-history belts along the three seam classes."""
    cfg = world.config
    ranges = cfg.get("startRanges", {})
    max_hops = max(1, int(math.ceil(3000.0 / max(float(world.mesh.edge_km), 1.0))))
    for kind_name in ("inner", "ring", "major"):
        kind = StartSeam[kind_name.upper()]
        settings = ranges.get(kind_name) if isinstance(ranges, dict) else None
        if not settings or len(settings) < 4:
            continue
        width_km, work_km, age_myr, keep = (float(value) for value in settings[:4])
        if work_km <= 0 or width_km <= 0:
            continue
        sources = seams == kind
        if not np.any(sources):
            continue
        hops = _distance_hops(
            world.mesh,
            sources,
            can_enter=lambda cell: bool(land_mask[cell]),
            max_hops=max_hops,
        )
        reached = (hops >= 0) & land_mask & (hops.astype(np.float64) * world.mesh.edge_km <= 2.0 * width_km)
        cells = np.flatnonzero(reached)
        if not cells.size:
            continue
        distance = hops[cells].astype(np.float64) * world.mesh.edge_km
        profile = np.clip(1.0 - distance / (2.0 * width_km), 0.0, 1.0)
        strength = keep * profile
        crust = plate.crust
        crust.oroType[cells] = OrogenyType["URAL"]
        crust.oroWork[cells] = (work_km * profile).astype(np.float32)
        crust.oroStart[cells] = -age_myr
        crust.oroLast[cells] = -age_myr
        crust.thickKm[cells] += (cfg.get("thickeningPerWorkKm", 0.05) * work_km * profile).astype(np.float32)
        crust.rootKm[cells] += (0.9 * strength).astype(np.float32)
        crust.upliftKm[cells] += (0.25 * strength).astype(np.float32)
        crust.rugged[cells] += (0.15 * strength).astype(np.float32)


def build_start_world(sim_mesh: Any, seed: int, overrides: dict | None = None) -> dict[str, Any]:
    """Build a deterministic supercontinent and passive pre-history ocean.

    The return shape follows upstream buildStartWorld. The continental cell
    fraction is exact at mesh resolution; craton count is reduced on coarse
    meshes so every core remains represented by cells.
    """
    world = TectonicWorld(sim_mesh, seed, overrides=overrides)
    cfg = world.config
    fraction = float(cfg.get("continentalFraction", 0.29))
    if not math.isfinite(fraction) or not 0.0 < fraction < 1.0:
        raise ValueError("continentalFraction must be finite and strictly between 0 and 1")
    n = int(sim_mesh.num_cells)
    if n < 4:
        raise ValueError("start-world generation requires at least four mesh cells")
    xyz = _mesh_xyz(sim_mesh)

    # Match the upstream attempt stream offsets: layout and passive margin are
    # independent so draws made for one mechanism cannot perturb the other.
    layout_rng = make_rng(seed * 31 + 17)
    passive_rng = make_rng(seed * 31 + 29)
    centre = np.asarray(random_point(layout_rng), dtype=np.float64)
    target_land = max(1, min(n - 1, int(math.floor(n * fraction + 0.5))))
    land_cells, land_radius = _land_cells(sim_mesh, centre, target_land)
    land_mask = np.zeros(n, dtype=bool)
    land_mask[land_cells] = True

    share = float(cfg.get("cratonShare", 0.24))
    target_cratons = max(1, int(round(target_land * max(0.0, min(1.0, share)))))
    requested = int(cfg.get("cratonCount", 14))
    craton_count = min(max(1, requested), max(1, target_land // 12), target_cratons)
    centres = _pick_craton_centres(
        xyz, land_cells, centre, land_radius, craton_count,
        float(cfg.get("cratonMinGapKm", 500.0)), float(sim_mesh.edge_km), layout_rng,
    )
    terrane_owner, core_mask, local_owner = _assign_cratons(xyz, land_cells, centres, share)
    terrane_for_cell = np.zeros(n, dtype=np.uint16)
    terrane_for_cell[land_cells] = terrane_owner.astype(np.uint16)
    craton_mask = np.zeros(n, dtype=bool)
    craton_mask[land_cells[core_mask]] = True

    group_count = min(4, max(1, int(round(math.sqrt(craton_count)))))
    group_for_terrane = np.floor(
        np.arange(craton_count, dtype=np.float64) * group_count / craton_count
    ).astype(np.int32)
    seams = _seams(
        sim_mesh, land_mask, land_cells, terrane_owner,
        centre, land_radius, group_for_terrane,
    )

    # Expand the continent's plate a short distance offshore. Distances are
    # graph hops on the same mesh used by the simulation.
    land_hops = _distance_hops(sim_mesh, land_mask)
    apron_km = max(0.0, float(cfg.get("apronKm", 250.0)))
    apron_hops = 0 if apron_km <= 0 else max(
        1, int(math.floor(apron_km / max(float(sim_mesh.edge_km), 1.0) + 0.5))
    )
    apron_mask = (~land_mask) & (land_hops >= 1) & (land_hops <= apron_hops)
    in_plate = land_mask | apron_mask

    continent = world.add_plate("".join(chr(65 + i) for i in range(craton_count)))
    ocean = world.add_plate("Pre-history ocean")
    ocean.isBackground = True
    continent.born = -float(cfg.get("riftTenureMyr", 30.0))

    land_types = np.full(land_cells.size, CrustType["CONTINENT"], dtype=np.uint8)
    land_types[core_mask] = CrustType["CRATON"]
    continent.crust.create_many(
        land_cells, land_types,
        -float(cfg.get("initialLandAgeMyr", 2000.0)),
        terrane_owner.astype(np.uint16),
    )
    apron_cells = np.flatnonzero(apron_mask)
    if apron_cells.size:
        continent.crust.create_many(
            apron_cells, CrustType["OCEAN"],
            -float(cfg.get("initialOceanAgeMyr", 150.0)),
        )
        continent.crust.flags[apron_cells] |= CrustFlag["APRON"]

    # One passive sector is left without a trench; the remaining rim starts
    # with a mature slab, matching the initial state consumed by stepper.py.
    east, north = tangent_basis(centre)
    rim_cells: list[int] = []
    offsets, adjacency = sim_mesh.adj_offset, sim_mesh.adj_list
    for cell in apron_cells:
        lo, hi = int(offsets[cell]), int(offsets[cell + 1])
        if any(not in_plate[int(other)] for other in adjacency[lo:hi]):
            rim_cells.append(int(cell))
    coverage = max(0.0, min(1.0, float(cfg.get("rimTrenchCoverage", 0.85))))
    passive_from = passive_rng() * 2.0 * math.pi
    passive_span = (1.0 - coverage) * 2.0 * math.pi
    if rim_cells and coverage > 0:
        rim = np.asarray(rim_cells, dtype=np.int32)
        angles = np.arctan2(xyz[rim] @ north, xyz[rim] @ east) % (2.0 * math.pi)
        passive = ((angles - passive_from) % (2.0 * math.pi)) < passive_span
        trench_cells = rim[~passive]
        if not trench_cells.size and coverage >= 0.5:
            trench_cells = rim
        continent.crust.flags[trench_cells] |= CrustFlag["TRENCH"]
        continent.crust.slabKm[trench_cells] = np.maximum(
            continent.crust.slabKm[trench_cells],
            float(cfg.get("slabMatureKm", 600.0)),
        )
    else:
        trench_cells = np.empty(0, dtype=np.int32)

    ocean_cells = np.flatnonzero(~in_plate)
    ocean.crust.create_many(
        ocean_cells, CrustType["OCEAN"],
        -float(cfg.get("initialOceanAgeMyr", 150.0)),
    )

    # Inner and major boundaries are weak sutures. Ring seams are accreted
    # belts and therefore remain stronger rift corridors.
    weak = (seams == StartSeam["INNER"]) | (seams == StartSeam["MAJOR"])
    continent.crust.flags[weak] |= CrustFlag["SUTURE"]
    _paint_old_ranges(world, continent, land_mask, seams)

    major_sources = seams == StartSeam["MAJOR"]
    ring_sources = seams == StartSeam["RING"]
    hop_limit = max(1, int(math.ceil(3000.0 / max(float(sim_mesh.edge_km), 1.0))))
    major_hops = _distance_hops(
        sim_mesh, major_sources,
        can_enter=lambda cell: bool(land_mask[cell]), max_hops=hop_limit,
    )
    ring_hops = _distance_hops(
        sim_mesh, ring_sources,
        can_enter=lambda cell: bool(land_mask[cell]), max_hops=hop_limit,
    )
    major_km = np.full(n, np.inf, dtype=np.float32)
    ring_km = np.full(n, np.inf, dtype=np.float32)
    major_valid, ring_valid = major_hops >= 0, ring_hops >= 0
    major_km[major_valid] = major_hops[major_valid].astype(np.float32) * sim_mesh.edge_km
    ring_km[ring_valid] = ring_hops[ring_valid].astype(np.float32) * sim_mesh.edge_km
    major_range = np.zeros(n, dtype=np.float32)
    major_range[major_sources] = 1.0
    world.startSeamKm = {"major": major_km, "majorRange": major_range, "ring": ring_km}

    for plate in world.plates:
        plate.refresh_extent(sim_mesh.xyz)
    world.terraneLetters = {index + 1: chr(65 + index) for index in range(craton_count)}
    world.log(
        EventKind["START"],
        f"Supercontinent {continent.name}: {craton_count} cratons, ringed by subduction",
        {"plate": continent.id, "cratons": craton_count},
    )

    cratons = []
    for index, cell in enumerate(centres):
        held = land_cells[terrane_owner == index + 1]
        cratons.append({
            "id": index + 1,
            "letter": chr(65 + index),
            "centreCell": int(cell),
            "centre": xyz[int(cell)].tolist(),
            "cells": int(np.count_nonzero(craton_mask & (terrane_for_cell == index + 1))),
            "terraneCells": int(held.size),
            "cluster": int(group_for_terrane[index]),
        })
    start = {
        "cratons": cratons,
        "centre": centre.tolist(),
        "rimCentre": centre.tolist(),
        "reference": east,
        "elongation": 1.0,
        "seams": seams,
        "owner": terrane_for_cell,
        "skirtKm": apron_km,
        "rim": np.asarray(rim_cells, dtype=np.int32),
        "trench": trench_cells,
        "passiveFrom": passive_from,
        "passiveTo": (passive_from + passive_span) % (2.0 * math.pi),
        "attempts": 1,
        "cratonAreaScale": 1.0,
        "fractionMet": True,
        "config": cfg,
        "continentId": continent.id,
        "oceanId": ocean.id,
        "landCells": target_land,
        "continentalFraction": target_land / n,
        "apronCells": int(apron_cells.size),
    }
    return {"world": world, "start": start}


# Public spelling used by JavaScript-side callers ported mechanically.
buildStartWorld = build_start_world
