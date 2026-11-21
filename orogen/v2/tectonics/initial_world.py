"""Deterministic V2 initial-world construction.

Production meshes use the seeded craton-packing and noise-shaped landmass
pipeline ported from the upstream V2 initializer. The implementation preserves
the Python world/raster contracts and uses mesh-native sampling for coastline,
apron and trench assignment. Numerical output is not claimed to be bit-for-bit
identical to the JavaScript engine; differential fixtures define the supported
compatibility envelope.
"""

from __future__ import annotations

import heapq
import math
from typing import Any

import numpy as np

from ..core.fast_noise import FastNoise
from ..core.mesh_bfs import hop_distance
from ..core.seed import make_rng
from ..core.sim_mesh import km_to_hops
from ..core.sphere_geometry import (
    cell_distance_km,
    polar_frame,
    random_point,
    tangent_basis,
)
from .crust import CrustFlag, CrustType, OrogenyType
from .far_field import spread_push
from .spread_belt import spread_belt
from .world import EventKind, TectonicWorld

StartSeam = {"NONE": 0, "INNER": 1, "RING": 2, "MAJOR": 3}


def _mesh_xyz(sim_mesh: Any) -> np.ndarray:
    xyz = np.asarray(sim_mesh.xyz, dtype=np.float64).reshape((-1, 3))
    if xyz.shape[0] != int(sim_mesh.num_cells):
        raise ValueError("mesh xyz does not match num_cells")
    return xyz


def _distance_hops(
    sim_mesh: Any, sources: np.ndarray, *, can_enter=None, max_hops=None
) -> np.ndarray:
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


def _land_cells(
    sim_mesh: Any, centre: np.ndarray, count: int
) -> tuple[np.ndarray, float]:
    xyz = _mesh_xyz(sim_mesh)
    count = max(1, min(int(count), int(sim_mesh.num_cells) - 1))
    cosine = xyz @ centre
    # Resolve equal-distance mesh cells by index, not by argpartition's
    # implementation-dependent ordering at the selection boundary.
    cells = np.lexsort((np.arange(cosine.size), -cosine))[:count]
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
    count = min(int(count), int(core.size))
    if count <= 0:
        return np.empty(0, dtype=np.int32)
    random_order = np.argsort(
        np.fromiter((rng() for _ in range(core.size)), dtype=np.float64)
    )
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
            kind = (
                StartSeam["MAJOR"]
                if groups[other] != groups[cell]
                else StartSeam["INNER"]
            )
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


def _paint_old_ranges(
    world: TectonicWorld,
    plate,
    land_mask: np.ndarray,
    seams: np.ndarray,
    pairs: np.ndarray | None = None,
    pair_range: np.ndarray | None = None,
) -> None:
    """Build inherited ranges with spatially varying source strength."""
    cfg, mesh, crust = world.config, world.mesh, plate.crust
    ranges = cfg.get("startRanges", {})
    if not isinstance(ranges, dict):
        return
    xyz = _mesh_xyz(mesh)
    noise = FastNoise((int(world.seed) ^ 0x5A7E) & 0xFFFFFFFF)
    vary = max(0.0, float(cfg.get("startRangeVary", 0.5)))
    scale_km = max(1.0, float(cfg.get("startRangeVaryKm", 1600.0)))
    frequency = 6371.0 / scale_km
    offsets, adjacency = mesh.adj_offset, mesh.adj_list

    for kind_name in ("inner", "ring", "major"):
        kind = StartSeam[kind_name.upper()]
        settings = ranges.get(kind_name)
        if not settings or len(settings) < 4:
            continue
        width_km, work_km, age_myr, keep = (float(value) for value in settings[:4])
        if width_km <= 0.0 or work_km <= 0.0:
            continue
        sources = np.flatnonzero((seams == kind) & land_mask)
        if not sources.size:
            continue

        strength = np.empty(sources.size, dtype=np.float64)
        for index, cell_value in enumerate(sources):
            cell = int(cell_value)
            x, y, z = xyz[cell]
            value = noise.noise3d(
                x * frequency + 7.1 * kind, y * frequency, z * frequency
            ) + 0.4 * noise.noise3d(
                x * 2.7 * frequency + 3.3,
                y * 2.7 * frequency + 8.1 * kind,
                z * 2.7 * frequency + 1.9,
            )
            strength[index] = 1.0 - vary * np.clip(0.5 + 0.6 * value, 0.0, 1.0)
            if (
                kind == StartSeam["MAJOR"]
                and pairs is not None
                and pair_range is not None
            ):
                pair_id = int(pairs[cell])
                if 0 <= pair_id < len(pair_range):
                    strength[index] *= float(pair_range[pair_id])

        before = {
            "root": crust.rootKm.copy(),
            "uplift": crust.upliftKm.copy(),
            "rugged": crust.rugged.copy(),
        }
        spread_belt(
            plate,
            mesh,
            sources.tolist(),
            {
                "type": OrogenyType["URAL"],
                "time": -age_myr,
                "widthKm": width_km,
                "workKm": work_km,
                "works": (work_km * strength).tolist(),
                "widths": (width_km * (0.7 + 0.3 * strength)).tolist(),
            },
            cfg,
            allow=lambda cell: bool(land_mask[cell]),
        )

        # Carry the nearest source's strength inland, as the reference does
        # when it scales the worn relief left by each inherited seam.
        nearest = np.full(int(mesh.num_cells), np.inf, dtype=np.float64)
        nearest_strength = np.ones(int(mesh.num_cells), dtype=np.float64)
        queue: list[tuple[float, int]] = []
        for cell_value, value in zip(sources, strength, strict=True):
            cell = int(cell_value)
            nearest[cell] = 0.0
            nearest_strength[cell] = float(value)
            heapq.heappush(queue, (0.0, cell))
        limit = 2.0 * width_km
        while queue:
            distance, cell = heapq.heappop(queue)
            if distance > nearest[cell] + 1e-9 or distance > limit:
                continue
            lo, hi = int(offsets[cell]), int(offsets[cell + 1])
            for other_value in adjacency[lo:hi]:
                other = int(other_value)
                if not crust.mask[other]:
                    continue
                candidate = distance + cell_distance_km(mesh.xyz, cell, other)
                if candidate >= nearest[other] or candidate > limit:
                    continue
                nearest[other] = candidate
                nearest_strength[other] = nearest_strength[cell]
                heapq.heappush(queue, (candidate, other))

        affected = np.isfinite(nearest)
        relief_keep = keep * np.where(affected, nearest_strength, 1.0)
        for field, previous in (
            (crust.rootKm, before["root"]),
            (crust.upliftKm, before["uplift"]),
            (crust.rugged, before["rugged"]),
        ):
            field[:] = previous + (field - previous) * relief_keep


def _distance_km(
    sim_mesh: Any,
    sources: np.ndarray,
    *,
    can_enter=None,
    max_km: float = 3000.0,
    source_strength: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return weighted geodesic distance and nearest-source strength."""
    n = int(sim_mesh.num_cells)
    source_mask = np.asarray(sources, dtype=bool)
    if source_mask.shape != (n,):
        raise ValueError("source mask must contain one value per mesh cell")
    distance = np.full(n, np.inf, dtype=np.float64)
    strength = np.zeros(n, dtype=np.float32)
    queue: list[tuple[float, int]] = []
    for cell_value in np.flatnonzero(source_mask):
        cell = int(cell_value)
        distance[cell] = 0.0
        strength[cell] = (
            float(source_strength[cell]) if source_strength is not None else 1.0
        )
        heapq.heappush(queue, (0.0, cell))
    offsets, adjacency = sim_mesh.adj_offset, sim_mesh.adj_list
    while queue:
        current, cell = heapq.heappop(queue)
        if current > distance[cell] + 1e-9 or current > max_km:
            continue
        lo, hi = int(offsets[cell]), int(offsets[cell + 1])
        for other_value in adjacency[lo:hi]:
            other = int(other_value)
            if can_enter is not None and not can_enter(other):
                continue
            candidate = current + cell_distance_km(sim_mesh.xyz, cell, other)
            if candidate >= distance[other] or candidate > max_km:
                continue
            distance[other] = candidate
            strength[other] = strength[cell]
            heapq.heappush(queue, (candidate, other))
    return distance.astype(np.float32), strength


def _build_coarse_start_world(
    sim_mesh: Any, seed: int, overrides: dict | None = None
) -> dict[str, Any]:
    """Build a deterministic supercontinent and passive pre-history ocean.

    The return shape follows upstream buildStartWorld. The continental cell
    fraction is exact at mesh resolution; craton count is reduced on coarse
    meshes so every core remains represented by cells.
    """
    world = TectonicWorld(sim_mesh, seed, overrides=overrides)
    cfg = world.config
    fraction = float(cfg.get("continentalFraction", 0.29))
    if not math.isfinite(fraction) or not 0.0 < fraction < 1.0:
        raise ValueError(
            "continentalFraction must be finite and strictly between 0 and 1"
        )
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
        xyz,
        land_cells,
        centre,
        land_radius,
        craton_count,
        float(cfg.get("cratonMinGapKm", 500.0)),
        float(sim_mesh.edge_km),
        layout_rng,
    )
    terrane_owner, core_mask, local_owner = _assign_cratons(
        xyz, land_cells, centres, share
    )
    terrane_for_cell = np.zeros(n, dtype=np.uint16)
    terrane_for_cell[land_cells] = terrane_owner.astype(np.uint16)
    craton_mask = np.zeros(n, dtype=bool)
    craton_mask[land_cells[core_mask]] = True

    group_count = min(4, max(1, int(round(math.sqrt(craton_count)))))
    group_for_terrane = np.floor(
        np.arange(craton_count, dtype=np.float64) * group_count / craton_count
    ).astype(np.int32)
    seams = _seams(
        sim_mesh,
        land_mask,
        land_cells,
        terrane_owner,
        centre,
        land_radius,
        group_for_terrane,
    )

    # Reproduce the upstream morphological closing: grow land by apron + smoothing,
    # then erode the grown mask by the smoothing radius. This fills narrow bays and
    # leaves a smoother offshore plate rim than a simple apron dilation.
    apron_km = max(0.0, float(cfg.get("apronKm", 250.0)))
    smooth_km = max(0.0, float(cfg.get("apronSmoothKm", 1400.0)))
    apron_hops = max(
        1, int(math.floor(apron_km / max(float(sim_mesh.edge_km), 1.0) + 0.5))
    )
    smooth_hops = max(
        1, int(math.floor(smooth_km / max(float(sim_mesh.edge_km), 1.0) + 0.5))
    )
    grown_distance = _distance_hops(
        sim_mesh, land_mask, max_hops=apron_hops + smooth_hops
    )
    grown = grown_distance >= 0
    from_outside = _distance_hops(sim_mesh, ~grown, max_hops=smooth_hops)
    in_plate = grown & (from_outside < 0)
    apron_mask = in_plate & ~land_mask

    continent = world.add_plate("".join(chr(65 + i) for i in range(craton_count)))
    ocean = world.add_plate("Pre-history ocean")
    ocean.isBackground = True
    continent.born = -float(cfg.get("riftTenureMyr", 30.0))

    land_types = np.full(land_cells.size, CrustType["CONTINENT"], dtype=np.uint8)
    land_types[core_mask] = CrustType["CRATON"]
    continent.crust.create_many(
        land_cells,
        land_types,
        -float(cfg.get("initialLandAgeMyr", 2000.0)),
        terrane_owner.astype(np.uint16),
    )
    apron_cells = np.flatnonzero(apron_mask)
    if apron_cells.size:
        continent.crust.create_many(
            apron_cells,
            CrustType["OCEAN"],
            -float(cfg.get("initialOceanAgeMyr", 150.0)),
        )
        continent.crust.flags[apron_cells] |= CrustFlag["APRON"]

    # The active trench follows the smoothed outer rim, leaving one passive
    # azimuthal sector. The sector is sampled before any rim cells are visited.
    east, north = tangent_basis(centre.tolist())
    rim_mask = np.zeros(n, dtype=bool)
    offsets, adjacency = sim_mesh.adj_offset, sim_mesh.adj_list
    for cell in np.flatnonzero(in_plate):
        lo, hi = int(offsets[cell]), int(offsets[cell + 1])
        rim_mask[cell] = any(not in_plate[int(other)] for other in adjacency[lo:hi])
    rim_cells = np.flatnonzero(rim_mask).astype(np.int32)
    coverage = max(0.0, min(1.0, float(cfg.get("rimTrenchCoverage", 0.85))))
    passive_from = passive_rng() * 360.0 - 180.0
    passive_span = 360.0 * (1.0 - coverage)
    if rim_cells.size and coverage > 0:
        angles = np.degrees(np.arctan2(xyz[rim_cells] @ north, xyz[rim_cells] @ east))
        offset_angles = np.mod(angles - passive_from, 360.0)
        trench_cells = rim_cells[offset_angles >= passive_span]
        if not trench_cells.size and coverage >= 0.5:
            trench_cells = rim_cells
        continent.crust.flags[trench_cells] |= CrustFlag["TRENCH"]
        continent.crust.slabKm[trench_cells] = np.maximum(
            continent.crust.slabKm[trench_cells],
            float(cfg.get("slabMatureKm", 600.0)),
        )
    else:
        trench_cells = np.empty(0, dtype=np.int32)

    ocean_cells = np.flatnonzero(~in_plate)
    ocean.crust.create_many(
        ocean_cells,
        CrustType["OCEAN"],
        -float(cfg.get("initialOceanAgeMyr", 150.0)),
    )

    # Inner and major boundaries are weak sutures. Ring seams are accreted
    # belts and therefore remain stronger rift corridors.
    weak = (seams == StartSeam["INNER"]) | (seams == StartSeam["MAJOR"])
    continent.crust.flags[weak] |= CrustFlag["SUTURE"]
    _paint_old_ranges(world, continent, land_mask, seams)

    # Each pair of proto-continents leaves its own far-field shortening fan.
    # Grouping fronts independently preserves the source strength and length
    # scaling used by the upstream startPushes pass.
    far_start_work = float(cfg.get("farStartWorkKm", 0.0))
    if far_start_work > 0.0:
        offsets, adjacency = sim_mesh.adj_offset, sim_mesh.adj_list
        by_pair: dict[tuple[int, int], dict[int, float]] = {}
        for cell in np.flatnonzero((seams == StartSeam["MAJOR"]) & land_mask):
            cell_terrane = int(terrane_for_cell[cell])
            if cell_terrane <= 0:
                continue
            cell_group = int(group_for_terrane[cell_terrane - 1])
            neighbours = set()
            lo, hi = int(offsets[cell]), int(offsets[cell + 1])
            for other_value in adjacency[lo:hi]:
                other = int(other_value)
                if not land_mask[other]:
                    continue
                other_terrane = int(terrane_for_cell[other])
                if other_terrane <= 0:
                    continue
                other_group = int(group_for_terrane[other_terrane - 1])
                if other_group != cell_group:
                    neighbours.add(other_group)
            for other_group in neighbours:
                pair = (
                    min(cell_group, other_group),
                    max(cell_group, other_group),
                )
                by_pair.setdefault(pair, {})[int(cell)] = far_start_work
        for pair in sorted(by_pair):
            sources = by_pair[pair]
            spread_push(
                world,
                continent,
                sources,
                float(len(sources)) * float(sim_mesh.edge_km),
                -float(cfg.get("startRanges", {}).get("major", [0, 0, 0])[2]),
            )

    major_sources = (seams == StartSeam["MAJOR"]) & land_mask
    ring_sources = (seams == StartSeam["RING"]) & land_mask
    major_km, major_range = _distance_km(
        sim_mesh, major_sources, can_enter=lambda cell: bool(land_mask[cell])
    )
    ring_km, _ = _distance_km(
        sim_mesh, ring_sources, can_enter=lambda cell: bool(land_mask[cell])
    )
    world.startSeamKm = {
        "major": major_km,
        "majorRange": major_range,
        "ring": ring_km,
    }

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
        cratons.append(
            {
                "id": index + 1,
                "letter": chr(65 + index),
                "centreCell": int(cell),
                "centre": xyz[int(cell)].tolist(),
                "cells": int(
                    np.count_nonzero(craton_mask & (terrane_for_cell == index + 1))
                ),
                "terraneCells": int(held.size),
                "cluster": int(group_for_terrane[index]),
            }
        )
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
        "passiveTo": passive_from + passive_span,
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


_MAX_LAYOUT_ATTEMPTS = 24
_LAYOUT_RELAX_SPAN_AFTER = 18
_LAYOUT_FRACTION_TOLERANCE = 0.004


def _passive_stretch(coverage: float, rng) -> dict[str, float]:
    """Choose the passive azimuthal sector before drawing the continental coast."""
    return {
        "from": rng() * 360.0 - 180.0,
        "span": 360.0 * (1.0 - min(1.0, max(0.0, coverage))),
    }


def _plate_outline_with_apron(
    sim_mesh: Any, is_land: np.ndarray, config: dict
) -> tuple[np.ndarray, np.ndarray]:
    """Morphologically close the continental mask and return its outer rim."""
    count = int(sim_mesh.num_cells)
    apron = km_to_hops(sim_mesh, float(config["apronKm"]))
    smooth = km_to_hops(sim_mesh, float(config["apronSmoothKm"]))
    from_land = hop_distance(
        sim_mesh, lambda cell: bool(is_land[cell]), max_hops=apron + smooth
    )
    grown = from_land >= 0
    from_outside = hop_distance(
        sim_mesh, lambda cell: not bool(grown[cell]), max_hops=smooth
    )
    in_plate = grown & (from_outside < 0)
    rim = np.zeros(count, dtype=bool)
    offsets, adjacency = sim_mesh.adj_offset, sim_mesh.adj_list
    for cell in np.flatnonzero(in_plate):
        lo, hi = int(offsets[cell]), int(offsets[cell + 1])
        rim[cell] = any(not in_plate[int(other)] for other in adjacency[lo:hi])
    return in_plate, np.flatnonzero(rim).astype(np.int32)


def _choose_active_rim(
    sim_mesh: Any,
    rim: np.ndarray,
    centre: list[float],
    reference: list[float],
    passive: dict[str, float],
) -> tuple[np.ndarray, float, float]:
    """Choose mature trench cells outside the passive azimuthal sector."""
    xyz = _mesh_xyz(sim_mesh)
    to_polar = polar_frame(centre, reference)
    active = np.zeros(int(sim_mesh.num_cells), dtype=bool)
    for cell_value in rim:
        cell = int(cell_value)
        azimuth = to_polar(*xyz[cell])["azimuth"]
        offset = ((azimuth - passive["from"]) % 360.0 + 360.0) % 360.0
        if offset >= passive["span"]:
            active[cell] = True
    return active, passive["from"], passive["from"] + passive["span"]


def _shrink_steps(floor: float) -> list[float]:
    lowest = max(0.05, min(1.0, float(floor)))
    steps = []
    value = 1.0
    while value > lowest:
        steps.append(value)
        value *= 0.85
    steps.append(lowest)
    return steps


def _shape_attempt(
    sim_mesh: Any, seed: int, config: dict, attempt: int, layout: dict, stretch: float
):
    from .initial_landmass import shape_landmass

    passive = _passive_stretch(
        float(config["rimTrenchCoverage"]),
        make_rng(seed * 31 + attempt * 7919 + 29),
    )
    noise = FastNoise(seed + attempt * 101)
    rng = make_rng(seed * 31 + attempt * 7919 + 41)
    land = shape_landmass(sim_mesh, layout, noise, rng, config, stretch, passive)
    return land


def _evaluate_layout_attempt(
    sim_mesh: Any, seed: int, config: dict, attempt: int
) -> tuple[list[dict], dict | None]:
    from .initial_cratons import place_cratons

    tried: list[dict] = []
    stretch = 1.0 - attempt / _MAX_LAYOUT_ATTEMPTS
    for area_scale in _shrink_steps(config["cratonMinAreaScale"]):
        rng = make_rng(seed * 31 + attempt * 7919 + 17)
        layout = place_cratons(rng, config, stretch, area_scale)
        if layout is None:
            continue
        land = _shape_attempt(sim_mesh, seed, config, attempt, layout, stretch)
        if land is None:
            continue
        land_count = int(np.count_nonzero(land["type"] != CrustType["OCEAN"]))
        excess = land_count / int(sim_mesh.num_cells) - float(
            config["continentalFraction"]
        )
        candidate = {
            "layout": layout,
            "land": land,
            "areaScale": area_scale,
            "excess": excess,
            "attempts": attempt + 1,
        }
        tried.append(candidate)
        if abs(excess) <= _LAYOUT_FRACTION_TOLERANCE:
            return tried, candidate
    return tried, None


def _land_centroid(sim_mesh: Any, land_type: np.ndarray) -> list[float]:
    xyz = _mesh_xyz(sim_mesh)
    selected = land_type != CrustType["OCEAN"]
    total = np.sum(xyz[selected], axis=0, dtype=np.float64)
    length = float(np.linalg.norm(total)) or 1.0
    return (total / length).tolist()


def _half_span_deg(sim_mesh: Any, land_type: np.ndarray, centre: list[float]) -> float:
    xyz = _mesh_xyz(sim_mesh)
    selected = land_type != CrustType["OCEAN"]
    if not np.any(selected):
        return 0.0
    farthest_dot = float(np.min(xyz[selected] @ np.asarray(centre, dtype=np.float64)))
    return math.degrees(math.acos(max(-1.0, min(1.0, farthest_dot))))


def _find_layout(sim_mesh: Any, seed: int, config: dict) -> dict | None:
    """Search layouts in deterministic upstream attempt order."""
    round_span = math.degrees(
        math.acos(1.0 - 2.0 * float(config["continentalFraction"]))
    )
    closest: dict | None = None
    for attempt in range(_MAX_LAYOUT_ATTEMPTS):
        tried, fit = _evaluate_layout_attempt(sim_mesh, seed, config, attempt)
        for candidate in tried:
            if closest is None or abs(candidate["excess"]) < abs(closest["excess"]):
                closest = candidate
        chosen = fit
        if chosen is None and attempt == _MAX_LAYOUT_ATTEMPTS - 1:
            chosen = closest
        if chosen is None:
            continue
        middle = _land_centroid(sim_mesh, chosen["land"]["type"])
        too_wide = _half_span_deg(
            sim_mesh, chosen["land"]["type"], middle
        ) > round_span + float(config["spanAllowanceDeg"])
        if fit is not None and too_wide and attempt < _LAYOUT_RELAX_SPAN_AFTER:
            continue
        return {
            **chosen,
            "middle": middle,
            "fractionMet": fit is not None,
        }
    return None


def _record_start_pushes(world: TectonicWorld, continent, land: dict) -> None:
    config = world.config
    work = float(config["farStartWorkKm"])
    if work <= 0.0:
        return
    pair_ids = land["pair"]
    pair_range = land["pairRange"]
    major = (land["suture"] == StartSeam["MAJOR"]) & (
        land["type"] != CrustType["OCEAN"]
    )
    for pair_id in sorted(int(value) for value in np.unique(pair_ids[major]) if value):
        cells = np.flatnonzero(major & (pair_ids == pair_id))
        amount = work * float(pair_range[pair_id])
        sources = {int(cell): amount for cell in cells}
        spread_push(
            world,
            continent,
            sources,
            float(len(sources)) * float(world.mesh.edge_km),
            -float(config["startRanges"]["major"][2]),
        )


def _assemble_upstream_start_world(
    world: TectonicWorld, seed: int, found: dict
) -> dict[str, Any]:
    from .initial_landmass import Seam

    mesh, config = world.mesh, world.config
    land, layout = found["land"], found["layout"]
    is_land = land["type"] != CrustType["OCEAN"]
    in_plate, rim = _plate_outline_with_apron(mesh, is_land, config)
    reference = tangent_basis(land["rimCentre"])[0]
    active, passive_from, passive_to = _choose_active_rim(
        mesh, rim, land["rimCentre"], reference, land["passive"]
    )
    continent_name = "".join(item["letter"] for item in layout["cratons"])
    continent = world.add_plate(continent_name)
    ocean = world.add_plate("Pre-history ocean")
    ocean.isBackground = True

    ocean_age = -float(config["initialOceanAgeMyr"])
    land_age = -float(config["initialLandAgeMyr"])
    slab_mature = float(config["slabMatureKm"])
    apron_cells = []
    trench_cells = []
    weak_sutures = (land["suture"] == Seam["MAJOR"]) | (land["suture"] == Seam["INNER"])
    for cell in range(int(mesh.num_cells)):
        if not in_plate[cell]:
            ocean.crust.create(cell, CrustType["OCEAN"], ocean_age)
            continue
        if land["type"][cell] == CrustType["OCEAN"]:
            continent.crust.create(cell, CrustType["OCEAN"], ocean_age)
            continent.crust.flags[cell] |= CrustFlag["APRON"]
            apron_cells.append(cell)
            if active[cell]:
                continent.crust.arm_trench(cell, slab_mature)
                trench_cells.append(cell)
        else:
            continent.crust.create(
                cell,
                int(land["type"][cell]),
                land_age,
                int(land["terrane"][cell]),
            )
            if weak_sutures[cell]:
                continent.crust.flags[cell] |= CrustFlag["SUTURE"]

    _paint_old_ranges(
        world,
        continent,
        is_land,
        land["suture"],
        land["pair"],
        land["pairRange"],
    )
    major_sources = land["suture"] == Seam["MAJOR"]
    ring_sources = land["suture"] == Seam["RING"]
    major_strength = np.zeros(int(mesh.num_cells), dtype=np.float32)
    major_cells = np.flatnonzero(major_sources)
    if major_cells.size:
        major_strength[major_cells] = land["pairRange"][land["pair"][major_cells]]
    major_km, major_range = _distance_km(
        mesh,
        major_sources,
        can_enter=lambda cell: bool(is_land[cell]),
        source_strength=major_strength,
    )
    ring_km, _ = _distance_km(
        mesh,
        ring_sources,
        can_enter=lambda cell: bool(is_land[cell]),
    )
    world.startSeamKm = {
        "major": major_km,
        "majorRange": major_range,
        "ring": ring_km,
    }
    _record_start_pushes(world, continent, land)

    for plate in world.plates:
        plate.refresh_extent(mesh.xyz)
    world.terraneLetters = {
        int(item["id"]): str(item["letter"]) for item in layout["cratons"]
    }
    continent.born = -float(config["riftTenureMyr"])
    world.log(
        EventKind["START"],
        f"Supercontinent {continent.name}: {len(layout['cratons'])} cratons, ringed by subduction",
        {"plate": continent.id, "cratons": len(layout["cratons"])},
    )
    start = {
        "cratons": layout["cratons"],
        "centre": found["middle"],
        "rimCentre": land["rimCentre"],
        "reference": reference,
        "elongation": layout["elongation"],
        "seams": land["suture"],
        "owner": land["owner"],
        "skirtKm": land["skirtKm"],
        "rim": rim,
        "passiveFrom": passive_from,
        "passiveTo": passive_to,
        "attempts": found["attempts"],
        "cratonAreaScale": found["areaScale"],
        "fractionMet": found["fractionMet"],
        "config": config,
        "continentId": continent.id,
        "oceanId": ocean.id,
        "landCells": int(np.count_nonzero(is_land)),
        "continentalFraction": float(np.count_nonzero(is_land)) / int(mesh.num_cells),
        "apronCells": len(apron_cells),
        "trench": np.asarray(trench_cells, dtype=np.int32),
    }
    return {"world": world, "start": start}


def build_start_world(
    sim_mesh: Any, seed: int, overrides: dict | None = None
) -> dict[str, Any]:
    """Build the upstream craton-assembled initial supercontinent.

    Meshes below 512 cells retain the small deterministic fixture builder:
    the upstream geometric packing has no meaningful spatial resolution there.
    """
    if int(sim_mesh.num_cells) < 512:
        return _build_coarse_start_world(sim_mesh, seed, overrides)
    world = TectonicWorld(sim_mesh, seed, overrides=overrides)
    found = _find_layout(sim_mesh, int(seed), world.config)
    if found is None:
        raise ValueError(
            "Initial-world settings do not fit the planet: craton spacing, coast "
            "margins and connected land cannot satisfy continentalFraction."
        )
    if not found["fractionMet"]:
        actual = int(np.count_nonzero(found["land"]["type"] != CrustType["OCEAN"]))
        achieved = actual / int(sim_mesh.num_cells)
        target = float(world.config["continentalFraction"])
        raise ValueError(
            "Initial-world settings could not meet continentalFraction after "
            f"{found['attempts']} attempts (target={target:.4f}, "
            f"closest={achieved:.4f}); adjust craton spacing, coast margins or "
            "continentalFraction."
        )
    return _assemble_upstream_start_world(world, int(seed), found)


# Public spelling used by JavaScript-side callers ported mechanically.
buildStartWorld = build_start_world  # noqa: N816 -- preserve upstream V2 public spelling
