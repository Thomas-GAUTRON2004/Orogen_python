"""Collision-time plate welding and raster redraw.

The operation is a direct Python port of the geometry in upstream
`v2/js/tectonics/ops/merge-plates.js`. It deforms only at the suture event:
the upper plate survives, the lower plate is resampled into its frame, and
the overlap lens drives an inverse warp of the upper crust.
"""

from __future__ import annotations

import math

import numpy as np

from ...core.mesh_bfs import hops_from
from ...core.sphere_geometry import PLANET_RADIUS_KM, centroid_of_cells, tangent_toward
from ..crust import BeltSide, CrustFlag, CrustRaster, CrustType, OrogenyType
from ..orogeny import belt_width_km
from ..plate import frame_rotation
from ..spread_belt import spread_belt
from .plate_accretion import close_seas_between


def merge_plates(world, upper, lower, overlap_world_cells, collision) -> dict:
    """Weld two plates in the upper plate's frame and retire the lower plate.

    Returns the land-loss, indentation and sea-closure counters reported by
    the upstream suture event. The caller is responsible for setting the
    collision record's `sutured` flag and rebuilding world ownership.
    """
    mesh, config, time = world.mesh, world.config, float(world.time)
    xyz = mesh.xyz
    num_cells = int(mesh.num_cells)
    offsets, adjacency, index = mesh.adj_offset, mesh.adj_list, mesh.index
    edge_km = float(mesh.edge_km)

    upper_crust, lower_crust = upper.crust, lower.crust
    lower_to_upper = frame_rotation(lower, upper)
    land_before = int(upper.stats["continentalCells"]) + int(
        lower.stats["continentalCells"]
    )
    hue = (
        lower.hue
        if int(lower.stats["cells"]) > int(upper.stats["cells"])
        else upper.hue
    )

    # Resample the lower plate once into the upper plate's coordinate frame.
    from_lower = _resample_lower_cells(mesh, upper, lower, lower_crust)

    def upper_land(cell: int) -> bool:
        return bool(
            upper_crust.mask[cell] and upper_crust.type[cell] != CrustType["OCEAN"]
        )

    def lower_land(cell: int) -> bool:
        source = int(from_lower[cell])
        return bool(source >= 0 and lower_crust.type[source] != CrustType["OCEAN"])

    overlap = np.zeros(num_cells, dtype=np.uint8)
    overlap_count = 0
    for cell in range(num_cells):
        if upper_land(cell) and lower_land(cell):
            overlap[cell] = 1
            overlap_count += 1
    if overlap_count == 0:
        for world_cell in overlap_world_cells:
            cell = upper.cell_at(
                index,
                float(xyz[3 * world_cell]),
                float(xyz[3 * world_cell + 1]),
                float(xyz[3 * world_cell + 2]),
            )
            if not overlap[cell]:
                overlap[cell] = 1
                overlap_count += 1

    # Identify the rear and nose edges of the overlapping indenter.
    rear_side: list[int] = []
    nose_side: list[int] = []
    for cell in np.flatnonzero(overlap):
        cell = int(cell)
        for edge in range(int(offsets[cell]), int(offsets[cell + 1])):
            neighbour = int(adjacency[edge])
            if overlap[neighbour]:
                continue
            if upper_land(neighbour):
                rear_side.append(cell)
                break
        for edge in range(int(offsets[cell]), int(offsets[cell + 1])):
            neighbour = int(adjacency[edge])
            if overlap[neighbour]:
                continue
            if lower_land(neighbour):
                nose_side.append(cell)
                break

    def within(cell: int) -> bool:
        return bool(overlap[cell])

    rear_dist, rear_source = hops_from(mesh, rear_side, within)
    nose_dist, nose_source = hops_from(mesh, nose_side, within)
    overlap_cells = [int(cell) for cell in np.flatnonzero(overlap)]
    centre = centroid_of_cells(xyz, overlap_cells)
    end_a = _farthest(xyz, overlap_cells, centre)
    end_b = _farthest(
        xyz,
        overlap_cells,
        xyz[3 * end_a : 3 * end_a + 3],
    )
    end_a_dist, _ = hops_from(mesh, [end_a], within)
    end_b_dist, _ = hops_from(mesh, [end_b], within)

    depth_km = np.zeros(num_cells, dtype=np.float32)
    along = np.zeros(num_cells, dtype=np.float32)
    normal = np.zeros((num_cells, 3), dtype=np.float32)
    max_indent = 0.0
    sum_indent = 0.0
    lower_world = lower.to_world(*map(float, lower.capCenter))
    lower_centre_in_upper = upper.to_plate_frame(*map(float, lower_world))
    for cell in overlap_cells:
        rear_hops = max(0, int(rear_dist[cell]))
        nose_hops = max(0, int(nose_dist[cell]))
        depth = (rear_hops + nose_hops + 1) * edge_km * math.sqrt(3.0) / 2.0
        depth_km[cell] = depth
        max_indent = max(max_indent, depth)
        sum_indent += depth
        a = max(0, int(end_a_dist[cell]))
        b = max(0, int(end_b_dist[cell]))
        along[cell] = (b - a) / (a + b + 1e-9)

        from_cell = int(nose_source[cell]) if nose_source[cell] >= 0 else cell
        to_cell = int(rear_source[cell]) if rear_source[cell] >= 0 else cell
        here = _position(xyz, cell)
        if from_cell == to_cell:
            direction = tangent_toward(
                here,
                lower_centre_in_upper,
                upper.capCenter,
            )
        else:
            direction = tangent_toward(
                here,
                _position(xyz, from_cell),
                _position(xyz, to_cell),
            )
        normal[cell] = direction

    # Inverse displacement of the upper raster, decaying behind the indenter.
    falloff_km = float(config["indentFalloffKm"])
    indent_gain = float(config["indentGain"])
    max_hops = math.ceil(falloff_km / max(edge_km, 1e-12)) + 1
    reach_dist, reach_source = hops_from(
        mesh, overlap_cells, lambda _cell: True, max_hops
    )
    displacement = np.zeros((num_cells, 3), dtype=np.float32)
    position_a = _position(xyz, end_a)
    position_b = _position(xyz, end_b)
    for cell in range(num_cells):
        near = int(reach_source[cell])
        if near < 0:
            continue
        distance_km = int(reach_dist[cell]) * edge_km * math.sqrt(3.0) / 2.0
        if distance_km >= falloff_km:
            continue
        taper = (1.0 - distance_km / falloff_km) ** 1.5
        magnitude = indent_gain * float(depth_km[near]) * taper
        if magnitude <= 0:
            continue
        lateral_share = float(config["escapeShare"]) * abs(float(along[near]))
        here = _position(xyz, cell)
        end = position_a if along[near] > 0 else position_b
        lateral = tangent_toward(here, here, end)
        normal_vector = normal[near]
        displacement[cell] = magnitude * (
            (1.0 - lateral_share) * normal_vector
            + lateral_share * np.asarray(lateral, dtype=np.float32)
        )

    # Build the output raster in the upper plate's local frame.
    output = CrustRaster(num_cells)
    origin = np.zeros(num_cells, dtype=np.int8)
    moved = np.zeros(3, dtype=np.float64)
    upper_targets = []
    upper_sources = []
    for cell in range(num_cells):
        source = cell
        delta = displacement[cell]
        if float(np.dot(delta, delta)) > 0.0:
            moved[:] = (
                np.asarray(xyz[3 * cell : 3 * cell + 3], dtype=np.float64)
                - delta / PLANET_RADIUS_KM
            )
            length = float(np.linalg.norm(moved))
            if length > 0:
                source = int(index.nearest(*map(float, moved / length)))
        if upper_crust.mask[source]:
            upper_targets.append(cell)
            upper_sources.append(source)

    if upper_targets:
        upper_targets_array = np.asarray(upper_targets, dtype=np.int32)
        upper_sources_array = np.asarray(upper_sources, dtype=np.int32)
        output.copy_many(upper_targets_array, upper_crust, upper_sources_array)
        origin[upper_targets_array] = 1
        upper_cratons = upper_crust.type[upper_sources_array] == CrustType["CRATON"]
        output.type[upper_targets_array[upper_cratons]] = CrustType["CONTINENT"]

    _stamp_cratons_rigidly(output, upper_crust, displacement, mesh)

    lower_targets = np.flatnonzero(from_lower >= 0).astype(np.int32)
    if lower_targets.size:
        lower_sources = from_lower[lower_targets].astype(np.int32, copy=False)
        occupied = output.mask[lower_targets] != 0
        lower_land = lower_crust.type[lower_sources] != CrustType["OCEAN"]
        keep_upper_craton = occupied & (
            output.type[lower_targets] == CrustType["CRATON"]
        )
        accept = (lower_land | ~occupied) & ~keep_upper_craton
        lower_targets = lower_targets[accept]
        lower_sources = lower_sources[accept]
        if lower_targets.size:
            output.copy_many(
                lower_targets,
                lower_crust,
                lower_sources,
                lower_to_upper,
            )
            origin[lower_targets] = 2

    # Fill interior holes left by the inverse warp; preserve genuine plate edges.
    for _pass in range(2):
        fills: list[tuple[int, int]] = []
        for cell in range(num_cells):
            if output.mask[cell]:
                continue
            crust_count = 0
            land_count = 0
            land_source = -1
            sea_source = -1
            for edge in range(int(offsets[cell]), int(offsets[cell + 1])):
                neighbour = int(adjacency[edge])
                if not output.mask[neighbour]:
                    continue
                crust_count += 1
                if output.type[neighbour] != CrustType["OCEAN"]:
                    land_count += 1
                    if land_source < 0:
                        land_source = neighbour
                elif sea_source < 0:
                    sea_source = neighbour
            if crust_count < 4:
                continue
            source = land_source if 2 * land_count >= crust_count else sea_source
            if source >= 0:
                fills.append((cell, source))
        if fills:
            fill_targets = np.fromiter(
                (cell for cell, _source in fills),
                dtype=np.int32,
                count=len(fills),
            )
            fill_sources = np.fromiter(
                (source for _cell, source in fills),
                dtype=np.int32,
                count=len(fills),
            )
            output.copy_many(fill_targets, output, fill_sources)
            filled_cratons = output.type[fill_targets] == CrustType["CRATON"]
            output.type[fill_targets[filled_cratons]] = CrustType["CONTINENT"]
            origin[fill_targets] = origin[fill_sources]
        if not fills:
            break

    sealed = close_seas_between(output, origin, mesh, upper, config, time)

    # The interface between the two origins becomes the suture zone.
    seam: list[int] = []
    for cell in range(num_cells):
        if not output.mask[cell] or output.type[cell] == CrustType["OCEAN"]:
            continue
        for edge in range(int(offsets[cell]), int(offsets[cell + 1])):
            neighbour = int(adjacency[edge])
            if (
                output.mask[neighbour]
                and output.type[neighbour] != CrustType["OCEAN"]
                and origin[neighbour] != origin[cell]
            ):
                seam.append(cell)
                break

    half_width = max(
        1,
        math.floor(float(config["sutureHalfWidthKm"]) / max(edge_km, 1e-12) + 0.5),
    )
    suture_dist, _ = hops_from(
        mesh,
        seam,
        lambda cell: bool(
            output.mask[cell] and output.type[cell] != CrustType["OCEAN"]
        ),
        half_width,
    )
    for cell in range(num_cells):
        if suture_dist[cell] >= 0 and output.type[cell] != CrustType["CRATON"]:
            output.flags[cell] |= CrustFlag["SUTURE"]

    # Redraw plateau and foreland records over their respective sides.
    merged = type("MergedPlate", (), {})()
    merged.crust = output
    mean_indent = sum_indent / max(1, overlap_count)
    rear_seam = [cell for cell in seam if origin[cell] == 1]
    nose_seam = [cell for cell in seam if origin[cell] == 2]
    total_work = float(collision["shorteningKm"]) + mean_indent
    rate = float(collision["meanRateCmYr"])
    if rear_seam:
        spread_belt(
            merged,
            mesh,
            rear_seam,
            {
                "type": OrogenyType["HIMALAYAN"],
                "widthKm": belt_width_km("himalayan", total_work, rate, config),
                "workKm": mean_indent,
                "time": time,
                "side": BeltSide["UPPER"],
                "keepShear": True,
            },
            config,
            lambda cell: origin[cell] == 1,
        )
    if nose_seam:
        spread_belt(
            merged,
            mesh,
            nose_seam,
            {
                "type": OrogenyType["HIMALAYAN"],
                "widthKm": belt_width_km("foreland", total_work, rate, config),
                "workKm": mean_indent,
                "time": time,
                "side": BeltSide["LOWER"],
                "keepShear": True,
            },
            config,
            lambda cell: origin[cell] == 2,
        )

    upper.crust = output
    upper.wings.clear()
    upper.mark_reshaped()
    upper.refresh_extent(xyz)
    upper.name = world.name_by_cratons(upper, f"{upper.name}+{lower.name}")
    upper.born = time
    upper.bornOfRift = False
    lower.alive = False
    upper.hue = hue
    lower.crust = CrustRaster(num_cells)
    lower.omega = [0.0, 0.0, 0.0]
    lower.dynamics = None
    lower.refresh_extent(xyz)

    return {
        "landLostCells": land_before - (int(upper.stats["continentalCells"]) - sealed),
        "maxIndentKm": max_indent,
        "meanIndentKm": mean_indent,
        "overlapCells": overlap_count,
        "sealedCells": sealed,
    }


def _resample_lower_cells(mesh, upper, lower, lower_crust) -> np.ndarray:
    """Map every upper-frame world cell to its occupied lower-plate source."""
    num_cells = int(mesh.num_cells)
    xyz, index = mesh.xyz, mesh.index
    from_lower = np.full(num_cells, -1, dtype=np.int32)
    for cell in range(num_cells):
        point = upper.to_world(
            float(xyz[3 * cell]),
            float(xyz[3 * cell + 1]),
            float(xyz[3 * cell + 2]),
        )
        local = lower.cell_at(index, *map(float, point))
        if lower_crust.mask[local]:
            from_lower[cell] = local
    return from_lower


def _position(xyz, cell: int) -> np.ndarray:
    return np.asarray(xyz[3 * cell : 3 * cell + 3], dtype=np.float64)


def _farthest(xyz, cells, from_point) -> int:
    best = int(cells[0])
    best_dot = 2.0
    for cell in cells:
        offset = 3 * int(cell)
        value = (
            float(xyz[offset]) * float(from_point[0])
            + float(xyz[offset + 1]) * float(from_point[1])
            + float(xyz[offset + 2]) * float(from_point[2])
        )
        if value < best_dot:
            best_dot = value
            best = int(cell)
    return best


def _stamp_cratons_rigidly(output, upper_crust, displacement, mesh) -> None:
    """Move each upper-plate craton as a rigid terrane rather than warping it."""
    xyz, index, num_cells = mesh.xyz, mesh.index, int(mesh.num_cells)
    blocks: dict[int, dict] = {}
    for cell in range(num_cells):
        if not upper_crust.mask[cell] or upper_crust.type[cell] != CrustType["CRATON"]:
            continue
        terrane = int(upper_crust.terrane[cell])
        block = blocks.setdefault(terrane, {"cells": [], "displacement": np.zeros(3)})
        block["cells"].append(cell)
        block["displacement"] += displacement[cell]

    for block in blocks.values():
        delta = block["displacement"] / max(1, len(block["cells"])) / PLANET_RADIUS_KM
        for cell in block["cells"]:
            point = _position(xyz, cell) + delta
            length = float(np.linalg.norm(point))
            if length <= 0:
                continue
            target = int(index.nearest(*map(float, point / length)))
            output.copy_from(target, upper_crust, cell)
