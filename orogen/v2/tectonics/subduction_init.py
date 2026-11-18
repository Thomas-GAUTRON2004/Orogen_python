"""Subduction initiation along passive ocean margins, ported from Orogen V2."""

from __future__ import annotations

import math

import numpy as np

from ..core.mesh_bfs import largest_piece_of
from .crust import CrustFlag, CrustType
from .ops.birth_ocean_plate import birth_ocean_plate, ocean_plate_name
from .ops.plate_transfer import transfer_cells
from .ops.rift_partition import exterior_sea
from .world import EventKind


def initiate_subduction(world, dt: float) -> int:
    """Found new trenches along old, non-diverging oceanic plate contacts."""
    mesh = world.mesh
    owner, owner_cell = world.owner, world.ownerCell
    plates = world.plates
    convergence = world.convergence
    config = world.config
    time = float(world.time)
    num_cells = int(mesh.num_cells)
    offsets, adjacency = mesh.adj_offset, mesh.adj_list
    edge_km = float(mesh.edge_km)

    if float(config["subductionInitRatePerMyr"]) <= 0.0:
        return 0

    def stale(plate, cell: int) -> bool:
        return bool(
            (plate.reshaped and not plate.isBackground) or not plate.crust.mask[cell]
        )

    by_pair: dict[tuple[int, int], dict] = {}
    for world_cell in range(num_cells):
        plate_a_id = int(owner[world_cell])
        if plate_a_id < 0 or plates[plate_a_id].isBackground:
            continue
        plate_a = plates[plate_a_id]
        cell_a = int(owner_cell[world_cell])
        crust_a = plate_a.crust
        if (
            stale(plate_a, cell_a)
            or crust_a.type[cell_a] != CrustType["OCEAN"]
            or int(crust_a.flags[cell_a]) & CrustFlag["TRENCH"]
        ):
            continue

        for edge in range(
            int(offsets[world_cell]),
            int(offsets[world_cell + 1]),
        ):
            neighbour_world = int(adjacency[edge])
            plate_b_id = int(owner[neighbour_world])
            if plate_b_id < 0 or plate_b_id == plate_a_id:
                continue

            plate_b = plates[plate_b_id]
            cell_b = int(owner_cell[neighbour_world])
            crust_b = plate_b.crust
            if (
                stale(plate_b, cell_b)
                or int(crust_b.flags[cell_b]) & CrustFlag["TRENCH"]
            ):
                continue
            if float(convergence[world_cell]) < -float(
                config["subductionInitMaxOpeningCmYr"]
            ):
                continue

            key = (
                (plate_a_id, plate_b_id)
                if plate_a_id < plate_b_id
                else (plate_b_id, plate_a_id)
            )
            entry = by_pair.setdefault(key, {"cells": [], "ages": 0.0})
            entry["cells"].append(world_cell)
            entry["ages"] += time - min(
                float(crust_a.formedAt[cell_a]),
                float(crust_b.formedAt[cell_b]),
            )
            break

    started = 0
    for (plate_a_id, plate_b_id), entry in by_pair.items():
        plate_a, plate_b = plates[plate_a_id], plates[plate_b_id]
        if not plate_a.alive or not plate_b.alive:
            continue

        length_km = len(entry["cells"]) * edge_km
        age_myr = float(entry["ages"]) / len(entry["cells"])
        minimum_length = float(config["subductionInitMinLengthKm"])
        minimum_age = float(config["subductionInitAgeMyr"])
        if length_km < minimum_length or age_myr < minimum_age:
            continue

        over_age = age_myr / minimum_age - 1.0
        chance = (
            float(config["subductionInitRatePerMyr"])
            * float(dt)
            * min(float(config["subductionInitAgeCap"]), over_age)
            * (length_km / minimum_length)
        )
        cycle = getattr(world, "cycle", None)
        if (
            cycle is not None
            and cycle.get("phase") == "dispersing"
            and age_myr <= time - float(cycle["breakupAt"]) + 1e-9
        ):
            chance *= float(config["disperseFounderFactor"])
        if float(world.rngSubduction()) >= chance:
            continue

        segment = largest_piece_of(mesh, entry["cells"])
        if len(segment) * edge_km < minimum_length:
            continue

        if plate_a.isBackground or plate_b.isBackground:
            background = plate_a if plate_a.isBackground else plate_b
            born = birth_ocean_plate(
                world,
                background,
                segment,
                float(config["oceanPlateReachKm"]),
                ocean_plate_name(world),
            )
            if born is None:
                continue
            if plate_a.isBackground:
                plate_a = born
            else:
                plate_b = born

        upper, lower = choose_foundering(world, plate_a, plate_b, segment)
        arm_margin(world, upper, lower, segment)
        started += 1

        segment_length = len(segment) * edge_km
        world.log(
            EventKind["SUBDUCTION_INIT"],
            (
                f"a new trench founders along {segment_length:.0f} km of "
                f"{lower.name}’s passive margin, beneath {upper.name}: "
                f"ocean {age_myr:.0f} Myr old begins to go down"
            ),
            {
                "upper": int(upper.id),
                "lower": int(lower.id),
                "lengthKm": segment_length,
                "ageMyr": age_myr,
            },
        )

    return started


def choose_foundering(world, plate_a, plate_b, segment):
    """Choose the overriding and downgoing sides from land share and crust age."""
    mesh = world.mesh
    xyz, index = mesh.xyz, mesh.index
    age_a = age_b = 0.0
    land_a = land_b = 0
    for world_cell in segment:
        offset = 3 * int(world_cell)
        point = (
            float(xyz[offset]),
            float(xyz[offset + 1]),
            float(xyz[offset + 2]),
        )
        cell_a = plate_a.cell_at(index, *point)
        cell_b = plate_b.cell_at(index, *point)
        crust_a, crust_b = plate_a.crust, plate_b.crust
        if crust_a.mask[cell_a]:
            age_a += float(crust_a.formedAt[cell_a])
            if crust_a.type[cell_a] != CrustType["OCEAN"]:
                land_a += 1
        if crust_b.mask[cell_b]:
            age_b += float(crust_b.formedAt[cell_b])
            if crust_b.type[cell_b] != CrustType["OCEAN"]:
                land_b += 1

    if land_a != land_b:
        return (plate_a, plate_b) if land_a > land_b else (plate_b, plate_a)
    return (plate_a, plate_b) if age_a <= age_b else (plate_b, plate_a)


def arm_margin(world, upper, lower, segment) -> None:
    """Arm a continuous trench on the overriding plate along a world segment."""
    mesh = world.mesh
    xyz, index = mesh.xyz, mesh.index
    config = world.config
    crust = upper.crust
    slab_km = float(config["subductionInitSlabKm"])

    for world_cell in segment:
        offset = 3 * int(world_cell)
        cell = upper.cell_at(
            index,
            float(xyz[offset]),
            float(xyz[offset + 1]),
            float(xyz[offset + 2]),
        )
        if not crust.mask[cell]:
            continue
        crust.arm_trench(cell, slab_km)

        for edge in range(
            int(mesh.adj_offset[cell]),
            int(mesh.adj_offset[cell + 1]),
        ):
            neighbour = int(mesh.adj_list[edge])
            if (
                crust.mask[neighbour]
                and crust.type[neighbour] == CrustType["OCEAN"]
                and is_on_segment(world, lower, neighbour, upper)
            ):
                crust.arm_trench(neighbour, slab_km)

    upper.dirty = True


def is_on_segment(world, lower, cell: int, upper) -> bool:
    """Return whether an upper-plate raster cell touches lower-plate crust."""
    mesh = world.mesh
    offset = 3 * int(cell)
    point = upper.to_world(
        float(mesh.xyz[offset]),
        float(mesh.xyz[offset + 1]),
        float(mesh.xyz[offset + 2]),
    )
    lower_cell = lower.cell_at(
        mesh.index,
        float(point[0]),
        float(point[1]),
        float(point[2]),
    )
    if lower.crust.mask[lower_cell]:
        return True
    return any(
        lower.crust.mask[int(mesh.adj_list[edge])]
        for edge in range(
            int(mesh.adj_offset[lower_cell]),
            int(mesh.adj_offset[lower_cell + 1]),
        )
    )


__all__ = [
    "arm_margin",
    "choose_foundering",
    "initiate_subduction",
    "is_on_segment",
    "founder_within_plates",
]


def founder_within_plates(world, dt: float) -> int:
    """Found old ocean along a large plate's own outer continental margin.

    A contiguous old-ocean coast is selected by age, length and a seeded
    per-Myr chance. The ocean immediately outboard of that coast is carved
    into a new mobile plate; the continental plate keeps the trench margin.
    """
    mesh, plates, config = world.mesh, world.plates, world.config
    time = float(world.time)
    if float(config["marginFounderRatePerMyr"]) <= 0.0:
        return 0

    num_cells = int(mesh.num_cells)
    offsets, adjacency = mesh.adj_offset, mesh.adj_list
    edge_km = float(mesh.edge_km)
    cell_area = float(mesh.cell_area_km2)
    minimum_cells = float(config["oceanPlateMinAreaKm2"]) / cell_area
    reach_hops = max(
        1,
        math.floor(float(config["oceanPlateReachKm"]) / edge_km + 0.5),
    )
    minimum_age = float(config["subductionInitAgeMyr"])
    minimum_length = float(config["subductionInitMinLengthKm"])
    age_cap = float(config["subductionInitAgeCap"])
    started = 0

    for plate in list(plates):
        if (
            not plate.alive
            or plate.isBackground
            or plate.crust.count == 0
            or int(plate.stats["continentalCells"]) <= 0
            or (int(plate.stats["cells"]) - int(plate.stats["continentalCells"]))
            < minimum_cells
        ):
            continue
        if (
            int(plate.stats["cells"])
            < float(config["marginFounderMinShare"]) * num_cells
        ):
            continue

        crust = plate.crust
        mask, kinds = crust.mask, crust.type
        flags, formed_at = crust.flags, crust.formedAt

        def land(cell: int) -> bool:
            return bool(
                mask[cell]
                and kinds[cell]
                in (
                    CrustType["CONTINENT"],
                    CrustType["CRATON"],
                )
            )

        coast = np.zeros(num_cells, dtype=np.uint8)
        old = []
        for cell in range(num_cells):
            if not mask[cell] or kinds[cell] != CrustType["OCEAN"]:
                continue
            for edge in range(int(offsets[cell]), int(offsets[cell + 1])):
                neighbour = int(adjacency[edge])
                if not land(neighbour):
                    continue
                coast[cell] = 1
                if (
                    not (int(flags[cell]) & CrustFlag["TRENCH"])
                    and time - float(formed_at[cell]) >= minimum_age
                ):
                    old.append(cell)
                break

        in_old = np.zeros(num_cells, dtype=np.uint8)
        run_of = np.full(num_cells, -1, dtype=np.int32)
        for cell in old:
            in_old[cell] = 1

        runs = []
        for start in old:
            if run_of[start] >= 0:
                continue
            run = [start]
            run_of[start] = len(runs)
            cursor = 0
            while cursor < len(run):
                cell = run[cursor]
                cursor += 1
                for edge in range(int(offsets[cell]), int(offsets[cell + 1])):
                    neighbour = int(adjacency[edge])
                    if in_old[neighbour] and run_of[neighbour] < 0:
                        run_of[neighbour] = len(runs)
                        run.append(neighbour)
            runs.append(run)

        for run in runs:
            length_km = len(run) * edge_km
            if length_km < minimum_length:
                continue
            age_myr = sum(time - float(formed_at[cell]) for cell in run) / len(run)
            chance = (
                float(config["marginFounderRatePerMyr"])
                * float(dt)
                * min(age_cap, age_myr / minimum_age - 1.0)
                * (length_km / minimum_length)
            )
            cycle = getattr(world, "cycle", None)
            if (
                cycle is not None
                and cycle.get("phase") == "dispersing"
                and age_myr <= time - float(cycle["breakupAt"]) + 1e-9
            ):
                chance *= float(config["disperseFounderFactor"])
            if float(world.rngMarginFounder()) >= chance:
                continue

            exterior = exterior_sea(mesh, crust)
            if not any(exterior[cell] for cell in run):
                continue

            labels = np.zeros(num_cells, dtype=np.uint8)
            mine = int(run_of[run[0]])
            frontier = []
            for cell in range(num_cells):
                if coast[cell]:
                    labels[cell] = 1 if int(run_of[cell]) == mine else 2
                    frontier.append(cell)

            take = []
            for _ in range(reach_hops):
                if not frontier:
                    break
                following = []
                for cell in frontier:
                    label = labels[cell]
                    for edge in range(int(offsets[cell]), int(offsets[cell + 1])):
                        neighbour = int(adjacency[edge])
                        if (
                            labels[neighbour]
                            or not mask[neighbour]
                            or kinds[neighbour] != CrustType["OCEAN"]
                        ):
                            continue
                        labels[neighbour] = label
                        following.append(neighbour)
                        if label == 1:
                            take.append(neighbour)
                frontier = following

            if len(take) < minimum_cells:
                continue

            fresh = world.add_plate(ocean_plate_name(world))
            fresh.set_quat(plate.quat.copy())
            fresh.omega = plate.omega.copy()
            fresh.isOceanBorn = True

            for tip, direction in list(plate.wings.items()):
                if labels[int(tip)] == 1 and not coast[int(tip)]:
                    fresh.wings[tip] = direction
                    del plate.wings[tip]

            moved = transfer_cells(world, plate, fresh, take)
            if moved == 0:
                fresh.alive = False
                continue

            fresh.born = time
            for cell in run:
                plate.crust.arm_trench(cell, float(config["subductionInitSlabKm"]))
            plate.dirty = True
            started += 1

            area_mkm2 = len(take) * cell_area / 1e6
            world.log(
                EventKind["PLATE_BORN"],
                (
                    f"{fresh.name} is born: {area_mkm2:.1f} Mkm² of "
                    f"{plate.name}’s old ocean breaks away"
                ),
                {"plate": fresh.id, "areaMkm2": area_mkm2},
            )
            world.log(
                EventKind["SUBDUCTION_INIT"],
                (
                    f"a new trench founders along {length_km:.0f} km of "
                    f"{plate.name}’s passive margin: {fresh.name}, ocean "
                    f"{age_myr:.0f} Myr old, begins to go down beneath it"
                ),
                {
                    "upper": plate.id,
                    "lower": fresh.id,
                    "lengthKm": length_km,
                    "ageMyr": age_myr,
                },
            )
            break

    return started
