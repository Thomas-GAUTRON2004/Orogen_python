"""Mantle-plume lifecycle, LIP provinces, and volcanic tracks from V2 hotspots.js.

Execution of queued plume-triggered rifts remains a later tectonic stage.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from ..core.fast_noise import FastNoise
from ..core.seed import make_rng
from ..core.sphere_geometry import PLANET_RADIUS_KM, normalize, tangent_basis
from .crust import CrustFlag, CrustType, VolcanicType
from .thermal import (
    insulate_at_start,
    pick_plume_site,
    plume_heat_factor,
    update_mantle_heat,
)
from .world import EventKind

_ONSETS = (
    "K",
    "V",
    "T",
    "R",
    "M",
    "S",
    "L",
    "N",
    "D",
    "H",
    "Th",
    "Z",
    "B",
    "G",
    "Sh",
    "F",
    "P",
    "Kh",
)
_NUCLEI = ("a", "e", "i", "o", "u", "a", "o", "ai", "au", "ei")
_CODAS = ("", "", "n", "r", "l", "s", "th", "k", "m", "rn")


def _value(record, name: str, default=None):
    return (
        record.get(name, default)
        if isinstance(record, dict)
        else getattr(record, name, default)
    )


def _assign(record, name: str, value) -> None:
    if isinstance(record, dict):
        record[name] = value
    else:
        setattr(record, name, value)


def _plume_name(world) -> str:
    rng = world.rngHotspot
    for _ in range(30):
        syllables = 3 if rng() < 0.3 else 2
        name = "".join(
            _ONSETS[int(rng() * len(_ONSETS))]
            + _NUCLEI[int(rng() * len(_NUCLEI))]
            + (_CODAS[int(rng() * len(_CODAS))] if i == syllables - 1 else "")
            for i in range(syllables)
        )
        name = name[0] + name[1:].lower()
        if not any(_value(h, "name") == name for h in world.hotspots):
            return name
    return f"Plume {len(world.hotspots) + 1}"


def _nearest_cell(mesh, xyz: np.ndarray) -> int:
    if hasattr(mesh, "index") and hasattr(mesh.index, "nearest"):
        return int(mesh.index.nearest(*map(float, xyz)))
    points = np.asarray(mesh.xyz, dtype=np.float64).reshape(-1, 3)
    if points.shape[0] != int(mesh.num_cells):
        raise ValueError("mesh coordinates must match num_cells")
    return int(np.argmax(points @ xyz))


def _continental_area_km2(plate, mesh) -> float:
    crust = plate.crust
    mask = np.asarray(crust.mask, dtype=bool)
    continental = mask & (
        (crust.type == CrustType["CONTINENT"]) | (crust.type == CrustType["CRATON"])
    )
    return float(np.count_nonzero(continental)) * float(mesh.cell_area_km2)


def birth_plume(
    world,
    xyz,
    *,
    head: bool = True,
    rift="chance",
    rift_plate: int = -1,
    rift_cell: int = -1,
    name: str | None = None,
    life_myr: float | None = None,
    strength: float | None = None,
    born_at: float | None = None,
) -> dict:
    """Register a mantle-frame plume, erupt its head LIP, and optionally queue a rift.

    Active plume tails stamp tracks and pulse-built edifices; queued rift execution
    remains a separate tectonic stage.
    """
    position = np.asarray(xyz, dtype=np.float64).reshape(-1)
    if position.size != 3 or not np.all(np.isfinite(position)):
        raise ValueError("plume xyz must contain three finite coordinates")
    length = float(np.linalg.norm(position))
    if length <= 0.0:
        raise ValueError("plume xyz must have non-zero length")
    position /= length

    if not hasattr(world, "hotspots"):
        world.hotspots = []
    if not hasattr(world, "pendingPlumeRifts"):
        world.pendingPlumeRifts = []
    rng = world.rngHotspot
    config = world.config
    start = float(world.time if born_at is None else born_at)
    plume_name = str(name) if name is not None else _plume_name(world)
    if life_myr is None:
        minimum = float(config["plumeLifeMinMyr"])
        maximum = float(config["plumeLifeMaxMyr"])
        life = minimum + float(rng()) * (maximum - minimum)
    else:
        life = float(life_myr)
    plume_strength = 0.6 + 0.8 * float(rng()) if strength is None else float(strength)
    hotspot_id = len(world.hotspots)
    plume = {
        "id": hotspot_id,
        "name": plume_name,
        "xyz": position.tolist(),
        "bornAt": start,
        "dieAt": start + life,
        "alive": True,
        "lip": None,
        "strength": plume_strength,
        "pulseUntil": None,
        "bursting": False,
        "pulseIntensity": 0.0,
    }
    world.hotspots.append(plume)

    cell = _nearest_cell(world.mesh, position)
    host_id = int(world.owner[cell])
    host = world.plates[host_id] if 0 <= host_id < len(world.plates) else None
    host_name = (
        host.name if host is not None and not host.isBackground else "the open ocean"
    )
    world.log(
        EventKind["PLUME_RISE"],
        f"the {plume_name} plume {'rises' if head else 'is already rising'} beneath {host_name}",
        {
            "hotspot": hotspot_id,
            "name": plume_name,
            "plate": host_id if host is not None and not host.isBackground else -1,
            "head": bool(head),
        },
    )

    if head:
        _erupt_lip(world, plume, cell)

    if rift is not False and host is not None and not host.isBackground:
        target_id = int(rift_plate) if int(rift_plate) >= 0 else host_id
        target = world.plates[target_id] if 0 <= target_id < len(world.plates) else None
        if target is not None:
            forced = rift is True
            eligible = _continental_area_km2(target, world.mesh) >= float(
                config.get("riftMinContinentKm2", math.inf)
            )
            chance = (
                float(config.get("plumeRiftChance", 0.0))
                * plume_heat_factor(world, target)
                if eligible and not forced
                else 0.0
            )
            if forced or (rift == "chance" and eligible and float(rng()) < chance):
                world.pendingPlumeRifts.append(
                    {
                        "plate": target.id,
                        "xyz": position.tolist(),
                        "cell": int(rift_cell),
                        "cause": {
                            "text": f"the {plume_name} plume",
                            "kind": "plume",
                            "hotspot": hotspot_id,
                        },
                    }
                )
    return plume


def update_plumes(world, dt: float) -> dict:
    """Update mantle heat and plume birth/death records for one simulation step.

    Scheduled entries may specify xyz or a plate/cell pair. Random births use the
    dedicated hotspot RNG stream. Head births construct LIPs; active tails stamp
    tracks and pulse-built edifices. Queued rifts remain for a later stage.
    """
    step = float(dt)
    if not math.isfinite(step) or step < 0.0:
        raise ValueError("dt must be finite and non-negative")

    config = world.config
    if not getattr(world, "hotspotsReady", False):
        insulate_at_start(world)
        world.hotspotsReady = True
        initial_count = int(config.get("hotspotInitialCount", 0))
        for _ in range(max(0, initial_count)):
            cell = pick_plume_site(world, world.rngHotspot)
            if cell < 0:
                break
            age = float(world.rngHotspot()) * float(config["plumeLifeMinMyr"]) * 0.6
            point = np.asarray(world.mesh.xyz, dtype=np.float64).reshape(-1, 3)[cell]
            birth_plume(
                world, point, head=False, rift=False, born_at=float(world.time) - age
            )

    update_mantle_heat(world, step)

    scheduled = getattr(world, "plumes", ())
    for item in scheduled:
        item_time = float(_value(item, "time", world.time))
        if _value(item, "done", False) or item_time > float(world.time) + 1e-9:
            continue
        _assign(item, "done", True)
        scheduled_xyz = _value(item, "xyz")
        if scheduled_xyz is not None:
            birth_plume(
                world,
                scheduled_xyz,
                head=_value(item, "head", True),
                rift=_value(item, "rift", "chance"),
                name=_value(item, "name"),
                life_myr=_value(item, "lifeMyr"),
                strength=_value(item, "strength"),
                born_at=item_time,
            )
            continue

        plate_id = int(_value(item, "plate", -1))
        cell = int(_value(item, "cell", -1))
        if not 0 <= plate_id < len(world.plates):
            continue
        plate = world.plates[plate_id]
        if not plate.alive or not 0 <= cell < int(world.mesh.num_cells):
            continue
        point = np.asarray(world.mesh.xyz, dtype=np.float64).reshape(-1, 3)[cell]
        mantle_position = plate.to_world(*map(float, point))
        birth_plume(
            world,
            mantle_position,
            head=True,
            rift=True,
            rift_plate=plate_id,
            rift_cell=cell,
            born_at=item_time,
        )

    rate = float(config.get("plumeRatePerMyr", 0.0))
    rng = getattr(world, "rngHotspot", None)
    if rate > 0.0 and rng is not None and float(rng()) < rate * step:
        cell = pick_plume_site(world, rng)
        if cell >= 0:
            point = np.asarray(world.mesh.xyz, dtype=np.float64).reshape(-1, 3)[cell]
            birth_plume(world, point, head=True, rift="chance")

    expired = 0
    for plume in getattr(world, "hotspots", ()):
        if not _value(plume, "alive", True):
            continue
        if float(world.time) >= float(_value(plume, "dieAt", math.inf)):
            _assign(plume, "alive", False)
            born = float(_value(plume, "bornAt", world.time))
            name = str(_value(plume, "name", "unknown"))
            life = float(world.time) - born
            world.log(
                EventKind["PLUME_DEATH"],
                f"the {name} hotspot dies out after {life:.0f} Myr",
                {
                    "hotspot": int(_value(plume, "id", -1)),
                    "name": name,
                    "lifeMyr": life,
                },
            )
            expired += 1
            continue
        _stamp_track(world, plume, step)

    alive = sum(
        1 for plume in getattr(world, "hotspots", ()) if _value(plume, "alive", True)
    )
    return {
        "born": len(getattr(world, "hotspots", ())),
        "expired": expired,
        "alive": alive,
    }


def _erupt_lip(world, plume: dict, anchor: int) -> None:
    """Rasterize a plume-head flood-basalt province from upstream hotspots.js."""
    config, mesh = world.config, world.mesh
    owner, owner_cell = world.owner, world.ownerCell
    provisional = world.provisional
    xyz = np.asarray(mesh.xyz).reshape(-1)
    adj_offset, adj_list = mesh.adj_offset, mesh.adj_list
    cell_area = float(mesh.cell_area_km2)

    minimum = float(config["lipAreaMinMkm2"])
    maximum = float(config["lipAreaMaxMkm2"])
    if minimum <= 0.0 or maximum < minimum:
        raise ValueError("LIP area bounds must be positive and ordered")

    u = float(world.rngHotspot())
    rv = world.rngVolcano
    ragged = 0.4 + 0.6 * float(rv())
    lobes = [
        {
            "k": k,
            "amp": ragged * float(rv()) * float(config["lipLobes"]) / math.sqrt(k),
            "phase": float(rv()) * 2.0 * math.pi,
        }
        for k in range(1, 6)
    ]
    noise_seed = math.floor(float(rv()) * 1e9)
    noise = FastNoise(noise_seed)
    rs = make_rng((noise_seed ^ 0x11B5) & 0xFFFFFFFF)
    area_km2 = 1e6 * minimum * (maximum / minimum) ** u
    radius = math.sqrt(area_km2 / math.pi) / PLANET_RADIUS_KM

    centre = np.asarray(plume["xyz"], dtype=np.float64)
    axis = float(rs()) * math.pi
    basis1, basis2 = tangent_basis(tuple(map(float, centre)))
    flow_count = 1 + math.floor(float(rs()) * float(rs()) * 4.0)
    flows: list[dict[str, Any]] = []
    total_weight = 0.0
    for i in range(flow_count):
        weight = 1.0 if i == 0 else 0.25 + 0.6 * float(rs())
        if i == 0:
            along = 0.0
        else:
            along_distance = 0.6 + 0.6 * float(rs())
            direction_sign = -1.0 if float(rs()) < 0.5 else 1.0
            along = along_distance * direction_sign
        offset = 0.0 if i == 0 else radius * along
        direction = axis + (float(rs()) - 0.5) * 1.2
        cosine_offset, sine_offset = math.cos(offset), math.sin(offset)
        flow_centre = normalize(
            [
                cosine_offset * centre[0]
                + sine_offset
                * (math.cos(direction) * basis1[0] + math.sin(direction) * basis2[0]),
                cosine_offset * centre[1]
                + sine_offset
                * (math.cos(direction) * basis1[1] + math.sin(direction) * basis2[1]),
                cosine_offset * centre[2]
                + sine_offset
                * (math.cos(direction) * basis1[2] + math.sin(direction) * basis2[2]),
            ]
        )
        aspect_a = 1.0 + 2.0 * float(rs()) * float(rs())
        turn = (axis if i == 0 else direction) + (float(rs()) - 0.5) * 0.8
        flow_basis1, flow_basis2 = tangent_basis(flow_centre)
        flows.append(
            {
                "f1": flow_basis1,
                "f2": flow_basis2,
                "w": weight,
                "aspect": aspect_a,
                "cos": math.cos(turn),
                "sin": math.sin(turn),
            }
        )
        total_weight += weight

    for flow in flows:
        flow_radius = radius * math.sqrt(1.25 * flow["w"] / total_weight)
        flow["a"] = flow_radius * math.sqrt(flow["aspect"])
        flow["b"] = flow_radius / math.sqrt(flow["aspect"])

    warp_frequency = 2.5 / radius
    edge_frequency = 5.0 / radius

    def depth(cell: int) -> float:
        offset = 3 * cell
        x, y, z = float(xyz[offset]), float(xyz[offset + 1]), float(xyz[offset + 2])
        wx = (
            0.35
            * radius
            * noise.fbm(
                x * warp_frequency + 3.1, y * warp_frequency, z * warp_frequency, 2
            )
        )
        wy = (
            0.35
            * radius
            * noise.fbm(
                x * warp_frequency, y * warp_frequency + 5.3, z * warp_frequency, 2
            )
        )
        wz = (
            0.35
            * radius
            * noise.fbm(
                x * warp_frequency, y * warp_frequency, z * warp_frequency + 7.7, 2
            )
        )
        x, y, z = x + wx, y + wy, z + wz
        best = -math.inf
        for flow in flows:
            px = x * flow["f1"][0] + y * flow["f1"][1] + z * flow["f1"][2]
            py = x * flow["f2"][0] + y * flow["f2"][1] + z * flow["f2"][2]
            pu = px * flow["cos"] + py * flow["sin"]
            pv = -px * flow["sin"] + py * flow["cos"]
            q = math.hypot(pu / flow["a"], pv / flow["b"])
            best = max(best, 1.0 - q)

        theta = math.atan2(
            x * basis2[0] + y * basis2[1] + z * basis2[2],
            x * basis1[0] + y * basis1[1] + z * basis1[2],
        )
        lobe = sum(
            item["amp"] * math.cos(item["k"] * theta + item["phase"]) for item in lobes
        )
        return (
            best
            + 0.5 * lobe
            + float(config["lipEdgeNoise"])
            * noise.fbm(x * edge_frequency, y * edge_frequency, z * edge_frequency, 3)
        )

    depth_cache: dict[int, float] = {}

    def cell_depth(cell: int) -> float:
        if cell not in depth_cache:
            depth_cache[cell] = depth(cell)
        return depth_cache[cell]

    cells = 0
    land_cells = 0
    seen = {int(anchor)}
    queue = [int(anchor)]
    for cell in queue:
        if int(owner[cell]) >= 0 and not provisional[cell]:
            plate_id = int(owner[cell])
            crust_cell = int(owner_cell[cell])
            crust = world.plates[plate_id].crust
            d = 1.0 - max(0.0, min(1.0, cell_depth(cell)))
            inner = 1.0 - d
            thickness = (
                float(config["lipThicknessKm"])
                * (0.8 + 0.2 * inner)
                * (0.35 + 0.65 * min(1.0, inner / 0.15))
            )
            crust.volcType[crust_cell] = VolcanicType["LIP"]
            crust.volcAt[crust_cell] = world.time
            crust.volcKm[crust_cell] = max(float(crust.volcKm[crust_cell]), thickness)
            crust.volcSource[crust_cell] = int(plume["id"]) + 1
            crust.flags[crust_cell] |= CrustFlag["LIP"]
            cells += 1
            if crust.type[crust_cell] != CrustType["OCEAN"]:
                land_cells += 1

        for edge in range(int(adj_offset[cell]), int(adj_offset[cell + 1])):
            neighbour = int(adj_list[edge])
            if neighbour not in seen and cell_depth(neighbour) > 0.0:
                seen.add(neighbour)
                queue.append(neighbour)

    on_land = land_cells * 2 > cells
    host_id = int(owner[anchor])
    host_plate = world.plates[host_id] if 0 <= host_id < len(world.plates) else None
    host_name = (
        host_plate.name
        if host_plate is not None and not host_plate.isBackground
        else "the open ocean"
    )
    suffix = "Traps" if on_land else "Plateau"
    lip_name = str(plume["name"]) + " " + suffix
    lip_area = cells * cell_area / 1e6
    plume["lip"] = {
        "name": lip_name,
        "time": float(world.time),
        "radiusKm": radius * PLANET_RADIUS_KM,
        "flows": flow_count,
        "cells": cells,
        "onLand": on_land,
    }
    verb = "erupt" if on_land else "erupts"
    world.log(
        EventKind["LIP"],
        f"the {lip_name} {verb}: {lip_area:.1f} Mkm² of flood basalt on {host_name}",
        {
            "hotspot": int(plume["id"]),
            "name": lip_name,
            "onLand": on_land,
            "areaMkm2": lip_area,
            "plate": host_id
            if host_plate is not None and not host_plate.isBackground
            else -1,
        },
    )


def _erupted_during(world, plume: dict, start: float, end: float) -> float:
    """Integrate plume burst intensity over a simulation interval."""
    config = world.config
    rng = world.rngVolcano
    plume.setdefault("pulseUntil", None)
    plume.setdefault("bursting", False)
    plume.setdefault("pulseIntensity", 0.0)
    if plume["pulseUntil"] is None:
        plume["bursting"] = False
        plume["pulseUntil"] = start + float(rng()) * float(config["hotspotQuietMaxMyr"])

    total = 0.0
    time = start
    while time < end:
        pulse_end = min(end, float(plume["pulseUntil"]))
        if plume["bursting"]:
            total += float(plume["pulseIntensity"]) * (pulse_end - time)
        if pulse_end < float(plume["pulseUntil"]):
            break

        if plume["bursting"]:
            plume["bursting"] = False
            plume["pulseUntil"] = (
                pulse_end
                + float(config["hotspotQuietMinMyr"])
                + float(rng())
                * (
                    float(config["hotspotQuietMaxMyr"])
                    - float(config["hotspotQuietMinMyr"])
                )
            )
        else:
            plume["bursting"] = True
            plume["pulseIntensity"] = 0.5 + 0.9 * float(rng())
            plume["pulseUntil"] = (
                pulse_end
                + float(config["hotspotBurstMinMyr"])
                + float(rng())
                * (
                    float(config["hotspotBurstMaxMyr"])
                    - float(config["hotspotBurstMinMyr"])
                )
            )
        time = pulse_end
    return total


def _stamp_track(world, plume: dict, dt: float) -> None:
    """Stamp the plume's moving tail on the plate currently passing above it."""
    config, mesh = world.config, world.mesh
    owner, provisional = world.owner, world.provisional
    xyz = np.asarray(mesh.xyz).reshape(-1)
    anchor = int(mesh.index.nearest(*map(float, plume["xyz"])))
    if int(owner[anchor]) < 0 or provisional[anchor]:
        return

    plate = world.plates[int(owner[anchor])]
    crust = plate.crust
    crust_anchor = plate.cell_at(mesh.index, *map(float, plume["xyz"]))
    if not crust.mask[crust_anchor]:
        return

    local = plate.to_plate_frame(*map(float, plume["xyz"]))
    reach = math.cos(float(config["hotspotTrackKm"]) / PLANET_RADIUS_KM)
    edifice = math.cos(float(config["hotspotEdificeKm"]) / PLANET_RADIUS_KM)

    def closeness(cell: int) -> float:
        offset = 3 * cell
        return (
            float(xyz[offset]) * float(local[0])
            + float(xyz[offset + 1]) * float(local[1])
            + float(xyz[offset + 2]) * float(local[2])
        )

    cells = [crust_anchor]
    seen = {crust_anchor}
    for cell in cells:
        for edge in range(int(mesh.adj_offset[cell]), int(mesh.adj_offset[cell + 1])):
            neighbour = int(mesh.adj_list[edge])
            if (
                neighbour not in seen
                and crust.mask[neighbour]
                and closeness(neighbour) >= reach
            ):
                seen.add(neighbour)
                cells.append(neighbour)

    build = (
        float(config["hotspotBuildKmPerMyr"])
        * float(plume.get("strength", 0.6))
        * _erupted_during(world, plume, float(world.time) - dt, float(world.time))
    )
    for cell in cells:
        if crust.volcType[cell] != VolcanicType["LIP"]:
            crust.volcType[cell] = VolcanicType["TRACK"]
        crust.volcAt[cell] = world.time
        crust.volcSource[cell] = int(plume["id"]) + 1
        if build > 0.0 and (cell == crust_anchor or closeness(cell) >= edifice):
            crust.volcKm[cell] = max(
                float(crust.volcKm[cell]),
                min(
                    float(config["hotspotMaxKm"]),
                    float(crust.volcKm[cell]) + build,
                ),
            )
