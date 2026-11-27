"""Noise-shaped supercontinent assembly and sutures for the V2 start world.

Port of v2/js/tectonics/init/landmass.js at upstream v3.0.1.
"""

from __future__ import annotations

import math
from bisect import bisect_right
from collections.abc import Callable, Mapping
from typing import Any, Literal, TypedDict

import numpy as np

from ..core.mesh_bfs import flood_from, within_km
from ..core.sim_mesh import SimMesh
from ..core.simplex_noise import SimplexNoise
from ..core.sphere_geometry import (
    PLANET_RADIUS_KM,
    angle_between,
    polar_frame,
    tangent_basis,
)
from .crust import CrustType
from .initial_cratons import Craton, CratonLayout, craton_edge


class ClusterFrame(TypedDict):
    heart: list[float]
    e: list[float]
    n: list[float]
    bias: float
    towards: float
    ringKm: float


class LandmassResult(TypedDict):
    type: np.ndarray
    terrane: np.ndarray
    suture: np.ndarray
    owner: np.ndarray
    pair: np.ndarray
    pairRange: np.ndarray
    skirtKm: float
    rimCentre: list[float]
    passive: dict[str, float]


Seam = {"NONE": 0, "INNER": 1, "RING": 2, "MAJOR": 3}
MAX_RINGS = 4


def _ieee_div(numerator: float, denominator: float) -> float:
    """Division with IEEE results (no exception), as JavaScript `/`."""
    if denominator != 0.0:
        return numerator / denominator
    if numerator == 0.0 or math.isnan(numerator):
        return math.nan
    sign = math.copysign(1.0, numerator) * math.copysign(1.0, denominator)
    return math.inf if sign > 0.0 else -math.inf


def _ieee_exp(value: float) -> float:
    """Exponential with IEEE overflow (inf), as JavaScript `Math.exp`."""
    try:
        return math.exp(value)
    except OverflowError:
        return math.inf


def _ieee_log(value: float) -> float:
    """Natural log with IEEE results (-inf at 0, NaN below 0), as `Math.log`."""
    if math.isnan(value) or value < 0.0:
        return math.nan
    if value == 0.0:
        return -math.inf
    return math.log(value)


def _js_max(first: float, second: float) -> float:
    """`Math.max` for two values: NaN wins, and max(0, -0) is +0."""
    if math.isnan(first) or math.isnan(second):
        return math.nan
    if first == second == 0.0:
        return first if math.copysign(1.0, first) > 0 else second
    return first if first >= second else second


def _smoothstep(value: float) -> float:
    if value <= 0.0:
        return 0.0
    if value >= 1.0:
        return 1.0
    return value * value * (3.0 - 2.0 * value)


def _clamp(value: float, lower: float, upper: float) -> float:
    return min(upper, max(lower, value))


def _lerp(bounds: list[float] | tuple[float, float], value: float) -> float:
    return float(bounds[0]) + (float(bounds[1]) - float(bounds[0])) * value


def _heart_of(cratons: list[Craton]) -> list[float]:
    total = np.zeros(3, dtype=np.float64)
    for craton in cratons:
        weight = float(craton["radius"]) ** 2
        total += np.asarray(craton["centre"], dtype=np.float64) * weight
    length = float(np.linalg.norm(total)) or 1.0
    return (total / length).tolist()


def _margin_profile(
    rng: Callable[[], float],
    cfg: Mapping[str, Any],
    passive: Mapping[str, float],
) -> Callable[[float], dict[str, float]]:
    for key in ("startArcKm", "startStepKm"):
        if any(float(width) <= 0.0 for width in cfg[key]):
            raise ValueError(f"{key} widths must be positive")
    km_per_degree = (
        math.sin(math.acos(1.0 - 2.0 * float(cfg["continentalFraction"])))
        * 2.0
        * math.pi
        * PLANET_RADIUS_KM
        / 360.0
    )
    span = _clamp(float(passive["span"]), 0.0, 360.0)
    arcs: list[dict[str, float]] = []
    steps: list[dict[str, float]] = []
    offset = span
    while offset < 360.0:
        length = _lerp(cfg["startArcKm"], rng())
        radius = _lerp(cfg["startArcRadiusKm"], rng())
        width = length / km_per_degree
        arcs.append(
            {
                "from": offset,
                "to": min(360.0, offset + width),
                "height": min(0.2 * length, length * length / (8.0 * radius)),
            }
        )
        offset += width
    offset = 0.0
    while offset < span:
        width = _lerp(cfg["startStepKm"], rng()) / km_per_degree
        sign = -1.0 if rng() < 0.5 else 1.0
        steps.append(
            {
                "from": offset,
                "to": min(span, offset + width),
                "offset": sign * _lerp(cfg["startStepOffsetKm"], rng()),
            }
        )
        offset += width

    def at(azimuth: float) -> dict[str, float]:
        position = (
            (math.fmod(azimuth - float(passive["from"]), 360.0)) + 360.0
        ) % 360.0
        active = (
            1.0
            if span <= 0.0
            else 0.0
            if position < span
            else _smoothstep(min(position - span, 360.0 - position) / 4.0)
        )
        bulge = 0.0
        step_offset = 0.0
        if position >= span:
            # Arc intervals tile [span, 360); the index is therefore defined.
            arc_index = bisect_right([arc["from"] for arc in arcs], position) - 1
            arc = arcs[arc_index]
            ratio = (position - arc["from"]) / (arc["to"] - arc["from"])
            bulge = arc["height"] * (1.0 - (2.0 * ratio - 1.0) ** 2) * active
        else:
            # Step intervals tile [0, span) whenever this branch is reachable.
            segment_index = (
                bisect_right([segment["from"] for segment in steps], position) - 1
            )
            segment = steps[segment_index]
            following = (
                steps[segment_index + 1] if segment_index + 1 < len(steps) else None
            )
            blend = (
                _smoothstep((position - (segment["to"] - 0.5)) / 0.5)
                if following is not None
                else 0.0
            )
            next_offset = float(following["offset"]) if following else 0.0
            step_offset = (
                (float(segment["offset"]) * (1.0 - blend) + next_offset * blend)
                * (1.0 - active)
                * _smoothstep(min(position, span - position) / 4.0)
            )
        return {"bulgeKm": bulge, "stepKm": step_offset, "active": active}

    return at


def _bridges(
    sim_mesh: SimMesh, cratons: list[Craton], half_width: float
) -> np.ndarray:
    xyz = np.asarray(sim_mesh.xyz, dtype=np.float64).reshape((-1, 3))
    count = len(cratons)
    result = np.zeros(int(sim_mesh.num_cells), dtype=np.uint8)
    if count < 2:
        return result
    centres = [np.asarray(item["centre"], dtype=np.float64) for item in cratons]
    in_tree = [False] * count
    in_tree[0] = True
    edges = []
    for _ in range(1, count):
        # The complete craton graph always has an edge from the current tree
        # to an unvisited node, so seed the minimum with a valid candidate.
        first = next(index for index, included in enumerate(in_tree) if included)
        second = next(index for index, included in enumerate(in_tree) if not included)
        best = (
            first,
            second,
            angle_between(centres[first].tolist(), centres[second].tolist()),
        )
        for i in range(count):
            if not in_tree[i]:
                continue
            for j in range(count):
                if in_tree[j]:
                    continue
                distance = angle_between(centres[i].tolist(), centres[j].tolist())
                if distance < best[2]:
                    best = (i, j, distance)
        in_tree[best[1]] = True
        edges.append(best)

    for first, second, distance in edges:
        a, b = centres[first], centres[second]
        normal = np.cross(a, b)
        length = float(np.linalg.norm(normal)) or 1.0
        signed = np.clip((xyz @ normal) / length, -1.0, 1.0)
        cross_track = np.arcsin(signed)
        eligible = np.flatnonzero(np.abs(cross_track) <= half_width)
        if not eligible.size:
            continue
        # Triangle inequality rejects cells beyond either end of the segment.
        points = xyz[eligible]
        along_a = np.arccos(np.clip(points @ a, -1.0, 1.0))
        along_b = np.arccos(np.clip(points @ b, -1.0, 1.0))
        accepted = eligible[
            (along_a <= distance + half_width) & (along_b <= distance + half_width)
        ]
        result[accepted] = 1
    return result


def shape_landmass(
    sim_mesh: SimMesh,
    layout: CratonLayout,
    noise: SimplexNoise,
    rng: Callable[[], float],
    cfg: Mapping[str, Any],
    stretch: float = 1.0,
    passive: Mapping[str, float] | None = None,
) -> LandmassResult | None:
    """Turn a packed craton layout into connected, terrane-labelled land."""
    if passive is None:
        passive = {"from": 0.0, "span": 0.0}
    cratons = list(layout["cratons"])
    cluster_count = int(layout.get("clusters", 1))
    xyz = np.asarray(sim_mesh.xyz, dtype=np.float64).reshape((-1, 3))
    num_cells = int(sim_mesh.num_cells)

    def km(value: float | int) -> float:
        return float(value) / PLANET_RADIUS_KM

    edge_noise = km(cfg["cratonEdgeNoiseKm"])
    soft = km(cfg["coastUnionSoftnessKm"])
    wander = km(cfg["terraneBorderWanderKm"])
    grain_frequency = PLANET_RADIUS_KM / float(cfg["cratonGrainNoiseKm"][0])
    across_frequency = PLANET_RADIUS_KM / float(cfg["cratonGrainNoiseKm"][1])
    seam_half = km(cfg["sutureHalfWidthKm"])
    core = km(cfg["startCoreKm"])
    seam_wander = km(cfg["startSeamWanderKm"])
    core_soft = 0.55 * km(
        float(cfg["cratonMinGapKm"]) + 2.0 * float(cfg["cratonEdgeNoiseKm"])
    )
    small_craton = km(800.0)
    clusters: list[ClusterFrame] = []
    for cluster in range(cluster_count):
        members = [item for item in cratons if int(item.get("cluster", 0)) == cluster]
        heart = _heart_of(members)
        east, north = tangent_basis(heart)
        clusters.append(
            {
                "heart": heart,
                "e": east,
                "n": north,
                "bias": float(cfg["startGrowthBias"]) * (0.25 + 0.75 * rng()),
                "towards": rng() * 2.0 * math.pi,
                "ringKm": _lerp(cfg["startRingKm"], rng()),
            }
        )
    rim_centre = _heart_of(cratons)
    reference = tangent_basis(rim_centre)[0]
    to_polar = polar_frame(rim_centre, reference)
    margin_at = _margin_profile(rng, cfg, passive)

    types = np.full(num_cells, CrustType["OCEAN"], dtype=np.uint8)
    terrane = np.zeros(num_cells, dtype=np.uint16)
    suture = np.zeros(num_cells, dtype=np.uint8)
    owner = np.full(num_cells, -1, dtype=np.int8)
    rings = np.zeros(num_cells, dtype=np.uint8)
    pairs = np.zeros(num_cells, dtype=np.uint16)
    field = np.zeros(num_cells, dtype=np.float32)
    shape = np.zeros(num_cells, dtype=np.float32)
    growth = np.zeros(cluster_count, dtype=np.float64)
    weighted = np.zeros(cluster_count, dtype=np.float64)
    true_distance = np.zeros(cluster_count, dtype=np.float64)
    round_span = math.acos(1.0 - 2.0 * float(cfg["continentalFraction"]))

    for cell in range(num_cells):
        x, y, z = (float(value) for value in xyz[cell])
        growth.fill(0.0)
        nearest = math.inf
        nearest_id = 0
        nearest_rift = along = across = size = 0.0
        for craton in cratons:
            edge = craton_edge(craton, x, y, z)
            distance = edge["d"]
            if distance < nearest:
                nearest = distance
                nearest_id = int(craton["id"])
                nearest_rift = edge["rift"]
                along, across = edge["along"], edge["across"]
                size = float(craton["radius"])
            cluster = int(craton.get("cluster", 0))
            growth[cluster] += math.exp(-max(distance, -core_soft * 4.0) / core_soft)

        first = 0
        second = -1
        for cluster in range(cluster_count):
            frame = clusters[cluster]
            theta = math.atan2(
                x * frame["n"][0] + y * frame["n"][1] + z * frame["n"][2],
                x * frame["e"][0] + y * frame["e"][1] + z * frame["e"][2],
            )
            growth[cluster] = true_distance[cluster] = (
                -core_soft * math.log(growth[cluster])
                if growth[cluster] > 0.0
                else math.inf
            )
            if growth[cluster] > 0.0:
                growth[cluster] /= 1.0 + float(frame["bias"]) * math.cos(
                    theta - float(frame["towards"])
                )
            value = growth[cluster] + seam_wander * (
                noise.noise3D(
                    x * 1.5 + 11.3 * cluster,
                    y * 1.5 - 5.1 * cluster,
                    z * 1.5 + 2.9 * cluster,
                )
                + 0.2
                * noise.noise3D(
                    x * 4.5 - 3.7 * cluster,
                    y * 4.5 + 8.2 * cluster,
                    z * 4.5 + 1.3 * cluster,
                )
            )
            weighted[cluster] = value
            if cluster > 0 and value < weighted[first]:
                second, first = first, cluster
            elif cluster != first and (second < 0 or value < weighted[second]):
                second = cluster

        owner[cell] = first
        if second >= 0 and weighted[second] - weighted[first] < 2.0 * seam_half:
            suture[cell] = Seam["MAJOR"]
            pairs[cell] = 1 + min(first, second) * cluster_count + max(first, second)

        if true_distance[first] < core:
            nearest_one = math.inf
            nearest_two = math.inf
            who = 0
            for craton in cratons:
                if int(craton.get("cluster", 0)) != first:
                    continue
                craton_id = int(craton["id"])
                border = craton_edge(craton, x, y, z)["d"] + wander * noise.noise3D(
                    x * 2.2 + 7.3 * craton_id,
                    y * 2.2 - 3.1 * craton_id,
                    z * 2.2 + 1.7 * craton_id,
                )
                if border < nearest_one:
                    nearest_two, nearest_one, who = nearest_one, border, craton_id
                elif border < nearest_two:
                    nearest_two = border
            terrane[cell] = who
            if (
                suture[cell] == Seam["NONE"]
                and nearest_two - nearest_one < 2.0 * seam_half
            ):
                suture[cell] = Seam["INNER"]
        else:
            ring = 1 + math.floor(
                (true_distance[first] - core) / km(clusters[first]["ringKm"])
                + 0.35
                * noise.noise3D(
                    x * 3.0 + 4.4 * first, y * 3.0 + 1.7, z * 3.0 - 2.2 * first
                )
            )
            ring = max(1, min(MAX_RINGS, ring))
            rings[cell] = ring
            terrane[cell] = len(cratons) + 1 + first * MAX_RINGS + ring - 1

        wobble = (
            _clamp(
                1.4
                * noise.fbm(
                    along * grain_frequency + 3.1 * nearest_id,
                    across * across_frequency - 1.7 * nearest_id,
                    5.3 + 0.37 * nearest_id,
                    3,
                    0.55,
                ),
                -1.0,
                1.0,
            )
            * edge_noise
            * (1.0 - (1.0 - float(cfg["cratonRiftNoise"])) * nearest_rift)
            * min(1.0, size / small_craton)
        )
        if nearest + wobble < 0.0:
            types[cell] = CrustType["CRATON"]
            terrane[cell] = nearest_id

        # IEEE arithmetic as in upstream JavaScript: with coastUnionSoftnessKm = 0
        # the divisions give NaN instead of raising, and the field becomes NaN.
        union = 0.0
        for index in range(cluster_count):
            union += _ieee_exp(_ieee_div(-_js_max(growth[index], -soft * 4.0), soft))
        field[cell] = -soft * _ieee_log(union)
        polar = to_polar(x, y, z)
        at = margin_at(float(polar["azimuth"]))
        outer = _smoothstep(
            (float(polar["colat"]) * math.pi / 180.0 - 0.45 * round_span)
            / (0.35 * round_span)
        )
        detail = (
            km(cfg["startCoastDetailKm"])
            * (1.0 - 0.6 * at["active"])
            * _clamp(
                1.3 * noise.fbm(x * 9.0 + 23.0, y * 9.0 + 23.0, z * 9.0 + 23.0, 3, 0.5),
                -1.0,
                1.0,
            )
        )
        shape[cell] = (
            outer * (km(at["stepKm"]) - km(at["bulgeKm"]))
            + detail
            + km(cfg["coastReliefKm"])
            * (0.5 + 0.5 * stretch)
            * _clamp(
                1.3 * noise.noise3D(x * 0.9 + 31.0, y * 0.9 + 31.0, z * 0.9 + 31.0),
                -1.0,
                1.0,
            )
        )

    for cell in range(num_cells):
        if suture[cell] or rings[cell] < 2:
            continue
        lo = int(sim_mesh.adj_offset[cell])
        hi = int(sim_mesh.adj_offset[cell + 1])
        for other_value in sim_mesh.adj_list[lo:hi]:
            other = int(other_value)
            if owner[other] == owner[cell] and rings[other] < rings[cell]:
                suture[cell] = Seam["RING"]
                break

    near_craton = within_km(
        sim_mesh,
        lambda cell: types[cell] == CrustType["CRATON"],
        float(cfg["cratonMinCoastKm"]) + float(sim_mesh.edge_km),
    )
    bridge = _bridges(sim_mesh, cratons, km(float(cfg["bridgeHalfWidthKm"])))
    antipode = int(np.argmax(xyz @ -np.asarray(layout["centre"], dtype=np.float64)))
    heart_cell = int(
        np.argmax(xyz @ np.asarray(cratons[0]["centre"], dtype=np.float64))
    )

    # JavaScript reads Float32Array values as Numbers before adding them.
    # Keep this sum in float64 instead of rounding it back to float32 as NumPy does.
    coast_level = field.astype(np.float64) + shape.astype(np.float64)

    def landmass(skirt: float) -> tuple[np.ndarray, int] | Literal[False] | None:
        made = types.copy()
        add_land = (near_craton != 0) | (bridge != 0) | (coast_level < skirt)
        made[(made != CrustType["CRATON"]) & add_land] = CrustType["CONTINENT"]
        if made[antipode] != CrustType["OCEAN"]:
            return None
        open_ocean = flood_from(
            sim_mesh, antipode, lambda cell: made[cell] == CrustType["OCEAN"]
        )
        enclosed = (made == CrustType["OCEAN"]) & (open_ocean == 0)
        made[enclosed] = CrustType["CONTINENT"]
        mainland = flood_from(
            sim_mesh, heart_cell, lambda cell: made[cell] != CrustType["OCEAN"]
        )
        land_cells = made != CrustType["OCEAN"]
        if np.any((made == CrustType["CRATON"]) & (mainland == 0)):
            return False
        made[land_cells & (mainland == 0)] = CrustType["OCEAN"]
        return made, int(np.count_nonzero(made != CrustType["OCEAN"]))

    lower = float(np.min(coast_level)) - 0.01
    upper = 0.7
    for _ in range(24):
        skirt = (lower + upper) / 2.0
        made = landmass(skirt)
        # As upstream (`if (!made) return null;`): an antipode on land or a
        # craton cut off from the main landmass rejects the whole attempt.
        if made is None or made is False:
            return None
        if made[1] / num_cells < float(cfg["continentalFraction"]):
            lower = skirt
        else:
            upper = skirt
    skirt = (lower + upper) / 2.0
    final_land = landmass(skirt)
    if final_land is None or final_land is False:
        return None
    types[:] = final_land[0]
    ocean = types == CrustType["OCEAN"]
    terrane[ocean] = 0
    suture[ocean] = Seam["NONE"]
    owner[ocean] = -1
    suture[types == CrustType["CRATON"]] = Seam["NONE"]

    major_pairs = sorted(int(value) for value in np.unique(pairs) if value > 0)
    pair_range = np.zeros(1 + cluster_count * cluster_count, dtype=np.float32)
    last = major_pairs[math.floor(rng() * len(major_pairs))] if major_pairs else 0
    for pair in major_pairs:
        pair_range[pair] = 1.0 if pair == last else 0.25 + 0.45 * rng()
    pairs[suture != Seam["MAJOR"]] = 0
    return {
        "type": types,
        "terrane": terrane,
        "suture": suture,
        "owner": owner,
        "pair": pairs,
        "pairRange": pair_range,
        "skirtKm": skirt * PLANET_RADIUS_KM,
        "rimCentre": rim_centre,
        "passive": dict(passive),
    }
