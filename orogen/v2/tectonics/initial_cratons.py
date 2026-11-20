"""Upstream-compatible craton packing for the V2 initial-world generator.

Port of v2/js/tectonics/init/cratons.js at upstream v3.0.1.
Distances in the block frame are azimuthal-equidistant radians.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from typing import Any

import numpy as np

from ..core.sphere_geometry import (
    PLANET_RADIUS_KM,
    angle_between,
    bearing_to,
    cap_fraction,
    destination,
    random_point,
    tangent_basis,
)

TAU = 2.0 * math.pi
OUTLINE_SAMPLES = 72
CORNER = 0.11
CLUSTER_PULL = 0.8
MIN_EDGE_TURN = 0.9
STEP_WIDTH = 0.04


def _smoothstep(value: float) -> float:
    if value <= 0.0:
        return 0.0
    if value >= 1.0:
        return 1.0
    return value * value * (3.0 - 2.0 * value)


def _smax(a: float, b: float, width: float) -> float:
    h = max(width - abs(a - b), 0.0) / width
    return max(a, b) + h * h * width * 0.25


def _piece_shape(piece: Mapping[str, Any], u: float, v: float) -> dict[str, float]:
    body = piece["body"]
    du = u - float(body["u0"])
    dv = v - float(body["v0"])
    angle = math.atan2(dv, du) - float(body["psi"])
    cosine, sine = math.cos(angle), math.sin(angle)
    lobe = 1.0
    for lobe_def in body["lobes"]:
        lobe += float(lobe_def["amp"]) * math.cos(
            float(lobe_def["k"]) * angle + float(lobe_def["phase"])
        )
    radius = (
        float(body["a"])
        * float(body["b"])
        / math.sqrt(
            float(body["b"]) ** 2 * cosine**2
            + float(body["a"]) ** 2 * sine**2
        )
        * lobe
    )
    body_distance = math.hypot(du, dv) - radius
    distance = body_distance
    cut = -math.inf
    for edge in piece["cuts"]:
        p = u * float(edge["cs"]) + v * float(edge["sn"])
        q = v * float(edge["cs"]) - u * float(edge["sn"])
        at = float(edge["d"])
        for step in edge["steps"]:
            at += float(step["jump"]) * _smoothstep(
                (q - float(step["at"])) / STEP_WIDTH + 0.5
            )
        edge_distance = at - p if edge["flip"] else p - at
        cut = max(cut, edge_distance)
        distance = _smax(distance, edge_distance, CORNER)
    rift = 0.0 if cut == -math.inf else _smoothstep(
        (cut - body_distance) / (2.0 * CORNER) + 0.5
    )
    return {"d": distance, "rift": rift}


def _body_radius(body: Mapping[str, Any], theta: float) -> float:
    angle = theta - float(body["psi"])
    cosine, sine = math.cos(angle), math.sin(angle)
    lobe = 1.0
    for lobe_def in body["lobes"]:
        lobe += float(lobe_def["amp"]) * math.cos(
            float(lobe_def["k"]) * angle + float(lobe_def["phase"])
        )
    return (
        float(body["a"])
        * float(body["b"])
        / math.sqrt(
            float(body["b"]) ** 2 * cosine**2
            + float(body["a"]) ** 2 * sine**2
        )
        * lobe
    )


def _rift_edge(
    rng: Callable[[], float],
    cfg: Mapping[str, Any],
    body: Mapping[str, Any],
    bearing: float,
    depth: float,
    km_per_unit: float,
    flip: bool = False,
) -> dict[str, Any]:
    radius = _body_radius(body, bearing)
    offset = radius * depth
    half = math.sqrt(max(0.05, radius * radius - offset * offset))
    steps = []
    count = math.floor(rng() * (int(cfg["cratonRiftSteps"]) + 1))
    lower, upper = cfg["cratonRiftStepKm"]
    for _ in range(count):
        steps.append(
            {
                "at": (rng() * 1.4 - 0.7) * half,
                "jump": (-1.0 if rng() < 0.5 else 1.0)
                * (float(lower) + (float(upper) - float(lower)) * rng())
                / km_per_unit,
            }
        )
    return {
        "cs": math.cos(bearing),
        "sn": math.sin(bearing),
        "d": offset,
        "steps": steps,
        "flip": flip,
    }


def _outer_edges(
    rng: Callable[[], float],
    cfg: Mapping[str, Any],
    body: Mapping[str, Any],
    count: int,
    start: float,
    span: float,
    km_per_unit: float,
    ease: float = 0.0,
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    bearings: list[float] = []
    lower, upper = cfg["cratonRiftDepth"]
    accepted = 0
    attempts = 0
    while accepted < count and attempts < 40:
        attempts += 1
        beta = start + span * rng()
        if any(
            abs(math.atan2(math.sin(beta - prior), math.cos(beta - prior)))
            < MIN_EDGE_TURN
            for prior in bearings
        ):
            continue
        bearings.append(beta)
        accepted += 1
        depth = min(
            0.97,
            float(lower) + ease + (float(upper) - float(lower)) * rng(),
        )
        result.append(_rift_edge(rng, cfg, body, beta, depth, km_per_unit))
    return result


def _edge_count(rng: Callable[[], float], cfg: Mapping[str, Any]) -> int:
    lower, upper = cfg["cratonRiftEdges"]
    return int(lower) + math.floor(rng() * (int(upper) - int(lower) + 1))


def _trace_outline(piece: Mapping[str, Any]) -> dict[str, Any]:
    best = math.inf
    centre_u = centre_v = 0.0
    for i in range(-24, 25):
        for j in range(-24, 25):
            u = i * 0.1 + float(piece["tu"])
            v = j * 0.1 + float(piece["tv"])
            distance = _piece_shape(
                piece, u - float(piece["tu"]), v - float(piece["tv"])
            )["d"]
            if distance < best:
                best, centre_u, centre_v = distance, u, v

    points: list[list[float]] = []
    for sample in range(OUTLINE_SAMPLES):
        angle = sample * TAU / OUTLINE_SAMPLES
        du, dv = math.cos(angle), math.sin(angle)
        low = high = 0.0
        for step in range(1, 134):
            distance = step * 0.03
            value = _piece_shape(
                piece,
                centre_u + du * distance - float(piece["tu"]),
                centre_v + dv * distance - float(piece["tv"]),
            )["d"]
            if value > 0.0:
                high = distance
                break
            low = distance
        if high == 0.0:
            high = 4.0
        for _ in range(14):
            middle = (low + high) / 2.0
            value = _piece_shape(
                piece,
                centre_u + du * middle - float(piece["tu"]),
                centre_v + dv * middle - float(piece["tv"]),
            )["d"]
            if value > 0.0:
                high = middle
            else:
                low = middle
        points.append([centre_u + du * low, centre_v + dv * low])

    area = gx = gy = 0.0
    for index, point in enumerate(points):
        next_point = points[(index + 1) % len(points)]
        weight = point[0] * next_point[1] - next_point[0] * point[1]
        area += weight
        gx += (point[0] + next_point[0]) * weight
        gy += (point[1] + next_point[1]) * weight
    area /= 2.0
    if abs(area) < 1e-14:
        return {"points": points, "area": area, "cx": 0.0, "cy": 0.0, "compact": 0.0}
    cx, cy = gx / (6.0 * area), gy / (6.0 * area)
    compact = min(math.hypot(x - cx, y - cy) for x, y in points) / math.sqrt(
        abs(area) / math.pi
    )
    return {"points": points, "area": area, "cx": cx, "cy": cy, "compact": compact}


def _planar_gap(first: list[list[float]], second: list[list[float]]) -> float:
    return min(
        math.hypot(a[0] - b[0], a[1] - b[1])
        for a in first
        for b in second
    )


def _make_block(
    rng: Callable[[], float],
    cfg: Mapping[str, Any],
    targets: list[float],
    opening: float,
) -> dict[str, Any]:
    total_radius = math.sqrt(sum(radius * radius for radius in targets))
    km_per_unit = total_radius * PLANET_RADIUS_KM
    longest = float(cfg["cratonMaxDiameterKm"]) * 1.15 / (2.0 * km_per_unit)
    aspect = min(
        1.0 + (float(cfg["cratonMaxAspect"]) - 1.0) * rng() ** 1.5,
        max(1.0, longest * longest),
    )
    ragged = 0.4 + 0.6 * rng()
    psi = rng() * math.pi
    body = {
        "u0": 0.0,
        "v0": 0.0,
        "a": math.sqrt(aspect),
        "b": 1.0 / math.sqrt(aspect),
        "psi": psi,
        "lobes": [
            {
                "k": k,
                "amp": ragged * rng() * float(cfg["cratonLobes"]) / math.sqrt(k),
                "phase": rng() * TAU,
            }
            for k in (1, 2, 3)
        ],
    }
    pieces = [
        {"body": body, "cuts": [], "tu": 0.0, "tv": 0.0, "sibling": len(targets) > 1}
        for _ in targets
    ]
    traced: list[dict[str, Any]] = []
    beta = 0.0
    for tries in range(7):
        ease = 0.07 * tries
        fewer = tries >> 1

        def count(more: int, fewer_edges: int = fewer) -> int:
            return max(0, _edge_count(rng, cfg) - more - fewer_edges)

        if len(pieces) == 1:
            pieces[0]["cuts"] = _outer_edges(
                rng, cfg, body, count(0), 0.0, TAU, km_per_unit, ease
            )
        else:
            beta = psi + (rng() - 0.5)
            split = _rift_edge(
                rng,
                cfg,
                body,
                beta,
                (rng() - 0.5) * 0.4 / (1.0 + tries),
                km_per_unit,
            )
            pieces[0]["cuts"] = [
                split,
                *_outer_edges(
                    rng, cfg, body, count(1), beta + math.pi / 2.0, math.pi, km_per_unit, ease
                ),
            ]
            pieces[1]["cuts"] = [
                {**split, "flip": True},
                *_outer_edges(
                    rng, cfg, body, count(1), beta - math.pi / 2.0, math.pi, km_per_unit, ease
                ),
            ]
        traced = [_trace_outline(piece) for piece in pieces]
        total_area = sum(float(item["area"]) for item in traced)
        if tries >= 6 or all(
            float(item["compact"]) >= float(cfg["cratonMinCompact"])
            and float(item["area"]) >= 0.25 * total_area
            for item in traced
        ):
            break

    total_area = sum(float(item["area"]) for item in traced)
    if total_area <= 0.0:
        raise ValueError("craton block has a degenerate outline")
    scale = math.sqrt(
        sum(math.pi * radius * radius for radius in targets) / total_area
    )
    if len(pieces) > 1:
        slip = (rng() - 0.5) * 0.6
        gap = opening / scale
        for _ in range(256):
            pieces[1]["tu"] = gap * math.cos(beta) - slip * math.sin(beta)
            pieces[1]["tv"] = gap * math.sin(beta) + slip * math.cos(beta)
            traced[1] = _trace_outline(pieces[1])
            if (
                _planar_gap(traced[0]["points"], traced[1]["points"]) * scale
                >= opening * 1.05
            ):
                break
            gap *= 1.1

    total_area = sum(float(item["area"]) for item in traced)
    cx = sum(float(item["cx"]) * float(item["area"]) for item in traced) / total_area
    cy = sum(float(item["cy"]) * float(item["area"]) for item in traced) / total_area
    block: dict[str, Any] = {
        "scale": scale,
        "cx": cx,
        "cy": cy,
        "pieces": pieces,
        "frame": None,
    }
    for piece, outline in zip(pieces, traced, strict=True):
        local = [
            [(x - cx) * scale, (y - cy) * scale]
            for x, y in outline["points"]
        ]
        centre2 = [
            (float(outline["cx"]) - cx) * scale,
            (float(outline["cy"]) - cy) * scale,
        ]
        radii = [math.hypot(p[0] - centre2[0], p[1] - centre2[1]) for p in local]
        piece["local"] = local
        piece["centre2"] = centre2
        piece["radius"] = math.sqrt(float(outline["area"]) / math.pi) * scale
        piece["reach"] = max(radii)
        piece["inner"] = min(radii)
    block["reach"] = max(
        math.hypot(float(piece["centre2"][0]), float(piece["centre2"][1]))
        + float(piece["reach"])
        for piece in pieces
    )
    block["inner"] = min(
        max(
            0.0,
            float(piece["inner"])
            - math.hypot(float(piece["centre2"][0]), float(piece["centre2"][1])),
        )
        for piece in pieces
    )
    return block


def _from_frame(frame: Mapping[str, Any], u: float, v: float) -> list[float]:
    rho = math.hypot(u, v)
    if rho < 1e-12:
        return list(frame["o"])
    scale = math.sin(rho) / rho
    cosine = math.cos(rho)
    return [
        float(frame["o"][axis]) * cosine
        + (
            float(frame["e"][axis]) * u + float(frame["n"][axis]) * v
        )
        * scale
        for axis in range(3)
    ]


def _frame_at(origin: list[float], turn: float) -> dict[str, list[float]]:
    east0, north0 = tangent_basis(origin)
    cosine, sine = math.cos(turn), math.sin(turn)
    return {
        "o": origin,
        "e": [
            east0[i] * cosine + north0[i] * sine for i in range(3)
        ],
        "n": [
            north0[i] * cosine - east0[i] * sine for i in range(3)
        ],
    }


def _placed(
    block: dict[str, Any], frame: dict[str, list[float]]
) -> list[dict[str, Any]]:
    result = []
    for piece in block["pieces"]:
        result.append(
            {
                "pc": piece,
                "block": block,
                "centre": _from_frame(frame, *piece["centre2"]),
                "outline": [
                    _from_frame(frame, u, v) for u, v in piece["local"]
                ],
                "reach": piece["reach"],
                "inner": piece["inner"],
                "radius": piece["radius"],
            }
        )
    return result


def craton_edge(
    craton: Mapping[str, Any], x: float, y: float, z: float
) -> dict[str, float]:
    block = craton["block"]
    piece = craton["piece"]
    frame = block["frame"]
    co = x * frame["o"][0] + y * frame["o"][1] + z * frame["o"][2]
    pe = x * frame["e"][0] + y * frame["e"][1] + z * frame["e"][2]
    pn = x * frame["n"][0] + y * frame["n"][1] + z * frame["n"][2]
    sine = math.hypot(pe, pn)
    rho = math.atan2(sine, co)
    u = rho * pe / sine if sine > 1e-12 else 0.0
    v = rho * pn / sine if sine > 1e-12 else 0.0
    shaped = _piece_shape(
        piece,
        u / float(block["scale"]) + float(block["cx"]) - float(piece["tu"]),
        v / float(block["scale"]) + float(block["cy"]) - float(piece["tv"]),
    )
    grain = float(piece["body"]["psi"])
    cosine, sine = math.cos(grain), math.sin(grain)
    return {
        "d": shaped["d"] * float(block["scale"]),
        "rift": shaped["rift"],
        "along": u * cosine + v * sine,
        "across": v * cosine - u * sine,
    }


def _craton_radii(
    rng: Callable[[], float], cfg: Mapping[str, Any], area_scale: float
) -> list[float]:
    count = int(cfg["cratonCount"])
    target = (
        float(cfg["cratonShare"])
        * float(cfg["continentalFraction"])
        * area_scale
    )
    ratio = float(cfg["cratonSizeRatio"])
    while True:
        shape = [
            ratio ** ((i / (count - 1)) ** float(cfg["cratonSizeSkew"]) if count > 1 else 0.0)
            * (0.9 + 0.2 * rng())
            for i in range(count)
        ]
        low, high = 0.0, 1.0
        for _ in range(40):
            factor = (low + high) / 2.0
            covered = sum(
                cap_fraction(min(math.pi, value * factor)) for value in shape
            )
            if covered < target:
                low = factor
            else:
                high = factor
        radii = [value * low for value in shape]
        if 2.0 * max(radii) * PLANET_RADIUS_KM <= float(cfg["cratonMaxDiameterKm"]) or ratio < 1.3:
            return radii
        ratio *= 0.93


def _outline_gap(first: Mapping[str, Any], second: Mapping[str, Any]) -> float:
    distance = angle_between(first["centre"], second["centre"])
    if distance >= float(first["reach"]) + float(second["reach"]):
        return distance - float(first["reach"]) - float(second["reach"])
    if distance <= float(first["inner"]) + float(second["inner"]):
        return 0.0
    best = max(
        sum(a * b for a, b in zip(p, q, strict=True))
        for p in first["outline"]
        for q in second["outline"]
    )
    gap = math.acos(min(1.0, best))
    return max(
        0.0,
        gap - max(float(first["reach"]), float(second["reach"])) * math.pi / OUTLINE_SAMPLES,
    )


def place_cratons(
    rng: Callable[[], float],
    cfg: Mapping[str, Any],
    stretch: float = 1.0,
    area_scale: float = 1.0,
) -> dict[str, Any] | None:
    """Pack shaped cratons into clustered proto-continents on a sphere."""
    radii = _craton_radii(rng, cfg, area_scale)
    reserve = (
        float(cfg["cratonMinGapKm"]) + 2.0 * float(cfg["cratonEdgeNoiseKm"])
    ) / PLANET_RADIUS_KM
    order = list(range(len(radii)))
    for index in range(len(order) - 1, 0, -1):
        other = math.floor(rng() * (index + 1))
        order[index], order[other] = order[other], order[index]
    blocks = []
    cursor = 0
    while cursor < len(order):
        paired = cursor + 1 < len(order) and rng() < float(cfg["cratonSiblingChance"])
        targets = [radii[order[cursor]]]
        cursor += 1
        if paired:
            targets.append(radii[order[cursor]])
            cursor += 1
        opening = reserve * (1.0 + 0.25 * rng())
        blocks.append(_make_block(rng, cfg, targets, opening))

    def block_size(block: Mapping[str, Any]) -> float:
        return sum(float(piece["radius"]) ** 2 for piece in block["pieces"])

    lower, upper = cfg["clusterBlocks"]
    cluster_count = max(
        1,
        min(
            len(blocks),
            min(
                6,
                math.floor(
                    len(blocks) / (float(lower) + (float(upper) - float(lower)) * rng())
                    + 0.5
                ),
            ),
        ),
    )
    weights = [
        (0.6 + 0.8 * rng()) / math.pow(index + 1, 0.6)
        for index in range(cluster_count)
    ]
    counts = [1] * cluster_count
    for _ in range(len(blocks) - cluster_count):
        total_weight = sum(weights)
        total_count = sum(counts)
        deficits = [
            weights[k] / total_weight * (total_count + 1) - counts[k]
            for k in range(cluster_count)
        ]
        counts[deficits.index(max(deficits))] += 1

    heart = max(blocks, key=block_size)
    rest = [block for block in blocks if block is not heart]
    for index in range(len(rest) - 1, 0, -1):
        other = math.floor(rng() * (index + 1))
        rest[index], rest[other] = rest[other], rest[index]
    clusters: list[list[dict[str, Any]]] = []
    offset = 0
    for cluster in range(cluster_count):
        if cluster == 0:
            members = [heart, *rest[: counts[0] - 1]]
            offset = counts[0] - 1
        else:
            members = rest[offset : offset + counts[cluster]]
            offset += counts[cluster]
        clusters.append(sorted(members, key=block_size, reverse=True))

    centre = random_point(rng)
    max_latitude = math.sin(math.radians(float(cfg["startMaxLatitudeDeg"])))
    while abs(centre[2]) > max_latitude:
        centre = random_point(rng)
    axis = rng() * math.pi
    elongation = rng() * float(cfg["clusterElongation"]) * stretch
    apart_lo, apart_hi = cfg["clusterApart"]
    all_pieces: list[dict[str, Any]] = []

    def cluster_centre(cluster: int) -> list[float]:
        total = np.zeros(3, dtype=np.float64)
        for piece in all_pieces:
            if piece["cluster"] == cluster:
                weight = float(piece["radius"]) ** 2
                total += np.asarray(piece["centre"], dtype=np.float64) * weight
        length = float(np.linalg.norm(total)) or 1.0
        return (total / length).tolist()

    def settle(block: dict[str, Any], cluster: int, frame: dict[str, Any]) -> None:
        block["frame"] = frame
        for piece in _placed(block, frame):
            piece["cluster"] = cluster
            all_pieces.append(piece)

    for cluster, members in enumerate(clusters):
        for block in members:
            if not all_pieces:
                settle(block, cluster, _frame_at(centre, rng() * TAU))
                continue
            own = [piece for piece in all_pieces if piece["cluster"] == cluster]
            founding = not own
            home = centre if founding else cluster_centre(cluster)
            best_frame = None
            best_score = math.inf
            for relaxation in range(4):
                apart = max(1.0, float(apart_lo) - relaxation * (float(apart_lo) - 1.0) / 3.0)
                if best_frame is not None:
                    break
                for tries in range(400):
                    if tries >= 80 and best_frame is not None:
                        break
                    pool = all_pieces if founding else own
                    anchor = pool[math.floor(rng() * len(pool))]
                    if founding:
                        gap = reserve * (
                            float(apart_lo) + (float(apart_hi) - float(apart_lo)) * rng()
                        ) * (apart / float(apart_lo) if relaxation else 1.0)
                    else:
                        gap = reserve * (
                            1.0 + rng() * float(cfg["clusterGapVariety"]) * stretch
                        )
                    bearing = rng() * TAU
                    turn = rng() * TAU
                    near = float(anchor["inner"]) + float(block["inner"]) + gap
                    far = float(anchor["reach"]) + float(block["reach"]) + gap
                    steps = max(1, math.ceil((far - near) / (0.25 * gap)))
                    candidate: list[dict[str, Any]] = []
                    point = centre
                    for step in range(steps + 1):
                        point = destination(
                            anchor["centre"], bearing, near + (far - near) * step / steps
                        )
                        candidate = _placed(block, _frame_at(point, turn))
                        if step == steps or all(
                            _outline_gap(anchor, item) >= gap for item in candidate
                        ):
                            break
                    clear = all(
                        _outline_gap(other_piece, item)
                        >= (
                            gap * 0.999
                            if other_piece is anchor
                            else reserve * (1.0 if other_piece["cluster"] == cluster else apart)
                        )
                        for item in candidate
                        for other_piece in all_pieces
                    )
                    if not clear:
                        continue
                    across = abs(math.sin(bearing_to(centre, point) - axis))
                    spread = angle_between(point, centre) * (
                        1.0 + 1.6 * elongation * across
                    )
                    score = (
                        spread
                        if founding
                        else angle_between(point, home) + CLUSTER_PULL * spread
                    ) + rng() * float(cfg["clusterLooseness"]) * stretch
                    if score < best_score:
                        best_score = score
                        best_frame = _frame_at(point, turn)
            if best_frame is None:
                return None
            settle(block, cluster, best_frame)

    cratons = []
    for index, item in enumerate(all_pieces):
        block = item["block"]
        piece = item["pc"]
        cratons.append(
            {
                "id": index + 1,
                "letter": chr(65 + index),
                "cluster": item["cluster"],
                "centre": item["centre"],
                "radius": item["radius"],
                "reach": item["reach"],
                "inner": item["inner"],
                "grain": piece["body"]["psi"],
                "block": block,
                "piece": piece,
                "outline": item["outline"],
            }
        )
    return {
        "cratons": cratons,
        "centre": centre,
        "elongation": elongation,
        "clusters": cluster_count,
    }
