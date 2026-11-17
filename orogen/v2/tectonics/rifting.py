"""Continental rifting stage ported from Orogen V2 step/rifting.js."""

from __future__ import annotations

import math

from .ops.rift import (
    choose_junction,
    plume_junctions,
    try_rift,
)
from .supercontinent import is_continental


_JUNCTION_TRIES = 12
_FORCED_JUNCTION_TRIES = 48
_LOOSE_SHAPE = {
    "riftTortuosityMax": 2.0,
    "riftMinChordKm": 1000.0,
    "riftMinPieceShare": 0.12,
    "riftCoastHugKm": 1500.0,
}


def pace_of(age: float, config) -> float:
    """Return the upstream smooth breakup pace multiplier at a piece's age."""
    x = (
        float(age) - float(config["riftMinGapMyr"])
    ) / max(
        1e-9,
        float(config["riftSpacingMyr"]) - float(config["riftMinGapMyr"]),
    )
    x = 0.0 if x <= 0.0 else 1.0 if x >= 1.0 else x
    late = max(
        0.0,
        float(age) - float(config["riftDueMyr"]),
    ) / float(config["riftDueRampMyr"])
    return x * x * (3.0 - 2.0 * x) * (1.0 + late * late)


def search_junctions(world, request: dict) -> list:
    """Evaluate each supplied rift junction in order (serial worker equivalent)."""
    from .ops.rift import find_rift, rift_context

    plate = request["plate"]
    context = request.get("ctx") or rift_context(world, plate)
    pull = request.get("pull")
    need_cratons = bool(request.get("needCratonsBothSides", True))
    return [
        None
        if int(junction) < 0
        else find_rift(
            world,
            plate,
            int(junction),
            pull,
            need_cratons,
            context,
        )
        for junction in request["junctions"]
    ]


def rift_continents(world, dt: float) -> int:
    """Execute queued plume rifts, overdue breakups and spontaneous rifting.

    This is the synchronous Python counterpart of the upstream generator.
    Junction searches are performed locally rather than handed to helper
    workers; each result is evaluated in the same deterministic order.
    """
    config = world.config
    plates = world.plates
    time = float(world.time)
    mesh = world.mesh
    step_dt = float(dt)
    pending = getattr(world, "pendingPlumeRifts", [])
    rate = float(config.get("riftRatePerMyr", 0.0))
    if rate <= 0.0 and not pending:
        return 0

    min_gap = float(config.get("riftMinGapMyr", 0.0))
    paced = rate > 0.0 and min_gap > 0.0
    last_rift = getattr(world, "lastRiftAt", None)
    world_waits = (
        paced
        and last_rift is not None
        and time - float(last_rift)
        < float(config["riftGlobalGapMyr"]) - 1e-9
    )

    def age_of(plate) -> float:
        return time - float(plate.born)

    def area_of(plate) -> float:
        return (
            float(plate.stats["continentalCells"])
            * float(mesh.cell_area_km2)
        )

    def too_young(plate) -> bool:
        return paced and age_of(plate) < min_gap - 1e-9

    min_area = float(config.get("riftMinContinentKm2", math.inf))
    for plate in plates:
        if not plate.alive or plate.isBackground:
            continue
        if area_of(plate) >= min_area:
            if getattr(plate, "bigSince", None) is None:
                plate.bigSince = time
        else:
            plate.bigSince = None

    def big_age(plate) -> float:
        since = getattr(plate, "bigSince", None)
        return time - max(float(plate.born), time if since is None else float(since))

    cycle = getattr(world, "cycle", None)
    gather_into = (
        set(cycle.get("largestPlates", []))
        if cycle is not None and float(cycle.get("gather", 0.0)) > 0.0
        else None
    )

    def overdue(plate) -> bool:
        return (
            paced
            and big_age(plate) + step_dt
            >= float(config["riftMaxGapMyr"])
            - float(config["riftGlobalGapMyr"])
            - 1e-9
            and area_of(plate) >= min_area
        )

    super_held = None
    if (
        cycle is not None
        and cycle.get("phase") == "assembled"
        and float(cycle.get("formedAt", 0.0)) > 0.0
        and time - float(cycle["formedAt"])
        < float(config["superTenureMyr"])
    ):
        super_held = set(cycle.get("largestPlates", []))

    def held(plate) -> bool:
        return bool(
            super_held is not None
            and plate.id in super_held
            and not overdue(plate)
        )

    # Pending plume heads keep their original plate-frame junctions while the
    # continental piece is too young or the global rift clock is closed.
    waiting = []
    splits = 0
    for request in pending:
        plate_id = int(request["plate"])
        plate = plates[plate_id] if 0 <= plate_id < len(plates) else None
        if plate is None or not plate.alive:
            continue
        if "since" not in request:
            request["since"] = time
            request["junctions"] = (
                [int(request["cell"])]
                if int(request.get("cell", -1)) >= 0
                else plume_junctions(world, plate, request["xyz"])
            )
        since = float(request["since"])
        if (
            world_waits
            or (paced and splits)
            or held(plate)
            or too_young(plate)
        ):
            if (
                paced
                and time - since
                < float(config["plumeRiftWaitMyr"]) - 1e-9
            ):
                waiting.append(request)
            continue

        cause = request["cause"]
        success = try_rift(
            world, plate, request["junctions"], cause, False
        )
        if not success and time > since:
            success = _with_loose_shape(
                config,
                lambda: try_rift(
                    world, plate, request["junctions"], cause, False
                ),
            )
        if success:
            splits += 1

    world.pendingPlumeRifts = waiting
    if world_waits or (paced and splits):
        return splits

    heat = (
        _heat_under_continents(world)
        if float(config.get("riftHeatGain", 0.0)) > 0.0
        else None
    )

    # A sufficiently large continental piece that has reached its maximum
    # breakup interval gets the first chance, regardless of its random rate.
    if paced:
        due = [
            plate for plate in plates
            if (
                plate.alive
                and not plate.isBackground
                and plate.dynamics
                and overdue(plate)
            )
        ]
        due.sort(key=big_age, reverse=True)
        for plate in due:
            if _force_rift(world, plate, age_of(plate)):
                return 1

    # Spontaneous rifting is probabilistic, weighted by tension, size, mantle
    # heat, dispersal state and the piece's age since its last rift or weld.
    for plate in list(plates):
        if (
            not plate.alive
            or plate.isBackground
            or not plate.dynamics
            or held(plate)
            or too_young(plate)
        ):
            continue
        area_km2 = area_of(plate)
        too_soon = bool(
            cycle is not None
            and plate.id in cycle.get("clump", [])
        )
        mantle_heat = 0.0 if heat is None else float(heat.get(plate.id, 0.0))

        if paced:
            tenure = 0.0
        elif too_soon:
            tenure = float(config["riftTenureHotMyr"])
        elif heat is not None and plate.bornOfRift:
            tenure = (
                float(config["riftTenureMyr"])
                + (
                    float(config["riftTenureHotMyr"])
                    - float(config["riftTenureMyr"])
                ) * mantle_heat
            )
        else:
            tenure = float(config["riftTenureMyr"])
        if area_km2 < min_area or age_of(plate) < tenure:
            continue

        if heat is not None:
            heat_factor = (
                float(config["riftHeatFloor"])
                + float(config["riftHeatGain"])
                * mantle_heat ** float(config["riftHeatPower"])
            )
        else:
            heat_factor = 1.0
        if too_soon:
            heat_factor = (
                max(heat_factor, 1.0)
                * float(config["dispersalRiftBoost"])
            )
        if heat is not None and gather_into and plate.id in gather_into:
            heat_factor *= (
                1.0
                - (1.0 - float(config["gatherRiftFactor"]))
                * float(cycle["gather"])
            )

        tension = (
            max(
                float(plate.dynamics["tension"]),
                float(config["riftHeatTension"]) * mantle_heat,
            )
            if heat is not None
            else float(plate.dynamics["tension"])
        )
        pace = pace_of(age_of(plate), config) if paced else 1.0
        chance = (
            pace
            * rate
            * step_dt
            * tension
            * heat_factor
            * min(
                float(config["riftSizeCap"]),
                area_km2 / min_area - 1.0,
            )
        )
        if float(world.rng()) >= chance:
            continue

        junctions = [
            choose_junction(world, plate, world.rng)
            for _ in range(_JUNCTION_TRIES)
        ]
        cause = {
            "text": f"tension {tension:.2f}"
            + (
                f", mantle heat {mantle_heat:.2f}"
                if heat is not None
                else ""
            ),
            "kind": "tension",
            "hotspot": -1,
        }
        if try_rift(world, plate, junctions, cause):
            splits += 1
            if paced:
                break

    return splits


def _force_rift(world, plate, age: float) -> bool:
    """Attempt an overdue breakup with normal, permissive, then loose rules."""
    cause = {
        "text": f"no rift for {math.floor(float(age) + 0.5)} Myr",
        "kind": "overdue",
        "hotspot": -1,
    }
    junctions = [
        choose_junction(world, plate, world.rng)
        for _ in range(_FORCED_JUNCTION_TRIES)
    ]

    def attempt() -> bool:
        return try_rift(
            world, plate, junctions, cause
        ) or try_rift(
            world, plate, junctions, cause, False
        )

    return attempt() or _with_loose_shape(
        world.config, attempt
    )


def _with_loose_shape(config, action):
    """Temporarily relax the geometric constraints, restoring them on failure too."""
    saved = {
        key: config[key]
        for key in _LOOSE_SHAPE
    }
    config.update(_LOOSE_SHAPE)
    try:
        return action()
    finally:
        config.update(saved)


def _heat_under_continents(world) -> dict[int, float]:
    """Mean mantle heat under each plate's continental and cratonic cells."""
    totals = [0.0] * len(world.plates)
    counts = [0] * len(world.plates)
    for cell in range(len(world.owner)):
        if not is_continental(world, cell):
            continue
        owner = int(world.owner[cell])
        totals[owner] += float(world.mantleHeat[cell])
        counts[owner] += 1
    return {
        plate_id: totals[plate_id] / counts[plate_id] if counts[plate_id] else 0.0
        for plate_id in range(len(world.plates))
    }


__all__ = ["pace_of", "rift_continents", "search_junctions"]
