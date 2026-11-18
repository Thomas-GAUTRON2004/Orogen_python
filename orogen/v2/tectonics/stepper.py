"""Single-step orchestration for the currently ported tectonic subsystems.

This is intentionally not the full upstream simulation loop: collision
tracking and configured suture welding are partial; ocean-plate evolution,
arc maturation, and slow terrain/geology stages are not fully orchestrated.
Continental rifting, rollback-driven back-arcs, and far-field diffusion are
included when their configuration is present.
"""

from __future__ import annotations

import math

from .backarc import open_back_arcs, steer_back_arc_strips
from .boundaries import classify_contact_boundaries
from .collision import track_collisions
from .dynamics import update_plate_motion
from .far_field import evolve_far_field
from .kinematics import advance_plates
from .plumes import update_plumes
from .resolve import resolve
from .rifting import rift_continents
from .simulation import choose_step, closing_rate
from .subduction_init import initiate_subduction
from .supercontinent import track_cycle


def step_world(
    world,
    dt: float | None = None,
    *,
    update_motion: bool = True,
    update_cycle: bool = True,
) -> dict:
    """Advance one step using the ported resolve/classify/force stages.

    The current plate angular velocities move plates through dt first.
    Ownership and crust consumption are rebuilt, boundaries and trench
    bookkeeping are classified, plumes and the supercontinent cycle advance,
    then configured rifting/subduction and collision stages run. The collision
    stage can merge eligible sutures when the full redraw configuration exists.
    When any topology-changing stage changes the crust or plate set, ownership
    and boundaries are rebuilt before force balance.

    dt=None selects the adaptive step from choose_step. Pass
    update_motion=False when an external driver will set plate velocities;
    that driver should call steer_back_arc_strips after its own force solve.
    """
    step_dt = float(choose_step(world) if dt is None else dt)
    if not math.isfinite(step_dt) or step_dt <= 0:
        raise ValueError("dt must be a finite positive number")

    cells_moved = (
        closing_rate(world) * step_dt / float(world.mesh.edge_rad)
        if float(world.mesh.edge_rad) > 0
        else 0.0
    )
    advance_plates(world, step_dt)
    world.lastDt = step_dt
    tally = resolve(world)
    classified = classify_contact_boundaries(world, step_dt)
    plume_state = update_plumes(world, step_dt)
    cycle = track_cycle(world) if update_cycle else getattr(world, "cycle", None)
    rifts = 0
    if "riftRatePerMyr" in world.config or getattr(world, "pendingPlumeRifts", None):
        rifts = rift_continents(world, step_dt)

    back_arcs = open_back_arcs(world, step_dt)

    subductions = 0
    if float(world.config.get("subductionInitRatePerMyr", 0.0)) > 0.0:
        subductions = initiate_subduction(world, step_dt)

    # Rifting, foundering, and back-arc opening/rejoining can add, remove, or
    # reassign crust. Rebuild the world view before a later stage reads it.
    if rifts or subductions or back_arcs:
        tally = resolve(world)
        classified = classify_contact_boundaries(world, 0.0)

    collision_state = {
        "tracked": 0,
        "started": 0,
        "underthrustCells": 0,
        "changed": False,
    }
    if "collisionForgetMyr" in world.config:
        collision_state = track_collisions(world, step_dt)
        if collision_state["changed"]:
            tally = resolve(world)
            classified = classify_contact_boundaries(world, 0.0)

    far_field_state = 0
    if "farFieldEveryMyr" in world.config:
        far_field_state = evolve_far_field(world, step_dt)

    steered_back_arcs = 0
    if update_motion:
        update_plate_motion(world)
        steered_back_arcs = steer_back_arc_strips(world)

    return {
        "time": float(world.time),
        "step": int(world.step),
        "dt": step_dt,
        "cellsMoved": cells_moved,
        "resolve": tally,
        "classifiedBoundaries": int(classified),
        "cyclePhase": None if cycle is None else cycle.get("phase"),
        "plumes": plume_state,
        "rifts": int(rifts),
        "subductions": int(subductions),
        "backArcs": int(back_arcs),
        "steeredBackArcs": int(steered_back_arcs),
        "collisions": collision_state,
        "farField": int(far_field_state),
        "motionUpdated": bool(update_motion),
    }
