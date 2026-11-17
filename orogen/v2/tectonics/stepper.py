"""Single-step orchestration for the currently ported tectonic subsystems.

This is intentionally not the full upstream simulation loop: event-driven
collision/orogeny, execution of queued rifts, ocean-plate evolution, arcs,
LIP/track volcanism, and slow terrain/geology stages are not orchestrated here.
"""
from __future__ import annotations

import math

from .boundaries import classify_contact_boundaries
from .dynamics import update_plate_motion
from .kinematics import advance_plates
from .resolve import resolve
from .simulation import choose_step, closing_rate
from .plumes import update_plumes
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
    Ownership and crust consumption are then rebuilt, boundaries and trench
    bookkeeping are classified, the supercontinent cycle is advanced (when
    enabled), and finally a new force-balanced angular velocity is computed
    for the next step.

    dt=None selects the adaptive step from choose_step. Pass
    update_motion=False when an external driver will set plate velocities.
    """
    step_dt = float(choose_step(world) if dt is None else dt)
    if not math.isfinite(step_dt) or step_dt <= 0:
        raise ValueError("dt must be a finite positive number")

    cells_moved = (
        closing_rate(world) * step_dt / float(world.mesh.edge_rad)
        if float(world.mesh.edge_rad) > 0 else 0.0
    )
    advance_plates(world, step_dt)
    world.lastDt = step_dt
    tally = resolve(world)
    classified = classify_contact_boundaries(world, step_dt)
    plume_state = update_plumes(world, step_dt)
    cycle = track_cycle(world) if update_cycle else getattr(world, "cycle", None)
    if update_motion:
        update_plate_motion(world)

    return {
        "time": float(world.time),
        "step": int(world.step),
        "dt": step_dt,
        "cellsMoved": cells_moved,
        "resolve": tally,
        "classifiedBoundaries": int(classified),
        "cyclePhase": None if cycle is None else cycle.get("phase"),
        "plumes": plume_state,
        "motionUpdated": bool(update_motion),
    }
