"""Shared construction helpers for deterministic tectonic scenario scenes."""

from __future__ import annotations

import math
from collections.abc import Sequence

from ..core.sim_mesh import SimMesh
from ..core.simplex_noise import SimplexNoise
from ..core.sphere_geometry import PLANET_RADIUS_KM, cross, dot, normalize
from .scenarios import SceneCanvas, add_ocean, scene_canvas
from .plate import Plate
from .world import TectonicWorld


def begin_scene(
    sim_mesh: SimMesh, seed: int
) -> tuple[TectonicWorld, SceneCanvas, SimplexNoise]:
    """Create a world, its scene canvas and the seeded noise source."""
    world = TectonicWorld(sim_mesh, seed)
    return world, scene_canvas(sim_mesh, world.config), SimplexNoise(seed)


def add_ocean_corridor(
    canvas: SceneCanvas,
    plate: Plate,
    first: Sequence[float],
    second: Sequence[float],
    skirt_km: float,
    half_width_km: float,
) -> None:
    """Paint an old-ocean skirt and a corridor along the great circle between two centres."""
    axis = normalize(cross(first, second))
    span = math.acos(max(-1.0, min(1.0, dot(first, second))))

    def wanted(x: float, y: float, z: float) -> bool:
        point = [x, y, z]
        distance = math.acos(max(-1.0, min(1.0, dot(point, first))))
        if distance * PLANET_RADIUS_KM < skirt_km:
            return True
        off_track = (
            abs(math.asin(max(-1.0, min(1.0, dot(point, axis))))) * PLANET_RADIUS_KM
        )
        along = math.atan2(dot(cross(first, point), axis), dot(first, point))
        return off_track < half_width_km and 0.0 < along < span

    add_ocean(canvas, plate, wanted, -90.0)
