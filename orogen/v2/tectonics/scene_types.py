"""Shared typing boundary for the heterogeneous tectonic scene catalogues."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import cast

from ..core.sim_mesh import SimMesh
from .world import TectonicWorld

SceneBuilder = Callable[[SimMesh, int], TectonicWorld]


def build_from_catalogue(
    definition: Mapping[str, object], sim_mesh: SimMesh, seed: int
) -> TectonicWorld:
    """Invoke the typed builder stored in a scene metadata mapping."""
    builder = cast(SceneBuilder, definition["build"])
    return builder(sim_mesh, seed)
