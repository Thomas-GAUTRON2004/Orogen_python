"""Shared typing boundary for the heterogeneous tectonic scene catalogues."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import NotRequired, TypedDict

from ..core.sim_mesh import SimMesh
from .world import TectonicWorld

SceneBuilder = Callable[[SimMesh, int], TectonicWorld]
SceneDriver = Callable[[TectonicWorld], None]
SceneExpectation = bool | list[str]


class SceneDefinition(TypedDict):
    """Required metadata and callable used by every tectonic scene catalogue."""

    id: str
    label: str
    maxMyr: float
    view: tuple[float, float] | None
    about: str
    build: SceneBuilder


class OptionalSceneMetadata(TypedDict, total=False):
    """Optional display and simulation metadata for a scene."""

    drive: SceneDriver
    expect: dict[str, SceneExpectation]
    checklist: list[str]
    speedCmYr: float
    trenchLon: float
    plume: tuple[float, float]
    bendMyr: float
    legs: tuple[float, float]
    arrivesMyr: float


class CompleteSceneDefinition(SceneDefinition, OptionalSceneMetadata):
    """A catalogue entry including any optional scene-specific metadata."""


def build_from_catalogue(
    definition: CompleteSceneDefinition, sim_mesh: SimMesh, seed: int
) -> TectonicWorld:
    """Invoke the statically typed builder stored in a scene definition."""
    return definition["build"](sim_mesh, seed)
