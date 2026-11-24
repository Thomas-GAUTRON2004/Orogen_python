"""Scene catalogue parity with the upstream engine (ed466f9).

The fixture was dumped from upstream ``v2/js/tectonics/scenarios`` by
``tests/reference/v2/scenes/dump_scenes.mjs`` (768 cells, seed 941). Each
scene's build step must give the same plates, cell counts and crust types.
"""

from __future__ import annotations

import importlib
import json
import os
from pathlib import Path

import numpy as np
import pytest

from orogen.v2.core.sim_mesh import build_sim_mesh

FIXTURE = (
    Path(__file__).parent / "reference" / "v2" / "scenes" / "js-scene-dump-768-941.json"
)
MODULES = {
    "BUDGET_SCENES": "budget_scenes",
    "COLLISION_SCENES": "collision_scenes",
    "EVENT_SCENES": "event_scenes",
    "FORCE_SCENES": "force_scenes",
    "HOTSPOT_SCENES": "hotspot_scenes",
    "MARGIN_SCENES": "margin_scenes",
    "RIFT_SCENES": "rift_scenes",
    "TERRAIN_SCENES": "terrain_scenes",
}
SEED = 941
EXPECTED = json.loads(FIXTURE.read_text(encoding="utf-8"))


def _python_scenes() -> dict[str, dict]:
    scenes = {}
    for array, module in MODULES.items():
        mod = importlib.import_module(f"orogen.v2.tectonics.{module}")
        for definition in getattr(mod, array):
            if definition["id"] in ("history", "cycle"):
                continue
            scenes[f"{array}:{definition['id']}"] = definition
    return scenes


@pytest.fixture(scope="module")
def mesh():
    return build_sim_mesh(768, SEED)


@pytest.mark.parametrize("key", sorted(EXPECTED))
def test_scene_build_matches_upstream(key: str, mesh) -> None:
    definition = _python_scenes()[key]
    world = definition["build"](mesh, SEED)
    expected = EXPECTED[key]

    assert len(world.plates) == expected["plates"]
    cells = [int(np.asarray(p.crust.mask, bool).sum()) for p in world.plates]
    assert cells == expected["cells"]
    types: dict[str, int] = {}
    for plate in world.plates:
        mask = np.asarray(plate.crust.mask, bool)
        for value in np.asarray(plate.crust.type)[mask].tolist():
            types[str(int(value))] = types.get(str(int(value)), 0) + 1
    assert types == expected["types"]


def test_fixture_is_current_with_the_submodule() -> None:
    """The stored fixture must equal a fresh dump of the pinned submodule.

    Skipped when Node or the submodule's dependencies are not available.
    """
    import shutil
    import subprocess

    node = shutil.which("node")
    if node is None:
        if os.environ.get("CI"):
            pytest.fail("node is not installed")
        pytest.skip("node is not installed")
    dump = Path(__file__).parent / "reference" / "v2" / "scenes" / "dump_scenes.mjs"
    try:
        result = subprocess.run(
            [node, str(dump), "768", str(SEED)],
            check=True,
            capture_output=True,
            text=True,
            timeout=600,
        )
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as error:
        if os.environ.get("CI"):
            pytest.fail(f"upstream dump unavailable: {error}")
        pytest.skip(f"upstream dump unavailable: {error}")
    assert json.loads(result.stdout) == EXPECTED
