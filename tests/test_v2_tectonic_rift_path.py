import math
from types import SimpleNamespace

import numpy as np
import pytest

from orogen.v2.tectonics.crust import CrustRaster, CrustType
from orogen.v2.tectonics.ops import find_rift, grain_of, rift_context, rift_path
from orogen.v2.tectonics.ops.rift_path import (
    _angular_separation,
    _cheapest_paths,
    _crosses_rather_than_hugs,
    _is_monotone,
    _nearby_rift_cuts,
    _trace_parent_path,
)


class Mesh:
    def __init__(self, points, neighbours, edge_km=63.71, edge_rad=0.01):
        self.num_cells = len(points)
        self.xyz = np.asarray(points, dtype=np.float32).reshape(-1)
        self.edge_km = float(edge_km)
        self.edge_rad = float(edge_rad)
        offsets = [0]
        adjacency = []
        for row in neighbours:
            adjacency.extend(row)
            offsets.append(len(adjacency))
        self.adj_offset = np.asarray(offsets, dtype=np.int32)
        self.adj_list = np.asarray(adjacency, dtype=np.int32)


def square_world():
    points = []
    neighbours = []
    for row in range(7):
        latitude = (row - 3) * 0.01
        for col in range(7):
            longitude = (col - 3) * 0.01
            points.append(
                [
                    math.cos(latitude) * math.cos(longitude),
                    math.cos(latitude) * math.sin(longitude),
                    math.sin(latitude),
                ]
            )
            cell = row * 7 + col
            adjacent = []
            if row:
                adjacent.append(cell - 7)
            if col:
                adjacent.append(cell - 1)
            if col < 6:
                adjacent.append(cell + 1)
            if row < 6:
                adjacent.append(cell + 7)
            neighbours.append(adjacent)

    mesh = Mesh(points, neighbours)
    crust = CrustRaster(49)
    for row in range(1, 6):
        for col in range(1, 6):
            cell = row * 7 + col
            crust.create(cell, CrustType["CONTINENT"], 0.0, cell + 1)

    config = {
        "riftCurveAmp": 0.0,
        "riftJagAmp": 0.0,
        "riftCurveKm": 2000.0,
        "riftJagKm": 400.0,
        "riftWeaknessCost": 0.4,
        "riftMinArmKm": 10.0,
        "riftAcrossPullBias": 0.0,
        "riftArmEndCoastKm": 3000.0,
        "riftMinChordKm": 100.0,
        "riftTortuosityMax": 4.0,
        "riftCostSlack": 10.0,
        "riftOverhangKm": 200.0,
        "riftCoastClearanceKm": 0.0,
        "riftCoastHugKm": 200.0,
        "riftMinPieceShare": 0.1,
        "riftElongationBias": 0.0,
        "riftFailedArms": 1,
        "riftFailedArmKm": 100.0,
    }
    world = SimpleNamespace(mesh=mesh, config=config, seed=42)
    plate = SimpleNamespace(crust=crust)
    return world, plate


def test_grain_of_off_switch_and_cached_noise(monkeypatch):
    world, _ = square_world()
    flat_grain = grain_of(world)
    assert flat_grain(0) == 1.0
    assert not hasattr(world, "riftGrain")

    world.config.update(riftCurveAmp=0.5, riftJagAmp=0.25)

    class ConstantNoise:
        def __init__(self, seed):
            self.seed = seed

        def noise3d(self, x, y, z):
            return -100.0

    monkeypatch.setattr(rift_path, "FastNoise", ConstantNoise)
    grain = grain_of(world)
    first = grain(10)
    assert first == 0.2
    assert grain(10) == first
    assert grain(11) == 0.2
    assert world.riftGrainOf[10] == 0.2
    assert world.riftGrain.seed == (world.seed ^ 0x6A41)


def test_rift_context_precomputes_coast_distance_and_grain():
    world, plate = square_world()
    ctx = rift_context(world, plate)
    assert len(ctx["cells"]) == 25
    assert ctx["exterior"].dtype == np.uint8
    assert ctx["from_coast"][3 * 7 + 3] == 2
    assert ctx["from_coast"][0] == -1
    assert ctx["grain"](3 * 7 + 3) == 1.0


def test_coast_distance_seeds_outer_land_and_floods_inward():
    world, plate = square_world()
    ctx = rift_context(world, plate)
    assert ctx["from_coast"][1 * 7 + 3] == 0
    assert ctx["from_coast"][2 * 7 + 3] == 1
    assert ctx["from_coast"][3 * 7 + 3] == 2


def test_cheapest_paths_respect_ocean_and_craton_barriers():
    points = [[math.cos(i * 0.01), 0.0, math.sin(i * 0.01)] for i in range(5)]
    mesh = Mesh(points, [[1], [0, 2], [1, 3], [2, 4], [3]])
    crust = CrustRaster(5)
    crust.create(0, CrustType["CONTINENT"], 0.0)
    crust.create(1, CrustType["CONTINENT"], 0.0)
    crust.create(2, CrustType["CRATON"], 0.0)
    crust.create(3, CrustType["CONTINENT"], 0.0)
    crust.create(4, CrustType["OCEAN"], 0.0)

    result = _cheapest_paths(mesh, crust, 0, lambda _cell: 1.0)
    assert math.isfinite(result["cost"][1])
    assert result["from"][1] == 0
    assert math.isinf(result["cost"][2])
    assert math.isinf(result["cost"][3])
    assert math.isinf(result["cost"][4])


def test_crosses_rather_than_hugs_rejects_long_coastal_runs():
    mesh = SimpleNamespace(edge_km=50.0)
    path = [0, 1, 2, 3, 4, 5]
    coast_distance_values = np.asarray([0, 2, 1, 1, 2, 0], dtype=np.int16)
    config = {"riftCoastClearanceKm": 100.0, "riftCoastHugKm": 50.0}
    assert not _crosses_rather_than_hugs(mesh, path, coast_distance_values, config)

    config["riftCoastHugKm"] = 100.0
    assert _crosses_rather_than_hugs(mesh, path, coast_distance_values, config)


def test_is_monotone_accepts_forward_path_and_rejects_backstep():
    angles = [-0.02, -0.01, 0.0, 0.01, 0.02]
    points = [[math.cos(a), math.sin(a), 0.0] for a in angles]
    mesh = Mesh(points, [[] for _ in points], edge_rad=0.001)
    config = {"riftOverhangKm": 0.0}

    assert _is_monotone(mesh, [0, 1, 2, 3, 4], config)
    assert not _is_monotone(mesh, [0, 1, 3, 2, 4], config)


def test_find_rift_builds_a_coast_to_coast_path_and_failed_arm():
    world, plate = square_world()
    # A sealed ocean pocket is not land and cannot enter the path-search tree.
    plate.crust.type[2 * 7 + 2] = CrustType["OCEAN"]
    world.config["riftFailedArmKm"] = 50.0
    result = find_rift(
        world,
        plate,
        junction=24,
        pull_axis=[1.0, 0.0, 0.0],
        need_cratons_both_sides=False,
    )
    assert result is not None
    assert result["path"][0] != result["path"][-1]
    assert 24 in result["path"]
    assert len(result["arms"]) == 3
    assert result["arms"][0]
    assert result["arms"][1]
    assert result["arms"][2]
    assert result["failed_arm"]
    assert len(result["across"]) == 3
    assert math.isfinite(result["value"])


def test_find_rift_rejects_ocean_craton_and_single_coast_junctions():
    world, plate = square_world()
    assert find_rift(world, plate, junction=0) is None

    craton_cell = 2 * 7 + 2
    plate.crust.type[craton_cell] = CrustType["CRATON"]
    assert find_rift(world, plate, junction=craton_cell) is None

    mesh = Mesh([[1, 0, 0], [0, 1, 0]], [[1], [0]])
    crust = CrustRaster(2)
    crust.create(0, CrustType["CONTINENT"], 0.0)
    tiny_world = SimpleNamespace(
        mesh=mesh,
        config={
            "riftCurveAmp": 0.0,
            "riftJagAmp": 0.0,
            "riftMinArmKm": 0.0,
            "riftAcrossPullBias": 0.0,
            "riftWeaknessCost": 1.0,
        },
        seed=1,
    )
    tiny_plate = SimpleNamespace(crust=crust)
    assert find_rift(tiny_world, tiny_plate, junction=0) is None


def test_find_rift_rejects_short_chords():
    world, plate = square_world()
    world.config["riftMinChordKm"] = 100_000.0
    assert find_rift(world, plate, junction=24, need_cratons_both_sides=False) is None


def test_find_rift_continues_after_non_monotone_candidates(monkeypatch):
    world, plate = square_world()
    monkeypatch.setattr(rift_path, "_is_monotone", lambda *_args: False)
    assert find_rift(world, plate, junction=24, need_cratons_both_sides=False) is None


def test_find_rift_continues_after_coast_hugging_candidates(monkeypatch):
    world, plate = square_world()
    monkeypatch.setattr(rift_path, "_crosses_rather_than_hugs", lambda *_args: False)
    assert find_rift(world, plate, junction=24, need_cratons_both_sides=False) is None


def test_find_rift_rejects_poorly_balanced_land_blocks(monkeypatch):
    world, plate = square_world()
    monkeypatch.setattr(
        rift_path,
        "sides_of",
        lambda *_args: {
            "share": 0.0,
            "elongation": 1.0,
            "cratonsBothSides": False,
        },
    )
    assert find_rift(world, plate, junction=24, need_cratons_both_sides=False) is None


def test_find_rift_stops_when_pair_cost_exceeds_slack(monkeypatch):
    world, plate = square_world()
    world.config["riftCostSlack"] = 0.0
    monkeypatch.setattr(
        rift_path,
        "sides_of",
        lambda *_args: {
            "share": 0.0,
            "elongation": 1.0,
            "cratonsBothSides": False,
        },
    )
    assert find_rift(world, plate, junction=24, need_cratons_both_sides=False) is None


def test_grain_of_single_noise_component_and_precomputed_context(monkeypatch):
    world, plate = square_world()
    world.config["riftCurveAmp"] = 0.25
    world.config["riftJagAmp"] = 0.0

    class ConstantNoise:
        def __init__(self, _seed):
            pass

        def noise3d(self, *_coords):
            return 0.5

    monkeypatch.setattr(rift_path, "FastNoise", ConstantNoise)
    grain = grain_of(world)
    assert grain(10) == pytest.approx(1.125)
    assert grain(10) == pytest.approx(1.125)
    ctx = rift_context(world, plate)
    assert (
        find_rift(
            world,
            plate,
            24,
            need_cratons_both_sides=False,
            ctx=ctx,
        )
        is not None
    )


def test_find_rift_without_optional_failed_arm():
    world, plate = square_world()
    world.config["riftFailedArms"] = 0
    result = find_rift(
        world,
        plate,
        24,
        need_cratons_both_sides=False,
    )
    assert result is not None
    assert result["failed_arm"] == []
    assert result["arms"][2] == []


def test_coast_distance_satisfies_breadth_first_distance_invariants():
    from orogen.v2.tectonics.ops.rift_path import coast_distance

    for seed in (3, 7, 19):
        world, plate = square_world()
        rng = np.random.default_rng(seed)
        plate.crust = CrustRaster(49)
        for row in range(1, 6):
            for col in range(1, 6):
                cell = row * 7 + col
                if rng.random() < 0.72:
                    plate.crust.create(cell, CrustType["CONTINENT"], 0.0, cell)

        context = rift_context(world, plate)
        distance = coast_distance(world.mesh, plate.crust, context["exterior"])
        offsets, adjacency = world.mesh.adj_offset, world.mesh.adj_list

        for cell in range(world.mesh.num_cells):
            neighbours = [
                int(adjacency[edge])
                for edge in range(int(offsets[cell]), int(offsets[cell + 1]))
            ]
            is_land = bool(
                plate.crust.mask[cell] and plate.crust.type[cell] != CrustType["OCEAN"]
            )
            if not is_land or distance[cell] < 0:
                continue

            reachable_land = [
                neighbour
                for neighbour in neighbours
                if plate.crust.mask[neighbour]
                and plate.crust.type[neighbour] != CrustType["OCEAN"]
                and distance[neighbour] >= 0
            ]
            if distance[cell] == 0:
                assert any(context["exterior"][neighbour] for neighbour in neighbours)
            else:
                assert any(
                    distance[neighbour] == distance[cell] - 1
                    for neighbour in reachable_land
                )

            assert all(
                abs(int(distance[cell]) - int(distance[neighbour])) <= 1
                for neighbour in reachable_land
            )

def test_rift_path_geometry_helpers_preserve_parent_and_distance_rules():
    parent = np.asarray([-1, 0, 1, 1, 3], dtype=np.int32)
    assert _trace_parent_path(4, parent) == [0, 1, 3, 4]
    assert _trace_parent_path(2, parent) == [0, 1, 2]

    first = {"direction": [1.0, 0.0, 0.0], "cell": 0}
    perpendicular = {"direction": [0.0, 1.0, 0.0], "cell": 2}
    opposite = {"direction": [-1.0, 0.0, 0.0], "cell": 4}
    assert _angular_separation(first, perpendicular) == pytest.approx(90.0)
    assert _angular_separation(first, opposite) == pytest.approx(180.0)

    xyz = np.asarray(
        [
            1.0, 0.0, 0.0,
            math.cos(0.01), math.sin(0.01), 0.0,
            -1.0, 0.0, 0.0,
        ],
        dtype=np.float32,
    )
    close = {"cell": 0}
    nearby = {"cell": 1}
    distant = {"cell": 2}
    assert _nearby_rift_cuts(xyz, close, nearby)
    assert not _nearby_rift_cuts(xyz, close, distant)
