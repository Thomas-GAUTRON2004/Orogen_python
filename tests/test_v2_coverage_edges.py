"""Targeted edge-case tests to close V2 core statement coverage."""

import math
from types import SimpleNamespace

import numpy as np
import pytest

from orogen.v2.core import fast_noise, seed, sphere_mesh
from orogen.v2.core.mesh_bfs import (
    flood_from,
    hop_distance,
    hops_from,
    largest_piece_of,
    pieces_of,
    reaches_within,
    touches,
    within_km,
)
from orogen.v2.core.mesh_locate import locate
from orogen.v2.core.mesh_rings import rings_of
from orogen.v2.core.min_heap import MinHeap, _edge_km_of, forget_edge_km, spread_km
from orogen.v2.core.rotation import (
    quat_from_axis_angle,
    quat_integrate,
    quat_to_euler_pole,
    quat_to_mat3,
)
from orogen.v2.core.sim_mesh import build_sim_mesh, mesh_from_arrays
from orogen.v2.tectonics.occupancy import bin_index, mark_bin, may_have_crust_near


class Graph:
    def __init__(self, offsets, adjacency, xyz=None):
        self.num_cells = len(offsets) - 1
        self.adj_offset = np.asarray(offsets, dtype=np.int32)
        self.adj_list = np.asarray(adjacency, dtype=np.int32)
        self.xyz = np.asarray(
            xyz if xyz is not None else np.tile([1.0, 0.0, 0.0], self.num_cells)
        )


def test_seed_parser_edge_cases_and_random_seed_range(monkeypatch):
    assert seed._js_number("") == 0
    assert seed._js_number("Infinity") == math.inf
    assert seed._js_number("-Infinity") == -math.inf
    assert seed._js_number("0x10") == 16
    assert seed._js_number("0b11") == 3
    assert seed._js_number("0o10") == 8
    assert seed._js_number("1.25e2") == 125
    assert seed._js_number("0x") is None
    assert seed._js_number("not-a-number") is None
    sampler = seed.make_rand_int(123)
    assert 0 <= sampler(10) < 10
    with pytest.raises(ValueError):
        sampler(0)
    assert seed.is_word_seed(None) is False
    assert seed.is_word_seed("  ") is False
    assert seed.is_word_seed("Infinity") is True
    assert seed.seed_of(None) == 0
    assert seed.seed_of("Pangaea") == seed.seed_of("Pangaea")
    assert seed.is_word_seed("Pangaea") is True
    assert seed.seed_of("Pangaea") == seed.seed_of("Pangaea")
    assert seed.is_word_seed("Pangaea") is True
    assert seed.seed_of("0") == 0
    assert seed.seed_of("1.5") == 1.5
    monkeypatch.setattr(seed.secrets, "randbelow", lambda n: n - 1)
    assert seed.random_seed() == seed.SEEDS - 1


def test_quaternion_zero_rate_output_buffer_and_zero_pole():
    q = [1.0, 0.0, 0.0, 0.0]
    assert quat_integrate(q, [0.0, 0.0, 0.0], 10) == q
    out = [0.0] * 9
    assert quat_to_mat3(quat_from_axis_angle(0, 0, 1, math.pi / 2), out) is out
    assert quat_to_euler_pole(q) == {"lat": 90.0, "lon": 0.0, "angle": 0.0}


def test_heap_errors_ties_and_edge_cache_lifecycle():
    with pytest.raises(ValueError):
        MinHeap(0)
    heap = MinHeap(1)
    with pytest.raises(IndexError):
        heap.pop()
    heap.push(1.0, 10)
    assert heap.peek_key() == 1.0
    assert heap.pop() == 10
    m = Graph([0, 1, 2], [1, 0], [1.0, 0.0, 0.0, 0.0, 1.0, 0.0])
    first = _edge_km_of(m)
    assert _edge_km_of(m) is first
    forget_edge_km(m)
    assert _edge_km_of(m) is not first
    distance = np.array([0.0, math.inf])
    seen = []
    spread_km(
        m,
        distance,
        can_enter=lambda a, b: False,
        on_reach=lambda a, b: seen.append((a, b)),
    )
    assert math.isinf(distance[1]) and not seen


def test_mesh_bfs_uncovered_empty_and_filter_paths():
    # A source with no neighbours exercises the no-boundary fast path.
    isolated = Graph([0, 0, 0], [])
    assert within_km(isolated, lambda c: c == 0, 100).tolist() == [1, 0]
    m = Graph([0, 1, 3, 3], [1, 0, 2], [1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0])
    assert hop_distance(m, lambda c: c == 0, can_enter=lambda c: c != 1).tolist() == [
        0,
        -1,
        -1,
    ]
    assert hops_from(m, [0], can_enter=lambda c: c != 1)[0].tolist() == [0, -1, -1]
    assert flood_from(m, 0, lambda c: False).tolist() == [0, 0, 0]
    assert touches(m, 2, lambda c: True) is False
    assert reaches_within(m, 0, 3, lambda c: False, lambda c: c == 2) is False
    assert pieces_of(m, []) == []
    assert largest_piece_of(m, []) == []


def test_locate_skips_degenerate_candidate_triangles():
    # Collinear/zero-volume candidate around the nearest cell: no valid weights.
    mesh = SimpleNamespace(
        xyz=np.array([1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0]),
        adj_offset=np.array([0, 2, 2, 2]),
        adj_list=np.array([1, 2]),
        index=SimpleNamespace(nearest=lambda *_: 0),
    )
    cells, weights = locate(mesh, 1.0, 0.0, 0.0)
    assert cells.tolist() == [0, 0, 0]
    assert weights.tolist() == [1.0, 0.0, 0.0]


def test_ring_builder_grows_storage_and_handles_disconnected_mesh():
    # Dense cycle exceeds the initial 40*n storage only at larger radii.
    n = 50
    offsets = [0]
    adjacency = []
    for i in range(n):
        adjacency.extend(j for j in range(n) if j != i)
        offsets.append(len(adjacency))
    m = Graph(offsets, adjacency)
    rings = rings_of(m)
    assert len(rings["cells"]) > n * 40
    assert rings_of(m) is rings


def test_sphere_mesh_validation_and_exception_translation(monkeypatch):
    with pytest.raises(ValueError, match="at least 3"):
        sphere_mesh.generate_fibonacci_sphere(2, 0.0, lambda: 0.5)

    class BrokenDelaunay:
        def __init__(self, points):
            raise RuntimeError("bad geometry")

    monkeypatch.setattr(sphere_mesh, "Delaunay", BrokenDelaunay)
    with pytest.raises(ValueError, match="Delaunay triangulation failed"):
        sphere_mesh._closed_triangles(np.zeros((4, 2)), 4)

    class EmptyDelaunay:
        def __init__(self, points):
            self.simplices = np.empty((0, 3), dtype=np.int32)

    monkeypatch.setattr(sphere_mesh, "Delaunay", EmptyDelaunay)
    with pytest.raises(ValueError, match="invalid 2D triangulation"):
        sphere_mesh._closed_triangles(np.zeros((4, 2)), 4)


def test_delaunay_rejects_invalid_simplex_shape_and_indices(monkeypatch):
    class WrongShape:
        def __init__(self, points):
            self.simplices = np.array([0, 1, 2], dtype=np.int32)

    monkeypatch.setattr(sphere_mesh, "Delaunay", WrongShape)
    with pytest.raises(ValueError, match="invalid 2D triangulation"):
        sphere_mesh._closed_triangles(
            np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]]), 4
        )

    class OutOfRange:
        def __init__(self, points):
            self.simplices = np.array([[0, 1, 4]], dtype=np.int32)

    monkeypatch.setattr(sphere_mesh, "Delaunay", OutOfRange)
    with pytest.raises(ValueError, match="out-of-range vertex index"):
        sphere_mesh._closed_triangles(
            np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]]), 4
        )

    class NoHullDelaunay:
        def __init__(self, points):
            # Duplicate simplices leave no boundary edges in the edge counter.
            self.simplices = np.array([[0, 1, 2], [0, 1, 2]], dtype=np.int32)

    monkeypatch.setattr(sphere_mesh, "Delaunay", NoHullDelaunay)
    with pytest.raises(ValueError, match="no closed convex hull"):
        sphere_mesh._closed_triangles(
            np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]]), 4
        )


def test_sphere_mesh_orientation_and_invalid_cyclic_adjacency():
    points = np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0]])
    closed = sphere_mesh._closed_triangles(points, 3)
    assert len(closed) == 4
    with pytest.raises(ValueError, match="no neighbours"):
        sphere_mesh._cyclic_region_adjacency(np.empty((0, 3), dtype=np.int32), 1)
    with pytest.raises(ValueError, match="non-manifold"):
        sphere_mesh._cyclic_region_adjacency(np.array([[0, 1, 2], [0, 1, 3]]), 4)
    with pytest.raises(ValueError, match="broken cyclic"):
        sphere_mesh._cyclic_region_adjacency(
            np.array([[0, 1, 2], [0, 3, 4], [0, 5, 6]]), 7
        )
    with pytest.raises(ValueError, match="one ring"):
        sphere_mesh._cyclic_region_adjacency(
            np.array([[0, 1, 2], [0, 2, 1], [0, 3, 4], [0, 4, 3]]), 5
        )


def test_sim_mesh_input_validation():
    with pytest.raises(ValueError, match="exactly 3"):
        mesh_from_arrays(
            num_cells=2,
            xyz=np.zeros(3),
            adj_offset=np.zeros(3),
            adj_list=np.array([], dtype=int),
        )
    with pytest.raises(ValueError, match="CSR"):
        mesh_from_arrays(
            num_cells=2,
            xyz=np.zeros(6),
            adj_offset=np.zeros(2),
            adj_list=np.array([], dtype=int),
        )
    with pytest.raises(ValueError, match="at least 3"):
        build_sim_mesh(2, 1)


def test_occupancy_non_cacheable_mesh_and_missing_marks():
    class SlottedMesh:
        __slots__ = ("xyz", "edge_km")

        def __init__(self):
            self.xyz = np.array([1.0, 0.0, 0.0])
            self.edge_km = 100.0

    m = SlottedMesh()
    assert bin_index(m).count == 1800
    bins = bin_index(m)
    plate = SimpleNamespace()
    assert may_have_crust_near(plate, 0, 1, bins) is False
    mark_bin(plate, 0, 7, bins.count)
    assert may_have_crust_near(plate, 0, 7, bins) is True
    assert may_have_crust_near(plate, 0, 8, bins) is False


def test_simplex_noise_and_fbm_cover_corner_and_octave_paths():
    noise = fast_noise.FastNoise(123)
    # A broad deterministic lattice exercises all simplex corner attenuation branches.
    values = [
        noise.noise3d(x / 7, y / 9, z / 11)
        for x in range(-5, 6)
        for y in range(-5, 6)
        for z in range(-5, 6)
    ]
    assert all(math.isfinite(v) for v in values)
    assert math.isfinite(noise.fbm(0.1, 0.2, 0.3, octaves=1))
    assert math.isfinite(noise.fbm(0.1, 0.2, 0.3, octaves=4, persistence=0.5))


def test_world_craton_naming_fallback_and_sorted_letters():
    from orogen.v2.tectonics.world import TectonicWorld

    world = TectonicWorld(build_sim_mesh(12, 2), 3, {"simCells": 12})
    plate = world.add_plate("fallback")
    assert world.name_by_cratons(plate, "fallback") == "fallback"
    plate.crust.create(0, 2, -1000, 7)
    world.terraneLetters = {7: "Z"}
    assert world.name_by_cratons(plate, "fallback") == "Z"
    world.terraneLetters = {}
    assert world.name_by_cratons(plate, "fallback") == "fallback"


def test_sphere_mesh_normalizes_clockwise_simplex_winding(monkeypatch):
    class ClockwiseDelaunay:
        def __init__(self, points):
            self.simplices = np.array([[0, 1, 2]], dtype=np.int32)

    monkeypatch.setattr(sphere_mesh, "Delaunay", ClockwiseDelaunay)
    points = np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0]])
    triangles = sphere_mesh._closed_triangles(points, 3)
    assert triangles[0].tolist() == [0, 2, 1]


def test_bfs_duplicate_sources_and_locate_reused_output_buffers():
    mesh = Graph([0, 1, 2, 2], [1, 0], [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0])
    distances, origins = hops_from(mesh, [0, 0, 1, 1], max_hops=0)
    assert distances.tolist() == [0, 0, -1]
    assert origins.tolist() == [0, 1, -1]

    # Supplying both output arrays takes the reuse path instead of allocating.
    cells = np.empty(3, dtype=np.int32)
    weights = np.empty(3, dtype=np.float64)
    tri_mesh = SimpleNamespace(
        xyz=np.array(
            [
                1.0,
                0.0,
                0.0,
                0.0,
                1.0,
                0.0,
                0.0,
                0.0,
                1.0,
                -1.0,
                0.0,
                0.0,
            ],
            dtype=np.float32,
        ),
        adj_offset=np.array([0, 3, 3, 3, 3], dtype=np.int32),
        adj_list=np.array([1, 2, 3], dtype=np.int32),
        index=SimpleNamespace(nearest=lambda *_: 0),
    )
    got_cells, got_weights = locate(tri_mesh, 1.0, 1.0, 1.0, cells, weights)
    assert got_cells is cells and got_weights is weights
    assert cells.tolist() == [0, 1, 2]
    assert np.isclose(weights.sum(), 1.0)


def test_fibonacci_sphere_without_jitter_and_kinematic_pose_paths():
    xyz = sphere_mesh.generate_fibonacci_sphere(8, 0.0, lambda: 0.5)
    assert xyz.shape == (24,)
    from orogen.v2.tectonics.kinematics import pose_at_constant_omega

    q = [1.0, 0.0, 0.0, 0.0]
    assert pose_at_constant_omega(q, [0.0, 0.0, 0.0], 5.0) == q
    moved = pose_at_constant_omega(q, [0.0, 0.0, 1.0], math.pi / 2)
    assert np.allclose(np.linalg.norm(moved), 1.0)
    assert not np.allclose(moved, q)


def test_advance_plates_updates_only_living_plates():
    from orogen.v2.tectonics.kinematics import advance_plates

    class StubPlate:
        def __init__(self, alive):
            self.alive = alive
            self.quat = [1.0, 0.0, 0.0, 0.0]
            self.omega = [0.0, 0.0, 0.0]
            self.updated = False

        def set_quat(self, value):
            self.updated = True
            self.quat = value

    living, dead = StubPlate(True), StubPlate(False)
    world = SimpleNamespace(plates=[living, dead], time=2.0, step=4)
    advance_plates(world, 0.5)
    assert living.updated and not dead.updated
    assert world.time == 2.5 and world.step == 5
