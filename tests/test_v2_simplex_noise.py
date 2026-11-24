"""Simplex noise matches upstream js/simplex-noise.js (ed466f9) bit for bit.

Golden values were produced by running the upstream file under Node 22 on the
points below; they cover noise3D, fbm (5 octaves, default persistence) and
ridgedFbm (defaults). Regenerate with tests/reference/v2/js/simplex-noise.mjs.
"""

from __future__ import annotations

import pytest

from orogen.v2.core.simplex_noise import SimplexNoise

GOLDEN = [
    {
        "seed": 0,
        "points": [
            [-16.135338859695633, 13.359785095490416, 17.908099907407582],
            [-18.564856200742003, -19.538165870838874, -17.953791188985942],
            [10.631486713248997, 3.397189575898082, 16.565202119092085],
            [11.352015580680227, -6.674135507398349, -12.19547284403605],
        ],
        "n3": [
            0.3144704512704344,
            0.2778193615196438,
            0.32671908410018957,
            -0.396457970989767,
        ],
        "fbm": [
            0.27876731192603416,
            -0.3015144201547745,
            0.14705370196900677,
            -0.29713313873378727,
        ],
        "rf": [
            0.3647913577526721,
            0.2740260819730654,
            0.28748624001388134,
            0.2850742393882197,
        ],
    },
    {
        "seed": 941,
        "points": [
            [-16.135338859695633, 13.359785095490416, 17.908099907407582],
            [-18.564856200742003, -19.538165870838874, -17.953791188985942],
            [10.631486713248997, 3.397189575898082, 16.565202119092085],
            [11.352015580680227, -6.674135507398349, -12.19547284403605],
        ],
        "n3": [
            0.02809790919231812,
            -0.046634912329075906,
            0.5769980776963249,
            -0.648072275188376,
        ],
        "fbm": [
            0.1296324649057189,
            -0.1842877305520704,
            0.07299707989356148,
            -0.29523646997298497,
        ],
        "rf": [
            0.7753830383814733,
            0.5369988850960945,
            0.12078501709342968,
            0.08189536123168546,
        ],
    },
]


def _force_pure_python(monkeypatch, module) -> None:
    """Make the Python path really pure.

    fbm and ridged fBm call the sample by global name; importing with Numba rebinds
    that name to the compiled sample, so the Python path must rebind it back.
    """
    from orogen.v2.core import _numba_kernels as kernels

    monkeypatch.setattr(module, "NUMBA_AVAILABLE", False)
    monkeypatch.setattr(kernels, "_simplex_noise3d_kernel", kernels.simplex_noise3d_py)


@pytest.mark.parametrize("path", ["numba", "python"])
@pytest.mark.parametrize("block", GOLDEN, ids=lambda b: f"seed-{b['seed']}")
def test_simplex_noise_matches_upstream_bit_for_bit(block, path, monkeypatch) -> None:
    import orogen.v2.core.simplex_noise as module

    if path == "python":
        _force_pure_python(monkeypatch, module)
    elif not module.NUMBA_AVAILABLE:
        pytest.skip("numba is not installed")
    noise = SimplexNoise(block["seed"])
    for index, (x, y, z) in enumerate(block["points"]):
        assert noise.noise3D(x, y, z) == block["n3"][index]
        assert noise.fbm(x, y, z, 5) == block["fbm"][index]
        assert noise.ridged_fbm(x, y, z) == block["rf"][index]


@pytest.mark.parametrize("seed", [0, 941, 5501])
def test_numba_and_python_paths_are_bit_identical(seed, monkeypatch):
    import numpy as np

    import orogen.v2.core.simplex_noise as module

    if not module.NUMBA_AVAILABLE:
        pytest.skip("numba is not installed")
    rng = np.random.default_rng(seed)
    points = rng.uniform(-30.0, 30.0, size=(400, 3))
    fast = SimplexNoise(seed)
    fast_values = [
        (fast.noise3D(*p), fast.fbm(*p, 5), fast.ridged_fbm(*p)) for p in points
    ]
    _force_pure_python(monkeypatch, module)
    slow = SimplexNoise(seed)
    slow_values = [
        (slow.noise3D(*p), slow.fbm(*p, 5), slow.ridged_fbm(*p)) for p in points
    ]
    assert fast_values == slow_values
