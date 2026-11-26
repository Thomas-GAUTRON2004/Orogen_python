"""Le repli pur Python reste exécuté, même sans Numba (job pure-python-fallback)."""

import numpy as np
import pytest

from orogen.v2.core.fast_noise import FastNoise


def test_python_fallback_remains_available_without_numba(monkeypatch):
    import orogen.v2.core.fast_noise as fast_noise_module

    monkeypatch.setattr(fast_noise_module, "NUMBA_AVAILABLE", False)
    noise = FastNoise(123)
    point = (0.125, -0.75, 1.5)
    expected = noise._fbm_python(*point, 4, 0.6)
    assert noise.noise3d(*point) == noise._noise3d_python(*point)
    assert noise.fbm(*point, 4, 0.6) == expected
    values = noise.fbm_many(np.asarray([point], dtype=np.float64), 4, 0.6)
    np.testing.assert_array_equal(values, np.asarray([expected]))
    with pytest.raises(ValueError, match="shape"):
        noise.fbm_many(np.zeros((2, 2)), 4, 0.6)
