import math

from orogen.v2.core.device import SMALL_DEVICE, is_small_device
from orogen.v2.core.sphere_geometry import polar_frame


def test_polar_frame_matches_euler_pole_convention():
    frame = polar_frame([0.0, 0.0, 1.0], [1.0, 0.0, 0.0])
    east = frame(1.0, 0.0, 0.0)
    north = frame(0.0, 1.0, 0.0)
    pole = frame(0.0, 0.0, 1.0)
    assert math.isclose(east["colat"], 90.0, abs_tol=1e-12)
    assert math.isclose(east["azimuth"], 0.0, abs_tol=1e-12)
    assert math.isclose(north["colat"], 90.0, abs_tol=1e-12)
    assert math.isclose(north["azimuth"], 90.0, abs_tol=1e-12)
    assert math.isclose(pole["colat"], 0.0, abs_tol=1e-12)


def test_small_device_rule_uses_memory_or_coarse_pointer():
    assert SMALL_DEVICE is False
    assert is_small_device(4.0, False)
    assert is_small_device(2.0, False)
    assert is_small_device(8.0, True)
    assert not is_small_device(8.0, False)
    assert not is_small_device(None, False)
