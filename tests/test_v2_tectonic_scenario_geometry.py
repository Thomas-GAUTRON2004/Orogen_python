import numpy as np

from orogen.v2.tectonics.scenario_geometry import heading, omega_for


def test_heading_and_angular_velocity_form_a_unit_spherical_motion():
    point = np.asarray([1.0, 0.0, 0.0])
    direction = heading(point, 90.0)
    assert np.isclose(np.linalg.norm(direction), 1.0)
    assert np.isclose(np.dot(point, direction), 0.0)
    omega = omega_for(point, direction, 5.0)
    assert np.isfinite(omega).all()
    assert np.linalg.norm(omega) > 0.0


def test_heading_accepts_list_input_and_signed_bearings():
    point = [0.0, 1.0, 0.0]
    east = heading(point, 90.0)
    west = heading(point, -90.0)
    assert np.isclose(np.linalg.norm(east), 1.0)
    assert np.isclose(np.linalg.norm(west), 1.0)
    assert np.isclose(np.dot(east, west), -1.0)
