"""Mid-ocean-ridge segment and fracture-zone coordinates (V2 model/segments.js)."""
from __future__ import annotations

import math

PLANET_RADIUS_KM = 6371.0
SEGMENT_KM = 400.0


def fracture_coordinate(px: float, py: float, pz: float, omega) -> float:
    """Return the ridge-floor coordinate in segment lengths about a spreading pole.

    Zero is reserved for cells with no recorded fracture-zone coordinate. A zero
    relative rotation uses the conventional half-segment coordinate.
    """
    ax, ay, az = map(float, omega)
    length = math.hypot(ax, ay, az)
    if length < 1e-12:
        return 0.5
    ax, ay, az = ax / length, ay / length, az / length
    cosine = max(-1.0, min(1.0, px * ax + py * ay + pz * az))
    distance = math.acos(cosine) * PLANET_RADIUS_KM / SEGMENT_KM
    return 1e-6 if distance == 0.0 else distance
