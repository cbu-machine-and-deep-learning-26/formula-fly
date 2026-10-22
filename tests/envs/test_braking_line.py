"""The braking line's speed profile (GH-27): it must brake where a driver would."""

from __future__ import annotations

import numpy as np
import pytest

from fly_driver.envs.braking_line import BRAKE, speed_profile
from fly_driver.envs.centerline import Centerline
from fly_driver.envs.segments import find_segments

CENTERLINE = Centerline.load()
PROFILE = speed_profile(CENTERLINE)
#: The heavy-braking corners after Silverstone's straights, by where they sit on our lap.
HEAVY_BRAKING_M = {"Brooklands": 2000.0, "Stowe": 5100.0, "Vale": 5500.0}


def _corner_near(arclength_m: float):
    corners = [segment for segment in find_segments(CENTERLINE) if segment.is_corner]
    return min(corners, key=lambda corner: abs(corner.start_m - arclength_m))


@pytest.mark.parametrize("name", list(HEAVY_BRAKING_M))
def test_it_brakes_hard_before_the_heavy_braking_corners(name):
    corner = _corner_near(HEAVY_BRAKING_M[name])
    approach = (PROFILE.arclength_m > corner.start_m - 150.0) & (
        PROFILE.arclength_m < corner.start_m + 30.0
    )
    braking = approach & (PROFILE.bands == BRAKE)
    spacing = float(np.median(np.diff(PROFILE.arclength_m)))
    assert braking.sum() * spacing >= 40.0, f"{name}: under 40 m of braking"
    drop_kmh = 3.6 * (PROFILE.speed_mps[approach].max() - PROFILE.speed_mps[approach].min())
    assert drop_kmh >= 80.0, f"{name}: only {drop_kmh:.0f} km/h scrubbed"


def test_the_profile_is_physical():
    """Never above the corner limit, and a lap between the scripted 121 s and an F1 87 s."""
    assert np.all(PROFILE.speed_mps <= PROFILE.limit_mps + 1e-9)
    steps = np.linalg.norm(np.roll(PROFILE.points, -1, axis=0) - PROFILE.points, axis=1)
    lap_s = float(np.sum(steps / PROFILE.speed_mps))
    assert 87.0 < lap_s < 121.0
