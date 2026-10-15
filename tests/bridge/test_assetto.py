"""ControlVector onto AssettoCorsaGym's absolute action space, without AC installed."""

from __future__ import annotations

import numpy as np

from fly_driver.bridge.assetto import to_ac_action
from fly_driver.interface import ControlVector

# dallara_f317's norm_steer_at_max, from upstream's steer_map.csv.
F317_MAX_STEER = 0.5341796875


def test_extremes_map_to_full_lock_and_released_or_full_pedals() -> None:
    np.testing.assert_allclose(
        to_ac_action(ControlVector(-1.0, 0.0, 1.0), F317_MAX_STEER), [-F317_MAX_STEER, -1, 1]
    )
    np.testing.assert_allclose(
        to_ac_action(ControlVector(1.0, 1.0, 0.0), F317_MAX_STEER), [F317_MAX_STEER, 1, -1]
    )


def test_neutral_is_centred_with_both_pedals_released() -> None:
    np.testing.assert_allclose(to_ac_action(ControlVector.neutral(), F317_MAX_STEER), [0, -1, -1])
