"""Shared T4/T5 direction-selectivity gate used by the flyvis-backed eye tests.

Both the hex-resampler gate and the ``FlyvisEye`` gate feed the same drifting
gratings and apply the same paired T4+T5 preference criterion, so the check
lives here once.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from fly_driver.eyes.stimuli import DIRECTIONS

PRESTIMULUS_FRAMES = 25
DRIFT_FRAMES = 50
SETTLING_DRIFT_FRAMES = 10
MOTION_READOUTS = ("T4a", "T4b", "T4c", "T4d", "T5a", "T5b", "T5c", "T5d")
MINIMUM_PREFERENCE_RATIO = 1.25
EXPECTED_PREFERENCES = (
    ("a", "left", "right"),
    ("b", "right", "left"),
    ("c", "up", "down"),
    ("d", "down", "up"),
)

Amplitudes = dict[str, dict[str, float]]
ReadoutResponse = Callable[[str, str], np.ndarray]


def measure_q95_amplitudes(readout_response: ReadoutResponse) -> Amplitudes:
    """Measure a robust near-peak response for every direction and readout.

    Args:
        readout_response: ``(direction, readout) -> (time, columns)`` responses
            to the grating drifting in ``direction``.

    Returns:
        ``amplitudes[direction][readout]``, the 95th percentile over columns and
        the settled part of the drift.
    """
    amplitudes: Amplitudes = {}
    response_start = PRESTIMULUS_FRAMES + SETTLING_DRIFT_FRAMES
    for direction in DIRECTIONS:
        amplitudes[direction] = {}
        for readout in MOTION_READOUTS:
            response = np.asarray(readout_response(direction, readout))
            amplitudes[direction][readout] = float(np.quantile(response[response_start:], 0.95))
    return amplitudes


def combine_t4_t5(amplitudes: Amplitudes, direction: str, subtype: str) -> float:
    """Average ON-pathway T4 and OFF-pathway T5 amplitudes for one subtype."""
    return (amplitudes[direction][f"T4{subtype}"] + amplitudes[direction][f"T5{subtype}"]) / 2


def assert_t4_t5_direction_selectivity(amplitudes: Amplitudes) -> None:
    """Assert opposite T4/T5 subtype channels prefer opposite grating motion."""
    for subtype, preferred_direction, opposite_direction in EXPECTED_PREFERENCES:
        preferred_response = combine_t4_t5(amplitudes, preferred_direction, subtype)
        opposite_response = combine_t4_t5(amplitudes, opposite_direction, subtype)
        assert preferred_response > MINIMUM_PREFERENCE_RATIO * opposite_response, (
            f"T4/T5{subtype} q95 response {preferred_response:.4f} for "
            f"{preferred_direction} did not exceed {opposite_direction} "
            f"{opposite_response:.4f} by {MINIMUM_PREFERENCE_RATIO:.2f}x; "
            f"individual amplitudes: {amplitudes}"
        )
