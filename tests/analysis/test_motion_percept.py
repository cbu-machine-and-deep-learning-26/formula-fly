"""The shared motion-percept maths (GH-75): one encoding for the webcam demo and the eye view.

The webcam demo's own tests (``tests/scripts/test_flyvis_eye_live.py``) still exercise the
fused colours through the script. These pin :func:`direction_channels`, which only the eye
view uses, and the colour wheel's anchor points, so the two tools cannot drift apart.
"""

from __future__ import annotations

import numpy as np
import pytest

from fly_driver.analysis.motion_percept import (
    METER_DIRECTIONS,
    SUBTYPE_DIRECTIONS,
    compute_motion_percept,
    direction_channels,
)

COLUMNS = 7


def _readouts(**active: float) -> dict[str, np.ndarray]:
    """All eight T4/T5 subtypes at zero, except ``active`` (name -> value) in column 3."""
    readouts = {f"{family}{sub}": np.zeros(COLUMNS) for family in ("T4", "T5") for sub in "abcd"}
    for name, value in active.items():
        readouts[name][3] = value
    return readouts


def _rest() -> dict[str, np.ndarray]:
    return _readouts()


class TestDirectionChannels:
    def test_every_subtype_lands_on_its_preferred_direction(self):
        for subtype, direction in SUBTYPE_DIRECTIONS.items():
            for family in ("T4", "T5"):
                channels = direction_channels(_readouts(**{family + subtype: 1.0}), _rest())
                assert channels[direction][3] == pytest.approx(1.0), family + subtype
                others = [name for name in METER_DIRECTIONS if name != direction]
                assert all(channels[name][3] == 0.0 for name in others)

    def test_t4_and_t5_of_one_direction_add(self):
        channels = direction_channels(_readouts(T4b=0.5, T5b=0.25), _rest())
        assert channels["right"][3] == pytest.approx(0.75)

    def test_activity_is_measured_from_rest(self):
        """A neuron sitting at its resting level is not motion, however high that level is."""
        rest = _readouts(T4a=2.0)
        assert direction_channels(_readouts(T4a=2.0), rest)["left"][3] == 0.0
        assert direction_channels(_readouts(T4a=2.5), rest)["left"][3] == pytest.approx(0.5)

    def test_below_rest_is_rectified_to_zero(self):
        rest = _readouts(T4c=1.0)
        assert direction_channels(_readouts(T4c=0.2), rest)["up"][3] == 0.0

    def test_other_cell_types_are_ignored(self):
        readouts = _readouts()
        readouts["Tm3"] = np.full(COLUMNS, 9.0)
        rest = _rest()
        rest["Tm3"] = np.zeros(COLUMNS)
        channels = direction_channels(readouts, rest)
        assert all(not channel.any() for channel in channels.values())

    def test_returns_all_four_directions(self):
        assert set(direction_channels(_readouts(), _rest())) == set(METER_DIRECTIONS)


class TestColourWheel:
    """The anchors the legend promises: right red, up yellow-green, left cyan, down violet."""

    @pytest.mark.parametrize(
        ("direction", "dominant"),
        [("right", "red"), ("left", "cyan"), ("down", "violet"), ("up", "yellow-green")],
    )
    def test_each_direction_has_its_hue(self, direction, dominant):
        channels = {name: np.zeros(1) for name in METER_DIRECTIONS}
        channels[direction][0] = 1.0
        ((red, green, blue),) = compute_motion_percept(channels, peak=1.0)[0]
        if dominant == "red":
            assert red > 0.9 and green < 0.1 and blue < 0.1
        elif dominant == "cyan":
            assert red < 0.1 and green > 0.9 and blue > 0.9
        elif dominant == "violet":
            assert blue > 0.9 and green < 0.1 and 0.4 < red < 0.6
        else:
            assert green > 0.9 and blue < 0.1 and 0.4 < red < 0.6

    def test_still_is_black(self):
        channels = {name: np.zeros(3) for name in METER_DIRECTIONS}
        colours, vectors = compute_motion_percept(channels, peak=1.0)
        assert not colours.any() and not vectors.any()
