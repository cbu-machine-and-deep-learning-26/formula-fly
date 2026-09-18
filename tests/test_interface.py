"""Shared-contract tests (GH-16).

`AGENTS.md` §11 lists "action ranges/clipping" and "no silent resizing" as failures that
look plausible when wrong: a saturated action is invisible in a reward curve, and a
quietly resized frame corrupts the hex resampler's retinal geometry rather than raising.
"""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import numpy as np
import pytest

from fly_driver.interface import CONTROL_DTYPE, ControlVector, validate_frame


class TestControlVectorRanges:
    def test_accepts_in_range_values(self):
        control = ControlVector(steer=-0.5, throttle=0.8, brake=0.0)
        assert (control.steer, control.throttle, control.brake) == (-0.5, 0.8, 0.0)

    def test_accepts_exact_bounds(self):
        assert ControlVector(steer=-1.0, throttle=0.0, brake=0.0).steer == -1.0
        assert ControlVector(steer=1.0, throttle=1.0, brake=1.0).brake == 1.0

    @pytest.mark.parametrize(
        ("kwargs", "field"),
        [
            ({"steer": 1.5, "throttle": 0.0, "brake": 0.0}, "steer"),
            ({"steer": -1.01, "throttle": 0.0, "brake": 0.0}, "steer"),
            ({"steer": 0.0, "throttle": 1.2, "brake": 0.0}, "throttle"),
            ({"steer": 0.0, "throttle": -0.1, "brake": 0.0}, "throttle"),
            ({"steer": 0.0, "throttle": 0.0, "brake": 2.0}, "brake"),
            ({"steer": 0.0, "throttle": 0.0, "brake": -0.5}, "brake"),
        ],
    )
    def test_rejects_out_of_range(self, kwargs, field):
        """Must raise, never silently saturate."""
        with pytest.raises(ValueError, match=field):
            ControlVector(**kwargs)

    @pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
    def test_rejects_non_finite(self, bad):
        with pytest.raises(ValueError, match="finite"):
            ControlVector(steer=bad, throttle=0.5, brake=0.0)

    def test_rejects_non_numbers(self):
        with pytest.raises(TypeError, match="real number"):
            ControlVector(steer="0.5", throttle=0.5, brake=0.0)

    def test_is_immutable(self):
        control = ControlVector.neutral()
        with pytest.raises(FrozenInstanceError):
            control.steer = 0.5  # type: ignore[misc]


class TestControlVectorClipping:
    def test_clipping_is_explicit_and_correct(self):
        control = ControlVector.clipped(steer=-3.0, throttle=7.0, brake=-2.0)
        assert (control.steer, control.throttle, control.brake) == (-1.0, 1.0, 0.0)

    def test_clipping_leaves_in_range_values_alone(self):
        control = ControlVector.clipped(steer=0.25, throttle=0.5, brake=0.125)
        assert (control.steer, control.throttle, control.brake) == (0.25, 0.5, 0.125)

    def test_clipping_still_rejects_nan(self):
        """A clipped NaN would silently become a bound -- the worst kind of wrong."""
        with pytest.raises(ValueError, match="finite"):
            ControlVector.clipped(steer=float("nan"), throttle=0.5, brake=0.0)


class TestControlVectorArrays:
    def test_to_array_shape_dtype_and_order(self):
        array = ControlVector(steer=-0.25, throttle=0.5, brake=0.75).to_array()
        assert array.shape == (3,)
        assert array.dtype == CONTROL_DTYPE
        np.testing.assert_allclose(array, [-0.25, 0.5, 0.75])

    def test_round_trips(self):
        original = ControlVector(steer=0.1, throttle=0.2, brake=0.3)
        restored = ControlVector.from_array(original.to_array())
        np.testing.assert_allclose(restored.to_array(), original.to_array())

    def test_from_array_rejects_wrong_length(self):
        with pytest.raises(ValueError, match="3 components"):
            ControlVector.from_array([0.0, 0.0])

    def test_from_array_rejects_out_of_range_unless_clip_requested(self):
        with pytest.raises(ValueError):
            ControlVector.from_array([2.0, 0.0, 0.0])
        assert ControlVector.from_array([2.0, 0.0, 0.0], clip=True).steer == 1.0

    def test_neutral_is_all_zero(self):
        np.testing.assert_allclose(ControlVector.neutral().to_array(), [0.0, 0.0, 0.0])


class TestValidateFrame:
    def test_accepts_and_returns_the_same_object(self):
        frame = np.zeros((64, 64, 3), dtype=np.uint8)
        assert validate_frame(frame, (64, 64, 3)) is frame

    def test_rejects_wrong_shape_without_resizing(self):
        """A frame that quietly changed size would corrupt the hex lattice (GH-13)."""
        with pytest.raises(ValueError, match="No implicit resize"):
            validate_frame(np.zeros((32, 32, 3), dtype=np.uint8), (64, 64, 3))

    def test_rejects_float_frames_without_casting(self):
        with pytest.raises(TypeError, match="No implicit cast"):
            validate_frame(np.zeros((64, 64, 3), dtype=np.float32), (64, 64, 3))

    def test_rejects_missing_channel_dimension(self):
        with pytest.raises(ValueError):
            validate_frame(np.zeros((64, 64), dtype=np.uint8), (64, 64, 3))

    def test_rejects_non_array(self):
        with pytest.raises(TypeError, match="numpy array"):
            validate_frame([[0, 0, 0]], (64, 64, 3))
