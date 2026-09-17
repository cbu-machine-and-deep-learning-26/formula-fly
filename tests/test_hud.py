"""Telemetry HUD tests (GH-16).

The numbers-to-pixels maths is pure and tested everywhere. The window is only opened where
a display exists; on a headless runner that test skips rather than fails.
"""

from __future__ import annotations

import pytest

from fly_driver.hud import TelemetryHUD, bar_fill, rpm_fraction, shift_light_on, steer_to_x


class TestRpm:
    def test_zero_at_rest(self):
        assert rpm_fraction(0.0, 15000.0) == 0.0

    def test_full_at_the_limiter(self):
        assert rpm_fraction(15000.0, 15000.0) == 1.0

    def test_clips_past_the_limiter(self):
        assert rpm_fraction(20000.0, 15000.0) == 1.0

    def test_rejects_a_zero_limiter(self):
        with pytest.raises(ValueError):
            rpm_fraction(1000.0, 0.0)

    def test_shift_light_comes_on_at_the_shift_point_not_the_limiter(self):
        assert not shift_light_on(14000.0, 14100.0)
        assert shift_light_on(14100.0, 14100.0)
        assert shift_light_on(14900.0, 14100.0)


class TestPedalBars:
    def test_fill_is_the_value(self):
        assert bar_fill(0.35) == 0.35

    def test_clips_both_ends(self):
        assert bar_fill(-0.2) == 0.0
        assert bar_fill(1.7) == 1.0


class TestSteeringBar:
    """A horizontal bar from -1 to +1, the ControlVector's own range."""

    def test_full_left_is_the_left_edge(self):
        assert steer_to_x(-1.0, 100.0, 300.0) == 100.0

    def test_full_right_is_the_right_edge(self):
        assert steer_to_x(1.0, 100.0, 300.0) == 300.0

    def test_centred_is_the_middle(self):
        assert steer_to_x(0.0, 100.0, 300.0) == 200.0

    def test_half_is_halfway(self):
        assert steer_to_x(0.5, 100.0, 300.0) == 250.0

    def test_clips_beyond_the_range(self):
        assert steer_to_x(3.0, 100.0, 300.0) == 300.0
        assert steer_to_x(-3.0, 100.0, 300.0) == 100.0

    def test_rejects_an_inverted_bar(self):
        with pytest.raises(ValueError):
            steer_to_x(0.0, 300.0, 100.0)


class TestWindow:
    def test_rejects_a_shift_point_past_the_limiter(self):
        with pytest.raises(ValueError):
            TelemetryHUD(limiter_rpm=15000.0, shift_rpm=16000.0)

    def test_opens_updates_and_closes_where_a_display_exists(self):
        try:
            hud = TelemetryHUD(limiter_rpm=15000.0, shift_rpm=14100.0, title="test")
        except RuntimeError as exc:  # pragma: no cover - headless runner
            pytest.skip(str(exc))
        try:
            hud.update(speed_kmh=217.0, gear=5, rpm=14500.0, throttle=0.8, brake=0.0, steer=-0.3)
            hud.update(speed_kmh=0.0, gear=1, rpm=0.0, throttle=0.0, brake=1.0, steer=1.0)
        finally:
            hud.close()
        assert hud.closed
        hud.update(speed_kmh=1.0, gear=1, rpm=1.0, throttle=0.0, brake=0.0, steer=0.0)  # no-op
