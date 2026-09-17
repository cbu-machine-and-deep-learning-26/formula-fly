"""Powertrain tests (GH-16).

The gearbox is the part most likely to be quietly wrong. A box that never leaves first
gear, or that hunts between two ratios every step, still produces a car that drives — it
just has the wrong acceleration everywhere, and nothing raises. Both of those actually
happened while writing this: the first ratio set reached 150 km/h in first and never used
gears six to eight.
"""

from __future__ import annotations

import numpy as np
import pytest

from fly_driver.envs.aero import SF70H_AERO, drag_n
from fly_driver.envs.powertrain import (
    SF70H_POWERTRAIN,
    PowertrainConfig,
    brake_torque,
    drive_torque,
    engine_speed_rads,
    engine_torque,
    select_gear,
)

P = SF70H_POWERTRAIN
WHEEL_RADIUS = 0.335
MASS_KG = 728.0
G = 9.81


def wheel_rads_at(speed_kmh: float) -> float:
    return (speed_kmh / 3.6) / WHEEL_RADIUS


def settled_gear(speed_kmh: float) -> int:
    """Gear the box converges on at a steady speed, from a standing start."""
    wheel = wheel_rads_at(speed_kmh)
    gear = 0
    for _ in range(P.num_gears * 2):
        gear = select_gear(wheel, gear, P)
    return gear


def tractive_n(speed_kmh: float) -> float:
    gear = settled_gear(speed_kmh)
    return 2.0 * drive_torque(1.0, wheel_rads_at(speed_kmh), gear, P) / WHEEL_RADIUS


class TestEngineTorque:
    def test_flat_below_the_power_crossover(self):
        low = engine_torque(P.idle_engine_rads * 1.2, P)
        assert low == pytest.approx(P.peak_torque_nm)

    def test_falls_as_power_limited_at_high_rpm(self):
        assert engine_torque(P.max_engine_rads, P) < P.peak_torque_nm

    def test_power_at_the_limiter_matches_the_rating(self):
        power = engine_torque(P.max_engine_rads, P) * P.max_engine_rads
        assert power == pytest.approx(P.peak_power_w, rel=0.01)

    def test_never_exceeds_rated_power(self):
        for rads in np.linspace(P.idle_engine_rads, P.max_engine_rads, 50):
            assert engine_torque(float(rads), P) * float(rads) <= P.peak_power_w * 1.001

    def test_below_idle_does_not_go_negative(self):
        """A stationary car must not be dragged backwards by engine braking."""
        assert engine_torque(0.0, P) > 0


class TestGearbox:
    def test_uses_first_gear_from_rest(self):
        assert settled_gear(5.0) == 0

    def test_reaches_top_gear_at_top_speed(self):
        assert settled_gear(340.0) == P.num_gears - 1

    def test_every_gear_is_used_somewhere(self):
        """The first ratio set never used gears 6-8. Nothing failed; it was just wrong."""
        used = {settled_gear(kmh) for kmh in range(10, 350, 5)}
        assert used == set(range(P.num_gears))

    def test_gear_rises_monotonically_with_speed(self):
        gears = [settled_gear(kmh) for kmh in range(10, 350, 10)]
        assert all(b >= a for a, b in zip(gears, gears[1:], strict=False))

    def test_first_gear_tops_out_at_a_realistic_speed(self):
        """F1 first gear runs out around 120 km/h. 150+ means the box is far too tall."""
        top_of_first = max(kmh for kmh in range(10, 350) if settled_gear(kmh) == 0)
        assert 90 <= top_of_first <= 150

    def test_engine_stays_under_the_limiter_in_the_chosen_gear(self):
        for kmh in range(20, 345, 5):
            rads = engine_speed_rads(wheel_rads_at(kmh), settled_gear(kmh), P)
            assert rads <= P.max_engine_rads * 1.001, f"over-revving at {kmh} km/h"

    def test_does_not_hunt_at_a_steady_speed(self):
        """Hysteresis check: once settled, the gear must not oscillate."""
        for kmh in (80, 120, 160, 200, 240, 280, 320):
            gear = settled_gear(kmh)
            wheel = wheel_rads_at(kmh)
            assert all(select_gear(wheel, gear, P) == gear for _ in range(20))

    def test_downshifts_when_slowing(self):
        fast = settled_gear(300.0)
        assert select_gear(wheel_rads_at(60.0), fast, P) < fast

    def test_gear_index_is_clamped(self):
        assert 0 <= select_gear(wheel_rads_at(100), 99, P) < P.num_gears
        assert 0 <= select_gear(wheel_rads_at(100), -5, P) < P.num_gears


class TestTractiveForce:
    def test_falls_with_speed(self):
        """Torque-limited low down, power-limited up top. A flat curve means the gearbox
        is not doing anything."""
        assert tractive_n(340.0) < tractive_n(200.0) < tractive_n(50.0)

    def test_tractive_force_and_drag_cross_at_a_realistic_top_speed(self):
        """The independent check that gearing and aero agree. Top speed is where tractive
        force falls to drag, so the car must still be pulling at 300 km/h and be beaten by
        drag by 350. Gearing and aero were derived separately, so them crossing in the
        right window is evidence rather than a tuned coincidence.

        Not asserted *at* 340 km/h: that is past this car's top speed, where the rev
        limiter has already cut torque to near zero, which is correct but makes the
        comparison meaningless."""
        assert tractive_n(300.0) > float(drag_n(300.0 / 3.6, SF70H_AERO))
        assert tractive_n(350.0) < float(drag_n(350.0 / 3.6, SF70H_AERO))

    def test_launch_force_exceeds_available_grip(self):
        """Not a defect. A real F1 car has several times more torque than grip in first,
        which is why launches are traction-limited and drivers modulate the throttle."""
        grip = 1.7 * MASS_KG * G * 0.55
        assert tractive_n(20.0) > grip

    def test_no_throttle_gives_no_torque(self):
        assert drive_torque(0.0, wheel_rads_at(100), 2, P) == 0.0

    def test_torque_scales_with_throttle(self):
        full = drive_torque(1.0, wheel_rads_at(100), 2, P)
        half = drive_torque(0.5, wheel_rads_at(100), 2, P)
        assert half == pytest.approx(full * 0.5)

    def test_throttle_is_clamped(self):
        assert drive_torque(5.0, wheel_rads_at(100), 2, P) == pytest.approx(
            drive_torque(1.0, wheel_rads_at(100), 2, P)
        )


class TestBrakes:
    DT = 0.002
    INERTIA = 1.0

    def _torque(self, brake, wheel_rads, front=True):
        return brake_torque(brake, wheel_rads, front, P, dt=self.DT, wheel_inertia=self.INERTIA)

    def test_opposes_forward_rotation(self):
        assert self._torque(1.0, 200.0) < 0

    def test_opposes_backward_rotation(self):
        assert self._torque(1.0, -200.0) > 0

    def test_never_reverses_a_wheel(self):
        """The property the old damper gave for free. Torque is clamped to exactly what
        stops the wheel this step, so it can reach zero but not cross it."""
        for omega in (0.01, 0.5, 5.0, 50.0):
            torque = self._torque(1.0, omega)
            delta_omega = torque / self.INERTIA * self.DT
            assert omega + delta_omega >= -1e-9

    def test_stopped_wheel_gets_no_torque(self):
        assert self._torque(1.0, 0.0) == 0.0

    def test_no_pedal_gives_no_torque(self):
        assert self._torque(0.0, 200.0) == 0.0

    def test_does_not_fade_with_speed(self):
        """The whole reason for replacing the damper. At 1.37 g from 150 km/h but 0.6 g
        from 64 km/h, the old brakes faded exactly when trying to stop."""
        fast = abs(self._torque(1.0, 250.0))
        slow = abs(self._torque(1.0, 80.0))
        assert slow == pytest.approx(fast)

    def test_front_gets_more_torque_than_rear(self):
        assert abs(self._torque(1.0, 200.0, front=True)) > abs(
            self._torque(1.0, 200.0, front=False)
        )

    def test_bias_split_matches_the_config(self):
        front = abs(self._torque(1.0, 200.0, front=True))
        rear = abs(self._torque(1.0, 200.0, front=False))
        assert front / (front + rear) == pytest.approx(P.brake_bias_front)

    def test_scales_with_pedal(self):
        assert abs(self._torque(0.5, 200.0)) == pytest.approx(abs(self._torque(1.0, 200.0)) * 0.5)

    def test_total_torque_can_exceed_tyre_grip(self):
        """Brakes must be able to lock a wheel; otherwise a policy can never get braking
        wrong, and the tyres stop being the limiting factor they are on a real car."""
        grip_torque = 1.7 * MASS_KG * G * WHEEL_RADIUS
        assert P.max_brake_torque_nm > grip_torque

    @pytest.mark.parametrize("bad", [{"dt": 0.0}, {"wheel_inertia": 0.0}])
    def test_rejects_bad_clamp_arguments(self, bad):
        kwargs = {"dt": self.DT, "wheel_inertia": self.INERTIA, **bad}
        with pytest.raises(ValueError):
            brake_torque(1.0, 100.0, True, P, **kwargs)


class TestAbsFactor:
    """The slip limiter. Measured without it: 40% pedal at 150 km/h locked the fronts
    88% of the time, and the car went straight on while the steering shook."""

    R = WHEEL_RADIUS

    def _factor(self, speed, wheel_rads, config=P):
        from fly_driver.envs.powertrain import abs_factor

        return abs_factor(speed, wheel_rads, self.R, config)

    def test_freely_rolling_wheel_gets_full_torque(self):
        assert self._factor(40.0, 40.0 / self.R) == 1.0

    def test_locked_wheel_gets_no_torque(self):
        assert self._factor(40.0, 0.0) == 0.0

    def test_ramps_between_the_thresholds(self):
        mid = (P.abs_slip_full + P.abs_slip_release) / 2
        wheel = (1.0 - mid) * 40.0 / self.R
        assert self._factor(40.0, wheel) == pytest.approx(0.5)

    def test_stands_down_at_walking_pace_so_the_car_can_stop(self):
        assert self._factor(P.abs_min_speed_mps * 0.5, 0.0) == 1.0

    def test_disabled_means_no_limiting(self):
        off = PowertrainConfig(
            peak_power_w=P.peak_power_w,
            peak_torque_nm=P.peak_torque_nm,
            max_engine_rads=P.max_engine_rads,
            idle_engine_rads=P.idle_engine_rads,
            max_brake_torque_nm=P.max_brake_torque_nm,
            abs_enabled=False,
        )
        assert self._factor(40.0, 0.0, off) == 1.0

    def test_reversing_is_symmetric(self):
        assert self._factor(10.0, -10.0 / self.R) == 1.0

    def test_monotonic_in_slip(self):
        speeds = [(1.0 - s) * 40.0 / self.R for s in (0.0, 0.1, 0.15, 0.2, 0.25, 0.3, 0.5)]
        factors = [self._factor(40.0, w) for w in speeds]
        assert factors == sorted(factors, reverse=True)


class TestPowertrainConfigValidation:
    def _valid(self, **overrides):
        base = {
            "peak_power_w": 746_000.0,
            "peak_torque_nm": 700.0,
            "max_engine_rads": 1570.0,
            "idle_engine_rads": 420.0,
            "max_brake_torque_nm": 16_000.0,
        }
        return {**base, **overrides}

    @pytest.mark.parametrize(
        "overrides",
        [
            {"peak_power_w": 0.0},
            {"peak_torque_nm": -1.0},
            {"max_brake_torque_nm": 0.0},
            {"final_drive": 0.0},
            {"gear_ratios": ()},
            {"gear_ratios": (1.0, 2.0, 3.0)},  # ascending
            {"gear_ratios": (2.0, 2.0)},  # duplicate
            {"driveline_efficiency": 1.5},
            {"brake_bias_front": 0.0},
            {"idle_engine_rads": 2000.0},  # above the limiter
            {"shift_up_fraction": 0.4, "shift_down_fraction": 0.6},  # inverted
            {"abs_slip_full": 0.5, "abs_slip_release": 0.3},  # inverted
            {"abs_slip_release": 1.5},
            {"abs_min_speed_mps": -1.0},
        ],
    )
    def test_rejects_bad_values(self, overrides):
        with pytest.raises(ValueError):
            PowertrainConfig(**self._valid(**overrides))

    def test_total_ratio_rejects_an_out_of_range_gear(self):
        with pytest.raises(IndexError):
            P.total_ratio(P.num_gears)

    def test_ratios_descend(self):
        assert list(P.gear_ratios) == sorted(P.gear_ratios, reverse=True)
