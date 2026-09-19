"""Performance-envelope tests: does the car behave like a Ferrari SF70H? (GH-16)

Everything else in the suite checks that the car works. This file checks that it is the
*right* car. Every previous fidelity bug passed all the structural tests: the wheels turned,
the brakes opposed motion, the camera pointed forward, and the car still took 4.85 s to
reach 100 km/h with no top speed at all.

Measurements are slow -- each one accelerates a car to racing speed -- so they are computed
once per module and shared. ``scripts/validate_car.py`` prints the same numbers for a human.
"""

from __future__ import annotations

import numpy as np
import pytest

from fly_driver.envs.aero import SF70H_AERO, downforce_n
from fly_driver.envs.car import SF70H_REFERENCE, CarConfig
from fly_driver.envs.performance import (
    G,
    PerformanceBed,
    measure_acceleration,
    measure_braking,
    measure_lateral,
    measure_sideslip,
    measure_top_speed,
)

pytestmark = pytest.mark.slow


@pytest.fixture(scope="module")
def bed() -> PerformanceBed:
    return PerformanceBed.build()


@pytest.fixture(scope="module")
def acceleration(bed) -> dict[str, float]:
    return measure_acceleration(bed)


@pytest.fixture(scope="module")
def top_speed(bed) -> dict[str, float]:
    return measure_top_speed(bed)


@pytest.fixture(scope="module")
def high_speed_braking(bed) -> dict[str, float]:
    return measure_braking(bed, 300.0, 194.0)


@pytest.fixture(scope="module")
def lateral(bed) -> dict[str, float]:
    return measure_lateral(bed, 250.0)


@pytest.fixture(scope="module")
def provoked(bed) -> dict[str, float]:
    """Full lock and full throttle at 60 km/h: the clumsiest input a driver can give."""
    return measure_sideslip(bed, 60.0)


class TestAcceleration:
    def test_zero_to_100(self, acceleration):
        """Published SF70H figure is ~2.6 s. Before this work it was 4.85 s."""
        assert 2.2 <= acceleration["0-100 km/h (s)"] <= 3.0

    def test_zero_to_200(self, acceleration):
        assert 4.0 <= acceleration["0-200 km/h (s)"] <= 9.0

    def test_zero_to_300(self, acceleration):
        """Reaching 300 km/h at all requires the gearbox to work through every ratio."""
        assert 9.0 <= acceleration["0-300 km/h (s)"] <= 16.0

    def test_launch_is_traction_limited(self, acceleration):
        """An F1 start is limited by grip, not power. Static analysis gives ~1.1 g and
        published figures quote 1.5-2 g with weight transfer. Well above this band means
        the tyre model is inventing grip that a real car does not have."""
        assert 1.0 <= acceleration["launch peak (g)"] <= 2.0

    def test_the_last_increment_is_the_slowest(self, acceleration):
        """200-300 km/h must take longer than 100-200: drag rises with v^2 and the engine
        is power-limited up there. Equal increments would mean constant tractive force,
        which is exactly the bug the powertrain replaced."""
        second = acceleration["0-200 km/h (s)"] - acceleration["0-100 km/h (s)"]
        third = acceleration["0-300 km/h (s)"] - acceleration["0-200 km/h (s)"]
        assert third > second

    def test_launch_is_slower_than_the_mid_range(self, acceleration):
        """A real F1 car's quickest 100 km/h is the *second* one, not the first: the start
        is limited by grip, and once rolling the car has full traction and no meaningful
        drag yet. Measured 2.66 s to 100 but only 2.26 s from 100 to 200. A model whose
        launch was its strongest phase would be producing grip out of nowhere."""
        first = acceleration["0-100 km/h (s)"]
        second = acceleration["0-200 km/h (s)"] - first
        assert first > second


class TestTopSpeed:
    def test_reaches_a_realistic_top_speed(self, top_speed):
        assert 320.0 <= top_speed["top speed (km/h)"] <= 360.0

    def test_top_speed_is_finite_at_all(self, top_speed):
        """The measurement only terminates because drag exists. Without aerodynamics the
        car accelerated until the integrator produced NaN at 667 km/h, so this passing is
        itself the check that aero is being applied."""
        assert np.isfinite(top_speed["top speed (km/h)"])

    def test_top_speed_is_reached_in_top_gear(self, top_speed):
        """If the car tops out in an intermediate gear the ratios are wrong, even though
        the speed might happen to look plausible."""
        assert top_speed["top gear"] == 8.0


class TestBraking:
    def test_high_speed_braking_distance(self, high_speed_braking):
        assert 30.0 <= high_speed_braking["brake 300->194 km/h (m)"] <= 60.0

    def test_high_speed_deceleration(self, high_speed_braking):
        """Published figure for Vale is about 5.5 g."""
        assert 3.5 <= high_speed_braking["brake 300->194 km/h avg (g)"] <= 6.0

    def test_peak_deceleration(self, high_speed_braking):
        assert high_speed_braking["brake 300->194 km/h peak (g)"] >= 4.0

    def test_low_speed_braking_saturates_the_tyres(self, bed):
        """The regression guard for the old damper brakes, whose force was proportional to
        wheel speed: they gave 0.6 g from 64 km/h, less than half the grip available.

        Low-speed deceleration is *expected* to be far below the high-speed figure, and
        that is not fade -- there is no downforce at 60 km/h, so mu*g is the ceiling. The
        real question is whether the brakes can reach that ceiling. They should get close
        to it, and anything well under means the brakes, not the tyres, are the limit."""
        mu = CarConfig().wheel_friction[0]
        low = measure_braking(bed, 100.0, 20.0)["brake 100->20 km/h avg (g)"]
        assert low >= 0.8 * mu, f"{low:.2f} g against a tyre ceiling of {mu:.2f} g"

    def test_car_actually_stops(self, bed):
        result = measure_braking(bed, 150.0, 0.0)
        assert 25.0 <= result["brake 150->0 km/h (m)"] <= 55.0


class TestCornering:
    def test_lateral_grip_at_speed(self, lateral):
        """Without downforce the ceiling is mu*g ~ 1.7 g. Anything in this band is only
        reachable because aerodynamic load is multiplying tyre grip."""
        assert 3.5 <= lateral["lateral at 250 km/h (g)"] <= 6.5

    def test_exceeds_what_tyres_alone_could_give(self, lateral):
        tyre_only = CarConfig().wheel_friction[0]
        assert lateral["lateral at 250 km/h (g)"] > tyre_only * 1.5


class TestHandlingBalance:
    """Payton: "increase rear grip so oversteer is not a significant issue (but still
    possible just not as easy)." Both halves of that are asserted here, because a grip
    balance is exactly the kind of thing that fails silently -- the car posts the same
    lap-time numbers either way and simply spins whenever you open the throttle.

    Worst sideslip reached under the same clumsy input, measured:

    ======================================  ======  ======
    Provocation                             mu 1.7  mu 1.8
    ======================================  ======  ======
    60 km/h, full lock and full throttle    63 deg  21 deg
    150 km/h, trail-brake then hard on gas  24 deg   5 deg
    ======================================  ======  ======
    """

    #: Past this the car has swapped ends rather than slid.
    SPIN_DEG = 45.0

    def test_full_lock_and_full_throttle_does_not_spin_the_car(self, provoked):
        angle = provoked["sideslip at 60 km/h (deg)"]
        assert angle < self.SPIN_DEG, f"spun to {angle:.0f} degrees of sideslip"

    def test_the_car_underneath_is_still_lively(self, bed):
        """Not a rail. The limiter should be shaping how the car slides, not standing in
        for a car that cannot. Switch it off and the back must step out much further than
        it does with the aid on, or the rear grip itself has gone too far."""
        from dataclasses import replace

        from fly_driver.envs.car import CarDynamics
        from fly_driver.envs.powertrain import SF70H_POWERTRAIN

        no_tc = replace(SF70H_POWERTRAIN, traction_control_enabled=False)
        bed.dynamics = CarDynamics(bed.model, bed.car, powertrain=no_tc)
        try:
            loose = measure_sideslip(bed, 60.0)["sideslip at 60 km/h (deg)"]
        finally:
            bed.dynamics = CarDynamics(bed.model, bed.car)
        assert loose > 20.0, f"only {loose:.1f} degrees with the aid off"

    def test_the_rear_can_be_provoked(self, provoked):
        """The point of loosening the limiter to 0.20. At AC's 0.10 this input produced
        1.9 degrees, which is a rail rather than a car, and a driver -- or a policy --
        cannot learn to catch a slide that never happens."""
        assert provoked["sideslip at 60 km/h (deg)"] > 6.0

    def test_the_limiter_is_still_doing_the_larger_part(self, bed, provoked):
        """Provokable, not undamped. Turning the aid off must still be a big step, or the
        limiter has been loosened until it may as well not be there."""
        from dataclasses import replace

        from fly_driver.envs.car import CarDynamics
        from fly_driver.envs.powertrain import SF70H_POWERTRAIN

        no_tc = replace(SF70H_POWERTRAIN, traction_control_enabled=False)
        bed.dynamics = CarDynamics(bed.model, bed.car, powertrain=no_tc)
        try:
            loose = measure_sideslip(bed, 60.0)["sideslip at 60 km/h (deg)"]
        finally:
            bed.dynamics = CarDynamics(bed.model, bed.car)
        assert loose > 2.0 * provoked["sideslip at 60 km/h (deg)"]

    def test_a_fast_corner_stays_planted(self, bed):
        """Where downforce dominates, nothing the driver does should unstick it."""
        result = measure_sideslip(bed, 200.0, steer=-0.4, throttle=1.0, seconds=2.0)
        assert result["sideslip at 200 km/h (deg)"] < 10.0


class TestAeroLoad:
    def test_downforce_exceeds_car_weight_at_speed(self):
        car = CarConfig()
        speed = SF70H_REFERENCE["copse_speed_kmh"] / 3.6
        assert float(downforce_n(speed, SF70H_AERO)) > car.mass_kg * G

    def test_copse_is_possible(self):
        """Copse was taken at 290 km/h in 2017, needing roughly 5.5 g."""
        car = CarConfig()
        weight = car.mass_kg * G
        speed = SF70H_REFERENCE["copse_speed_kmh"] / 3.6
        load = weight + float(downforce_n(speed, SF70H_AERO))
        assert car.wheel_friction[0] * load / car.mass_kg / G > 4.5


class TestBrakingStability:
    """Payton: "when braking, even small amounts, the front wobble or shake aggressively,
    putting the car out of control." Two causes, both measured, both pinned here."""

    def _brake_with_a_touch_of_steering(self, bed, brake: float):
        import mujoco

        from fly_driver.interface import ControlVector

        joint = mujoco.mj_name2id(bed.model, mujoco.mjtObj.mjOBJ_JOINT, "steer_fl")
        address = bed.model.jnt_qposadr[joint]
        front = bed.dynamics._wheel_dof["fl"]
        radius = bed.car.wheel_radius_m
        bed.reset()
        bed.accelerate_to(150.0 / 3.6)
        angles, slips = [], []
        for _ in range(int(1.5 / bed.dt)):
            bed.dynamics.step(ControlVector(steer=0.08, throttle=0.0, brake=brake), bed.data, 1)
            speed = bed.speed()
            angles.append(float(bed.data.qpos[address]))
            slips.append((speed - abs(float(bed.data.qvel[front])) * radius) / max(speed, 1.0))
        angles = np.degrees(np.array(angles))
        return float(angles.max() - angles.min()), float((np.array(slips) > 0.5).mean())

    def test_front_wheels_do_not_shimmy_under_braking(self, bed):
        """Was 27.9 degrees peak-to-peak at 19 Hz with the undamped kingpin."""
        wobble, _ = self._brake_with_a_touch_of_steering(bed, 0.4)
        assert wobble < 5.0, f"{wobble:.1f} degrees of steering oscillation"

    def test_front_wheels_do_not_lock_at_a_moderate_pedal(self, bed):
        """Was locked 88% of the time at 40% pedal. A locked, steered wheel steers nothing."""
        _, locked = self._brake_with_a_touch_of_steering(bed, 0.4)
        assert locked < 0.2, f"front wheel locked {100 * locked:.0f}% of the time"

    def test_full_pedal_still_stops_the_car(self, bed):
        """The limiter must never make the brakes weaker overall."""
        result = measure_braking(bed, 150.0, 0.0)
        assert result["brake 150->0 km/h (m)"] < 55.0


class TestSuspensionAtSpeed:
    def test_roll_stays_flat_in_a_fast_corner(self, bed):
        """F1 cars roll about a degree. Much more and the springs or bars are too soft;
        the aero platform would move and downforce with it."""

        from fly_driver.interface import ControlVector

        bed.reset()
        bed.accelerate_to(200.0 / 3.6)
        rolls = []
        for i in range(int(2.0 / bed.dt)):
            bed.dynamics.step(ControlVector(steer=-0.25, throttle=0.55, brake=0.0), bed.data, 1)
            if i > int(1.0 / bed.dt):
                rotation = bed.data.xmat[bed.dynamics._body].reshape(3, 3)
                rolls.append(abs(np.degrees(np.arctan2(rotation[2, 1], rotation[2, 2]))))
        assert np.mean(rolls) < 1.5, f"{np.mean(rolls):.2f} degrees of roll"

    def test_floor_never_touches_the_road(self, bed):
        """Aero load compresses the springs ~20 mm at top speed and braking pitches the
        nose; the collision box must clear the ground through all of it, or the chassis
        grinds and the car stops dead."""
        import mujoco

        from fly_driver.interface import ControlVector

        chassis = mujoco.mj_name2id(bed.model, mujoco.mjtObj.mjOBJ_GEOM, "chassis")
        bed.reset()
        bed.accelerate_to(320.0 / 3.6)
        clearance = float("inf")
        for _ in range(int(4.0 / bed.dt)):
            bed.dynamics.step(ControlVector(steer=0.0, throttle=0.0, brake=1.0), bed.data, 1)
            bottom = bed.data.geom_xpos[chassis][2] - bed.model.geom_size[chassis][2]
            clearance = min(clearance, float(bottom))
            if bed.speed() < 2.0:
                break
        assert clearance > 0.05, f"box came within {1000 * clearance:.0f} mm of the road"


class TestNumericalStability:
    def test_no_nan_over_a_long_hard_rollout(self, bed):
        """`AGENTS.md` §11. Two separate blow-ups happened while building this: the car
        with no drag reached 667 km/h and produced NaN, and an unlimited driven wheel
        reached 4321 rad/s and did the same."""
        from fly_driver.interface import ControlVector

        bed.reset()
        rng = np.random.default_rng(0)
        for _ in range(60):
            control = ControlVector.clipped(
                steer=float(rng.uniform(-1, 1)),
                throttle=float(rng.uniform(0, 1)),
                brake=float(rng.uniform(0, 0.4)),
            )
            bed.dynamics.step(control, bed.data, int(0.1 / bed.dt))
            assert np.all(np.isfinite(bed.data.qpos))
            assert np.all(np.isfinite(bed.data.qvel))
            assert np.all(np.isfinite(bed.data.qacc))

    def test_wheel_speed_stays_bounded_under_full_throttle(self, bed):
        """The rev limiter is the only thing bounding a spinning wheel. Without it the
        rear wheel reached 4321 rad/s with 1435 m/s of slip."""
        from fly_driver.interface import ControlVector

        bed.reset()
        full = ControlVector(steer=0.0, throttle=1.0, brake=0.0)
        bed.dynamics.step(full, bed.data, int(10.0 / bed.dt))
        for side in ("rl", "rr"):
            omega = abs(float(bed.data.qvel[bed.dynamics._wheel_dof[side]]))
            assert omega < 400.0, f"{side} wheel at {omega:.0f} rad/s"
