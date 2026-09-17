"""Aerodynamics tests (GH-16).

Two jobs. The first is the ordinary one: the quadratic law, the edge cases, the validation.

The second is checking that :data:`~fly_driver.envs.aero.SF70H_AERO` actually reproduces the
published behaviour of a 2017 Formula 1 car. Those numbers were not copied off a spec sheet
-- teams do not publish ClA or CdA -- they were derived from on-track figures, so they need
to be checked *against* those figures or they are just two numbers someone made up.
"""

from __future__ import annotations

import numpy as np
import pytest

from fly_driver.envs.aero import (
    SF70H_AERO,
    AeroConfig,
    downforce_n,
    drag_n,
    terminal_speed_mps,
)

#: Ferrari SF70H mass including driver, kg (2017 FIA minimum).
SF70H_MASS_KG = 728.0
#: Pirelli slick, dry, in its operating window. Published range is 1.5-1.8.
TYRE_MU = 1.7
G = 9.81


def kmh(speed_kmh: float) -> float:
    return speed_kmh / 3.6


class TestQuadraticLaw:
    def test_doubling_speed_quadruples_downforce(self):
        single = downforce_n(50.0, SF70H_AERO)
        double = downforce_n(100.0, SF70H_AERO)
        assert float(double / single) == pytest.approx(4.0)

    def test_doubling_speed_quadruples_drag(self):
        assert float(drag_n(100.0, SF70H_AERO) / drag_n(50.0, SF70H_AERO)) == pytest.approx(4.0)

    def test_zero_speed_gives_zero_force(self):
        assert float(downforce_n(0.0, SF70H_AERO)) == 0.0
        assert float(drag_n(0.0, SF70H_AERO)) == 0.0

    def test_matches_the_closed_form(self):
        speed = 70.0
        expected = 0.5 * SF70H_AERO.air_density * SF70H_AERO.cla * speed**2
        assert float(downforce_n(speed, SF70H_AERO)) == pytest.approx(expected)

    def test_accepts_arrays(self):
        speeds = np.array([0.0, 25.0, 50.0, 100.0])
        forces = downforce_n(speeds, SF70H_AERO)
        assert forces.shape == speeds.shape
        assert np.all(np.diff(forces) > 0)


class TestReversing:
    """An RL policy early in training reverses constantly. The model must stay sane."""

    def test_downforce_is_never_negative(self):
        assert float(downforce_n(-50.0, SF70H_AERO)) > 0

    def test_reversing_produces_the_same_magnitude(self):
        assert float(downforce_n(-50.0, SF70H_AERO)) == pytest.approx(
            float(downforce_n(50.0, SF70H_AERO))
        )

    def test_drag_magnitude_is_never_negative(self):
        assert float(drag_n(-50.0, SF70H_AERO)) > 0


class TestAeroConfigValidation:
    @pytest.mark.parametrize(
        "kwargs",
        [
            {"cla": 0.0, "cda": 1.0},
            {"cla": -1.0, "cda": 1.0},
            {"cla": 1.0, "cda": 0.0},
            {"cla": 1.0, "cda": 1.0, "air_density": 0.0},
            {"cla": 1.0, "cda": 1.0, "balance_front": 0.0},
            {"cla": 1.0, "cda": 1.0, "balance_front": 1.0},
            {"cla": 1.0, "cda": 1.0, "balance_front": 1.5},
        ],
    )
    def test_rejects_bad_values(self, kwargs):
        with pytest.raises(ValueError):
            AeroConfig(**kwargs)

    def test_air_density_is_configurable_for_altitude(self):
        """Mexico City runs ~0.9 kg/m^3 and cars are set up differently for it."""
        thin = AeroConfig(cla=4.0, cda=1.35, air_density=0.9)
        assert float(downforce_n(70.0, thin)) < float(downforce_n(70.0, SF70H_AERO))


class TestSF70HMatchesPublishedPerformance:
    """The coefficients were derived from on-track figures. Check they reproduce them."""

    def test_lift_to_drag_is_realistic_for_2017(self):
        """2017-era F1 sat around 3. Well outside 2-4.5 means the pair is inconsistent."""
        assert 2.0 < SF70H_AERO.lift_to_drag < 4.5

    def test_terminal_speed_matches_published_top_speed(self):
        """~746 kW at the crank, ~90% to the wheels. Published top speed is ~340 km/h."""
        wheel_power = 746_000 * 0.9
        top_speed_kmh = terminal_speed_mps(wheel_power, SF70H_AERO) * 3.6
        assert 310.0 < top_speed_kmh < 375.0, f"cda gives {top_speed_kmh:.0f} km/h"

    def test_copse_is_possible_at_290_kmh(self):
        """Copse was the fastest proper corner in F1 in 2017, taken at 290 km/h.

        Grip available is ``mu * (weight + downforce)``. Without the downforce term this
        is ~1.7 g and Copse is impossible, which is the entire reason this module exists.
        """
        weight = SF70H_MASS_KG * G
        load = weight + float(downforce_n(kmh(290.0), SF70H_AERO))
        lateral_g = TYRE_MU * load / SF70H_MASS_KG / G
        assert lateral_g > 4.5, f"only {lateral_g:.1f} g available at Copse speed"

    def test_braking_from_300_kmh_reaches_published_deceleration(self):
        """Vale: 300 -> 194 km/h in ~71 m, about 5.5 g."""
        weight = SF70H_MASS_KG * G
        speed = kmh(300.0)
        load = weight + float(downforce_n(speed, SF70H_AERO))
        total_force = TYRE_MU * load + float(drag_n(speed, SF70H_AERO))
        decel_g = total_force / SF70H_MASS_KG / G
        assert decel_g > 5.0, f"only {decel_g:.1f} g available braking from 300 km/h"

    def test_downforce_exceeds_car_weight_at_speed(self):
        """The headline property of a modern F1 car."""
        weight = SF70H_MASS_KG * G
        assert float(downforce_n(kmh(250.0), SF70H_AERO)) > weight

    def test_low_speed_grip_is_not_inflated(self):
        """At 50 km/h aero is nearly irrelevant, so grip must be roughly mu * g. Catches a
        cla so large the car is glued down everywhere, which would make the whole
        performance envelope wrong in a way the high-speed tests would not see."""
        weight = SF70H_MASS_KG * G
        load = weight + float(downforce_n(kmh(50.0), SF70H_AERO))
        assert TYRE_MU * load / SF70H_MASS_KG / G < 2.5

    def test_drag_at_top_speed_is_a_plausible_power_draw(self):
        """Sanity on cda from the other direction: drag power at 340 km/h should be most
        of, but not more than, the car's output."""
        speed = kmh(340.0)
        drag_power_kw = float(drag_n(speed, SF70H_AERO)) * speed / 1000.0
        assert 400.0 < drag_power_kw < 800.0, f"{drag_power_kw:.0f} kW to overcome drag"


class TestTerminalSpeed:
    def test_more_power_gives_more_speed(self):
        assert terminal_speed_mps(800_000, SF70H_AERO) > terminal_speed_mps(400_000, SF70H_AERO)

    def test_more_drag_gives_less_speed(self):
        draggy = AeroConfig(cla=4.0, cda=2.7)
        assert terminal_speed_mps(700_000, draggy) < terminal_speed_mps(700_000, SF70H_AERO)

    def test_cube_root_scaling(self):
        """Eight times the power is twice the speed -- the signature of a v^3 law."""
        low = terminal_speed_mps(100_000, SF70H_AERO)
        high = terminal_speed_mps(800_000, SF70H_AERO)
        assert high / low == pytest.approx(2.0)

    def test_rejects_non_positive_power(self):
        with pytest.raises(ValueError):
            terminal_speed_mps(0.0, SF70H_AERO)
