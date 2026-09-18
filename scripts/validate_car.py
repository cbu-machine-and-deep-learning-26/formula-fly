"""Print the car's performance envelope against published SF70H figures (GH-16).

``AGENTS.md`` §11 says to test numerics rather than eyeball them.
``tests/test_car_performance.py`` does the asserting; this script exists so a human can see
the whole envelope at once and judge whether the thing is an F1 car, which no single
assertion conveys.

Run it::

    ./.venv/Scripts/python.exe scripts/validate_car.py

The measurement code lives in :mod:`fly_driver.envs.performance` so the script and the
tests cannot drift apart.
"""

from __future__ import annotations

from fly_driver.envs.aero import SF70H_AERO, downforce_n
from fly_driver.envs.car import SF70H_REFERENCE
from fly_driver.envs.performance import (
    G,
    PerformanceBed,
    measure_acceleration,
    measure_braking,
    measure_lateral,
    measure_top_speed,
)

#: Acceptance bands, derived from physics wherever the published figures disagree with
#: themselves. One widely quoted description of Vale says 300 -> 194 km/h "in roughly 71
#: metres, about 5.5 g" -- but 5.5 g over that speed change is 37 m, not 71. The
#: deceleration figure is consistent with everything else, so the distance band comes from
#: it rather than from the quoted distance.
#:
#: Launch g is allowed up to 2.0. An F1 start is traction-limited, not power-limited: the
#: static analysis gives ~1.1 g, and published figures quote 1.5-2 g once weight transfer
#: loads the rear axle. Far above this band means the tyre model is inventing grip.
TARGETS: dict[str, str] = {
    "0-100 km/h (s)": "2.2 - 3.0",
    "0-200 km/h (s)": "4 - 9",
    "0-300 km/h (s)": "9 - 16",
    "launch peak (g)": "1.0 - 2.0 (traction limited)",
    "top speed (km/h)": "320 - 360",
    "brake 300->194 km/h (m)": "30 - 60",
    "brake 300->194 km/h avg (g)": "3.5 - 6.0",
    "brake 300->194 km/h peak (g)": ">= 4.0",
    "brake 150->0 km/h (m)": "25 - 55",
    "lateral at 250 km/h (g)": "3.5 - 6.5",
}


def main() -> int:
    bed = PerformanceBed.build()
    results: dict[str, float] = {}
    results.update(measure_acceleration(bed))
    results.update(measure_top_speed(bed))
    results.update(measure_braking(bed, 300.0, 194.0))
    results.update(measure_braking(bed, 150.0, 0.0))
    results.update(measure_lateral(bed, 250.0))

    speed = SF70H_REFERENCE["copse_speed_kmh"] / 3.6
    weight = bed.car.mass_kg * G
    downforce = float(downforce_n(speed, SF70H_AERO))
    results["downforce at 290 km/h (x weight)"] = downforce / weight
    results["grip at 290 km/h (g)"] = (
        bed.car.wheel_friction[0] * (weight + downforce) / bed.car.mass_kg / G
    )

    print("\nFerrari SF70H -- measured in MuJoCo")
    print(f"{'metric':<34}{'measured':>12}   target")
    print("-" * 78)
    for name, value in results.items():
        print(f"{name:<34}{value:>12.2f}   {TARGETS.get(name, '')}")

    print(
        "\nKnown simplification: MuJoCo uses one Coulomb mu per contact, but real tyres lose\n"
        "grip coefficient as vertical load rises. Under heavy downforce this over-predicts\n"
        "high-speed grip. Calibratable once Assetto Corsa's tyre data is available."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
