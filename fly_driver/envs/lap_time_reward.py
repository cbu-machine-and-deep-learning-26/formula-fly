"""A reward that pays for pace and charges for every early end (GH-27).

Round 3 charged for time, so every second below ~50 km/h lost reward and ending the episode
was the cheapest policy: all 10 seeds learned to stall. Round 4 paid pace with no crash
penalty, so an early end cost only the reward it forfeited. Its best seeds drove 2.8 km at up
to 140 km/h, but the off-track cost made a run of kerb clipping worth less than ending it, so
creeping off the track paid, and a car that never passed 2 m/s parked for the whole episode.
Round 5 adds a real penalty for every early end and keeps every step on track worth >= 0:

    reward = progress_m * clip(v / v_target, pace_floor, pace_cap)   forward; v = progress_m / dt
           = progress_m                                              backwards
           - min(v_target * time_weight * depth * dt, pace)          past the kerb's outer edge
           - crash_penalty_m                                         once: crash, off track, stall
           + lap_bonus_m * target / lap_time                         once, per completed lap
           + target_bonus_m                                          once, if under the target

Everything is in metres, then multiplied by ``scale``. PPO here does not normalise returns
or values, so the scale keeps the critic's targets small: at ``scale=0.01`` a lap at target
pace is about +89.

**Slowing down to delay a crash never pays.** A discounted constant penalty is cheaper the
later it lands. For a crash D metres ahead at constant speed v, with ``lam = -ln(gamma) / dt``
and ``E = exp(-lam * D / v)``, the return is ``G = S - P * E`` with pace
``S = v^2 / (v_target * lam) * (1 - E)`` (``pace_floor`` 0, below the cap). Faster is better
while ``dG/dv > 0``, that is while ``P < S'(v) / E'(v)``.
The bound is tightest at the slowest speed the env allows and the shortest distance: at
gamma 0.997, a 105 s target, 30 km/h (the slow rule's limit) and D = 50 m it is 18.44 m, so
v5 uses ``crash_penalty_m: 18``. Below 30 km/h the env's slow rule ends the episode anyway.

**An exit never pays.** The cost past the kerb is capped at the step's own pace, so a step
on or off the track is worth >= 0 while the car moves forward. Steering back on, or riding
the excursion out, is worth at least 0; ending it costs ``crash_penalty_m``. Below the cap
the cost is what the env's lap-time penalty would add
(:attr:`~fly_driver.envs.penalty.PenaltyWeights.time_weight` seconds per unit depth) in
metres at target pace. The kerb counts as track, as in F1:
:func:`~fly_driver.envs.centerline.off_track_fraction` measures from the car's outer edge to
the kerb's *outer* edge, so two wheels on the kerb cost nothing.

**The lap is the prize.** The bonus reads ``info["lap_time"]``, which already includes the
off-track penalty seconds, and the env's :class:`~fly_driver.envs.lap.LapTimer` only counts a
full lap of forward progress from a crossing of the line, so laps cannot be farmed.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from fly_driver.envs.centerline import Centerline
from fly_driver.envs.penalty import DEFAULT_PENALTY_WEIGHTS, PenaltyWeights

__all__ = ["DEFAULT_TARGET_LAP_TIME_S", "LapTimeReward"]

#: Roughly F1 race pace at Silverstone (the record in issue #32 is 1:27.097). The simulated
#: car cannot reach it: see docs/training.md for the measured lap times.
DEFAULT_TARGET_LAP_TIME_S = 90.0

#: Silverstone's centreline length, read once so the reward needs no env to build.
_LAP_LENGTH_M = Centerline.load().length


@dataclass(frozen=True)
class LapTimeReward:
    """Pace-weighted progress, with a crash/stall penalty and a lap-time bonus.

    Args:
        target_lap_time_s: The lap time the reward is calibrated to.
        pace_cap: Most a forward metre can be weighted, as a multiple of target pace.
        pace_floor: Least a forward metre is weighted, however slow the car.
        crash_penalty_m: Cost of every early end: a crash, an off-track termination or a
            stall. See the module docstring for the largest that keeps faster better.
        lap_bonus_m: Bonus for a lap at exactly the target time, scaled by
            ``target / lap_time``.
        target_bonus_m: Extra bonus for a lap at or under the target.
        penalty_weights: The env's off-track rule. ``time_weight`` sets the cost past the
            kerb, ``minimum_depth`` and ``maximum_depth`` bound the depth it applies to.
        lap_length_m: Track length, for target pace.
        scale: Multiplies every term before it is returned and logged.

    Raises:
        ValueError: If a time or length is not positive, or a weight is negative.
    """

    target_lap_time_s: float = DEFAULT_TARGET_LAP_TIME_S
    pace_cap: float = 2.0
    pace_floor: float = 0.25
    crash_penalty_m: float = 0.0
    lap_bonus_m: float = 2000.0
    target_bonus_m: float = 1000.0
    penalty_weights: PenaltyWeights = field(default_factory=lambda: DEFAULT_PENALTY_WEIGHTS)
    lap_length_m: float = _LAP_LENGTH_M
    scale: float = 1.0

    def __post_init__(self) -> None:
        for name in ("target_lap_time_s", "pace_cap", "lap_length_m", "scale"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive, got {getattr(self, name)}")
        for name in ("pace_floor", "crash_penalty_m", "lap_bonus_m", "target_bonus_m"):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must be non-negative, got {getattr(self, name)}")

    @property
    def target_speed_mps(self) -> float:
        """Average speed of a lap at the target time."""
        return self.lap_length_m / self.target_lap_time_s

    def __call__(self, info: Mapping[str, Any], dt: float) -> tuple[float, dict[str, float]]:
        weights = self.penalty_weights
        depth = min(float(info["off_track_fraction"]), weights.maximum_depth)
        beyond_kerb = depth if depth > weights.minimum_depth else 0.0
        progress_m = float(info["progress_m"])
        if progress_m > 0.0:
            weight = min(progress_m / dt / self.target_speed_mps, self.pace_cap)
            weight = max(weight, self.pace_floor)
            pace = progress_m * weight
        else:
            pace = progress_m
        track_limits = self.target_speed_mps * weights.time_weight * beyond_kerb * dt
        terms = {
            "pace": pace,
            # Capped at the step's pace, so no step that moves forward is worth less than 0.
            "track_limits": -min(track_limits, max(pace, 0.0)),
            "crash": -self.crash_penalty_m if info["crashed"] or info["stalled"] else 0.0,
            "lap_bonus": 0.0,
        }
        if info["lap_complete"]:
            lap_time = float(info["lap_time"])
            terms["lap_bonus"] = self.lap_bonus_m * self.target_lap_time_s / lap_time
            if lap_time <= self.target_lap_time_s:
                terms["lap_bonus"] += self.target_bonus_m
        terms = {name: self.scale * value for name, value in terms.items()}
        return float(sum(terms.values())), terms
