"""A reward that pays for pace, not distance (GH-27).

Round 2 trained on :class:`~fly_driver.envs.practice_track.ProgressReward`, which pays metres
and nothing else. With a five-minute episode, crawling round the kerb at 28 km/h earned more
than driving fast and crashing at the next corner, so that is what one seed learned. This
reward makes time cost something, and makes ending the episode early cost more than any
amount of slow driving could:

    reward = progress_m                                 metres along the centreline
           - time_cost_fraction * v_target * dt         v_target = lap length / target lap time
           - v_target * time_weight * depth * dt        only past the kerb's outer edge
           - (crash_penalty_m + time cost * horizon_s)  once, on a crash or a stall
           + lap_bonus_m * target / lap_time            once, per completed lap
           + target_bonus_m                             once, if that lap beat the target

Everything is in metres, so the terms add up in one unit, and then multiplied by ``scale``.
PPO here does not normalise returns or values, so the scale is what keeps the critic's
targets small: at ``scale=0.01`` a crash is about -47 instead of -4,700. It multiplies every
term, so the ratios between them, and every property below, are unchanged.

**Pace beats distance.** Per second the car earns ``v - time_cost_fraction * v_target``, which
is negative below that pace (59 km/h for a 90 s target) and grows with speed. Standing still
and crawling lose reward, and target pace earns three times what it costs.

**Crashing is never a shortcut.** The crash term charges the time cost for a whole episode
(``horizon_s``) on top of ``crash_penalty_m``. No continuation with non-negative progress can
bleed more than that before the episode is truncated, so ending the episode early never
saves reward, discounted or not. The term is a constant rather than "the time left" because
the policy cannot see the clock, and a penalty it cannot predict is noise to the critic.
Riding past the kerb is the one cost the bound does not cover; the track limit ends that
within 0.15 of a car's width.

**Track limits.** The kerb counts as track, as in F1: the cost is driven by
:func:`~fly_driver.envs.centerline.off_track_fraction`, which measures from the car's outer
edge to the kerb's *outer* edge, so a car clipping the kerb with two wheels pays nothing. Past
it, each second costs what the env's lap-time penalty would add
(:attr:`~fly_driver.envs.penalty.PenaltyWeights.time_weight` seconds per unit depth), turned
into metres at target pace.

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
    """Progress less a time cost, with a crash/stall penalty and a lap-time bonus.

    Args:
        target_lap_time_s: The lap time the reward is calibrated to.
        time_cost_fraction: Fraction of target pace charged per second. Below
            ``time_cost_fraction * v_target`` the car loses reward.
        crash_penalty_m: Base cost of a crash or a stall, on top of the episode's time cost.
        horizon_s: Longest possible episode in seconds. The crash term charges the time cost
            over all of it. :class:`~fly_driver.envs.practice_track.PracticeTrack` fills it
            in from ``max_steps`` when built from a mapping.
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
    time_cost_fraction: float = 0.25
    crash_penalty_m: float = 500.0
    horizon_s: float = 300.0
    lap_bonus_m: float = 2000.0
    target_bonus_m: float = 1000.0
    penalty_weights: PenaltyWeights = field(default_factory=lambda: DEFAULT_PENALTY_WEIGHTS)
    lap_length_m: float = _LAP_LENGTH_M
    scale: float = 1.0

    def __post_init__(self) -> None:
        for name in ("target_lap_time_s", "horizon_s", "lap_length_m", "scale"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive, got {getattr(self, name)}")
        for name in ("time_cost_fraction", "crash_penalty_m", "lap_bonus_m", "target_bonus_m"):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must be non-negative, got {getattr(self, name)}")

    @property
    def target_speed_mps(self) -> float:
        """Average speed of a lap at the target time."""
        return self.lap_length_m / self.target_lap_time_s

    @property
    def time_cost_mps(self) -> float:
        """Metres charged per second: the pace below which driving loses reward."""
        return self.time_cost_fraction * self.target_speed_mps

    @property
    def crash_cost_m(self) -> float:
        """What a crash or a stall costs, as a positive number."""
        return self.crash_penalty_m + self.time_cost_mps * self.horizon_s

    def __call__(self, info: Mapping[str, Any], dt: float) -> tuple[float, dict[str, float]]:
        weights = self.penalty_weights
        depth = min(float(info["off_track_fraction"]), weights.maximum_depth)
        beyond_kerb = depth if depth > weights.minimum_depth else 0.0
        terms = {
            "progress": float(info["progress_m"]),
            "time": -self.time_cost_mps * dt,
            "track_limits": -self.target_speed_mps * weights.time_weight * beyond_kerb * dt,
            "crash": -self.crash_cost_m if info["crashed"] or info["stalled"] else 0.0,
            "lap_bonus": 0.0,
        }
        if info["lap_complete"]:
            lap_time = float(info["lap_time"])
            terms["lap_bonus"] = self.lap_bonus_m * self.target_lap_time_s / lap_time
            if lap_time <= self.target_lap_time_s:
                terms["lap_bonus"] += self.target_bonus_m
        terms = {name: self.scale * value for name, value in terms.items()}
        return float(sum(terms.values())), terms
