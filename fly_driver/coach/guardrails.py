"""What the reward coach may change, and the checks every proposed recipe must pass (GH-27).

Claude proposes; this module disposes. A proposal names dotted config keys and new values.
It becomes a config file only if every key is on :data:`KNOBS`, every value is inside that
knob's bounds, the result still loads as a :class:`~fly_driver.training.config.TrainConfig`
(which refuses unknown keys), and the reward still keeps the guarantees rounds 3-7 paid for:

- **Slowing down to delay a crash never pays.** :func:`max_crash_penalty_m` computes the bound
  derived in :mod:`fly_driver.envs.lap_time_reward` for the proposal's own ``gamma``, target
  lap time and slow-rule speed, and the proposed ``crash_penalty_m`` must stay under it. At
  v7's settings the bound is 18.44 m, as that module's docstring says; a test pins it.
- **One proposal tests one idea.** At most :data:`MAX_CHANGES_PER_PROPOSAL` knobs, because
  the Vale test changed two at once and could not say which one mattered (docs/training.md).
- **No scale changes.** ``reward.scale`` sets the critic's target size; PPO here does not
  normalise returns, so it is not a reward-shaping knob.
"""

from __future__ import annotations

import copy
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from fly_driver.training.config import TrainConfig

__all__ = [
    "KNOBS",
    "MAX_CHANGES_PER_PROPOSAL",
    "Knob",
    "ProposalCheck",
    "apply_changes",
    "check_proposal",
    "max_crash_penalty_m",
]

MAX_CHANGES_PER_PROPOSAL = 2
_LAP_M = 5886.8  # fly_driver.envs.centerline's Silverstone length, rounded
_DT_S = 1.0 / 50.0


@dataclass(frozen=True)
class Knob:
    """One setting the coach may change.

    Args:
        key: Dotted path into the run config.
        low: Smallest value allowed.
        high: Largest value allowed.
        meaning: What it does, for the prompt.
        integer: Must be a whole number.
    """

    key: str
    low: float
    high: float
    meaning: str
    integer: bool = False


KNOBS: dict[str, Knob] = {
    knob.key: knob
    for knob in (
        Knob(
            "env.params.reward.target_lap_time_s",
            100.0,
            130.0,
            "lap time the pace weight is calibrated to; target pace = lap length / this. "
            "The car's measured limit is ~100 s and the scripted lap 121 s",
        ),
        Knob("env.params.reward.pace_cap", 1.0, 3.0, "most a forward metre is weighted"),
        Knob("env.params.reward.pace_floor", 0.0, 0.5, "least a forward metre is weighted"),
        Knob(
            "env.params.reward.crash_penalty_m",
            0.0,
            100.0,
            "cost of every early end (crash, off-track termination, stall); must also stay "
            "under the slow-down-never-pays bound for the recipe's gamma, target and slow rule",
        ),
        Knob(
            "env.params.reward.corner_bonus_m",
            0.0,
            300.0,
            "paid once per corner per lap on a forward exit",
        ),
        Knob("env.params.reward.lap_bonus_m", 0.0, 5000.0, "lap bonus at the target time"),
        Knob("env.params.reward.target_bonus_m", 0.0, 3000.0, "extra for a lap under target"),
        Knob(
            "env.params.stall_speed_mps",
            4.0,
            12.0,
            "slow rule: under this speed for stall_seconds ends the episode",
        ),
        Knob("env.params.stall_seconds", 1.5, 6.0, "slow rule duration"),
        Knob(
            "env.params.track_limit",
            0.05,
            0.3,
            "fraction of car width past the kerb's outer edge that ends the episode",
        ),
        # The zones are the recipe's (a person picks them from docs/training.md); the coach
        # only moves how many training starts go to them.
        Knob(
            "env.params.start_focus.share",
            0.0,
            0.8,
            "share of training starts placed in the recipe's start_focus zones, split evenly "
            "between them; 0 = all uniform. Only when the recipe already has zones",
        ),
        Knob("ppo.entropy_coef", 0.0, 0.02, "entropy bonus"),
        Knob("ppo.learning_rate", 1e-5, 3e-4, "Adam step size (annealed)"),
        Knob("ppo.target_kl", 0.003, 0.03, "stop an update's epochs past this approx_kl"),
        Knob("ppo.gamma", 0.99, 0.999, "discount; 0.997 is a 6.7 s horizon at 50 Hz"),
        Knob("ppo.gae_lambda", 0.9, 0.99, "GAE trace decay"),
        Knob(
            "ppo.teacher_coef",
            0.0,
            2.0,
            "weight of the training-only braking teacher (MSE on mean throttle and brake toward "
            "a physics speed profile, from privileged arclength and speed); 0 turns it off. "
            "Fade it toward 0 once corners pass, so the policy's own pace takes over",
        ),
        Knob(
            "ppo.teacher_steer_coef",
            0.0,
            2.0,
            "weight of the training-only steering teacher (a Stanley controller on the "
            "centreline, from privileged lateral offset and heading); 0 turns it off. Fade it "
            "once the policy holds the road, so its own line takes over",
        ),
        Knob(
            "ppo.teacher_speed_fraction",
            0.7,
            1.0,
            "share of the physics speed profile the teacher brakes to; 0.85 passes every corner",
        ),
    )
}


def max_crash_penalty_m(
    *,
    gamma: float,
    target_lap_time_s: float,
    slow_speed_mps: float,
    pace_cap: float = 2.0,
    shortest_crash_m: float = 50.0,
    longest_crash_m: float = 2000.0,
) -> float:
    """The largest crash penalty for which driving faster to a fixed crash always pays.

    :mod:`fly_driver.envs.lap_time_reward` derives it: for a crash ``D`` metres ahead at
    constant speed ``v``, the discounted return is ``G = S - P * E`` with
    ``E = exp(-lam * D / v)``, ``lam = -ln(gamma) / dt`` and pace
    ``S = v^2 / (v_target * lam) * (1 - E)``. Faster is better while ``P < S'(v) / E'(v)``.
    This takes the minimum of that ratio over speeds from the slow rule's limit to the pace
    cap and crash distances from ``shortest_crash_m`` to ``longest_crash_m``.

    Returns:
        The bound in metres (before ``scale``).
    """
    lam = -math.log(gamma) / _DT_S
    v_target = _LAP_M / target_lap_time_s

    def ratio(v: float, distance: float) -> float:
        exponent = math.exp(-lam * distance / v)
        # d/dv of S and E, by hand; S' includes the (1 - E) factor's derivative.
        d_e = exponent * lam * distance / v**2
        d_s = (2 * v / (v_target * lam)) * (1 - exponent) - (v**2 / (v_target * lam)) * d_e
        return d_s / d_e if d_e > 0 else math.inf

    speeds = [slow_speed_mps + (pace_cap * v_target - slow_speed_mps) * i / 200 for i in range(201)]
    distances = [
        shortest_crash_m * (longest_crash_m / shortest_crash_m) ** (j / 60) for j in range(61)
    ]
    return min(ratio(v, d) for v in speeds for d in distances)


def _set_dotted(config: dict[str, Any], key: str, value: Any) -> None:
    parts = key.split(".")
    node = config
    for part in parts[:-1]:
        child = node.get(part)
        if not isinstance(child, dict):
            child = {}
            node[part] = child
        node = child
    node[parts[-1]] = value


def _get_dotted(config: Mapping[str, Any], key: str, default: Any = None) -> Any:
    node: Any = config
    for part in key.split("."):
        if not isinstance(node, Mapping) or part not in node:
            return default
        node = node[part]
    return node


def apply_changes(base: Mapping[str, Any], changes: Mapping[str, float]) -> dict[str, Any]:
    """A deep copy of ``base`` with each dotted ``key: value`` set."""
    config = copy.deepcopy(dict(base))
    for key, value in changes.items():
        knob = KNOBS[key]
        _set_dotted(config, key, int(value) if knob.integer else float(value))
    return config


@dataclass(frozen=True)
class ProposalCheck:
    """The verdict on one proposal.

    Args:
        accepted: Every check passed.
        problems: Why not, one line each.
        config: The full proposed config when accepted, else ``None``.
    """

    accepted: bool
    problems: tuple[str, ...]
    config: dict[str, Any] | None


def check_proposal(base: Mapping[str, Any], changes: Sequence[Mapping[str, Any]]) -> ProposalCheck:
    """Validate a proposal's ``[{"key": ..., "value": ...}]`` against ``base``.

    Args:
        base: The run config the proposal edits, as loaded from its ``config.yaml``.
        changes: The proposal's changes.
    """
    problems: list[str] = []
    flat: dict[str, float] = {}
    for change in changes:
        key, value = change.get("key"), change.get("value")
        if key not in KNOBS:
            problems.append(f"{key!r} is not a knob the coach may change")
            continue
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
        ):
            problems.append(f"{key}: value {value!r} is not a finite number")
            continue
        knob = KNOBS[key]
        if not knob.low <= value <= knob.high:
            problems.append(f"{key}: {value} is outside [{knob.low}, {knob.high}]")
        if key in flat:
            problems.append(f"{key} is changed twice")
        flat[str(key)] = float(value)
    if not flat and not problems:
        problems.append("the proposal changes nothing")
    if len(flat) > MAX_CHANGES_PER_PROPOSAL:
        problems.append(
            f"{len(flat)} knobs changed; one proposal may change at most "
            f"{MAX_CHANGES_PER_PROPOSAL}, so its result can be attributed"
        )
    if problems:
        return ProposalCheck(False, tuple(problems), None)

    config = apply_changes(base, flat)
    params = _get_dotted(config, "env.params", {}) or {}
    focus = params.get("start_focus")
    if isinstance(focus, Mapping):
        zones = focus.get("zones") or []
        if "share" not in focus or not zones:
            problems.append("start_focus needs a share and the recipe's zones")
        elif any(not 0.0 <= float(low) < float(high) <= _LAP_M for low, high in zones):
            problems.append(f"every start_focus zone must be [from_m, to_m] inside 0-{_LAP_M} m")
        elif not params.get("random_start"):
            problems.append("start_focus needs env.params.random_start: true")
        elif float(focus["share"]) == 0.0:
            params.pop("start_focus")

    reward = params.get("reward") or {}
    if reward.get("type") == "lap_time":
        bound = max_crash_penalty_m(
            gamma=float(_get_dotted(config, "ppo.gamma", 0.99)),
            target_lap_time_s=float(reward.get("target_lap_time_s", 90.0)),
            slow_speed_mps=float(params.get("stall_speed_mps") or 2.0),
            pace_cap=float(reward.get("pace_cap", 2.0)),
        )
        penalty = float(reward.get("crash_penalty_m", 0.0))
        if penalty > bound:
            problems.append(
                f"crash_penalty_m {penalty:g} is over {bound:.2f}, the largest for which slowing "
                "down to delay a crash never pays at this gamma, target and slow rule "
                "(fly_driver/envs/lap_time_reward.py; round 3 stalled on exactly this)"
            )
    if not problems:
        try:
            TrainConfig.from_dict(config)
        except (TypeError, ValueError) as error:
            problems.append(f"the proposed config does not load: {error}")
    if problems:
        return ProposalCheck(False, tuple(problems), None)
    return ProposalCheck(True, (), config)
