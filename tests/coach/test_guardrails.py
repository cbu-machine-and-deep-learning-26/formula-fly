"""The reward coach's guardrails: only listed knobs, inside bounds, keeping the reward's rules."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from fly_driver.coach.guardrails import check_proposal, max_crash_penalty_m

V7A = Path(__file__).resolve().parents[2] / "fly_driver/configs/train_flyvis_cx_v7a.yaml"


@pytest.fixture()
def base() -> dict:
    return yaml.safe_load(V7A.read_text(encoding="utf-8"))


def test_the_crash_bound_is_the_one_lap_time_reward_derives() -> None:
    """lap_time_reward.py: 18.44 m at gamma 0.997, a 105 s target and the 30 km/h slow rule."""
    bound = max_crash_penalty_m(gamma=0.997, target_lap_time_s=105.0, slow_speed_mps=8.3)
    assert 18.0 < bound < 19.0


def test_a_lower_slow_rule_tightens_the_bound() -> None:
    """Slower crawling is allowed, so a smaller penalty already makes delaying a crash pay."""
    bounds = [
        max_crash_penalty_m(gamma=0.997, target_lap_time_s=105.0, slow_speed_mps=speed)
        for speed in (4.0, 8.3, 12.0)
    ]
    assert bounds == sorted(bounds)
    assert bounds[0] < 18.0


def test_a_valid_proposal_becomes_a_loadable_config(base: dict) -> None:
    check = check_proposal(base, [{"key": "env.params.reward.corner_bonus_m", "value": 120}])
    assert check.accepted, check.problems
    assert check.config["env"]["params"]["reward"]["corner_bonus_m"] == 120.0
    assert base["env"]["params"]["reward"]["corner_bonus_m"] == 50.0  # the base is untouched


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ([{"key": "env.params.reward.scale", "value": 0.1}], "not a knob"),
        ([{"key": "ppo.entropy_coef", "value": 0.5}], "outside"),
        ([{"key": "env.params.reward.crash_penalty_m", "value": 40}], "slowing down"),
        # v7a's 18 m penalty is over the bound once the slow rule allows 4 m/s.
        ([{"key": "env.params.stall_speed_mps", "value": 4.0}], "slowing down"),
        (
            [
                {"key": "ppo.entropy_coef", "value": 0.01},
                {"key": "ppo.learning_rate", "value": 5e-5},
                {"key": "ppo.target_kl", "value": 0.01},
            ],
            "at most 2",
        ),
        ([], "changes nothing"),
        ([{"key": "ppo.entropy_coef", "value": True}], "finite number"),
    ],
)
def test_bad_proposals_are_rejected(base: dict, changes: list, message: str) -> None:
    check = check_proposal(base, changes)
    assert not check.accepted
    assert check.config is None
    assert any(message in problem for problem in check.problems), check.problems


def test_the_coach_moves_the_focus_share_but_not_the_zones(base: dict) -> None:
    no_zones = check_proposal(base, [{"key": "env.params.start_focus.share", "value": 0.4}])
    assert not no_zones.accepted
    assert any("recipe's zones" in p for p in no_zones.problems)

    base["env"]["params"]["start_focus"] = {"share": 0.5, "zones": [[470.0, 620.0]]}
    more = check_proposal(base, [{"key": "env.params.start_focus.share", "value": 0.6}])
    assert more.accepted, more.problems
    assert more.config["env"]["params"]["start_focus"] == {"share": 0.6, "zones": [[470.0, 620.0]]}
    moved = check_proposal(base, [{"key": "env.params.start_focus.from_m", "value": 700}])
    assert not moved.accepted

    base["env"]["params"]["start_focus"]["zones"] = [[620.0, 470.0]]
    backwards = check_proposal(base, [{"key": "env.params.start_focus.share", "value": 0.4}])
    assert not backwards.accepted
