"""Pace reward tests (GH-27), on fixed synthetic ``info`` streams.

Each test is one of the properties the reward is built to have. They drive the reward
directly rather than through the env, so a failure points at the formula and not at the car.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from fly_driver.envs.centerline import Centerline, Projection, off_track_fraction
from fly_driver.envs.lap import LapTimer
from fly_driver.envs.lap_time_reward import LapTimeReward
from fly_driver.envs.practice_track import DEFAULT_TRACK_LIMIT, ProgressReward
from fly_driver.interface import FRAME_RATE_HZ

DT = 1.0 / FRAME_RATE_HZ
GAMMA = 0.997
#: v5's reward (train_flyvis_cx_v5.yaml): no floor, every early end costs 18 m.
REWARD = LapTimeReward(target_lap_time_s=105.0, pace_floor=0.0, crash_penalty_m=18.0)
#: v5's slow rule: under 30 km/h for 3 s ends the episode.
MIN_SPEED_MPS = 8.3


def _info(
    *,
    speed_mps: float = 0.0,
    depth: float = 0.0,
    crashed: bool = False,
    stalled: bool = False,
    lap_time: float | None = None,
) -> dict:
    """One step's ``info`` with only the keys the rewards read."""
    return {
        "progress_m": speed_mps * DT,
        "lateral_m": 0.0,
        "off_track_fraction": depth,
        "off_track": depth > 0.0,
        "crashed": crashed,
        "stalled": stalled,
        "lap_complete": lap_time is not None,
        "lap_time": lap_time,
    }


def _drive(
    speed_mps: float, seconds: float, reward: LapTimeReward = REWARD, gamma: float = 1.0
) -> float:
    """Discounted return of holding one speed on track for ``seconds``."""
    step = reward(_info(speed_mps=speed_mps), DT)[0]
    return sum(step * gamma**t for t in range(round(seconds / DT)))


def _excursion(
    speed_mps: float, depths: list[float], reward: LapTimeReward = REWARD, *, exits: bool
) -> float:
    """Discounted return of an off-track excursion through ``depths``, one per step, that
    either ends in an off-track termination or steers back onto the track."""
    total = 0.0
    for step, depth in enumerate(depths):
        crashed = exits and step == len(depths) - 1
        info = _info(speed_mps=speed_mps, depth=depth, crashed=crashed)
        total += GAMMA**step * reward(info, DT)[0]
    return total


#: From the slow rule's limit to twice target pace, the range the bound on P covers.
SPEEDS_MPS = np.linspace(MIN_SPEED_MPS, 2.0 * REWARD.target_speed_mps, 60)


def _crash_at(speed_mps: float, crash_at_m: float, reward: LapTimeReward = REWARD) -> float:
    """Discounted return of driving ``crash_at_m`` at one speed and then crashing.

    The step count ``D / (v * dt)`` is kept fractional: rounding it to whole steps moves the
    crash by up to a step's distance, which at 100 m/s and 50 m is a 4% change in D and
    swamps the effect under test.
    """
    steps = crash_at_m / (speed_mps * DT)
    per_step = reward(_info(speed_mps=speed_mps), DT)[0]
    crash = reward(_info(crashed=True), DT)[0]
    return per_step * (1.0 - GAMMA**steps) / (1.0 - GAMMA) + GAMMA**steps * crash


class TestSlowingDownNeverPays:
    @pytest.mark.parametrize("crash_at_m", [50.0, 100.0, 300.0, 1000.0])
    def test_slowing_down_to_delay_a_crash_never_pays(self, crash_at_m):
        """With a crash a fixed distance ahead, the discounted return rises with speed."""
        returns = [_crash_at(v, crash_at_m) for v in SPEEDS_MPS]
        assert np.all(np.diff(returns) > 0.0)

    def test_the_penalty_is_the_largest_whole_metre_that_keeps_it_so(self):
        """One metre more and, 50 m from a crash, 30 km/h beats 30.4 km/h."""
        for penalty_m, is_faster_better in ((18.0, True), (19.0, False)):
            reward = replace(REWARD, crash_penalty_m=penalty_m)
            slow = _crash_at(MIN_SPEED_MPS, 50.0, reward)
            faster = _crash_at(MIN_SPEED_MPS + 0.1, 50.0, reward)
            assert (faster > slow) is is_faster_better, penalty_m

    def test_every_early_end_costs_the_same(self):
        crash = REWARD(_info(crashed=True), DT)[1]["crash"]
        stall = REWARD(_info(stalled=True), DT)[1]["crash"]
        assert crash == stall == -REWARD.crash_penalty_m < 0.0


class TestAnExitNeverPays:
    @pytest.mark.parametrize("speed_mps", [MIN_SPEED_MPS, 20.0, REWARD.target_speed_mps, 100.0])
    @pytest.mark.parametrize("seconds", [0.5, 2.0, 10.0])
    @pytest.mark.parametrize("depth", [0.02, 0.1, 0.149])
    def test_exiting_off_track_never_beats_steering_back_on(self, speed_mps, seconds, depth):
        """At ``depth`` past the kerb, the car either carries on out past the 0.15 limit
        and is terminated, or steers back to the kerb, each over ``seconds``. Steering back
        wins even with nothing earned afterwards."""
        steps = round(seconds / DT)
        exits = list(np.linspace(depth, 0.16, steps))
        recovers = list(np.linspace(depth, 0.0, steps))
        back_on = _excursion(speed_mps, recovers, exits=False)
        assert back_on >= 0.0
        assert _excursion(speed_mps, exits, exits=True) < back_on

    @pytest.mark.parametrize("speed_mps", [MIN_SPEED_MPS, 20.0, REWARD.target_speed_mps])
    @pytest.mark.parametrize("depth", [0.05, 0.1, 0.149])
    def test_no_step_forward_is_worth_less_than_zero(self, speed_mps, depth):
        """The cost past the kerb is capped at the step's pace, so riding it out never
        loses reward and ending it never saves any."""
        reward, terms = REWARD(_info(speed_mps=speed_mps, depth=depth), DT)
        assert reward >= 0.0 and terms["track_limits"] <= 0.0


class TestPace:
    def test_a_faster_metre_pays_more_up_to_the_cap(self):
        def per_metre(speed_mps: float) -> float:
            return REWARD(_info(speed_mps=speed_mps), DT)[0] / (speed_mps * DT)

        target = REWARD.target_speed_mps
        assert per_metre(0.1 * target) < per_metre(target) < per_metre(2.0 * target)
        assert per_metre(0.1 * target) == pytest.approx(0.1)
        assert per_metre(3.0 * target) == pytest.approx(REWARD.pace_cap)


class TestReversing:
    @pytest.mark.parametrize("speed_mps", [-0.1, -10.0, -REWARD.target_speed_mps])
    def test_reversing_costs_its_metres(self, speed_mps):
        reward, terms = REWARD(_info(speed_mps=speed_mps), DT)
        assert terms["pace"] == pytest.approx(speed_mps * DT) and reward < 0.0


class TestTheLapIsThePrize:
    def test_a_faster_lap_earns_a_bigger_bonus(self):
        bonuses = [REWARD(_info(lap_time=t), DT)[1]["lap_bonus"] for t in (150.0, 120.0, 100.0)]
        assert bonuses == sorted(bonuses) and len(set(bonuses)) == 3

    def test_beating_the_target_adds_the_target_bonus(self):
        target = REWARD.target_lap_time_s
        just_under = REWARD(_info(lap_time=target - 1e-6), DT)[1]["lap_bonus"]
        just_over = REWARD(_info(lap_time=target + 1e-6), DT)[1]["lap_bonus"]
        assert just_under - just_over == pytest.approx(REWARD.target_bonus_m)

    def test_no_lap_credit_without_a_full_lap_from_just_behind_the_line(self):
        """A random start 5 m behind the line crosses it almost at once. Only a full lap
        of progress after that crossing may complete a lap; rocking back and forth over
        the line must not."""
        centerline = Centerline.load()
        length = centerline.length
        timer = LapTimer(centerline)
        position, now = length - 5.0, 0.0
        timer.reset(position, now)

        def advance(metres: float) -> list[float]:
            nonlocal position, now
            laps = []
            for _ in range(round(abs(metres))):
                position = (position + np.sign(metres)) % length
                now += DT
                lap = timer.update(position, now)
                if lap is not None:
                    laps.append(lap)
            return laps

        assert advance(10.0) == []  # over the line: the out lap ends, timing starts
        for _ in range(20):  # circle the line
            assert advance(-10.0) + advance(10.0) == []
        assert advance(length - 10.0) == []
        assert len(advance(10.0)) == 1


class TestTrackLimits:
    """The kerb is free. Only the part of the car past the kerb's outer edge costs."""

    HALF_WIDTH_M = 5.0
    KERB_M = 1.0
    CAR_M = 2.0

    def _depth(self, lateral_m: float) -> float:
        projection = Projection(
            arclength=0.0,
            lateral=lateral_m,
            half_width_left=self.HALF_WIDTH_M,
            half_width_right=self.HALF_WIDTH_M,
            segment=0,
        )
        return off_track_fraction(projection, car_width_m=self.CAR_M, kerb_width_m=self.KERB_M)

    def _cost(self, lateral_m: float) -> float:
        info = _info(speed_mps=REWARD.target_speed_mps, depth=self._depth(lateral_m))
        return REWARD(info, DT)[1]["track_limits"]

    def test_two_wheels_on_the_kerb_cost_nothing(self):
        inner_wheels_on_the_edge = self.HALF_WIDTH_M  # outer wheels span the whole kerb
        for lateral in (self.HALF_WIDTH_M - 0.5, inner_wheels_on_the_edge):
            assert self._cost(lateral) == 0.0
            assert self._cost(-lateral) == 0.0

    def test_beyond_the_kerb_costs_more_the_further_over_the_car_is(self):
        costs = [self._cost(self.HALF_WIDTH_M + over) for over in (0.1, 0.2, 0.3)]
        assert 0.0 > costs[0] > costs[1] > costs[2]

    def test_it_costs_more_than_the_old_lateral_term(self):
        lateral = self.HALF_WIDTH_M + 0.2
        info = _info(speed_mps=REWARD.target_speed_mps, depth=self._depth(lateral))
        old = ProgressReward()(info | {"lateral_m": lateral}, DT)
        assert self._cost(lateral) < 10.0 * old[1]["lateral"]

    def test_the_kerb_never_reaches_the_termination_limit(self):
        """Termination is the env's, unchanged: a car on the kerb is nowhere near it."""
        assert self._depth(self.HALF_WIDTH_M) == 0.0 < DEFAULT_TRACK_LIMIT


class TestTerms:
    def test_the_terms_add_up_to_the_reward(self):
        info = _info(speed_mps=50.0, depth=0.1, lap_time=110.0)
        reward, terms = REWARD(info, DT)
        assert set(terms) == {"pace", "track_limits", "crash", "lap_bonus"}
        assert sum(terms.values()) == pytest.approx(reward)


class TestScale:
    SCALED = replace(REWARD, scale=0.01)

    def test_it_multiplies_every_term(self):
        info = _info(speed_mps=50.0, depth=0.1, crashed=True, lap_time=110.0)
        reward, terms = REWARD(info, DT)
        scaled_reward, scaled_terms = self.SCALED(info, DT)
        assert scaled_terms == pytest.approx({name: 0.01 * v for name, v in terms.items()})
        assert scaled_reward == pytest.approx(0.01 * reward)

    def test_a_lap_at_target_pace_lands_in_a_range_the_critic_can_fit(self):
        """Unnormalised PPO needs small returns: a target-pace lap should be roughly +60 to
        +150 at the scale the configs use."""
        reward = replace(REWARD, scale=0.01)
        driving = _drive(reward.target_speed_mps, reward.target_lap_time_s, reward)
        bonus = reward(_info(lap_time=reward.target_lap_time_s), DT)[1]["lap_bonus"]
        assert 60.0 < driving + bonus < 150.0


class TestTheDefaultIsUnchanged:
    def test_progress_reward_scores_as_before(self):
        """Hand-computed against the round-2 formula, so old runs reproduce exactly."""
        reward = ProgressReward()
        info = _info(speed_mps=50.0, depth=0.3) | {"lateral_m": -4.0}
        total, terms = reward(info, DT)
        assert terms == pytest.approx(
            {"progress": 1.0, "lateral": -0.008, "lap_bonus": 0.0, "off_track": -10.0}
        )
        assert total == pytest.approx(1.0 - 0.008 - 10.0)
        lap = _info(speed_mps=50.0, lap_time=100.0)
        assert reward(lap, DT)[1]["lap_bonus"] == 100.0
