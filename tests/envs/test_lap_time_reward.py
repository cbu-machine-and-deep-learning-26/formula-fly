"""Lap-time reward tests (GH-27), on fixed synthetic ``info`` streams.

Each test is one of the properties the reward is built to have. They drive the reward
directly rather than through the env, so a failure points at the formula and not at the car.
"""

from __future__ import annotations

import numpy as np
import pytest

from fly_driver.envs.centerline import Centerline, Projection, off_track_fraction
from fly_driver.envs.lap import LapTimer
from fly_driver.envs.lap_time_reward import LapTimeReward
from fly_driver.envs.practice_track import DEFAULT_TRACK_LIMIT, ProgressReward
from fly_driver.interface import FRAME_RATE_HZ

DT = 1.0 / FRAME_RATE_HZ
REWARD = LapTimeReward()


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


def _drive(speed_mps: float, seconds: float, reward: LapTimeReward = REWARD) -> float:
    """Return of holding one speed on track for ``seconds``."""
    return sum(reward(_info(speed_mps=speed_mps), DT)[0] for _ in range(round(seconds / DT)))


def _crash_after(speed_mps: float, seconds: float) -> float:
    """Return of driving at ``speed_mps`` for ``seconds`` and then crashing."""
    return _drive(speed_mps, seconds) + REWARD(_info(speed_mps=speed_mps, crashed=True), DT)[0]


class TestPaceBeatsDistance:
    def test_still_is_worse_than_crawling_is_worse_than_target_pace(self):
        horizon_s = 60.0
        still = _drive(0.0, horizon_s)
        crawling = _drive(30.0 / 3.6, horizon_s)
        pace = _drive(REWARD.target_speed_mps, horizon_s)
        assert still < crawling < 0.0 < pace

    def test_target_pace_earns_clearly_more_than_slow_driving_over_the_same_time(self):
        slow = _drive(28.0 / 3.6, 60.0)  # round 2's kerb-hugger
        pace = _drive(REWARD.target_speed_mps, 60.0)
        assert pace - slow > 2000.0


class TestCrashingIsNeverAShortcut:
    @pytest.mark.parametrize("speed_mps", [REWARD.time_cost_mps, REWARD.target_speed_mps])
    @pytest.mark.parametrize("crash_at_s", [1.0, 30.0, 150.0, 299.0])
    def test_crashing_early_never_beats_continuing_at_or_above_the_pace_threshold(
        self, speed_mps, crash_at_s
    ):
        assert _crash_after(speed_mps, crash_at_s) < _drive(speed_mps, REWARD.horizon_s)

    def test_crashing_costs_more_than_standing_still_for_the_rest_of_the_episode(self):
        """The bound behind the property: the slowest legal continuation, zero progress
        until truncation, still beats the crash term."""
        still_to_the_end = _drive(0.0, REWARD.horizon_s)
        crash = REWARD(_info(crashed=True), DT)[0]
        assert crash < still_to_the_end

    def test_a_stall_costs_the_same_as_a_crash(self):
        crash = REWARD(_info(crashed=True), DT)[1]["crash"]
        stall = REWARD(_info(stalled=True), DT)[1]["crash"]
        assert stall == crash == -REWARD.crash_cost_m


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
    """The kerb is track. Only the part of the car past the kerb's outer edge costs."""

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
        return REWARD(_info(depth=self._depth(lateral_m)), DT)[1]["track_limits"]

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
        old = ProgressReward()(_info(depth=self._depth(lateral)) | {"lateral_m": lateral}, DT)
        assert self._cost(lateral) < 10.0 * old[1]["lateral"]

    def test_the_kerb_never_reaches_the_termination_limit(self):
        """Termination is the env's, unchanged: a car on the kerb is nowhere near it."""
        assert self._depth(self.HALF_WIDTH_M) == 0.0 < DEFAULT_TRACK_LIMIT


class TestTerms:
    def test_the_terms_add_up_to_the_reward(self):
        info = _info(speed_mps=50.0, depth=0.1, lap_time=110.0)
        reward, terms = REWARD(info, DT)
        assert set(terms) == {"progress", "time", "track_limits", "crash", "lap_bonus"}
        assert sum(terms.values()) == pytest.approx(reward)


class TestScale:
    SCALED = LapTimeReward(scale=0.01)

    def test_it_multiplies_every_term(self):
        info = _info(speed_mps=50.0, depth=0.1, crashed=True, lap_time=110.0)
        reward, terms = REWARD(info, DT)
        scaled_reward, scaled_terms = self.SCALED(info, DT)
        assert scaled_terms == pytest.approx({name: 0.01 * v for name, v in terms.items()})
        assert scaled_reward == pytest.approx(0.01 * reward)

    def test_it_preserves_the_orderings(self):
        def drive(speed_mps: float) -> float:
            return _drive(speed_mps, 60.0, self.SCALED)

        assert drive(0.0) < drive(30.0 / 3.6) < 0.0 < drive(self.SCALED.target_speed_mps)
        crash = self.SCALED(_info(crashed=True), DT)[0]
        assert crash < drive(0.0) * self.SCALED.horizon_s / 60.0


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
