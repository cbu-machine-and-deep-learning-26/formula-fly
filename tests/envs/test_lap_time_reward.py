"""Pace reward tests (GH-27), on fixed synthetic ``info`` streams.

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


def _drive(
    speed_mps: float, seconds: float, reward: LapTimeReward = REWARD, gamma: float = 1.0
) -> float:
    """Discounted return of holding one speed on track for ``seconds``."""
    step = reward(_info(speed_mps=speed_mps), DT)[0]
    return sum(step * gamma**t for t in range(round(seconds / DT)))


def _crash_after(
    speed_mps: float, seconds: float, reward: LapTimeReward = REWARD, gamma: float = 1.0
) -> float:
    """Return of driving at ``speed_mps`` for ``seconds``, then crashing on the next step."""
    crash = reward(_info(speed_mps=speed_mps, crashed=True), DT)[0]
    return _drive(speed_mps, seconds, reward, gamma) + crash * gamma ** round(seconds / DT)


SPEEDS_MPS = [0.1, 28.0 / 3.6, REWARD.target_speed_mps, 3.0 * REWARD.target_speed_mps]


FLOORS = [pytest.param(LapTimeReward(pace_floor=f), id=f"floor_{f}") for f in (0.0, 0.25)]


@pytest.mark.parametrize("reward", FLOORS)
class TestNoEarlyExitPays:
    @pytest.mark.parametrize("gamma", [1.0, 0.997])
    @pytest.mark.parametrize("speed_mps", SPEEDS_MPS)
    @pytest.mark.parametrize("crash_at_s", [0.02, 1.0, 30.0, 299.0])
    def test_crashing_now_is_worse_than_driving_on_and_crashing_later(
        self, reward, gamma, speed_mps, crash_at_s
    ):
        now = reward(_info(speed_mps=speed_mps, crashed=True), DT)[0]
        assert now < _crash_after(speed_mps, crash_at_s, reward, gamma)

    @pytest.mark.parametrize("gamma", [1.0, 0.997])
    @pytest.mark.parametrize("speed_mps", SPEEDS_MPS)
    def test_stalling_never_beats_driving(self, reward, gamma, speed_mps):
        """Parked until the stall ends the episode, against any forward drive that crashes."""
        stall_s = 2.5
        parked = (
            _drive(0.0, stall_s, reward, gamma)
            + gamma ** round(stall_s / DT) * reward(_info(stalled=True), DT)[0]
        )
        assert parked < _crash_after(speed_mps, stall_s, reward, gamma)

    @pytest.mark.parametrize("crash_at_m", [30.0, 100.0, 300.0, 1000.0])
    def test_slowing_down_to_delay_a_crash_never_pays(self, reward, crash_at_m):
        """With a crash a fixed distance ahead, discounting must not make slower better."""
        speeds = np.linspace(2.0, 80.0, 40)
        returns = [_crash_after(v, crash_at_m / v, reward, 0.997) for v in speeds]
        assert np.all(np.diff(returns) > 0.0)


class TestPaceBeatsDistance:
    def test_crawling_the_whole_horizon_earns_less_than_target_pace_then_a_crash(self):
        crawl = _drive(28.0 / 3.6, 300.0)  # round 2's kerb-hugger, to truncation
        sprint = _crash_after(REWARD.target_speed_mps, 20.0)
        assert crawl < sprint

    def test_a_faster_metre_pays_more_between_the_floor_and_the_cap(self):
        def per_metre(speed_mps: float) -> float:
            return REWARD(_info(speed_mps=speed_mps), DT)[0] / (speed_mps * DT)

        target = REWARD.target_speed_mps
        assert per_metre(0.5 * target) < per_metre(target) < per_metre(2.0 * target)
        assert per_metre(3.0 * target) == pytest.approx(REWARD.pace_cap)
        assert per_metre(0.1 * target) == pytest.approx(REWARD.pace_floor)


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
        assert set(terms) == {"pace", "track_limits", "crash", "lap_bonus"}
        assert sum(terms.values()) == pytest.approx(reward)


class TestScale:
    SCALED = LapTimeReward(scale=0.01)

    def test_it_multiplies_every_term(self):
        info = _info(speed_mps=50.0, depth=0.1, crashed=True, lap_time=110.0)
        reward, terms = REWARD(info, DT)
        scaled_reward, scaled_terms = self.SCALED(info, DT)
        assert scaled_terms == pytest.approx({name: 0.01 * v for name, v in terms.items()})
        assert scaled_reward == pytest.approx(0.01 * reward)

    def test_a_lap_at_target_pace_lands_in_a_range_the_critic_can_fit(self):
        """Unnormalised PPO needs small returns: a target-pace lap should be roughly +60 to
        +150 at the scale the configs use."""
        reward = LapTimeReward(target_lap_time_s=105.0, scale=0.01)
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
