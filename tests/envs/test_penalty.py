"""The off-track time penalty (GH-16).

Every failure here is a silent one. A penalty that is charged twice, or never banked, or
quietly reset by the next step still produces a plausible-looking lap time -- nothing
raises, and the only symptom is a number that is wrong by a few seconds. So the anchors
the rule was designed around are pinned as numbers, and the accounting is tested for the
ways it can leak: an excursion that is never closed, a peak that carries into the next
one, and a lap that launders a cut by timing it across the line.
"""

from __future__ import annotations

import pytest

from fly_driver.envs.penalty import DEFAULT_PENALTY_WEIGHTS, OffTrackPenalty, PenaltyWeights


def _drive_off(penalty: OffTrackPenalty, depth: float, seconds: float, dt: float = 0.02) -> None:
    """Hold the car at one depth for a while, then bring it back onto the circuit."""
    for _ in range(round(seconds / dt)):
        penalty.update(depth, dt)
    penalty.update(0.0, dt)


class TestTheAnchors:
    """The two cases the weights were chosen to produce.

    These are the rule's definition as far as a driver is concerned, so a weight change
    that moves them should have to come here and say so.
    """

    def test_brushing_a_kerb_costs_about_half_a_second(self):
        penalty = OffTrackPenalty()
        _drive_off(penalty, depth=0.2, seconds=0.5)
        assert penalty.finish_lap() == pytest.approx(0.57, abs=0.02)

    def test_cutting_a_corner_costs_about_seven_seconds(self):
        penalty = OffTrackPenalty()
        _drive_off(penalty, depth=1.2, seconds=3.0)
        assert penalty.finish_lap() == pytest.approx(7.10, abs=0.05)

    def test_a_clean_lap_costs_nothing(self):
        penalty = OffTrackPenalty()
        for _ in range(500):
            penalty.update(0.0, 0.02)
        assert penalty.finish_lap() == 0.0
        assert penalty.excursions == 0

    def test_a_cut_costs_more_than_a_brush(self):
        """The ordering is the point of the rule, and must not depend on the weights
        happening to be the ones above."""
        brush, cut = OffTrackPenalty(), OffTrackPenalty()
        _drive_off(brush, depth=0.2, seconds=0.5)
        _drive_off(cut, depth=1.2, seconds=3.0)
        assert cut.finish_lap() > 5.0 * brush.finish_lap()


class TestTheAccounting:
    def test_each_excursion_is_charged_separately(self):
        """A lap-wide peak would make the second trip off free, so two mistakes would cost
        the same as one."""
        once, twice = OffTrackPenalty(), OffTrackPenalty()
        _drive_off(once, depth=0.5, seconds=1.0)
        _drive_off(twice, depth=0.5, seconds=1.0)
        _drive_off(twice, depth=0.5, seconds=1.0)
        assert twice.finish_lap() == pytest.approx(2.0 * once.finish_lap(), rel=0.05)
        assert twice.excursions == 2

    def test_the_peak_does_not_carry_into_the_next_excursion(self):
        """Otherwise one early cut makes every later brush cost as much as the cut did."""
        deep_then_shallow, shallow = OffTrackPenalty(), OffTrackPenalty()
        _drive_off(deep_then_shallow, depth=1.5, seconds=0.5)
        banked = deep_then_shallow.seconds
        _drive_off(deep_then_shallow, depth=0.1, seconds=0.5)
        _drive_off(shallow, depth=0.1, seconds=0.5)
        assert deep_then_shallow.seconds - banked == pytest.approx(shallow.finish_lap())

    def test_the_worst_moment_is_what_is_charged(self):
        """Depth is sampled per step, so a brief deep moment inside a long shallow
        excursion is still charged at its worst -- capped, since 0.8 is under the ceiling
        and 1.4 would not be."""
        penalty = OffTrackPenalty()
        penalty.update(0.1, 0.02)
        penalty.update(0.8, 0.02)
        penalty.update(0.1, 0.02)
        penalty.update(0.0, 0.02)
        assert penalty.seconds > DEFAULT_PENALTY_WEIGHTS.peak_weight * 0.8

    def test_crossing_the_line_while_still_off_track_does_not_launder_it(self):
        """The reason finish_lap banks rather than discards. Without it, running wide over
        the line is free, which is exactly where a fast lap would learn to do it."""
        penalty = OffTrackPenalty()
        for _ in range(150):
            penalty.update(1.0, 0.02)
        assert penalty.seconds == 0.0, "nothing banked yet, the car is still out"
        assert penalty.finish_lap() > 0.0

    def test_a_penalty_is_visible_while_it_is_being_earned(self):
        """So a panel can show the number climbing, rather than having it appear from
        nowhere when the car rejoins."""
        penalty = OffTrackPenalty()
        penalty.update(0.5, 0.02)
        assert penalty.is_off_track
        assert penalty.pending_seconds > 0.0
        assert penalty.seconds == 0.0
        assert penalty.total_seconds == pytest.approx(penalty.pending_seconds)

    def test_pending_moves_into_the_total_without_changing_it(self):
        penalty = OffTrackPenalty()
        for _ in range(50):
            penalty.update(0.5, 0.02)
        before = penalty.total_seconds
        penalty.update(0.0, 0.02)
        assert not penalty.is_off_track
        assert penalty.total_seconds == pytest.approx(before)

    def test_longer_off_costs_more(self):
        brief, sustained = OffTrackPenalty(), OffTrackPenalty()
        _drive_off(brief, depth=0.5, seconds=0.5)
        _drive_off(sustained, depth=0.5, seconds=2.0)
        assert sustained.finish_lap() > brief.finish_lap()

    def test_deeper_costs_more(self):
        shallow, deep = OffTrackPenalty(), OffTrackPenalty()
        _drive_off(shallow, depth=0.3, seconds=1.0)
        _drive_off(deep, depth=0.9, seconds=1.0)
        assert deep.finish_lap() > shallow.finish_lap()

    def test_reset_starts_the_next_lap_owing_nothing(self):
        penalty = OffTrackPenalty()
        _drive_off(penalty, depth=1.0, seconds=1.0)
        penalty.reset()
        assert penalty.total_seconds == 0.0
        assert penalty.excursions == 0
        assert not penalty.is_off_track

    def test_reset_clears_an_excursion_still_in_progress(self):
        """Resetting mid-excursion happens on every env reset that follows a car being
        stopped off the circuit."""
        penalty = OffTrackPenalty()
        penalty.update(1.0, 0.02)
        penalty.reset()
        assert penalty.finish_lap() == 0.0


class TestTheThreshold:
    def test_a_millimetre_over_the_line_is_not_an_excursion(self):
        """off_track_fraction returns small positive values for a tyre barely across, and
        charging for those would make every lap dirty and every segment ineligible."""
        penalty = OffTrackPenalty()
        for _ in range(100):
            penalty.update(DEFAULT_PENALTY_WEIGHTS.minimum_depth * 0.5, 0.02)
        assert penalty.finish_lap() == 0.0
        assert penalty.excursions == 0

    def test_negative_depth_is_treated_as_on_track(self):
        penalty = OffTrackPenalty()
        penalty.update(-0.5, 0.02)
        assert penalty.finish_lap() == 0.0


class TestValidation:
    def test_time_cannot_run_backwards(self):
        with pytest.raises(ValueError, match="non-negative"):
            OffTrackPenalty().update(0.5, -0.02)

    @pytest.mark.parametrize("name", ["peak_weight", "time_weight"])
    def test_weights_cannot_be_negative(self, name):
        with pytest.raises(ValueError, match="non-negative"):
            PenaltyWeights(**{name: -1.0})

    @pytest.mark.parametrize("value", [-0.1, 1.0, 2.0])
    def test_the_threshold_must_be_a_fraction(self, value):
        with pytest.raises(ValueError, match=r"\[0, 1\)"):
            PenaltyWeights(minimum_depth=value)

    def test_weights_are_configurable(self):
        penalty = OffTrackPenalty(PenaltyWeights(peak_weight=10.0, time_weight=0.0))
        _drive_off(penalty, depth=0.5, seconds=2.0)
        assert penalty.finish_lap() == pytest.approx(5.0), "only the peak term should count"

    def test_the_shared_default_is_not_mutated_by_use(self):
        """OffTrackPenalty holds the module-level default rather than a copy, so the
        weights being frozen is what keeps one env from rewriting another's rules."""
        OffTrackPenalty().update(1.0, 0.02)
        assert DEFAULT_PENALTY_WEIGHTS.peak_weight == 2.0


class TestTheCap:
    """``off_track_fraction`` is unbounded -- metres past the kerb over the car's width --
    so a spin into the infield reads 20, not 1.2.

    Without a ceiling the first draft of these weights charged 147 seconds for one such
    excursion on a real hand-driven lap, which is longer than the lap it was added to. The
    numbers here come from that drive.
    """

    def test_a_spin_into_the_infield_does_not_cost_more_than_the_lap(self):
        penalty = OffTrackPenalty()
        _drive_off(penalty, depth=20.4, seconds=8.0)  # measured: 47 m off at Silverstone
        assert penalty.finish_lap() < 20.0

    def test_beyond_the_cap_going_further_off_costs_nothing_more(self):
        """The cap's actual claim. Past a car's width off, the car is simply off, and how
        much further is no longer information about how bad it was."""
        at_cap, far_beyond = OffTrackPenalty(), OffTrackPenalty()
        _drive_off(at_cap, depth=DEFAULT_PENALTY_WEIGHTS.maximum_depth, seconds=2.0)
        _drive_off(far_beyond, depth=50.0, seconds=2.0)
        assert far_beyond.finish_lap() == pytest.approx(at_cap.finish_lap())

    def test_below_the_cap_depth_still_matters(self):
        """Otherwise the cap would have flattened the whole scale rather than its tail."""
        shallow, deeper = OffTrackPenalty(), OffTrackPenalty()
        _drive_off(shallow, depth=0.25, seconds=1.0)
        _drive_off(deeper, depth=0.75, seconds=1.0)
        assert deeper.finish_lap() > shallow.finish_lap() * 2.0

    def test_time_still_accrues_past_the_cap(self):
        """Once depth stops carrying information, duration is all that is left to tell a
        brief excursion from a long one -- so it must keep counting."""
        brief, sustained = OffTrackPenalty(), OffTrackPenalty()
        _drive_off(brief, depth=30.0, seconds=1.0)
        _drive_off(sustained, depth=30.0, seconds=6.0)
        assert sustained.finish_lap() > brief.finish_lap()

    def test_the_cap_cannot_make_a_deep_excursion_read_as_on_track(self):
        """It is applied before the on-track threshold, not after. A cap below the
        threshold would otherwise make the worst excursions free."""
        penalty = OffTrackPenalty(PenaltyWeights(minimum_depth=0.02, maximum_depth=0.05))
        _drive_off(penalty, depth=50.0, seconds=1.0)
        assert penalty.excursions == 1
        assert penalty.finish_lap() > 0.0

    def test_the_cap_is_configurable(self):
        loose = OffTrackPenalty(PenaltyWeights(maximum_depth=5.0))
        _drive_off(loose, depth=50.0, seconds=1.0)
        tight = OffTrackPenalty()
        _drive_off(tight, depth=50.0, seconds=1.0)
        assert loose.finish_lap() > tight.finish_lap()

    @pytest.mark.parametrize("maximum", [0.02, 0.01, 0.0, -1.0])
    def test_a_cap_at_or_below_the_threshold_is_refused(self, maximum):
        with pytest.raises(ValueError, match="above minimum_depth"):
            PenaltyWeights(minimum_depth=0.02, maximum_depth=maximum)
