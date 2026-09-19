"""Paying for leaving the circuit in seconds rather than in a restart (GH-16).

The practice track used to throw the lap away the moment 15% of the car's width was past
the kerb. That is unforgiving to drive, and worse, it destroys the data: a terminated
episode teaches a learning agent "something ended" and nothing about how badly. A time
penalty keeps the lap alive and grades the mistake.

**Two terms, because one number cannot say both things.** How far off the car went and how
long it stayed out are different mistakes. A wheel on the grass for two seconds is untidy;
cutting the corner entirely is cheating, even if it takes less time. So:

    penalty = peak_weight * worst_depth  +  time_weight * integral(depth * dt)

``depth`` is :func:`~fly_driver.envs.centerline.off_track_fraction` -- the fraction of the
car's width past the kerb, already measured every step for the old rule -- so this needs no
new geometry. The first term is charged once per excursion and scales with how far off the
car got at its worst; the second accrues for as long as it stays out.

**Charged per excursion, not per lap.** Going off twice costs more than going off once, and
the peak term resets each time the car comes back. A lap with two separate cuts is worse
than a lap with one, which a lap-wide peak would hide.

The defaults land on the two cases that were asked for:

===========================  =======  ========  =========
case                          depth    seconds   penalty
===========================  =======  ========  =========
brushing a kerb                 0.2       0.5      0.53 s
cutting a corner completely     1.2       3.0      7.08 s
never leaving the circuit        --        --      0
===========================  =======  ========  =========

Those are computed from the weights below, against the two cases the rule was asked to
produce. They are not yet a measurement of anything a car did: the depths and durations
are estimates of what a brush and a cut look like, and the weights want re-checking once
someone has actually driven off at both extremes and read the number back.
"""

from __future__ import annotations

from dataclasses import dataclass, field

__all__ = ["DEFAULT_PENALTY_WEIGHTS", "OffTrackPenalty", "PenaltyWeights"]


@dataclass(frozen=True)
class PenaltyWeights:
    """What an excursion costs.

    Args:
        peak_weight: Seconds charged per unit of worst depth, once per excursion. Depth is
            a fraction of the car's width, so 1.0 means the whole car is a full width past
            the kerb.
        time_weight: Seconds charged per unit of depth per second spent out.
        minimum_depth: Depth below which the car counts as on track. Not zero, because
            ``off_track_fraction`` returns tiny positive values when a tyre is a
            millimetre over the line, and charging for those would make every lap dirty.

    Raises:
        ValueError: If a weight is negative, or the threshold is not in ``[0, 1)``.
    """

    peak_weight: float = 2.0
    time_weight: float = 1.3
    minimum_depth: float = 0.02

    def __post_init__(self) -> None:
        for name in ("peak_weight", "time_weight"):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must be non-negative, got {getattr(self, name)}")
        if not 0.0 <= self.minimum_depth < 1.0:
            raise ValueError(f"minimum_depth must be in [0, 1), got {self.minimum_depth}")


#: What an excursion costs unless a caller says otherwise.
DEFAULT_PENALTY_WEIGHTS = PenaltyWeights()


@dataclass
class OffTrackPenalty:
    """Accumulates the seconds a lap has earned by leaving the circuit.

    Args:
        weights: What an excursion costs. Defaults to :data:`DEFAULT_PENALTY_WEIGHTS`.
    """

    weights: PenaltyWeights = field(default_factory=lambda: DEFAULT_PENALTY_WEIGHTS)
    seconds: float = field(default=0.0, init=False)
    excursions: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        self._peak = 0.0
        self._integral = 0.0

    @property
    def is_off_track(self) -> bool:
        """Whether the car is outside the limits right now."""
        return self._peak > 0.0

    @property
    def pending_seconds(self) -> float:
        """What the excursion in progress has earned so far.

        Separate from :attr:`seconds` so a panel can show a penalty growing while the car
        is still off, rather than having it appear from nowhere when the car returns.
        """
        if not self.is_off_track:
            return 0.0
        return self.weights.peak_weight * self._peak + self.weights.time_weight * self._integral

    @property
    def total_seconds(self) -> float:
        """Everything earned this lap, including the excursion in progress."""
        return self.seconds + self.pending_seconds

    def reset(self) -> None:
        """Start a fresh lap with nothing owed."""
        self.seconds = 0.0
        self.excursions = 0
        self._peak = 0.0
        self._integral = 0.0

    def update(self, depth: float, dt: float) -> None:
        """Feed one step.

        Args:
            depth: Fraction of the car's width past the kerb, from
                :func:`~fly_driver.envs.centerline.off_track_fraction`. Zero on track.
            dt: Seconds this step covered.

        Raises:
            ValueError: If ``dt`` is negative.
        """
        if dt < 0:
            raise ValueError(f"dt must be non-negative, got {dt}")
        depth = max(0.0, float(depth))

        if depth > self.weights.minimum_depth:
            if not self.is_off_track:
                self.excursions += 1
            self._peak = max(self._peak, depth)
            self._integral += depth * dt
        elif self.is_off_track:
            self._bank()

    def finish_lap(self) -> float:
        """Close any excursion in progress and return the lap's total penalty.

        A car still off the track as it crosses the line has still earned what it earned;
        leaving that unbanked would let an excursion be laundered by timing it right.
        """
        if self.is_off_track:
            self._bank()
        return self.seconds

    def _bank(self) -> None:
        """Charge for the excursion that just ended and start the next one clean."""
        self.seconds += (
            self.weights.peak_weight * self._peak + self.weights.time_weight * self._integral
        )
        self._peak = 0.0
        self._integral = 0.0
