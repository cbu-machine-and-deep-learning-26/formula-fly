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

**Depth is capped at :attr:`PenaltyWeights.maximum_depth`, and that cap is load-bearing.**
``off_track_fraction`` is unbounded: it is metres past the kerb over the car's width, so
spinning into the infield reads 20 and not 1.2. Measured on a real hand-driven lap, a car
that ran 47 m wide came back with a depth of 20.4, which at the first draft of these
weights was a 147-second penalty for one mistake -- longer than the lap it was added to.
Past roughly a car's width off the circuit, "further off" stops carrying information: the
car is simply off. What still scales is how long it takes to rejoin, and the time term
handles that.

**Charged per excursion, not per lap.** Going off twice costs more than going off once, and
the peak term resets each time the car comes back. A lap with two separate cuts is worse
than a lap with one, which a lap-wide peak would hide.

The defaults land on the two cases that were asked for, plus the one the drive turned up:

===========================  =======  ========  =========
case                          depth    seconds   penalty
===========================  =======  ========  =========
brushing a kerb                 0.2       0.5      0.57 s
cutting a corner completely     1.2       3.0      7.10 s
spinning into the infield      20.4       8.0     15.60 s
never leaving the circuit        --        --      0
===========================  =======  ========  =========

The first two are the cases the rule was asked to produce and the weights are fitted to
them. The third is a real excursion from a hand-driven lap, and it is in the table because
it is what the cap exists for.
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
        maximum_depth: Depth above which going further off stops costing more. Bounds an
            otherwise unbounded measurement; see the module docstring for the lap that
            made it necessary.

    Raises:
        ValueError: If a weight is negative, if the threshold is not in ``[0, 1)``, or if
            the cap is not above the threshold.
    """

    peak_weight: float = 2.0
    time_weight: float = 1.7
    minimum_depth: float = 0.02
    maximum_depth: float = 1.0

    def __post_init__(self) -> None:
        for name in ("peak_weight", "time_weight"):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must be non-negative, got {getattr(self, name)}")
        if not 0.0 <= self.minimum_depth < 1.0:
            raise ValueError(f"minimum_depth must be in [0, 1), got {self.minimum_depth}")
        if self.maximum_depth <= self.minimum_depth:
            raise ValueError(
                f"maximum_depth must be above minimum_depth, got {self.maximum_depth} "
                f"and {self.minimum_depth}"
            )


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
                Values well above 1 are ordinary -- the measurement is unbounded -- and are
                capped at :attr:`PenaltyWeights.maximum_depth` before being charged.
            dt: Seconds this step covered.

        Raises:
            ValueError: If ``dt`` is negative.
        """
        if dt < 0:
            raise ValueError(f"dt must be non-negative, got {dt}")
        # Clamped before the threshold test, not after, so the cap can never make a car
        # that is deep off the circuit read as being on it.
        depth = min(max(0.0, float(depth)), self.weights.maximum_depth)

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
