"""Extra training starts on a stretch of track the policy keeps failing at (GH-103).

Uniform random starts share practice out by track length, whatever a corner costs. Measured
over the runs of 2026-11-03, 95% of the drives that reached Vale ended there, and most drives
never reached it. :class:`StartFocus` starts a share of the episodes inside a chosen zone and
leaves the rest uniform.

The uniform share stays exactly the old distribution, so it stays comparable with every run
before this one. That is what ``fly_driver.training.lab`` scores: a focused drive is short by
design, and counting it would make a focused run look like a collapse.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = ["StartFocus", "draw_start"]


@dataclass(frozen=True)
class StartFocus:
    """Where the focused share of the episodes starts.

    Args:
        share: Fraction of the episodes that start in the zone. Strictly between 0 and 1: at 1
            no drive would be left for the lab supervisor to score, and off is ``None``.
        from_m: Arclength the zone starts at, metres from the line.
        to_m: Arclength the zone ends at. Must be above ``from_m``, so a zone cannot wrap
            round the line.

    Raises:
        ValueError: If any of the above does not hold.
    """

    share: float
    from_m: float
    to_m: float

    def __post_init__(self) -> None:
        if not 0.0 < self.share < 1.0:
            raise ValueError(f"start_focus.share must be strictly in (0, 1), got {self.share}")
        if not 0.0 <= self.from_m < self.to_m:
            raise ValueError(
                f"start_focus needs 0 <= from_m < to_m, got from_m={self.from_m}, to_m={self.to_m}"
            )


def draw_start(
    rng: np.random.Generator, lap_length_m: float, focus: StartFocus | None
) -> tuple[float, bool]:
    """Draw one episode's start.

    Without a focus this makes exactly one draw, so a reset seed gives the start it always
    gave and a run that does not ask for a focus is unchanged to the byte.

    Args:
        rng: The episode's generator.
        lap_length_m: Length of the lap.
        focus: The zone and its share, or ``None`` for uniform starts only.

    Returns:
        The start's arclength, and whether it came from the focused share. A uniform start
        that happens to land in the zone is not focused.
    """
    if focus is not None and float(rng.random()) < focus.share:
        return float(rng.uniform(focus.from_m, focus.to_m)), True
    return float(rng.uniform(0.0, lap_length_m)), False
