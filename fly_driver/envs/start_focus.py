"""Extra training starts on the stretches of track the policy keeps failing at (GH-103).

Uniform random starts share practice out by track length, whatever a corner costs. Measured
over the runs of 2026-11-03 and 04, 1 of 2,416 arrivals got through T3 and 9 of 6,210 through
T10, and a lap needs both. :class:`StartFocus` starts a share of the episodes inside chosen
zones and leaves the rest uniform.

The uniform share stays exactly the old distribution, so it stays comparable with every run
before this one. That is what ``fly_driver.training.lab`` scores: a focused drive is short by
design, and counting it would make a focused run look like a collapse.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

__all__ = ["StartFocus", "draw_start"]


@dataclass(frozen=True)
class StartFocus:
    """Where the focused share of the episodes starts.

    Args:
        share: Fraction of the episodes that start in a zone. Strictly between 0 and 1: at 1 no
            drive would be left for the lab supervisor to score, and off is ``None``.
        zones: ``(from_m, to_m)`` arclengths, metres from the line, each with ``from_m`` below
            ``to_m`` so a zone cannot wrap round the line. Each zone gets the same number of
            starts, however long it is, so a short approach gets as much practice as a long one.

    Raises:
        ValueError: If any of the above does not hold.
    """

    share: float
    zones: Sequence[Sequence[float]]

    def __post_init__(self) -> None:
        if not 0.0 < self.share < 1.0:
            raise ValueError(f"start_focus.share must be strictly in (0, 1), got {self.share}")
        if not self.zones:
            raise ValueError("start_focus.zones must name at least one zone")
        zones = []
        for zone in self.zones:
            if len(zone) != 2 or not 0.0 <= zone[0] < zone[1]:
                raise ValueError(f"start_focus zones need 0 <= from_m < to_m, got {list(zone)}")
            zones.append((float(zone[0]), float(zone[1])))
        # YAML gives lists. Tuples make two equal focuses compare equal and keep this frozen.
        object.__setattr__(self, "zones", tuple(zones))


def draw_start(
    rng: np.random.Generator, lap_length_m: float, focus: StartFocus | None
) -> tuple[float, bool]:
    """Draw one episode's start.

    Without a focus this makes exactly one draw, so a reset seed gives the start it always
    gave and a run that does not ask for a focus is unchanged to the byte.

    Args:
        rng: The episode's generator.
        lap_length_m: Length of the lap.
        focus: The zones and their share, or ``None`` for uniform starts only.

    Returns:
        The start's arclength, and whether it came from the focused share. A uniform start
        that happens to land in a zone is not focused.
    """
    if focus is not None and float(rng.random()) < focus.share:
        from_m, to_m = focus.zones[int(rng.integers(len(focus.zones)))]
        return float(rng.uniform(from_m, to_m)), True
    return float(rng.uniform(0.0, lap_length_m)), False
