"""How the central complex's neurons split into compass, steering and descending cells.

Torch-free, so code that only reads a brain's activity (the cockpit pose, GH-26) can index it
without importing PyTorch. :class:`~fly_driver.brains.central_complex.CentralComplexBrain`
sizes itself with :func:`split_populations`; its neurons are ordered compass, then steering,
then descending.
"""

from __future__ import annotations

from dataclasses import dataclass

from fly_driver.brains.constants import MIN_COMPASS_COUNT

__all__ = ["PopulationCounts", "default_output_dim", "split_populations"]


@dataclass(frozen=True)
class PopulationCounts:
    """How ``neuron_count`` was split across the three synthetic compartments."""

    compass: int
    steering: int
    descending: int

    @property
    def total(self) -> int:
        """Sum of the three compartments."""
        return self.compass + self.steering + self.descending


def default_output_dim(neuron_count: int) -> int:
    """Descending width when none is given: about a fifth, leaving room for the ring."""
    room = neuron_count - (MIN_COMPASS_COUNT + 1)
    return min(max(4, neuron_count // 5), room)


def split_populations(neuron_count: int, output_dim: int) -> PopulationCounts:
    """Split ``neuron_count`` into a compass ring, steering cells and ``output_dim`` readouts.

    Raises:
        ValueError: If ``output_dim`` leaves too few cells for a ring and a steering cell.
    """
    remaining = neuron_count - output_dim
    if remaining < MIN_COMPASS_COUNT + 1:
        raise ValueError(
            f"output_dim={output_dim} leaves {remaining} neurons for the compass and "
            f"steering populations; neuron_count={neuron_count} needs at least "
            f"{MIN_COMPASS_COUNT + 1} besides the descending readout"
        )
    compass = max(MIN_COMPASS_COUNT, remaining // 5)
    if compass > remaining - 1:
        compass = remaining - 1
    steering = remaining - compass
    return PopulationCounts(compass=compass, steering=steering, descending=output_dim)
