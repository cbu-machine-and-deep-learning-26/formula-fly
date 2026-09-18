"""The measurement GH-23 turns on: can a brain keep up with the frame budget?

One environment step is one frame, and at :data:`~fly_driver.interface.FRAME_RATE_HZ`
that is 20 ms of simulated time. A brain in the loop has to simulate 20 ms of *biology*
in the wall-clock time left after the eye and the environment have taken theirs. So the
number that decides the backend is **wall-milliseconds per 20 ms of biology**, and this
module defines it once so both backends report the same thing.

The neuron constants are Shiu et al.'s, with their sources; they are here rather than
in the backends so the two implementations cannot quietly drift into simulating
different animals.
"""

from __future__ import annotations

from dataclasses import dataclass

from fly_driver.interface import FRAME_RATE_HZ

__all__ = [
    "BIOLOGICAL_MS_PER_FRAME",
    "BenchmarkConfig",
    "NeuronParameters",
    "ThroughputResult",
]

#: Milliseconds of biology one environment step has to cover.
BIOLOGICAL_MS_PER_FRAME = 1000.0 / FRAME_RATE_HZ

#: What the eye and the environment already spend of the 20 ms frame, in milliseconds.
#: The eye's figure is published in ``docs/running-the-stacks.md``; the environment's
#: was measured for GH-16. Both are on this machine and both belong in the docs, not
#: buried here -- they are defaults for the budget, not facts about the brain.
_EYE_MS = 7.9
_ENV_MS = 2.2


@dataclass(frozen=True)
class NeuronParameters:
    """Leaky integrate-and-fire constants, from Shiu et al.

    Defaults are the paper's ``default_params``, with its citations. A backend that
    changes one of these is no longer the published model and should say so.

    Args:
        resting_mv: Resting potential, also the reset potential after a spike.
        threshold_mv: Spike threshold.
        membrane_ms: Membrane time constant (Kakaria and de Bivort 2017).
        synapse_ms: Synaptic time constant (Juergensen et al. 2021).
        refractory_ms: Refractory period (Lazar et al. 2021).
        delay_ms: Synaptic transmission delay (Paul et al. 2015).
        synapse_weight_mv: Millivolts per synapse; the model's one free parameter.
    """

    resting_mv: float = -52.0
    threshold_mv: float = -45.0
    membrane_ms: float = 20.0
    synapse_ms: float = 5.0
    refractory_ms: float = 2.2
    delay_ms: float = 1.8
    synapse_weight_mv: float = 0.275

    def __post_init__(self) -> None:
        if self.threshold_mv <= self.resting_mv:
            raise ValueError(
                f"threshold_mv={self.threshold_mv} must be above "
                f"resting_mv={self.resting_mv}, or every neuron spikes forever"
            )
        for name in ("membrane_ms", "synapse_ms", "refractory_ms", "delay_ms"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive, got {getattr(self, name)}")


@dataclass(frozen=True)
class BenchmarkConfig:
    """How to run one throughput measurement.

    Args:
        dt_ms: Integration timestep. The single biggest lever on cost, because the
            work is proportional to steps per frame: at 0.1 ms that is 200 steps for
            every 20 ms of biology. It is not free, though -- the synaptic delay is
            1.8 ms and the refractory period 2.2 ms, so a coarse ``dt`` quantises
            both, and the measured firing rate is the thing to watch.
        biological_ms: How much biology to time. Long enough that per-run overhead
            does not dominate.
        warmup_ms: Simulated before the clock starts, so first-call costs -- code
            generation, allocation, caches -- are not counted as throughput.
        drive_fraction: Fraction of neurons given Poisson input. A silent network is
            the cheapest possible case and flatters every backend, so the default
            drives it.
        drive_rate_hz: Poisson rate, the paper's optogenetic activation default.
        drive_scale: Multiplier on the synaptic weight for driven inputs; the paper's
            ``f_poi``, large enough to make targets actually spike.
        budget_ms: Wall-clock milliseconds per frame the brain may use. Defaults to
            the 20 ms frame less the eye's 7.9 ms and the environment's 2.2 ms.

    Raises:
        ValueError: On a non-positive duration, a ``dt`` that does not divide the
            frame, or a drive fraction outside ``[0, 1]``.
    """

    dt_ms: float = 0.5
    biological_ms: float = 100.0
    warmup_ms: float = 20.0
    drive_fraction: float = 0.01
    drive_rate_hz: float = 150.0
    drive_scale: float = 250.0
    budget_ms: float = BIOLOGICAL_MS_PER_FRAME - _EYE_MS - _ENV_MS

    def __post_init__(self) -> None:
        if self.dt_ms <= 0:
            raise ValueError(f"dt_ms must be positive, got {self.dt_ms}")
        if self.dt_ms > BIOLOGICAL_MS_PER_FRAME:
            raise ValueError(
                f"dt_ms={self.dt_ms} is longer than the {BIOLOGICAL_MS_PER_FRAME} ms "
                "frame; one frame must be at least one integration step"
            )
        if self.biological_ms <= 0:
            raise ValueError(f"biological_ms must be positive, got {self.biological_ms}")
        if self.warmup_ms < 0:
            raise ValueError(f"warmup_ms must be non-negative, got {self.warmup_ms}")
        if not 0.0 <= self.drive_fraction <= 1.0:
            raise ValueError(f"drive_fraction must be in [0, 1], got {self.drive_fraction}")
        if self.drive_rate_hz < 0:
            raise ValueError(f"drive_rate_hz must be non-negative, got {self.drive_rate_hz}")

    @property
    def steps_per_frame(self) -> float:
        """Integration steps needed to cover one environment frame."""
        return BIOLOGICAL_MS_PER_FRAME / self.dt_ms


@dataclass(frozen=True)
class ThroughputResult:
    """What one backend managed on one network.

    Args:
        backend: Which implementation produced this.
        num_neurons: Neurons integrated.
        num_synapses: Synapses integrated.
        dt_ms: Timestep used.
        wall_ms_per_frame: **The answer** -- wall-clock milliseconds to simulate 20 ms
            of biology.
        spike_rate_hz: Mean firing rate over the timed window. Reported beside every
            timing because `AGENTS.md` 11 asks for it: a backend that is fast because
            its neurons stopped spiking, or because they all saturated, would
            otherwise look like the winner.
        budget_ms: The budget it was measured against.
        build_s: Wall-clock seconds to construct the network, excluded from the
            timing. Paid once per episode at most, not per frame.
        notes: Anything that qualifies the number, such as an uncompiled backend.
    """

    backend: str
    num_neurons: int
    num_synapses: int
    dt_ms: float
    wall_ms_per_frame: float
    spike_rate_hz: float
    budget_ms: float
    build_s: float = 0.0
    notes: str = ""

    @property
    def fits_budget(self) -> bool:
        """Whether this network could run inside the frame budget in real time."""
        return self.wall_ms_per_frame <= self.budget_ms

    @property
    def real_time_factor(self) -> float:
        """Frames of biology per frame of wall clock. Above 1.0 is faster than life."""
        return BIOLOGICAL_MS_PER_FRAME / self.wall_ms_per_frame

    def __str__(self) -> str:
        verdict = "fits" if self.fits_budget else "OVER"
        return (
            f"{self.backend:<12} {self.num_neurons:>8,} neurons "
            f"{self.num_synapses:>11,} synapses  dt={self.dt_ms:<5.2f} "
            f"{self.wall_ms_per_frame:>8.1f} ms/frame  {verdict:>4}  "
            f"{self.spike_rate_hz:>6.2f} Hz/neuron"
        )
