"""Running a brain one frame at a time, so something can watch it (GH-23).

:mod:`fly_driver.brains.shiu` and :mod:`fly_driver.brains.torch_lif` answer "how long
does a frame take" and then throw the spikes away. This runs the same networks in
20 ms slices -- one environment frame each, the way the loop will -- and hands back
what happened, so a viewer can draw it while it runs.

Both backends look the same from out here. That is the point: the comparison in
``docs/running-the-stacks.md`` is only honest because both are handed the identical
:class:`~fly_driver.brains.connectome.Subnetwork` and asked for the identical work,
and a viewer showing one against the other has to preserve that.

Nothing here imports a simulator at module scope.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np
import numpy.typing as npt

from fly_driver.brains.benchmark import (
    BIOLOGICAL_MS_PER_FRAME,
    BenchmarkConfig,
    NeuronParameters,
)
from fly_driver.brains.connectome import Subnetwork

if TYPE_CHECKING:  # pragma: no cover - typing only
    from fly_driver.brains.benchmark import ThroughputResult

__all__ = ["BACKENDS", "FrameActivity", "SpikeStream", "measure_in_loop"]

#: Backends a stream can run, by the name a caller passes.
BACKENDS = ("brian2", "torch")


@dataclass(frozen=True)
class FrameActivity:
    """What one 20 ms frame of biology produced.

    Args:
        neuron_indices: Which neuron each spike came from.
        spike_times_ms: When each spike happened, in simulation time.
        wall_ms: Wall-clock milliseconds this frame cost. The live version of the
            benchmark's headline number.
        sim_time_ms: Simulation time at the end of the frame.
        num_neurons: Size of the network, so a rate can be derived without it.
    """

    neuron_indices: npt.NDArray[np.int64]
    spike_times_ms: npt.NDArray[np.float64]
    wall_ms: float
    sim_time_ms: float
    num_neurons: int

    @property
    def num_spikes(self) -> int:
        """Spikes in this frame."""
        return int(len(self.neuron_indices))

    @property
    def rate_hz(self) -> float:
        """Mean firing rate across the network over this frame."""
        seconds = BIOLOGICAL_MS_PER_FRAME / 1000.0
        return self.num_spikes / (self.num_neurons * seconds)


@dataclass
class SpikeStream:
    """A brain advanced one environment frame at a time.

    Args:
        subnetwork: The network to run.
        backend: ``"brian2"`` or ``"torch"``.
        parameters: Neuron constants. Defaults to the paper's.
        config: Timestep and drive. Defaults to the standard benchmark settings.
        device: Torch device; ignored by the Brian2 backend.

    Raises:
        ValueError: On an unknown backend.
    """

    subnetwork: Subnetwork
    backend: str = "brian2"
    parameters: NeuronParameters | None = None
    config: BenchmarkConfig | None = None
    device: str | None = None
    sim_time_ms: float = field(default=0.0, init=False)
    frames: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        if self.backend not in BACKENDS:
            raise ValueError(
                f"unknown backend {self.backend!r}; known backends are {', '.join(BACKENDS)}"
            )
        self.parameters = self.parameters or NeuronParameters()
        self.config = self.config or BenchmarkConfig()
        self._network: Any = None
        self._monitor: Any = None
        self._model: Any = None
        self._cursor = 0
        self._build()

    def _build(self) -> None:
        if self.backend == "brian2":
            from fly_driver.brains.shiu import build_network

            self._network, self._monitor = build_network(
                self.subnetwork, self.parameters, self.config
            )
        else:
            from fly_driver.brains.torch_lif import TorchLIF

            self._model = TorchLIF(
                self.subnetwork, self.parameters, self.config, device=self.device
            )

    @property
    def num_neurons(self) -> int:
        """Neurons in the network being streamed."""
        return self.subnetwork.num_neurons

    @property
    def label(self) -> str:
        """A short name for this stream, for a legend or a title."""
        if self.backend == "torch":
            return f"torch:{self._model.device.type}"
        return "brian2"

    def warm_up(self) -> None:
        """Run the configured warm-up without timing it.

        Worth doing before anything is drawn: the first frame pays for code
        generation and allocation, and it would otherwise appear as a spike in the
        cost panel that means nothing about the backend.
        """
        assert self.config is not None
        if self.config.warmup_ms <= 0:
            return
        if self.backend == "brian2":
            import brian2

            self._network.run(self.config.warmup_ms * brian2.ms)
            self._cursor = int(self._monitor.num_spikes)
        else:
            self._model.run(self.config.warmup_ms)
        self.sim_time_ms += self.config.warmup_ms

    def advance(self) -> FrameActivity:
        """Run one 20 ms frame and return what it produced."""
        started = time.perf_counter()
        if self.backend == "brian2":
            indices, times = self._advance_brian2()
        else:
            indices, times = self._advance_torch()
        wall_ms = 1000.0 * (time.perf_counter() - started)

        self.sim_time_ms += BIOLOGICAL_MS_PER_FRAME
        self.frames += 1
        return FrameActivity(
            neuron_indices=indices,
            spike_times_ms=times,
            wall_ms=wall_ms,
            sim_time_ms=self.sim_time_ms,
            num_neurons=self.num_neurons,
        )

    def _advance_brian2(self) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.float64]]:
        import brian2

        self._network.run(BIOLOGICAL_MS_PER_FRAME * brian2.ms)
        # The monitor accumulates for the whole session, so take only what is new.
        total = int(self._monitor.num_spikes)
        indices = np.asarray(self._monitor.i[self._cursor : total], dtype=np.int64)
        times = np.asarray(self._monitor.t[self._cursor : total] / brian2.ms, dtype=np.float64)
        self._cursor = total
        return indices, times

    def _advance_torch(self) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.float64]]:
        assert self.config is not None
        step_ms = self.config.dt_ms
        steps = int(round(BIOLOGICAL_MS_PER_FRAME / step_ms))
        indices: list[npt.NDArray[np.int64]] = []
        times: list[npt.NDArray[np.float64]] = []
        for step in range(steps):
            spiked = self._model.step()
            fired = spiked.nonzero().flatten().cpu().numpy().astype(np.int64)
            if fired.size:
                indices.append(fired)
                times.append(np.full(fired.size, self.sim_time_ms + step * step_ms, dtype=float))
        if not indices:
            return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float64)
        return np.concatenate(indices), np.concatenate(times)


def measure_in_loop(
    subnetwork: Subnetwork,
    backend: str = "brian2",
    parameters: NeuronParameters | None = None,
    config: BenchmarkConfig | None = None,
    device: str | None = None,
) -> ThroughputResult:
    """Time a brain the way the loop will actually drive it: 20 ms at a time.

    This is not the same measurement as :func:`fly_driver.brains.shiu.measure`, and the
    difference is the whole point. That one asks Brian2 for 100 ms of biology in a
    single ``run()`` call and divides. A brain between an eye and a policy cannot do
    that -- it has to advance one environment frame, hand its activity over, and be
    advanced again -- so it pays whatever a backend charges *per call*.

    Brian2 charges a lot: a fixed 33-52 ms of setup per ``run()``, independent of
    network size, because it re-prepares the network every call. Measured on 3,000
    neurons that is 4.5 ms per frame batched against 54.7 ms stepped. PyTorch charges
    nothing, because a step is just tensor operations. Quoting the batched number for
    an in-loop brain overstates it by an order of magnitude, which is exactly the kind
    of plausible-looking wrong number `AGENTS.md` §11 exists to catch.

    Args:
        subnetwork: The network to run.
        backend: ``"brian2"`` or ``"torch"``.
        parameters: Neuron constants. Defaults to the paper's.
        config: Timestep, durations and drive.
        device: Torch device; ignored by the Brian2 backend.

    Returns:
        A :class:`~fly_driver.brains.benchmark.ThroughputResult` whose
        ``wall_ms_per_frame`` is the cost of one frame *as the loop will pay it*.
    """
    from fly_driver.brains.benchmark import ThroughputResult

    config = config or BenchmarkConfig()
    started = time.perf_counter()
    stream = SpikeStream(subnetwork, backend, parameters, config, device)
    build_s = time.perf_counter() - started
    stream.warm_up()

    frames = max(1, int(round(config.biological_ms / BIOLOGICAL_MS_PER_FRAME)))
    costs: list[float] = []
    spikes = 0
    for _ in range(frames):
        activity = stream.advance()
        costs.append(activity.wall_ms)
        spikes += activity.num_spikes

    neuron_seconds = subnetwork.num_neurons * frames * BIOLOGICAL_MS_PER_FRAME / 1000.0
    note = "stepped one frame at a time, as the loop will"
    if backend == "brian2":
        from fly_driver.brains.shiu import codegen_target

        if codegen_target() == "numpy":
            note += "; uncompiled numpy target"
    return ThroughputResult(
        backend=stream.label,
        num_neurons=subnetwork.num_neurons,
        num_synapses=subnetwork.num_synapses,
        dt_ms=config.dt_ms,
        wall_ms_per_frame=float(np.median(costs)),
        spike_rate_hz=spikes / neuron_seconds if neuron_seconds else 0.0,
        budget_ms=config.budget_ms,
        build_s=build_s,
        notes=note,
    )
