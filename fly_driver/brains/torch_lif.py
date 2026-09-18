"""The same leaky integrate-and-fire network, in PyTorch (GH-23).

#23 asks us to "decide Brian2 vs a PyTorch LIF by measuring throughput". This is the
second half of that comparison: identical dynamics, identical constants, identical
network -- both backends take the same
:class:`~fly_driver.brains.connectome.Subnetwork` -- so the only thing the measurement
varies is the implementation.

**The integration is exact, not Euler.** Brian2 solves this model with ``method="linear"``,
which integrates the coupled linear system analytically, so a forward-Euler PyTorch
version would be a different model that happens to be faster. Over one step of length
``h``, with membrane constant ``T`` and synaptic constant ``tau``::

    g <- g * exp(-h / tau)
    v <- v_0 + (v - v_0) * exp(-h / T)
             + g * tau / (tau - T) * (exp(-h / tau) - exp(-h / T))

which is the closed-form solution of ``dv/dt = (v_0 - v + g) / T`` with ``g`` decaying
exponentially. `AGENTS.md` §11 lists LIF numerics as a thing that fails silently and
looks plausible; a fast backend simulating subtly different neurons is the worst
outcome this ticket could produce, so the agreement with Brian2 is a test.

``torch`` is imported lazily. ``import fly_driver.brains`` works without it.
"""

from __future__ import annotations

import math
import time
from typing import Any

import numpy as np

from fly_driver.brains.benchmark import BenchmarkConfig, NeuronParameters, ThroughputResult
from fly_driver.brains.connectome import Subnetwork

__all__ = ["TorchLIF", "TorchNotInstalledError", "measure"]

BACKEND_NAME = "torch"


class TorchNotInstalledError(RuntimeError):
    """Raised when the PyTorch backend is asked for and torch is not installed."""


def _import_torch() -> Any:
    """Import torch, or explain how to get it."""
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - depends on the local install
        raise TorchNotInstalledError(
            "the PyTorch backend needs torch, which is not part of the base install. "
            "It lives in its own environment -- see docs/running-the-stacks.md."
        ) from exc
    return torch


class TorchLIF:
    """A whole-brain LIF network as sparse matrix products.

    Args:
        subnetwork: Neurons and weighted synapses.
        parameters: Neuron constants. Defaults to the paper's.
        config: Supplies ``dt_ms`` and the Poisson drive.
        device: ``"cpu"``, ``"cuda"``, or ``None`` to prefer CUDA when present.
        seed: Seeds the Poisson drive, so a run is reproducible.

    Raises:
        TorchNotInstalledError: If torch is not installed.
    """

    def __init__(
        self,
        subnetwork: Subnetwork,
        parameters: NeuronParameters | None = None,
        config: BenchmarkConfig | None = None,
        device: str | None = None,
        seed: int = 0,
    ) -> None:
        torch = _import_torch()
        self._torch = torch
        self.subnetwork = subnetwork
        self.parameters = parameters or NeuronParameters()
        self.config = config or BenchmarkConfig()
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.generator = torch.Generator(device=self.device).manual_seed(seed)

        step_ms = self.config.dt_ms
        membrane_ms = self.parameters.membrane_ms
        synapse_ms = self.parameters.synapse_ms
        if math.isclose(synapse_ms, membrane_ms):
            raise ValueError(
                "synapse_ms and membrane_ms must differ; the closed-form solution "
                "divides by (tau - T)"
            )

        self._decay_g = math.exp(-step_ms / synapse_ms)
        self._decay_v = math.exp(-step_ms / membrane_ms)
        self._coupling = synapse_ms / (synapse_ms - membrane_ms) * (self._decay_g - self._decay_v)

        # Brian2 rounds a synaptic delay to whole timesteps; match it, or the two
        # backends are running different circuits at coarse dt. 1.8 ms at dt=0.5 is
        # 4 steps in both.
        self.delay_steps = max(1, round(self.parameters.delay_ms / step_ms))
        self.refractory_steps = max(0, round(self.parameters.refractory_ms / step_ms))

        num_neurons = subnetwork.num_neurons
        weights_mv = subnetwork.weight * self.parameters.synapse_weight_mv
        indices = torch.tensor(np.stack([subnetwork.post, subnetwork.pre]), dtype=torch.int64)
        values = torch.tensor(weights_mv, dtype=torch.float32)
        self._weights = torch.sparse_coo_tensor(
            indices, values, (num_neurons, num_neurons), device=self.device
        ).coalesce()

        self._num_driven = int(num_neurons * self.config.drive_fraction)
        self._drive_probability = self.config.drive_rate_hz * step_ms / 1000.0
        self._drive_amplitude = self.parameters.synapse_weight_mv * self.config.drive_scale
        self.reset()

    def reset(self) -> None:
        """Return every neuron to rest and empty the delay line."""
        torch = self._torch
        num_neurons = self.subnetwork.num_neurons
        zeros = torch.zeros(num_neurons, dtype=torch.float32, device=self.device)
        self._v = torch.full_like(zeros, self.parameters.resting_mv)
        self._g = zeros.clone()
        self._refractory = torch.zeros(num_neurons, dtype=torch.int32, device=self.device)
        self._delay_line = [zeros.clone() for _ in range(self.delay_steps)]
        self._delay_index = 0
        self.spike_count = 0

    def step(self) -> Any:
        """Advance one timestep. Returns the boolean spike vector."""
        torch = self._torch
        arrived = self._delay_line[self._delay_index]

        active = self._refractory <= 0
        decayed_g = self._g * self._decay_g
        updated_v = (
            self.parameters.resting_mv
            + (self._v - self.parameters.resting_mv) * self._decay_v
            + self._g * self._coupling
        )
        # `unless refractory` in Brian2 pauses the *differential equations*, holding v
        # and g where they are. It does not gate the synaptic pathway: `on_pre` is a
        # Synapses event and keeps adding to g throughout the refractory period. So
        # arriving input is added unconditionally and only the decay is held. Getting
        # this backwards silently deletes every synapse that lands on a neuron in the
        # 2.2 ms after it fires, which in a busy network is most of them.
        self._g = torch.where(active, decayed_g, self._g) + arrived
        self._v = torch.where(active, updated_v, self._v)
        self._refractory = torch.clamp(self._refractory - 1, min=0)

        spiked = (self._v > self.parameters.threshold_mv) & active
        if bool(spiked.any()):
            self._v = torch.where(
                spiked, torch.full_like(self._v, self.parameters.resting_mv), self._v
            )
            self._g = torch.where(spiked, torch.zeros_like(self._g), self._g)
            self._refractory = torch.where(
                spiked,
                torch.full_like(self._refractory, self.refractory_steps),
                self._refractory,
            )
            self.spike_count += int(spiked.sum())

        # Synaptic transmission, delayed: this step's spikes land delay_steps later.
        transmitted = torch.sparse.mm(self._weights, spiked.to(torch.float32).unsqueeze(1)).squeeze(
            1
        )
        if self._num_driven and self._drive_probability > 0:
            noise = torch.rand(self._num_driven, generator=self.generator, device=self.device)
            transmitted[: self._num_driven] += (noise < self._drive_probability).to(
                torch.float32
            ) * self._drive_amplitude

        self._delay_line[self._delay_index] = transmitted
        self._delay_index = (self._delay_index + 1) % self.delay_steps
        return spiked

    def run(self, biological_ms: float) -> int:
        """Integrate for a stretch of biology. Returns the spikes counted."""
        before = self.spike_count
        for _ in range(int(round(biological_ms / self.config.dt_ms))):
            self.step()
        return self.spike_count - before

    @property
    def membrane_mv(self) -> Any:
        """Membrane potential per neuron, for comparing against another backend."""
        return self._v


def measure(
    subnetwork: Subnetwork,
    parameters: NeuronParameters | None = None,
    config: BenchmarkConfig | None = None,
    device: str | None = None,
) -> ThroughputResult:
    """Time how long PyTorch takes to simulate one frame's worth of biology.

    The warm-up is not timed: the first CUDA call initialises the context, and the
    first sparse product picks a kernel.

    Raises:
        TorchNotInstalledError: If torch is not installed.
    """
    config = config or BenchmarkConfig()
    started = time.perf_counter()
    network = TorchLIF(subnetwork, parameters, config, device=device)
    build_s = time.perf_counter() - started

    if config.warmup_ms > 0:
        network.run(config.warmup_ms)

    torch = network._torch
    if network.device.type == "cuda":
        torch.cuda.synchronize()
    started = time.perf_counter()
    counted = network.run(config.biological_ms)
    if network.device.type == "cuda":
        torch.cuda.synchronize()
    wall_s = time.perf_counter() - started

    neuron_seconds = subnetwork.num_neurons * config.biological_ms / 1000.0
    return ThroughputResult(
        backend=f"{BACKEND_NAME}:{network.device.type}",
        num_neurons=subnetwork.num_neurons,
        num_synapses=subnetwork.num_synapses,
        dt_ms=config.dt_ms,
        wall_ms_per_frame=1000.0 * wall_s / config.biological_ms * 20.0,
        spike_rate_hz=counted / neuron_seconds if neuron_seconds else 0.0,
        budget_ms=config.budget_ms,
        build_s=build_s,
    )
