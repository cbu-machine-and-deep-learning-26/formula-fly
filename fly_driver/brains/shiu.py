"""The Shiu et al. whole-brain model, in Brian2 (GH-23).

A leaky integrate-and-fire network over the FlyWire connectome, from *"A leaky
integrate-and-fire computational model based on the connectome of the entire adult
Drosophila brain reveals insights into sensorimotor processing"*. The equations,
constants and reset rule below are the published ones, taken verbatim from the
upstream repository's ``model.py`` so that this stays the same animal.

What is *not* upstream is where the network comes from. Upstream reads its files
inside the model constructor; this takes a :class:`~fly_driver.brains.connectome.Subnetwork`
instead, so the Brian2 backend and the PyTorch one can be handed the identical network
and their timings compared honestly.

``brian2`` is imported lazily. ``import fly_driver.brains`` works without it.

Upstream also ships the two manipulations RQ3 will want: neurons are addressed by
FlyWire ``root_id``, and silencing is "set all synapses to and from those neurons to
zero" -- which is the lesion map of #30, already solved.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any

from fly_driver.brains.benchmark import BenchmarkConfig, NeuronParameters, ThroughputResult
from fly_driver.brains.connectome import Subnetwork

if TYPE_CHECKING:  # pragma: no cover - typing only
    pass

__all__ = ["Brian2NotInstalledError", "build_network", "codegen_target", "measure"]

BACKEND_NAME = "brian2"

#: Upstream's equations, unchanged. ``v`` is the membrane potential and ``g`` the
#: synaptic conductance expressed as a voltage; ``rfc`` is per-neuron so the refractory
#: period can be zeroed on neurons receiving Poisson drive, as the paper does.
_EQUATIONS = """
dv/dt = (v_0 - v + g) / t_mbr : volt (unless refractory)
dg/dt = -g / tau               : volt (unless refractory)
rfc                            : second
"""

_THRESHOLD = "v > v_th"
_RESET = "v = v_rst; w = 0; g = 0 * mV"


class Brian2NotInstalledError(RuntimeError):
    """Raised when the Brian2 backend is asked for and Brian2 is not installed."""


def _import_brian2() -> Any:
    """Import Brian2, or explain how to get it."""
    try:
        import brian2
    except ImportError as exc:  # pragma: no cover - depends on the local install
        raise Brian2NotInstalledError(
            "the Brian2 backend needs brian2, which is not part of the base install. "
            "It lives in its own environment -- see docs/running-the-stacks.md."
        ) from exc
    return brian2


def codegen_target() -> str:
    """Which code generation target Brian2 will actually use.

    **Read this before believing any timing.** Brian2's ``codegen.target`` defaults to
    ``auto``, which compiles through Cython when a C++ compiler is present and falls
    back to plain numpy when one is not -- with a warning and nothing else. The
    fallback is several times slower, so a measurement taken without checking reports
    the backend as slow for a reason that has nothing to do with the backend.

    Returns:
        ``"cython"`` when the compiled path is live, ``"numpy"`` when it is not, or
        whatever explicit target has been set in ``brian2.prefs``.
    """
    brian2 = _import_brian2()
    preference = str(brian2.prefs["codegen.target"])
    if preference != "auto":
        return preference
    from brian2.codegen.runtime.cython_rt.cython_rt import CythonCodeObject

    try:
        return "cython" if CythonCodeObject.is_available() else "numpy"
    except Exception:  # pragma: no cover - probing failure means it is unavailable
        return "numpy"


def build_network(
    subnetwork: Subnetwork,
    parameters: NeuronParameters | None = None,
    config: BenchmarkConfig | None = None,
) -> tuple[Any, Any]:
    """Build the Brian2 objects for a subnetwork.

    Args:
        subnetwork: Neurons and weighted synapses.
        parameters: Neuron constants. Defaults to the paper's.
        config: Supplies ``dt_ms`` and the Poisson drive. Defaults to
            :class:`~fly_driver.brains.benchmark.BenchmarkConfig`.

    Returns:
        ``(network, spike_monitor)``, ready to run.

    Raises:
        Brian2NotInstalledError: If Brian2 is not installed.
    """
    brian2 = _import_brian2()
    parameters = parameters or NeuronParameters()
    config = config or BenchmarkConfig()

    millivolt, millisecond, hertz = brian2.mV, brian2.ms, brian2.Hz
    brian2.defaultclock.dt = config.dt_ms * millisecond

    namespace = {
        "v_0": parameters.resting_mv * millivolt,
        "v_rst": parameters.resting_mv * millivolt,
        "v_th": parameters.threshold_mv * millivolt,
        "t_mbr": parameters.membrane_ms * millisecond,
        "tau": parameters.synapse_ms * millisecond,
    }

    neurons = brian2.NeuronGroup(
        N=subnetwork.num_neurons,
        model=_EQUATIONS,
        method="linear",
        threshold=_THRESHOLD,
        reset=_RESET,
        refractory="rfc",
        namespace=namespace,
        name="brain_neurons",
    )
    neurons.v = namespace["v_0"]
    neurons.g = 0
    neurons.rfc = parameters.refractory_ms * millisecond

    synapses = brian2.Synapses(
        neurons,
        neurons,
        "w : volt",
        on_pre="g += w",
        delay=parameters.delay_ms * millisecond,
        name="brain_synapses",
    )
    objects = [neurons]
    # Brian2 refuses to run a Synapses object that was never connected, and a network
    # with no synapses is a legitimate control condition -- the disconnected baseline
    # RQ1 wants. So leave it out entirely rather than adding an inert one.
    if subnetwork.num_synapses:
        synapses.connect(i=subnetwork.pre, j=subnetwork.post)
        synapses.w = subnetwork.weight * parameters.synapse_weight_mv * millivolt
        objects.append(synapses)

    num_driven = int(subnetwork.num_neurons * config.drive_fraction)
    if num_driven >= 1 and config.drive_rate_hz > 0:
        # The paper removes the refractory period on driven neurons so the Poisson
        # input is not silently rate-limited by it.
        neurons.rfc[:num_driven] = 0 * millisecond
        objects.append(
            brian2.PoissonInput(
                neurons[:num_driven],
                "g",
                1,
                config.drive_rate_hz * hertz,
                parameters.synapse_weight_mv * config.drive_scale * millivolt,
            )
        )

    spikes = brian2.SpikeMonitor(neurons)
    objects.append(spikes)
    return brian2.Network(*objects), spikes


def measure(
    subnetwork: Subnetwork,
    parameters: NeuronParameters | None = None,
    config: BenchmarkConfig | None = None,
) -> ThroughputResult:
    """Time how long Brian2 takes to simulate one frame's worth of biology.

    The warm-up run is not timed: Brian2 generates and caches code on the first run,
    and counting that as throughput would make short measurements look far worse than
    a real episode.

    Args:
        subnetwork: The network to integrate.
        parameters: Neuron constants. Defaults to the paper's.
        config: Timestep, durations and drive. Defaults to the standard benchmark.

    Raises:
        Brian2NotInstalledError: If Brian2 is not installed.
    """
    brian2 = _import_brian2()
    config = config or BenchmarkConfig()
    millisecond = brian2.ms

    started = time.perf_counter()
    network, spikes = build_network(subnetwork, parameters, config)
    build_s = time.perf_counter() - started

    if config.warmup_ms > 0:
        network.run(config.warmup_ms * millisecond)
    spikes_before = int(spikes.num_spikes)

    started = time.perf_counter()
    network.run(config.biological_ms * millisecond)
    wall_s = time.perf_counter() - started

    counted = int(spikes.num_spikes) - spikes_before
    neuron_seconds = subnetwork.num_neurons * config.biological_ms / 1000.0
    target = codegen_target()
    return ThroughputResult(
        backend=BACKEND_NAME,
        num_neurons=subnetwork.num_neurons,
        num_synapses=subnetwork.num_synapses,
        dt_ms=config.dt_ms,
        wall_ms_per_frame=1000.0 * wall_s / config.biological_ms * 20.0,
        spike_rate_hz=counted / neuron_seconds if neuron_seconds else 0.0,
        budget_ms=config.budget_ms,
        build_s=build_s,
        notes=(
            "uncompiled numpy target: a floor, not Brian2's best"
            if target == "numpy"
            else f"{target} target"
        ),
    )
