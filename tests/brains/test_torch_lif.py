"""The PyTorch LIF, and whether it is the same animal as Brian2 (GH-23).

`AGENTS.md` §11 names LIF numerics as a thing that fails silently and looks plausible.
The risk this file exists to cover is specific: #23 chooses a backend on throughput, so
a PyTorch implementation that is fast *because* it quietly simulates something simpler
would win the comparison and lose the science. The dynamics are pinned against closed
form, and the whole network is pinned against Brian2.

Both backends are optional and live in their own environment, so everything here skips
cleanly on the base install (``docs/running-the-stacks.md``).
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from fly_driver.brains.benchmark import BenchmarkConfig, NeuronParameters
from fly_driver.brains.connectome import subnetwork_from_arrays

torch = pytest.importorskip("torch")

from fly_driver.brains.torch_lif import (  # noqa: E402
    TorchLIF,
    describe_device,
    resolve_device,
)

PARAMETERS = NeuronParameters()


def _pair(weight: float, device: str = "cpu", dt_ms: float = 0.5) -> TorchLIF:
    """Two neurons, one synapse, no Poisson drive."""
    network = subnetwork_from_arrays([0, 1], [0], [1], [weight])
    config = BenchmarkConfig(dt_ms=dt_ms, drive_fraction=0.0)
    return TorchLIF(network, PARAMETERS, config, device=device, seed=0)


def _fire_first_neuron(model: TorchLIF) -> None:
    """Push neuron 0 over threshold and step once, leaving the rest at rest.

    Only neuron 0: driving the whole group makes the postsynaptic neuron fire too,
    and a refractory target is a different test than the one intended.
    """
    model._v[0] = PARAMETERS.threshold_mv + 1.0
    spiked = model.step()
    assert bool(spiked[0]) and not bool(spiked[1])


class TestTheDynamicsAreTheOnesOnPaper:
    def test_an_undisturbed_neuron_sits_exactly_at_rest(self, device):
        """No input means no drift. A leak implemented against the wrong reference
        would show up here as a slow slide and nowhere else."""
        model = _pair(0.0, device)
        model.run(200.0)
        assert model.membrane_mv.numpy() == pytest.approx(PARAMETERS.resting_mv)

    def test_the_membrane_follows_the_closed_form(self, device):
        """Brian2 integrates this model exactly (`method="linear"`). Forward Euler
        would be a different model that happens to be faster, so the step is checked
        against the analytic solution rather than against itself."""
        model = _pair(0.0, device)
        start_mv = PARAMETERS.resting_mv + 3.0
        conductance = 1.5
        model._v = torch.full_like(model._v, start_mv)
        model._g = torch.full_like(model._g, conductance)
        model.step()

        step_ms = model.config.dt_ms
        membrane, synapse = PARAMETERS.membrane_ms, PARAMETERS.synapse_ms
        decay_v = math.exp(-step_ms / membrane)
        decay_g = math.exp(-step_ms / synapse)
        expected = (
            PARAMETERS.resting_mv
            + (start_mv - PARAMETERS.resting_mv) * decay_v
            + conductance * synapse / (synapse - membrane) * (decay_g - decay_v)
        )
        assert float(model.membrane_mv[0]) == pytest.approx(expected, rel=1e-5)

    def test_it_spikes_resets_and_then_holds_through_the_refractory(self, device):
        model = _pair(0.0, device)
        _fire_first_neuron(model)
        assert float(model.membrane_mv[0]) == pytest.approx(PARAMETERS.resting_mv)
        # Held, not merely prevented from spiking: pushing it over threshold during
        # the refractory period must do nothing at all.
        model._v[0] = PARAMETERS.threshold_mv + 1.0
        assert not bool(model.step()[0])

    def test_a_synapse_takes_the_delay_to_arrive(self, device):
        """1.8 ms is not instant and is not one step. Losing the delay line would
        make the network faster and differently behaved."""
        model = _pair(50.0, device)
        _fire_first_neuron(model)
        for _ in range(model.delay_steps - 1):
            assert float(model._g[1]) == pytest.approx(0.0), "arrived early"
            model.step()
        model.step()
        assert float(model._g[1]) > 0.0, "never arrived"

    def test_the_delay_is_rounded_to_whole_steps_as_brian2_does(self, device):
        assert _pair(0.0, device, dt_ms=0.5).delay_steps == 4  # 1.8 ms / 0.5 ms -> 4
        assert _pair(0.0, device, dt_ms=0.1).delay_steps == 18

    def test_an_inhibitory_synapse_lowers_the_target(self, device):
        """The connectome's weights are signed. An absolute value anywhere would make
        every inhibitory synapse excitatory, and the network would still run."""
        model = _pair(-50.0, device)
        _fire_first_neuron(model)
        for _ in range(model.delay_steps):
            model.step()
        assert float(model._g[1]) < 0.0

    def test_a_matching_membrane_and_synapse_constant_is_refused(self):
        """The closed form divides by (tau - T). Returning NaN here would be a silent
        all-zero network."""
        network = subnetwork_from_arrays([0], [], [], [])
        with pytest.raises(ValueError, match="must differ"):
            TorchLIF(network, NeuronParameters(synapse_ms=20.0, membrane_ms=20.0))


def _random_network(num_neurons: int, num_synapses: int, seed: int = 0):
    generator = np.random.default_rng(seed)
    return subnetwork_from_arrays(
        root_ids=np.arange(num_neurons),
        pre=generator.integers(0, num_neurons, num_synapses),
        post=generator.integers(0, num_neurons, num_synapses),
        weight=generator.normal(0.0, 4.0, num_synapses),
    )


class TestStability:
    """`AGENTS.md` §11: no NaN or exploding activity over long rollouts, rates in a
    plausible range."""

    def test_a_long_rollout_stays_finite_and_plausible(self, device):
        network = _random_network(500, 5_000)
        model = TorchLIF(network, PARAMETERS, BenchmarkConfig(dt_ms=0.5), device=device)
        spikes = model.run(2_000.0)  # two seconds of biology
        membrane = model.membrane_mv.numpy()
        assert np.all(np.isfinite(membrane)), "membrane potential went non-finite"
        assert membrane.min() > -500.0, "runaway inhibition"
        assert membrane.max() <= PARAMETERS.threshold_mv + 1e-3, "above threshold"
        rate = spikes / (network.num_neurons * 2.0)
        assert 0.0 < rate < 200.0, f"implausible firing rate {rate:.1f} Hz"

    def test_the_same_seed_gives_the_same_spikes(self, device):
        """The Poisson drive is the only stochastic part, and the eval protocol has
        to be reproducible (§11)."""
        network = _random_network(300, 2_000)
        config = BenchmarkConfig(dt_ms=0.5)
        counts = [
            TorchLIF(network, PARAMETERS, config, device=device, seed=3).run(200.0)
            for _ in range(2)
        ]
        assert counts[0] == counts[1]

    def test_reset_puts_it_back_where_it_started(self, device):
        network = _random_network(200, 1_000)
        model = TorchLIF(network, PARAMETERS, BenchmarkConfig(dt_ms=0.5), device=device)
        model.run(100.0)
        model.reset()
        assert model.spike_count == 0
        assert model.membrane_mv.numpy() == pytest.approx(PARAMETERS.resting_mv)


class TestItAgreesWithBrian2:
    """The comparison #23 turns on. A faster backend is only an alternative if it is
    running the same model."""

    @pytest.fixture(scope="class")
    def brian2(self):
        return pytest.importorskip("brian2")

    def test_an_isolated_neuron_decays_identically(self, brian2, device):
        from fly_driver.brains.shiu import build_network

        network = subnetwork_from_arrays([0], [], [], [])
        config = BenchmarkConfig(dt_ms=0.5, drive_fraction=0.0)
        start_mv = PARAMETERS.resting_mv + 5.0

        simulation, _ = build_network(network, PARAMETERS, config)
        neurons = simulation["brain_neurons"]
        neurons.v = start_mv * brian2.mV
        simulation.run(20 * brian2.ms)
        reference_mv = float(neurons.v[0] / brian2.mV)

        model = TorchLIF(network, PARAMETERS, config, device=device)
        model._v = torch.full_like(model._v, start_mv)
        model.run(20.0)
        assert float(model.membrane_mv[0]) == pytest.approx(reference_mv, abs=1e-3)

    def test_the_two_backends_fire_at_the_same_rate(self, brian2, device):
        """Spike-for-spike equality is not the bar -- the Poisson drives are
        different generators -- but the rates must match, or the dynamics differ."""
        from fly_driver.brains import shiu

        network = _random_network(400, 4_000, seed=7)
        config = BenchmarkConfig(dt_ms=0.5, biological_ms=500.0, warmup_ms=50.0)

        reference = shiu.measure(network, PARAMETERS, config)
        candidate = TorchLIF(network, PARAMETERS, config, device=device)
        candidate.run(config.warmup_ms)
        spikes = candidate.run(config.biological_ms)
        rate = spikes / (network.num_neurons * config.biological_ms / 1000.0)

        assert rate == pytest.approx(reference.spike_rate_hz, rel=0.25), (
            f"torch {rate:.2f} Hz vs brian2 {reference.spike_rate_hz:.2f} Hz"
        )


class TestDeviceSelection:
    """Runs everywhere, including the CPU-only box this was written on."""

    def test_no_device_asked_for_picks_one(self):
        resolved = resolve_device(None)
        assert resolved.type in ("cpu", "cuda")

    def test_asking_for_cuda_without_cuda_raises_rather_than_using_the_cpu(self):
        """The Brian2 codegen fallback already taught this lesson on this ticket: a run
        that silently uses different hardware than it was told to succeeds, produces a
        plausible number, and is measuring the wrong machine."""
        if torch.cuda.is_available():
            pytest.skip("CUDA is available, so the refusal cannot be provoked")
        with pytest.raises(RuntimeError, match="cuda"):
            resolve_device("cuda")

    def test_the_cpu_is_always_nameable(self):
        assert describe_device("cpu") == "cpu"

    def test_a_cuda_device_names_its_card(self, cuda_device):
        """So a recorded timing can never be misattributed to the wrong machine."""
        described = describe_device(cuda_device)
        assert described.startswith("cuda:")
        assert len(described) > len("cuda:0 ")


class TestOnCuda:
    """The first CUDA execution of this code will be on someone else's machine.

    These are the tests that have to catch a device or dtype mistake on that first run,
    because nobody got to make it earlier.
    """

    def test_the_gpu_computes_what_the_cpu_computes(self, cuda_device):
        """Same seed, same network, drive off, both devices: identical spikes.

        The drive is off deliberately. CPU and CUDA generators produce different random
        streams from the same seed, so a driven comparison would fail for a reason that
        is not a bug and would teach us to ignore this test.
        """
        network = _random_network(400, 3_000, seed=11)
        config = BenchmarkConfig(dt_ms=0.5, drive_fraction=0.0)

        on_cpu = TorchLIF(network, PARAMETERS, config, device="cpu", seed=5)
        on_gpu = TorchLIF(network, PARAMETERS, config, device=cuda_device, seed=5)
        # Identical starting state, or the two are simply different experiments.
        start = PARAMETERS.resting_mv + np.linspace(0.0, 6.0, network.num_neurons)
        on_cpu._v = torch.tensor(start, dtype=torch.float32)
        on_gpu._v = torch.tensor(start, dtype=torch.float32, device=on_gpu.device)

        cpu_spikes = on_cpu.run(200.0)
        gpu_spikes = on_gpu.run(200.0)
        assert gpu_spikes == cpu_spikes, f"cuda {gpu_spikes} spikes vs cpu {cpu_spikes}"

        difference = (on_gpu.membrane_mv.cpu() - on_cpu.membrane_mv).abs().max()
        assert float(difference) < 1e-2, (
            f"membrane potentials diverged by {float(difference):.4f} mV. Both are "
            "float32, so a small gap is arithmetic ordering; a large one is a bug."
        )

    def test_the_step_loop_never_touches_the_host(self, cuda_device):
        """The performance bug this backend was rewritten to remove, pinned.

        A single `.item()` or `bool(...)` inside `step()` stalls the pipeline forty times
        a frame and makes a 4090 look slower than a laptop -- while still producing
        correct spikes, so nothing else would catch it.
        """
        network = _random_network(300, 2_000, seed=3)
        model = TorchLIF(network, PARAMETERS, BenchmarkConfig(dt_ms=0.5), device=cuda_device)
        model.run(20.0)  # warm up outside the strict window
        torch.cuda.set_sync_debug_mode("error")
        try:
            for _ in range(40):
                model.step()
        except RuntimeError as exc:  # pragma: no cover - only on a regression
            pytest.fail(f"step() synchronised with the host: {exc}")
        finally:
            torch.cuda.set_sync_debug_mode("default")

    def test_the_weights_are_csr_on_the_device(self, cuda_device):
        """COO has to sort or hash coordinates on every product; CSR is what cuSPARSE
        wants for a matrix multiplied forty times a frame forever."""
        model = TorchLIF(_random_network(200, 900), PARAMETERS, device=cuda_device)
        assert model._weights.layout == torch.sparse_csr
        assert model._weights.device.type == "cuda"

    def test_every_tensor_lives_on_the_device(self, cuda_device):
        """One tensor left on the CPU makes every step a round trip."""
        model = TorchLIF(_random_network(200, 900), PARAMETERS, device=cuda_device)
        model.step()
        for name in ("_v", "_g", "_refractory", "_spike_total", "_resting", "_zeros"):
            assert getattr(model, name).device.type == "cuda", f"{name} is not on the GPU"
