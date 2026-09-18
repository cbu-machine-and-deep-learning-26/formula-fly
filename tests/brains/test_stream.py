"""Frame-at-a-time streaming, and the measurement that corrected the decision (GH-23).

The point of this module is that a brain in the loop is driven 20 ms at a time, and
that this costs differently from asking a simulator for a second of biology in one
call. Brian2 charges a fixed 33-52 ms of setup per ``run()``; measured batched it looks
ten times cheaper than it is. These tests pin the shape of the measurement so the
distinction cannot quietly collapse back into one number.
"""

from __future__ import annotations

import numpy as np
import pytest

from fly_driver.brains.benchmark import BIOLOGICAL_MS_PER_FRAME, BenchmarkConfig
from fly_driver.brains.connectome import subnetwork_from_arrays
from fly_driver.brains.stream import BACKENDS, FrameActivity, SpikeStream, measure_in_loop

torch = pytest.importorskip("torch")


def _network(num_neurons: int = 200, num_synapses: int = 1_500, seed: int = 0):
    generator = np.random.default_rng(seed)
    return subnetwork_from_arrays(
        root_ids=np.arange(num_neurons),
        pre=generator.integers(0, num_neurons, num_synapses),
        post=generator.integers(0, num_neurons, num_synapses),
        weight=generator.normal(0.0, 4.0, num_synapses),
    )


class TestFrameActivity:
    def test_a_rate_is_derived_from_the_frame_not_guessed(self):
        activity = FrameActivity(
            neuron_indices=np.arange(50),
            spike_times_ms=np.zeros(50),
            wall_ms=1.0,
            sim_time_ms=20.0,
            num_neurons=500,
        )
        assert activity.num_spikes == 50
        # 50 spikes from 500 neurons in 20 ms is 5 Hz each.
        assert activity.rate_hz == pytest.approx(5.0)

    def test_a_silent_frame_is_zero_and_not_a_division_error(self):
        empty = np.empty(0)
        activity = FrameActivity(
            neuron_indices=empty.astype(np.int64),
            spike_times_ms=empty,
            wall_ms=1.0,
            sim_time_ms=20.0,
            num_neurons=10,
        )
        assert activity.rate_hz == 0.0


class TestStreaming:
    def test_an_unknown_backend_lists_the_known_ones(self):
        with pytest.raises(ValueError, match="brian2"):
            SpikeStream(_network(), backend="neuromorphic-toaster")

    @pytest.mark.parametrize("backend", ["torch"])
    def test_each_frame_advances_exactly_one_frame_of_biology(self, backend):
        """The stream is the loop's clock. If a frame is not 20 ms the eye beside it
        integrates at a dt it was never fitted at, and nothing downstream can tell."""
        stream = SpikeStream(_network(), backend, config=BenchmarkConfig(dt_ms=0.5))
        before = stream.sim_time_ms
        stream.advance()
        assert stream.sim_time_ms - before == pytest.approx(BIOLOGICAL_MS_PER_FRAME)
        assert stream.frames == 1

    def test_spikes_are_reported_once_and_not_re_reported(self):
        """The Brian2 monitor accumulates for the session, so a stream that forgot its
        cursor would re-report every earlier spike and the rate would climb forever."""
        stream = SpikeStream(_network(), "torch", config=BenchmarkConfig(dt_ms=0.5))
        stream.warm_up()
        rates = [stream.advance().rate_hz for _ in range(6)]
        assert max(rates) < 200.0, f"rates ran away: {rates}"

    def test_spike_times_fall_inside_the_frame_they_came_from(self):
        stream = SpikeStream(_network(), "torch", config=BenchmarkConfig(dt_ms=0.5))
        start = stream.sim_time_ms
        activity = stream.advance()
        if activity.num_spikes:
            assert activity.spike_times_ms.min() >= start - 1e-9
            assert activity.spike_times_ms.max() < start + BIOLOGICAL_MS_PER_FRAME

    def test_indices_stay_inside_the_network(self):
        stream = SpikeStream(_network(), "torch", config=BenchmarkConfig(dt_ms=0.5))
        for _ in range(4):
            activity = stream.advance()
            if activity.num_spikes:
                assert activity.neuron_indices.max() < stream.num_neurons
                assert activity.neuron_indices.min() >= 0

    def test_the_label_names_the_device_for_torch(self):
        assert SpikeStream(_network(), "torch", device="cpu").label == "torch:cpu"

    def test_backends_is_the_list_the_error_message_promises(self):
        assert set(BACKENDS) == {"brian2", "torch"}


class TestMeasureInLoop:
    def test_it_reports_a_per_frame_cost_and_says_how_it_was_measured(self):
        config = BenchmarkConfig(dt_ms=0.5, biological_ms=100.0)
        result = measure_in_loop(_network(), "torch", config=config, device="cpu")
        assert result.wall_ms_per_frame > 0.0
        assert result.num_neurons == 200
        assert "stepped" in result.notes, "the method has to travel with the number"

    def test_the_build_is_excluded_from_the_frame_cost(self):
        """Constructing the network is paid once per episode at most. Folding it into
        the per-frame number would make a small network look worse than a large one."""
        config = BenchmarkConfig(dt_ms=0.5, biological_ms=100.0)
        result = measure_in_loop(_network(), "torch", config=config, device="cpu")
        assert result.build_s >= 0.0
        assert result.wall_ms_per_frame < 1000.0 * result.build_s + 1000.0

    @pytest.mark.slow
    def test_stepping_costs_more_than_batching_for_brian2(self):
        """The finding this module exists for, pinned. Brian2 re-prepares the network
        on every `run()` call, so the in-loop cost is several times the batched one --
        and quoting the batched number for a brain in the loop understates it by an
        order of magnitude."""
        pytest.importorskip("brian2")
        from fly_driver.brains import shiu

        network = _network(500, 4_000)
        config = BenchmarkConfig(dt_ms=0.5, biological_ms=200.0)
        batched = shiu.measure(network, config=config).wall_ms_per_frame
        stepped = measure_in_loop(network, "brian2", config=config).wall_ms_per_frame
        assert stepped > batched * 2.0, (
            f"stepped {stepped:.1f} ms vs batched {batched:.1f} ms -- if these are now "
            "close, Brian2's per-call overhead has changed and the GH-23 decision in "
            "AGENTS.md 10.12 should be re-measured"
        )
