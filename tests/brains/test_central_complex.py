"""Stability, wiring, parameter counts, and the frame-time curve (GH-23).

No flyvis. Torch is optional: the module skips when it is not installed, which is
the base CI image.
"""

from __future__ import annotations

import math
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from fly_driver.brains.blank import BlankWidthNet, blank_trainable_parameter_count  # noqa: E402
from fly_driver.brains.central_complex import CentralComplexBrain  # noqa: E402
from fly_driver.brains.constants import (  # noqa: E402
    DEFAULT_INPUT_IN_DEGREE,
    DEFAULT_RECURRENT_IN_DEGREE,
    VOLTAGE_LIMIT,
)
from fly_driver.brains.sweep import frame_budget_ms, sweep_frame_times  # noqa: E402
from fly_driver.interface import FRAME_RATE_HZ, Brain  # noqa: E402

pytestmark = pytest.mark.usefixtures("cpu_default_device")

LONG_ROLLOUT_STEPS = 2000  # 40 s of simulated time at 50 Hz
INPUT_DIM = 64
NEURON_COUNT = 64
OUTPUT_DIM = 16


def _brain(**overrides) -> CentralComplexBrain:
    settings = {
        "input_dim": INPUT_DIM,
        "neuron_count": NEURON_COUNT,
        "output_dim": OUTPUT_DIM,
        "seed": 0,
    }
    settings.update(overrides)
    return CentralComplexBrain(**settings)


def _noise(seed: int, steps: int, input_dim: int, scale: float, offset: float) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return (rng.random((steps, input_dim)).astype(np.float32) * scale + offset).astype(np.float32)


class TestTheModuleMeetsTheContract:
    def test_it_is_a_brain_and_the_populations_sum_to_the_width(self):
        brain = _brain()
        assert isinstance(brain, Brain)
        assert brain.populations.total == NEURON_COUNT
        assert brain.output_dim == brain.descending_count == OUTPUT_DIM
        assert brain.compass_count >= 8
        assert brain.steering_count >= 1

    def test_a_width_that_cannot_hold_the_ring_is_refused(self):
        with pytest.raises(ValueError, match="at least"):
            CentralComplexBrain(8, neuron_count=12)
        with pytest.raises(ValueError, match="descending"):
            CentralComplexBrain(8, neuron_count=16, output_dim=8)

    def test_step_returns_finite_float32_of_the_descending_width(self):
        brain = _brain()
        brain.reset()
        features = np.linspace(-1.0, 1.0, INPUT_DIM, dtype=np.float32)
        output = brain.step(features)
        assert output.shape == (OUTPUT_DIM,)
        assert output.dtype == np.float32
        assert np.all(np.isfinite(output))

    def test_non_finite_features_are_refused(self):
        brain = _brain()
        bad = np.zeros(INPUT_DIM, dtype=np.float32)
        bad[0] = np.nan
        with pytest.raises(ValueError, match="finite"):
            brain.step(bad)

    def test_the_same_seed_replays_and_reset_restarts_it(self):
        drive = _noise(1, 5, INPUT_DIM, scale=2.0, offset=-1.0)
        first = _brain()
        second = _brain()
        first.reset()
        second.reset()
        played = [first.step(frame) for frame in drive]
        replayed = [second.step(frame) for frame in drive]
        for left, right in zip(played, replayed, strict=True):
            assert np.array_equal(left, right)
        first.reset()
        assert np.array_equal(first.step(drive[0]), played[0])

    def test_a_different_seed_changes_the_wiring(self):
        left = _brain(seed=0)
        right = _brain(seed=1)
        assert not np.array_equal(left.input_pre.cpu().numpy(), right.input_pre.cpu().numpy())


class TestSyntheticWiring:
    def test_degrees_match_the_requested_masks(self):
        brain = _brain()
        input_degrees = brain.input_in_degrees()
        recurrent_degrees = brain.recurrent_in_degrees()
        assert np.all(input_degrees == min(DEFAULT_INPUT_IN_DEGREE, INPUT_DIM))
        assert np.all(recurrent_degrees[: brain.compass_count] == 5)
        assert np.all(recurrent_degrees[brain.compass_count :] == DEFAULT_RECURRENT_IN_DEGREE)

    def test_recurrent_edges_are_not_self_synapses(self):
        brain = _brain()
        pre = brain.recurrent_pre.cpu().numpy()
        post = brain.recurrent_post.cpu().numpy()
        assert np.all(pre != post)

    def test_descending_cells_only_listen_upstream(self):
        brain = _brain()
        pre = brain.recurrent_pre.cpu().numpy()
        post = brain.recurrent_post.cpu().numpy()
        descending = post >= brain.descending_start
        assert np.all(pre[descending] < brain.descending_start)

    def test_input_degree_clips_to_the_feature_count(self):
        brain = CentralComplexBrain(4, neuron_count=32, output_dim=8, input_in_degree=16, seed=0)
        assert np.all(brain.input_in_degrees() == 4)


class TestParameterCounts:
    def test_the_blank_count_matches_its_parameters(self):
        blank = BlankWidthNet(INPUT_DIM, NEURON_COUNT, OUTPUT_DIM)
        assert blank.trainable_parameter_count() == blank_trainable_parameter_count(
            INPUT_DIM, NEURON_COUNT
        )

    def test_the_constrained_module_is_smaller_than_the_blank_net(self):
        comparison = _brain().compare_parameters()
        assert comparison.constrained_trainable < comparison.blank_trainable
        assert comparison.constrained_is_smaller
        assert comparison.blank_trainable == blank_trainable_parameter_count(
            INPUT_DIM, NEURON_COUNT
        )

    def test_blank_steps_stay_finite(self):
        blank = BlankWidthNet(INPUT_DIM, NEURON_COUNT, OUTPUT_DIM, device="cpu")
        blank.reset()
        for frame in _noise(2, 30, INPUT_DIM, scale=2.0, offset=-0.5):
            output = blank.step(frame)
            assert output.shape == (OUTPUT_DIM,)
            assert np.all(np.isfinite(output))


class TestLongRolloutStability:
    @pytest.mark.parametrize(
        ("scale", "offset"),
        [(1.0, 0.0), (9.0, -3.0)],  # pixels in [0, 1], and a flyvis-like span
    )
    def test_a_long_rollout_stays_finite_and_in_a_plausible_rate_band(self, scale, offset):
        brain = _brain()
        brain.reset()
        rates_hz: list[np.ndarray] = []
        outputs: list[np.ndarray] = []
        for frame in _noise(3, LONG_ROLLOUT_STEPS, INPUT_DIM, scale=scale, offset=offset):
            outputs.append(brain.step(frame))
            rates_hz.append(brain.firing_rates_hz())
        stacked_outputs = np.stack(outputs)
        stacked_rates = np.stack(rates_hz)
        activity = brain.activity()
        assert np.all(np.isfinite(stacked_outputs))
        assert np.all(np.isfinite(activity))
        assert np.max(np.abs(activity)) <= VOLTAGE_LIMIT
        assert np.all(stacked_rates >= 0.0)
        assert np.all(stacked_rates <= FRAME_RATE_HZ + 1e-3)
        # Responds, and does not sit on the 50 Hz cap for the whole population.
        assert float(stacked_outputs.std()) > 1e-4
        assert 0.0 < float(stacked_rates.mean()) < FRAME_RATE_HZ * 0.5

    def test_huge_weights_stay_clamped_rather_than_producing_nans(self):
        brain = _brain()
        with torch.no_grad():
            brain.input_weight.mul_(1e6)
            brain.recurrent_weight.mul_(1e6)
        brain.reset()
        drive = np.ones(INPUT_DIM, dtype=np.float32)
        for _ in range(100):
            output = brain.step(drive)
            assert np.all(np.isfinite(output))
        assert np.max(np.abs(brain.activity())) <= VOLTAGE_LIMIT + 1e-5


class TestFrameTimeCurve:
    def test_small_widths_fit_the_frame_and_report_both_parameter_counts(self):
        points = sweep_frame_times(
            (32, 128, 512),
            input_dim=INPUT_DIM,
            output_dim=8,
            steps=15,
            warmup=5,
            seed=0,
        )
        budget = frame_budget_ms()
        assert budget == pytest.approx(1000.0 / FRAME_RATE_HZ)
        assert [point.neuron_count for point in points] == [32, 128, 512]
        previous = 0
        for point in points:
            assert point.constrained_trainable < point.blank_trainable
            assert point.constrained_trainable > previous
            previous = point.constrained_trainable
            assert math.isfinite(point.median_frame_ms) and point.median_frame_ms > 0.0
            assert point.p95_frame_ms >= point.median_frame_ms
            assert point.frame_budget_ms == pytest.approx(budget)
        assert points[0].meets_frame_budget
        assert points[1].meets_frame_budget

    def test_the_curve_script_writes_a_csv(self, tmp_path: Path):
        out = tmp_path / "curve.csv"
        completed = subprocess.run(
            [
                sys.executable,
                "scripts/brain_frame_time.py",
                "--counts",
                "32,64",
                "--input-dim",
                "16",
                "--output-dim",
                "8",
                "--steps",
                "4",
                "--warmup",
                "1",
                "--out",
                str(out),
            ],
            check=False,
            capture_output=True,
            text=True,
        )
        assert completed.returncode == 0, completed.stderr
        text = out.read_text(encoding="utf-8")
        assert "meets_frame_budget" in text
        assert "constrained_trainable" in text
        assert completed.stdout.splitlines()[0].startswith("frame budget")
