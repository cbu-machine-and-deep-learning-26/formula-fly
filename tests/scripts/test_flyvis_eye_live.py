"""Tests for the live flyvis eye demo script."""

from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest

from fly_driver.eyes.hex_resampler import DEFAULT_FRAME_SHAPE, HEX_COLUMN_COUNT
from fly_driver.eyes.stimuli import GREY_LEVEL

SCRIPT_PATH = Path(__file__).parents[2] / "scripts" / "flyvis_eye_live.py"
READOUTS = ("T4a", "T4b", "T4c", "T4d", "T5a", "T5b", "T5c", "T5d")


def _load_live_module() -> ModuleType:
    module_spec = importlib.util.spec_from_file_location("flyvis_eye_live", SCRIPT_PATH)
    if module_spec is None or module_spec.loader is None:
        raise ImportError(f"could not load {SCRIPT_PATH}")
    module = importlib.util.module_from_spec(module_spec)
    # Dataclasses with postponed annotations look the module up in sys.modules.
    sys.modules[module_spec.name] = module
    module_spec.loader.exec_module(module)
    return module


def _features_with_active_readout(name: str, level: float) -> np.ndarray:
    """Zero features except one readout whose columns all sit at ``level``."""
    features = np.zeros(len(READOUTS) * HEX_COLUMN_COUNT, dtype=np.float32)
    index = READOUTS.index(name)
    features[index * HEX_COLUMN_COUNT : (index + 1) * HEX_COLUMN_COUNT] = level
    return features


def test_skips_cleanly_without_optional_stack(tmp_path: Path) -> None:
    """Exit 0 with a SKIP line when flyvis or its checkpoint is absent."""
    environment = os.environ.copy()
    environment["FLYVIS_ROOT_DIR"] = str(tmp_path)

    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT_PATH),
            "--source",
            "synthetic",
            "--no-display",
            "--frames",
            "3",
            "--show",
            "R1",
        ],
        check=False,
        capture_output=True,
        env=environment,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.startswith("SKIP:")


def _skip_without_pretrained_eye() -> None:
    pytest.importorskip("flyvis")
    from fly_driver.eyes.flyvis_eye import resolve_checkpoint_dir

    try:
        resolve_checkpoint_dir()
    except FileNotFoundError:
        pytest.skip("run `flyvis download-pretrained` to enable this test")


def _run_headless(*extra_args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            str(SCRIPT_PATH),
            "--source",
            "synthetic",
            "--no-display",
            "--fps-cap",
            "0",
            *extra_args,
        ],
        check=False,
        capture_output=True,
        text=True,
    )


def test_headless_synthetic_run_prints_direction_meter() -> None:
    """Run N synthetic frames without a window and report the meter and timing."""
    _skip_without_pretrained_eye()

    result = _run_headless("--frames", "30", "--print-every", "10")

    assert result.returncode == 0, result.stderr
    assert "SKIP" not in result.stdout
    meter_lines = [line for line in result.stdout.splitlines() if "meter left=" in line]
    assert len(meter_lines) == 3, result.stdout
    assert "frames: 30" in result.stdout
    assert "eye latency: median" in result.stdout


def test_headless_figure_has_retina_and_show_panels(tmp_path: Path) -> None:
    """Snapshot on Agg with the retina panel and extra --show cell types."""
    _skip_without_pretrained_eye()

    result = _run_headless(
        "--frames",
        "3",
        "--print-every",
        "0",
        "--save-dir",
        str(tmp_path),
        "--save-every",
        "2",
        "--show",
        "R1, Tm3,T4a",
        "--hide-t5",
    )

    assert result.returncode == 0, result.stderr
    assert "readouts: T4a, T4b, T4c, T4d\n" in result.stdout
    assert (
        "panels: camera, photoreceptors, motion percept, T4a, T4b, T4c, T4d, R1, Tm3\n"
        in result.stdout
    )
    assert "figure 12.50x6.80 in at 100 dpi = 1250x680 logical px" in result.stdout
    assert sorted(path.name for path in tmp_path.glob("*.png")) == [
        "flyvis_eye_live_00000.png",
        "flyvis_eye_live_00002.png",
    ]


def test_headless_retina_panel_and_hidpi_figure_fit(tmp_path: Path) -> None:
    """--show-retina adds the camera-derived panel; --dpi 200 shrinks to the screen."""
    _skip_without_pretrained_eye()

    result = _run_headless(
        "--frames",
        "1",
        "--print-every",
        "1",
        "--save-dir",
        str(tmp_path),
        "--show-retina",
        "--dpi",
        "200",
    )

    assert result.returncode == 0, result.stderr
    assert (
        "panels: camera, retina, photoreceptors, motion percept, T4a, T4b, T4c, T4d, "
        "T5a, T5b, T5c, T5d\n" in result.stdout
    )
    assert "in at 200 dpi = 1378x750 logical px" in result.stdout
    assert "motion" in result.stdout and "deg" in result.stdout
    assert (tmp_path / "flyvis_eye_live_00000.png").exists()


def test_headless_rejects_unknown_show_types() -> None:
    """Unknown or partial-lattice cell types fail fast with a clear message."""
    _skip_without_pretrained_eye()

    unknown = _run_headless("--frames", "1", "--show", "R1,NotACell")
    partial = _run_headless("--frames", "1", "--show", "Lawf1")

    assert unknown.returncode == 2
    assert "unknown cell types ['NotACell']" in unknown.stderr
    assert "choose from" in unknown.stderr
    assert partial.returncode == 2
    assert "721-column lattice" in partial.stderr


def test_motion_percept_hue_follows_direction_and_still_is_dark() -> None:
    """Right is red, up is yellow-green, left is cyan, down is violet; no motion is black."""
    live = _load_live_module()
    zeros = np.zeros(HEX_COLUMN_COUNT, dtype=np.float32)
    ones = np.ones(HEX_COLUMN_COUNT, dtype=np.float32)
    expected_rgb = {
        "right": [1.0, 0.0, 0.0],
        "up": [0.5, 1.0, 0.0],
        "left": [0.0, 1.0, 1.0],
        "down": [0.5, 0.0, 1.0],
    }

    for direction, rgb in expected_rgb.items():
        channels = {name: zeros for name in live.METER_DIRECTIONS}
        channels[direction] = ones
        colors, vectors = live.compute_motion_percept(channels, peak=1.0)
        assert colors.shape == (HEX_COLUMN_COUNT, 3)
        assert np.allclose(colors[0], rgb, atol=1e-6), direction
        angle, magnitude = live.summarise_motion(vectors)
        assert magnitude == pytest.approx(1.0)
        assert angle == pytest.approx(
            {"right": 0, "up": 90, "left": 180, "down": 270}[direction]
        )

    still = {name: zeros for name in live.METER_DIRECTIONS}
    colors, _ = live.compute_motion_percept(still, peak=1.0)
    assert np.all(colors == 0.0)
    half = {name: zeros for name in live.METER_DIRECTIONS}
    half["right"] = ones * 0.5
    colors, _ = live.compute_motion_percept(half, peak=1.0)
    assert np.allclose(colors[0], [0.5, 0.0, 0.0])
    wheel = live.hue_wheel_image(16)
    assert wheel.shape == (16, 16, 4)
    assert wheel[0, 0, 3] == 0.0 and wheel[8, 8, 3] == 1.0


def test_running_peak_decays_toward_floor() -> None:
    """The peak follows spikes immediately and relaxes slowly afterwards."""
    live = _load_live_module()
    peak = live.RunningPeak(floor=0.1, decay=0.5)

    assert peak.update(2.0) == 2.0
    assert peak.update(0.0) == 1.0
    assert peak.update(0.0) == 0.5
    assert peak.update(0.0) == 0.25
    assert peak.update(0.0) == 0.125
    assert peak.update(0.0) == 0.1


def test_figure_fit_and_font_scale_follow_the_screen() -> None:
    """The figure shrinks to the screen at high dpi and fonts shrink with it."""
    live = _load_live_module()

    assert live.fit_figure_size(2, 100.0) == (12.5, 6.8)
    # Three rows want 8.5 in of height; the 900 px screen allows 7.5, so the
    # whole figure shrinks uniformly.
    assert live.fit_figure_size(3, 100.0) == (11.03, 7.5)
    width, height = live.fit_figure_size(3, 200.0)
    assert width * 200 <= 1400 and height * 200 <= 750
    assert width / height == pytest.approx(12.5 / 8.5, rel=0.01)
    assert live.fit_figure_size(2, 100.0, requested=(6.0, 4.0)) == (6.0, 4.0)
    assert live.fit_figure_size(2, 100.0, screen_px=(1024, 640)) == (
        pytest.approx(9.0, abs=0.02),
        pytest.approx(4.9, abs=0.02),
    )
    assert live.font_scale(12.5) == pytest.approx(1.0)
    assert live.font_scale(6.25) == pytest.approx(0.5)
    assert live.font_scale(6.25, user_scale=2.0) == pytest.approx(1.0)
    assert live.parse_figsize(None) is None
    assert live.parse_figsize("10,5.5") == (10.0, 5.5)
    assert live.parse_figsize("8x4") == (8.0, 4.0)
    with pytest.raises(ValueError, match="W,H"):
        live.parse_figsize("10")
    with pytest.raises(ValueError, match="positive"):
        live.parse_figsize("0,4")


def test_show_types_are_parsed_and_ordered_after_readouts() -> None:
    """--show splits on commas, drops blanks and repeats, and follows T4/T5."""
    live = _load_live_module()

    assert live.parse_show_types(None) == []
    assert live.parse_show_types(" R1, L1,,Mi1,R1 ") == ["R1", "L1", "Mi1"]
    assert live.build_readout_names() == READOUTS
    assert live.build_readout_names(hide_t5=True) == READOUTS[:4]
    assert live.build_panel_names(READOUTS[:4], ["Tm3", "T4a", "R1"]) == [
        *READOUTS[:4],
        "Tm3",
        "R1",
    ]


def test_direction_meter_maps_subtypes_to_directions() -> None:
    """T4/T5 a, b, c, d drive left, right, up, down; the loudest one wins."""
    live = _load_live_module()
    expected = {"T4a": "left", "T5b": "right", "T4c": "up", "T5d": "down"}

    for readout, direction in expected.items():
        meter = live.compute_direction_meter(
            _features_with_active_readout(readout, 2.0), READOUTS
        )
        assert set(meter) == set(live.METER_DIRECTIONS)
        # Pooling one active readout with its silent sibling halves the mean but
        # leaves the 95th percentile on the active half.
        assert meter[direction] == pytest.approx(2.0)
        assert all(meter[other] == 0.0 for other in meter if other != direction)
        assert live.find_winning_direction(meter) == direction

    mean_meter = live.compute_direction_meter(
        _features_with_active_readout("T4a", 2.0), READOUTS, statistic="mean"
    )
    assert mean_meter["left"] == pytest.approx(1.0)


def test_direction_meter_subtracts_baseline_and_clips_at_zero() -> None:
    """Resting activity is removed and negative excursions never show as motion."""
    live = _load_live_module()
    features = _features_with_active_readout("T4b", -1.5)
    features += _features_with_active_readout("T4d", 0.4)
    baseline = {"down": 0.4}

    meter = live.compute_direction_meter(features, READOUTS, baseline=baseline)

    assert meter["right"] == 0.0
    assert meter["down"] == pytest.approx(0.0, abs=1e-6)
    assert live.find_winning_direction(meter) is None
    with pytest.raises(ValueError, match="statistic"):
        live.compute_direction_meter(features, READOUTS, statistic="max")


def test_direction_meter_ignores_non_motion_readouts() -> None:
    """Readouts outside T4/T5 (for example Tm9) contribute nothing."""
    live = _load_live_module()
    readouts = ("Tm9", "T4b")
    features = np.zeros(2 * HEX_COLUMN_COUNT, dtype=np.float32)
    features[:HEX_COLUMN_COUNT] = 5.0
    features[HEX_COLUMN_COUNT:] = 1.0

    meter = live.compute_direction_meter(features, readouts)

    assert meter == {"left": 0.0, "right": pytest.approx(1.0), "up": 0.0, "down": 0.0}


def test_center_crop_resize_keeps_the_middle_square() -> None:
    """A landscape camera frame is cropped around its center, then shrunk."""
    live = _load_live_module()
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    frame[:, 80:560] = 40  # the central 480x480 square survives the crop
    frame[200:280, 280:360, 0] = 255  # a red block at the exact center

    resized = live.center_crop_resize(frame)

    assert resized.shape == DEFAULT_FRAME_SHAPE
    assert resized.dtype == np.uint8
    assert resized[0, 0].tolist() == [40, 40, 40]
    assert resized[48, 48].tolist() == [255, 40, 40]
    assert resized[48, 8].tolist() == [40, 40, 40]


def test_center_crop_resize_is_identity_on_contract_frames() -> None:
    """Frames already in the eye's shape pass through unchanged."""
    live = _load_live_module()
    generator = np.random.default_rng(0)
    frame = generator.integers(0, 256, size=DEFAULT_FRAME_SHAPE, dtype=np.uint8)

    assert np.array_equal(live.center_crop_resize(frame), frame)
    with pytest.raises(TypeError, match="uint8"):
        live.center_crop_resize(frame.astype(np.float32))
    with pytest.raises(ValueError, match="height, width, 3"):
        live.center_crop_resize(frame[..., 0])


def test_synthetic_bar_cycle_visits_all_directions() -> None:
    """The bar sweeps left, right, up, down on grey, ~2 s each, in the contract."""
    live = _load_live_module()

    frames, labels = live.synthetic_bar_cycle()

    assert frames.dtype == np.uint8
    assert frames.shape == (400, *DEFAULT_FRAME_SHAPE)
    assert len(labels) == len(frames)
    assert [labels.count(name) for name in live.METER_DIRECTIONS] == [90] * 4
    assert set(np.unique(frames)) == {GREY_LEVEL, 255}
    right_frames = [index for index, label in enumerate(labels) if label == "right"]
    bar_columns = [
        np.flatnonzero(frames[index, 48, :, 0] == 255).mean() for index in right_frames
    ]
    assert bar_columns == sorted(bar_columns)
    assert bar_columns[0] < 10 and bar_columns[-1] > 85

    source = live.SyntheticBarSource()
    first = source.read()
    assert first is not None and first.frame.shape == DEFAULT_FRAME_SHAPE
    assert first.label == "grey"
