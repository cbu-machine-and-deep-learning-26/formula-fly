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
    assert "panels: camera, retina, T4a, T4b, T4c, T4d, R1, Tm3\n" in result.stdout
    assert sorted(path.name for path in tmp_path.glob("*.png")) == [
        "flyvis_eye_live_00000.png",
        "flyvis_eye_live_00002.png",
    ]


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
