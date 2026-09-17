#!/usr/bin/env python3
"""Run the frozen FlyvisEye on a moving-edge stimulus, time it, and plot readouts."""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from fly_driver.eyes.hex_resampler import (
    HEX_COLUMN_COUNT,
    hex_receptor_centers,
)
from fly_driver.eyes.stimuli import (
    DIRECTIONS,
    drifting_grating_frames,
    moving_edge_frames,
)

PLOT_DIRECTIONS = ("right", "left")
PLOT_COLUMNS = 8
PRESTIMULUS_FRAMES = 25
SWEEP_FRAMES = 50
HOLD_FRAMES = 10
SETTLING_DRIFT_FRAMES = 10


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("outputs/flyvis_eye_demo"),
        help="Directory for the PNGs (created if missing).",
    )
    parser.add_argument(
        "--timing-frames",
        type=int,
        default=200,
        help="Frames streamed through encode() for the latency measurement.",
    )
    return parser.parse_args()


def _time_streaming(eye: Any, frames: np.ndarray) -> dict[str, float]:
    eye.reset()
    durations = []
    for frame in frames:
        start = time.perf_counter()
        eye.encode(frame)
        durations.append((time.perf_counter() - start) * 1000)
    return {
        "median_ms": statistics.median(durations),
        "mean_ms": statistics.fmean(durations),
        "p95_ms": float(np.percentile(durations, 95)),
        "max_ms": max(durations),
    }


def _time_sequence(eye: Any, frames: np.ndarray) -> float:
    start = time.perf_counter()
    eye.encode_sequence(frames)
    return (time.perf_counter() - start) * 1000 / len(frames)


def _time_reset(eye: Any) -> float:
    start = time.perf_counter()
    eye.reset()
    return (time.perf_counter() - start) * 1000


def _readout_maps(eye: Any, features: np.ndarray) -> dict[str, np.ndarray]:
    """Split ``(time, feature_dim)`` features into per-readout ``(time, 721)`` maps."""
    return {
        name: features[:, index * HEX_COLUMN_COUNT : (index + 1) * HEX_COLUMN_COUNT]
        for index, name in enumerate(eye.readout_names)
    }


def _hex_scatter(axes: Any, values: np.ndarray, **kwargs: Any) -> Any:
    """Draw one 721-column map in camera orientation (row down, column right)."""
    centers = hex_receptor_centers().numpy()
    axes.set_facecolor("#808080")
    scatter = axes.scatter(
        centers[:, 1], -centers[:, 0], c=values, s=9, marker="h", linewidths=0, **kwargs
    )
    axes.set_aspect("equal")
    axes.set_xticks([])
    axes.set_yticks([])
    return scatter


def _plot_edge_response(
    eye: Any, frames: np.ndarray, features: np.ndarray, direction: str, path: Path
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    maps = _readout_maps(eye, features)
    plot_frames = [
        PRESTIMULUS_FRAMES - 1,
        *np.linspace(
            PRESTIMULUS_FRAMES, PRESTIMULUS_FRAMES + SWEEP_FRAMES - 1, PLOT_COLUMNS - 2
        )
        .round()
        .astype(int),
        len(frames) - 1,
    ]
    limit = float(max(np.abs(maps[name]).max() for name in eye.readout_names))
    rows = 1 + len(eye.readout_names)
    figure, axes = plt.subplots(
        rows,
        len(plot_frames),
        figsize=(1.5 * len(plot_frames), 1.4 * rows),
        layout="constrained",
        squeeze=False,
    )
    for column, frame_index in enumerate(plot_frames):
        axes[0, column].imshow(frames[frame_index])
        axes[0, column].set_title(
            f"{frame_index / eye.frame_rate_hz:.2f} s", fontsize=8
        )
        axes[0, column].set_xticks([])
        axes[0, column].set_yticks([])
        for row, name in enumerate(eye.readout_names, start=1):
            scatter = _hex_scatter(
                axes[row, column],
                maps[name][frame_index],
                cmap="coolwarm",
                vmin=-limit,
                vmax=limit,
            )
    axes[0, 0].set_ylabel("camera", fontsize=8)
    for row, name in enumerate(eye.readout_names, start=1):
        axes[row, 0].set_ylabel(name, fontsize=8, fontweight="bold")
    figure.suptitle(f"FlyvisEye readouts for a bright edge moving {direction}")
    figure.colorbar(
        scatter, ax=axes.ravel().tolist(), label="activity (a.u.)", shrink=0.5
    )
    figure.savefig(path, dpi=110)
    plt.close(figure)


def _plot_direction_preference(
    eye: Any, amplitudes: dict[str, np.ndarray], path: Path
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(figsize=(8, 3.2), layout="constrained")
    readouts = list(eye.readout_names)
    positions = np.arange(len(readouts))
    width = 0.8 / len(DIRECTIONS)
    for offset, direction in enumerate(DIRECTIONS):
        axes.bar(
            positions + (offset - (len(DIRECTIONS) - 1) / 2) * width,
            amplitudes[direction],
            width=width,
            label=f"grating moving {direction}",
        )
    axes.set_xticks(positions)
    axes.set_xticklabels(readouts)
    axes.set_ylabel("q95 activity during drift (a.u.)")
    axes.set_title(
        "Which direction each T4/T5 subtype prefers (FlyvisEye.encode_sequence)"
    )
    axes.legend(fontsize=8, ncol=2)
    figure.savefig(path, dpi=110)
    plt.close(figure)


def _measure_direction_preference(eye: Any) -> dict[str, np.ndarray]:
    amplitudes: dict[str, np.ndarray] = {}
    response_start = PRESTIMULUS_FRAMES + SETTLING_DRIFT_FRAMES
    for direction in DIRECTIONS:
        features = eye.encode_sequence(drifting_grating_frames(direction))
        maps = _readout_maps(eye, features)
        amplitudes[direction] = np.array(
            [
                np.quantile(maps[name][response_start:], 0.95)
                for name in eye.readout_names
            ]
        )
    return amplitudes


def main() -> int:
    """Run the demo, or skip successfully when the optional stack is absent."""
    args = _parse_args()
    from fly_driver.eyes.flyvis_eye import (
        FlyvisEye,
        FlyvisNotInstalledError,
        resolve_checkpoint_dir,
    )

    try:
        resolve_checkpoint_dir()
    except FlyvisNotInstalledError:
        if sys.version_info >= (3, 13):
            print("SKIP: flyvis needs Python 3.9-3.12.")
        else:
            print(
                "SKIP: flyvis is not installed; run "
                "`python -m pip install -r requirements-flyvis.txt`."
            )
        return 0
    except FileNotFoundError as error:
        print(f"SKIP: {error}")
        return 0

    import torch

    load_start = time.perf_counter()
    eye = FlyvisEye()
    print(f"checkpoint: {eye.checkpoint_dir}")
    print(f"load time: {time.perf_counter() - load_start:.1f} s")
    print(f"device: {eye.device}; torch threads: {torch.get_num_threads()}")
    print(f"readouts ({len(eye.readout_names)}): {', '.join(eye.readout_names)}")
    print(f"feature_dim: {eye.feature_dim}")
    print(
        f"cells: {eye.cell_count}; parameters: {eye.parameter_count()} "
        f"(trainable: {eye.trainable_parameters()})"
    )
    print(
        f"frame rate: {eye.frame_rate_hz:.0f} Hz (dt={eye.dt:.3f} s); "
        f"warm-up: {eye.warmup_seconds:.1f} s at luminance {eye.warmup_luminance}"
    )

    args.out.mkdir(parents=True, exist_ok=True)
    for direction in PLOT_DIRECTIONS:
        frames = moving_edge_frames(
            direction,
            prestimulus_frames=PRESTIMULUS_FRAMES,
            sweep_frames=SWEEP_FRAMES,
            hold_frames=HOLD_FRAMES,
        )
        features = eye.encode_sequence(frames)
        path = args.out / f"flyvis_eye_edge_{direction}.png"
        _plot_edge_response(eye, frames, features, direction, path)
        maps = _readout_maps(eye, features)
        sweep = slice(PRESTIMULUS_FRAMES, PRESTIMULUS_FRAMES + SWEEP_FRAMES)
        means = {name: float(maps[name][sweep].mean()) for name in eye.readout_names}
        winners = {
            family: max((f"{family}a", f"{family}b"), key=means.__getitem__)
            for family in ("T4", "T5")
        }
        print(
            f"edge {direction}: sweep means "
            + ", ".join(f"{name}={value:.3f}" for name, value in means.items())
        )
        print(f"edge {direction}: horizontal winners {winners}")
        print(f"plot: {path}")

    amplitudes = _measure_direction_preference(eye)
    preference_path = args.out / "flyvis_eye_direction_preference.png"
    _plot_direction_preference(eye, amplitudes, preference_path)
    print(f"plot: {preference_path}")

    timing_frames = moving_edge_frames(
        "right", prestimulus_frames=0, sweep_frames=args.timing_frames, hold_frames=0
    )
    streaming = _time_streaming(eye, timing_frames)
    sequence_ms = _time_sequence(eye, timing_frames)
    reset_ms = _time_reset(eye)
    print(
        f"encode() latency over {args.timing_frames} frames: "
        f"median {streaming['median_ms']:.2f} ms, mean {streaming['mean_ms']:.2f} ms, "
        f"p95 {streaming['p95_ms']:.2f} ms, max {streaming['max_ms']:.2f} ms"
    )
    print(f"encode_sequence() throughput: {sequence_ms:.2f} ms/frame")
    print(f"reset() warm-up: {reset_ms:.0f} ms")
    print(
        f"frame budget at {eye.frame_rate_hz:.0f} Hz: {1000 / eye.frame_rate_hz:.0f} ms"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
