#!/usr/bin/env python3
"""Run a pretrained flyvis optic-lobe network on a synthetic short sequence."""

from __future__ import annotations

import argparse
import operator
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

MOTION_READOUTS = (
    "T4a",
    "T4b",
    "T4c",
    "T4d",
    "T5a",
    "T5b",
    "T5c",
    "T5d",
)
FRAME_COUNT = 4
HEXAL_COUNT = 721
FRAME_INTERVAL_SECONDS = 1 / 50
EDGE_PRE_STIMULUS_FRAMES = 25
EDGE_SWEEP_FRAMES = 50
EDGE_POST_STIMULUS_FRAMES = 10


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--direction",
        choices=("ltr", "rtl"),
        default="ltr",
        help="Edge sweep direction (default: ltr).",
    )
    parser.add_argument(
        "--plot",
        type=Path,
        help="Output prefix for input and response PNGs.",
    )
    parser.add_argument(
        "--readout",
        action="append",
        dest="readouts",
        help="Cell type to report; repeat for multiple types (default: T4/T5).",
    )
    parser.add_argument(
        "--stimulus",
        choices=("ramp", "edge"),
        default="ramp",
        help="Synthetic stimulus to simulate (default: ramp).",
    )
    return parser.parse_args()


def _create_ramp_sequence(torch: Any) -> Any:
    base_frame = torch.linspace(0.1, 0.9, HEXAL_COUNT)
    frames = [torch.roll(base_frame, shifts=frame) for frame in range(FRAME_COUNT)]
    return torch.stack(frames)[None, :, None, :]


def _create_edge_frames(horizontal_positions: Sequence[float], direction: str) -> list[list[float]]:
    if not horizontal_positions:
        raise ValueError("horizontal_positions must not be empty")
    if direction not in {"ltr", "rtl"}:
        raise ValueError("direction must be 'ltr' or 'rtl'")

    left_edge = min(horizontal_positions)
    right_edge = max(horizontal_positions)
    baseline = [0.5] * len(horizontal_positions)
    thresholds = [
        left_edge + (right_edge - left_edge) * frame / (EDGE_SWEEP_FRAMES - 1)
        for frame in range(EDGE_SWEEP_FRAMES)
    ]
    if direction == "rtl":
        thresholds.reverse()
    is_bright = operator.le if direction == "ltr" else operator.ge
    sweep_frames = [
        [1.0 if is_bright(position, threshold) else 0.0 for position in horizontal_positions]
        for threshold in thresholds
    ]
    return (
        [baseline.copy() for _ in range(EDGE_PRE_STIMULUS_FRAMES)]
        + sweep_frames
        + [sweep_frames[-1].copy() for _ in range(EDGE_POST_STIMULUS_FRAMES)]
    )


def _create_edge_sequence(torch: Any, direction: str) -> Any:
    from flyvis.utils.hex_utils import get_hex_coords, get_hextent, hex_to_pixel

    extent = get_hextent(HEXAL_COUNT)
    horizontal_hex, vertical_hex = get_hex_coords(extent)
    horizontal_pixels, _ = hex_to_pixel(horizontal_hex, vertical_hex)
    frames = _create_edge_frames(horizontal_pixels.tolist(), direction)
    return torch.tensor(frames, dtype=torch.float32)[None, :, None, :]


def _find_missing_readouts(
    requested_readouts: Sequence[str], available_readouts: Sequence[str]
) -> list[str]:
    return sorted(set(requested_readouts) - set(available_readouts))


def _get_plot_frames(stimulus_name: str, frame_count: int) -> list[int]:
    if stimulus_name != "edge":
        return list(range(frame_count))

    sweep_start = EDGE_PRE_STIMULUS_FRAMES
    sweep_frames = [
        sweep_start
        + round(
            sample * (EDGE_SWEEP_FRAMES - 1) / 6,
        )
        for sample in range(7)
    ]
    return [0, sweep_start - 1, *sweep_frames, frame_count - 1]


def _print_edge_metrics(responses: Any, direction: str) -> None:
    sweep_start = EDGE_PRE_STIMULUS_FRAMES
    sweep_stop = sweep_start + EDGE_SWEEP_FRAMES
    means = {
        readout: float(responses[readout][:, sweep_start:sweep_stop].mean())
        for readout in MOTION_READOUTS
    }
    for readout in MOTION_READOUTS:
        print(f"edge {direction} sweep mean {readout}: {means[readout]:.6f}")

    for family in ("T4", "T5"):
        for first_subtype, second_subtype in (("a", "b"), ("c", "d")):
            first_readout = f"{family}{first_subtype}"
            second_readout = f"{family}{second_subtype}"
            denominator = abs(means[first_readout]) + abs(means[second_readout])
            contrast = (
                (means[first_readout] - means[second_readout]) / denominator if denominator else 0.0
            )
            print(
                f"edge {direction} contrast ({first_readout}-{second_readout})/"
                f"(|{first_readout}|+|{second_readout}|): {contrast:.6f}"
            )
        horizontal_winner = max((f"{family}a", f"{family}b"), key=means.__getitem__)
        print(f"edge {direction} horizontal winner {family}: {horizontal_winner}")


def _save_plots(
    plot_prefix: Path,
    stimulus_name: str,
    direction: str,
    sequence: Any,
    responses: Any,
) -> tuple[Path, Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import torch
    from flyvis.analysis.visualization.plots import quick_hex_scatter

    if plot_prefix.suffix.lower() == ".png":
        plot_prefix = plot_prefix.with_suffix("")
    plot_prefix.parent.mkdir(parents=True, exist_ok=True)
    input_path = Path(f"{plot_prefix}_input.png")
    response_path = Path(f"{plot_prefix}_responses.png")

    frame_count = sequence.shape[1]
    plot_frames = _get_plot_frames(stimulus_name, frame_count)
    input_figure, input_axes = plt.subplots(
        1,
        len(plot_frames),
        figsize=(1.6 * len(plot_frames), 2),
        layout="constrained",
        squeeze=False,
    )
    input_scalarmapper = None
    for column, frame in enumerate(plot_frames):
        input_axes[0, column].set_facecolor("#808080")
        _, _, (_, input_scalarmapper) = quick_hex_scatter(
            sequence[0, frame, 0],
            ax=input_axes[0, column],
            cbar=False,
            cmap=plt.get_cmap("gray"),
            fig=input_figure,
            title=f"{frame * FRAME_INTERVAL_SECONDS:.2f} s",
            vmin=0,
            vmax=1,
        )
    stimulus_label = f"{stimulus_name} {direction}" if stimulus_name == "edge" else stimulus_name
    input_figure.suptitle(f"{stimulus_label.capitalize()} input on flyvis retina")
    input_figure.colorbar(
        input_scalarmapper,
        ax=input_axes.ravel().tolist(),
        label="luminance",
        shrink=0.7,
    )
    input_figure.savefig(input_path, dpi=180)
    plt.close(input_figure)

    motion_responses = torch.stack([responses[readout][0] for readout in MOTION_READOUTS])
    response_limit = float(motion_responses.abs().max())
    response_figure, response_axes = plt.subplots(
        len(MOTION_READOUTS),
        len(plot_frames),
        figsize=(1.6 * len(plot_frames), 1.45 * len(MOTION_READOUTS)),
        layout="constrained",
        squeeze=False,
    )
    response_scalarmapper = None
    for row, readout in enumerate(MOTION_READOUTS):
        for column, frame in enumerate(plot_frames):
            response_axes[row, column].set_facecolor("#808080")
            _, _, (_, response_scalarmapper) = quick_hex_scatter(
                motion_responses[row, frame],
                ax=response_axes[row, column],
                cbar=False,
                cmap=plt.get_cmap("coolwarm"),
                fig=response_figure,
                midpoint=0,
                title=(f"{frame * FRAME_INTERVAL_SECONDS:.2f} s" if row == 0 else ""),
                vmin=-response_limit,
                vmax=response_limit,
            )
            if column == 0:
                response_axes[row, column].text(
                    -0.2,
                    0.5,
                    readout,
                    fontsize=8,
                    fontweight="bold",
                    ha="right",
                    rotation=90,
                    transform=response_axes[row, column].transAxes,
                    va="center",
                )
    response_figure.suptitle(f"T4/T5 responses to {stimulus_label} stimulus (shared scale)")
    response_figure.colorbar(
        response_scalarmapper,
        ax=response_axes.ravel().tolist(),
        label="response (a.u.)",
        shrink=0.5,
    )
    response_figure.savefig(response_path, dpi=180)
    plt.close(response_figure)
    return input_path, response_path


def main() -> int:
    """Run the smoke test, or skip successfully when optional data is absent."""
    args = _parse_args()
    try:
        import flyvis
    except ModuleNotFoundError as error:
        if error.name == "flyvis":
            if sys.version_info >= (3, 13):
                print("SKIP: flyvis needs Python 3.9-3.12.")
            else:
                print(
                    "SKIP: flyvis is not installed; run "
                    "`python -m pip install -r requirements-flyvis.txt`."
                )
            return 0
        raise

    checkpoint_dir = Path(flyvis.results_dir) / "flow" / "0000" / "000"
    if not checkpoint_dir.is_dir():
        print(
            f"SKIP: pretrained flyvis checkpoint not found at {checkpoint_dir}; "
            "run `flyvis download-pretrained`."
        )
        return 0

    import torch
    from flyvis.utils.activity_utils import LayerActivity

    requested_readouts = tuple(args.readouts or MOTION_READOUTS)
    network_view = flyvis.NetworkView(checkpoint_dir)
    network = network_view.init_network()
    network.eval()
    network.requires_grad_(False)

    if args.stimulus == "edge":
        sequence = _create_edge_sequence(torch, args.direction)
    else:
        sequence = _create_ramp_sequence(torch)
    responses = network.simulate(
        sequence,
        dt=FRAME_INTERVAL_SECONDS,
        as_layer_activity=True,
    )
    if not isinstance(responses, LayerActivity):
        raise TypeError("flyvis did not return cell-type-indexed activity")

    available_readouts = tuple(responses.output_cell_types)
    missing_readouts = _find_missing_readouts(requested_readouts, available_readouts)
    if missing_readouts:
        print(f"ERROR: unavailable readouts: {', '.join(missing_readouts)}")
        return 2

    selected_responses = [responses[readout] for readout in requested_readouts]
    readout_vector = torch.cat(selected_responses, dim=-1)

    print(f"flyvis version: {flyvis.__version__}")
    print(f"checkpoint: {checkpoint_dir}")
    print(f"input shape: {tuple(sequence.shape)}")
    print(f"full response shape: {tuple(responses.activity.shape)}")
    print(
        f"available output cell types ({len(available_readouts)}): {', '.join(available_readouts)}"
    )
    for readout, response in zip(requested_readouts, selected_responses, strict=True):
        print(f"{readout} shape: {tuple(response.shape)}")
    print(f"concatenated readout shape: {tuple(readout_vector.shape)}")
    if args.stimulus == "edge":
        _print_edge_metrics(responses, args.direction)
    if args.plot is not None:
        input_path, response_path = _save_plots(
            args.plot,
            args.stimulus,
            args.direction,
            sequence,
            responses,
        )
        print(f"input plot: {input_path}")
        print(f"response plot: {response_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
