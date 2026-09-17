#!/usr/bin/env python3
"""Run a pretrained flyvis optic-lobe network on a synthetic short sequence."""

from __future__ import annotations

import argparse
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


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--readout",
        action="append",
        dest="readouts",
        help="Cell type to report; repeat for multiple types (default: T4/T5).",
    )
    return parser.parse_args()


def _create_sequence(torch: Any) -> Any:
    base_frame = torch.linspace(0.1, 0.9, HEXAL_COUNT)
    frames = [torch.roll(base_frame, shifts=frame) for frame in range(FRAME_COUNT)]
    return torch.stack(frames)[None, :, None, :]


def _find_missing_readouts(
    requested_readouts: Sequence[str], available_readouts: Sequence[str]
) -> list[str]:
    return sorted(set(requested_readouts) - set(available_readouts))


def main() -> int:
    """Run the smoke test, or skip successfully when optional data is absent."""
    try:
        import flyvis
    except ModuleNotFoundError as error:
        if error.name == "flyvis":
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

    requested_readouts = tuple(_parse_args().readouts or MOTION_READOUTS)
    network_view = flyvis.NetworkView(checkpoint_dir)
    network = network_view.init_network()
    network.eval()
    network.requires_grad_(False)

    sequence = _create_sequence(torch)
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
        f"available output cell types ({len(available_readouts)}): "
        f"{', '.join(available_readouts)}"
    )
    for readout, response in zip(requested_readouts, selected_responses):
        print(f"{readout} shape: {tuple(response.shape)}")
    print(f"concatenated readout shape: {tuple(readout_vector.shape)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
