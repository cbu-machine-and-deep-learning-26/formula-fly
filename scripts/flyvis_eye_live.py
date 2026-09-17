#!/usr/bin/env python3
"""Watch the frozen flyvis eye respond to a webcam (or a synthetic bar) live.

One window shows, left to right, the pipeline the fly sees through: the 96x96
frame the eye receives, the **retina** (the 721-column hex-resampled luminance
that actually enters the network, from ``HexResampler``), and the T4/T5
hexagonal maps updating through ``FlyvisEye.encode`` (streaming, state carried
between frames), plus a direction meter with one bar per motion direction. Wave
a hand across the camera and the bar for that direction jumps. ``--show
R1,L1,Mi1,Tm3`` adds panels for any of the model's cell types: the eye keeps
its T4/T5 readouts for the meter, and the extra panels are read from
``FlyvisEye.state_activity`` (the full network state after each step, which
also covers the 31 non-output types such as photoreceptors and lamina cells);
``--hide-t5`` drops the T5 row; ``--no-retina`` hides the retina panel.

**Display choice.** matplotlib with blitting, not pygame. matplotlib is already
used by every other script in the repo, so the live demo adds only
``opencv-python``; the hexagonal map drawing is shared with
``scripts/flyvis_eye_demo.py`` through ``fly_driver.analysis.hex_plots``; and
the same figure renders headlessly on the Agg backend for tests and PNG/GIF
evidence. Blitting redraws only the frame image, the eight maps, the meter
bars, and the overlay text. The maps are painted as small images
(``HexRaster``, one lookup per pixel) instead of 8 x 721 scatter markers, which
halves the draw cost; on a 4-core CPU VM the window runs at ~24 fps with the
eye at ~9 ms per frame, close to a webcam's ~30 fps. pygame would be smoother
for full-screen video, which this is not, and would need its own hex drawing.

**Timing.** Every frame is one 20 ms Euler step of the optic lobe regardless of
how fast frames arrive, so a ~30 fps webcam plays back through the eye at
50 Hz, about 1.7x faster than real time. That is fine for a demo and is stated
on screen; the synthetic source is paced to 50 Hz so it is real time.

Headless check (no window, prints the direction meter):

    python scripts/flyvis_eye_live.py --source synthetic --frames 200 --no-display

The script exits 0 with a ``SKIP`` message when flyvis, its checkpoint, or
(for the webcam) OpenCV is missing; a missing camera falls back to synthetic.
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import torch
from torch.nn import functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from fly_driver.analysis.hex_plots import HexRaster, split_readout_maps
from fly_driver.eyes.hex_resampler import DEFAULT_FRAME_SHAPE, HEX_COLUMN_COUNT
from fly_driver.eyes.stimuli import (
    FRAME_RATE_HZ,
    GREY_LEVEL,
    grey_frames,
    moving_edge_frames,
)

Frame = npt.NDArray[np.uint8]

T4_READOUTS = ("T4a", "T4b", "T4c", "T4d")
T5_READOUTS = ("T5a", "T5b", "T5c", "T5d")
MAPS_PER_ROW = 4
METER_DIRECTIONS = ("left", "right", "up", "down")
# Image-coordinate preferred direction of each T4/T5 subtype, the mapping the
# direction-selectivity gate in tests/eyes asserts against the pretrained eye.
SUBTYPE_DIRECTIONS = {"a": "left", "b": "right", "c": "up", "d": "down"}
METER_STATISTICS = ("q95", "mean")
SYNTHETIC_SECONDS_PER_DIRECTION = 2.0
SYNTHETIC_PAUSE_SECONDS = 0.2
SYNTHETIC_BAR_WIDTH_FRAMES = 6
DEFAULT_FPS_CAP = FRAME_RATE_HZ
# |activity| peaks near 2.3 a.u. for the synthetic bar; 1.5 keeps the maps readable.
DEFAULT_COLOR_LIMIT = 1.5
EMA_WEIGHT = 0.1
METER_PEAK_DECAY = 0.995
METER_PEAK_FLOOR = 0.05


# --------------------------------------------------------------------------
# Frame sources
# --------------------------------------------------------------------------


@dataclass
class SourceFrame:
    """One frame in the eye's contract plus an optional stimulus label."""

    frame: Frame
    label: str = ""


def center_crop_resize(
    frame: npt.NDArray[np.uint8],
    output_shape: tuple[int, int, int] = DEFAULT_FRAME_SHAPE,
) -> Frame:
    """Center-crop a camera frame to a square and resize it to the eye's shape.

    The eye enforces its declared frame shape and never resizes. The webcam is
    not part of that contract, so this is the one place a resize is allowed:
    it turns an arbitrary camera resolution into the declared ``output_shape``
    before the frame reaches the contract boundary.

    Args:
        frame: uint8 RGB frame ``(height, width, 3)`` of any size.
        output_shape: Declared eye frame shape ``(height, width, 3)``.

    Returns:
        A uint8 RGB frame with ``output_shape`` (area-averaged when shrinking).
    """
    if frame.ndim != 3 or frame.shape[2] != 3:
        raise ValueError(f"frame must be (height, width, 3), got {frame.shape}")
    if frame.dtype != np.uint8:
        raise TypeError(f"frame must be uint8, got {frame.dtype}")
    height, width, _ = frame.shape
    side = min(height, width)
    top = (height - side) // 2
    left = (width - side) // 2
    cropped = np.ascontiguousarray(frame[top : top + side, left : left + side])

    target_height, target_width, _ = output_shape
    if cropped.shape[:2] == (target_height, target_width):
        return cropped
    tensor = torch.from_numpy(cropped).permute(2, 0, 1)[None].float()
    resized = F.interpolate(tensor, size=(target_height, target_width), mode="area")
    return (
        resized.round().clamp(0, 255).to(torch.uint8)[0].permute(1, 2, 0).numpy().copy()
    )


def synthetic_bar_cycle(
    *,
    seconds_per_direction: float = SYNTHETIC_SECONDS_PER_DIRECTION,
    pause_seconds: float = SYNTHETIC_PAUSE_SECONDS,
    bar_width_frames: int = SYNTHETIC_BAR_WIDTH_FRAMES,
    frame_rate_hz: float = FRAME_RATE_HZ,
    frame_shape: tuple[int, int, int] = DEFAULT_FRAME_SHAPE,
) -> tuple[Frame, list[str]]:
    """Build one cycle of a bright bar sweeping left, right, up, then down.

    Each direction lasts ``seconds_per_direction``: a short grey pause, then a
    bar carved from :func:`fly_driver.eyes.stimuli.moving_edge_frames` (the
    bright region behind the leading edge minus the region ``bar_width_frames``
    earlier) on the grey background.

    Args:
        seconds_per_direction: Duration of one direction including the pause.
        pause_seconds: Grey frames before each sweep.
        bar_width_frames: Bar thickness in sweep frames.
        frame_rate_hz: Frame rate used to convert seconds to frames.
        frame_shape: Camera frame shape ``(height, width, 3)``.

    Returns:
        ``(frames, labels)``: a uint8 ``(time, height, width, 3)`` array and the
        direction label of every frame (``"grey"`` during pauses).
    """
    frames_per_direction = round(seconds_per_direction * frame_rate_hz)
    pause_frames = round(pause_seconds * frame_rate_hz)
    sweep_frames = frames_per_direction - pause_frames
    if sweep_frames < bar_width_frames + 2:
        raise ValueError("seconds_per_direction is too short for the bar to sweep")

    height, width, _ = frame_shape
    cycle: list[Frame] = []
    labels: list[str] = []
    for direction in METER_DIRECTIONS:
        edge = moving_edge_frames(
            direction, prestimulus_frames=0, sweep_frames=sweep_frames, hold_frames=0
        )
        is_bright = edge[..., 0] > GREY_LEVEL
        trailing = np.concatenate(
            [np.zeros((bar_width_frames, height, width), dtype=bool), is_bright]
        )[:sweep_frames]
        luminance = np.where(is_bright & ~trailing, 255, GREY_LEVEL).astype(np.uint8)
        cycle.append(grey_frames(pause_frames, frame_shape=frame_shape))
        cycle.append(np.repeat(luminance[..., None], 3, axis=-1))
        labels.extend(["grey"] * pause_frames + [direction] * sweep_frames)
    return np.concatenate(cycle, axis=0), labels


class SyntheticBarSource:
    """Cycle through the synthetic bar sweep, one frame per :meth:`read`."""

    nominal_fps = FRAME_RATE_HZ
    description = f"synthetic bar, {FRAME_RATE_HZ:.0f} Hz"

    def __init__(self, frame_shape: tuple[int, int, int] = DEFAULT_FRAME_SHAPE) -> None:
        self.frames, self.labels = synthetic_bar_cycle(frame_shape=frame_shape)
        self._index = 0

    def read(self) -> SourceFrame | None:
        """Return the next frame of the cycle (wraps around forever)."""
        index = self._index % len(self.frames)
        self._index += 1
        return SourceFrame(self.frames[index], self.labels[index])

    def close(self) -> None:
        """Nothing to release."""


class WebcamSource:
    """Read RGB frames from an OpenCV camera and fit them to the eye's shape."""

    description = "webcam"

    def __init__(
        self, camera_index: int, frame_shape: tuple[int, int, int] = DEFAULT_FRAME_SHAPE
    ) -> None:
        import cv2

        self._cv2 = cv2
        self.frame_shape = frame_shape
        self.capture = cv2.VideoCapture(camera_index)
        self.is_open = bool(self.capture.isOpened())
        self.nominal_fps = float(self.capture.get(cv2.CAP_PROP_FPS) or 0.0)
        if self.is_open:
            width = int(self.capture.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self.description = f"webcam {camera_index} ({width}x{height})"

    def read(self) -> SourceFrame | None:
        """Return the next camera frame in the eye's contract, or ``None``."""
        is_ok, bgr = self.capture.read()
        if not is_ok or bgr is None:
            return None
        rgb = np.ascontiguousarray(bgr[..., ::-1], dtype=np.uint8)
        return SourceFrame(center_crop_resize(rgb, self.frame_shape))

    def close(self) -> None:
        """Release the camera."""
        self.capture.release()


def open_source(
    source: str, camera_index: int, frame_shape: tuple[int, int, int]
) -> Any:
    """Open the requested frame source, falling back to synthetic without a camera."""
    if source == "webcam":
        webcam = WebcamSource(camera_index, frame_shape)
        if webcam.is_open:
            return webcam
        webcam.close()
        print(
            f"NOTICE: no camera opened at index {camera_index}; "
            "falling back to the synthetic bar."
        )
    return SyntheticBarSource(frame_shape)


# --------------------------------------------------------------------------
# Readout selection
# --------------------------------------------------------------------------


def parse_show_types(show: str | None) -> list[str]:
    """Split a ``--show`` value such as ``"R1,L1, Mi1"`` into unique cell types.

    Args:
        show: Comma-separated cell type names, or ``None``.

    Returns:
        The names in the order given, stripped, without repeats or blanks.
    """
    if not show:
        return []
    names: list[str] = []
    for raw_name in show.split(","):
        name = raw_name.strip()
        if name and name not in names:
            names.append(name)
    return names


def build_readout_names(*, hide_t5: bool = False) -> tuple[str, ...]:
    """Return the eye readouts feeding the direction meter: T4a-d, then T5a-d.

    Args:
        hide_t5: Drop the T5 row.

    Returns:
        The readout names to construct the eye with.
    """
    return T4_READOUTS + (() if hide_t5 else T5_READOUTS)


def build_panel_names(
    readout_names: Sequence[str], show: Sequence[str] = ()
) -> list[str]:
    """Return the hex panels to draw: the readouts, then ``--show`` extras.

    Extras that repeat a readout are dropped so every cell type has one panel.

    Args:
        readout_names: The eye's readouts.
        show: Extra cell types requested with ``--show``.

    Returns:
        Panel names in display order.
    """
    names = list(readout_names)
    names.extend(name for name in show if name not in names)
    return names


class UnknownCellTypesError(ValueError):
    """Raised when ``--show`` names cell types the model cannot draw."""


def resolve_cell_type_indices(eye: Any, names: Sequence[str]) -> dict[str, np.ndarray]:
    """Map extra cell types to their neuron indices in ``eye.state_activity``.

    Any of the model's cell types can be shown, not only the 34 output types
    the eye exposes as readouts, because the full network state is available
    after every ``encode``. Types must occupy the whole 721-column lattice in
    flyvis hex order (all but ``Lawf1``/``Lawf2``).

    Args:
        eye: A constructed ``FlyvisEye``.
        names: Cell types requested with ``--show``.

    Returns:
        ``{name: indices}`` with one index per hex column, in column order.

    Raises:
        UnknownCellTypesError: For names the model lacks or cannot draw.
    """
    nodes = eye.network.connectome.nodes
    available = sorted(str(name) for name in np.unique(nodes.type[:].astype(str)))
    unknown = [name for name in names if name not in available]
    if unknown:
        raise UnknownCellTypesError(
            f"--show: unknown cell types {unknown}; choose from {', '.join(available)}"
        )
    indices = {name: np.asarray(nodes.layer_index[name][:]) for name in names}
    partial = [
        name for name, index in indices.items() if len(index) != HEX_COLUMN_COUNT
    ]
    if partial:
        raise UnknownCellTypesError(
            f"--show: {partial} do not cover the {HEX_COLUMN_COUNT}-column lattice "
            "and cannot be drawn as hex maps"
        )
    return indices


# --------------------------------------------------------------------------
# Direction meter
# --------------------------------------------------------------------------


def compute_direction_meter(
    features: npt.NDArray[np.floating[Any]],
    readout_names: Sequence[str],
    *,
    statistic: str = "q95",
    baseline: dict[str, float] | None = None,
) -> dict[str, float]:
    """Reduce T4/T5 readouts to one non-negative value per motion direction.

    For each direction the columns of every readout whose subtype prefers it
    (``T4a`` and ``T5a`` for left, and so on) are pooled and summarised with
    ``statistic``; the resting ``baseline`` is subtracted and the result is
    clipped at zero. Readouts that are not T4/T5 subtypes are ignored.

    Args:
        features: ``(len(readout_names) * 721,)`` eye features.
        readout_names: Readouts in the order they are concatenated.
        statistic: ``"q95"`` (95th percentile over columns) or ``"mean"``.
        baseline: Resting meter values to subtract, per direction.

    Returns:
        ``{direction: value}`` for every direction in :data:`METER_DIRECTIONS`.
    """
    if statistic not in METER_STATISTICS:
        raise ValueError(
            f"statistic must be one of {METER_STATISTICS}, got {statistic!r}"
        )
    maps = split_readout_maps(np.asarray(features, dtype=np.float32), readout_names)
    meter: dict[str, float] = {}
    for direction in METER_DIRECTIONS:
        pooled = [
            maps[name]
            for name in readout_names
            if name[:2] in ("T4", "T5")
            and SUBTYPE_DIRECTIONS.get(name[2:]) == direction
        ]
        if not pooled:
            meter[direction] = 0.0
            continue
        values = np.concatenate(pooled)
        summary = float(
            np.quantile(values, 0.95) if statistic == "q95" else values.mean()
        )
        if baseline is not None:
            summary -= baseline.get(direction, 0.0)
        meter[direction] = max(summary, 0.0)
    return meter


def find_winning_direction(
    meter: dict[str, float], minimum: float = METER_PEAK_FLOOR
) -> str | None:
    """Return the direction with the largest meter value; ``None`` if all are quiet."""
    direction = max(METER_DIRECTIONS, key=lambda name: meter.get(name, 0.0))
    return direction if meter.get(direction, 0.0) > minimum else None


def format_meter(meter: dict[str, float], *, compact: bool = False) -> str:
    """Format the meter as ``left=0.12 right=1.30 up=0.00 down=0.04 -> right``.

    Args:
        meter: Direction meter values.
        compact: Use ``L 0.12  R 1.30  U 0.00  D 0.04`` for narrow on-screen text.

    Returns:
        The formatted line.
    """
    if compact:
        return "  ".join(
            f"{name[0].upper()} {meter[name]:.2f}" for name in METER_DIRECTIONS
        )
    values = " ".join(f"{name}={meter[name]:.3f}" for name in METER_DIRECTIONS)
    return f"{values} -> {find_winning_direction(meter) or 'none'}"


# --------------------------------------------------------------------------
# Display
# --------------------------------------------------------------------------


@dataclass
class LoopStats:
    """Running timing statistics shown in the overlay and printed at the end."""

    display_fps: float = 0.0
    camera_fps: float = 0.0
    eye_latency_ms: float = 0.0
    eye_latencies_ms: list[float] = field(default_factory=list)
    loop_periods_s: list[float] = field(default_factory=list)

    def record_eye_latency(self, latency_ms: float) -> None:
        """Track one ``encode`` latency (exponential average for the overlay)."""
        self.eye_latencies_ms.append(latency_ms)
        self.eye_latency_ms = _ema(self.eye_latency_ms, latency_ms)

    def record_loop_period(self, period_s: float) -> None:
        """Track one display loop period."""
        self.loop_periods_s.append(period_s)
        if period_s > 0:
            self.display_fps = _ema(self.display_fps, 1.0 / period_s)

    def record_camera_period(self, period_s: float) -> None:
        """Track the interval between two source frames."""
        if period_s > 0:
            self.camera_fps = _ema(self.camera_fps, 1.0 / period_s)


def _ema(current: float, sample: float) -> float:
    return (
        sample if current == 0.0 else (1 - EMA_WEIGHT) * current + EMA_WEIGHT * sample
    )


class LiveView:
    """matplotlib figure with blitted camera frame, retina, hex maps, and meter.

    Layout: the left block holds the camera frame, the retina next to it, and
    the direction meter below; the right block holds the readout hex maps in
    rows of :data:`MAPS_PER_ROW`, so the pipeline reads camera → retina → cell
    types from left to right. With one readout row (``--hide-t5`` and nothing
    shown) the maps span both rows.

    Args:
        panel_names: Cell types to draw as hex maps (readouts first).
        color_limit: Symmetric colour limit for the maps in activity units.
        display: Open an interactive window. ``False`` uses the Agg backend and
            only supports :meth:`save`.
        show_retina: Draw the hex-resampled luminance entering the network.
        relative_panels: Panels whose values are deviations from rest; their
            titles say so.
    """

    def __init__(
        self,
        panel_names: Sequence[str],
        color_limit: float,
        display: bool,
        *,
        show_retina: bool = True,
        relative_panels: Sequence[str] = (),
    ) -> None:
        import matplotlib

        if not display:
            matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        self.display = display
        self.show_retina = show_retina
        self.paused = False
        self.reset_requested = False
        self.is_open = True
        self._needs_background = True
        self._meter_peak = METER_PEAK_FLOOR
        self._plt = plt
        self._raster = HexRaster()

        readout_rows = [
            list(panel_names[start : start + MAPS_PER_ROW])
            for start in range(0, len(panel_names), MAPS_PER_ROW)
        ]
        row_count = max(2, len(readout_rows))
        left_columns = 2
        self.figure = plt.figure(
            figsize=(12, min(8.0, 2.0 + 2.25 * row_count)), facecolor="white"
        )
        grid = self.figure.add_gridspec(
            row_count,
            left_columns + MAPS_PER_ROW,
            left=0.03,
            right=0.99,
            top=0.85 if row_count == 2 else 0.9,
            bottom=0.1 if row_count == 2 else 0.06,
            wspace=0.15,
            hspace=0.35,
        )
        camera_axes = self.figure.add_subplot(
            grid[0, 0] if show_retina else grid[0, :2]
        )
        retina_axes = self.figure.add_subplot(grid[0, 1]) if show_retina else None
        meter_axes = self.figure.add_subplot(grid[1, :left_columns])
        map_axes = []
        map_names = []
        for row_index, names in enumerate(readout_rows):
            for column_index, name in enumerate(names):
                column = left_columns + column_index
                cell = (
                    grid[:, column]
                    if len(readout_rows) == 1
                    else grid[row_index, column]
                )
                map_axes.append(self.figure.add_subplot(cell))
                map_names.append(name)

        blank = np.full(DEFAULT_FRAME_SHAPE, GREY_LEVEL, dtype=np.uint8)
        self._image = camera_axes.imshow(blank, animated=display)
        camera_axes.set_title("eye input (96x96)", fontsize=10)
        camera_axes.set_xticks([])
        camera_axes.set_yticks([])

        self._retina = None
        if retina_axes is not None:
            self._retina = self._raster.imshow(
                retina_axes,
                np.full(HEX_COLUMN_COUNT, 0.5, dtype=np.float32),
                cmap="gray",
                vmin=0.0,
                vmax=1.0,
                interpolation="nearest",
            )
            self._retina.set_animated(display)
            retina_axes.set_title("retina (721 columns)", fontsize=10)

        self._bars = meter_axes.bar(
            METER_DIRECTIONS, [0.0] * len(METER_DIRECTIONS), color="#4c72b0"
        )
        for bar in self._bars:
            bar.set_animated(display)
        meter_axes.set_ylim(0, 1.05)
        meter_axes.set_yticks([])
        meter_families = "/".join(
            family
            for family in ("T4", "T5")
            if any(name.startswith(family) for name in panel_names)
        )
        meter_axes.set_title(
            f"direction meter ({meter_families}, relative to peak)", fontsize=10
        )
        self._meter_text = meter_axes.text(
            0.5,
            0.95,
            "",
            transform=meter_axes.transAxes,
            ha="center",
            va="top",
            fontsize=8,
            animated=display,
        )

        self._maps = []
        for axes, name in zip(map_axes, map_names):
            image = self._raster.imshow(
                axes,
                np.zeros(HEX_COLUMN_COUNT, dtype=np.float32),
                cmap="coolwarm",
                vmin=-color_limit,
                vmax=color_limit,
                interpolation="nearest",
            )
            image.set_animated(display)
            title = f"{name} − rest" if name in relative_panels else name
            axes.set_title(title, fontsize=10, fontweight="bold")
            self._maps.append(image)
        self.figure.colorbar(
            self._maps[0], ax=map_axes, label="activity (a.u.)", shrink=0.6, pad=0.02
        )
        self.panel_names = ["camera"] + (["retina"] if show_retina else []) + map_names

        self._overlay = self.figure.text(
            0.03, 0.97, "", fontsize=10, family="monospace", va="top", animated=display
        )
        self.figure.text(
            0.99,
            0.015,
            "space pause/resume · r reset eye · q/Esc quit",
            fontsize=9,
            ha="right",
            va="bottom",
            color="#555555",
        )
        self._animated = [
            self._image,
            self._meter_text,
            self._overlay,
            *self._bars,
            *self._maps,
        ]
        if self._retina is not None:
            self._animated.append(self._retina)

        if display:
            self.figure.canvas.mpl_connect("key_press_event", self._on_key)
            self.figure.canvas.mpl_connect("close_event", self._on_close)
            self.figure.canvas.mpl_connect("resize_event", self._on_resize)
            plt.show(block=False)

    def _on_key(self, event: Any) -> None:
        if event.key == " ":
            self.paused = not self.paused
        elif event.key == "r":
            self.reset_requested = True
        elif event.key in ("q", "escape"):
            self.is_open = False

    def _on_close(self, _event: Any) -> None:
        self.is_open = False

    def _on_resize(self, _event: Any) -> None:
        self._needs_background = True

    def poll(self) -> None:
        """Process pending window events (key presses, close, resize)."""
        if self.display:
            self.figure.canvas.flush_events()

    def update(
        self,
        frame: Frame,
        retina: npt.NDArray[np.floating[Any]] | None,
        maps: dict[str, npt.NDArray[np.floating[Any]]],
        meter: dict[str, float],
        overlay_text: str,
    ) -> None:
        """Push new data into the artists and redraw only them.

        Args:
            frame: The 96x96x3 frame the eye received.
            retina: ``(721,)`` resampled luminance in ``[0, 1]``, or ``None``.
            maps: Readout maps in panel order (see ``split_readout_maps``).
            meter: Direction meter values.
            overlay_text: Status line for the top-left overlay.
        """
        self._image.set_data(frame)
        if self._retina is not None and retina is not None:
            self._retina.set_data(self._raster.render(retina))
        for image, values in zip(self._maps, maps.values()):
            image.set_data(self._raster.render(values))
        self._meter_peak = max(
            self._meter_peak * METER_PEAK_DECAY, max(meter.values()), METER_PEAK_FLOOR
        )
        winner = find_winning_direction(meter)
        for bar, direction in zip(self._bars, METER_DIRECTIONS):
            bar.set_height(meter[direction] / self._meter_peak)
            bar.set_color("#dd8452" if direction == winner else "#4c72b0")
        self._meter_text.set_text(format_meter(meter, compact=True))
        self._overlay.set_text(overlay_text)
        if self.display:
            self._blit()

    def _blit(self) -> None:
        # Typed as Any: blitting lives on the concrete Agg-based canvases only.
        canvas: Any = self.figure.canvas
        if self._needs_background:
            canvas.draw()
            self._background = canvas.copy_from_bbox(self.figure.bbox)
            self._needs_background = False
        canvas.restore_region(self._background)
        for artist in self._animated:
            self.figure.draw_artist(artist)
        canvas.blit(self.figure.bbox)
        canvas.flush_events()

    def save(self, path: Path) -> None:
        """Write the current figure (including blitted artists) to ``path``."""
        for artist in self._animated:
            artist.set_animated(False)
        try:
            self.figure.savefig(path, dpi=100)
        finally:
            for artist in self._animated:
                artist.set_animated(self.display)
        self._needs_background = True

    def close(self) -> None:
        """Close the figure."""
        self._plt.close(self.figure)


# --------------------------------------------------------------------------
# Main loop
# --------------------------------------------------------------------------


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--source", choices=("webcam", "synthetic"), default="webcam")
    parser.add_argument("--camera-index", type=int, default=0)
    parser.add_argument(
        "--fps-cap",
        type=float,
        default=DEFAULT_FPS_CAP,
        help="Upper bound on frames per second (0 = uncapped).",
    )
    parser.add_argument(
        "--frames", type=int, default=0, help="Stop after N frames (0 = until quit)."
    )
    parser.add_argument("--no-display", action="store_true", help="No window.")
    parser.add_argument(
        "--print-every",
        type=int,
        default=10,
        help="Print the meter every N frames when --no-display is set (0 = never).",
    )
    parser.add_argument(
        "--save-dir", type=Path, default=None, help="Save PNG snapshots here."
    )
    parser.add_argument("--save-every", type=int, default=50, help="Snapshot period.")
    parser.add_argument("--meter-statistic", choices=METER_STATISTICS, default="q95")
    parser.add_argument("--color-limit", type=float, default=DEFAULT_COLOR_LIMIT)
    parser.add_argument(
        "--show",
        default=None,
        metavar="TYPE[,TYPE...]",
        help="Extra flyvis output cell types to draw, e.g. R1,L1,Mi1,Tm3.",
    )
    parser.add_argument(
        "--hide-t5", action="store_true", help="Drop the T5a-d row (compact view)."
    )
    parser.add_argument(
        "--no-retina",
        action="store_true",
        help="Hide the retina panel (the 721-column luminance entering the eye).",
    )
    parser.add_argument("--checkpoint", default=None, help="flyvis checkpoint.")
    return parser.parse_args(argv)


def _try_load_eye(checkpoint: str | None, readouts: Sequence[str]) -> Any:
    """Return a FlyvisEye, or ``None`` after printing why the demo is skipped."""
    from fly_driver.eyes.flyvis_eye import (
        DEFAULT_CHECKPOINT,
        FlyvisEye,
        FlyvisNotInstalledError,
        resolve_checkpoint_dir,
    )

    checkpoint = checkpoint or DEFAULT_CHECKPOINT
    try:
        resolve_checkpoint_dir(checkpoint)
    except FlyvisNotInstalledError:
        if sys.version_info >= (3, 13):
            print("SKIP: flyvis needs Python 3.9-3.12.")
        else:
            print(
                "SKIP: flyvis is not installed; run "
                "`python -m pip install -r requirements-flyvis.txt`."
            )
        return None
    except FileNotFoundError as error:
        print(f"SKIP: {error}")
        return None
    return FlyvisEye(checkpoint=checkpoint, readouts=readouts)


def _has_opencv() -> bool:
    try:
        import cv2  # noqa: F401
    except ImportError:
        return False
    return True


def _measure_baseline(
    eye: Any, statistic: str
) -> tuple[dict[str, float], npt.NDArray[np.float32]]:
    """Reset the eye and read the resting meter and network state from grey.

    Returns:
        The resting direction meter (subtracted from live values) and the
        resting activity of every neuron (subtracted from ``--show`` panels).
    """
    eye.reset()
    grey = grey_frames(1, frame_shape=eye.frame_shape)[0]
    features = eye.encode(grey)
    resting_state = np.array(eye.state_activity, dtype=np.float32)
    return (
        compute_direction_meter(features, eye.readout_names, statistic=statistic),
        resting_state,
    )


def _format_overlay(
    stats: LoopStats,
    source: Any,
    stimulus_label: str,
    has_display: bool,
    eye_rate_hz: float,
) -> str:
    camera_note = (
        f"camera ~{stats.camera_fps:.0f} fps"
        if isinstance(source, WebcamSource)
        else f"synthetic bar {stats.camera_fps:.0f} fps"
    )
    rate_label = "display" if has_display else "loop"
    overlay = (
        f"{rate_label} {stats.display_fps:5.1f} fps"
        f" | eye {stats.eye_latency_ms:5.1f} ms"
        f" | {camera_note}, eye stepped at {eye_rate_hz:.0f} Hz"
    )
    if stimulus_label:
        overlay += f" | stimulus: {stimulus_label}"
    return overlay


def _summarise(
    stats: LoopStats, agreement: dict[str, list[bool]], rate_label: str
) -> None:
    if stats.loop_periods_s:
        mean_fps = 1.0 / statistics.fmean(stats.loop_periods_s)
        print(
            f"{rate_label}: {mean_fps:.1f} fps mean over "
            f"{len(stats.loop_periods_s)} frames"
        )
    if stats.eye_latencies_ms:
        latencies = stats.eye_latencies_ms
        print(
            f"eye latency: median {statistics.median(latencies):.2f} ms, "
            f"p95 {float(np.percentile(latencies, 95)):.2f} ms, "
            f"max {max(latencies):.2f} ms (budget {1000 / FRAME_RATE_HZ:.0f} ms)"
        )
    for direction in METER_DIRECTIONS:
        votes = agreement.get(direction, [])
        if votes:
            print(
                f"meter agreement while bar moves {direction}: "
                f"{sum(votes) / len(votes):.2f} ({len(votes)} frames)"
            )


def main(argv: Sequence[str] | None = None) -> int:
    """Run the live demo, or skip successfully when the optional stack is absent."""
    args = _parse_args(argv)
    if args.source == "webcam" and not _has_opencv():
        print(
            "SKIP: opencv-python is not installed (needed for --source webcam); "
            "install it or run with --source synthetic."
        )
        return 0
    eye = _try_load_eye(args.checkpoint, build_readout_names(hide_t5=args.hide_t5))
    if eye is None:
        return 0
    show_types = [
        name for name in parse_show_types(args.show) if name not in eye.readout_names
    ]
    try:
        extra_indices = resolve_cell_type_indices(eye, show_types)
    except UnknownCellTypesError as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2
    panel_names = build_panel_names(eye.readout_names, show_types)
    print(f"checkpoint: {eye.checkpoint_dir}; device: {eye.device}")
    print(f"readouts: {', '.join(eye.readout_names)}")

    source = open_source(args.source, args.camera_index, eye.frame_shape)
    print(f"source: {source.description}")
    print(f"eye stepped at {eye.frame_rate_hz:.0f} Hz (one frame = one step)")
    baseline, resting_state = _measure_baseline(eye, args.meter_statistic)
    print(f"resting meter (subtracted): {format_meter(baseline)}")

    view: LiveView | None = None
    if not args.no_display or args.save_dir is not None:
        view = LiveView(
            panel_names,
            args.color_limit,
            display=not args.no_display,
            show_retina=not args.no_retina,
            relative_panels=show_types,
        )
        print(f"panels: {', '.join(view.panel_names)}")
    if args.save_dir is not None:
        args.save_dir.mkdir(parents=True, exist_ok=True)

    stats = LoopStats()
    agreement: dict[str, list[bool]] = {}
    frame_count = 0
    last_loop = time.perf_counter()
    last_read: float | None = None
    minimum_period = 1.0 / args.fps_cap if args.fps_cap > 0 else 0.0
    try:
        while (view is None or view.is_open) and (
            args.frames <= 0 or frame_count < args.frames
        ):
            if view is not None:
                view.poll()
                if view.reset_requested:
                    baseline, resting_state = _measure_baseline(
                        eye, args.meter_statistic
                    )
                    view.reset_requested = False
                    print("eye state reset")
                if view.paused:
                    time.sleep(0.02)
                    last_loop = time.perf_counter()
                    continue

            source_frame = source.read()
            if source_frame is None:
                print("camera returned no frame; stopping")
                break
            now = time.perf_counter()
            if last_read is not None:
                stats.record_camera_period(now - last_read)
            last_read = now

            encode_start = time.perf_counter()
            features = eye.encode(source_frame.frame)
            stats.record_eye_latency((time.perf_counter() - encode_start) * 1000)
            meter = compute_direction_meter(
                features,
                eye.readout_names,
                statistic=args.meter_statistic,
                baseline=baseline,
            )
            winner = find_winning_direction(meter)
            if source_frame.label in METER_DIRECTIONS:
                agreement.setdefault(source_frame.label, []).append(
                    winner == source_frame.label
                )

            if view is not None:
                # The retina is what the network integrates: the same resampler
                # call encode() made, repeated here (~1.5 ms) purely for display.
                retina = (
                    eye.resampler.frame(source_frame.frame)[0, 0, 0].numpy()
                    if view.show_retina
                    else None
                )
                maps = split_readout_maps(features, eye.readout_names)
                if extra_indices:
                    state = eye.state_activity
                    for name, indices in extra_indices.items():
                        maps[name] = state[indices] - resting_state[indices]
                view.update(
                    source_frame.frame,
                    retina,
                    {name: maps[name] for name in panel_names},
                    meter,
                    _format_overlay(
                        stats,
                        source,
                        source_frame.label,
                        view.display,
                        eye.frame_rate_hz,
                    ),
                )
                if args.save_dir is not None and frame_count % args.save_every == 0:
                    path = args.save_dir / f"flyvis_eye_live_{frame_count:05d}.png"
                    view.save(path)
                    print(f"saved {path}")
            if (
                args.no_display
                and args.print_every > 0
                and frame_count % args.print_every == 0
            ):
                label = f" stim={source_frame.label:<5}" if source_frame.label else ""
                print(
                    f"frame {frame_count:05d}{label} "
                    f"eye {stats.eye_latencies_ms[-1]:5.2f} ms "
                    f"meter {format_meter(meter)}"
                )

            frame_count += 1
            elapsed = time.perf_counter() - last_loop
            if minimum_period > elapsed:
                time.sleep(minimum_period - elapsed)
            now = time.perf_counter()
            stats.record_loop_period(now - last_loop)
            last_loop = now
    except KeyboardInterrupt:
        print("interrupted")
    finally:
        source.close()
        if view is not None:
            view.close()

    print(f"frames: {frame_count}")
    _summarise(stats, agreement, "display" if not args.no_display else "loop")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
