#!/usr/bin/env python3
"""Watch the frozen flyvis eye respond to a webcam (or a synthetic bar) live.

One window shows the pipeline the simulated fly sees through, in order:

**Camera-derived panels** (pixels, no neurons involved):

- ``camera``: the 96x96 frame the eye receives.
- ``retina`` (``--show-retina``, off by default): the 721-column hex-resampled
  luminance that ``HexResampler`` feeds into the network.

**Neural activity panels** (read from the network after every step):

- ``photoreceptors R1-R6``: the first neural stage, mean R1-R6 activity per
  column relative to rest. This is what the brain actually receives.
- ``motion percept (T4/T5)``: the eight T4/T5 direction channels fused into one
  picture: per column a motion vector (right − left, up − down) drawn as hue =
  direction, brightness = strength (relative to a running peak). It is the
  5,768 T4/T5 numbers the driving policy will get, as one image; still scenes
  are dark.
- ``T4a-d`` / ``T5a-d``: the individual motion-detector maps
  (``FlyvisEye.encode`` readouts, streaming with state carried between frames).
- ``--show R1,L1,Mi1,Tm3``: any other cell type, from
  ``FlyvisEye.state_activity`` relative to rest.

Plus a direction meter (left/right/up/down from T4/T5 a/b/c/d). Wave a hand
across the camera and the bar for that direction jumps. ``--hide-t5`` drops the
T5 row.

**Display choice.** matplotlib with blitting, not pygame. matplotlib is already
used by every other script in the repo, so the live demo adds only
``opencv-python``; the hexagonal map drawing is shared with
``scripts/flyvis_eye_demo.py`` through ``fly_driver.analysis.hex_plots``; and
the same figure renders headlessly on the Agg backend for tests and PNG/GIF
evidence. Blitting redraws only the images, the meter bars, and the overlay
text; the maps are painted as small images (``HexRaster``) instead of 8 x 721
scatter markers. On a 4-core CPU VM the window runs at ~24 fps with the eye at
~9 ms per frame, close to a webcam's ~30 fps.

**HiDPI-safe layout.** The figure uses a constrained-layout ``GridSpec`` (no
hand-placed axes), the status overlay lives in the figure's own suptitle row,
and every font size is a fraction of the figure width, recomputed when the
window is resized, so a Retina Mac or a shrunk window cannot make titles collide.
The default size fits a 1440x900 logical screen with the toolbar;
``--figsize W,H`` and ``--scale`` override it.

**Timing.** Every frame is one 20 ms Euler step of the optic lobe regardless of
how fast frames arrive, so a ~30 fps webcam plays back through the eye at
50 Hz, about 1.7x faster than real time. That is fine for a demo and is stated
on screen; the synthetic source is paced to 50 Hz so it is real time.

Headless check (no window, prints the direction meter and motion percept):

    python scripts/flyvis_eye_live.py --source synthetic --frames 200 --no-display

The script exits 0 with a ``SKIP`` message when flyvis, its checkpoint, or
(for the webcam) OpenCV is missing; a missing camera falls back to synthetic.
"""

from __future__ import annotations

import argparse
import contextlib
import statistics
import sys
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import torch
from torch.nn import functional as F

# The flyvis virtualenv installs requirements-flyvis.txt, not the fly_driver
# package (docs/running-the-stacks.md), so like the other flyvis scripts this one
# puts the repo root on sys.path before importing it.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from fly_driver.analysis.hex_plots import HexRaster, split_readout_maps  # noqa: E402
from fly_driver.eyes.hex_resampler import (  # noqa: E402
    DEFAULT_FRAME_SHAPE,
    HEX_COLUMN_COUNT,
)
from fly_driver.eyes.stimuli import (  # noqa: E402
    FRAME_RATE_HZ,
    GREY_LEVEL,
    grey_frames,
    moving_edge_frames,
)

Frame = npt.NDArray[np.uint8]
FloatArray = npt.NDArray[np.floating[Any]]

T4_READOUTS = ("T4a", "T4b", "T4c", "T4d")
T5_READOUTS = ("T5a", "T5b", "T5c", "T5d")
PHOTORECEPTOR_TYPES = ("R1", "R2", "R3", "R4", "R5", "R6")
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
PEAK_DECAY = 0.995
PEAK_FLOOR = 0.05
# Layout: sizes are in inches at the reference width; fonts scale with width.
REFERENCE_WIDTH_IN = 12.5
DEFAULT_SCREEN_LOGICAL_PX = (1440, 900)
WINDOW_CHROME_PX = (40, 150)
LEFT_COLUMNS = 3
BASE_FONT_PT = {"title": 9.0, "overlay": 9.0, "hint": 8.0, "meter": 8.0, "tick": 7.0}
DEFAULT_FIGURE_DPI = 100.0
# `import flyvis` restyles matplotlib for 300 dpi paper figures (figure.dpi,
# savefig.*, 5 pt fonts, hidden spines). A 12 in figure at 300 dpi is wider than
# any laptop screen, the window manager shrinks the axes and the point-sized
# fonts do not follow, which is the overlap seen on macOS. Restore these keys.
RC_KEYS_RESET_TO_DEFAULT = (
    "figure.dpi",
    "savefig.dpi",
    "savefig.bbox",
    "savefig.format",
    "font.size",
    "figure.titlesize",
    "axes.titlesize",
    "axes.labelsize",
    "axes.linewidth",
    "axes.spines.right",
    "axes.spines.top",
    "xtick.labelsize",
    "ytick.labelsize",
    "xtick.major.width",
    "ytick.major.width",
    "image.interpolation",
    "image.resample",
)


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
    return resized.round().clamp(0, 255).to(torch.uint8)[0].permute(1, 2, 0).numpy().copy()


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


def open_source(source: str, camera_index: int, frame_shape: tuple[int, int, int]) -> Any:
    """Open the requested frame source, falling back to synthetic without a camera."""
    if source == "webcam":
        webcam = WebcamSource(camera_index, frame_shape)
        if webcam.is_open:
            return webcam
        webcam.close()
        print(
            f"NOTICE: no camera opened at index {camera_index}; falling back to the synthetic bar."
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


def build_panel_names(readout_names: Sequence[str], show: Sequence[str] = ()) -> list[str]:
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
    """Map cell types to their neuron indices in ``eye.state_activity``.

    Any of the model's cell types can be shown, not only the 34 output types
    the eye exposes as readouts, because the full network state is available
    after every ``encode``. Types must occupy the whole 721-column lattice in
    flyvis hex order (all but ``Lawf1``/``Lawf2``).

    Args:
        eye: A constructed ``FlyvisEye``.
        names: Cell types to resolve.

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
    partial = [name for name, index in indices.items() if len(index) != HEX_COLUMN_COUNT]
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
    features: FloatArray,
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
        raise ValueError(f"statistic must be one of {METER_STATISTICS}, got {statistic!r}")
    maps = split_readout_maps(np.asarray(features, dtype=np.float32), readout_names)
    meter: dict[str, float] = {}
    for direction in METER_DIRECTIONS:
        pooled = [
            maps[name]
            for name in readout_names
            if name[:2] in ("T4", "T5") and SUBTYPE_DIRECTIONS.get(name[2:]) == direction
        ]
        if not pooled:
            meter[direction] = 0.0
            continue
        values = np.concatenate(pooled)
        summary = float(np.quantile(values, 0.95) if statistic == "q95" else values.mean())
        if baseline is not None:
            summary -= baseline.get(direction, 0.0)
        meter[direction] = max(summary, 0.0)
    return meter


def find_winning_direction(meter: dict[str, float], minimum: float = PEAK_FLOOR) -> str | None:
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
        return "  ".join(f"{name[0].upper()} {meter[name]:.2f}" for name in METER_DIRECTIONS)
    values = " ".join(f"{name}={meter[name]:.3f}" for name in METER_DIRECTIONS)
    return f"{values} -> {find_winning_direction(meter) or 'none'}"


# --------------------------------------------------------------------------
# Neural view: photoreceptors and the fused motion percept
# --------------------------------------------------------------------------


@dataclass
class NeuralIndices:
    """Neuron indices into ``FlyvisEye.state_activity`` for the neural panels."""

    photoreceptors: list[np.ndarray]
    motion_channels: dict[str, list[np.ndarray]]

    @classmethod
    def resolve(cls, eye: Any) -> NeuralIndices:
        """Look up R1-R6 and every T4/T5 subtype in the eye's connectome."""
        names = list(PHOTORECEPTOR_TYPES) + list(T4_READOUTS) + list(T5_READOUTS)
        indices = resolve_cell_type_indices(eye, names)
        channels: dict[str, list[np.ndarray]] = {direction: [] for direction in METER_DIRECTIONS}
        for name in T4_READOUTS + T5_READOUTS:
            channels[SUBTYPE_DIRECTIONS[name[2:]]].append(indices[name])
        return cls([indices[name] for name in PHOTORECEPTOR_TYPES], channels)


def photoreceptor_map(
    state: FloatArray, resting_state: FloatArray, indices: Sequence[np.ndarray]
) -> FloatArray:
    """Return mean R1-R6 activity per column relative to rest, ``(721,)``."""
    return np.mean([state[index] - resting_state[index] for index in indices], axis=0)


def motion_channels(
    state: FloatArray,
    resting_state: FloatArray,
    channels: Mapping[str, Sequence[np.ndarray]],
) -> dict[str, FloatArray]:
    """Return rectified T4+T5 activity above rest per direction, ``(721,)`` each."""
    return {
        direction: np.sum(
            [np.maximum(state[index] - resting_state[index], 0.0) for index in indices],
            axis=0,
        )
        for direction, indices in channels.items()
    }


def hsv_to_rgb(hue: FloatArray, saturation: FloatArray, value: FloatArray) -> FloatArray:
    """Convert HSV arrays in ``[0, 1]`` to an RGB array with a trailing axis of 3."""
    hue6 = (np.asarray(hue) % 1.0) * 6.0
    sector = np.floor(hue6).astype(int) % 6
    fraction = hue6 - np.floor(hue6)
    low = value * (1 - saturation)
    falling = value * (1 - fraction * saturation)
    rising = value * (1 - (1 - fraction) * saturation)
    red = np.choose(sector, [value, falling, low, low, rising, value])
    green = np.choose(sector, [rising, value, value, falling, low, low])
    blue = np.choose(sector, [low, low, rising, value, value, falling])
    return np.stack([red, green, blue], axis=-1).astype(np.float32)


def compute_motion_percept(
    channels: dict[str, FloatArray], peak: float
) -> tuple[FloatArray, FloatArray]:
    """Fuse four direction channels into per-column motion vectors and colours.

    Per column ``x = right - left`` and ``y = up - down``; hue encodes the
    vector angle (right = red, up = yellow-green, left = cyan, down = violet)
    and brightness the magnitude relative to ``peak``, so still columns are
    black.

    Args:
        channels: ``{"left" | "right" | "up" | "down": (721,) non-negative}``.
        peak: Magnitude drawn at full brightness.

    Returns:
        ``(rgb, vectors)``: ``(721, 3)`` colours in ``[0, 1]`` and ``(721, 2)``
        ``(x, y)`` motion vectors.
    """
    x = np.asarray(channels["right"]) - np.asarray(channels["left"])
    y = np.asarray(channels["up"]) - np.asarray(channels["down"])
    magnitude = np.hypot(x, y)
    hue = (np.arctan2(y, x) / (2 * np.pi)) % 1.0
    value = np.clip(magnitude / max(peak, 1e-6), 0.0, 1.0)
    rgb = hsv_to_rgb(hue, np.ones_like(hue), value)
    return rgb, np.stack([x, y], axis=-1)


def summarise_motion(vectors: FloatArray) -> tuple[float, float]:
    """Return the mean motion vector as ``(angle_degrees, magnitude)``.

    Angles follow the hue wheel: 0 = right, 90 = up, 180 = left, 270 = down.
    """
    mean_x, mean_y = np.mean(vectors, axis=0)
    angle = float(np.degrees(np.arctan2(mean_y, mean_x)) % 360.0)
    return angle, float(np.hypot(mean_x, mean_y))


def hue_wheel_image(size: int = 64) -> FloatArray:
    """Return an ``(size, size, 4)`` RGBA hue wheel legend (transparent outside)."""
    coordinates = (np.arange(size) + 0.5) / size * 2 - 1
    x, y = np.meshgrid(coordinates, -coordinates)
    radius = np.hypot(x, y)
    hue = (np.arctan2(y, x) / (2 * np.pi)) % 1.0
    saturation = np.clip(radius, 0.0, 1.0)
    rgb = hsv_to_rgb(hue, saturation, np.ones_like(hue))
    alpha = (radius <= 1.0).astype(np.float32)
    return np.concatenate([rgb, alpha[..., None]], axis=-1)


class RunningPeak:
    """Slowly decaying maximum used to normalise panels to their recent range."""

    def __init__(self, floor: float = PEAK_FLOOR, decay: float = PEAK_DECAY) -> None:
        self.value = floor
        self._floor = floor
        self._decay = decay

    def update(self, sample: float) -> float:
        """Fold one sample in and return the current peak."""
        self.value = max(self.value * self._decay, float(sample), self._floor)
        return self.value


# --------------------------------------------------------------------------
# Layout helpers
# --------------------------------------------------------------------------


def parse_figsize(text: str | None) -> tuple[float, float] | None:
    """Parse ``"W,H"`` or ``"WxH"`` inches into a tuple, or ``None`` if absent."""
    if not text:
        return None
    parts = text.replace("x", ",").split(",")
    if len(parts) != 2:
        raise ValueError(f"--figsize must be W,H in inches, got {text!r}")
    width, height = (float(part) for part in parts)
    if width <= 0 or height <= 0:
        raise ValueError("--figsize values must be positive")
    return width, height


def fit_figure_size(
    row_count: int,
    dpi: float,
    screen_px: tuple[int, int] = DEFAULT_SCREEN_LOGICAL_PX,
    requested: tuple[float, float] | None = None,
) -> tuple[float, float]:
    """Return a figure size in inches that fits the screen with window chrome.

    The natural size is :data:`REFERENCE_WIDTH_IN` wide and grows with the
    number of panel rows; it is shrunk uniformly until ``width * dpi`` and
    ``height * dpi`` fit inside ``screen_px`` minus :data:`WINDOW_CHROME_PX`.

    Args:
        row_count: Panel grid rows (at least 2).
        dpi: Figure dpi (logical pixels per inch).
        screen_px: Logical screen size in pixels.
        requested: Explicit ``(width, height)`` from ``--figsize``; returned as
            given.

    Returns:
        ``(width_in, height_in)``.
    """
    if requested is not None:
        return requested
    width = REFERENCE_WIDTH_IN
    height = min(8.5, 2.2 + 2.3 * max(row_count, 2))
    max_width = (screen_px[0] - WINDOW_CHROME_PX[0]) / dpi
    max_height = (screen_px[1] - WINDOW_CHROME_PX[1]) / dpi
    shrink = min(1.0, max_width / width, max_height / height)
    return round(width * shrink, 2), round(height * shrink, 2)


def font_scale(width_in: float, user_scale: float = 1.0) -> float:
    """Return the font multiplier for a figure ``width_in`` inches wide.

    Fonts are a fixed fraction of the figure width, so they shrink with the
    window instead of swallowing the panels.
    """
    return user_scale * width_in / REFERENCE_WIDTH_IN


def _screen_logical_size(figure: Any) -> tuple[int, int]:
    """Best-effort logical screen size from the window, capped at the default."""
    width, height = DEFAULT_SCREEN_LOGICAL_PX
    manager = getattr(figure.canvas, "manager", None)
    window: Any = getattr(manager, "window", None)
    # Any toolkit failure (no window yet, headless Qt) keeps the default.
    with contextlib.suppress(Exception):
        if hasattr(window, "screen"):  # Qt
            geometry = window.screen().availableGeometry()
            width, height = geometry.width(), geometry.height()
        elif hasattr(window, "winfo_screenwidth"):  # Tk
            width, height = window.winfo_screenwidth(), window.winfo_screenheight()
    return min(width, DEFAULT_SCREEN_LOGICAL_PX[0]), min(height, DEFAULT_SCREEN_LOGICAL_PX[1])


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
    return sample if current == 0.0 else (1 - EMA_WEIGHT) * current + EMA_WEIGHT * sample


@dataclass
class ViewFrame:
    """Everything :meth:`LiveView.update` draws for one frame."""

    frame: Frame
    photoreceptors: FloatArray
    motion_rgb: FloatArray
    maps: dict[str, FloatArray]
    meter: dict[str, float]
    overlay_text: str
    retina: FloatArray | None = None


class LiveView:
    """Blitted matplotlib figure: camera, neural panels, hex maps, meter, overlay.

    The grid is a constrained-layout ``GridSpec``. Row 0 of the left block reads
    camera → photoreceptors → motion percept; row 1 holds the optional retina
    and the direction meter; the right block holds the hex maps in rows of
    :data:`MAPS_PER_ROW` (a single row spans both grid rows). The overlay is
    the figure suptitle and the key hint its supxlabel, so constrained layout
    reserves room for them. Font sizes are a fraction of the figure width and
    are recomputed on every resize.

    Args:
        panel_names: Cell types to draw as hex maps on the right.
        color_limit: Symmetric colour limit for the maps in activity units.
        display: Open an interactive window. ``False`` uses the Agg backend and
            only supports :meth:`save`.
        show_retina: Also draw the resampled luminance entering the network.
        relative_panels: ``--show`` panels drawn relative to rest (noted once
            in the hint line).
        scale: Extra multiplier on every font size.
        figsize: Explicit figure size in inches; otherwise fitted to the screen.
        dpi: Figure dpi override (``None`` keeps matplotlib's default).
    """

    def __init__(
        self,
        panel_names: Sequence[str],
        color_limit: float,
        display: bool,
        *,
        show_retina: bool = False,
        relative_panels: Sequence[str] = (),
        scale: float = 1.0,
        figsize: tuple[float, float] | None = None,
        dpi: float | None = None,
    ) -> None:
        import matplotlib

        if not display:
            matplotlib.use("Agg")
        defaults: Any = matplotlib.rcParamsDefault
        rc_params: Any = matplotlib.rcParams
        rc_params.update({key: defaults[key] for key in RC_KEYS_RESET_TO_DEFAULT})
        import matplotlib.pyplot as plt

        self.display = display
        self.show_retina = show_retina
        self.paused = False
        self.reset_requested = False
        self.is_open = True
        self._needs_background = True
        self._user_scale = scale
        self._meter_peak = RunningPeak()
        self._plt = plt
        self._raster = HexRaster()
        self._scaled_texts: list[tuple[Any, float]] = []
        self._tick_axes: list[Any] = []

        readout_rows = [
            list(panel_names[start : start + MAPS_PER_ROW])
            for start in range(0, len(panel_names), MAPS_PER_ROW)
        ]
        row_count = max(2, len(readout_rows))

        self.figure = plt.figure(
            figsize=(8, 5), dpi=dpi or DEFAULT_FIGURE_DPI, layout="constrained"
        )
        layout_engine: Any = self.figure.get_layout_engine()
        layout_engine.set(w_pad=0.04, h_pad=0.04, wspace=0.04, hspace=0.06)
        width, height = fit_figure_size(
            row_count, self.figure.dpi, _screen_logical_size(self.figure), figsize
        )
        self.figure.set_size_inches(width, height, forward=True)
        grid = self.figure.add_gridspec(row_count, LEFT_COLUMNS + MAPS_PER_ROW)

        camera_axes = self.figure.add_subplot(grid[0, 0])
        photoreceptor_axes = self.figure.add_subplot(grid[0, 1])
        motion_axes = self.figure.add_subplot(grid[0, 2])
        retina_axes = self.figure.add_subplot(grid[1, 0]) if show_retina else None
        meter_axes = self.figure.add_subplot(
            grid[1, 1:LEFT_COLUMNS] if show_retina else grid[1, :LEFT_COLUMNS]
        )
        map_axes = []
        map_names = []
        for row_index, names in enumerate(readout_rows):
            for column_index, name in enumerate(names):
                column = LEFT_COLUMNS + column_index
                cell = grid[:, column] if len(readout_rows) == 1 else grid[row_index, column]
                map_axes.append(self.figure.add_subplot(cell))
                map_names.append(name)

        blank = np.full(DEFAULT_FRAME_SHAPE, GREY_LEVEL, dtype=np.uint8)
        self._image = camera_axes.imshow(blank, animated=display)
        camera_axes.set_xticks([])
        camera_axes.set_yticks([])
        self._title(camera_axes, "camera (96x96)")

        grey_columns = np.zeros(HEX_COLUMN_COUNT, dtype=np.float32)
        self._photoreceptors = self._raster.imshow(
            photoreceptor_axes,
            grey_columns,
            cmap="gray",
            vmin=-1.0,
            vmax=1.0,
            interpolation="nearest",
        )
        self._photoreceptors.set_animated(display)
        self._title(photoreceptor_axes, "photoreceptors R1-R6")

        self._motion = motion_axes.imshow(
            self._raster.render_rgb(np.zeros((HEX_COLUMN_COUNT, 3), dtype=np.float32)),
            interpolation="nearest",
            animated=display,
        )
        motion_axes.set_facecolor("#808080")
        motion_axes.set_xticks([])
        motion_axes.set_yticks([])
        self._title(motion_axes, "motion percept (T4/T5)")
        wheel_axes = motion_axes.inset_axes((0.74, 0.0, 0.26, 0.26))
        wheel_axes.imshow(hue_wheel_image(), interpolation="bilinear")
        wheel_axes.set_axis_off()

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
            self._title(retina_axes, "retina (721 columns)")

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
        self._title(meter_axes, f"direction meter ({meter_families})")
        self._scaled_texts.append((meter_axes.set_ylabel("relative to peak"), BASE_FONT_PT["tick"]))
        self._tick_axes.append(meter_axes)
        self._meter_text = meter_axes.text(
            0.5,
            0.95,
            "",
            transform=meter_axes.transAxes,
            ha="center",
            va="top",
            animated=display,
        )
        self._scaled_texts.append((self._meter_text, BASE_FONT_PT["meter"]))

        self._maps = []
        for axes, name in zip(map_axes, map_names, strict=True):
            image = self._raster.imshow(
                axes,
                grey_columns,
                cmap="coolwarm",
                vmin=-color_limit,
                vmax=color_limit,
                interpolation="nearest",
            )
            image.set_animated(display)
            self._title(axes, name, bold=True)
            self._maps.append(image)
        colorbar = self.figure.colorbar(self._maps[0], ax=map_axes, shrink=0.7, pad=0.01, aspect=30)
        self._scaled_texts.append((colorbar.ax.set_ylabel("activity (a.u.)"), BASE_FONT_PT["tick"]))
        self._tick_axes.append(colorbar.ax)

        left_panels = ["camera"] + (["retina"] if show_retina else [])
        left_panels += ["photoreceptors", "motion percept"]
        self.panel_names = left_panels + map_names

        self._overlay = self.figure.suptitle(
            "display   0.0 fps | eye   0.0 ms | camera, eye stepped at 50 Hz",
            family="monospace",
            animated=display,
        )
        self._scaled_texts.append((self._overlay, BASE_FONT_PT["overlay"]))
        hint = "space pause/resume · r reset eye · q/Esc quit"
        if relative_panels:
            hint = f"{', '.join(relative_panels)} shown relative to rest   |   {hint}"
        self._scaled_texts.append(
            (self.figure.supxlabel(hint, color="#555555"), BASE_FONT_PT["hint"])
        )
        self._apply_font_scale()

        self._animated: list[Any] = [
            self._image,
            self._photoreceptors,
            self._motion,
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

    def _title(self, axes: Any, text: str, *, bold: bool = False) -> None:
        title = axes.set_title(text, fontweight="bold" if bold else "normal")
        self._scaled_texts.append((title, BASE_FONT_PT["title"]))

    def _apply_font_scale(self) -> None:
        scale = font_scale(self.figure.get_size_inches()[0], self._user_scale)
        for text, base_points in self._scaled_texts:
            text.set_fontsize(base_points * scale)
        for axes in self._tick_axes:
            axes.tick_params(labelsize=BASE_FONT_PT["tick"] * scale)

    def describe_geometry(self) -> str:
        """Return figure size, dpi, device pixel ratio, and font scale for logs."""
        width, height = self.figure.get_size_inches()
        dpi = self.figure.dpi
        ratio = getattr(self.figure.canvas, "device_pixel_ratio", 1.0)
        return (
            f"figure {width:.2f}x{height:.2f} in at {dpi:.0f} dpi = "
            f"{width * dpi:.0f}x{height * dpi:.0f} logical px; device pixel ratio "
            f"{ratio}; font scale {font_scale(width, self._user_scale):.2f}"
        )

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
        self._apply_font_scale()
        self._needs_background = True

    def poll(self) -> None:
        """Process pending window events (key presses, close, resize)."""
        if self.display:
            self.figure.canvas.flush_events()

    def update(self, data: ViewFrame) -> None:
        """Push new data into the artists and redraw only them."""
        self._image.set_data(data.frame)
        self._photoreceptors.set_data(self._raster.render(data.photoreceptors))
        self._motion.set_data(self._raster.render_rgb(data.motion_rgb))
        if self._retina is not None and data.retina is not None:
            self._retina.set_data(self._raster.render(data.retina))
        for image, values in zip(self._maps, data.maps.values(), strict=True):
            image.set_data(self._raster.render(values))
        peak = self._meter_peak.update(max(data.meter.values()))
        winner = find_winning_direction(data.meter)
        for bar, direction in zip(self._bars, METER_DIRECTIONS, strict=True):
            bar.set_height(data.meter[direction] / peak)
            bar.set_color("#dd8452" if direction == winner else "#4c72b0")
        self._meter_text.set_text(format_meter(data.meter, compact=True))
        self._overlay.set_text(data.overlay_text)
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
            self.figure.savefig(path)
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
    parser.add_argument("--save-dir", type=Path, default=None, help="Save PNG snapshots here.")
    parser.add_argument("--save-every", type=int, default=50, help="Snapshot period.")
    parser.add_argument("--meter-statistic", choices=METER_STATISTICS, default="q95")
    parser.add_argument("--color-limit", type=float, default=DEFAULT_COLOR_LIMIT)
    parser.add_argument(
        "--show",
        default=None,
        metavar="TYPE[,TYPE...]",
        help="Extra cell types to draw as hex maps, e.g. R1,L1,Mi1,Tm3.",
    )
    parser.add_argument("--hide-t5", action="store_true", help="Drop the T5a-d row (compact view).")
    parser.add_argument(
        "--show-retina",
        action="store_true",
        help="Also draw the resampled luminance entering the eye (camera-derived).",
    )
    parser.add_argument("--scale", type=float, default=1.0, help="Multiply every font size.")
    parser.add_argument(
        "--figsize",
        default=None,
        metavar="W,H",
        help="Figure size in inches (default: fitted to a 1440x900 logical screen).",
    )
    parser.add_argument("--dpi", type=float, default=None, help="Figure dpi (for HiDPI testing).")
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


def _measure_baseline(eye: Any, statistic: str) -> tuple[dict[str, float], npt.NDArray[np.float32]]:
    """Reset the eye and read the resting meter and network state from grey.

    Returns:
        The resting direction meter (subtracted from live values) and the
        resting activity of every neuron (subtracted from the neural panels).
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


def _summarise(stats: LoopStats, agreement: dict[str, list[bool]], rate_label: str) -> None:
    if stats.loop_periods_s:
        mean_fps = 1.0 / statistics.fmean(stats.loop_periods_s)
        print(f"{rate_label}: {mean_fps:.1f} fps mean over {len(stats.loop_periods_s)} frames")
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
    try:
        figsize = parse_figsize(args.figsize)
    except ValueError as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2
    eye = _try_load_eye(args.checkpoint, build_readout_names(hide_t5=args.hide_t5))
    if eye is None:
        return 0
    show_types = [name for name in parse_show_types(args.show) if name not in eye.readout_names]
    try:
        extra_indices = resolve_cell_type_indices(eye, show_types)
    except UnknownCellTypesError as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2
    neural = NeuralIndices.resolve(eye)
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
            show_retina=args.show_retina,
            relative_panels=show_types,
            scale=args.scale,
            figsize=figsize,
            dpi=args.dpi,
        )
        print(f"panels: {', '.join(view.panel_names)}")
        print(view.describe_geometry())
    if args.save_dir is not None:
        args.save_dir.mkdir(parents=True, exist_ok=True)

    stats = LoopStats()
    motion_peak = RunningPeak()
    photoreceptor_peak = RunningPeak()
    agreement: dict[str, list[bool]] = {}
    frame_count = 0
    last_loop = time.perf_counter()
    last_read: float | None = None
    minimum_period = 1.0 / args.fps_cap if args.fps_cap > 0 else 0.0
    try:
        while (view is None or view.is_open) and (args.frames <= 0 or frame_count < args.frames):
            if view is not None:
                view.poll()
                if view.reset_requested:
                    baseline, resting_state = _measure_baseline(eye, args.meter_statistic)
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
                agreement.setdefault(source_frame.label, []).append(winner == source_frame.label)

            state = eye.state_activity
            channels = motion_channels(state, resting_state, neural.motion_channels)
            _, vectors = compute_motion_percept(channels, motion_peak.value)
            motion_peak.update(float(np.quantile(np.hypot(*vectors.T), 0.99)))

            if view is not None:
                motion_rgb, _ = compute_motion_percept(channels, motion_peak.value)
                photoreceptors = photoreceptor_map(state, resting_state, neural.photoreceptors)
                photoreceptor_peak.update(float(np.quantile(np.abs(photoreceptors), 0.99)))
                maps = split_readout_maps(features, eye.readout_names)
                for name, indices in extra_indices.items():
                    maps[name] = state[indices] - resting_state[indices]
                view.update(
                    ViewFrame(
                        frame=source_frame.frame,
                        photoreceptors=photoreceptors / photoreceptor_peak.value,
                        motion_rgb=motion_rgb,
                        maps={name: maps[name] for name in panel_names},
                        meter=meter,
                        overlay_text=_format_overlay(
                            stats,
                            source,
                            source_frame.label,
                            view.display,
                            eye.frame_rate_hz,
                        ),
                        # The retina is the same resampler call encode() made,
                        # repeated (~1.5 ms) purely for display.
                        retina=(
                            eye.resampler.frame(source_frame.frame)[0, 0, 0].numpy()
                            if view.show_retina
                            else None
                        ),
                    )
                )
                if args.save_dir is not None and frame_count % args.save_every == 0:
                    path = args.save_dir / f"flyvis_eye_live_{frame_count:05d}.png"
                    view.save(path)
                    print(f"saved {path}")
            if args.no_display and args.print_every > 0 and frame_count % args.print_every == 0:
                label = f" stim={source_frame.label:<5}" if source_frame.label else ""
                angle, magnitude = summarise_motion(vectors)
                print(
                    f"frame {frame_count:05d}{label} "
                    f"eye {stats.eye_latencies_ms[-1]:5.2f} ms "
                    f"meter {format_meter(meter)} "
                    f"motion {angle:5.1f} deg x{magnitude:.3f}"
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
