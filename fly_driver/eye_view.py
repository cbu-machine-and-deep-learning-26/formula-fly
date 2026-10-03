"""See the track through the fly's eye, inside the MuJoCo viewer window (GH-75).

``scripts/drive.py --eye-view`` renders the car's ``fly_head`` camera every control step,
pushes it through :class:`~fly_driver.eyes.flyvis_eye.FlyvisEye`, and draws what the optic
lobe made of it over the 3D view -- the same ``set_images`` path the telemetry panel uses
(:mod:`fly_driver.hud`), so it is one window and no extra toolkit.

What is drawn, in the encoding the webcam demo (GH-46) already uses
(:mod:`fly_driver.analysis.motion_percept`):

* **retina** -- the 721 hexagonal columns of luminance the eye actually receives, after the
  hex resampler. Coarse on purpose: that is the fly's resolution, not a rendering fault.
* **motion** -- T4/T5 activity above rest, fused per column into a vector: hue is its
  direction (right red, up yellow-green, left cyan, down violet), brightness its strength.
  "Rest" is the eye settled on the still view in front of it (:meth:`FlyEyeView.reset`),
  so a parked car is black. Driving should light it up and turning sweep it sideways --
  the egocentric optic flow AGENTS.md section 6 chose a first-person track for.
* **filter** (the big view) -- both at once: the retina dimmed, the motion coloured over it.
* **fly** (the whole window) -- the retina alone, in grey, one flat facet per column with a
  dark line between neighbours: the closest thing here to what the fly is looking at. Its
  field is the ``fly_head`` camera's 75 degrees across about 31 columns, so about 2.4
  degrees a facet; a real fly's eye spans nearly 360 degrees at about 5 degrees a facet.
  Greyscale because flyvis takes luminance alone, not because flies are colour-blind.

``F10`` cycles corner -> big -> fly -> off; MuJoCo's viewer binds F1-F6, F8, F9, letters
and digits, and handles them as well as passing them on, so F10 is the free key.

**Placement trusts only the top and left edges.** ``viewer.viewport`` over-reports the
drawable area (see :class:`~fly_driver.hud.ViewerHUD`), so nothing is anchored right or
bottom: the corner panel sits under the telemetry panel and the big view beside it. The fly
view covers the whole reported viewport -- what is off screen is simply clipped -- and
centres the eye in the part assumed visible (:data:`VISIBLE_FRACTION`).

**The eye is stepped every control step, whatever is showing.** It is a dynamical system
integrated at 50 Hz (``FlyvisEye``); skipping frames while the view is hidden would make the
next frames shown a different eye, so turning the view off only stops the drawing.

Nothing here imports flyvis or torch. The eye is handed in already built, so the default
``drive.py`` path, and these tests, stay flyvis-free.
"""

from __future__ import annotations

import functools
import time
from enum import Enum
from typing import Any, Protocol

import mujoco
import numpy as np
import numpy.typing as npt

from fly_driver.analysis.hex_plots import HexRaster, split_readout_maps
from fly_driver.analysis.motion_percept import (
    RunningPeak,
    compute_motion_percept,
    direction_channels,
)
from fly_driver.eyes.hex_resampler import HEX_COLUMN_COUNT
from fly_driver.eyes.stimuli import GREY_LEVEL
from fly_driver.hud import Overlay

__all__ = [
    "EYE_VIEW_KEY",
    "EYE_VIEW_KEY_NAME",
    "EyePicture",
    "EyeViewMode",
    "FacetLayout",
    "FlyEyeView",
    "facet_raster",
    "big_view_side",
    "place_overlay",
]

#: GLFW key code for F10 (GLFW_KEY_F1 is 290). Unbound by MuJoCo's viewer.
EYE_VIEW_KEY = 299
EYE_VIEW_KEY_NAME = "F10"

#: Side of one hex panel in the corner view, in pixels. The lattice is about 31 columns
#: across, so this is about six pixels a column -- enough to read, small enough to sit
#: under the telemetry panel.
PANEL_PX = 192
PANEL_GAP_PX = 8
#: The big view's side as a fraction of the reported viewport height. The viewport
#: over-reports by up to 25% (a 125% display scale), so 0.7 still clears the bottom edge.
BIG_VIEW_FRACTION = 0.7
#: How bright the retina is under the motion colours in the big view: dim enough that
#: motion reads as colour over it, bright enough to see the track's shape.
RETINA_DIM = 0.45
#: Painted outside the hexagonal field.
BACKGROUND = 0.06
#: The fly view's lattice is rasterised once at this size and scaled up by nearest
#: neighbour: fine enough that the staircase on the facet edges is a pixel or two, cheap
#: enough (about half a second, once) that switching to the view does not stall the car.
FLY_VIEW_BASE_PX = 384
#: The fraction of the reported viewport assumed to be on screen, for centring the fly
#: view. ``viewport`` over-reports by up to 25% (see :class:`~fly_driver.hud.ViewerHUD`);
#: on a window that does not, the eye just sits a little up and to the left of centre.
VISIBLE_FRACTION = 0.8
#: The quantile of per-column motion the brightness is normalised to, as in the webcam demo.
PEAK_QUANTILE = 0.99
#: How long the eye is shown a still view to settle on it, as its resting baseline. The
#: same one second ``FlyvisEye`` warms up for on grey.
SETTLE_SECONDS = 1.0

FloatArray = npt.NDArray[np.floating[Any]]
Frame = npt.NDArray[np.uint8]


class Eye(Protocol):
    """What the eye view needs from an eye: :class:`~fly_driver.eyes.flyvis_eye.FlyvisEye`."""

    frame_shape: tuple[int, int, int]
    frame_rate_hz: float
    readout_names: tuple[str, ...]
    resampler: Any

    def encode(self, frame: Frame) -> FloatArray: ...

    def encode_sequence(self, frames: Frame, *, reset: bool = True) -> FloatArray: ...


class EyeViewMode(Enum):
    """What the overlay shows. ``F10`` steps through them in this order."""

    CORNER = "corner"
    BIG = "big"
    FLY = "fly"
    OFF = "off"

    def next(self) -> EyeViewMode:
        """The mode one key press away."""
        order = list(EyeViewMode)
        return order[(order.index(self) + 1) % len(order)]


class EyePicture:
    """What the eye made of the latest frame, as images. Pure numpy.

    Args:
        readout_names: The eye's readout cell types, in feature order.
        resting_features: The eye's features at rest. Subtracted before rectifying, so a
            neuron's resting level never shows as motion.
        resolution: Side of each rendered hex panel, in pixels.
    """

    def __init__(
        self,
        readout_names: tuple[str, ...],
        resting_features: FloatArray,
        *,
        resolution: int = PANEL_PX,
    ) -> None:
        self.readout_names = tuple(readout_names)
        self.resolution = int(resolution)
        self._raster = HexRaster(self.resolution)
        self._layout: FacetLayout | None = None
        self.set_resting(resting_features)
        self.clear()

    def set_resting(self, resting_features: FloatArray) -> None:
        """Replace the resting baseline, after the eye has been reset."""
        self._resting = split_readout_maps(
            np.asarray(resting_features, dtype=np.float32), self.readout_names
        )

    def clear(self) -> None:
        """Forget the last frame: black motion, black retina, brightness scale from scratch."""
        #: Mean per-column motion-vector length of the last frame, before any brightness
        #: scaling: how much the eye saw move, comparable across frames.
        self.motion_strength = 0.0
        self.retina = np.zeros(HEX_COLUMN_COUNT, dtype=np.float32)
        self.motion_rgb = np.zeros((HEX_COLUMN_COUNT, 3), dtype=np.float32)
        self._peak = RunningPeak()

    def update(self, features: FloatArray, retina: FloatArray) -> None:
        """Take in one step of the eye.

        Args:
            features: The eye's ``(len(readout_names) * 721,)`` output for this frame.
            retina: The ``(721,)`` luminance the eye received, in ``[0, 1]``.
        """
        retina = np.asarray(retina, dtype=np.float32).reshape(-1)
        if retina.shape != (HEX_COLUMN_COUNT,):
            raise ValueError(f"retina must hold {HEX_COLUMN_COUNT} columns, got {retina.shape}")
        readouts = split_readout_maps(np.asarray(features, dtype=np.float32), self.readout_names)
        channels = direction_channels(readouts, self._resting)
        _, vectors = compute_motion_percept(channels, self._peak.value)
        magnitudes = np.hypot(*vectors.T)
        self.motion_strength = float(np.mean(magnitudes))
        self._peak.update(float(np.quantile(magnitudes, PEAK_QUANTILE)))
        self.motion_rgb, _ = compute_motion_percept(channels, self._peak.value)
        self.retina = np.clip(retina, 0.0, 1.0)

    def retina_image(self) -> FloatArray:
        """The retina as a ``(resolution, resolution, 3)`` grey image in ``[0, 1]``."""
        grey = np.repeat(self.retina[:, None], 3, axis=1)
        return self._raster.render_rgb(grey, background=BACKGROUND)

    def motion_image(self) -> FloatArray:
        """The motion percept as a ``(resolution, resolution, 3)`` image in ``[0, 1]``."""
        return self._raster.render_rgb(self.motion_rgb, background=BACKGROUND)

    def filter_image(self) -> FloatArray:
        """Retina dimmed, motion coloured over it: the track as the fly's motion cells see it."""
        fused = np.clip(RETINA_DIM * self.retina[:, None] + self.motion_rgb, 0.0, 1.0)
        return self._raster.render_rgb(fused, background=BACKGROUND)

    def corner_image(self) -> npt.NDArray[np.uint8]:
        """Retina and motion side by side: ``(resolution, 2 * resolution + gap, 3)`` uint8."""
        gap = np.full((self.resolution, PANEL_GAP_PX, 3), BACKGROUND, dtype=np.float32)
        return _to_uint8(np.concatenate([self.retina_image(), gap, self.motion_image()], axis=1))

    def fly_image(self, width: int, height: int) -> npt.NDArray[np.uint8]:
        """The retina alone, in grey, filling a ``(height, width, 3)`` uint8 window.

        The returned array is reused by the next call; the viewer copies it on the way in.
        """
        if self._layout is None or self._layout.shape != (height, width):
            self._layout = FacetLayout(width, height, facet_raster())
        return self._layout.paint(self.retina)

    def big_image(self, side: int) -> npt.NDArray[np.uint8]:
        """The filter view scaled (nearest neighbour) to ``(side, side, 3)`` uint8.

        Nearest neighbour keeps the hexagons crisp, and costs one gather rather than
        re-rasterising the lattice at every window size.
        """
        if side < 1:
            raise ValueError(f"side must be positive, got {side}")
        index = (np.arange(side) * self.resolution) // side
        return _to_uint8(self.filter_image()[index][:, index])


@functools.cache
def facet_raster() -> HexRaster:
    """The lattice the fly view is scaled from, built once per process.

    About two and a half seconds to build, so :class:`FlyEyeView` asks for it at start-up,
    while the eye loads, rather than on the first press of F10 with the car moving.
    """
    return HexRaster(FLY_VIEW_BASE_PX)


class FacetLayout:
    """Which retina column paints each pixel of a full-window fly view.

    Worked out once per window size. A frame is then one table lookup over the square the
    eye sits in -- everything outside it is always black, so it is painted once and left --
    which measured 2.5 ms for a 768-pixel eye in a 1706x960 window, against 10.7 ms for a
    naive lookup over every pixel.

    Args:
        width: Window width in pixels.
        height: Window height in pixels.
        raster: The lattice at some resolution; scaled to the view by nearest neighbour.

    Raises:
        ValueError: If the window has no area.
    """

    #: Index of the black entry: outside the eye, and the lines between facets.
    DARK = HEX_COLUMN_COUNT

    def __init__(self, width: int, height: int, raster: HexRaster) -> None:
        if width < 1 or height < 1:
            raise ValueError(f"the window must have an area, got {width}x{height}")
        self.shape = (int(height), int(width))
        visible_width = max(1, int(width * VISIBLE_FRACTION))
        visible_height = max(1, int(height * VISIBLE_FRACTION))
        side = min(visible_width, visible_height)
        self.left = (visible_width - side) // 2
        self.top = (visible_height - side) // 2
        self.side = side

        resolution = raster.index.shape[0]
        scale = (np.arange(side) * resolution) // side
        square = raster.index[scale][:, scale].astype(np.intp)
        square[square < 0] = self.DARK
        # One-pixel lines wherever a pixel's facet differs from its left or upper
        # neighbour's: the facets read as facets, and the eye's rim as an edge.
        edge = np.zeros(square.shape, dtype=bool)
        edge[:, 1:] |= square[:, 1:] != square[:, :-1]
        edge[1:, :] |= square[1:, :] != square[:-1, :]
        square[edge] = self.DARK
        self.square_index = square

        # Native-size indices and three-byte rows: one gather per pixel, not per channel,
        # and no index conversion on every frame.
        self._square_flat = np.ascontiguousarray(square).ravel()
        self._table = np.zeros((HEX_COLUMN_COUNT + 1, 3), dtype=np.uint8)
        self._rows = self._table.view("V3").ravel()
        self._square = np.empty(self._square_flat.size, dtype="V3")
        self._window = np.zeros((*self.shape, 3), dtype=np.uint8)

    def paint(self, retina: FloatArray) -> npt.NDArray[np.uint8]:
        """The window, each facet the grey level of its column. The array is reused."""
        grey = np.clip(np.asarray(retina, dtype=np.float32), 0.0, 1.0) * 255.0 + 0.5
        self._table[:HEX_COLUMN_COUNT] = grey.astype(np.uint8)[:, None]
        np.take(self._rows, self._square_flat, out=self._square)
        top, left, side = self.top, self.left, self.side
        self._window[top : top + side, left : left + side] = self._square.view(np.uint8).reshape(
            side, side, 3
        )
        return self._window


def _to_uint8(image: FloatArray) -> npt.NDArray[np.uint8]:
    return (np.clip(image, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)


def place_overlay(
    viewport_height: int, *, left_px: int, top_px: int, width: int, height: int
) -> mujoco.MjrRect:
    """A rect ``top_px`` down from the top edge and ``left_px`` in from the left.

    ``MjrRect.bottom`` is measured up from the framebuffer's bottom edge, which is the
    edge that cannot be trusted; measuring down from ``viewport_height`` lands on the top.
    """
    return mujoco.MjrRect(int(left_px), int(viewport_height - height - top_px), width, height)


def big_view_side(
    viewport_width: int, viewport_height: int, *, left_px: int, margin_px: int
) -> int:
    """The big view's side in pixels: most of the height, never past the right edge."""
    by_height = int(BIG_VIEW_FRACTION * viewport_height)
    by_width = int(BIG_VIEW_FRACTION * (viewport_width - left_px - margin_px))
    return max(0, min(by_height, by_width))


class FlyEyeView:
    """Renders the fly's camera, steps the eye, and draws what it saw.

    Args:
        model: The car-and-track model; it must have ``camera``.
        eye: A built eye, e.g. :class:`~fly_driver.eyes.flyvis_eye.FlyvisEye`. Stepped once
            per :meth:`step`, so it must be built for the control rate.
        camera: The camera the eye looks through.
        mode: What to show first.
        margin_px: Gap from the window's top and left edges, and between panels.
        below_px: Height of whatever already sits in the top-left corner (the telemetry
            panel), so the corner view goes under it. ``0`` when there is nothing there.
        beside_px: Width of that same thing, so the big view goes beside it.
    """

    #: The GLFW key that cycles :attr:`mode`, for the viewer's ``key_callback``.
    key = EYE_VIEW_KEY

    def __init__(
        self,
        model: mujoco.MjModel,
        eye: Eye,
        *,
        camera: str = "fly_head",
        mode: EyeViewMode = EyeViewMode.CORNER,
        margin_px: int = 12,
        below_px: int = 0,
        beside_px: int = 0,
    ) -> None:
        if mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, camera) < 0:
            raise ValueError(f"the model has no camera named {camera!r}")
        height, width, _ = eye.frame_shape
        self.eye = eye
        self.camera = camera
        self.mode = mode
        self.margin_px = int(margin_px)
        self.below_px = int(below_px)
        self.beside_px = int(beside_px)
        self.last_step_ms = 0.0
        self._renderer = mujoco.Renderer(model, height, width)
        self._settle_frames = max(1, round(SETTLE_SECONDS * eye.frame_rate_hz))
        # Grey until the first reset() shows the eye the real view; drive.py does that as
        # soon as the car is on the grid.
        grey = np.full(eye.frame_shape, GREY_LEVEL, dtype=np.uint8)
        self.picture = EyePicture(eye.readout_names, self._settle(grey))
        facet_raster()  # now, while the eye loads, not on the first F10 mid-drive

    def _settle(self, frame: Frame) -> FloatArray:
        """Hold ``frame`` still in front of the eye until it settles; return its features.

        That is the baseline motion is measured from. Not a grey screen: a still, textured
        world keeps T4/T5 off their grey-screen rest (they answer static contrast too), and
        measured from grey a parked car read 0.058 against 0.070 at 60 m/s -- the colours
        were mostly that offset. Settled on the real view it read 0.003 against 0.035.
        The eye's state carries on from here, so streaming continues seamlessly.
        """
        held = np.repeat(np.asarray(frame, dtype=np.uint8)[None], self._settle_frames, axis=0)
        return np.asarray(self.eye.encode_sequence(held, reset=True)[-1], dtype=np.float32)

    def render_frame(self, data: mujoco.MjData) -> Frame:
        """What the fly's camera sees right now, as the eye receives it."""
        self._renderer.update_scene(data, camera=self.camera)
        return np.array(self._renderer.render(), dtype=np.uint8, copy=True)

    def reset(self, data: mujoco.MjData) -> None:
        """Settle the eye on the current view: on the grid at the start, and after a reset.

        Without this the teleport back to the grid is a frame of the whole world jumping,
        and the motion cells answer it with a burst that has nothing to do with driving.
        Costs about half a second (one batched second of eye time).
        """
        self.picture.set_resting(self._settle(self.render_frame(data)))
        self.picture.clear()

    @property
    def covers_window(self) -> bool:
        """Whether the view hides everything else, so drawing anything under it is waste."""
        return self.mode is EyeViewMode.FLY

    def cycle_mode(self) -> EyeViewMode:
        """Step to the next mode; returns it. Safe to call from the viewer's key thread."""
        self.mode = self.mode.next()
        return self.mode

    def step(self, data: mujoco.MjData) -> None:
        """Render the fly's camera and advance the eye by one frame.

        Called once per control step whatever the mode: the eye is a dynamical system, and
        one that skipped frames while hidden would be a different eye when shown again.
        """
        frame = self.render_frame(data)
        start = time.perf_counter()
        features = self.eye.encode(frame)
        retina = np.asarray(self.eye.resampler.frame(frame), dtype=np.float32)
        self.last_step_ms = (time.perf_counter() - start) * 1000.0
        self.picture.update(features, retina)

    def overlay(self, viewport: mujoco.MjrRect | None) -> Overlay | None:
        """This frame's image and where it goes; ``None`` when off or it does not fit."""
        if viewport is None or self.mode is EyeViewMode.OFF:
            return None
        if self.mode is EyeViewMode.FLY:
            if viewport.width < 1 or viewport.height < 1:
                return None
            image = self.picture.fly_image(viewport.width, viewport.height)
            rect = place_overlay(
                viewport.height, left_px=0, top_px=0, width=viewport.width, height=viewport.height
            )
            return rect, image
        if self.mode is EyeViewMode.CORNER:
            image = self.picture.corner_image()
            top = self.margin_px + (self.below_px + self.margin_px if self.below_px else 0)
            left = self.margin_px
        else:
            left = self.margin_px + (self.beside_px + self.margin_px if self.beside_px else 0)
            top = self.margin_px
            side = big_view_side(
                viewport.width, viewport.height, left_px=left, margin_px=self.margin_px
            )
            if side < self.picture.resolution // 2:
                return None
            image = self.picture.big_image(side)
        height, width, _ = image.shape
        if left + width > viewport.width or top + height > viewport.height:
            return None
        rect = place_overlay(viewport.height, left_px=left, top_px=top, width=width, height=height)
        return rect, image

    def close(self) -> None:
        """Release the offscreen renderer."""
        self._renderer.close()
