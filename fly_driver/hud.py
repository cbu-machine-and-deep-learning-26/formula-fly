"""Telemetry overlay drawn inside the MuJoCo viewer window (GH-16).

Speed in mph, gear, an rpm bar with a row of shift lights, throttle and brake as bars,
steering as a horizontal bar whose marker runs -1 to +1 -- the
:class:`~fly_driver.interface.ControlVector` range -- and the travel of each coilover.
Watching the input bars is the quickest way to tell whether a feel problem is the car or
the input: if the steering bar is flicking, the car is being told to flick. The suspension
bars exist because F1 springs are so stiff that nothing else shows them working.

The panel is rendered here as a plain RGB array with numpy and handed to the passive
viewer's ``set_images``, which blits it over the 3D view each frame. Two things about that
API are not obvious and both cost a working panel once: it will only draw when a text
overlay is also set, and it positions from the top of the window rather than the bottom
that ``viewport`` implies. Both are explained at :class:`ViewerHUD`. That keeps the whole
thing in one window, needs no toolkit, and makes every pixel testable without a display.
Digits and labels come from a small bitmap font below rather than the viewer's text
overlay, so the layout is under our control and the same on every machine.

The overlay lives only in the viewer's scene. The fly's observation is rendered by
:class:`mujoco.Renderer` from the ``fly_head`` camera and never sees it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import mujoco
import numpy as np
import numpy.typing as npt

from fly_driver.envs.lap import format_lap_time

__all__ = [
    "HEIGHT",
    "MPS_TO_MPH",
    "LED_COUNT",
    "WIDTH",
    "Telemetry",
    "MAP_RECT",
    "TrackMap",
    "ViewerHUD",
    "bar_fill",
    "leds_lit",
    "render",
    "rpm_fraction",
    "shift_light_on",
    "steer_to_x",
    "text_width",
]

WIDTH, HEIGHT = 650, 244

#: Metres per second to miles per hour. The panel is handed the simulation's own unit and
#: converts here, once, so no caller has to remember which unit the display wants and a
#: change of unit stays a one-line change.
MPS_TO_MPH = 2.236936
LED_COUNT = 10

#: The first shift light comes on at this fraction of the limiter; the last at the shift
#: point. Real F1 wheels light theirs over roughly the top quarter of the range.
LED_START_FRACTION = 0.72

Colour = tuple[int, int, int]

_BACKGROUND: Colour = (16, 18, 20)
_TRACK: Colour = (52, 56, 62)
_TEXT: Colour = (232, 232, 232)
_DIM: Colour = (130, 136, 142)
_THROTTLE: Colour = (57, 197, 90)
_BRAKE: Colour = (224, 59, 59)
_STEER: Colour = (74, 163, 255)
_RPM: Colour = (208, 208, 208)
_RPM_RED_ZONE: Colour = (110, 30, 30)
_SUSP_COMPRESS: Colour = (255, 170, 60)
_SUSP_EXTEND: Colour = (120, 170, 220)
_LED_ON: tuple[Colour, ...] = ((40, 200, 80),) * 4 + ((235, 60, 60),) * 4 + ((90, 150, 255),) * 2
_LED_OFF: tuple[Colour, ...] = ((28, 52, 36),) * 4 + ((60, 28, 28),) * 4 + ((30, 40, 72),) * 2
_LED_SHIFT: Colour = (120, 180, 255)
_MAP_BACKGROUND: Colour = (24, 27, 30)
_MAP_TRACK: Colour = (108, 116, 124)
_MAP_START: Colour = (235, 235, 235)
_FLY_BODY: Colour = (255, 82, 62)
_FLY_WING: Colour = (198, 216, 246)
_BEST: Colour = (255, 205, 90)
#: Green for time gained, red for time lost. Separate names from the pedal colours they
#: happen to match, because these mean something else and may want to diverge.
_FASTER: Colour = (57, 197, 90)
_SLOWER: Colour = (224, 59, 59)

# Layout, in panel pixels with y down. One place, so the drawing code reads as geometry.
_MARGIN = 10
_SPEED_XY = (10, 10)
_UNIT_XY = (92, 24)
_RPM_TEXT_XY = (160, 24)
_GEAR_LABEL_XY = (334, 24)
_GEAR_XY = (390, 10)
_LED_Y = (48, 58)
_LED_PITCH, _LED_WIDTH = 40, 34
_RPM_BAR = (10, 64, 410, 74)  # x0, y0, x1, y1
_PEDAL_Y = (86, 146)
_THROTTLE_X = (10, 30)
_BRAKE_X = (38, 58)
_STEER_BAR = (80, 100, 240, 114)
_STEER_MARKER_HALF_WIDTH = 4
_SUSP_Y = (100, 150)
_SUSP_X = {"fl": (270, 290), "fr": (298, 318), "rl": (334, 354), "rr": (362, 382)}
_LAP_X = 10
_LAP_ROWS = (176, 198, 220)
_LAP_VALUE_X = 76
_LAP_COUNT_X = 250
#: The segment block shares the lap block's rows, in the gap between the lap count and the
#: map. There is no vertical room for rows of its own: the panel is 244 px tall and the
#: last lap row already ends at 234.
_SEGMENT_X = 250
MAP_RECT = (432, 8, 642, 236)  # x0, y0, x1, y1
#: How wide the segment block may draw before it would run into the map.
_SEGMENT_MAX_W = MAP_RECT[0] - 8 - _SEGMENT_X

# 5x7 bitmap font: only what the panel prints. '#' is ink.
_FONT: dict[str, tuple[str, ...]] = {
    "0": (".###.", "#...#", "#..##", "#.#.#", "##..#", "#...#", ".###."),
    "1": ("..#..", ".##..", "..#..", "..#..", "..#..", "..#..", ".###."),
    "2": (".###.", "#...#", "....#", "...#.", "..#..", ".#...", "#####"),
    "3": ("#####", "...#.", "..#..", "...#.", "....#", "#...#", ".###."),
    "4": ("...#.", "..##.", ".#.#.", "#..#.", "#####", "...#.", "...#."),
    "5": ("#####", "#....", "####.", "....#", "....#", "#...#", ".###."),
    "6": ("..##.", ".#...", "#....", "####.", "#...#", "#...#", ".###."),
    "7": ("#####", "....#", "...#.", "..#..", ".#...", ".#...", ".#..."),
    "8": (".###.", "#...#", "#...#", ".###.", "#...#", "#...#", ".###."),
    "9": (".###.", "#...#", "#...#", ".####", "....#", "...#.", ".##.."),
    "A": (".###.", "#...#", "#...#", "#####", "#...#", "#...#", "#...#"),
    "B": ("####.", "#...#", "#...#", "####.", "#...#", "#...#", "####."),
    "C": (".####", "#....", "#....", "#....", "#....", "#....", ".####"),
    "E": ("#####", "#....", "#....", "####.", "#....", "#....", "#####"),
    "F": ("#####", "#....", "#....", "####.", "#....", "#....", "#...."),
    "G": (".###.", "#...#", "#....", "#.###", "#...#", "#...#", ".####"),
    "H": ("#...#", "#...#", "#...#", "#####", "#...#", "#...#", "#...#"),
    "K": ("#...#", "#..#.", "#.#..", "##...", "#.#..", "#..#.", "#...#"),
    "L": ("#....", "#....", "#....", "#....", "#....", "#....", "#####"),
    "M": ("#...#", "##.##", "#.#.#", "#.#.#", "#...#", "#...#", "#...#"),
    "N": ("#...#", "##..#", "#.#.#", "#..##", "#...#", "#...#", "#...#"),
    "O": (".###.", "#...#", "#...#", "#...#", "#...#", "#...#", ".###."),
    "P": ("####.", "#...#", "#...#", "####.", "#....", "#....", "#...."),
    "R": ("####.", "#...#", "#...#", "####.", "#.#..", "#..#.", "#...#"),
    "S": (".####", "#....", "#....", ".###.", "....#", "....#", "####."),
    "T": ("#####", "..#..", "..#..", "..#..", "..#..", "..#..", "..#.."),
    "U": ("#...#", "#...#", "#...#", "#...#", "#...#", "#...#", ".###."),
    "/": ("....#", "....#", "...#.", "..#..", ".#...", "#....", "#...."),
    ":": (".....", "..##.", "..##.", ".....", "..##.", "..##.", "....."),
    "-": (".....", ".....", ".....", "#####", ".....", ".....", "....."),
    "+": (".....", "..#..", "..#..", "#####", "..#..", "..#..", "....."),
    ".": (".....", ".....", ".....", ".....", ".....", ".##..", ".##.."),
    " ": (".....",) * 7,
}
GLYPH_W, GLYPH_H = 5, 7

#: The marker on the map. Payton asked for "a simple fly icon", so it is one shape read
#: from above: dark body down the middle, pale wings swept out either side. 'B' body,
#: 'W' wing, '.' transparent -- the map shows through the dots.
_FLY: tuple[str, ...] = (
    "....BBB....",
    "...BBBBB...",
    "..BBBBBBB..",
    ".WBBBBBBBW.",
    "WWWBBBBBWWW",
    "WWWWBBBWWWW",
    ".WWWBBBWWW.",
    "...BBBBB...",
    "....BBB....",
    "....BBB....",
    ".....B.....",
)
FLY_W, FLY_H = len(_FLY[0]), len(_FLY)


@dataclass(frozen=True)
class Telemetry:
    """One frame of what the panel shows.

    Args:
        speed_mps: Ground speed in metres per second, shown as mph.
        gear: Engaged gear, 1-indexed, as a driver counts them.
        rpm: Engine speed.
        throttle: 0 to 1.
        brake: 0 to 1.
        steer: -1 to 1, the ControlVector's own range.
        suspension: Coilover travel for ``fl, fr, rl, rr`` as a fraction of the bump-stop
            travel, compression positive, so -1 to 1.
        position: The car's ``(x, y)`` in world metres, for the map marker. ``None`` hides
            the marker but still draws the outline.
        lap_current: Seconds into the lap being driven.
        lap_last: The lap just completed.
        lap_best: The record, read from the lap-time document -- not from anything this
            process remembers, so deleting a row changes what is shown.
        lap_count: Laps completed this session.
        on_out_lap: True while the car is still driving up to the line for the first
            time, when there is no lap time to show yet.
        penalty_s: Seconds this lap has earned by leaving the circuit, including an
            excursion still in progress. Shown while it is being earned rather than at the
            line, so a driver can see what a mistake is costing them during it.
        segment: Name of the segment being driven, or ``None`` before the first update.
        segment_elapsed: Seconds since entering it.
        segment_delta: The last completed segment's time against its record. Negative is
            faster. ``None`` when nothing has completed, or when there was no record to
            compare against.
        segment_is_best: Whether that last completed segment set a new record.
    """

    speed_mps: float
    gear: int
    rpm: float
    throttle: float
    brake: float
    steer: float
    suspension: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)
    position: tuple[float, float] | None = None
    lap_current: float | None = None
    lap_last: float | None = None
    lap_best: float | None = None
    lap_count: int = 0
    on_out_lap: bool = False
    penalty_s: float = 0.0
    segment: str | None = None
    segment_elapsed: float = 0.0
    segment_delta: float | None = None
    segment_is_best: bool = False


def rpm_fraction(rpm: float, limiter_rpm: float) -> float:
    """Where the needle sits on the rpm bar, 0 to 1."""
    if limiter_rpm <= 0:
        raise ValueError(f"limiter_rpm must be positive, got {limiter_rpm}")
    return min(1.0, max(0.0, rpm / limiter_rpm))


def shift_light_on(rpm: float, shift_rpm: float) -> bool:
    """All lights on at the shift point, not the limiter: it is a cue to shift."""
    return rpm >= shift_rpm


def leds_lit(rpm: float, limiter_rpm: float, shift_rpm: float, count: int = LED_COUNT) -> int:
    """How many shift lights are on: none below :data:`LED_START_FRACTION` of the limiter,
    all at the shift point, filling in linearly between."""
    if not 0 < shift_rpm <= limiter_rpm:
        raise ValueError("need 0 < shift_rpm <= limiter_rpm")
    start = LED_START_FRACTION * limiter_rpm
    if rpm >= shift_rpm:
        return count
    if rpm <= start or shift_rpm <= start:
        return 0
    return int(math.ceil(count * (rpm - start) / (shift_rpm - start)))


def bar_fill(value: float) -> float:
    """Pedal bar fill, 0 to 1. Clipped: a bar cannot show 120%."""
    return min(1.0, max(0.0, value))


def steer_to_x(steer: float, left_px: float, right_px: float) -> float:
    """Marker x for a steering value: -1 at the left edge, +1 at the right, 0 centred."""
    if right_px <= left_px:
        raise ValueError("right_px must be greater than left_px")
    steer = min(1.0, max(-1.0, steer))
    return left_px + (steer + 1.0) / 2.0 * (right_px - left_px)


def _rect(
    img: npt.NDArray[np.uint8], x0: float, y0: float, x1: float, y1: float, colour: Colour
) -> None:
    """Fill [x0, x1) x [y0, y1), clipped to the panel. Coordinates may be floats."""
    xa, xb = sorted((int(round(x0)), int(round(x1))))
    ya, yb = sorted((int(round(y0)), int(round(y1))))
    xa, xb = max(0, xa), min(img.shape[1], xb)
    ya, yb = max(0, ya), min(img.shape[0], yb)
    if xa < xb and ya < yb:
        img[ya:yb, xa:xb] = colour


def _text(img: npt.NDArray[np.uint8], x: int, y: int, text: str, scale: int, colour: Colour) -> int:
    """Draw ``text`` with its top-left at (x, y). Returns the x just past the last glyph."""
    for char in text.upper():
        glyph = _FONT.get(char, _FONT[" "])
        for row, line in enumerate(glyph):
            for col, ink in enumerate(line):
                if ink == "#":
                    _rect(
                        img,
                        x + col * scale,
                        y + row * scale,
                        x + (col + 1) * scale,
                        y + (row + 1) * scale,
                        colour,
                    )
        x += (GLYPH_W + 1) * scale
    return x


def text_width(text: str, scale: int) -> int:
    """Pixel width of ``text`` at ``scale``, including the trailing gap."""
    return len(text) * (GLYPH_W + 1) * scale


def _line(
    img: npt.NDArray[np.uint8],
    x0: float,
    y0: float,
    x1: float,
    y1: float,
    colour: Colour,
    width: int = 1,
) -> None:
    """A straight line of square dots. Enough for a track outline, no anti-aliasing."""
    steps = int(max(abs(x1 - x0), abs(y1 - y0))) + 1
    half = width / 2.0
    for x, y in zip(np.linspace(x0, x1, steps), np.linspace(y0, y1, steps), strict=True):
        _rect(img, x - half, y - half, x + half, y + half, colour)


class TrackMap:
    """The track outline, scaled once into a fixed box, plus world-to-pixel mapping.

    The outline never changes, so it is drawn once into :attr:`base` and copied each
    frame; only the marker is redrawn. That keeps a 700-point polyline off the per-frame
    path entirely.

    The aspect ratio is preserved and the drawing centred, so the circuit keeps its real
    shape rather than being stretched to the box -- a squashed Silverstone would be
    actively misleading about where you are.

    Args:
        points: ``(N, 2)`` centerline points in world metres.
        width: Box width in pixels.
        height: Box height in pixels.
        padding: Pixels of clear space inside the box.

    Raises:
        ValueError: If fewer than two points are given, or the box is too small to draw in.
    """

    def __init__(
        self, points: npt.ArrayLike, *, width: int, height: int, padding: int = 10
    ) -> None:
        pts = np.asarray(points, dtype=float).reshape(-1, 2)
        if len(pts) < 2:
            raise ValueError(f"need at least two centerline points, got {len(pts)}")
        if width - 2 * padding < 2 or height - 2 * padding < 2:
            raise ValueError(f"{width}x{height} box is too small for {padding} px padding")

        self.width, self.height, self.padding = int(width), int(height), int(padding)
        low, high = pts.min(axis=0), pts.max(axis=0)
        span = np.maximum(high - low, 1e-9)
        usable = np.array([width - 2 * padding, height - 2 * padding], dtype=float)
        self._scale = float(np.min(usable / span))
        # Centre whatever room is left over once the aspect ratio has been preserved.
        self._offset = (usable - span * self._scale) / 2.0 + padding
        self._low = low

        base = np.empty((self.height, self.width, 3), dtype=np.uint8)
        base[:] = _MAP_BACKGROUND
        pixels = [self.to_pixel(float(x), float(y)) for x, y in pts]
        for (xa, ya), (xb, yb) in zip(pixels, pixels[1:] + pixels[:1], strict=True):
            _line(base, xa, ya, xb, yb, _MAP_TRACK, width=2)
        # Start/finish, so "where am I" has a reference point.
        sx, sy = pixels[0]
        _rect(base, sx - 3, sy - 3, sx + 3, sy + 3, _MAP_START)
        self.base = base

    def to_pixel(self, x: float, y: float) -> tuple[float, float]:
        """World metres to pixels inside the box. World +y is north, so pixel y flips."""
        px = self._offset[0] + (x - self._low[0]) * self._scale
        py = self.height - (self._offset[1] + (y - self._low[1]) * self._scale)
        return px, py


def _draw_fly(img: npt.NDArray[np.uint8], cx: float, cy: float) -> None:
    """Stamp the fly marker centred on ``(cx, cy)``, leaving '.' pixels untouched."""
    left, top = int(round(cx)) - FLY_W // 2, int(round(cy)) - FLY_H // 2
    for row, line in enumerate(_FLY):
        for col, mark in enumerate(line):
            if mark == ".":
                continue
            colour = _FLY_BODY if mark == "B" else _FLY_WING
            _rect(img, left + col, top + row, left + col + 1, top + row + 1, colour)


def _lap_text(seconds: float | None) -> str:
    """A lap time for the panel, or a dash when there is not one yet."""
    return "--" if seconds is None else format_lap_time(seconds)


def _fit(text: str, scale: int, max_width: int) -> str:
    """Trim ``text`` from the right until it fits in ``max_width`` pixels.

    The segment block draws into a fixed gap beside the lap times, and `_text` happily
    writes past it and over the track map. Names are short today -- ``T4``, ``S11`` -- so
    this never bites, which is exactly why it is enforced here rather than assumed.
    """
    while text and text_width(text, scale) > max_width:
        text = text[:-1]
    return text


def delta_colour(delta: float | None, *, is_best: bool = False) -> Colour:
    """Green for a gain, red for a loss, amber for a record.

    A new best is amber rather than green so it reads as different from an ordinary
    improvement -- it matches the BEST row, which is the thing it just changed.
    """
    if is_best:
        return _BEST
    if delta is None:
        return _DIM
    return _FASTER if delta < 0 else _SLOWER


def delta_text(delta: float | None) -> str:
    """A signed delta for the panel, or a dash when there is nothing to compare to."""
    return "--" if delta is None else f"{delta:+.3f}"


def _draw_segment(img: npt.NDArray[np.uint8], telemetry: Telemetry) -> None:
    """The segment block: which one, how long so far, and how the last one went."""
    if telemetry.segment is not None:
        _text(
            img,
            _SEGMENT_X,
            _LAP_ROWS[1],
            _fit(f"{telemetry.segment} {telemetry.segment_elapsed:5.1f}", 2, _SEGMENT_MAX_W),
            2,
            _TEXT,
        )
    if telemetry.segment_delta is not None or telemetry.segment_is_best:
        label = delta_text(telemetry.segment_delta)
        if telemetry.segment_is_best:
            label = f"{label} BEST" if telemetry.segment_delta is not None else "BEST"
        _text(
            img,
            _SEGMENT_X,
            _LAP_ROWS[2],
            _fit(label, 2, _SEGMENT_MAX_W),
            2,
            delta_colour(telemetry.segment_delta, is_best=telemetry.segment_is_best),
        )


def render(
    telemetry: Telemetry,
    *,
    limiter_rpm: float,
    shift_rpm: float,
    track: TrackMap | None = None,
) -> npt.NDArray[np.uint8]:
    """Draw the panel. Returns an ``(HEIGHT, WIDTH, 3)`` uint8 RGB array, y down."""
    if not 0 < shift_rpm <= limiter_rpm:
        raise ValueError("need 0 < shift_rpm <= limiter_rpm, both positive")
    img = np.empty((HEIGHT, WIDTH, 3), dtype=np.uint8)
    img[:] = _BACKGROUND

    # Speed, rpm and gear across the top.
    _text(img, *_SPEED_XY, f"{telemetry.speed_mps * MPS_TO_MPH:3.0f}", 4, _TEXT)
    _text(img, *_UNIT_XY, "MPH", 2, _DIM)
    _text(img, *_RPM_TEXT_XY, f"{telemetry.rpm:5.0f} RPM", 2, _DIM)
    _text(img, *_GEAR_LABEL_XY, "GEAR", 2, _DIM)
    _text(img, *_GEAR_XY, str(telemetry.gear)[:1], 4, _TEXT)

    # Shift lights.
    lit = leds_lit(telemetry.rpm, limiter_rpm, shift_rpm)
    shifting = shift_light_on(telemetry.rpm, shift_rpm)
    for i in range(LED_COUNT):
        x0 = _MARGIN + i * _LED_PITCH
        colour = _LED_SHIFT if shifting else (_LED_ON[i] if i < lit else _LED_OFF[i])
        _rect(img, x0, _LED_Y[0], x0 + _LED_WIDTH, _LED_Y[1], colour)

    # rpm bar with the red zone from the shift point.
    x0, y0, x1, y1 = _RPM_BAR
    _rect(img, x0, y0, x1, y1, _TRACK)
    shift_x = x0 + rpm_fraction(shift_rpm, limiter_rpm) * (x1 - x0)
    _rect(img, shift_x, y0, x1, y1, _RPM_RED_ZONE)
    _rect(img, x0, y0, x0 + rpm_fraction(telemetry.rpm, limiter_rpm) * (x1 - x0), y1, _RPM)

    # Pedals: vertical bars filling upward.
    top, bottom = _PEDAL_Y
    for (xa, xb), value, colour, label in (
        (_THROTTLE_X, telemetry.throttle, _THROTTLE, "T"),
        (_BRAKE_X, telemetry.brake, _BRAKE, "B"),
    ):
        _rect(img, xa, top, xb, bottom, _TRACK)
        _rect(img, xa, bottom - bar_fill(value) * (bottom - top), xb, bottom, colour)
        _text(img, (xa + xb) // 2 - GLYPH_W // 2, bottom + 5, label, 1, _DIM)

    # Steering: a horizontal bar, -1 to +1, centre tick, moving marker.
    x0, y0, x1, y1 = _STEER_BAR
    _text(img, (x0 + x1) // 2 - text_width("STEER", 1) // 2, y0 - 12, "STEER", 1, _DIM)
    _rect(img, x0, y0, x1, y1, _TRACK)
    centre = steer_to_x(0.0, x0, x1)
    _rect(img, centre - 1, y0 - 3, centre + 1, y1 + 3, _DIM)
    marker = steer_to_x(telemetry.steer, x0, x1)
    _rect(
        img,
        marker - _STEER_MARKER_HALF_WIDTH,
        y0 - 2,
        marker + _STEER_MARKER_HALF_WIDTH,
        y1 + 2,
        _STEER,
    )
    _text(img, x0, y1 + 5, "-1", 1, _DIM)
    _text(img, x1 - text_width("+1", 1) + 1, y1 + 5, "+1", 1, _DIM)

    # Suspension: one bar per corner, compression up from the zero line, extension down.
    top, bottom = _SUSP_Y
    mid = (top + bottom) / 2
    label_x = (_SUSP_X["fl"][0] + _SUSP_X["rr"][1]) // 2 - text_width("SUSP", 1) // 2
    _text(img, label_x, top - 12, "SUSP", 1, _DIM)
    for corner, travel in zip(("fl", "fr", "rl", "rr"), telemetry.suspension, strict=True):
        xa, xb = _SUSP_X[corner]
        _rect(img, xa, top, xb, bottom, _TRACK)
        fraction = min(1.0, max(-1.0, travel))
        if fraction >= 0:
            _rect(img, xa, mid - fraction * (mid - top), xb, mid, _SUSP_COMPRESS)
        else:
            _rect(img, xa, mid, xb, mid - fraction * (bottom - mid), _SUSP_EXTEND)
        _rect(img, xa - 1, mid - 1, xb + 1, mid + 1, _DIM)
        _text(img, (xa + xb) // 2 - text_width(corner, 1) // 2 + 1, bottom + 5, corner, 1, _DIM)

    # Lap times. BEST comes from the lap-time document, so it reflects whatever rows are
    # in the file right now -- delete one and this changes.
    current = "OUT" if telemetry.on_out_lap else _lap_text(telemetry.lap_current)
    rows = (
        ("CUR", current, _TEXT),
        ("LAST", _lap_text(telemetry.lap_last), _DIM),
        ("BEST", _lap_text(telemetry.lap_best), _BEST),
    )
    for y, (label, value, colour) in zip(_LAP_ROWS, rows, strict=True):
        _text(img, _LAP_X, y, label, 2, _DIM)
        _text(img, _LAP_VALUE_X, y, value, 2, colour)
    count_text = f"LAP {telemetry.lap_count}"
    _text(img, _LAP_COUNT_X, _LAP_ROWS[0], count_text, 2, _DIM)
    if telemetry.penalty_s > 0.0:
        offset = text_width(count_text + " ", 2)
        _text(
            img,
            _LAP_COUNT_X + offset,
            _LAP_ROWS[0],
            _fit(f"+{telemetry.penalty_s:.2f}", 2, _SEGMENT_MAX_W - offset),
            2,
            _SLOWER,
        )
    _draw_segment(img, telemetry)

    # The map, with the fly where the car is.
    if track is not None:
        x0, y0, x1, y1 = MAP_RECT
        img[y0:y1, x0:x1] = track.base
        if telemetry.position is not None:
            px, py = track.to_pixel(*telemetry.position)
            _draw_fly(img[y0:y1, x0:x1], px, py)
    return img


class ViewerHUD:
    """Draws the panel in the top-left corner of a passive viewer window.

    The top-left corner is not a style choice. ``viewer.viewport`` over-reports the
    drawable area -- on the machine this was built on it claimed 1706x960 while the real
    client area was 1365x766 -- and MuJoCo anchors its framebuffer to the window's *top*.
    Anything placed relative to the bottom therefore lands below the visible region: the
    first version of this panel used a 12 px bottom margin, drew itself every single frame
    without error, and was never on screen. Calibration blocks at known offsets put the
    cutoff at about 197 units above the reported bottom.

    So the rect is measured down from ``viewport.height``, which lines up with the top of
    the window and is the one edge that can be trusted.

    Args:
        viewer: The handle from :func:`mujoco.viewer.launch_passive`, or anything with its
            ``viewport`` and ``set_images``.
        limiter_rpm: The rev limit; the right end of the rpm bar.
        shift_rpm: Where the shift lights all come on.
        margin_px: Gap from the window's left and top edges.
        track: Outline to draw the map from. ``None`` leaves that corner empty.
    """

    def __init__(
        self,
        viewer,
        *,
        limiter_rpm: float,
        shift_rpm: float,
        margin_px: int = 12,
        track: TrackMap | None = None,
    ) -> None:
        if limiter_rpm <= 0 or not 0 < shift_rpm <= limiter_rpm:
            raise ValueError("need 0 < shift_rpm <= limiter_rpm, both positive")
        self._viewer = viewer
        self.track = track
        self.limiter_rpm = float(limiter_rpm)
        self.shift_rpm = float(shift_rpm)
        self.margin_px = int(margin_px)

    def rect(self, viewport: mujoco.MjrRect) -> mujoco.MjrRect:
        """Where the panel goes: top-left, measured down from the top of the window.

        ``MjrRect.bottom`` is measured up from the framebuffer's bottom edge, so hugging
        the top means subtracting the panel's height from the viewport height.
        """
        return mujoco.MjrRect(
            self.margin_px, viewport.height - HEIGHT - self.margin_px, WIDTH, HEIGHT
        )

    def update(self, telemetry: Telemetry) -> None:
        """Redraw with this frame's telemetry.

        Call this *before* ``viewer.sync()``: sync is what hands the frame to the render
        thread. Skipped while the window is too small to hold the panel.
        """
        viewport = self._viewer.viewport
        if viewport is None:
            return
        if viewport.width < WIDTH + 2 * self.margin_px:
            return
        if viewport.height < HEIGHT + 2 * self.margin_px:
            return
        image = render(
            telemetry,
            limiter_rpm=self.limiter_rpm,
            shift_rpm=self.shift_rpm,
            track=self.track,
        )
        # This empty text overlay is load-bearing. MuJoCo's passive viewer only runs its
        # overlay pass when a text overlay is set, so without it set_images() is accepted
        # every frame, raises nothing, and draws nothing at all -- which is exactly how
        # the first version of this panel shipped invisible. Proven by A/B: identical
        # rect and image, the only difference being this call, panel present vs absent.
        # The strings are empty so nothing but the panel is drawn.
        self._viewer.set_texts(
            (mujoco.mjtFontScale.mjFONTSCALE_100, mujoco.mjtGridPos.mjGRID_TOPRIGHT, "", "")
        )
        self._viewer.set_images([(self.rect(viewport), image)])
