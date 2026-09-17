"""Telemetry window for hand-driving: speed, gear, rpm with a shift light, and the three
control inputs as bars (GH-16).

The kind of overlay a sim-racing game shows in the corner: throttle and brake as vertical
bars, steering as a horizontal bar whose marker runs from -1 to +1, which is exactly the
:class:`~fly_driver.interface.ControlVector` range. Watching those three bars is the
quickest way to tell whether a feel problem is the car or the input -- if the steering bar
is flicking, the car is being told to flick.

tkinter, because it ships with Python on every platform we run and needs no window of its
own inside MuJoCo. The MuJoCo passive viewer runs its GUI on a background thread; tkinter
must stay on the main thread, which is the one running the sim loop, so the loop pumps the
window with :meth:`TelemetryHUD.update` each control step.

The numbers-to-pixels maths lives in small pure functions so it can be tested without a
display. The window itself is only exercised where one exists.
"""

from __future__ import annotations

import contextlib

__all__ = [
    "TelemetryHUD",
    "bar_fill",
    "rpm_fraction",
    "shift_light_on",
    "steer_to_x",
]

WIDTH, HEIGHT = 440, 250

# Layout, in pixels. One place, so the drawing code below reads as geometry.
_RPM_LEFT, _RPM_RIGHT, _RPM_TOP, _RPM_BOTTOM = 20, 340, 92, 118
_LIGHT_CENTRE, _LIGHT_RADIUS = (385, 105), 16
_PEDAL_TOP, _PEDAL_BOTTOM = 140, 236
_THROTTLE_LEFT, _BRAKE_LEFT, _PEDAL_WIDTH = 20, 62, 30
_STEER_LEFT, _STEER_RIGHT, _STEER_TOP, _STEER_BOTTOM = 130, 420, 176, 200
_STEER_MARKER_HALF_WIDTH = 5


def rpm_fraction(rpm: float, limiter_rpm: float) -> float:
    """Where the needle sits on the rpm bar, 0 to 1."""
    if limiter_rpm <= 0:
        raise ValueError(f"limiter_rpm must be positive, got {limiter_rpm}")
    return min(1.0, max(0.0, rpm / limiter_rpm))


def shift_light_on(rpm: float, shift_rpm: float) -> bool:
    """The light comes on at the shift point, not the limiter: it is a cue to shift."""
    return rpm >= shift_rpm


def bar_fill(value: float) -> float:
    """Pedal bar fill, 0 to 1. Clipped: a bar cannot show 120%."""
    return min(1.0, max(0.0, value))


def steer_to_x(steer: float, left_px: float, right_px: float) -> float:
    """Marker x for a steering value: -1 at the left edge, +1 at the right, 0 centred."""
    if right_px <= left_px:
        raise ValueError("right_px must be greater than left_px")
    steer = min(1.0, max(-1.0, steer))
    return left_px + (steer + 1.0) / 2.0 * (right_px - left_px)


class TelemetryHUD:
    """A small always-on-top window drawn with a tkinter canvas.

    Args:
        limiter_rpm: The rev limit; the right end of the rpm bar.
        shift_rpm: Where the shift light comes on.
        title: Window title.

    Raises:
        RuntimeError: If no display is available (tkinter cannot open a window). Callers
            that can run without a HUD should catch this and carry on.
    """

    def __init__(self, *, limiter_rpm: float, shift_rpm: float, title: str = "Telemetry") -> None:
        if limiter_rpm <= 0 or not 0 < shift_rpm <= limiter_rpm:
            raise ValueError("need 0 < shift_rpm <= limiter_rpm, both positive")
        try:
            import tkinter as tk
        except ImportError as exc:  # pragma: no cover - depends on the Python build
            raise RuntimeError("tkinter is not available in this Python") from exc

        self.limiter_rpm = float(limiter_rpm)
        self.shift_rpm = float(shift_rpm)
        self.closed = False

        try:
            self._root = tk.Tk()
        except tk.TclError as exc:  # no display
            raise RuntimeError(f"cannot open a telemetry window: {exc}") from exc
        self._root.title(title)
        self._root.resizable(False, False)
        self._root.attributes("-topmost", True)
        self._root.protocol("WM_DELETE_WINDOW", self._on_close)
        self._tk_error = tk.TclError

        canvas = tk.Canvas(
            self._root, width=WIDTH, height=HEIGHT, bg="#101214", highlightthickness=0
        )
        canvas.pack()
        self._canvas = canvas

        dim, text = "#3a3f45", "#e8e8e8"
        # Speed and gear.
        self._speed = canvas.create_text(
            20, 44, anchor="w", fill=text, font=("Consolas", 34, "bold"), text="0"
        )
        canvas.create_text(20, 74, anchor="w", fill=dim, font=("Consolas", 12), text="km/h")
        canvas.create_text(360, 74, anchor="e", fill=dim, font=("Consolas", 12), text="gear")
        self._gear = canvas.create_text(
            405, 44, anchor="e", fill=text, font=("Consolas", 34, "bold"), text="N"
        )
        # rpm bar: track, fill, red zone from the shift point, ticks.
        canvas.create_rectangle(_RPM_LEFT, _RPM_TOP, _RPM_RIGHT, _RPM_BOTTOM, fill=dim, width=0)
        self._rpm_fill = canvas.create_rectangle(
            _RPM_LEFT, _RPM_TOP, _RPM_LEFT, _RPM_BOTTOM, fill="#d0d0d0", width=0
        )
        shift_x = _RPM_LEFT + rpm_fraction(shift_rpm, limiter_rpm) * (_RPM_RIGHT - _RPM_LEFT)
        canvas.create_rectangle(shift_x, _RPM_TOP, _RPM_RIGHT, _RPM_BOTTOM, fill="#5a1a1a", width=0)
        canvas.create_line(shift_x, _RPM_TOP - 4, shift_x, _RPM_BOTTOM + 4, fill="#ff4040")
        for k in range(0, int(limiter_rpm) + 1, 5000):
            x = _RPM_LEFT + rpm_fraction(k, limiter_rpm) * (_RPM_RIGHT - _RPM_LEFT)
            canvas.create_text(
                x, _RPM_BOTTOM + 10, fill=dim, font=("Consolas", 9), text=f"{k // 1000}k"
            )
        self._rpm_text = canvas.create_text(
            _RPM_RIGHT, _RPM_TOP - 8, anchor="e", fill=dim, font=("Consolas", 10), text="0 rpm"
        )
        # Shift light.
        cx, cy = _LIGHT_CENTRE
        self._light = canvas.create_oval(
            cx - _LIGHT_RADIUS,
            cy - _LIGHT_RADIUS,
            cx + _LIGHT_RADIUS,
            cy + _LIGHT_RADIUS,
            fill="#2a1010",
            outline="#5a1a1a",
            width=2,
        )
        # Pedals.
        for left, label in ((_THROTTLE_LEFT, "thr"), (_BRAKE_LEFT, "brk")):
            canvas.create_rectangle(
                left, _PEDAL_TOP, left + _PEDAL_WIDTH, _PEDAL_BOTTOM, fill=dim, width=0
            )
            canvas.create_text(
                left + _PEDAL_WIDTH / 2,
                _PEDAL_BOTTOM + 8,
                fill=dim,
                font=("Consolas", 9),
                text=label,
            )
        self._throttle = canvas.create_rectangle(
            _THROTTLE_LEFT,
            _PEDAL_BOTTOM,
            _THROTTLE_LEFT + _PEDAL_WIDTH,
            _PEDAL_BOTTOM,
            fill="#39c55a",
            width=0,
        )
        self._brake = canvas.create_rectangle(
            _BRAKE_LEFT,
            _PEDAL_BOTTOM,
            _BRAKE_LEFT + _PEDAL_WIDTH,
            _PEDAL_BOTTOM,
            fill="#e03b3b",
            width=0,
        )
        # Steering: a horizontal bar, -1 to +1, centre tick, moving marker.
        canvas.create_rectangle(
            _STEER_LEFT, _STEER_TOP, _STEER_RIGHT, _STEER_BOTTOM, fill=dim, width=0
        )
        centre = steer_to_x(0.0, _STEER_LEFT, _STEER_RIGHT)
        canvas.create_line(centre, _STEER_TOP - 4, centre, _STEER_BOTTOM + 4, fill="#8a8f95")
        canvas.create_text(
            _STEER_LEFT, _STEER_BOTTOM + 12, fill=dim, font=("Consolas", 9), text="-1"
        )
        canvas.create_text(
            _STEER_RIGHT, _STEER_BOTTOM + 12, fill=dim, font=("Consolas", 9), text="1"
        )
        canvas.create_text(centre, _STEER_TOP - 10, fill=dim, font=("Consolas", 9), text="steer")
        self._steer = canvas.create_rectangle(
            centre - _STEER_MARKER_HALF_WIDTH,
            _STEER_TOP - 2,
            centre + _STEER_MARKER_HALF_WIDTH,
            _STEER_BOTTOM + 2,
            fill="#4aa3ff",
            width=0,
        )
        self._root.update()

    def _on_close(self) -> None:
        self.closed = True
        with contextlib.suppress(self._tk_error):
            self._root.destroy()

    def update(
        self,
        *,
        speed_kmh: float,
        gear: int,
        rpm: float,
        throttle: float,
        brake: float,
        steer: float,
    ) -> None:
        """Redraw with the latest telemetry and pump the window's events.

        Safe to call after the user has closed the window: it becomes a no-op.
        """
        if self.closed:
            return
        canvas = self._canvas
        canvas.itemconfigure(self._speed, text=f"{speed_kmh:.0f}")
        canvas.itemconfigure(self._gear, text=str(gear))
        fill_x = _RPM_LEFT + rpm_fraction(rpm, self.limiter_rpm) * (_RPM_RIGHT - _RPM_LEFT)
        canvas.coords(self._rpm_fill, _RPM_LEFT, _RPM_TOP, fill_x, _RPM_BOTTOM)
        canvas.itemconfigure(self._rpm_text, text=f"{rpm:.0f} rpm")
        lit = shift_light_on(rpm, self.shift_rpm)
        canvas.itemconfigure(
            self._light,
            fill="#ff2020" if lit else "#2a1010",
            outline="#ffb0b0" if lit else "#5a1a1a",
        )
        throttle_top = _PEDAL_BOTTOM - bar_fill(throttle) * (_PEDAL_BOTTOM - _PEDAL_TOP)
        canvas.coords(
            self._throttle,
            _THROTTLE_LEFT,
            throttle_top,
            _THROTTLE_LEFT + _PEDAL_WIDTH,
            _PEDAL_BOTTOM,
        )
        brake_top = _PEDAL_BOTTOM - bar_fill(brake) * (_PEDAL_BOTTOM - _PEDAL_TOP)
        canvas.coords(
            self._brake, _BRAKE_LEFT, brake_top, _BRAKE_LEFT + _PEDAL_WIDTH, _PEDAL_BOTTOM
        )
        x = steer_to_x(steer, _STEER_LEFT, _STEER_RIGHT)
        canvas.coords(
            self._steer,
            x - _STEER_MARKER_HALF_WIDTH,
            _STEER_TOP - 2,
            x + _STEER_MARKER_HALF_WIDTH,
            _STEER_BOTTOM + 2,
        )
        try:
            self._root.update()
        except self._tk_error:  # window destroyed underneath us
            self.closed = True

    def close(self) -> None:
        if not self.closed:
            self._on_close()
