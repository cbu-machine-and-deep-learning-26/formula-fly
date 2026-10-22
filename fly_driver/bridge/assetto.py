"""AssettoCorsaGym behind :class:`~fly_driver.bridge.car_side.CarEnv`.

AssettoCorsaGym, OmegaConf and its Python 3.9 environment exist only on the Windows box, so
they are imported inside :class:`AssettoCorsaCar` and nowhere else. :func:`to_ac_action` is
the one piece of numerics here and is tested without AC.

Upstream facts this module relies on (``AssettoCorsaEnv/ac_env.py``):

- Actions are ``(steer, pedal, brake)``. With ``use_relative_actions: False`` they are
  absolute: steer in ``[-norm_steer_at_max, norm_steer_at_max]`` (full lock, from the car's
  ``steer_map.csv``), pedal and brake in ``[-1, 1]`` with ``-1`` released. The upstream
  default is relative (rate deltas), which would turn a held control into a ramp, so it is
  forced off here.
- ``set_actions`` applies a control immediately and ``step(action=None)`` blocks until the
  next ego tick, which is the split :class:`~fly_driver.bridge.car_side.CarEnv` needs.
- ``get_current_image`` returns the plugin's last screen capture, ``(height, width, 3)``
  BGR uint8 when ``color_mode`` is ``bgr``. The plugin resizes AC's window to that size.
- The plugin sends one UDP state per ego tick whether or not anyone has answered, and
  ``step`` reads the *oldest* queued one. A caller that falls behind would read stale
  states in order, so :meth:`AssettoCorsaCar.step` drains the queue to the newest tick.
- ``reset`` closes the UDP link, sends ``reset`` to the plugin's management server, which
  calls CSP's ``ac.ext_resetCar()`` ("resets car to the road, invalidating lap"), sleeps
  1 s, reconnects (the tick counter restarts at 0) and flushes three ticks. There is no
  call that places the car at a chosen point on the lap.
"""

from __future__ import annotations

import select
import sys
from pathlib import Path

import numpy as np
import numpy.typing as npt

from fly_driver.bridge.wire import Telemetry
from fly_driver.interface import FRAME_RATE_HZ, FRAME_SHAPE, ControlVector, Frame

__all__ = ["AssettoCorsaCar", "to_ac_action"]


def to_ac_action(control: ControlVector, max_steer: float) -> npt.NDArray[np.float64]:
    """Map a :class:`ControlVector` onto AssettoCorsaGym's absolute action.

    Args:
        control: ``steer`` in ``[-1, 1]``, ``throttle`` and ``brake`` in ``[0, 1]``.
        max_steer: The env's ``norm_steer_at_max``, so ``steer=±1`` is full lock.

    Returns:
        ``(steer, pedal, brake)`` with pedal and brake in ``[-1, 1]``.
    """
    return np.array(
        [control.steer * max_steer, 2.0 * control.throttle - 1.0, 2.0 * control.brake - 1.0]
    )


class AssettoCorsaCar:
    """One AssettoCorsaGym env, configured for the fly's camera and frame rate.

    Args:
        gym_root: Checkout of github.com/dasGringuen/assetto_corsa_gym.
        config_path: Its ``config.yml``; the fields the bridge depends on are overridden.
        track: AC track id; Silverstone is ``ks_silverstone-gp``.
        car: AC car id; the Formula car is ``dallara_f317``.
        work_dir: Where upstream writes its lap logs.
        frame_shape: Capture size. Must equal ``final_image_*`` in the plugin's config.
        frame_rate_hz: Ego tick rate. Must equal ``ego_sampling_freq`` in the plugin.
        upstream_termination: Keep upstream's own episode ends (off track, 10 s below
            1 m/s, the 600 s cap). The bridge drives on them; the training env decides
            termination itself and turns them off.

    Upstream asserts that the plugin's config matches on the first reset.
    """

    def __init__(
        self,
        gym_root: Path,
        config_path: Path,
        track: str,
        car: str,
        work_dir: Path,
        frame_shape: tuple[int, int, int] = FRAME_SHAPE,
        frame_rate_hz: float = FRAME_RATE_HZ,
        upstream_termination: bool = True,
    ) -> None:
        # Upstream's own notebooks put the inner package directory on the path this way.
        sys.path.insert(0, str(Path(gym_root) / "assetto_corsa_gym"))
        import AssettoCorsaEnv.assettoCorsa as assetto_corsa
        from omegaconf import OmegaConf

        config = OmegaConf.load(config_path)
        ac = config.AssettoCorsa
        ac.track = track
        ac.car = car
        ac.screen_capture_enable = True
        ac.color_mode = "bgr"
        ac.final_image_height, ac.final_image_width = frame_shape[0], frame_shape[1]
        ac.ego_sampling_freq = int(frame_rate_hz)
        ac.use_relative_actions = False
        if not upstream_termination:
            ac.enable_out_of_track_termination = False
            ac.enable_low_speed_termination = False
            ac.max_laps_number = None
            ac.max_episode_py_time = 1e9
        self._env = assetto_corsa.make_ac_env(cfg=config, work_dir=str(work_dir))
        self._max_steer = float(self._env.norm_steer_at_max)

    def reset(self) -> Frame:
        """Reset the car on track and return the first frame."""
        self._env.reset()
        return self._frame()

    def apply(self, control: ControlVector) -> None:
        """Send the control to vJoy now."""
        self._env.set_actions(to_ac_action(control, self._max_steer))

    def step(self) -> tuple[Frame, bool]:
        """Wait for the next ego tick; if ticks are already queued, skip to the newest."""
        _, _, done, _ = self._env.step(action=None)
        sock = self._env.client.socket
        while sock is not None and select.select([sock], [], [], 0)[0]:
            _, _, late_done, _ = self._env.step(action=None)
            done = done or late_done
        return self._frame(), bool(done)

    def telemetry(self) -> Telemetry:
        """The last tick's state, from upstream's expanded state dict."""
        state = self._env.state
        return Telemetry(
            tick=int(state["steps"]),
            time_s=float(state["timestamp_ac"]),
            lap_dist_m=float(state["LapDist"]),
            track_length_m=float(self._env.track_length),
            lateral_m=float(state["gap"]),
            speed_mps=float(state["speed"]),
            tyres_out=int(state["numberOfTyresOut"]),
            centre_on_track=not bool(state["out_of_track_calc"]),
        )

    def close(self) -> None:
        """Release the controls and disconnect from the plugin."""
        self._env.close()

    def _frame(self) -> Frame:
        image = self._env.get_current_image()
        if image is None:
            raise RuntimeError("no screen capture: set screen_capture_enable in sensor_par")
        return np.ascontiguousarray(image[..., ::-1])  # BGR -> RGB
