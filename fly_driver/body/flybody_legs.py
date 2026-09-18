"""The tethered fly's trackball: legs drive throttle and brake (GH-21).

Wings have a pretrained generator to reuse (`flybody_wing.FlybodyWingBody`). Legs do not:
flybody ships no leg/gait pattern generator at all -- checked by reading
`flybody/tasks/pattern_generators.py`, which defines only `WingBeatPatternGenerator`.
`walk_on_ball` and `walk_imitation` are task *definitions*; producing a real tripod
gait from them is a full walking-policy training problem (flybody's own paper trains one
with RL), not something this ticket reuses off the shelf.

So this module is an honest approximation, the same spirit as the wing pattern
generator's own default ("a simple artificial base wing pattern approximation ... not a
substitute for a realistic base wing pattern"): a fixed-frequency alternating-tripod drive
(``T1_left``/``T2_right``/``T3_left`` vs. ``T1_right``/``T2_left``/``T3_right``, the two
tripod groups an insect actually alternates between) whose amplitude is set by
``throttle - brake``. It reads back `walker/ball_qvel`, the trackball's own measured
rotation -- confirmed non-zero and responsive to a driven gait, not assumed -- as the
realised throttle. Brake is not separately modelled here either; see
:mod:`fly_driver.body.flybody_wing` for the same call on wings.

flybody is imported lazily, so ``import fly_driver.body`` works without it.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from fly_driver.body._flybody_common import (
    _action_index_map,
    _import_flybody,
    _substeps_per_frame,
)
from fly_driver.interface import FRAME_RATE_HZ, ControlVector

__all__ = ["TrackballLegBody"]

#: Gait cycle frequency, Hz, at full throttle. Not measured against a target speed --
#: there is no pretrained target to match -- chosen so the tripod alternation is visible
#: over a 20 ms (50 Hz) control step rather than aliasing into noise.
DEFAULT_GAIT_FREQ_HZ = 6.0

#: How far a full-deflection throttle drives the coxa (fore-aft swing) actuators, as a
#: fraction of their own range. femur actuators are driven at half this, matching the
#: coxa-dominant, femur-assisting role coxa/femur play in fore-aft stepping.
DEFAULT_GAIT_GAIN = 0.6

#: Ball rotation, rad/s, read back as `ControlVector.throttle` == 1.0. Measured, not
#: guessed: 80 frame-averaged readings (properly frame-timed, see
#: `_substeps_per_frame`) under sustained full throttle gave mean 0.38 rad/s, std 0.20,
#: max 0.68 -- the gait's own sinusoidal swing means ball speed oscillates over the
#: stride rather than holding constant, which an insect's real walking speed also does.
#: 0.65 sits near the observed peak, so the gait's strongest pushes approach but do not
#: constantly pin the readout at 1.0, and the troughs read as a real, informative low
#: value instead of noise. Unlike the wing body's yaw rate, this one really is just a
#: scale to calibrate -- `walk_on_ball` fuses its walker's thorax to the world (a real
#: tether), so the underlying signal itself is stable; see `flybody_wing.py`'s module
#: docstring for the contrast.
DEFAULT_MAX_BALL_SPEED = 0.65

_TRIPOD_A = ("coxa_T1_left", "coxa_T2_right", "coxa_T3_left")
_TRIPOD_B = ("coxa_T1_right", "coxa_T2_left", "coxa_T3_right")
_FEMUR_A = ("femur_T1_left", "femur_T2_right", "femur_T3_left")
_FEMUR_B = ("femur_T1_right", "femur_T2_left", "femur_T3_right")


class TrackballLegBody:
    """Legs drive a trackball; the ball's own rotation is read back as throttle/brake.

    Implements :class:`~fly_driver.interface.Body`. ``steer`` passes through unchanged --
    this body never touches wings; compose with :class:`~fly_driver.body.FlybodyWingBody`
    via a combining `Body` for both.

    Args:
        gait_freq_hz: Tripod alternation frequency at full throttle magnitude.
        gait_gain: Fraction of the coxa/femur action range the gait swings through.
        max_ball_speed: Ball angular speed, rad/s, that reads back as throttle 1.0.
        env: A pre-built flybody environment matching ``flybody.fly_envs.walk_on_ball()``.
            Built lazily on first use if not given, so ``TrackballLegBody`` can be
            constructed without flybody installed, like the wing body.

    Raises:
        FlybodyNotInstalledError: On first use (not construction) if flybody is not
            importable.
    """

    def __init__(
        self,
        gait_freq_hz: float = DEFAULT_GAIT_FREQ_HZ,
        gait_gain: float = DEFAULT_GAIT_GAIN,
        max_ball_speed: float = DEFAULT_MAX_BALL_SPEED,
        frame_rate_hz: float = FRAME_RATE_HZ,
        env: Any | None = None,
    ) -> None:
        self.gait_freq_hz = float(gait_freq_hz)
        self.gait_gain = float(gait_gain)
        self.max_ball_speed = float(max_ball_speed)
        self.frame_rate_hz = float(frame_rate_hz)
        self._env = env
        self._action_index: dict[str, int] | None = None
        self._neutral_action: np.ndarray | None = None
        self._phase = 0.0
        self._substeps: int | None = None

    def _ensure_env(self) -> Any:
        if self._env is None:
            fly_envs = _import_flybody()
            self._env = fly_envs.walk_on_ball(disable_wings=True)
        return self._env

    def _ensure_indices(self, env: Any) -> dict[str, int]:
        if self._action_index is None:
            spec = env.action_spec()
            self._action_index = _action_index_map(spec.name)
            self._neutral_action = np.zeros(spec.shape, dtype=np.float64)
            self._substeps = _substeps_per_frame(env, self.frame_rate_hz)
        return self._action_index

    def reset(self) -> None:
        """Reset flybody's episode and the gait's own phase."""
        env = self._ensure_env()
        self._ensure_indices(env)
        self._phase = 0.0
        env.reset()

    def actuate(self, intent: ControlVector) -> ControlVector:
        """Drive an alternating-tripod gait scaled by throttle, read the ball back.

        Args:
            intent: The control the upstream policy (or brain) wants. ``steer`` is
                passed through unchanged.

        Returns:
            ``steer`` unchanged, ``throttle`` from the trackball's measured rotation
            speed (clipped to ``[0, 1]``), and ``brake`` fixed at ``0.0`` (not modelled
            by this body; see the module docstring).
        """
        env = self._ensure_env()
        index = self._ensure_indices(env)
        assert self._neutral_action is not None  # narrows for type checkers
        assert self._substeps is not None

        drive = float(np.clip(intent.throttle - intent.brake, -1.0, 1.0))
        swing = self.gait_gain * drive * np.sin(2 * np.pi * self._phase)
        self._phase += self.gait_freq_hz * abs(drive) / self.frame_rate_hz

        action = self._neutral_action.copy()
        for name in _TRIPOD_A:
            action[index[name]] = swing
        for name in _TRIPOD_B:
            action[index[name]] = -swing
        for name in _FEMUR_A:
            action[index[name]] = 0.5 * swing
        for name in _FEMUR_B:
            action[index[name]] = -0.5 * swing

        # Same reasoning as FlybodyWingBody.actuate: hold this gait posture for a whole
        # frame's worth of flybody's own (much finer) control steps rather than one, and
        # average the ball's measured velocity over that window -- see
        # _substeps_per_frame's docstring.
        ball_qvel_sum = np.zeros(3, dtype=np.float64)
        for _ in range(self._substeps):
            observation = env.step(action).observation
            ball_qvel_sum += np.asarray(observation["walker/ball_qvel"]).reshape(-1)
        ball_speed = float(np.linalg.norm(ball_qvel_sum / self._substeps))

        return ControlVector.clipped(
            steer=intent.steer,
            throttle=ball_speed / self.max_ball_speed if self.max_ball_speed else 0.0,
            brake=0.0,
        )
