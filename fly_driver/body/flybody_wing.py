"""The tethered fly body: flybody's vision-guided flight task as a `Body` (GH-21).

`AGENTS.md` §6 says reuse flybody's pretrained wing pattern generators and learn only a
low-dimensional mapping from a steering signal to controller modulation -- do not train
wing control from scratch. This module is that mapping.

**The mechanism, read from flybody's own ``before_step`` rather than assumed.** The
``vision_guided_flight`` task (`flybody.tasks.vision_flight.VisionFlightImitationWBPG`)
exposes a 12-dim action: three joints per wing (yaw, roll, pitch; left and right), head,
abdomen, and a single scalar ``user_0`` in ``[-1, 1]``. Each step, flybody's own
``WingBeatPatternGenerator`` computes a *symmetric* target wing trajectory at a beat
frequency set by ``user_0`` (``base_freq * (1 + rel_range * user_0)``), converts it to a
force-control delta, and adds that delta on top of whatever the six wing-joint actions
already are. So:

- ``user_0`` is a throttle-like knob -- it changes beat frequency for both wings equally,
  never asymmetrically, so it cannot steer.
- The six raw wing-joint actions are a *bias* added under a symmetric oscillation. Setting
  them asymmetrically between left and right is what makes one wing's effective stroke
  differ from the other's -- the steering mechanism.

This module never trains the pattern generator itself (flybody's, frozen, reused
verbatim). What it adds is the small mapping AGENTS.md asks for: ``ControlVector`` in,
asymmetric wing bias and a beat-frequency scalar out. Throttle here means wingbeat
frequency; the trackball throttle #21 also names is a separate task, in
:mod:`fly_driver.body.flybody_legs` -- :class:`~fly_driver.body.CombinedBody` composes
both into the one body the ticket asks for.

**Known limitation, found rather than assumed: the steer readout is noisy at the source,
not just under-sampled.** ``vision_guided_flight`` is genuinely free-flying -- checked
against ``walk_on_ball.py``, which explicitly removes its walker's freejoint to fuse the
thorax to the world (a real tether); nothing here does the equivalent. A constant
one-sided wing bias does not produce a clean, sustained turn on a free body, it tumbles.
Averaging the gyro over a whole frame (see ``DEFAULT_MAX_YAW_RATE``'s own comment for the
measured numbers) helps the *timestep* be right but cannot make an unstable body's own
signal stable. The real fix is a proper tethered-flight model (a weld constraint, the way
``walk_on_ball`` already does it for legs) -- out of this ticket's scope, worth its own
issue. ``brake`` always reads back ``0.0``; not modelled at all yet.

flybody is imported lazily, so ``import fly_driver.body`` works without it.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from fly_driver.body._flybody_common import (
    FlybodyNotInstalledError,
    _action_index_map,
    _import_flybody,
    _substeps_per_frame,
)
from fly_driver.interface import FRAME_RATE_HZ, ControlVector

__all__ = ["FlybodyNotInstalledError", "FlybodyWingBody"]

#: How far a full-deflection steer command biases one wing's yaw/roll/pitch actions away
#: from the other's, as a fraction of the action space's own ``[-1, 1]`` range. A starting
#: point, not a measured constant -- AGENTS.md's own words for the default wing pattern
#: ("an approximation ... not a substitute for a realistic base wing pattern") apply here
#: too. Tune once the body can be watched, not eyeballed from source alone.
DEFAULT_STEER_GAIN = 0.3

#: Averaged yaw rate, rad/s, that reads back as `ControlVector.steer` == 1.0.
#:
#: **This does not make the signal clean, and should not be read as if it does.**
#: Measured over 150 frame-averages (15,000 flybody substeps) under sustained
#: full-lock steer: mean -7.1 rad/s, std 88.7, median |value| 62, p90 |value| 139 --
#: i.e. close to zero-mean with huge spread, not a stable turning rate that got noisy.
#: `vision_guided_flight` is genuinely free-flying (`disable_legs=True` by default, no
#: freejoint removed, unlike `walk_on_ball` -- checked in flybody's own task source), so
#: a constant one-sided wing bias does not produce a clean turn, it tumbles. Frame
#: averaging (`_substeps_per_frame`) fixes the *timestep* being wrong (was 1/100th of a
#: frame) but cannot fix an unstable body producing an unstable signal. A runtime
#: "tether" (resetting the root freejoint's qpos/qvel every substep) was tried and made
#: it worse -- abrupt resets are large discontinuities, not free -- and is not shipped.
#: 139 (~p90) is picked so most frames read a real, non-saturated magnitude and only the
#: genuine outliers pin at +/-1, rather than a smaller constant that would pin almost
#: every frame regardless of this problem. The actual fix is a proper tethered-flight
#: model (a weld constraint at the MJCF level, the way `walk_on_ball.py` fuses its
#: walker's thorax to the world) -- out of this ticket's scope; worth its own issue.
DEFAULT_MAX_YAW_RATE = 139.0

#: Names of the six wing-joint action components, split by side, in the order flybody's
#: `WingBeatPatternGenerator` duplicates its base pattern -- yaw, roll, pitch -- for two
#: wings. Read from the task's own action_spec names rather than hardcoded indices, so a
#: flybody upgrade that reorders the action vector fails loudly instead of steering the
#: wrong joint.
_LEFT_WING_NAMES = ("wing_yaw_left", "wing_roll_left", "wing_pitch_left")
_RIGHT_WING_NAMES = ("wing_yaw_right", "wing_roll_right", "wing_pitch_right")
_USER_NAME = "user_0"


class FlybodyWingBody:
    """Wings steer via flybody's vision-guided flight task; frozen wing pattern generator.

    Implements :class:`~fly_driver.interface.Body`: an intended :class:`ControlVector` in,
    the control the cockpit actually reads back out, once flybody's physics has moved the
    simulated wings for one control step.

    Args:
        steer_gain: Fraction of the wing action range (``[-1, 1]`` per joint) that a
            full-deflection ``steer`` command biases left vs. right wing actions apart.
        env: A pre-built flybody environment satisfying the same action/observation
            contract as ``flybody.fly_envs.vision_guided_flight()``. Built lazily on
            first use if not given -- the constructor stays import-light so
            ``FlybodyWingBody`` can be constructed (though not used) without flybody
            installed, matching the eye's ``FlyvisEye`` pattern.

    Raises:
        FlybodyNotInstalledError: On first use (not construction) if flybody is not
            importable.
    """

    def __init__(
        self,
        steer_gain: float = DEFAULT_STEER_GAIN,
        max_yaw_rate: float = DEFAULT_MAX_YAW_RATE,
        frame_rate_hz: float = FRAME_RATE_HZ,
        env: Any | None = None,
    ) -> None:
        self.steer_gain = float(steer_gain)
        self.max_yaw_rate = float(max_yaw_rate)
        self.frame_rate_hz = float(frame_rate_hz)
        self._env = env
        self._action_index: dict[str, int] | None = None
        self._neutral_action: np.ndarray | None = None
        self._substeps: int | None = None

    def _ensure_env(self) -> Any:
        if self._env is None:
            fly_envs = _import_flybody()
            self._env = fly_envs.vision_guided_flight()
        return self._env

    def _ensure_indices(self, env: Any) -> dict[str, int]:
        if self._action_index is None:
            spec = env.action_spec()
            self._action_index = _action_index_map(spec.name)
            self._neutral_action = np.zeros(spec.shape, dtype=np.float64)
            self._substeps = _substeps_per_frame(env, self.frame_rate_hz)
        return self._action_index

    def reset(self) -> None:
        """Reset flybody's episode: the fly returns to rest, wing phase re-randomised."""
        env = self._ensure_env()
        self._ensure_indices(env)
        env.reset()

    def actuate(self, intent: ControlVector) -> ControlVector:
        """Bias the wings by ``intent``, step flybody's physics, read the result back.

        Args:
            intent: The control the upstream policy (or brain) wants.

        Returns:
            What the tethered cockpit actually reads: yaw rate for ``steer``, forward
            airspeed for ``throttle``, and ``0.0`` for ``brake`` (not modelled by flight;
            see the module docstring).
        """
        env = self._ensure_env()
        index = self._ensure_indices(env)
        assert self._neutral_action is not None  # narrows for type checkers
        assert self._substeps is not None

        action = self._neutral_action.copy()
        for name in _LEFT_WING_NAMES:
            action[index[name]] = self.steer_gain * intent.steer
        for name in _RIGHT_WING_NAMES:
            action[index[name]] = -self.steer_gain * intent.steer
        action[index[_USER_NAME]] = np.clip(intent.throttle - intent.brake, -1.0, 1.0)

        # Hold this action for a whole frame's worth of flybody's own (much finer)
        # control steps, the same way CarDynamics.step holds one ControlVector across
        # its physics substeps -- see _substeps_per_frame's docstring for why one
        # env.step() is not one frame. Average the sensors over the window: at 218 Hz
        # the wingbeat itself dominates a single sample (measured: raw gyro-z swings
        # +/-50-120 rad/s within one beat), so one instantaneous reading is mostly beat
        # phase, not net turning rate.
        yaw_rates = np.empty(self._substeps, dtype=np.float64)
        forward_speeds = np.empty(self._substeps, dtype=np.float64)
        for step in range(self._substeps):
            observation = env.step(action).observation
            yaw_rates[step] = np.asarray(observation["walker/gyro"]).reshape(-1)[2]
            forward_speeds[step] = np.asarray(observation["walker/velocimeter"]).reshape(-1)[0]

        steer = float(yaw_rates.mean()) / self.max_yaw_rate if self.max_yaw_rate else 0.0
        forward_speed = float(forward_speeds.mean())

        return ControlVector.clipped(
            steer=steer,
            throttle=max(forward_speed, 0.0),
            brake=0.0,
        )
