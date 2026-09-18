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
asymmetric wing bias and a beat-frequency scalar out.

**Not yet covered:** throttle here means faster wingbeat, not the trackball ``walk_on_ball``
task #21 also names -- that is a separate flybody task (legs, not wings) and needs its own
integration; ``brake`` always reads back ``0.0`` until it exists. Flagging rather than
guessing at a leg-based number that has not been measured.

flybody is imported lazily, so ``import fly_driver.body`` works without it.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from fly_driver.interface import ControlVector

__all__ = ["FlybodyNotInstalledError", "FlybodyWingBody"]

#: How far a full-deflection steer command biases one wing's yaw/roll/pitch actions away
#: from the other's, as a fraction of the action space's own ``[-1, 1]`` range. A starting
#: point, not a measured constant -- AGENTS.md's own words for the default wing pattern
#: ("an approximation ... not a substitute for a realistic base wing pattern") apply here
#: too. Tune once the body can be watched, not eyeballed from source alone.
DEFAULT_STEER_GAIN = 0.3

#: Names of the six wing-joint action components, split by side, in the order flybody's
#: `WingBeatPatternGenerator` duplicates its base pattern -- yaw, roll, pitch -- for two
#: wings. Read from the task's own action_spec names rather than hardcoded indices, so a
#: flybody upgrade that reorders the action vector fails loudly instead of steering the
#: wrong joint.
_LEFT_WING_NAMES = ("wing_yaw_left", "wing_roll_left", "wing_pitch_left")
_RIGHT_WING_NAMES = ("wing_yaw_right", "wing_roll_right", "wing_pitch_right")
_USER_NAME = "user_0"


class FlybodyNotInstalledError(ImportError):
    """Raised when the optional flybody stack is needed but not importable."""


def _import_flybody() -> Any:
    try:
        import flybody.fly_envs as fly_envs
    except ImportError as error:
        raise FlybodyNotInstalledError(
            "FlybodyWingBody needs the optional flybody stack. Install it with "
            "`python -m pip install "
            '"flybody @ git+https://github.com/TuragaLab/flybody.git@'
            'd015e9bfe441bd90ae431bac24c55cb74bdbce26"` '
            "in its own virtualenv -- see docs/running-the-stacks.md."
        ) from error
    return fly_envs


def _action_index_map(action_name: str) -> dict[str, int]:
    """``{component name: index}`` from a dm_control action_spec's tab-joined name."""
    names = action_name.split("\t")
    return {name: index for index, name in enumerate(names)}


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

    def __init__(self, steer_gain: float = DEFAULT_STEER_GAIN, env: Any | None = None) -> None:
        self.steer_gain = float(steer_gain)
        self._env = env
        self._action_index: dict[str, int] | None = None
        self._neutral_action: np.ndarray | None = None

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

        action = self._neutral_action.copy()
        for name in _LEFT_WING_NAMES:
            action[index[name]] = self.steer_gain * intent.steer
        for name in _RIGHT_WING_NAMES:
            action[index[name]] = -self.steer_gain * intent.steer
        action[index[_USER_NAME]] = np.clip(intent.throttle - intent.brake, -1.0, 1.0)

        time_step = env.step(action)
        observation = time_step.observation

        yaw_rate = float(np.asarray(observation["walker/gyro"]).reshape(-1)[2])
        forward_speed = float(np.asarray(observation["walker/velocimeter"]).reshape(-1)[0])

        return ControlVector.clipped(
            steer=yaw_rate,
            throttle=max(forward_speed, 0.0),
            brake=0.0,
        )
