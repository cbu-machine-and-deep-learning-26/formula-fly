"""What `flybody_wing.py` and `flybody_legs.py` both need: shared, not duplicated.

Two copies of the same lazy import and the same tab-joined action-name lookup would
drift apart the first time either needed a fix. Private (leading underscore, no
``__all__``) because nothing outside :mod:`fly_driver.body` should import from here
directly -- the public names are re-exported from the two body modules themselves.
"""

from __future__ import annotations

from typing import Any


class FlybodyNotInstalledError(ImportError):
    """Raised when the optional flybody stack is needed but not importable."""


def _import_flybody() -> Any:
    try:
        import flybody.fly_envs as fly_envs
    except ImportError as error:
        raise FlybodyNotInstalledError(
            "This body needs the optional flybody stack. Install it with "
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


def _substeps_per_frame(env: Any, frame_rate_hz: float) -> int:
    """How many ``env.step()`` calls fill one ``1/frame_rate_hz`` frame.

    flybody's own control timestep is not this project's frame rate -- measured, not
    assumed: ``vision_guided_flight()``/``walk_on_ball()`` both report
    ``env.control_timestep() == 0.0002`` (5,000 Hz, matched to the wing pattern
    generator's 218 Hz beat frequency needing several samples per cycle), where
    `fly_driver.interface.FRAME_RATE_HZ` is 50. A body that calls ``env.step()`` once
    per :meth:`~fly_driver.interface.Body.actuate` call -- what both bodies here did
    before this existed -- is advancing flybody's physics by 1/100th of a frame's worth
    of simulated time per frame, not one frame's worth, and reading a sensor that only
    ever saw 0.2 ms of it: fast, noisy, and not what "one frame" is supposed to mean
    anywhere else in this project. Queried from the env at runtime rather than hardcoded,
    so a flybody upgrade that changes its control timestep changes this too instead of
    silently going back to being wrong.
    """
    return max(1, round((1.0 / frame_rate_hz) / env.control_timestep()))
