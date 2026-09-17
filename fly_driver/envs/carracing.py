"""Gymnasium CarRacing wrapper. Optional extra: pip install 'fly-driver[train]'."""

from __future__ import annotations

from typing import Any

from fly_driver.interface import DEFAULT_FRAME_SHAPE, ControlVector


def make_carracing(seed: int = 0, **kwargs: Any) -> Any:
    """Build CarRacing-v3 if gymnasium[box2d] is installed."""
    try:
        import gymnasium as gym
    except ImportError as exc:
        raise ImportError(
            "gymnasium is optional. Install with: pip install 'fly-driver[train]'."
        ) from exc
    env = gym.make("CarRacing-v3", **kwargs)
    frame_shape = getattr(env.observation_space, "shape", None)
    if frame_shape != DEFAULT_FRAME_SHAPE:
        env.close()
        raise ValueError(
            f"CarRacing observation shape {frame_shape} != {DEFAULT_FRAME_SHAPE}; "
            "refusing to silently resize."
        )
    obs, _info = env.reset(seed=seed)
    del obs
    return env


def control_to_carracing(action: ControlVector):
    """Map clipped ControlVector onto CarRacing's (steer, gas, brake) box."""
    return action.clipped().as_array()
