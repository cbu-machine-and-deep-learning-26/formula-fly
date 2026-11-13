"""One eye for many tracks: every track's frame through the optic lobe in one batch (GH-27).

The vectorised trainer (:mod:`fly_driver.training.vector_train`) drives ``num_envs`` tracks
at once. :class:`BatchedFlyvisEye` runs the frozen flyvis network on all of their frames in
one call, keeping a separate optic-lobe state per track, so each track sees exactly what a
:class:`~fly_driver.eyes.flyvis_eye.FlyvisEye` of its own would.

**State per track.** flyvis's integrated state is two groups of tensors, ``nodes`` and
``edges``, each with the batch first; ``sources`` and ``targets`` are gathered from the
nodes by the network on every step. This class keeps one batched copy of ``nodes`` and
``edges`` and rebuilds the rest through the network's own ``_state_api``.

**Reset is a copy.** :meth:`FlyvisEye.reset` integrates a second of grey from the bias
state, which is the same every time. This class integrates it once, keeps the result, and
copies it into a track's row on reset: identical features, without 50 extra optic-lobe
steps at every episode start.

:class:`SharedEye` serves the stateless control eyes (pixels, random projection, CNN) from one
instance, which is also the instance a fine-tuned CNN trains. :class:`LoopedEyes` is the
fallback for any other stateful eye: one eye per track, encoded in turn.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any, Protocol

import numpy as np
import torch

from fly_driver.eyes.flyvis_eye import FlyvisEye
from fly_driver.eyes.hex_resampler import RGB_LUMA_WEIGHTS

__all__ = [
    "STATELESS_EYE_TYPES",
    "BatchEye",
    "BatchedFlyvisEye",
    "LoopedEyes",
    "SharedEye",
    "build_batch_eye",
]

#: Eyes that keep nothing between frames, so one instance can serve every track.
STATELESS_EYE_TYPES = frozenset({"pixels", "random_projection", "cnn"})


class BatchEye(Protocol):
    """An eye per track, behind one call."""

    feature_dim: int

    def reset(self, indices: Sequence[int]) -> None:
        """Put these tracks' eyes back to rest, as at an episode start."""

    def encode(self, frames: np.ndarray, indices: Sequence[int] | None = None) -> np.ndarray:
        """Advance the eyes one frame.

        Args:
            frames: ``(k, H, W, 3)`` uint8, one frame per track in ``indices``.
            indices: Which tracks the frames belong to; ``None`` is every track, in order.

        Returns:
            ``(k, feature_dim)`` float32.
        """


class LoopedEyes:
    """One eye per track, each encoded with its own :meth:`~fly_driver.interface.Eye.encode`.

    Args:
        make_eye: Builds one eye; called ``num_envs`` times.
        num_envs: Tracks.
    """

    def __init__(self, make_eye: Callable[[], Any], num_envs: int) -> None:
        self.eyes = [make_eye() for _ in range(num_envs)]
        self.feature_dim = int(self.eyes[0].feature_dim)

    def reset(self, indices: Sequence[int]) -> None:
        """Reset these tracks' eyes."""
        for index in indices:
            self.eyes[index].reset()

    def encode(self, frames: np.ndarray, indices: Sequence[int] | None = None) -> np.ndarray:
        """Encode each frame with its track's eye."""
        indices = range(len(self.eyes)) if indices is None else indices
        rows = [
            self.eyes[index].encode(frame) for index, frame in zip(indices, frames, strict=True)
        ]
        return np.stack(rows).astype(np.float32, copy=False)


class SharedEye:
    """One stateless eye for every track.

    Args:
        eye: An eye whose :meth:`encode` keeps no state between frames.
    """

    def __init__(self, eye: Any) -> None:
        self.eye = eye
        self.feature_dim = int(eye.feature_dim)

    def reset(self, indices: Sequence[int]) -> None:
        """Nothing to reset; the eye keeps no state."""

    def encode(self, frames: np.ndarray, indices: Sequence[int] | None = None) -> np.ndarray:
        """Encode every frame with the one eye."""
        if len(frames) == 0:
            return np.zeros((0, self.feature_dim), dtype=np.float32)
        return np.stack([self.eye.encode(frame) for frame in frames]).astype(np.float32, copy=False)


class BatchedFlyvisEye:
    """A frozen flyvis optic lobe with one state per track.

    Args:
        eye: A built :class:`FlyvisEye` (or subclass, such as the shuffled connectome). Its
            network, resampler, readouts and warm-up are used; its own single state is not.
        num_envs: Tracks.
    """

    def __init__(self, eye: FlyvisEye, num_envs: int) -> None:
        if num_envs < 1:
            raise ValueError("num_envs must be at least 1")
        self.eye = eye
        self.network = eye.network
        self.num_envs = int(num_envs)
        self.feature_dim = int(eye.feature_dim)
        self.device = eye.device
        self._height, self._width, _ = eye.frame_shape
        self._luma = torch.as_tensor(RGB_LUMA_WEIGHTS, dtype=torch.float32, device=self.device)
        eye.reset()
        rest = eye._state  # noqa: SLF001  the integrated grey warm-up, batch of one
        self._rest = {
            group: {name: tensor[0].detach().clone() for name, tensor in rest[group].items()}
            for group in ("nodes", "edges")
        }
        self._state = {
            group: {
                name: tensor.unsqueeze(0).repeat(self.num_envs, *([1] * tensor.dim()))
                for name, tensor in tensors.items()
            }
            for group, tensors in self._rest.items()
        }

    def reset(self, indices: Sequence[int]) -> None:
        """Copy the rest state into these tracks' rows."""
        rows = list(indices)
        if not rows:
            return
        for group, tensors in self._state.items():
            for name, tensor in tensors.items():
                tensor[rows] = self._rest[group][name]

    def _full_state(self, rows: list[int] | None) -> Any:
        from flyvis.utils.tensor_utils import AutoDeref  # type: ignore[import-untyped]

        def pick(tensor: torch.Tensor) -> torch.Tensor:
            return tensor if rows is None else tensor[rows]

        state = AutoDeref(
            nodes=AutoDeref(**{name: pick(t) for name, t in self._state["nodes"].items()}),
            edges=AutoDeref(**{name: pick(t) for name, t in self._state["edges"].items()}),
        )
        return self.network._state_api(state)  # noqa: SLF001  rebuilds sources and targets

    def _store(self, state: Any, rows: list[int] | None) -> None:
        for group in ("nodes", "edges"):
            for name, tensor in self._state[group].items():
                value = state[group][name]
                if rows is None:
                    tensor.copy_(value)
                else:
                    tensor[rows] = value

    def _retina(self, frames: np.ndarray) -> torch.Tensor:
        if frames.dtype != np.uint8 or frames.ndim != 4:
            raise TypeError(f"frames must be (k, H, W, 3) uint8, got {frames.dtype} {frames.shape}")
        if tuple(frames.shape[1:]) != tuple(self.eye.frame_shape):
            raise ValueError(
                f"frame shape {frames.shape[1:]} != expected {self.eye.frame_shape}; "
                "no implicit resize is performed"
            )
        pixels = torch.from_numpy(np.ascontiguousarray(frames)).to(self.device)
        gray = (pixels.to(torch.float32) / 255.0) @ self._luma
        return self.eye.resampler(gray[:, None])

    def encode(self, frames: np.ndarray, indices: Sequence[int] | None = None) -> np.ndarray:
        """Advance these tracks' optic lobes one frame and read out the configured cells."""
        rows = None if indices is None else list(indices)
        expected = self.num_envs if rows is None else len(rows)
        if frames.shape[0] != expected:
            raise ValueError(f"{frames.shape[0]} frames for {expected} tracks")
        if rows is not None and not rows:
            return np.zeros((0, self.feature_dim), dtype=np.float32)
        retina = self._retina(frames)
        with torch.no_grad():
            states = self.network.simulate(
                retina, self.eye.dt, initial_state=self._full_state(rows), as_states=True
            )
            last = states[-1]
            self._store(last, rows)
            features = last.nodes.activity[:, self.eye._readout_indices]  # noqa: SLF001
        return features.cpu().numpy().astype(np.float32, copy=False)


def build_batch_eye(eye_config: Any, eye: Any, env: Any, num_envs: int) -> BatchEye:
    """The batched eye for a built eye: one flyvis network, one shared eye, or one per track.

    Args:
        eye_config: The run's :class:`~fly_driver.training.config.EyeConfig`.
        eye: The eye already built for this run. A flyvis eye lends its network; a stateless
            eye is shared as it is, so a fine-tuned CNN is the one the optimiser trains.
        env: Anything with ``frame_shape`` and ``frame_rate_hz`` (the vector env), for
            building further eyes when one per track is needed.
        num_envs: Tracks.
    """
    from fly_driver.training.train import build_eye

    if isinstance(eye, FlyvisEye):
        return BatchedFlyvisEye(eye, num_envs)
    if eye_config.type in STATELESS_EYE_TYPES:
        return SharedEye(eye)
    return LoopedEyes(lambda: build_eye(eye_config, env), num_envs)
