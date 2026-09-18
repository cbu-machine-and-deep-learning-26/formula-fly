"""Frozen pretrained flyvis optic lobe as the default visual frontend.

The eye is the first stage of the pipeline: a camera frame goes in, a feature
vector of optic-lobe cell activity comes out. The pretrained flyvis network is
frozen (``AGENTS.md`` §6); only the policy on top learns.

**Temporal design.** flyvis is not a feed-forward encoder. It is a dynamical
system integrated with Euler steps of ``dt`` seconds, and every frame is one
step. The eye therefore keeps the network state between :meth:`FlyvisEye.encode`
calls, so a control loop that presents one frame per environment step runs the
optic lobe in "real time" at the environment's frame rate. Consequences:

- The frame rate must be the flyvis integration rate, 50 Hz (``dt = 1/50``),
  which is also Gymnasium CarRacing's ``FPS``. Slower rates make the Euler
  integration unreliable and are rejected; faster rates are accepted with a
  warning because they leave the project's 50 Hz assumption.
- :meth:`FlyvisEye.reset` re-initialises the state with a grey warm-up so each
  episode starts from the same resting state. One second of grey is the default:
  measured against a long grey run, 0.5 s leaves a 0.27 a.u. transient, 1 s
  0.027, and 2 s 0.010 (activity itself spans roughly -3 to 6 a.u.).
- Streaming frame-by-frame and encoding a whole sequence use the same integration
  loop and give identical features (asserted in the tests).

flyvis is imported lazily, so ``import fly_driver.eyes`` works without it.
"""

from __future__ import annotations

import os
import warnings
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import torch

from fly_driver.eyes.hex_resampler import (
    DEFAULT_FRAME_SHAPE,
    HEX_COLUMN_COUNT,
    HexResampler,
    frame_to_gray,
)

__all__ = [
    "DEFAULT_CHECKPOINT",
    "DEFAULT_MOTION_READOUTS",
    "DEFAULT_WARMUP_LUMINANCE",
    "DEFAULT_WARMUP_SECONDS",
    "FLYVIS_FRAME_RATE_HZ",
    "FlyvisEye",
    "FlyvisNotInstalledError",
    "resolve_checkpoint_dir",
]

DEFAULT_CHECKPOINT = "flow/0000/000"
DEFAULT_MOTION_READOUTS = ("T4a", "T4b", "T4c", "T4d", "T5a", "T5b", "T5c", "T5d")
FLYVIS_FRAME_RATE_HZ = 50.0
DEFAULT_WARMUP_SECONDS = 1.0
DEFAULT_WARMUP_LUMINANCE = 0.5

Frame = npt.NDArray[np.uint8]
Features = npt.NDArray[np.float32]


class FlyvisNotInstalledError(ImportError):
    """Raised when the optional flyvis stack is needed but not importable."""


def _import_flyvis() -> Any:
    try:
        import flyvis
    except ImportError as error:
        raise FlyvisNotInstalledError(
            "FlyvisEye needs the optional flyvis stack (Python 3.9-3.12). Install it "
            "with `python -m pip install -r requirements-flyvis.txt`, then run "
            "`flyvis download-pretrained`."
        ) from error
    return flyvis


def resolve_checkpoint_dir(
    checkpoint: str | os.PathLike[str] = DEFAULT_CHECKPOINT,
) -> Path:
    """Resolve a checkpoint name or path to an existing flyvis results directory.

    Relative names such as ``"flow/0000/000"`` resolve under ``flyvis.results_dir``
    (``$FLYVIS_ROOT_DIR/results``); existing paths are used as given.

    Args:
        checkpoint: Checkpoint name relative to flyvis's results directory, or a path.

    Returns:
        The checkpoint directory.

    Raises:
        FlyvisNotInstalledError: If flyvis cannot be imported.
        FileNotFoundError: If the directory does not exist.
    """
    candidate = Path(checkpoint).expanduser()
    if candidate.is_dir():
        return candidate
    flyvis = _import_flyvis()
    resolved = Path(flyvis.results_dir) / candidate
    if not resolved.is_dir():
        raise FileNotFoundError(
            f"pretrained flyvis checkpoint not found at {resolved} (checked "
            f"{candidate} first); run `flyvis download-pretrained` with "
            "FLYVIS_ROOT_DIR set."
        )
    return resolved


class FlyvisEye:
    """Frozen pretrained flyvis optic lobe: camera frames in, cell activity out.

    Args:
        checkpoint: Pretrained checkpoint name under ``flyvis.results_dir`` or path.
        readouts: Cell types whose activity is exposed, concatenated in this order.
            Each type contributes one value per hexagonal column (721), so
            ``feature_dim == len(readouts) * 721``. Any of flyvis's 34 output
            cell types is allowed.
        frame_shape: Declared camera frame shape ``(height, width, 3)``.
        device: Torch device for the network. ``None`` keeps flyvis's own choice
            (CUDA when available). flyvis sets the process-wide torch default
            device, so an explicit value changes that default too.
        frame_rate_hz: Environment frame rate; one frame is one integration step
            of ``dt = 1 / frame_rate_hz``. Must be at least 50 Hz.
        warmup_seconds: Grey stimulus duration used by :meth:`reset`.
        warmup_luminance: Grey level in ``[0, 1]`` used by :meth:`reset`.

    Raises:
        FlyvisNotInstalledError: If flyvis is not importable.
        FileNotFoundError: If the checkpoint is missing.
        ValueError: If readouts, frame rate, or warm-up settings are invalid.
    """

    def __init__(
        self,
        checkpoint: str | os.PathLike[str] = DEFAULT_CHECKPOINT,
        readouts: Sequence[str] = DEFAULT_MOTION_READOUTS,
        frame_shape: tuple[int, int, int] = DEFAULT_FRAME_SHAPE,
        device: str | torch.device | None = None,
        frame_rate_hz: float = FLYVIS_FRAME_RATE_HZ,
        warmup_seconds: float = DEFAULT_WARMUP_SECONDS,
        warmup_luminance: float = DEFAULT_WARMUP_LUMINANCE,
    ) -> None:
        if frame_rate_hz < FLYVIS_FRAME_RATE_HZ:
            raise ValueError(
                f"frame_rate_hz={frame_rate_hz} gives dt={1 / frame_rate_hz:.4f} s, "
                f"above flyvis's 1/{FLYVIS_FRAME_RATE_HZ:.0f} s integration limit."
            )
        if frame_rate_hz != FLYVIS_FRAME_RATE_HZ:
            warnings.warn(
                f"frame_rate_hz={frame_rate_hz} differs from the project's "
                f"{FLYVIS_FRAME_RATE_HZ:.0f} Hz assumption; the optic lobe still "
                "integrates one step per frame, at the shorter dt.",
                stacklevel=2,
            )
        if int(warmup_seconds * frame_rate_hz) < 1:
            raise ValueError(
                f"warmup_seconds={warmup_seconds} is shorter than one frame at "
                f"{frame_rate_hz} Hz; a reset must integrate at least one grey frame."
            )
        if not 0.0 <= warmup_luminance <= 1.0:
            raise ValueError("warmup_luminance must be in [0, 1]")
        if not readouts:
            raise ValueError("readouts must name at least one cell type")
        if len(set(readouts)) != len(readouts):
            raise ValueError(f"readouts must not repeat cell types: {list(readouts)}")

        flyvis = _import_flyvis()
        self.checkpoint_dir = resolve_checkpoint_dir(checkpoint)
        self.frame_rate_hz = float(frame_rate_hz)
        self.dt = 1.0 / self.frame_rate_hz
        self.warmup_seconds = float(warmup_seconds)
        self.warmup_luminance = float(warmup_luminance)
        self.resampler = HexResampler(expected_frame_shape=frame_shape)
        if self.resampler.hexals != HEX_COLUMN_COUNT:
            raise ValueError("HexResampler must produce flyvis's 721 columns")

        self.device = self._select_device(flyvis, device)
        network_view = flyvis.NetworkView(self.checkpoint_dir)
        self.network = network_view.init_network()
        self.network.eval()
        self.network.requires_grad_(False)

        available = tuple(self.network.connectome.output_cell_types[:].astype(str))
        self.available_readouts = available
        unknown = [name for name in readouts if name not in available]
        if unknown:
            raise ValueError(f"unknown readout cell types {unknown}; choose from {list(available)}")
        self.readout_names = tuple(readouts)
        layer_index = self.network.connectome.nodes.layer_index
        self._readout_indices = torch.as_tensor(
            np.concatenate([layer_index[name][:] for name in self.readout_names]),
            device=self.device,
        )
        self.feature_dim = int(self._readout_indices.numel())
        self._state: Any = None

    @staticmethod
    def _select_device(flyvis: Any, device: str | torch.device | None) -> torch.device:
        if device is None:
            return torch.device(flyvis.device)
        requested = torch.device(device)
        current = torch.get_default_device()
        if requested.type != current.type or (
            requested.index is not None and requested.index != current.index
        ):
            warnings.warn(
                f"flyvis runs on torch's default device; switching it from {current} "
                f"to {requested} for the whole process.",
                stacklevel=3,
            )
            torch.set_default_device(requested)
        return requested

    @property
    def frame_shape(self) -> tuple[int, int, int]:
        """Declared camera frame shape ``(height, width, 3)``."""
        return self.resampler.expected_frame_shape

    @property
    def cell_count(self) -> int:
        """Number of simulated neurons in the optic-lobe network."""
        return int(self.network.n_nodes)

    @property
    def state_activity(self) -> npt.NDArray[np.float32] | None:
        """Current activity of every neuron, or ``None`` before the first reset."""
        if self._state is None:
            return None
        return self._state.nodes.activity[0].detach().cpu().numpy().astype(np.float32)

    def trainable_parameters(self) -> int:
        """Return the number of parameters that require gradients (always zero)."""
        return sum(p.numel() for p in self.network.parameters() if p.requires_grad)

    def parameter_count(self) -> int:
        """Return the total number of (frozen) network parameters."""
        return sum(p.numel() for p in self.network.parameters())

    def reset(self) -> None:
        """Re-initialise the network state with the configured grey warm-up.

        The warm-up is a uniform grey *camera* frame pushed through the same
        resampler as real frames, not a flat value on every receptor. The
        resampler darkens the outermost columns (its mean filter runs past the
        frame edge), so warming up on the resampled frame leaves those columns
        at rest when the first real frame arrives.
        """
        warmup_frames = int(self.warmup_seconds * self.frame_rate_hz)
        height, width, _ = self.frame_shape
        grey = torch.full(
            (1, warmup_frames, height, width),
            self.warmup_luminance,
            dtype=torch.float32,
            device="cpu",
        )
        retina = self.resampler(grey)
        with torch.no_grad():
            states = self.network.simulate(
                retina.to(self.device), self.dt, initial_state=None, as_states=True
            )
        self._state = states[-1]

    def encode(self, frame: Frame) -> Features:
        """Advance the optic lobe by one frame and return the readout activity.

        The eye resets itself on the first call after construction. Frames must
        arrive at ``frame_rate_hz``.

        Args:
            frame: uint8 RGB frame with the declared ``frame_shape``.

        Returns:
            A float32 ``(feature_dim,)`` array: readout cell types concatenated
            in ``readout_names`` order, each in flyvis hexagonal column order.
        """
        gray = frame_to_gray(frame, expected_shape=self.frame_shape)
        retina = self.resampler(torch.from_numpy(gray)[None, None])
        return self._step(retina)[0]

    def encode_sequence(self, frames: Frame, *, reset: bool = True) -> Features:
        """Encode ``(time, height, width, 3)`` frames as one contiguous episode.

        Args:
            frames: uint8 RGB frames with the declared ``frame_shape`` per frame.
            reset: Warm up from grey before the first frame. With ``False`` the
                sequence continues from the current state.

        Returns:
            A float32 ``(time, feature_dim)`` array.
        """
        if not isinstance(frames, np.ndarray) or frames.ndim != 4:
            raise ValueError(
                "frames must be a (time, height, width, 3) uint8 array, got "
                f"{getattr(frames, 'shape', type(frames).__name__)}"
            )
        if frames.shape[0] == 0:
            raise ValueError("frames must contain at least one frame")
        gray = np.stack([frame_to_gray(frame, expected_shape=self.frame_shape) for frame in frames])
        retina = self.resampler(torch.from_numpy(gray)[None])
        if reset:
            self.reset()
        return self._step(retina)

    def _step(self, retina: torch.Tensor) -> Features:
        """Integrate ``(1, time, 1, 721)`` retina input from the current state."""
        if self._state is None:
            self.reset()
        with torch.no_grad():
            states = self.network.simulate(
                retina.to(self.device),
                self.dt,
                initial_state=self._state,
                as_states=True,
            )
            self._state = states[-1]
            activity = torch.stack([state.nodes.activity[0] for state in states])
            features = activity[:, self._readout_indices]
        return features.cpu().numpy().astype(np.float32, copy=False)
