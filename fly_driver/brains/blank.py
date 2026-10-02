"""Dense recurrent net of the same width as the central-complex module.

The blank control keeps the leaky rate neuron and the sliced readout, and replaces
the sparse synthetic wiring with dense input and recurrent maps. Trainable-parameter
count is the comparison RQ2 asks for; this module is not the live driver.
"""

from __future__ import annotations

import math

import numpy as np
import torch
from torch import nn

from fly_driver.brains.constants import (
    DEFAULT_TAU_SECONDS,
    MAX_TAU_SECONDS,
    MIN_TAU_SECONDS,
    VOLTAGE_LIMIT,
    WEIGHT_SCALE,
)
from fly_driver.interface import FRAME_RATE_HZ, Features, validate_features

__all__ = ["BlankWidthNet", "blank_trainable_parameter_count"]


def blank_trainable_parameter_count(input_dim: int, neuron_count: int) -> int:
    """Trainable scalars in a dense net of this width.

    The count is the input map, the recurrent map, one bias per neuron, and one log
    time-constant per neuron. The readout is a slice of the population, matching the
    constrained module, so it adds no extra parameters.

    Args:
        input_dim: Eye feature count.
        neuron_count: Hidden width, the same number the constrained module is sized to.

    Returns:
        The number of trainable parameters.
    """
    width = int(neuron_count)
    return width * (int(input_dim) + width + 2)


class BlankWidthNet(nn.Module):
    """Unconstrained leaky recurrent net with a dense map at the given width.

    Args:
        input_dim: Length of the feature vector :meth:`step` expects.
        neuron_count: Width of the recurrent population.
        output_dim: Length of the sliced readout. Must be at most ``neuron_count``.
        frame_rate_hz: One :meth:`step` integrates ``1 / frame_rate_hz`` seconds.
        device: Where the parameters live.

    Raises:
        ValueError: If a size is not positive or the readout is wider than the net.
    """

    def __init__(
        self,
        input_dim: int,
        neuron_count: int,
        output_dim: int,
        *,
        frame_rate_hz: float = FRAME_RATE_HZ,
        device: str | torch.device = "cpu",
    ) -> None:
        super().__init__()
        self.input_dim = _positive_int("input_dim", input_dim)
        self.neuron_count = _positive_int("neuron_count", neuron_count)
        self.output_dim = _positive_int("output_dim", output_dim)
        if self.output_dim > self.neuron_count:
            raise ValueError(
                f"output_dim={self.output_dim} is wider than neuron_count={self.neuron_count}"
            )
        if not math.isfinite(frame_rate_hz) or frame_rate_hz <= 0:
            raise ValueError(f"frame_rate_hz must be a positive finite number, got {frame_rate_hz}")
        self.frame_rate_hz = float(frame_rate_hz)
        self.dt = 1.0 / self.frame_rate_hz

        self.input_weight = nn.Parameter(torch.empty(self.neuron_count, self.input_dim))
        self.recurrent_weight = nn.Parameter(torch.empty(self.neuron_count, self.neuron_count))
        self.bias = nn.Parameter(torch.zeros(self.neuron_count))
        self.log_tau = nn.Parameter(torch.full((self.neuron_count,), math.log(DEFAULT_TAU_SECONDS)))
        nn.init.normal_(self.input_weight, std=WEIGHT_SCALE / math.sqrt(self.input_dim))
        nn.init.normal_(self.recurrent_weight, std=WEIGHT_SCALE / math.sqrt(self.neuron_count))
        self.register_buffer("voltage", torch.zeros(self.neuron_count))
        self.to(torch.device(device))

    def trainable_parameter_count(self) -> int:
        """Number of parameters with ``requires_grad``."""
        return sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)

    def reset(self) -> None:
        """Return the membrane to rest."""
        self.voltage.zero_()

    def step(self, features: Features) -> Features:
        """Advance one frame and return the sliced readout, ``(output_dim,)`` float32."""
        validated = validate_features(features, self.input_dim)
        drive = torch.from_numpy(np.ascontiguousarray(validated)).to(self.voltage.device)
        with torch.no_grad():
            output = self._integrate(drive)
        return np.ascontiguousarray(output.detach().cpu().numpy(), dtype=np.float32)

    def _integrate(self, drive: torch.Tensor) -> torch.Tensor:
        sensory = torch.tanh(drive)
        rate = torch.tanh(self.voltage)
        current = self.input_weight @ sensory + self.recurrent_weight @ rate + self.bias
        tau = self.log_tau.exp().clamp(MIN_TAU_SECONDS, MAX_TAU_SECONDS)
        decay = torch.exp(-self.dt / tau)
        updated = decay * self.voltage + (1.0 - decay) * current
        self.voltage.copy_(updated.clamp(-VOLTAGE_LIMIT, VOLTAGE_LIMIT))
        return torch.tanh(self.voltage[-self.output_dim :])


def _positive_int(name: str, value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an integer, got {value!r}")
    if value < 1:
        raise ValueError(f"{name} must be positive, got {value}")
    return int(value)
