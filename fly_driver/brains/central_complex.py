"""Sized central-complex and descending module for the closed loop.

The live path is PyTorch, one Euler step per camera frame. Brian2 stays offline
(:data:`fly_driver.brains.OFFLINE_SIMULATOR`) for the lesion map and the
sim-to-biology checks; this module does not import it.

Neuron count is a constructor argument. Nothing here locks a width: the sweep in
:mod:`fly_driver.brains.sweep` is what says which sizes still fit in a 50 Hz frame.

Wiring is synthetic until Codex cell-type names are in the repo. Three compartments
stand in for the central complex and the descending neurons:

- **Compass ring.** A ring attractor: each cell excites its ±1 and ±2 neighbours and
  inhibits the cell opposite it. This is a heading bump, not an EPG reconstruction.
- **Steering population.** Sparse fixed in-degree from the compass and steering cells.
- **Descending population.** Sparse fixed in-degree from the compass and steering
  cells, and no synapses back into the complex. Its bounded activation is the feature
  vector the policy reads.

What is trainable is the synapse strengths on those existing edges, a bias per neuron,
and a log time-constant per neuron. The mask itself is a buffer. A dense blank net of
the same width (:class:`~fly_driver.brains.blank.BlankWidthNet`) is the parameter-count
control; it is not wired into the driver.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

from fly_driver.brains.blank import blank_trainable_parameter_count
from fly_driver.brains.constants import (
    DEFAULT_INPUT_IN_DEGREE,
    DEFAULT_NEURON_COUNT,
    DEFAULT_RECURRENT_IN_DEGREE,
    DEFAULT_TAU_SECONDS,
    MAX_TAU_SECONDS,
    MIN_COMPASS_COUNT,
    MIN_NEURON_COUNT,
    MIN_TAU_SECONDS,
    RING_EXCITATION_FAR,
    RING_EXCITATION_NEAR,
    RING_INHIBITION,
    VOLTAGE_LIMIT,
    WEIGHT_SCALE,
)
from fly_driver.interface import FRAME_RATE_HZ, Features, validate_features

__all__ = ["CentralComplexBrain", "ParameterComparison", "PopulationCounts"]


@dataclass(frozen=True)
class PopulationCounts:
    """How ``neuron_count`` was split across the three synthetic compartments."""

    compass: int
    steering: int
    descending: int

    @property
    def total(self) -> int:
        """Sum of the three compartments."""
        return self.compass + self.steering + self.descending


@dataclass(frozen=True)
class ParameterComparison:
    """Trainable scalars in the constrained module and a dense blank of the same width."""

    neuron_count: int
    input_dim: int
    output_dim: int
    constrained_trainable: int
    blank_trainable: int

    @property
    def constrained_is_smaller(self) -> bool:
        """Whether the sparse module has strictly fewer trainable parameters."""
        return self.constrained_trainable < self.blank_trainable


class CentralComplexBrain(nn.Module):
    """Leaky rate neurons with synthetic central-complex wiring.

    One :meth:`step` is one frame. The sensory drive is ``tanh`` of the eye features,
    so a pixel eye in ``[0, 1]`` and a flyvis readout spanning a few activity units
    both stay inside the same bounded current. Membrane voltage is clamped.

    Args:
        input_dim: Eye feature count.
        neuron_count: Total cells across compass, steering, and descending.
        output_dim: Descending width, which is what the policy reads. ``None`` takes
            about a fifth of ``neuron_count``, leaving room for the ring.
        input_in_degree: Eye synapses onto each cell. Clipped to ``input_dim``.
        recurrent_in_degree: Partners per steering and descending cell.
        frame_rate_hz: Integration rate. One step lasts ``1 / frame_rate_hz`` seconds.
            The practice track's 50 Hz is the project default.
        seed: Fixes the synthetic mask and the initial synapse strengths.
        device: Where the parameters and the membrane live.

    Raises:
        ValueError: If a size cannot be split into the three compartments, or a degree
            or frame rate is not usable.
    """

    def __init__(
        self,
        input_dim: int,
        neuron_count: int = DEFAULT_NEURON_COUNT,
        output_dim: int | None = None,
        *,
        input_in_degree: int = DEFAULT_INPUT_IN_DEGREE,
        recurrent_in_degree: int = DEFAULT_RECURRENT_IN_DEGREE,
        frame_rate_hz: float = FRAME_RATE_HZ,
        seed: int = 0,
        device: str | torch.device = "cpu",
    ) -> None:
        super().__init__()
        self.input_dim = _positive_int("input_dim", input_dim, minimum=1)
        self.neuron_count = _positive_int("neuron_count", neuron_count, minimum=MIN_NEURON_COUNT)
        self.input_in_degree = _positive_int("input_in_degree", input_in_degree, minimum=1)
        self.recurrent_in_degree = _positive_int(
            "recurrent_in_degree", recurrent_in_degree, minimum=1
        )
        self.seed = _positive_int("seed", seed, minimum=0)
        if not math.isfinite(frame_rate_hz) or frame_rate_hz <= 0:
            raise ValueError(f"frame_rate_hz must be a positive finite number, got {frame_rate_hz}")
        self.frame_rate_hz = float(frame_rate_hz)
        self.dt = 1.0 / self.frame_rate_hz

        resolved_output = (
            _default_output_dim(self.neuron_count)
            if output_dim is None
            else _positive_int("output_dim", output_dim, minimum=1)
        )
        counts = _split_populations(self.neuron_count, resolved_output)
        self.compass_count = counts.compass
        self.steering_count = counts.steering
        self.descending_count = counts.descending
        self.output_dim = counts.descending
        self.descending_start = counts.compass + counts.steering

        wiring = _build_wiring(
            input_dim=self.input_dim,
            counts=counts,
            input_in_degree=self.input_in_degree,
            recurrent_in_degree=self.recurrent_in_degree,
            seed=self.seed,
        )
        self._register_edges("input", wiring["input"])
        self._register_edges("recurrent", wiring["recurrent"])
        self.bias = nn.Parameter(torch.zeros(self.neuron_count))
        self.log_tau = nn.Parameter(torch.full((self.neuron_count,), math.log(DEFAULT_TAU_SECONDS)))
        self.register_buffer("voltage", torch.zeros(self.neuron_count))
        self.to(torch.device(device))

    @property
    def populations(self) -> PopulationCounts:
        """Compass, steering, and descending widths. They sum to ``neuron_count``."""
        return PopulationCounts(self.compass_count, self.steering_count, self.descending_count)

    def trainable_parameter_count(self) -> int:
        """Synapse strengths, biases, and log time-constants. The mask is not counted."""
        return sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)

    def compare_parameters(self) -> ParameterComparison:
        """This module's trainable count next to a dense blank net of the same width."""
        return ParameterComparison(
            neuron_count=self.neuron_count,
            input_dim=self.input_dim,
            output_dim=self.output_dim,
            constrained_trainable=self.trainable_parameter_count(),
            blank_trainable=blank_trainable_parameter_count(self.input_dim, self.neuron_count),
        )

    def input_in_degrees(self) -> np.ndarray:
        """Eye synapses received by each neuron, shape ``(neuron_count,)`` int64."""
        return _in_degrees(self.input_post, self.neuron_count)

    def recurrent_in_degrees(self) -> np.ndarray:
        """In-module synapses received by each neuron, shape ``(neuron_count,)`` int64."""
        return _in_degrees(self.recurrent_post, self.neuron_count)

    def reset(self) -> None:
        """Return every membrane to rest. Call at the start of an episode."""
        self.voltage.zero_()

    def step(self, features: Features) -> Features:
        """Advance one frame. Returns descending activation, ``(output_dim,)`` float32.

        Raises:
            TypeError: If ``features`` is not float32.
            ValueError: If the shape is not ``(input_dim,)`` or a value is non-finite.
        """
        validated = validate_features(features, self.input_dim)
        drive = torch.from_numpy(np.ascontiguousarray(validated)).to(self.voltage.device)
        with torch.no_grad():
            output = self._integrate(drive)
        return np.ascontiguousarray(output.detach().cpu().numpy(), dtype=np.float32)

    def activity(self) -> np.ndarray:
        """Current membrane voltage, ``(neuron_count,)`` float32."""
        return self.voltage.detach().cpu().numpy().astype(np.float32, copy=True)

    def rates(self) -> np.ndarray:
        """Bounded activation ``tanh(voltage)``, ``(neuron_count,)`` float32, in ``(-1, 1)``."""
        activated = torch.tanh(self.voltage)
        return activated.detach().cpu().numpy().astype(np.float32, copy=True)

    def firing_rates_hz(self) -> np.ndarray:
        """Coarse non-negative rate in Hz, capped at the frame rate.

        One step can carry at most one event, so a cell cannot report more than
        ``frame_rate_hz``. The value is the positive part of :meth:`rates` times that
        cap. It is a stability diagnostic, not a spike raster.
        """
        positive = np.maximum(self.rates(), 0.0)
        return (positive * np.float32(self.frame_rate_hz)).astype(np.float32, copy=False)

    def _register_edges(self, name: str, edges: _Edges) -> None:
        self.register_buffer(f"{name}_pre", torch.from_numpy(edges.pre))
        self.register_buffer(f"{name}_post", torch.from_numpy(edges.post))
        weight = nn.Parameter(torch.from_numpy(edges.weight.copy()))
        self.register_parameter(f"{name}_weight", weight)

    def _integrate(self, drive: torch.Tensor) -> torch.Tensor:
        sensory = torch.tanh(drive)
        afferent = torch.zeros(self.neuron_count, dtype=torch.float32, device=drive.device)
        afferent.index_add_(0, self.input_post, self.input_weight * sensory[self.input_pre])
        rate = torch.tanh(self.voltage)
        recurrent = torch.zeros(self.neuron_count, dtype=torch.float32, device=drive.device)
        recurrent.index_add_(
            0, self.recurrent_post, self.recurrent_weight * rate[self.recurrent_pre]
        )
        current = afferent + recurrent + self.bias
        tau = self.log_tau.exp().clamp(MIN_TAU_SECONDS, MAX_TAU_SECONDS)
        decay = torch.exp(-self.dt / tau)
        updated = decay * self.voltage + (1.0 - decay) * current
        self.voltage.copy_(updated.clamp(-VOLTAGE_LIMIT, VOLTAGE_LIMIT))
        return torch.tanh(self.voltage[self.descending_start :])


@dataclass(frozen=True)
class _Edges:
    pre: np.ndarray
    post: np.ndarray
    weight: np.ndarray


def _build_wiring(
    *,
    input_dim: int,
    counts: PopulationCounts,
    input_in_degree: int,
    recurrent_in_degree: int,
    seed: int,
) -> dict[str, _Edges]:
    """Sample the mask. RNG order is input, then steering, then descending; the ring is fixed."""
    rng = np.random.default_rng(seed)
    input_edges = _input_edges(rng, counts.total, input_dim, input_in_degree)
    ring = _ring_edges(counts.compass)
    steering_posts = np.arange(counts.compass, counts.compass + counts.steering, dtype=np.int64)
    steering_pool = np.arange(0, counts.compass + counts.steering, dtype=np.int64)
    steering = _random_edges(
        rng, steering_posts, steering_pool, recurrent_in_degree, exclude_self=True
    )
    descending_posts = np.arange(counts.compass + counts.steering, counts.total, dtype=np.int64)
    descending_pool = np.arange(0, counts.compass + counts.steering, dtype=np.int64)
    descending = _random_edges(
        rng, descending_posts, descending_pool, recurrent_in_degree, exclude_self=False
    )
    recurrent = _concat_edges(ring, steering, descending)
    return {"input": input_edges, "recurrent": recurrent}


def _ring_edges(compass_count: int) -> _Edges:
    offsets = (
        (1, RING_EXCITATION_NEAR),
        (-1, RING_EXCITATION_NEAR),
        (2, RING_EXCITATION_FAR),
        (-2, RING_EXCITATION_FAR),
        (compass_count // 2, RING_INHIBITION),
    )
    pres: list[int] = []
    posts: list[int] = []
    weights: list[float] = []
    for post in range(compass_count):
        for offset, strength in offsets:
            pres.append((post + offset) % compass_count)
            posts.append(post)
            weights.append(strength)
    return _edges(pres, posts, weights)


def _input_edges(
    rng: np.random.Generator, neuron_count: int, input_dim: int, in_degree: int
) -> _Edges:
    take = min(in_degree, input_dim)
    chosen = np.empty((neuron_count, take), dtype=np.int64)
    for neuron in range(neuron_count):
        chosen[neuron] = rng.choice(input_dim, size=take, replace=False)
    pre = chosen.reshape(-1)
    post = np.repeat(np.arange(neuron_count, dtype=np.int64), take)
    weight = rng.normal(0.0, WEIGHT_SCALE / math.sqrt(take), size=pre.shape).astype(np.float32)
    return _Edges(pre=pre, post=post, weight=weight)


def _random_edges(
    rng: np.random.Generator,
    posts: np.ndarray,
    pool: np.ndarray,
    in_degree: int,
    *,
    exclude_self: bool,
) -> _Edges:
    pres: list[np.ndarray] = []
    post_index: list[np.ndarray] = []
    take: int | None = None
    for post in posts:
        candidates = pool[pool != post] if exclude_self else pool
        if candidates.size < 1:
            raise ValueError("synthetic wiring found a neuron with no partners")
        this_take = min(in_degree, int(candidates.size))
        take = this_take if take is None else take
        if this_take != take:
            raise ValueError("in-degree is not uniform; the pools were expected to match")
        pres.append(rng.choice(candidates, size=this_take, replace=False).astype(np.int64))
        post_index.append(np.full(this_take, int(post), dtype=np.int64))
    if take is None:
        raise ValueError("synthetic wiring was asked for an empty population")
    pre = np.concatenate(pres)
    post = np.concatenate(post_index)
    weight = rng.normal(0.0, WEIGHT_SCALE / math.sqrt(take), size=pre.shape).astype(np.float32)
    return _Edges(pre=pre, post=post, weight=weight)


def _concat_edges(*groups: _Edges) -> _Edges:
    return _Edges(
        pre=np.concatenate([group.pre for group in groups]),
        post=np.concatenate([group.post for group in groups]),
        weight=np.concatenate([group.weight for group in groups]),
    )


def _edges(pres: list[int], posts: list[int], weights: list[float]) -> _Edges:
    return _Edges(
        pre=np.asarray(pres, dtype=np.int64),
        post=np.asarray(posts, dtype=np.int64),
        weight=np.asarray(weights, dtype=np.float32),
    )


def _in_degrees(post_index: torch.Tensor, neuron_count: int) -> np.ndarray:
    posts = post_index.detach().cpu().numpy()
    return np.bincount(posts, minlength=neuron_count).astype(np.int64)


def _default_output_dim(neuron_count: int) -> int:
    room = neuron_count - (MIN_COMPASS_COUNT + 1)
    return min(max(4, neuron_count // 5), room)


def _split_populations(neuron_count: int, output_dim: int) -> PopulationCounts:
    remaining = neuron_count - output_dim
    if remaining < MIN_COMPASS_COUNT + 1:
        raise ValueError(
            f"output_dim={output_dim} leaves {remaining} neurons for the compass and "
            f"steering populations; neuron_count={neuron_count} needs at least "
            f"{MIN_COMPASS_COUNT + 1} besides the descending readout"
        )
    compass = max(MIN_COMPASS_COUNT, remaining // 5)
    if compass > remaining - 1:
        compass = remaining - 1
    steering = remaining - compass
    return PopulationCounts(compass=compass, steering=steering, descending=output_dim)


def _positive_int(name: str, value: int, *, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an integer, got {value!r}")
    if value < minimum:
        raise ValueError(f"{name} must be at least {minimum}, got {value}")
    return int(value)
