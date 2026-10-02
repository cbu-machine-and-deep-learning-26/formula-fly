"""Flyvis optic lobe with degree-matched shuffled wiring.

The RQ1 shuffled-connectome control (`AGENTS.md` §6). Same neurons, same
synapse parameters, same number of edges, and the same in-degree and
out-degree on every neuron as the frozen pretrained eye. The double-edge swap
in :func:`~fly_driver.eyes.degree_shuffle.degree_matched_shuffle` changes which
postsynaptic neuron each edge lands on, so the comparison isolates the
*pattern* of the wiring rather than sparsity or degree.

flyvis is optional, exactly as for :class:`~fly_driver.eyes.flyvis_eye.FlyvisEye`.
Constructing this eye without it raises :class:`FlyvisNotInstalledError`.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import torch

from fly_driver.eyes.degree_shuffle import degree_matched_shuffle, in_out_degrees
from fly_driver.eyes.flyvis_eye import (
    DEFAULT_CHECKPOINT,
    DEFAULT_MOTION_READOUTS,
    DEFAULT_WARMUP_LUMINANCE,
    DEFAULT_WARMUP_SECONDS,
    FLYVIS_FRAME_RATE_HZ,
    FlyvisEye,
)
from fly_driver.eyes.hex_resampler import DEFAULT_FRAME_SHAPE

__all__ = ["DEFAULT_SWAP_ATTEMPTS_PER_EDGE", "ShuffledConnectomeEye", "rewire_network"]

#: Swap attempts per edge. Most attempts on the optic lobe succeed (it is
#: sparse), so this is several full passes over the edge list.
DEFAULT_SWAP_ATTEMPTS_PER_EDGE = 10


class ShuffledConnectomeEye(FlyvisEye):
    """Frozen flyvis eye whose edges have been degree-preservingly rewired.

    Construction loads the same checkpoint and readout as :class:`FlyvisEye`,
    then rewires ``network._source_indices`` / ``_target_indices`` and the
    node-parameter source/target readers. Those are the two places the
    dynamics read the wiring; synapse strengths stay on the edge they were
    trained on and move with it to the new target.

    ``seed`` fixes the swaps. The default is 0, so a policy-seed sweep shares
    one shuffled connectome unless ``eye.params.seed`` overrides it.

    Args:
        checkpoint: Pretrained checkpoint name or path, as for :class:`FlyvisEye`.
        readouts: Cell types concatenated into the feature vector.
        frame_shape: Declared camera frame shape.
        device: Torch device for the network.
        frame_rate_hz: Environment frame rate; one frame is one integration step.
        warmup_seconds: Grey stimulus duration used by :meth:`reset`.
        warmup_luminance: Grey level in ``[0, 1]`` used by :meth:`reset`.
        seed: Seeds the degree-matched swaps.
        swap_attempts: How many swaps to attempt. ``None`` uses
            ``DEFAULT_SWAP_ATTEMPTS_PER_EDGE`` attempts per edge.

    Raises:
        FlyvisNotInstalledError: If flyvis is not importable.
        FileNotFoundError: If the checkpoint is missing.
        TypeError: If ``seed`` or ``swap_attempts`` is not an integer.
        ValueError: If flyvis settings are invalid, or the rewiring fails to
            change the edge list or to preserve degrees.
        RuntimeError: If every attempted swap was rejected, so the wiring is
            still the biological one.
    """

    def __init__(
        self,
        checkpoint: str = DEFAULT_CHECKPOINT,
        readouts: Sequence[str] = DEFAULT_MOTION_READOUTS,
        frame_shape: tuple[int, int, int] = DEFAULT_FRAME_SHAPE,
        device: str | torch.device | None = None,
        frame_rate_hz: float = FLYVIS_FRAME_RATE_HZ,
        warmup_seconds: float = DEFAULT_WARMUP_SECONDS,
        warmup_luminance: float = DEFAULT_WARMUP_LUMINANCE,
        seed: int = 0,
        swap_attempts: int | None = None,
    ) -> None:
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise TypeError(f"seed must be a non-negative integer, got {seed!r}")
        if seed < 0:
            raise ValueError(f"seed must be a non-negative integer, got {seed}")
        if swap_attempts is not None and (
            isinstance(swap_attempts, bool)
            or not isinstance(swap_attempts, int)
            or swap_attempts < 1
        ):
            raise ValueError(f"swap_attempts must be a positive integer, got {swap_attempts!r}")
        super().__init__(
            checkpoint=checkpoint,
            readouts=readouts,
            frame_shape=frame_shape,
            device=device,
            frame_rate_hz=frame_rate_hz,
            warmup_seconds=warmup_seconds,
            warmup_luminance=warmup_luminance,
        )
        self.seed = int(seed)
        sources = self._edge_index("_source_indices")
        targets = self._edge_index("_target_indices")
        n_nodes = int(self.network.n_nodes)
        n_edges = int(sources.shape[0])
        if swap_attempts is None:
            attempts = n_edges * DEFAULT_SWAP_ATTEMPTS_PER_EDGE
        else:
            attempts = int(swap_attempts)
        rewired_sources, rewired_targets = degree_matched_shuffle(
            sources, targets, n_nodes=n_nodes, seed=self.seed, attempts=attempts
        )
        before_in, before_out = in_out_degrees(sources, targets, n_nodes)
        after_in, after_out = in_out_degrees(rewired_sources, rewired_targets, n_nodes)
        if not (np.array_equal(before_in, after_in) and np.array_equal(before_out, after_out)):
            raise RuntimeError("degree-matched shuffle changed a neuron's in- or out-degree")
        if n_edges > 1 and np.array_equal(rewired_targets, targets):
            raise RuntimeError(
                "degree-matched shuffle left every edge in place; the control would be a "
                "copy of the biological connectome. Increase swap_attempts."
            )
        rewire_network(self.network, rewired_sources, rewired_targets)
        self.in_degree = after_in
        self.out_degree = after_out

    def _edge_index(self, name: str) -> np.ndarray:
        tensor = getattr(self.network, name).detach().to(device="cpu")
        return tensor.numpy().astype(np.int64, copy=True)


def rewire_network(network: object, sources: np.ndarray, targets: np.ndarray) -> None:
    """Point a flyvis ``Network`` at a new edge list.

    Updates the index tensors ``target_sum`` and the state gather read, plus
    the node-parameter ``sources`` / ``targets`` readers captured at init.
    Edge parameters stay put: a synapse keeps the strength it was trained
    with and the swap only changes its target.

    The stored connectome's ``source_index`` / ``target_index`` arrays are
    overwritten when they accept slice assignment. Spatial offsets (``du``,
    ``dv``) on that table are left as the pre-shuffle values; the dynamics do
    not read them after the parameters have been gathered.

    Args:
        network: A flyvis ``Network``, or a test double with the same attributes.
        sources: New presynaptic index per edge.
        targets: New postsynaptic index per edge.

    Raises:
        TypeError: If ``network`` has no index tensors.
        ValueError: If the new edge list disagrees in length with the old one.
    """
    old_sources = getattr(network, "_source_indices", None)
    old_targets = getattr(network, "_target_indices", None)
    if not isinstance(old_sources, torch.Tensor) or not isinstance(old_targets, torch.Tensor):
        raise TypeError("network has no _source_indices/_target_indices tensors to rewire")
    if old_sources.shape != old_targets.shape:
        raise ValueError("network source and target index tensors differ in length")
    source_array = np.asarray(sources, dtype=np.int64)
    target_array = np.asarray(targets, dtype=np.int64)
    if source_array.shape != tuple(old_sources.shape) or target_array.shape != tuple(
        old_targets.shape
    ):
        raise ValueError(
            f"rewiring has {source_array.shape[0]} edges, the network has {old_sources.shape[0]}"
        )
    source_tensor = torch.as_tensor(
        source_array, dtype=old_sources.dtype, device=old_sources.device
    )
    target_tensor = torch.as_tensor(
        target_array, dtype=old_targets.dtype, device=old_targets.device
    )
    network._source_indices = source_tensor
    network._target_indices = target_tensor
    node_params = getattr(network, "node_params", None)
    if node_params is not None:
        for param in node_params.values():
            _retarget_readers(param, source_tensor, target_tensor)
    edges = getattr(getattr(network, "connectome", None), "edges", None)
    _overwrite_index(edges, "source_index", source_array)
    _overwrite_index(edges, "target_index", target_array)


def _retarget_readers(param: object, sources: torch.Tensor, targets: torch.Tensor) -> None:
    readers = getattr(param, "readers", None)
    indices = getattr(param, "indices", None)
    if not isinstance(readers, dict) or not isinstance(indices, torch.Tensor):
        return
    if "sources" in readers:
        readers["sources"] = indices[sources.to(device=indices.device)]
    if "targets" in readers:
        readers["targets"] = indices[targets.to(device=indices.device)]


def _overwrite_index(edges: object, name: str, values: np.ndarray) -> None:
    if edges is None:
        return
    current = getattr(edges, name, None)
    if current is None:
        return
    try:
        if np.asarray(current).shape != values.shape:
            return
        current[:] = values
    except (TypeError, ValueError):
        return
