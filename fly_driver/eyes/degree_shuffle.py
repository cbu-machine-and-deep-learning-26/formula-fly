"""Degree-preserving rewiring of a directed connectome.

The RQ1 shuffled-connectome control has to change *who connects to whom* and
leave everything else alone (`AGENTS.md` §6): the same neurons, the same number
of edges, and the same in-degree and out-degree on every neuron. A free
permutation of the edge list would scramble the degree sequence and turn the
control into "a sparser or denser graph" rather than "the same sparsity, a
different pattern."

The rewiring is a directed double-edge swap. Two edges ``a → b`` and ``c → d``
become ``a → d`` and ``c → b`` when that stays a simple graph: no self-loop,
and no second edge onto a pair that already has one. Each swap keeps ``a`` and
``c``'s out-degree and ``b`` and ``d``'s in-degree, so after any number of
swaps the degree of every neuron is the one it started with.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

__all__ = ["degree_matched_shuffle", "in_out_degrees"]


def in_out_degrees(
    sources: npt.ArrayLike, targets: npt.ArrayLike, n_nodes: int
) -> tuple[np.ndarray, np.ndarray]:
    """Count the in-degree and out-degree of every neuron.

    Args:
        sources: Presynaptic index of each edge, shape ``(n_edges,)``.
        targets: Postsynaptic index of each edge, shape ``(n_edges,)``.
        n_nodes: Neurons to count, including ones with no edges. Indices are
            ``0 .. n_nodes - 1``.

    Returns:
        ``(in_degree, out_degree)``, each an int64 vector of length ``n_nodes``.
        In-degree is how many edges arrive at a neuron; out-degree is how many
        leave it.
    """
    source_index = _edge_index("sources", sources)
    target_index = _edge_index("targets", targets)
    if source_index.shape != target_index.shape:
        raise ValueError(
            f"sources and targets must have the same length, got {source_index.shape[0]} "
            f"and {target_index.shape[0]}"
        )
    node_count = _node_count(n_nodes)
    _check_in_range("sources", source_index, node_count)
    _check_in_range("targets", target_index, node_count)
    in_degree = np.bincount(target_index, minlength=node_count).astype(np.int64, copy=False)
    out_degree = np.bincount(source_index, minlength=node_count).astype(np.int64, copy=False)
    return in_degree, out_degree


def degree_matched_shuffle(
    sources: npt.ArrayLike,
    targets: npt.ArrayLike,
    *,
    n_nodes: int | None = None,
    seed: int,
    attempts: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Rewire a directed edge list without changing any neuron's degree.

    Args:
        sources: Presynaptic index of each edge.
        targets: Postsynaptic index of each edge.
        n_nodes: Neurons in the graph. Defaults to one past the largest index,
            which drops isolated neurons above that index; pass the real count
            to keep them in the degree vectors you compare against.
        seed: Seeds the swaps. The same seed and the same edge list produce the
            same rewiring.
        attempts: How many swap *attempts* to make. Rejected attempts (they
            would add a self-loop or a duplicate edge) still count. Zero
            attempts returns a copy of the input.

    Returns:
        New ``(sources, targets)`` int64 arrays. ``sources`` is unchanged,
        because each swap exchanges the two postsynaptic partners; both degree
        sequences match the input exactly.

    Raises:
        TypeError: If an index array is not an integer vector, or ``seed`` /
            ``attempts`` is not an integer.
        ValueError: If the edges disagree in length, an index is out of range,
            ``seed`` or ``attempts`` is negative, or the graph already has a
            self-loop or a duplicate pair.
    """
    source_index = _edge_index("sources", sources)
    target_index = _edge_index("targets", targets)
    if source_index.shape != target_index.shape:
        raise ValueError(
            f"sources and targets must have the same length, got {source_index.shape[0]} "
            f"and {target_index.shape[0]}"
        )
    swap_attempts = _attempt_count(attempts)
    swap_seed = _seed(seed)
    n_edges = int(source_index.shape[0])
    if n_edges == 0:
        if n_nodes is not None:
            _node_count(n_nodes)
        return source_index.copy(), target_index.copy()

    observed = int(max(int(source_index.max()), int(target_index.max())) + 1)
    node_count = observed if n_nodes is None else _node_count(n_nodes)
    if node_count < observed:
        raise ValueError(
            f"n_nodes={node_count} is smaller than an edge index (max index is {observed - 1})"
        )
    _check_in_range("sources", source_index, node_count)
    _check_in_range("targets", target_index, node_count)

    if np.any(source_index == target_index):
        raise ValueError("degree-matched shuffle expects no self-loops in the input graph")

    # Packing a pair into one int keeps the membership test one set of ints
    # rather than a set of tuples. n_nodes^2 has to fit in int64; the optic
    # lobe is ~45k neurons, so the product is ~2e9.
    if node_count > np.iinfo(np.int64).max // node_count:
        raise ValueError(f"n_nodes={node_count} is too large to pack an edge key")
    span = np.int64(node_count)
    keys = source_index * span + target_index
    if np.unique(keys).shape[0] != n_edges:
        raise ValueError(
            "degree-matched shuffle expects at most one directed edge per ordered pair"
        )

    rewired_sources = source_index.copy()
    rewired_targets = target_index.copy()
    if swap_attempts == 0:
        return rewired_sources, rewired_targets

    # Python ints, not numpy scalars: a numpy int64 does not always share a
    # hash with the Python int of the same value, and the membership test
    # would then miss an edge that is already there.
    occupied = {int(key) for key in keys.tolist()}
    source_of = [int(source) for source in rewired_sources.tolist()]
    generator = np.random.default_rng(swap_seed)
    # Draw in chunks so a multi-million-edge connectome does not allocate the
    # whole attempt list at once. The accept/reject step has to be serial: each
    # swap changes which pairs are occupied.
    remaining = swap_attempts
    chunk = 100_000
    while remaining:
        draw = min(chunk, remaining)
        left = generator.integers(0, n_edges, size=draw, dtype=np.int64)
        right = generator.integers(0, n_edges, size=draw, dtype=np.int64)
        _swap_chunk(source_of, rewired_targets, occupied, int(span), left, right)
        remaining -= draw
    return rewired_sources, rewired_targets


def _swap_chunk(
    source_of: list[int],
    targets: np.ndarray,
    occupied: set[int],
    span: int,
    left: np.ndarray,
    right: np.ndarray,
) -> None:
    """Apply one batch of attempted swaps in place."""
    # ``targets`` is mutated, so it is read from the array, not snapshotted.
    for edge_i, edge_j in zip(left.tolist(), right.tolist(), strict=True):
        if edge_i == edge_j:
            continue
        source_i = source_of[edge_i]
        source_j = source_of[edge_j]
        target_i = int(targets[edge_i])
        target_j = int(targets[edge_j])
        # Same source or same target: exchanging partners does not change the
        # edge set (or it only reorders two identical rows).
        if source_i == source_j or target_i == target_j:
            continue
        # a → d and c → b. A self-loop here would change no degree but would
        # invent a neuron that synapses onto itself, which the optic lobe
        # graph does not have.
        if source_i == target_j or source_j == target_i:
            continue
        key_i = source_i * span + target_i
        key_j = source_j * span + target_j
        new_i = source_i * span + target_j
        new_j = source_j * span + target_i
        if new_i in occupied or new_j in occupied:
            continue
        occupied.remove(key_i)
        occupied.remove(key_j)
        occupied.add(new_i)
        occupied.add(new_j)
        targets[edge_i] = target_j
        targets[edge_j] = target_i


def _edge_index(name: str, values: npt.ArrayLike) -> np.ndarray:
    array = np.asarray(values)
    if array.dtype.kind not in "iu":
        raise TypeError(f"{name} must be an integer array, got dtype {array.dtype}")
    if array.ndim != 1:
        raise ValueError(f"{name} must be a 1-d edge list, got shape {array.shape}")
    return array.astype(np.int64, copy=False)


def _node_count(n_nodes: int) -> int:
    if isinstance(n_nodes, bool) or not isinstance(n_nodes, int) or n_nodes < 0:
        raise ValueError(f"n_nodes must be a non-negative integer, got {n_nodes!r}")
    return n_nodes


def _seed(seed: int) -> int:
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError(f"seed must be a non-negative integer, got {seed!r}")
    if seed < 0:
        raise ValueError(f"seed must be a non-negative integer, got {seed}")
    return seed


def _attempt_count(attempts: int) -> int:
    if isinstance(attempts, bool) or not isinstance(attempts, int):
        raise TypeError(f"attempts must be a non-negative integer, got {attempts!r}")
    if attempts < 0:
        raise ValueError(f"attempts must be a non-negative integer, got {attempts}")
    return attempts


def _check_in_range(name: str, index: np.ndarray, n_nodes: int) -> None:
    if index.size and (int(index.min()) < 0 or int(index.max()) >= n_nodes):
        raise ValueError(f"{name} must be in 0 .. {n_nodes - 1}")
