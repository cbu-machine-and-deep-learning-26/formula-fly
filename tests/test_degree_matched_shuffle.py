"""The degree-matched shuffle, tested on the wiring itself (AGENTS.md §11).

No torch and no flyvis: this is the check CI runs. A flyvis network that
merely *called* the shuffle would not show that in-degree and out-degree
survived.
"""

from __future__ import annotations

import numpy as np
import pytest

from fly_driver.eyes.degree_shuffle import degree_matched_shuffle, in_out_degrees


def _random_digraph(n_nodes: int, n_edges: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """A simple directed graph with no self-loops."""
    generator = np.random.default_rng(seed)
    occupied: set[tuple[int, int]] = set()
    sources: list[int] = []
    targets: list[int] = []
    while len(sources) < n_edges:
        source = int(generator.integers(0, n_nodes))
        target = int(generator.integers(0, n_nodes))
        if source == target or (source, target) in occupied:
            continue
        occupied.add((source, target))
        sources.append(source)
        targets.append(target)
    return np.asarray(sources, dtype=np.int64), np.asarray(targets, dtype=np.int64)


def test_every_neurons_in_and_out_degree_is_unchanged() -> None:
    """Per-neuron degrees match, so the in- and out-degree distributions match.

    Isolated neurons are in the count (``n_nodes`` is larger than the edge
    indices) and stay at degree 0. Node 0 is given five extra outgoing edges
    on top of whatever the random graph drew, and that exact out-degree has
    to come back on node 0 -- a shuffle that only preserved the histogram
    could move it onto someone else.
    """
    sources, targets = _random_digraph(n_nodes=20, n_edges=60, seed=0)
    occupied = set(zip(sources.tolist(), targets.tolist(), strict=True))
    extra_targets = [node for node in range(1, 20) if (0, node) not in occupied][:5]
    assert len(extra_targets) == 5
    sources = np.concatenate([sources, np.zeros(len(extra_targets), dtype=np.int64)])
    targets = np.concatenate([targets, np.asarray(extra_targets, dtype=np.int64)])
    n_nodes = 25  # 20..24 are isolated
    before_in, before_out = in_out_degrees(sources, targets, n_nodes)
    assert before_out[0] >= 5
    assert before_out[20:].sum() == 0 and before_in[20:].sum() == 0

    rewired_sources, rewired_targets = degree_matched_shuffle(
        sources, targets, n_nodes=n_nodes, seed=3, attempts=2_000
    )
    after_in, after_out = in_out_degrees(rewired_sources, rewired_targets, n_nodes)
    np.testing.assert_array_equal(after_in, before_in)
    np.testing.assert_array_equal(after_out, before_out)


def test_sparsity_is_the_same_simple_graph() -> None:
    """Same edge count, still no self-loops, still at most one edge per pair."""
    sources, targets = _random_digraph(n_nodes=30, n_edges=100, seed=2)
    rewired_sources, rewired_targets = degree_matched_shuffle(
        sources, targets, n_nodes=30, seed=5, attempts=3_000
    )
    assert rewired_sources.shape == sources.shape
    assert rewired_targets.shape == targets.shape
    assert not np.any(rewired_sources == rewired_targets)
    pairs = list(zip(rewired_sources.tolist(), rewired_targets.tolist(), strict=True))
    assert len(pairs) == len(set(pairs))


def test_the_edge_set_actually_changes() -> None:
    """A shuffle that returns the input has not tested the wiring pattern."""
    sources, targets = _random_digraph(n_nodes=25, n_edges=80, seed=4)
    rewired_sources, rewired_targets = degree_matched_shuffle(
        sources, targets, n_nodes=25, seed=7, attempts=2_000
    )
    before = set(zip(sources.tolist(), targets.tolist(), strict=True))
    after = set(zip(rewired_sources.tolist(), rewired_targets.tolist(), strict=True))
    assert before != after


def test_the_same_seed_repeats_and_a_different_seed_does_not() -> None:
    sources, targets = _random_digraph(n_nodes=20, n_edges=50, seed=8)
    first = degree_matched_shuffle(sources, targets, n_nodes=20, seed=1, attempts=500)
    second = degree_matched_shuffle(sources, targets, n_nodes=20, seed=1, attempts=500)
    third = degree_matched_shuffle(sources, targets, n_nodes=20, seed=2, attempts=500)
    np.testing.assert_array_equal(first[0], second[0])
    np.testing.assert_array_equal(first[1], second[1])
    assert not np.array_equal(first[1], third[1])


def test_zero_attempts_copies_the_edges_without_aliasing() -> None:
    sources = np.array([0, 1], dtype=np.int64)
    targets = np.array([1, 0], dtype=np.int64)
    rewired_sources, rewired_targets = degree_matched_shuffle(sources, targets, seed=0, attempts=0)
    np.testing.assert_array_equal(rewired_sources, sources)
    np.testing.assert_array_equal(rewired_targets, targets)
    rewired_targets[0] = 7
    assert targets[0] == 1


def test_a_directed_cycle_has_no_legal_swap_and_keeps_its_degrees() -> None:
    """Every 2-swap on a 3-cycle would add a self-loop, so the graph stays."""
    sources = np.array([0, 1, 2])
    targets = np.array([1, 2, 0])
    rewired_sources, rewired_targets = degree_matched_shuffle(
        sources, targets, n_nodes=3, seed=0, attempts=100
    )
    np.testing.assert_array_equal(rewired_sources, sources)
    np.testing.assert_array_equal(np.sort(rewired_targets), np.sort(targets))
    before_in, before_out = in_out_degrees(sources, targets, 3)
    after_in, after_out = in_out_degrees(rewired_sources, rewired_targets, 3)
    np.testing.assert_array_equal(after_in, before_in)
    np.testing.assert_array_equal(after_out, before_out)


def test_known_seed_is_pinned() -> None:
    """The swap sequence for this edge list is part of the contract."""
    sources = np.array([0, 2, 1, 3, 0], dtype=np.int64)
    targets = np.array([1, 3, 2, 0, 2], dtype=np.int64)
    rewired_sources, rewired_targets = degree_matched_shuffle(
        sources, targets, n_nodes=4, seed=11, attempts=20
    )
    np.testing.assert_array_equal(rewired_sources, sources)
    # numpy Generator (PCG64) stream for seed 11 and 20 attempts.
    np.testing.assert_array_equal(rewired_targets, np.array([1, 3, 0, 2, 2]))


@pytest.mark.parametrize(
    "kwargs, match",
    [
        ({"sources": [0, 1], "targets": [1], "seed": 0, "attempts": 1}, "same length"),
        ({"sources": [0, -1], "targets": [1, 0], "seed": 0, "attempts": 1}, "0 .."),
        ({"sources": [0], "targets": [0], "seed": 0, "attempts": 1}, "self-loop"),
        (
            {"sources": [0, 0], "targets": [1, 1], "seed": 0, "attempts": 1},
            "at most one",
        ),
        (
            {"sources": [0], "targets": [1], "n_nodes": 1, "seed": 0, "attempts": 1},
            "smaller than",
        ),
        ({"sources": [0], "targets": [1], "seed": -1, "attempts": 1}, "seed"),
        ({"sources": [0], "targets": [1], "seed": 0, "attempts": -1}, "attempts"),
    ],
)
def test_bad_graphs_are_rejected(kwargs: dict[str, object], match: str) -> None:
    with pytest.raises(ValueError, match=match):
        degree_matched_shuffle(**kwargs)  # type: ignore[arg-type]


def test_non_integers_are_rejected() -> None:
    with pytest.raises(TypeError, match="seed"):
        degree_matched_shuffle([0], [1], seed=True, attempts=1)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="attempts"):
        degree_matched_shuffle([0], [1], seed=0, attempts=1.5)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="integer array"):
        degree_matched_shuffle([0.5], [1], seed=0, attempts=1)
