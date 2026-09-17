"""Shuffled connectome preserves in/out degree (CLAUDE.md §11)."""

from __future__ import annotations

import numpy as np

from fly_driver.eyes.shuffled import (
    ShuffledConnectomeEye,
    configuration_edges,
    degree_matched_shuffle,
    degrees,
)


def test_degree_matched_shuffle_preserves_degrees() -> None:
    rng = np.random.default_rng(0)
    n_src, n_dst = 40, 25
    out_d = rng.integers(1, 6, size=n_src, dtype=np.int64)
    in_d = np.zeros(n_dst, dtype=np.int64)
    stubs = int(out_d.sum())
    assignment = rng.integers(0, n_dst, size=stubs)
    for node in assignment:
        in_d[node] += 1
    src, dst = configuration_edges(out_d, in_d, rng)
    out0, in0 = degrees(src, dst, n_src, n_dst)
    src_s, dst_s = degree_matched_shuffle(src, dst, rng, n_swaps=2000)
    out1, in1 = degrees(src_s, dst_s, n_src, n_dst)
    np.testing.assert_array_equal(out0, out1)
    np.testing.assert_array_equal(in0, in1)
    # Pattern actually changed for this stub count.
    assert not np.array_equal(dst, dst_s)


def test_shuffled_eye_encode_is_finite(frame: np.ndarray) -> None:
    eye = ShuffledConnectomeEye(output_size=32, seed=0, mean_degree=3)
    out0, in0 = degrees(eye.sources_original, eye.targets_original, eye._n_in, eye.output_size)
    out1, in1 = degrees(eye.sources, eye.targets, eye._n_in, eye.output_size)
    np.testing.assert_array_equal(out0, out1)
    np.testing.assert_array_equal(in0, in1)
    features = eye.encode(frame)
    assert features.shape == (32,)
    assert np.isfinite(features).all()
