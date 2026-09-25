"""The pixel smoke eye: block-averaged grey, the interface's checks, nothing hidden (GH-17)."""

from __future__ import annotations

import numpy as np
import pytest

from fly_driver.eyes import PixelEye, frame_to_gray
from fly_driver.interface import FEATURE_DTYPE, FRAME_SHAPE, Eye, validate_features


def test_it_is_an_eye_with_the_dimensions_it_declares():
    eye = PixelEye(frame_shape=FRAME_SHAPE, downsample=4)
    assert isinstance(eye, Eye)
    assert eye.frame_shape == FRAME_SHAPE
    assert eye.feature_dim == (96 // 4) * (96 // 4) == 576


def test_features_are_the_block_means_of_the_resamplers_grey():
    rng = np.random.default_rng(0)
    frame = rng.integers(0, 256, size=FRAME_SHAPE, dtype=np.uint8)
    eye = PixelEye(frame_shape=FRAME_SHAPE, downsample=8)
    features = validate_features(eye.encode(frame), eye.feature_dim)

    grey = frame_to_gray(frame, expected_shape=FRAME_SHAPE)
    expected = grey.reshape(12, 8, 12, 8).mean(axis=(1, 3)).reshape(-1)
    np.testing.assert_allclose(features, expected, atol=1e-6)
    assert features.dtype == FEATURE_DTYPE
    assert features.min() >= 0.0 and features.max() <= 1.0


def test_row_major_layout_and_a_bright_corner():
    frame = np.zeros((16, 32, 3), dtype=np.uint8)
    frame[:8, :8] = 255  # top-left block
    features = PixelEye(frame_shape=(16, 32, 3), downsample=8).encode(frame)
    assert features.shape == (2 * 4,)
    assert features[0] == pytest.approx(1.0) and features[1:].max() == 0.0


def test_a_wrong_frame_is_refused_not_resized():
    eye = PixelEye(frame_shape=FRAME_SHAPE)
    with pytest.raises(ValueError, match="No implicit resize"):
        eye.encode(np.zeros((64, 64, 3), dtype=np.uint8))
    with pytest.raises(TypeError, match="uint8"):
        eye.encode(np.zeros(FRAME_SHAPE, dtype=np.float32))


@pytest.mark.parametrize(
    "kwargs, match",
    [
        ({"downsample": 5}, "does not tile"),
        ({"downsample": 0}, "positive integer"),
        ({"downsample": True}, "positive integer"),
        ({"frame_shape": (96, 96)}, "frame_shape"),
    ],
)
def test_bad_construction_fails_early(kwargs, match):
    with pytest.raises(ValueError, match=match):
        PixelEye(**kwargs)


def test_it_is_stateless():
    eye = PixelEye(frame_shape=(8, 8, 3), downsample=2)
    frame = np.full((8, 8, 3), 90, dtype=np.uint8)
    first = eye.encode(frame)
    eye.reset()
    np.testing.assert_array_equal(eye.encode(frame), first)
