"""Small CNN eye: same width as frozen flyvis, torch optional (GH-15)."""

from __future__ import annotations

import numpy as np
import pytest

from fly_driver.eyes.cnn_eye import CnnEye
from fly_driver.eyes.constants import FLYVIS_DEFAULT_FEATURE_DIM
from fly_driver.eyes.hex_resampler import RGB_LUMA_WEIGHTS
from fly_driver.interface import FRAME_SHAPE, Eye, validate_features

torch = pytest.importorskip("torch")


def test_default_width_matches_the_flyvis_readout() -> None:
    eye = CnnEye(frame_shape=(16, 16, 3))
    assert isinstance(eye, Eye)
    assert eye.feature_dim == FLYVIS_DEFAULT_FEATURE_DIM == 5768
    assert eye.frame_shape == (16, 16, 3)


def test_encode_matches_encode_batch_and_is_finite() -> None:
    eye = CnnEye(frame_shape=(16, 16, 3), seed=1)
    frame = np.arange(16 * 16 * 3, dtype=np.uint8).reshape(16, 16, 3)
    features = validate_features(eye.encode(frame), eye.feature_dim)
    batch = eye.encode_batch(torch.from_numpy(frame).unsqueeze(0))
    np.testing.assert_allclose(features, batch[0].detach().cpu().numpy(), atol=1e-5)
    assert np.all(np.isfinite(features))


def test_the_default_camera_shape_is_the_project_frame() -> None:
    eye = CnnEye(seed=0)
    assert eye.frame_shape == FRAME_SHAPE
    features = eye.encode(np.zeros(FRAME_SHAPE, dtype=np.uint8))
    assert features.shape == (FLYVIS_DEFAULT_FEATURE_DIM,)


def test_seed_fixes_the_weights_and_does_not_follow_the_global_rng() -> None:
    torch.manual_seed(123)
    first = CnnEye(frame_shape=(8, 8, 3), seed=4, feature_dim=16)
    torch.manual_seed(999)
    second = CnnEye(frame_shape=(8, 8, 3), seed=4, feature_dim=16)
    third = CnnEye(frame_shape=(8, 8, 3), seed=5, feature_dim=16)
    assert torch.equal(first.readout.weight, second.readout.weight)
    assert not torch.equal(first.readout.weight, third.readout.weight)
    frame = np.full((8, 8, 3), 30, dtype=np.uint8)
    np.testing.assert_array_equal(first.encode(frame), second.encode(frame))
    first.reset()
    np.testing.assert_array_equal(first.encode(frame), second.encode(frame))


def test_luminance_weights_match_the_resampler() -> None:
    eye = CnnEye(frame_shape=(8, 8, 3), seed=0, feature_dim=4)
    np.testing.assert_allclose(eye._luma.detach().cpu().numpy(), RGB_LUMA_WEIGHTS)


def test_parameters_are_trainable_until_the_loop_freezes_them() -> None:
    eye = CnnEye(frame_shape=(8, 8, 3), seed=0, feature_dim=4)
    assert any(parameter.requires_grad for parameter in eye.parameters())
    eye.requires_grad_(False)
    assert not any(parameter.requires_grad for parameter in eye.parameters())


def test_a_wrong_frame_is_refused() -> None:
    eye = CnnEye(frame_shape=(8, 8, 3), seed=0, feature_dim=4)
    with pytest.raises(ValueError, match="No implicit resize"):
        eye.encode(np.zeros((4, 4, 3), dtype=np.uint8))
    with pytest.raises(TypeError, match="uint8"):
        eye.encode_batch(torch.zeros(1, 8, 8, 3))


@pytest.mark.parametrize(
    "kwargs, match",
    [
        ({"seed": -1}, "seed"),
        ({"feature_dim": 0}, "feature_dim"),
        ({"frame_shape": (8, 8)}, "frame_shape"),
    ],
)
def test_bad_construction_fails_early(kwargs: dict[str, object], match: str) -> None:
    with pytest.raises(ValueError, match=match):
        CnnEye(**kwargs)  # type: ignore[arg-type]
