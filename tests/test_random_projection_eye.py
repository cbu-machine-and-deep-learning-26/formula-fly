"""Fixed random-projection eye: numpy only, flyvis readout width (GH-15)."""

from __future__ import annotations

import numpy as np
import pytest

from fly_driver.drivers import DirectDriveAgent
from fly_driver.eyes.constants import FLYVIS_DEFAULT_FEATURE_DIM
from fly_driver.eyes.random_projection_eye import RandomProjectionEye
from fly_driver.interface import FRAME_SHAPE, ControlVector, Eye, validate_features


def test_default_eye_matches_the_flyvis_readout_width() -> None:
    eye = RandomProjectionEye()
    assert isinstance(eye, Eye)
    assert eye.frame_shape == FRAME_SHAPE
    assert eye.feature_dim == FLYVIS_DEFAULT_FEATURE_DIM == 5768


def test_a_white_pixel_returns_that_row_of_the_matrix() -> None:
    """Row-major grey times the frozen matrix. White is luminance 1."""
    eye = RandomProjectionEye(frame_shape=(4, 5, 3), seed=0, feature_dim=7)
    frame = np.zeros((4, 5, 3), dtype=np.uint8)
    frame[1, 2] = 255
    features = validate_features(eye.encode(frame), eye.feature_dim)
    np.testing.assert_allclose(features, eye._projection[1 * 5 + 2], atol=1e-6)


def test_a_red_pixel_uses_the_bt601_red_weight() -> None:
    eye = RandomProjectionEye(frame_shape=(2, 2, 3), seed=4, feature_dim=3)
    frame = np.zeros((2, 2, 3), dtype=np.uint8)
    frame[0, 1, 0] = 255
    features = eye.encode(frame)
    np.testing.assert_allclose(features, eye._projection[1] * np.float32(0.299), atol=1e-5)


def test_the_matrix_is_fixed_by_its_seed() -> None:
    frame = np.full((3, 3, 3), 40, dtype=np.uint8)
    first = RandomProjectionEye(frame_shape=(3, 3, 3), seed=2, feature_dim=5)
    second = RandomProjectionEye(frame_shape=(3, 3, 3), seed=2, feature_dim=5)
    other = RandomProjectionEye(frame_shape=(3, 3, 3), seed=3, feature_dim=5)
    np.testing.assert_array_equal(first.encode(frame), second.encode(frame))
    assert not np.array_equal(first.encode(frame), other.encode(frame))
    first.reset()
    np.testing.assert_array_equal(first.encode(frame), second.encode(frame))


def test_it_drives_through_the_shared_contract() -> None:
    """The policy head sees 5768 features, the same width flyvis would emit."""
    eye = RandomProjectionEye(frame_shape=(8, 8, 3), seed=0)

    class HoldStill:
        feature_dim = eye.feature_dim

        def reset(self, seed: int | None = None) -> None:
            del seed

        def act(self, features: np.ndarray) -> ControlVector:
            assert features.shape == (eye.feature_dim,)
            return ControlVector(steer=0.0, throttle=0.2, brake=0.0)

    driver = DirectDriveAgent(eye, HoldStill())
    driver.reset(seed=0)
    control = driver.act(np.zeros((8, 8, 3), dtype=np.uint8))
    assert control.throttle == pytest.approx(0.2)


def test_a_wrong_frame_is_refused() -> None:
    eye = RandomProjectionEye(frame_shape=(8, 8, 3), seed=0, feature_dim=4)
    with pytest.raises(ValueError, match="No implicit resize"):
        eye.encode(np.zeros((4, 4, 3), dtype=np.uint8))
    with pytest.raises(TypeError, match="uint8"):
        eye.encode(np.zeros((8, 8, 3), dtype=np.float32))


@pytest.mark.parametrize(
    "kwargs, match",
    [
        ({"frame_shape": (8, 8)}, "frame_shape"),
        ({"frame_shape": (0, 8, 3)}, "frame_shape"),
        ({"seed": -1}, "seed"),
        ({"feature_dim": 0}, "feature_dim"),
        ({"feature_dim": True}, "feature_dim"),
    ],
)
def test_bad_construction_fails_early(kwargs: dict[str, object], match: str) -> None:
    arguments = {"frame_shape": (8, 8, 3), **kwargs}
    with pytest.raises(ValueError, match=match):
        RandomProjectionEye(**arguments)  # type: ignore[arg-type]


def test_a_non_integer_seed_is_a_type_error() -> None:
    with pytest.raises(TypeError, match="seed"):
        RandomProjectionEye(frame_shape=(8, 8, 3), seed=True)  # type: ignore[arg-type]
