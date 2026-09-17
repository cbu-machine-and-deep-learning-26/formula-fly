"""Interface contract: shapes, dtypes, ranges, no silent resize."""

from __future__ import annotations

import numpy as np
import pytest

from fly_driver.eyes import RandomProjectionEye, SmallCnnEye, build_eye
from fly_driver.interface import (
    BRAKE_RANGE,
    DEFAULT_FRAME_SHAPE,
    STEER_RANGE,
    THROTTLE_RANGE,
    ControlVector,
    DirectDriveAgent,
    require_frame,
)
from fly_driver.policies import LinearPolicy, RandomPolicy


def test_control_vector_clips_to_carracing_ranges() -> None:
    raw = ControlVector(steer=4.0, throttle=-2.0, brake=9.0).clipped()
    assert STEER_RANGE[0] <= raw.steer <= STEER_RANGE[1]
    assert THROTTLE_RANGE[0] <= raw.throttle <= THROTTLE_RANGE[1]
    assert BRAKE_RANGE[0] <= raw.brake <= BRAKE_RANGE[1]
    arr = raw.as_array()
    assert arr.dtype == np.float32
    assert arr.shape == (3,)


def test_require_frame_rejects_wrong_shape() -> None:
    with pytest.raises(ValueError, match="no silent resize"):
        require_frame(np.zeros((64, 64, 3), dtype=np.uint8))
    with pytest.raises(ValueError, match="no silent"):
        require_frame(np.zeros((96, 96), dtype=np.uint8))


def test_random_projection_encode_shape_dtype(frame: np.ndarray) -> None:
    eye = RandomProjectionEye(output_size=32, seed=0)
    features = eye.encode(frame)
    assert features.shape == (32,)
    assert features.dtype == np.float32
    assert np.isfinite(features).all()


def test_cnn_encode_matches_output_size(frame: np.ndarray) -> None:
    eye = SmallCnnEye(output_size=32, seed=1)
    features = eye.encode(frame)
    assert features.shape == (eye.output_size,)
    assert features.dtype == np.float32
    assert np.isfinite(features).all()


def test_direct_drive_agent_returns_clipped_controls(frame: np.ndarray) -> None:
    eye = RandomProjectionEye(output_size=64, seed=2)
    policy = LinearPolicy(feature_size=eye.output_size, seed=2)
    agent = DirectDriveAgent(eye, policy)
    action = agent.act(frame)
    assert STEER_RANGE[0] <= action.steer <= STEER_RANGE[1]
    assert THROTTLE_RANGE[0] <= action.throttle <= THROTTLE_RANGE[1]
    assert BRAKE_RANGE[0] <= action.brake <= BRAKE_RANGE[1]


def test_random_policy_stays_in_range(frame: np.ndarray) -> None:
    eye = RandomProjectionEye(output_size=16, seed=3)
    agent = DirectDriveAgent(eye, RandomPolicy(feature_size=16, seed=3))
    for _ in range(8):
        action = agent.act(frame)
        assert STEER_RANGE[0] <= action.steer <= STEER_RANGE[1]


def test_build_eye_registry_matched_output_size(frame: np.ndarray) -> None:
    size = 48
    for kind in ("cnn", "random_projection", "shuffled"):
        eye = build_eye(kind, output_size=size, seed=0)
        assert eye.output_size == size
        assert eye.input_shape == DEFAULT_FRAME_SHAPE
        out = eye.encode(frame)
        assert out.shape == (size,)


def test_build_eye_unknown_type() -> None:
    with pytest.raises(ValueError, match="Unknown eye_type"):
        build_eye("retina")
