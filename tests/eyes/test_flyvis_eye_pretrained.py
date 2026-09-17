"""Pretrained-checkpoint tests for ``FlyvisEye`` (skipped without flyvis + data)."""

from __future__ import annotations

import numpy as np
import pytest

from fly_driver.eyes.flyvis_eye import (
    DEFAULT_MOTION_READOUTS,
    FlyvisEye,
    resolve_checkpoint_dir,
)
from fly_driver.eyes.hex_resampler import (
    DEFAULT_FRAME_SHAPE,
    HEX_COLUMN_COUNT,
    hex_coordinates,
)
from fly_driver.eyes.stimuli import (
    DIRECTIONS,
    drifting_grating_frames,
    grey_frames,
    moving_edge_frames,
)
from tests.eyes.direction_selectivity import (
    DRIFT_FRAMES,
    PRESTIMULUS_FRAMES,
    assert_t4_t5_direction_selectivity,
    measure_q95_amplitudes,
)

pytest.importorskip("flyvis")

STABILITY_SECONDS = 10
# Observed |activity| peaks near 6 a.u. for gratings, edges, and noise; a blow-up
# passes this limit within a few frames.
PLAUSIBLE_ACTIVITY_LIMIT = 20.0


@pytest.fixture(scope="module")
def eye() -> FlyvisEye:
    """Load the default frozen eye once for the module."""
    try:
        resolve_checkpoint_dir()
    except FileNotFoundError:
        pytest.skip("run `flyvis download-pretrained` to enable these tests")
    return FlyvisEye()


def _random_frames(frame_count: int, seed: int) -> np.ndarray:
    """Mix pixel noise with coarse random blocks so both pathways get driven."""
    generator = np.random.default_rng(seed)
    height, width, _ = DEFAULT_FRAME_SHAPE
    noise = generator.integers(0, 256, size=(frame_count // 2, height, width, 3))
    block = 8
    coarse = generator.integers(
        0,
        256,
        size=(frame_count - frame_count // 2, height // block, width // block, 3),
    )
    blocks = coarse.repeat(block, axis=1).repeat(block, axis=2)
    return np.concatenate([noise, blocks]).astype(np.uint8)


def test_weights_are_frozen(eye: FlyvisEye) -> None:
    """Expose zero trainable parameters and accumulate no gradients."""
    assert eye.trainable_parameters() == 0
    assert eye.parameter_count() > 0
    assert not eye.network.training
    assert all(not parameter.requires_grad for parameter in eye.network.parameters())

    eye.encode_sequence(grey_frames(2))

    assert all(parameter.grad is None for parameter in eye.network.parameters())


def test_feature_layout(eye: FlyvisEye) -> None:
    """Concatenate one 721-column map per readout in flyvis hex order."""
    assert eye.readout_names == DEFAULT_MOTION_READOUTS
    assert eye.feature_dim == len(DEFAULT_MOTION_READOUTS) * HEX_COLUMN_COUNT
    assert eye.frame_shape == DEFAULT_FRAME_SHAPE
    assert len(eye.available_readouts) == 34
    assert eye.dt == pytest.approx(1 / 50)

    features = eye.encode(grey_frames(1)[0])
    assert isinstance(features, np.ndarray)
    assert features.dtype == np.float32
    assert features.shape == (eye.feature_dim,)

    expected_u, expected_v = hex_coordinates()
    nodes = eye.network.connectome.nodes
    for name in eye.readout_names:
        indices = nodes.layer_index[name][:]
        assert np.array_equal(nodes.u[:][indices], expected_u)
        assert np.array_equal(nodes.v[:][indices], expected_v)


def test_streaming_matches_sequence(eye: FlyvisEye) -> None:
    """Frame-by-frame encoding must equal one encode_sequence on the same frames."""
    frames = moving_edge_frames("right")

    sequence_features = eye.encode_sequence(frames)
    eye.reset()
    streamed_features = np.stack([eye.encode(frame) for frame in frames])

    assert sequence_features.shape == (len(frames), eye.feature_dim)
    assert np.allclose(streamed_features, sequence_features, atol=1e-6)
    assert np.abs(sequence_features).max() > 0.05


def test_reset_is_deterministic(eye: FlyvisEye) -> None:
    """Return the same features for the same frames after every reset."""
    frames = moving_edge_frames("down", prestimulus_frames=5, sweep_frames=20)

    first = eye.encode_sequence(frames)
    eye.encode_sequence(_random_frames(20, seed=3), reset=False)
    second = eye.encode_sequence(frames)

    assert np.array_equal(first, second)


def test_warmup_reaches_resting_state(eye: FlyvisEye) -> None:
    """Leave only a small transient after the default grey warm-up."""
    eye.reset()
    warm_state = eye.state_activity
    assert warm_state is not None

    eye.encode_sequence(grey_frames(100), reset=False)
    settled_state = eye.state_activity
    assert settled_state is not None

    assert np.abs(warm_state - settled_state).max() < 0.05


def test_t4_t5_direction_selectivity_through_eye(eye: FlyvisEye) -> None:
    """Opposite T4/T5 subtypes must prefer opposite grating motion end to end."""
    readout_offsets = {
        name: index * HEX_COLUMN_COUNT for index, name in enumerate(eye.readout_names)
    }
    responses = {
        direction: eye.encode_sequence(
            drifting_grating_frames(
                direction,
                prestimulus_frames=PRESTIMULUS_FRAMES,
                drift_frames=DRIFT_FRAMES,
            )
        )
        for direction in DIRECTIONS
    }

    def readout_response(direction: str, readout: str) -> np.ndarray:
        start = readout_offsets[readout]
        return responses[direction][:, start : start + HEX_COLUMN_COUNT]

    assert_t4_t5_direction_selectivity(measure_q95_amplitudes(readout_response))


def test_ten_seconds_of_noise_stays_finite_and_bounded(eye: FlyvisEye) -> None:
    """Keep every neuron finite and within plausible bounds over a long rollout."""
    frame_count = int(STABILITY_SECONDS * eye.frame_rate_hz)
    features = eye.encode_sequence(_random_frames(frame_count, seed=0))
    state = eye.state_activity

    assert features.shape == (frame_count, eye.feature_dim)
    assert np.isfinite(features).all()
    assert state is not None and np.isfinite(state).all()
    assert np.abs(features).max() < PLAUSIBLE_ACTIVITY_LIMIT
    assert np.abs(state).max() < PLAUSIBLE_ACTIVITY_LIMIT
    assert features.std() > 1e-3


def test_readouts_are_validated_against_output_types() -> None:
    """Reject unknown cell types and honour a custom readout subset."""
    try:
        resolve_checkpoint_dir()
    except FileNotFoundError:
        pytest.skip("run `flyvis download-pretrained` to enable these tests")

    with pytest.raises(ValueError, match="unknown readout"):
        FlyvisEye(readouts=("T4a", "Mi1"))

    custom_eye = FlyvisEye(readouts=("Tm9", "T4a"))
    assert custom_eye.readout_names == ("Tm9", "T4a")
    assert custom_eye.feature_dim == 2 * HEX_COLUMN_COUNT
    assert custom_eye.encode(grey_frames(1)[0]).shape == (2 * HEX_COLUMN_COUNT,)


def test_frame_contract_is_enforced(eye: FlyvisEye) -> None:
    """Reject wrong shapes and dtypes instead of resizing or casting."""
    with pytest.raises(ValueError, match="No implicit resize"):
        eye.encode(np.zeros((95, 96, 3), dtype=np.uint8))
    with pytest.raises(TypeError, match="uint8"):
        eye.encode(np.zeros(DEFAULT_FRAME_SHAPE, dtype=np.float32))
    with pytest.raises(ValueError, match="time, height, width"):
        eye.encode_sequence(np.zeros(DEFAULT_FRAME_SHAPE, dtype=np.uint8))
    with pytest.raises(ValueError, match="at least one frame"):
        eye.encode_sequence(np.zeros((0, *DEFAULT_FRAME_SHAPE), dtype=np.uint8))
