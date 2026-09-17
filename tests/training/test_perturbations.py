"""Tests for the robustness perturbation wrappers."""

from __future__ import annotations

import numpy as np
import pytest

from fly_driver.envs import DummyTrackEnv
from fly_driver.training.perturbations import (
    PERTURBATIONS,
    ActionDelay,
    Brightness,
    ObservationNoise,
    PerturbationSpec,
    apply_perturbation,
)


def test_observation_noise_is_seeded_and_bounded() -> None:
    """Same seed gives the same noisy frame; values stay uint8 in range."""
    frames = []
    for _ in range(2):
        env = ObservationNoise(DummyTrackEnv(), std=0.2)
        frame, _ = env.reset(seed=4)
        frames.append(frame)
    np.testing.assert_array_equal(frames[0], frames[1])
    clean, _ = DummyTrackEnv().reset(seed=4)
    assert frames[0].dtype == np.uint8 and not np.array_equal(frames[0], clean)

    other, _ = ObservationNoise(DummyTrackEnv(), std=0.2).reset(seed=5)
    assert not np.array_equal(frames[0], other)


def test_brightness_scales_and_clips() -> None:
    """Half brightness halves the frame; large scales saturate at 255."""
    clean, _ = DummyTrackEnv().reset(seed=0)
    dark, _ = Brightness(DummyTrackEnv(), scale=0.5).reset(seed=0)
    np.testing.assert_array_equal(
        dark, (clean.astype(np.float64) * 0.5).astype(np.uint8)
    )
    bright, _ = Brightness(DummyTrackEnv(), scale=10.0).reset(seed=0)
    assert bright.max() == 255 and bright.dtype == np.uint8


def test_action_delay_shifts_actions_by_n_steps() -> None:
    """The env sees neutral actions first, then the agent's actions in order."""
    seen: list[np.ndarray] = []

    class RecordingEnv:
        def reset(self, *, seed: int | None = None) -> tuple[np.ndarray, dict]:
            return np.zeros((2, 2, 3), dtype=np.uint8), {}

        def step(
            self, action: np.ndarray
        ) -> tuple[np.ndarray, float, bool, bool, dict]:
            seen.append(np.asarray(action, dtype=np.float32))
            return np.zeros((2, 2, 3), dtype=np.uint8), 0.0, False, False, {}

        def render(self) -> None:
            return None

    env = ActionDelay(RecordingEnv(), steps=2)
    env.reset(seed=0)
    for value in (0.1, 0.2, 0.3):
        env.step(np.array([value, 0.0, 0.0], dtype=np.float32))

    assert [float(action[0]) for action in seen] == pytest.approx([0.0, 0.0, 0.1])


def test_non_frame_observations_are_rejected() -> None:
    """Perturbations only make sense on uint8 frames."""

    class VectorEnv:
        def reset(self, *, seed: int | None = None) -> tuple[np.ndarray, dict]:
            return np.zeros(3, dtype=np.float32), {}

    with pytest.raises(TypeError):
        Brightness(VectorEnv(), scale=1.0).reset(seed=0)


def test_spec_parsing_labels_and_registry() -> None:
    """Specs round-trip through dicts and build the registered wrapper."""
    spec = PerturbationSpec.from_dict({"name": "action_delay", "steps": 3})
    assert spec.label == "action_delay(steps=3)"
    assert spec.to_dict() == {"name": "action_delay", "steps": 3}
    wrapped = apply_perturbation(DummyTrackEnv(), spec)
    assert isinstance(wrapped, ActionDelay) and wrapped.steps == 3
    assert set(PERTURBATIONS) == {"observation_noise", "brightness", "action_delay"}

    with pytest.raises(ValueError, match="unknown perturbation"):
        PerturbationSpec("fog")
    with pytest.raises(ValueError, match="name"):
        PerturbationSpec.from_dict({"std": 0.1})


@pytest.mark.parametrize(
    "factory",
    [
        lambda: ObservationNoise(DummyTrackEnv(), std=-1.0),
        lambda: Brightness(DummyTrackEnv(), scale=-0.5),
        lambda: ActionDelay(DummyTrackEnv(), steps=-1),
    ],
)
def test_negative_parameters_raise(factory: object) -> None:
    """Parameter validation happens at construction."""
    with pytest.raises(ValueError):
        factory()  # type: ignore[operator]
