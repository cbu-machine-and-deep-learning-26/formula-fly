"""Behaviour cloning: the loss, a run on a tiny drive, and its checkpoint through load_driver."""

from __future__ import annotations

import csv
from types import SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from fly_driver.interface import FRAME_RATE_HZ, FRAME_SHAPE  # noqa: E402
from fly_driver.training import (  # noqa: E402
    BrainConfig,
    EnvConfig,
    EyeConfig,
    PolicyConfig,
    TrainConfig,
)
from fly_driver.training.bc import BcConfig, action_loss, train_bc  # noqa: E402
from fly_driver.training.bc_data import HumanLaps  # noqa: E402
from fly_driver.training.train import (  # noqa: E402
    build_brain,
    build_eye,
    build_policy,
    load_driver,
)

pytestmark = pytest.mark.usefixtures("cpu_default_device")

FRAMES = SimpleNamespace(frame_shape=FRAME_SHAPE, frame_rate_hz=FRAME_RATE_HZ)
LAP_FRAMES = 120


def test_action_loss_ignores_errors_the_clip_absorbs() -> None:
    target = torch.tensor([[0.0, 1.0, 0.0]])
    # Brake below 0 for a target of 0 and throttle above 1 for a target of 1 cost nothing.
    assert float(action_loss(torch.tensor([[0.0, 1.4, -0.6]]), target)) == 0.0
    # The same overshoot on the wrong side of a target in range counts in full.
    loss = action_loss(torch.tensor([[0.0, 1.0, 0.6]]), target)
    assert float(loss) == pytest.approx(0.36 / 3)
    mean = torch.tensor([[1.5, 0.5, 0.0]], requires_grad=True)
    action_loss(mean, torch.tensor([[0.5, 0.5, 0.0]])).backward()
    assert mean.grad[0, 0] > 0  # past the steer bound, the gradient still pulls back


def _frames_and_controls(rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Frames bright on one side; the human steers towards the brighter side."""
    left = rng.uniform(0, 255, LAP_FRAMES)
    right = rng.uniform(0, 255, LAP_FRAMES)
    frames = np.zeros((LAP_FRAMES, *FRAME_SHAPE), np.uint8)
    frames[:, :, : FRAME_SHAPE[1] // 2] = left[:, None, None, None].astype(np.uint8)
    frames[:, :, FRAME_SHAPE[1] // 2 :] = right[:, None, None, None].astype(np.uint8)
    steer = (right - left) / 255.0
    controls = np.stack([steer, np.full(LAP_FRAMES, 0.7), np.zeros(LAP_FRAMES)], 1)
    return frames, controls.astype(np.float32)


@pytest.fixture(scope="module")
def run(tmp_path_factory: pytest.TempPathFactory) -> dict:
    """A base checkpoint, a three-lap drive, and a short BC run on it."""
    root = tmp_path_factory.mktemp("bc")
    base = TrainConfig(
        name="base",
        output_dir=str(root / "runs"),
        seeds=(0,),
        device="cpu",
        eye=EyeConfig(type="pixels", params={"downsample": 8}),
        brain=BrainConfig(type="central_complex", params={"neuron_count": 48, "output_dim": 8}),
        policy=PolicyConfig(hidden_sizes=(16,), init_action_mean=(0.0, 0.4, -0.6)),
        env=EnvConfig(type="dummy", max_steps=40),
    )
    base.to_yaml(root / "base.yaml")
    torch.manual_seed(0)
    eye = build_eye(base.eye, FRAMES)
    brain = build_brain(base.brain, eye, FRAMES)
    policy = build_policy(base.policy, brain.output_dim)
    policy.save(root / "warm" / "policy_update_00001.pt")
    torch.save(brain.state_dict(), root / "warm" / "brain_update_00001.pt")

    rng = np.random.default_rng(0)
    laps = [_frames_and_controls(rng) for _ in range(3)]
    HumanLaps(
        frames=np.concatenate([frames for frames, _ in laps]),
        controls=np.concatenate([controls for _, controls in laps]),
        segments=np.repeat(np.arange(3), LAP_FRAMES),
        laps=np.repeat(np.arange(3), LAP_FRAMES),
    ).save(root / "drive.npz")

    config = BcConfig(
        name="bc",
        output_dir=str(root / "runs"),
        base_config=str(root / "base.yaml"),
        init_from=str(root / "warm" / "policy_update_00001.pt"),
        datasets=(str(root / "drive.npz"),),
        holdout_laps=1,
        epochs=25,
        batch_size=32,
        learning_rate=3e-3,
        device="cpu",
    )
    history = train_bc(config, log=lambda message: None)
    return {"root": root, "config": config, "history": history, "laps": laps}


def test_bc_loss_drops_on_a_tiny_drive(run: dict) -> None:
    history = run["history"]
    assert history[-1].train_loss < 0.5 * history[0].train_loss
    with (run["config"].run_dir / "bc_log.csv").open() as handle:
        rows = list(csv.DictReader(handle))
    base_error = float(rows[0]["holdout_mae_steer"])
    best_error = min(float(row["holdout_mae_steer"]) for row in rows)
    assert best_error < 0.5 * base_error
    assert sum(row["is_best"] == "True" for row in rows) >= 2  # the base, then improvements


def test_bc_checkpoint_drives_through_load_driver(run: dict) -> None:
    run_dir = run["config"].run_dir
    config = TrainConfig.from_yaml(run_dir / "config.yaml")
    assert config.init_from is None
    driver = load_driver(config, run_dir / "policy_final.pt", FRAMES)

    # The held-out lap through the loaded driver, against the same files rebuilt by hand.
    frames, controls = run["laps"][-1]
    driver.reset()
    driven = np.array([driver.act(frame).to_array() for frame in frames])
    eye = build_eye(config.eye, FRAMES)
    brain = build_brain(config.brain, eye, FRAMES)
    brain.load_state_dict(torch.load(run_dir / "brain_final.pt", weights_only=True))
    policy = type(driver.policy).load(run_dir / "policy_final.pt")
    eye.reset()
    brain.reset()
    expected = np.array([policy.act(brain.step(eye.encode(frame))).to_array() for frame in frames])
    np.testing.assert_allclose(driven, expected, atol=1e-6)
    # And it learned the task: steer follows the brighter side.
    assert np.corrcoef(driven[:, 0], controls[:, 0])[0, 1] > 0.8


def test_bc_refuses_to_overwrite(run: dict) -> None:
    with pytest.raises(FileExistsError):
        train_bc(run["config"])
