"""The driving teacher: its targets, its loss, and that it only runs where it can."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from fly_driver.envs.braking_line import SpeedProfile  # noqa: E402
from fly_driver.policies.mlp_policy import MlpPolicy  # noqa: E402
from fly_driver.training import (  # noqa: E402
    EnvConfig,
    EyeConfig,
    PolicyConfig,
    PPOConfig,
    TrainConfig,
)
from fly_driver.training.ppo import PPOLearner, RolloutBuffer  # noqa: E402
from fly_driver.training.teacher import OFF_TARGET, DrivingTeacher, teacher_loss  # noqa: E402
from fly_driver.training.train import train_seed  # noqa: E402


def _teacher(**kwargs) -> DrivingTeacher:
    """A 1,000 m loop: 80 m/s everywhere except a 30 m/s corner at 500-600 m."""
    arclength = np.arange(0.0, 1000.0, 10.0)
    speed = np.where((arclength >= 500) & (arclength < 600), 30.0, 80.0)
    profile = SpeedProfile(
        points=np.zeros((len(arclength), 2)),
        arclength_m=arclength,
        limit_mps=speed,
        speed_mps=speed,
        bands=np.zeros(len(arclength), dtype=np.int64),
    )
    return DrivingTeacher(profile, 1000.0, **kwargs)


def test_it_brakes_over_the_target_and_accelerates_under_it() -> None:
    teacher = _teacher(fraction=1.0, lookahead_s=0.0, gain_mps=5.0)
    targets = teacher.pedal_targets([550.0, 550.0, 100.0, 100.0], [60.0, 32.0, 20.0, 78.0])
    # 60 m/s in the 30 m/s corner: full brake, throttle released.
    np.testing.assert_allclose(targets[0], [OFF_TARGET, 1.0])
    # 2 m/s over: brake 0.4.
    np.testing.assert_allclose(targets[1], [OFF_TARGET, 0.4], atol=1e-6)
    # Far under on the straight: full throttle, brake released.
    np.testing.assert_allclose(targets[2], [1.0, OFF_TARGET])
    # 2 m/s under: throttle 0.4.
    np.testing.assert_allclose(targets[3], [0.4, OFF_TARGET], atol=1e-6)


def test_it_reads_the_profile_ahead_and_wraps_past_the_line() -> None:
    teacher = _teacher(fraction=1.0, lookahead_s=1.0)
    # At 480 m and 50 m/s it reads 530 m: inside the corner, so it already brakes.
    assert teacher.pedal_targets([480.0], [50.0])[0, 1] > 0
    # Without the lookahead the same car sees the straight and keeps accelerating.
    assert _teacher(fraction=1.0, lookahead_s=0.0).pedal_targets([480.0], [50.0])[0, 0] > 0
    # 990 m + 50 m wraps to 40 m, on the straight.
    np.testing.assert_allclose(teacher.target_speed([990.0], [50.0]), [80.0])


def test_the_fraction_scales_the_target() -> None:
    np.testing.assert_allclose(
        _teacher(fraction=0.85, lookahead_s=0.0).target_speed([550.0], [0.0]), [25.5]
    )


def test_the_teacher_loss_pulls_the_mean_pedals_to_its_targets() -> None:
    torch.manual_seed(0)
    policy = MlpPolicy(4, hidden_sizes=(16,), normalize_features=False)
    config = PPOConfig(
        rollout_steps=64,
        minibatches=2,
        epochs=20,
        learning_rate=1e-2,
        teacher_coef=5.0,
        teacher_steer_coef=5.0,
    )
    learner = PPOLearner(policy, config, device="cpu")
    buffer = RolloutBuffer(64, 4)
    buffer.features[:] = np.random.default_rng(0).normal(size=(64, 4))
    buffer.teacher_targets[:] = [0.3, OFF_TARGET, 1.0]  # a little right, brake everywhere
    raw, log_probs, _ = policy.sample(torch.as_tensor(buffer.features))
    buffer.raw_actions[:] = raw.numpy()
    buffer.log_probs[:] = log_probs.numpy()
    buffer._filled = 64  # noqa: SLF001
    before = policy(torch.as_tensor(buffer.features))[0].mean(0)
    stats = learner.update(buffer, update_index=1, total_updates=1)
    after = policy(torch.as_tensor(buffer.features))[0].mean(0)
    target = torch.tensor([0.3, OFF_TARGET, 1.0])
    assert (after - target).abs().sum() < (before - target).abs().sum() / 2
    assert stats.teacher_loss > 0


def test_teacher_off_leaves_the_loss_alone() -> None:
    policy = MlpPolicy(4, hidden_sizes=(8,))
    learner = PPOLearner(policy, PPOConfig(rollout_steps=8, minibatches=1, epochs=1), device="cpu")
    buffer = RolloutBuffer(8, 4)
    buffer._filled = 8  # noqa: SLF001
    assert learner.update(buffer, update_index=1, total_updates=1).teacher_loss == 0.0


def _config(tmp_path: Path, **overrides) -> TrainConfig:
    base = TrainConfig(
        name="teacher",
        output_dir=str(tmp_path / "runs"),
        seeds=(0,),
        total_env_steps=128,
        device="cpu",
        eye=EyeConfig(type="pixels", frozen=True, params={"downsample": 8}),
        policy=PolicyConfig(hidden_sizes=(8,)),
        env=EnvConfig(type="dummy", max_steps=40),
        ppo=PPOConfig(rollout_steps=64, minibatches=4, epochs=2, teacher_coef=0.5),
        num_envs=4,
    )
    return replace(base, **overrides)


def test_the_teacher_needs_the_vectorized_trainer(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="num_envs >= 2"):
        _config(tmp_path, num_envs=1, ppo=PPOConfig(rollout_steps=64, teacher_coef=0.5))


def test_a_vectorized_run_trains_with_the_teacher(tmp_path: Path) -> None:
    result = train_seed(_config(tmp_path), 0)
    assert result.updates == 2


def test_the_loss_is_one_sided_at_released_and_full_pedals() -> None:
    targets = torch.tensor([[OFF_TARGET, 1.0], [0.4, OFF_TARGET]])
    # Further from pressing than "released", and past "full": no loss.
    assert teacher_loss(torch.tensor([[-3.0, 2.5], [0.4, -2.0]]), targets) == 0.0
    # A released pedal drifting up toward pressing, and a full pedal short of 1, cost.
    assert teacher_loss(torch.tensor([[0.0, 1.0], [0.4, OFF_TARGET]]), targets) > 0
    assert teacher_loss(torch.tensor([[OFF_TARGET, 0.5], [0.4, OFF_TARGET]]), targets) > 0
    # A partial press is two-sided.
    assert teacher_loss(torch.tensor([[OFF_TARGET, 1.0], [0.1, OFF_TARGET]]), targets) > 0


def test_it_steers_back_toward_the_centreline_and_its_direction() -> None:
    teacher = _teacher()
    # lateral and heading are positive left; steer +1 is right.
    left_of_centre = teacher.steer_targets([2.0], [0.0], [30.0])[0]
    pointing_left = teacher.steer_targets([0.0], [0.1], [30.0])[0]
    right_of_centre = teacher.steer_targets([-2.0], [0.0], [30.0])[0]
    assert left_of_centre > 0 and pointing_left > 0 and right_of_centre < 0
    assert teacher.steer_targets([0.0], [0.0], [30.0])[0] == 0.0
    assert teacher.steer_targets([0.0], [1.0], [30.0])[0] == 1.0  # clipped at full lock


def test_targets_come_in_the_policys_action_order() -> None:
    targets = _teacher(fraction=1.0, lookahead_s=0.0).targets([550.0], [60.0], [-2.0], [0.0])
    assert targets.shape == (1, 3)
    assert targets[0, 0] < 0  # right of centre: steer left
    np.testing.assert_allclose(targets[0, 1:], [OFF_TARGET, 1.0])  # over the corner's speed


def test_steering_alone_can_be_taught(tmp_path: Path) -> None:
    config = _config(tmp_path, ppo=PPOConfig(rollout_steps=64, teacher_steer_coef=0.5))
    assert config.ppo.uses_teacher


def test_a_negative_init_log_std_caps_a_warm_started_policys_noise(tmp_path: Path) -> None:
    from fly_driver.training.train import warm_start

    saved = MlpPolicy(4, hidden_sizes=(8,), init_log_std=0.2)
    with torch.no_grad():
        saved.log_std[:] = torch.tensor([-2.0, 0.2, 0.1])
    path = saved.save(tmp_path / "policy_update_00001.pt")

    capped = MlpPolicy(4, hidden_sizes=(8,))
    warm_start(path, capped, None, ceiling=-1.2)
    np.testing.assert_allclose(capped.log_std.detach().numpy(), [-2.0, -1.2, -1.2])

    untouched = MlpPolicy(4, hidden_sizes=(8,))
    warm_start(path, untouched, None, ceiling=0.0)
    np.testing.assert_allclose(untouched.log_std.detach().numpy(), [-2.0, 0.2, 0.1], rtol=1e-6)


def test_steps_the_teacher_executed_do_not_move_ppo() -> None:
    torch.manual_seed(0)
    policy = MlpPolicy(4, hidden_sizes=(8,), normalize_features=False)
    learner = PPOLearner(policy, PPOConfig(rollout_steps=16, minibatches=2, epochs=3), device="cpu")
    buffer = RolloutBuffer(16, 4)
    buffer.features[:] = np.random.default_rng(1).normal(size=(16, 4))
    buffer.raw_actions[:] = 5.0  # far from anything the policy would sample
    buffer.advantages[:] = np.random.default_rng(2).normal(size=16)
    buffer.returns[:] = 3.0
    buffer.policy_mask[:] = False
    buffer._filled = 16  # noqa: SLF001
    before = [p.detach().clone() for p in policy.parameters()]
    learner.update(buffer, update_index=1, total_updates=1)
    for old, new in zip(before, policy.parameters(), strict=True):
        assert torch.equal(old, new)


def test_the_env_never_sees_teacher_drive(tmp_path: Path) -> None:
    from fly_driver.training.env_factory import build_env

    env = build_env(EnvConfig(type="dummy", max_steps=10, params={"teacher_drive": {"share": 0.3}}))
    env.close()


def test_teacher_driven_episodes_are_logged_and_never_scored(tmp_path: Path) -> None:
    import csv

    config = _config(
        tmp_path,
        env=EnvConfig(type="dummy", max_steps=20, params={"teacher_drive": {"share": 0.5}}),
        total_env_steps=256,
    )
    result = train_seed(config, 0)
    with (result.run_dir / "episodes.csv").open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    driven = [row for row in rows if row["teacher_driven"] == "True"]
    assert 0 < len(driven) < len(rows)
    assert all(row["start_focused"] == "True" for row in driven)


def test_weights_make_braking_rows_count_more() -> None:
    targets = torch.tensor([[OFF_TARGET, 1.0], [1.0, OFF_TARGET]])
    prediction = torch.tensor([[OFF_TARGET, 0.0], [0.0, OFF_TARGET]])  # both rows 1.0 short
    plain = teacher_loss(prediction, targets)
    assert teacher_loss(prediction, targets, torch.ones(2)) == pytest.approx(float(plain))
    # Only the braking row is wrong: weighting it 5x raises the loss.
    braking_wrong = torch.tensor([[OFF_TARGET, 0.0], [1.0, OFF_TARGET]])
    unweighted = teacher_loss(braking_wrong, targets)
    weighted = teacher_loss(braking_wrong, targets, torch.tensor([5.0, 1.0]))
    assert float(weighted) > float(unweighted)


def test_a_run_with_brake_weights_trains(tmp_path: Path) -> None:
    config = _config(
        tmp_path,
        env=EnvConfig(type="dummy", max_steps=20, params={"teacher_weights": {"brake": 5.0}}),
    )
    assert train_seed(config, 0).updates == 2
