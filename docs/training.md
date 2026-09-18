# Training on the practice track

Reinforcement learning on the MuJoCo practice track (GH-17): a frozen eye (or a control
eye) plus a small policy head, trained with PPO. One YAML file is one experiment
condition -- eye type, brain, frozen or fine-tuned, seeds -- and every run writes its
per-term rewards to CSV. Weights & Biases is optional and off by default.

Lives in `fly_driver.training` (config, loggers, `ppo`, `train`),
`fly_driver.policies.mlp_policy` (the head), and `fly_driver.eyes.pixel_eye` (the smoke
eye). The entry point is `scripts/train.py`.

## Run it

```bash
python scripts/train.py                                   # default condition, seeds 0, 1, 2
python scripts/train.py --seed 0                          # one seed; put the others on other machines
python scripts/train.py --config conditions/cnn.yaml      # another condition
python scripts/train.py --env dummy --eye pixels --total-env-steps 20000 --name smoke
python scripts/train.py --wandb                           # CSV as always, plus a W&B mirror
```

Or from code:

```python
from fly_driver.training import TrainConfig
from fly_driver.training.train import train, train_seed

config = TrainConfig.from_yaml("fly_driver/configs/train_default.yaml")
results = train(config)              # every seed in config.seeds, one after another
result = train_seed(config, 0)       # just one
```

### Which virtualenv

Training needs torch, and the default condition needs the flyvis stack, so the natural
place to run it is the flyvis virtualenv from
[`running-the-stacks.md`](running-the-stacks.md) with the project installed into it:

```bash
source .venv-flyvis/bin/activate
python -m pip install -e ".[dev]"          # mujoco, pyyaml and the rest of the base install
export FLYVIS_ROOT_DIR="$HOME/.cache/flyvis"
python scripts/train.py --seed 0
```

torch is deliberately not in `pyproject.toml` (see the comment there). On a machine that
cannot install flyvis -- Python 3.13, or no Python 3.12 available -- the whole loop still
runs with the pixel eye from any venv that has torch:

```bash
py -3.13 -m venv .venv-train                # Windows; python3 -m venv on Linux/macOS
.venv-train/Scripts/python.exe -m pip install -e ".[dev]" torch   # CPU torch is enough
.venv-train/Scripts/python.exe scripts/train.py --eye pixels --env practice_track
```

`.venv-*` directories are git-ignored. W&B is `pip install wandb`, only if you turn it on.

## The condition is the config

`fly_driver/configs/train_default.yaml` is the locked-in default from `AGENTS.md` §6:
frozen pretrained flyvis eye, no brain, practice track, seeds 0, 1 and 2. Every key maps
to `fly_driver.training.TrainConfig`; **unknown keys are an error at every level**, so a
misspelt hyperparameter cannot silently keep its default.

| Key | Meaning |
|---|---|
| `name`, `output_dir` | Runs land in `<output_dir>/<name>/seed_<k>/` |
| `seeds` | The seeds this condition runs, each a complete independent run. Default `[0, 1, 2]` |
| `total_env_steps` | Env steps per seed, rounded down to whole rollouts |
| `device` | `auto` (CUDA if available), `cpu`, or a torch device string |
| `eye.type` | `flyvis` (default) or `pixels`. GH-15 registers `cnn`, `random_projection` and the shuffled connectome |
| `eye.frozen` | `true` trains only the head. `false` puts the eye's parameters in the optimiser; see below |
| `eye.params` | Constructor keyword arguments (`readouts`, `checkpoint`, `downsample`...). `frame_shape` and `frame_rate_hz` come from the env and may not be set here |
| `brain` | `none` (the policy reads the eye directly). GH-23 registers the whole-brain model |
| `policy` | `hidden_sizes`, `activation`, `normalize_features`, `init_log_std` |
| `env.type` | `practice_track` or `dummy` |
| `env.max_steps` | Episode truncation in frames. 15 000 = 5 minutes at 50 Hz, room for a slow clean lap |
| `env.params` | For the practice track: `racing_line` (turn it **off** for the RQ1 comparisons; the env's docstring says why), `track_limit`, and anything else `PracticeTrack` takes |
| `ppo` | CleanRL's continuous-control defaults: 2048-step rollouts, 32 minibatches, 10 epochs, lr 3e-4 annealed, γ 0.99, λ 0.95, clip 0.2 |
| `eval` | `every_updates` (0 = off), `episodes`, `max_steps`, `record_video` |
| `logging.wandb` | `enabled: false` by default; `project`, `entity`, `mode`, `tags` |
| `logging.checkpoint_every_updates` | 0 = only the final checkpoint |

To run another condition, copy the file, change `name` and the condition keys, keep the
seeds. To spread the seeds over the cluster, give every job the same file and a different
`--seed`; the run directories do not collide.

## What a run writes

```
runs/<name>/config.yaml              the condition as run
runs/<name>/seed_0/config.yaml       the same, narrowed to this seed
runs/<name>/seed_0/episodes.csv      one row per training episode
runs/<name>/seed_0/updates.csv       one row per PPO update
runs/<name>/seed_0/policy_final.pt   the head, with its feature-normaliser statistics
runs/<name>/seed_0/eye_final.pt      the eye's weights, only when it was fine-tuned
runs/<name>/seed_0/eval/step_<n>/    the evaluation harness's episodes.csv + summary.json
```

**`episodes.csv`** -- `seed, episode, update, env_steps, steps, total_return, lap_complete,
lap_time_s, terminated, truncated`, then one column per reward term:
`reward_progress, reward_lateral, reward_lap_bonus, reward_off_track` (sorted by name).
The term columns are what `AGENTS.md` §11 asks for: they always add up to `total_return`,
so an agent farming one term without progressing shows in the columns rather than being
inferred from a curve. The terms come from the env's `info["reward_terms"]`; both the
practice track and the dummy publish the same four, and the loop refuses an env that does
not publish any, or whose terms do not sum to its reward.

**`updates.csv`** -- `seed, update, env_steps, episodes_completed, mean_episode_return,
mean_episode_steps, policy_loss, value_loss, entropy, approx_kl, clip_fraction,
explained_variance, learning_rate, steps_per_second, elapsed_s`.

Rows are flushed as they are written, so a run that dies still leaves its numbers behind.

**Periodic evaluation** is the deterministic protocol from
[`evaluation-harness.md`](evaluation-harness.md), run on a `DirectDriveAgent` made of the
eye and the current policy with `env_steps` set, so `fly_driver.analysis.learning_curve`
turns a run's `eval/` directory into a learning curve and `steps_to_threshold` into a
sample-efficiency number. Lap completion -- the later success signal -- is read here, off
the deterministic policy, not off the noisy training episodes.

## Seeds and reproducibility

Each seed seeds Python, numpy and torch once, then draws every episode's env seed from a
generator seeded with it. The periodic evaluation reseeds the global generators (that is
the harness's protocol) and the loop restores all of them afterwards, so a run's training
rows are identical with evaluation on or off. On CPU, two runs of the same config are
byte-identical in `episodes.csv` and `updates.csv`; `tests/training/test_train.py` asserts
it. CUDA kernels are not all deterministic; compare returns, not bytes, across GPUs.

## Frozen and fine-tuned

`eye.frozen: true` (default) rolls out through the eye's numpy `encode` and trains the
head on the stored features. `eye.frozen: false` stores the rollout's frames too and, in
every minibatch, re-encodes them through the eye's `encode_batch` with gradients, with the
eye's parameters in the same Adam optimiser. That needs an eye that is feed-forward and
exposes `encode_batch` (`fly_driver.training.ppo.DifferentiableEye`); the CNN control eye
will, and a test eye does.

The flyvis eye is not feed-forward: it keeps the optic lobe's state between frames, so
training through it is backpropagation through the episode. That is the separate
frozen-versus-fine-tuned experimental condition `AGENTS.md` §6 names, and this loop does
not build it; asking for `frozen: false` with `flyvis` is refused with that explanation
rather than quietly training the head alone. `pixels` has no parameters and is refused
too.

## Adding an eye, a brain, an env

Two lists have to agree, and a test checks that they do:

- `fly_driver/training/config.py`: `KNOWN_EYE_TYPES`, `KNOWN_BRAINS`, `KNOWN_ENV_TYPES`,
  so a config validates without importing torch.
- `fly_driver/training/train.py`: `EYE_BUILDERS`, `BRAIN_BUILDERS`, `ENV_BUILDERS`, the
  functions that build the thing. An eye builder receives the env and must take
  `frame_shape` (and `frame_rate_hz` if it integrates in time) from it.

## What it does so far

Measured on the Windows dev box (Ryzen 5 7600, CPU-only torch, no flyvis):

| Setting | Throughput | Result |
|---|---|---|
| dummy track, pixel eye, 64 × 64 frames | ~2 000 steps/s | full laps (return ≈ 300) from ~40k steps, both seeds tried |
| practice track, pixel eye, 96 × 96 frames | ~120 steps/s | trains and logs; 8k steps is far too few to learn to drive |

The practice-track number is the MuJoCo render plus a CPU policy; the flyvis eye adds its
own ~8 ms per frame on CPU (see `running-the-stacks.md`). One env feeds one eye, so the
levers `AGENTS.md` §12 lists -- a GPU for the eye, batched envs -- are the next step once
the default condition is running on the Sparks.

## Tests

`tests/training/test_config.py` and `test_loggers.py` run everywhere. `test_ppo.py`
(the advantage recursion is pinned to a hand-computed case), `test_train.py` (writes,
seeds, evaluation, frozen versus fine-tuned, the env contract, and a `slow` test that the
return rises on the dummy track) and `tests/policies/test_mlp_policy.py` skip without
torch, as CI's base install does.
