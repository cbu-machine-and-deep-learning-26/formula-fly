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
python scripts/train.py --config fly_driver/configs/conditions/cnn.yaml  # an RQ1 eye condition
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
| `eye.type` | `flyvis` (default), `cnn`, `random_projection`, `shuffled`, or `pixels` (the GH-17 smoke eye, not an RQ1 control) |
| `eye.frozen` | `true` trains only the head. `false` puts the eye's parameters in the optimiser; see below |
| `eye.params` | Constructor keyword arguments (`readouts`, `checkpoint`, `downsample`...). `frame_shape` and `frame_rate_hz` come from the env and may not be set here |
| `brain` | `none` (the policy reads the eye directly), or `type: central_complex` with `params` (`neuron_count`, `output_dim`, `input_in_degree`, `recurrent_in_degree`, `seed` for the synthetic wiring). See [the brain condition](#the-brain-condition) |
| `policy` | `hidden_sizes`, `activation`, `normalize_features`, `init_log_std`, `init_action_mean` (default zeros) |
| `env.type` | `practice_track` or `dummy` |
| `env.max_steps` | Episode truncation in frames. 15 000 = 5 minutes at 50 Hz, room for a slow clean lap |
| `env.params` | For the practice track: `racing_line` (turn it **off** for the RQ1 comparisons; the env's docstring says why), `track_limit`, `random_start` (training env only, see below), and anything else `PracticeTrack` takes |
| `ppo` | CleanRL's continuous-control defaults: 2048-step rollouts, 32 minibatches, 10 epochs, lr 3e-4 annealed, γ 0.99, λ 0.95, clip 0.2 |
| `eval` | `every_updates` (0 = off), `episodes`, `max_steps`, `record_video` |
| `logging.wandb` | `enabled: false` by default; `project`, `entity`, `mode`, `tags` |
| `logging.checkpoint_every_updates` | 0 = only the final checkpoint |

**Random starts.** `env.params.random_start: true` starts every training episode at a
uniformly random point round the lap, on the centreline, facing along the track, at rest,
drawn from the episode's reset seed. The v1 RQ1 runs all started on the grid and plateaued
at about 555 m: each policy only ever practised the opening straight and ran off the first
bend. Lap timing needs no special case: the first crossing of the line ends the out lap, a
lap counts only line to line, and progress is a per-step delta that is safe across the
line. `build_env(..., training=True)` is the only path that passes it on. Periodic
evaluation, `scripts/evaluate.py` and robustness build the env without it and start on the
grid, so their lap times compare with every other run. `episodes.csv` records each
episode's `start_m`. The four `conditions/*.yaml` and `train_flyvis_cx.yaml` turn it on and
are renamed `rq1v2_<eye>` and `flyvis_cx_v2` so they do not overwrite the v1 runs.

To run another condition, copy the file, change `name` and the condition keys, keep the
seeds. To spread the seeds over the cluster, give every job the same file and a different
`--seed`; the run directories do not collide.

## What a run writes

```
runs/<name>/config.yaml              the condition as run
runs/<name>/seed_0/config.yaml       the same, narrowed to this seed
runs/<name>/seed_0/episodes.csv      one row per training episode
runs/<name>/seed_0/updates.csv       one row per PPO update
runs/<name>/seed_0/parameters.json   trainable parameter counts per stage (RQ2)
runs/<name>/seed_0/policy_final.pt   the head, with its feature-normaliser statistics
runs/<name>/seed_0/brain_final.pt    the brain's weights and mask, only with a brain
runs/<name>/seed_0/eye_final.pt      the eye's weights, only when it was fine-tuned
runs/<name>/seed_0/eval/step_<n>/    the evaluation harness's episodes.csv + summary.json
```

**`episodes.csv`** -- `seed, episode, update, env_steps, steps, total_return, lap_complete,
lap_time_s, terminated, truncated, start_m, distance_m` (the arclength the episode started
at, and metres driven along the centreline), then
one column per reward term:
`reward_progress, reward_lateral, reward_lap_bonus, reward_off_track` (sorted by name).
The term columns are what `AGENTS.md` §11 asks for: they always add up to `total_return`,
so an agent farming one term without progressing shows in the columns rather than being
inferred from a curve. The terms come from the env's `info["reward_terms"]`; both the
practice track and the dummy publish the same four under the default reward (the
[pace reward](#the-pace-reward-v4) publishes its own four), and the loop refuses an
env that does not publish any, or whose terms do not sum to its reward.

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
not build it; asking for `frozen: false` with `flyvis` or `shuffled` is refused with that
explanation rather than quietly training the head alone. `pixels` and `random_projection`
have no parameters and are refused too. `cnn` exposes `encode_batch`, so `frozen: false`
fine-tunes the convolutions and the linear map.

The three GH-15 controls emit the same 5768 features as frozen flyvis T4/T5. `cnn` is a
small conv net. `random_projection` is a frozen Gaussian matrix from grey pixels (`seed`
fixes it; default 0). `shuffled` is the pretrained flyvis eye with a degree-preserving
rewiring of its edges (`seed` and `swap_attempts` in `eye.params`). None of them is
`pixels`.

## The brain condition

`fly_driver/configs/train_flyvis_cx.yaml` is RQ2's condition: the default frozen flyvis eye,
then `CentralComplexBrain` (`fly_driver/brains/central_complex.py`), then the policy head,
with the racing line off. The brain's synapse strengths, biases and time constants train
with the head in the same Adam optimiser. Its mask does not train. The eye stays frozen.

```bash
python scripts/train.py --config fly_driver/configs/train_flyvis_cx.yaml --seed 0
python scripts/train.py --env dummy --eye pixels --brain central_complex --name cx_smoke \
    --total-env-steps 20000                # no flyvis needed; default width 256
python scripts/evaluate.py --checkpoint runs/flyvis_cx_v2/seed_0/policy_final.pt
```

`--brain central_complex` on another config uses the constructor defaults. Put
`neuron_count` and the rest under `brain.params` in YAML to change them. The config uses
4096 neurons and a 32-wide readout. `scripts/brain_frame_time.py` at the flyvis input width
puts that step at about 1.4 ms, which fits in what the 20 ms frame has left after the
render, the eye and the policy. 16384 neurons, at about 4.1 ms, does not.

`parameters.json` holds `eye_trainable`, `brain_trainable`, `head_trainable`, and
`blank_trainable`: the count for a dense `BlankWidthNet` of the same width and input. The
`none` condition writes the same file with zero brain parameters, so RQ2 reads both from
logs. The counts also go into the W&B run config.

**Single-step gradient.** The brain is recurrent: its membrane carries over between frames
and resets at each episode start. The rollout buffer stores each step's eye features and
the membrane before that step. The update replays that one step through
`CentralComplexBrain.transition` with gradients. There is no backpropagation through time,
so no credit reaches what earlier frames left in the membrane. The membrane still carries
information forward during rollouts. The brain cannot be trained to keep information it
needs later. Truncated BPTT over stored sequences is the upgrade if this turns out to
matter. `scripts/evaluate.py --checkpoint` reloads the brain from `brain_final.pt` (or
`brain_update_<n>.pt` beside a periodic checkpoint) and runs the deterministic protocol on
the run's env.

## The pace reward (v4)

Round 2 (`flyvis_cx_v2`) trained on `ProgressReward`, which pays metres and nothing else.
With a 300 s episode, slow driving earned more total distance than fast driving that
crashed. One seed learned to hug the kerb at 28 km/h, some checkpoints parked at full
lock, and no seed finished a lap.

Round 3 (`flyvis_cx_v3`) charged a time cost of a quarter of target pace per second, about
50 km/h at a 105 s target, plus a constant crash penalty (about -47 after scaling). Every
second below 50 km/h lost reward, and an untrained car cannot hold 50 km/h without
crashing, so ending the episode early was the cheapest policy. After about 200k steps all
10 seeds averaged 4-19 km/h, every episode ended in a crash or a stall, and training and
eval distances were shrinking (seed 7: 196 m to 65 m, seed 6's eval: 371 m to 0 m).

`fly_driver/configs/train_flyvis_cx_v4.yaml` keeps v3's stall termination, random starts,
discount and initial action, and pays pace-weighted progress (`LapTimeReward`,
`fly_driver/envs/lap_time_reward.py`):

```yaml
env:
  params:
    stall_speed_mps: 2.0      # slower than this for more than stall_seconds ends the episode
    stall_seconds: 2.5
    reward: {type: lap_time, target_lap_time_s: 105.0, pace_cap: 2.0, pace_floor: 0.25, crash_penalty_m: 0.0, scale: 0.01}
policy:
  init_action_mean: [0.0, 0.4, -0.6]   # steer, throttle, brake
```

The stall clock starts only once the car has first gone faster than `stall_speed_mps`.
Stopping after that is a crash. A car that never pulls away is not terminated: it earns
nothing until truncation. `policy.init_action_mean` sets the actor's initial output bias.
The default of zeros leaves throttle and brake clipped to 0 half the time, and in the first
20k-step v3 run every episode stalled at 2.5 s with the same return (explained variance 0).
Starting throttle at +0.4 and brake at -0.6 makes the untrained car pull away.

`reward` takes `type: progress` (the default, unchanged) or `type: lap_time`, plus that
class's keyword arguments. All terms are in metres, then multiplied by `scale`. At 0.01 a
lap at target pace returns about +89: PPO here does not normalise returns or
values, and unscaled returns gave the critic nothing it could fit. `v*` is lap length over
target lap time (202 km/h at 105 s), `v` is `progress_m / dt`, `dt` is 0.02 s:

| Term | Per step | Purpose |
|---|---|---|
| `pace` | `progress_m * max(clip(v / v*, 0, 2), 0.25)` forward, `progress_m` backwards | pays distance weighted by speed |
| `track_limits` | `-v* * 1.7 * depth * dt`, capped at the step's pace since v5 | charges running past the kerb, scaled by depth |
| `crash` | `-crash_penalty_m`, once; 0 in v4 | charged on a track-limit termination or a stall |
| `lap_bonus` | `2000 * target / lap_time`, plus 1000 if `lap_time <= target`, once | pays for a completed lap, more for a faster one |

- **Pace beats distance.** A metre at target pace pays twice a metre at half of it.
  Crawling at 28 km/h for the whole 300 s earns 583 m. Target pace for 20 s and then a
  crash earns 1,121 m. The 0.25 floor keeps a gradient toward faster at crawling speeds.
- **A crash costs only the reward it forfeits.** A first v4 smoke run with a 500 m crash
  penalty slid from 17 to 4 km/h: discounted, a constant penalty is cheaper the later it
  lands, so slowing down to delay a crash paid. Any penalty above 0 makes slow beat fast
  for some crash distance. With none, every forward step pays at least 0, so ending the
  episode early never pays and driving faster to a fixed crash always does.
- **The kerb is track.** `depth` is `off_track_fraction`: the fraction of the car's width
  past the kerb's *outer* edge. Two wheels on the kerb read 0 and cost nothing. Past the
  kerb, each second costs the seconds the env's lap-time penalty adds
  (`PenaltyWeights.time_weight` = 1.7 per unit depth), converted to metres at target pace.
  The existing termination at `track_limit` (0.15) is unchanged.
- **Laps can't be farmed.** `info["lap_time"]` already includes off-track penalty seconds,
  and `LapTimer` counts a lap only after a full lap of forward progress from a crossing of
  the line, so rocking over the line or starting just behind it earns nothing.

Every term is logged per episode as `reward_<term>` in `episodes.csv`, and `distance_m`
keeps the raw metres whatever the reward pays. Each step's `info` also carries `crashed`
and `stalled`.

### What the car can actually do

`target_lap_time_s` defaults to 90 s, which is about F1 race pace (the record in issue #32
is 1:27.097). The simulated car cannot get there. Measured on the practice track:

- **Scripted lap: 121.1 s** (122.7 s with penalties). The scripted driver uses pure
  pursuit on `racing_line()` with a curvature speed profile at 70% of the car's grip. At
  80% or more it leaves the track in the first sector.
- **Ideal profile: ~100 s.** A point-mass speed profile on the same line at the car's full
  measured envelope: 4.4 g lateral at 250 km/h, 3.7 g average braking from 300 to
  100 km/h, and the full-throttle acceleration curve recorded on the `PerformanceBed`.
  This is a bound, not a lap anyone drove.

v3 and v4 therefore set `target_lap_time_s: 105.0`. That is 5% off the bound and 13% faster than
the scripted lap, so the target bonus can be earned. A 90 s target would make it
unreachable.

### Discount

`ppo.gamma: 0.997`. 0.99 is a horizon of `1 / (1 - gamma)` = 100 steps, 2 s at 50 Hz.
Braking from 300 to 100 km/h takes about 1.3 s and 78 m, and the crash it prevents comes
after turn-in, 2–3 s after the brake point. Under 0.99 a crash 150 steps ahead is weighted
0.22. Under 0.997 it is weighted 0.64. 0.997 is a 333-step (6.7 s, about 400 m at target
pace) horizon, enough to see a braking zone and the corner after it without stretching
credit assignment over a whole lap.

### v4 runs

An A/B on `pace_floor`, paired on seeds 0-4, 5M steps each. `train_flyvis_cx_v4b.yaml` is
v4 with `pace_floor: 0.0`. Below the floor's speed (50 km/h at 0.25) a metre pays the same
however slow it is driven, while slower driving survives further. Without it, a slow metre
pays proportionally less. Whether the floor's early gradient or that incentive wins only
shows once the policy can steer:

```bash
# csds-pc-2
python scripts/train.py --config fly_driver/configs/train_flyvis_cx_v4.yaml --seed 0   # ... --seed 4
# csds-pc-1
python scripts/train.py --config fly_driver/configs/train_flyvis_cx_v4b.yaml --seed 0  # ... --seed 4
```

## Warm starts

Each round starts from the previous round's best checkpoint, not from scratch. `init_from:`
in a run config names a `policy_*.pt`; before PPO starts, the policy (with its
feature-normaliser statistics) loads from it and the brain from the `brain_*.pt` beside it,
as `load_driver` pairs them. The eye stays frozen and pretrained, the optimiser starts
fresh, and a policy or brain of another shape is an error. It works for every env type.

Checkpoints are not in git. Copy the chosen pair into the training checkout's `weights/`
(git-ignored) and point `init_from:` at it, relative to the checkout. Round 5
(`train_flyvis_cx_v5.yaml`, `train_flyvis_cx_ac_v5.yaml`) starts from v4 seed 0's
`policy_update_00250.pt`, before that seed collapsed, at its best 50-drive window (623 m
average), in `weights/round4_best/`.

## The v5 reward

Round 4's best seeds drove 2.8 km through six corners at up to 140 km/h, but seeds 0, 3 and 5
collapsed into two leaks. Kerb clipping cost about 14 m/s at the termination depth, so a
policy that kept clipping expected less from continuing than from creeping off the track and
ending the run. Seed 0 settled at 4 s, 7 m, 6 km/h episodes. And the stall rule only armed
after the car first passed 2 m/s, so seed 3 never did and sat out every 300 s episode.

v5 (`train_flyvis_cx_v5.yaml`) closes both:

- **Every early end costs 18 m:** a crash, an off-track termination or a stall. 18 is the
  largest whole metre for which, with a crash a fixed D ≥ 50 m ahead, the γ = 0.997
  discounted return still rises with speed from 30 km/h to twice target pace. The bound is
  18.44, tightest at 30 km/h and D = 50 m. The derivation is in
  `fly_driver/envs/lap_time_reward.py`.
- **Too slow ends the run.** Under 30 km/h (`stall_speed_mps: 8.3`) for 3 s ends the
  episode, counted from the first step. Flat out from rest passes 30 km/h in 0.7 s.
- **An exit never pays.** The cost past the kerb is capped at the step's pace, so a step that
  moves forward is worth at least 0, on or off the track. The kerb itself stays free.
- **Pace without the floor:** `pace_floor: 0`. The slow rule now forces the car to move.

## Adding an eye, a brain, an env

Two lists have to agree, and a test checks that they do:

- `fly_driver/training/config.py`: `KNOWN_EYE_TYPES`, `KNOWN_BRAINS`, `KNOWN_ENV_TYPES`,
  so a config validates without importing torch.
- `fly_driver/training/train.py`: `EYE_BUILDERS`, `BRAIN_BUILDERS`, `ENV_BUILDERS`, the
  functions that build the thing. An eye builder receives the env and must take
  `frame_shape` (and `frame_rate_hz` if it integrates in time) from it. A brain builder
  receives the eye's feature count and the env's frame rate.

## What it does so far

Measured on a Windows box: Ryzen 5 7600, AMD RX 7800 XT through AMD's ROCm build of PyTorch
(setup in `running-the-stacks.md`).

| Setting | Throughput | Result |
|---|---|---|
| dummy track, pixel eye, CPU | ~2 000 steps/s | full laps (return ≈ 300) from ~40k steps, both seeds tried |
| practice track, pixel eye, CPU | ~120 steps/s | trains and logs; 8k steps is far too few to learn to drive |
| practice track, **flyvis eye on the GPU** (the default condition) | ~55 steps/s | seeds 0, 1 and 2 train, log and evaluate end to end |

With the flyvis eye, one step costs about 7.5 ms of track (render plus physics), 4.9 ms of
eye and 1.1 ms (CPU) to 3.8 ms (GPU) of policy. The PPO update over 2 048 steps takes 3.3 s
on the GPU against 10.2 s on the CPU, so the two end up level. 1M steps is about five hours
per seed here. The track step is the largest share, which is what #63 is about.

On Windows, flyvis's connectome cache crashed on first build (datamate deletes an `.h5` file
it still holds open). `fly_driver/eyes/flyvis_eye.py` patches datamate's writer until
upstream fixes it. The eye also runs on a GPU host now, whatever device `import flyvis` makes
torch's default (GH-27).

## Tests

`tests/training/test_config.py` and `test_loggers.py` run everywhere. `test_ppo.py`
(the advantage recursion is pinned to a hand-computed case), `test_train.py` (writes,
seeds, evaluation, frozen versus fine-tuned, the env contract, and a `slow` test that the
return rises on the dummy track), `test_train_brain.py` (membrane resets, replay matches the
rollout, the mask stays fixed, a 15 000-step rollout stays bounded, checkpoints reload) and `tests/policies/test_mlp_policy.py` skip without
torch, as CI's base install does.
