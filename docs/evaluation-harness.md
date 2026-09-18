# Evaluation harness

One eval path for every track: same seed → same actions, with lap time,
sample efficiency, robustness, and optional episode video. Lives in
`fly_driver.training` (harness), `fly_driver.analysis` (curves), and
`fly_driver.envs.DummyTrackEnv` (numpy-only env for tests and CI).

## Run it

```bash
python scripts/evaluate.py --agent random --output runs/eval_random
```

```bash
python scripts/evaluate.py --config fly_driver/configs/eval_default.yaml --record-video --output runs/demo
```

Or from code:

```python
from fly_driver.envs import DummyTrackEnv
from fly_driver.training import EvalConfig, evaluate

config = EvalConfig.from_yaml("fly_driver/configs/eval_default.yaml")
report = evaluate(DummyTrackEnv, agent, config)
report.summary("clean").mean_lap_time_s
```

`evaluate(make_env, agent, config)` takes an env *factory* (one fresh env per
condition), an agent with `act(frame) -> (steer, throttle, brake)` (array-like
or an object with `to_array()`) and an optional `reset(seed=...)`, and an
`EvalConfig`.

## Protocol

- Episode `i` uses seed `config.seed + i` for the env, the agent, and every
  perturbation. `random`, `numpy.random`, and torch (only if already imported)
  are seeded once from `config.seed`.
- Conditions: `clean` first, then each configured perturbation, all with the
  same seeds. `EvalReport.actions_digest` is a SHA-256 over every action
  sequence; two runs with the same config must match it (tested in CI on
  `DummyTrackEnv`).
- **Lap time**: taken from `info["lap_time"]` when the env reports it on
  `info["lap_complete"]`, otherwise `steps / frame_rate_hz`. Every real env
  wrapper should set both keys.
- **Robustness**: per perturbation, `return_ratio_to_clean` (mean return over
  the clean mean; `None` if the clean mean is zero) plus lap completion rate.
  Built-in perturbations: `observation_noise(std)`, `brightness(scale)`,
  `action_delay(steps)`; add more in `fly_driver/training/perturbations.py`.
- **Sample efficiency**: set `env_steps` on each periodic evaluation during
  training, then `fly_driver.analysis.learning_curve(reports)` and
  `steps_to_threshold(env_steps, values, threshold)` give steps-to-threshold.
- Actions are validated (shape `(3,)`, finite) and never clipped by the
  harness; range checks belong to the env (`DummyTrackEnv` raises).

## Config (`fly_driver/configs/eval_default.yaml`)

| Key | Meaning |
|---|---|
| `episodes` | Episodes per condition |
| `seed` | Base seed |
| `max_steps` | Harness-side per-episode cap (`null` = env decides) |
| `frame_rate_hz` | Env rate; lap-time fallback and video fps |
| `record_video`, `video_episodes`, `video_backend` | Record the first N episodes per condition (`auto`, `imageio`, `cv2`) |
| `output_dir` | Where `episodes.csv`, `summary.json`, `videos/` go |
| `perturbations` | List of `{name, ...params}` |
| `env_steps` | Training steps seen by the evaluated policy |

Unknown keys are rejected.

## Outputs

`episodes.csv` has one row per episode per condition (`condition, episode,
seed, steps, total_return, lap_complete, lap_time_s, terminated, truncated,
video_path`). `summary.json` holds the config, per-condition aggregates, the
actions digest, and which video backend (if any) was used.

## Video

Videos need `imageio` + `imageio-ffmpeg` (or `opencv-python`); neither is a
base dependency. Without one, the harness warns once, records nothing, and
reports `video_backend: null`. With one, frames come from `env.render()` (or the
observation when `render()` returns `None`) and are written as mp4 at
`frame_rate_hz`.

```bash
python -m pip install imageio imageio-ffmpeg
```
