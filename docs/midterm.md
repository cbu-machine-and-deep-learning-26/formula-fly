# Midterm: does the connectome eye help? (GH-27, RQ1)

> **Status: waiting for the GPU runs.** The setup, launch plan and regeneration command below
> are final. The results and conclusion sections are placeholders until the 12 runs are
> copied back and `scripts/midterm_results.py` has been run on them.

## Setup

Four eyes, each frozen, each with a PPO-trained policy head, on the MuJoCo practice track
(Silverstone centerline). Seeds 0, 1 and 2 for each eye, so 12 runs.

| Eye | Config | What it is |
|---|---|---|
| `flyvis` | `fly_driver/configs/conditions/flyvis.yaml` | Pretrained flyvis optic lobe (`flow/0000/000`), T4a–d/T5a–d readout |
| `shuffled` | `.../conditions/shuffled.yaml` | The same network with degree-matched shuffled wiring. Each neuron keeps its in- and out-degree. The 12,380 self-loops are held fixed. |
| `cnn` | `.../conditions/cnn.yaml` | Small CNN, frozen at its seeded init |
| `random_projection` | `.../conditions/random_projection.yaml` | Fixed Gaussian projection of grey pixels |
| `flyvis+central_complex` | `fly_driver/configs/train_flyvis_cx.yaml` | **Fifth condition, not part of RQ1.** The flyvis eye feeding the central-complex brain. Its config is owned by the brain work. It is listed here so it can run alongside the others, and it is reported only if its runs exist. |

The four files differ only in `name` and `eye.type`. Every eye emits 5,768 features and is
frozen, so every condition trains the same `[64, 64]` tanh policy head with the same
parameter count (the matching set up in #73). The racing line is off (`racing_line: false`),
the track limit is 0.15, and episodes are truncated at 15,000 steps (5 simulated minutes).

**Budget:** `total_env_steps: 1000000` per seed for every eye. This rounds down to 488 PPO
updates of 2,048 steps, which is 999,424 steps. All four eyes get the same budget.

**Periodic evaluation:** every 25 updates (about 51k steps), one deterministic episode with
no step cap beyond the env's own. The default is every 10 updates. That was changed because an
early policy that barely moves runs the full 15,000-step cap: in the pipeline check, the CNN's
first deterministic episode covered 2.7 m in 1,500 steps. At every 10 updates, those evals
could add up to about 70% to a run's wall time. At every 25 updates the ceiling is about 30%,
and the curve still gets 19 points.

**Robustness:** after training, each seed's `policy_final.pt` is run through the evaluation
harness protocol (`docs/evaluation-harness.md`). That is clean driving plus the three
perturbations in `fly_driver/configs/eval_default.yaml`: observation noise (std 0.1),
brightness ×0.5 and a 2-step action delay. Each condition gets 2 episodes (seeds 0 and 1)
with the env's 15,000-step cap. The env's reset is deterministic, so the two clean episodes
are identical. The second episode only adds a sample under observation noise. The robustness
metric is return under the perturbation divided by clean return, plus lap completion and
progress. The harness does not report progress, so `fly_driver/analysis/robustness.py` adds
up each episode's `info["progress_m"]` and writes it to `robustness/progress.csv`.

**Metrics in the table:** each seed's mean over its evaluation episodes, then the mean ± sample
std over the 3 seeds. Lap time is averaged only over seeds that finished a lap, and the table
shows `n=` when that is fewer than 3. Progress is metres along the centerline, net of
reversing. A Silverstone lap is about 5.9 km.

## Pilot: go/no-go before the full runs

Before spending GPU-days, run one CNN seed for 200k steps on a single GPU. The CNN is the
fastest eye. The pilot answers one question: does the policy learn to commit throttle?

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/train.py --config fly_driver/configs/pilot_cnn.yaml
```

`fly_driver/configs/pilot_cnn.yaml` is `conditions/cnn.yaml` with seed 0 only and 200k
steps (97 updates). It evaluates every 5 updates, about every 10k steps, and caps each eval
at 3,000 steps (60 s). Read the clean-eval progress and lap completion at each eval:

```bash
python -c "
from pathlib import Path
from fly_driver.analysis.midterm import load_report
for path in sorted(Path('runs/pilot_cnn/seed_0/eval').glob('step_*/summary.json')):
    report = load_report(path)
    clean = report.summary()
    print(f'{report.config.env_steps:>7}  progress_m {clean.mean_progress_m:8.1f}  laps {clean.lap_completion_rate:.2f}')
"
```

The same numbers are in each `eval/step_*/episodes.csv` (`progress_m`, `lap_complete`).
Per training episode, `episodes.csv` has `reward_progress`, which is the same metres from the
stochastic policy.

**Go/no-go:** go if mean eval progress grows clearly beyond the untrained ~40 m by 200k
steps, and keeps growing rather than spiking once. If it is still around 40 m, the policy
has not learned to drive forward. Then 1M steps per seed is unlikely to produce a
comparison, and the reward or the budget should be revisited before the 12 runs.

## How to run

### 1. Each GPU machine, once

Linux:

```bash
git clone <repo> formula-fly && cd formula-fly && git checkout develop
python3.12 -m venv .venv-flyvis && source .venv-flyvis/bin/activate
python -m pip install torch --index-url https://download.pytorch.org/whl/cu128   # 5090s need CUDA >= 12.8
python -m pip install -r requirements-flyvis.txt -e ".[dev]"
export FLYVIS_ROOT_DIR="$HOME/.cache/flyvis" && flyvis download-pretrained
export MUJOCO_GL=egl                       # headless rendering; omit with a desktop session
python -c "import torch; print(torch.cuda.device_count(), torch.cuda.get_device_name(0))"
```

Windows (PowerShell):

```powershell
git clone <repo> formula-fly; cd formula-fly; git checkout develop
py -3.12 -m venv .venv-flyvis; .venv-flyvis\Scripts\Activate.ps1
python -m pip install torch --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements-flyvis.txt -e ".[dev]"
$env:FLYVIS_ROOT_DIR = "$HOME\.cache\flyvis"; flyvis download-pretrained
python -c "import torch; print(torch.cuda.device_count(), torch.cuda.get_device_name(0))"
```

Do **not** set `CUDA_VISIBLE_DEVICES=-1`. The flyvis eye now runs on the GPU even though
`import flyvis` makes CUDA torch's default device. `CUDA_VISIBLE_DEVICES=<index>` below only
picks which GPU a process sees.

### 2. Train: one GPU per eye, three seeds as concurrent processes

`<i>` is the GPU's index on its machine (from `nvidia-smi`). Linux:

```bash
EYE=flyvis   # 5090 #1: flyvis | 5090 #2: shuffled | 4090 #1: cnn | 4090 #2: random_projection
GPU=<i>
mkdir -p logs
for SEED in 0 1 2; do
  CUDA_VISIBLE_DEVICES=$GPU nohup python scripts/train.py \
    --config fly_driver/configs/conditions/$EYE.yaml --seed $SEED > logs/${EYE}_seed$SEED.log 2>&1 &
done
```

Windows:

```powershell
$EYE = "cnn"; $env:CUDA_VISIBLE_DEVICES = "<i>"; mkdir -Force logs
foreach ($SEED in 0, 1, 2) {
  Start-Process python -NoNewWindow -ArgumentList "scripts/train.py", "--config", `
    "fly_driver/configs/conditions/$EYE.yaml", "--seed", "$SEED" `
    -RedirectStandardOutput "logs/${EYE}_seed$SEED.log" -RedirectStandardError "logs/${EYE}_seed$SEED.err"
}
```

| GPU | Eye | Why |
|---|---|---|
| 5090 #1 | `flyvis` | Slowest eyes on the fastest cards. The optic lobe is simulated every frame. |
| 5090 #2 | `shuffled` | Same cost as flyvis, plus about a minute at startup for the shuffle. |
| 4090 #1 | `cnn` | The frozen CNN encodes on the CPU, so the GPU only runs the head. |
| 4090 #2 | `random_projection` | Same |
| 5090 #1, shared with flyvis | `train_flyvis_cx.yaml` | Optional. The same three-seed loop with `--config fly_driver/configs/train_flyvis_cx.yaml`. It shares the flyvis GPU, so both conditions run slower. Its run name is `flyvis_cx`, so step 3 uses `--condition flyvis_cx`. |

**Two GPUs only:** run six processes per GPU, the same loop twice. Put flyvis + cnn on one
GPU and shuffled + random_projection on the other, so that each GPU gets one slow eye. If the
fifth condition runs too, it goes on the flyvis GPU, which then has nine processes.

Each progress line prints `steps/s`. After the first few updates, a seed needs about
`1e6 / steps_per_s / 3600` hours plus up to 30% for evaluation. If that misses the deadline,
restart **all four eyes** with the same smaller `--total-env-steps` and record it here. The
budget must stay equal across eyes.

### 3. Robustness, on the same machine after training

```bash
for SEED in 0 1 2; do
  CUDA_VISIBLE_DEVICES=$GPU nohup python scripts/midterm_results.py runs --robustness-only \
    --condition rq1_$EYE --seed $SEED --episodes 2 > logs/${EYE}_robust$SEED.log 2>&1 &
done
```

### 4. Copy back

Each run writes to `runs/rq1_<eye>/seed_<k>/`: `config.yaml`, `episodes.csv`, `updates.csv`,
`eval/step_*/`, `policy_final.pt` and, after step 3, `robustness/`. Nothing in it is a video,
and the policy checkpoint is about 3 MB, so copy the whole `runs/rq1_*` directories (and the fifth condition's, if it ran) into one
`runs/` on the analysis machine. If step 3 was skipped somewhere, `policy_final.pt` lets the
next command run it locally, but slowly.

### 5. Regenerate the table, figure and robustness results

```bash
python scripts/midterm_results.py runs --episodes 2
```

This runs robustness for any seed that lacks it, then writes:

- `docs/results/midterm_results.csv`: per model × evaluation condition, mean/std/n over seeds
- `docs/results/midterm_per_seed.csv`: the per-seed numbers those come from
- `docs/results/midterm_table.md`: the two tables below
- `docs/figures/midterm_learning_curves.png`: clean evaluation return and lap completion against
  env steps, mean ± std band over seeds

Paste `midterm_table.md` into the results section and commit the CSVs and the PNG. Do not
commit `runs/`.

## Results

> **Placeholder: fill in from the real runs.** Paste `docs/results/midterm_table.md` here and
> record each machine's steps/s, wall time per seed, and any seed that crashed or did not
> learn.

![Learning curves](figures/midterm_learning_curves.png)

## Conclusion so far

> **Placeholder.** State whether the connectome constraint helped, hurt or made no difference
> so far: flyvis against shuffled first (the wiring pattern), then against cnn and
> random_projection. Judge it on lap completion and progress, with the seed spread from the
> table, and with these caveats:
> - 1M steps per seed is a midterm budget, not convergence.
> - With 3 seeds, a difference smaller than the std is not a difference.
> - There is one track and one shuffle (seed 0) of the connectome.
> - Return / clean is meaningless when the clean return is near zero, as it is for a policy
>   that barely moves. In that case, read lap completion and progress instead.
