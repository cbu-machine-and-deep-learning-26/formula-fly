# The reward coach: Claude reviews the reward (GH-27)

The fly's policy trains for millions of steps on a reward we wrote by hand. Rounds 2 to 7 each
found a new way that reward taught the wrong thing (parking, stalling, creeping off track,
crashing at braking zones). The coach has Claude Opus 5.5 read what a run actually did and
propose the next recipe, and checks every proposal before it can become a config.

```
runs/<run>/seed_*/episodes.csv, updates.csv, config.yaml
        │  fly_driver/coach/digest.py      corners, where drives end, reward terms, PPO health
        ▼
   digest.md ──► Claude Opus 5.5 ──► JSON review (diagnosis, hacking check, ≤4 proposals)
                                         │  fly_driver/coach/guardrails.py
                                         ▼
                     proposal_<k>_*.yaml (accepted)   review.md (accepted + rejected, and why)
```

Claude never scores individual steps or drives. A trainer makes over 1,000 steps a second, so a
per-step judge would be the bottleneck and the bill. The reward stays a cheap function the env
computes, and Claude improves that function between runs from evidence. This is the Eureka
pattern (Ma et al., ICLR 2024), with a person choosing what runs.

## Set up the key

Install the extra into the training venv, then put the key in `.env` at the checkout root.
`.env` is git-ignored:

```bash
.venv-flyvis/bin/python -m pip install -e ".[coach]"
echo 'ANTHROPIC_API_KEY=sk-ant-...' > .env
```

An `ANTHROPIC_API_KEY` already in the environment wins over `.env`. On a lab PC, the key sits in
that PC's folder. Delete `.env` when the key expires.

## Run it

```bash
python scripts/reward_coach.py runs/lab_20261104_g3                       # one review
python scripts/reward_coach.py runs/lab_20261104_g3 --dry-run              # digest + prompts, no API call
python scripts/reward_coach.py runs/lab_20261104_g3 runs/lab_20261104_g4   # pool several runs
python scripts/reward_coach.py runs/lab_20261104_g3 --note "Vale focus test, 50% starts at 4-5.2 km"
python scripts/reward_coach.py runs/lab_20261104_g3 --every-minutes 60     # while it trains
```

It only reads runs, so it is safe while a trainer writes them. Each review lands in
`<run>/coach/<timestamp>/`:

| File | What it is |
|---|---|
| `digest.md` | Exactly what Claude was shown |
| `review.md` | The diagnosis, bottlenecks, each proposal with its hypothesis and success metric, and which were accepted or rejected and why |
| `response.json` | The raw structured answer and token usage |
| `proposal_<k>_<title>.yaml` | A full run config per accepted proposal, renamed `<run>_coach<stamp>_<k>` |

`<run>/coach/history.jsonl` keeps one line per review. The next review sees the last five, so
Claude doesn't propose the same thing twice. `--every-minutes` skips a review until 200 new
drives have finished (`--min-new-drives`).

Claude Opus 5.5 reviews at `max` effort with adaptive thinking (`REVIEW_EFFORT` in
`fly_driver/coach/review.py`); at `high` the Nov 5 reviews thought for 4.6-7.8k tokens each.

A review costs roughly 10-20k input tokens and a few thousand output tokens at Opus 5.5 prices
($4 / $20 per million): cents per review. The request also turns on server-side refusal
fallbacks (`fallbacks: "default"`), so a safety-classifier false positive on a racing-telemetry
prompt still returns a review.

## What Claude sees

The digest (`fly_driver/coach/digest.py`) is built from the files every run already writes, so
it works on old runs and on runs copied off a lab PC:

1. **Recipe**: reward, slow rule, starts, PPO settings, brain, eye.
2. **Progress per seed**: drives, laps, the supervisor's last-50 uniform score, best window,
   longest drive, and mean distance over time in eight buckets.
3. **Corners**: arrivals and passes for T1-T10, overall and first half against second half.
4. **Where drives end**: per segment, with the reason (crash, stall, step cap) and median top
   speed. Runs from this change on log `end_reason` and `max_speed_kmh` per drive. Older runs
   show `ended`.
5. **Reward terms**: mean per drive and share of the total, plus return against distance (Pearson
   r and return per metre). That is the farming check `AGENTS.md` §11 asks for.
6. **PPO health**: entropy, KL, clip fraction, explained variance, controls and steps/s, for the
   first 10%, middle and last 10% of updates.

The system prompt holds the project context, `lap_time_reward.py`'s own derivation, what rounds
2-7 learned, and the knob table. It is identical for every review, so it is cached.

## What it may change: the guardrails

`fly_driver/coach/guardrails.py` decides, not Claude. A proposal becomes a config only if:

- every key is in `KNOBS`: reward weights (target lap time, pace cap/floor, crash, corner,
  lap and target bonuses), the slow rule, the track limit, the start-focus zone, and PPO's
  entropy, learning rate, target KL, gamma and λ, each within bounds. `reward.scale` isn't
  a knob, because PPO here doesn't normalise returns;
- it changes **at most two** knobs, so a result can be attributed (the Vale test changed two
  and couldn't say which mattered);
- `crash_penalty_m` stays under the **slow-down-never-pays bound** for the proposal's own
  gamma, target lap time, pace cap and slow-rule speed. `max_crash_penalty_m` computes it
  numerically from the derivation in `lap_time_reward.py`. At v7's settings it is about 18.4 m,
  as that file says. The bound depends on the other knobs: drop the slow rule to 4 m/s and it
  falls to about 9.3 m, so v7's 18 m penalty would quietly reopen round 3's stalling exploit.
  The guardrail rejects that;
- the result loads as a `TrainConfig`, which refuses unknown keys.

Anything Claude thinks needs more than the knobs, such as a new reward term, a curriculum, a
teacher policy or an observation change, goes under "Beyond the knobs" in `review.md` for a
person to judge.

## Running a proposal

The coach never starts training. A proposal is a recipe scored differently from `weights/best/`,
so it belongs in its own folder (AGENTS.md §15, "A second folder"):

```powershell
# in D:\formula-fly-<idea>, set up as AGENTS.md §15 describes
.\train start --config <path to proposal_1_*.yaml> --seeds 0
.venv-flyvis\Scripts\python.exe scripts\lab_compare.py D:\formula-fly D:\formula-fly-<idea>
```

Then review the test folder's run the same way. The previous review's `success_metric` says
what to look for.

## Tests

`tests/coach/test_guardrails.py` pins the crash bound against `lap_time_reward.py`'s 18.44 m,
and checks that each kind of bad proposal is rejected. `tests/coach/test_digest_and_review.py`
checks the corner and ending counts on a hand-built run and runs a whole review with the API
call replaced, so CI needs no key.
