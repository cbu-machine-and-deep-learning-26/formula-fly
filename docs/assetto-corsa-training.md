# Training in Assetto Corsa (GH-29)

`scripts/train.py` trains in Assetto Corsa (the original 2014 AC, through AssettoCorsaGym)
with the same PPO loop and YAML conditions as the MuJoCo practice track. Two processes:

- `scripts/ac_env_server.py` runs next to AC on the Windows box, in AssettoCorsaGym's
  Python 3.9 env. It owns the car and answers reset and step requests on TCP 2351.
- The trainer builds `env.type: assetto_corsa`, an `AssettoCorsaTrack`
  (`fly_driver/envs/assetto_corsa.py`) that connects to the server. It has
  `PracticeTrack`'s `reset`/`step` and `info` keys and takes the same `reward` callable.

This is not the driving bridge in [assetto-corsa-bridge.md](assetto-corsa-bridge.md). The
bridge pushes frames and never waits. The env server waits for the trainer to ask.

## How it behaves

**Resets.** Upstream's reset (`AssettoCorsaEnv/ac_env.py` → `ac_client.py` → the
`sensors_par` plugin) closes the UDP link, sends `reset` to the plugin's management port,
which calls Custom Shaders Patch's `ac.ext_resetCar()`, sleeps 1 s, reconnects and flushes
three ticks. CSP documents that call as "resets car to the road invalidating lap": the car
goes back on the road **where it stopped**, at rest. CSP's Python API has no call that puts
the car at a chosen point, and upstream adds none. So:

- There are **no random starts.** Each episode starts where the last one ended, which
  spreads starts round the lap on its own, biased towards the corners the policy fails at.
- There is **no fixed start** for evaluation either. Evaluation uses the same env and
  starts wherever the car is. Lap times are still comparable, because a lap is timed from
  the first crossing of the line (the out lap), as on the practice track.
- `reset(seed=...)` accepts the seed and ignores it.

The server checks every reset: the car must come back with its centre on the track and all
four tyres in, or it resets again, up to three times. After that the trainer's connection
closes and its env raises. Each reset is logged with its duration and position.

**Gear after a reset.** `ac.ext_resetCar()` leaves the SF70H in neutral (AC gear 1; 2 is
first). The auto-shifter only engages first under enough throttle: 0.5 or more takes
0.5-1 s, 0.33 takes 5.6 s, and 0.15 never makes it inside the 8 s slow window. In
`flyvis_cx_ac_v6`, all 177 episodes started in neutral and 34 never left it. So after a
reset lands on the road, the server presses upstream's shift-up (vJoy button 0) for 0.1 s
with the pedals released, checks `actualGear`, and presses once more if the car is still
not in first. A car still out of gear is logged as a warning, not a failed reset. The log
counts gear-ups and failures: `in gear 2 after a reset (N gear-ups so far)` and `still in
gear 1 after two gear-ups (N gear-ups, M failures so far)`.

Getting chosen starts would need a CSP **Lua** app (`physics.setCarPosition`) next to the
plugin. That is new plugin code and not part of this task.

**Episode ends.** Upstream's own terminations are switched off on the server, and the env
decides:

| End | Kind | Rule |
|---|---|---|
| Off track | terminated | `tyres_out / 4 > track_limit`; the default 0.5 is upstream's "three tyres out" |
| Stall | terminated | slower than `stall_speed_mps` (1 m/s) for `stall_s` seconds; off unless set |
| Step cap | truncated | `env.max_steps` |
| Late trainer | truncated | one step missed more than `max_missed_ticks` (5) ticks |
| Interrupted | truncated | AC ended the session, or another client reset the car |

**Real time.** A step applies its control at once and blocks until AC's next tick (20 ms
at 50 Hz). AC does not wait. When the trainer comes back late, the server drains the stale
UDP states and answers with the newest tick, so the observation is never old. The skipped
ticks show as a gap in AC's tick counter: `info["missed_ticks"]` per step,
`env.missed_ticks` in total, and a warning per episode that missed any. While the trainer is
away the car drives on with its last control, so a gap of more than `max_missed_ticks` ends
the episode instead of crediting the policy with driving it did not do. **Every PPO update
is such a gap**, so each rollout boundary costs one episode and one reset.

**Black frames.** Upstream's capture reader strips trailing zero bytes, so an all-black frame
(a camera change or fade) came back empty and the reshape error ended the env server.
`AssettoCorsaCar` reads the capture whole instead (`CaptureReader` in
`fly_driver/bridge/assetto.py`). A read that still comes back short reuses the last frame and
logs `screen capture read ... reusing the last frame (N so far)`.

**Periodic evaluation** opens a second connection while the training one sits idle. The
server serves both under one lock. The evaluation's reset moves the car, so the training
env's next step comes back `truncated` with `info["interrupted"]`, and training resets.

**Track position** comes from AC and upstream's track data: `arclength_m` is
`NormalizedSplinePosition × TrackLength` (0 at the line), `lateral_m` is upstream's distance
to AC's racing line, `on_track` is the car centre inside upstream's occupancy grid, and
`off_track_fraction` is `tyres_out / 4`. There is no off-track time penalty; `lap_time`
equals `lap_time_raw`.

**Protocol** (`fly_driver/bridge/wire.py`). On connect the server sends a JSON hello:
`track`, `car`, `frame_shape`, `frame_rate_hz`. The env refuses a server whose hello differs
from its config. Then the trainer sends `!Bddd` (kind, steer, throttle, brake), with kind 1
for reset and 2 for step. The server answers `!I?IdddddB?` (episode count, ended, tick,
time, lap distance, track length, lateral, speed, tyres out, centre on track) followed by
the bridge's frame record. See `fly_driver/bridge/env_link.py`.

## The AC box: csds-pc-2

Inspected read-only on 2026-10-21 over SSH as `stoff`:

| Needed | State |
|---|---|
| Assetto Corsa | Installed, `C:\Program Files (x86)\Steam\steamapps\common\assettocorsa`, build 14923034 |
| Silverstone | `content\tracks\ks_silverstone`, with the `gp` layout |
| `dallara_f317` | **Not installed.** The Formula cars present are `ks_ferrari_sf70h`, `ks_ferrari_sf15t`, `tatuusfa1` |
| Custom Shaders Patch | **Missing** (no `extension\`, no `dwrite.dll`). Without it there is no reset at all |
| `sensors_par` plugin | **Missing** from `apps\python` |
| Socket DLLs in `system\x64` | **Missing** (only `python33.dll`, `Python33.zip`, `openvr_api.dll`) |
| vJoy | **Missing** (no `C:\Program Files\vJoy`, no device) |
| Python 3.9 / conda | **Missing.** Only Python 3.11 (`C:\Program Files\Python311`); no conda |
| AssettoCorsaGym checkout | **Missing** everywhere under `C:\Users` and `C:\` |
| Occupancy grid `ks_silverstone-gp_0.1m.pkl` | **Missing** (comes from HuggingFace `dasgringuen/assettoCorsaGym`) |
| AC video settings (`CSDS 2`'s profile) | 3440×1440 fullscreen, no FPS cap. Upstream needs 640×480 windowed at 50 FPS |
| AC controls (`CSDS 2`'s profile) | `INPUT_METHOD=KEYBOARD`. Needs vJoy |
| Inbound firewall for 2351 | None. Only needed for a trainer on another machine |
| GPU | RTX 4090 |
| Fly stack | `C:\Users\student\formula-fly` with `.venv-flyvis` and earlier runs |

**The car.** Upstream ships steering maps only for `dallara_f317`, `bmw_z4_gt3` and
`ks_mazda_miata`, and `AssettoCorsaEnv` reads `AssettoCorsaConfigs/cars/<car>/steer_map.csv`
at start-up. Either buy the DLC that contains `dallara_f317`, or use `ks_ferrari_sf70h`,
which is the car the MuJoCo practice track models and the car GH-25's fly driver is fitted
to. For the SF70H, add `AssettoCorsaConfigs/cars/ks_ferrari_sf70h/steer_map.csv` in the
same two-row form as the Dallara's (`action_steerAngle,steerAngle`, then
`-a,-lock` and `a,lock`). Measure `lock` (degrees of steering wheel at full lock) in game,
and take `a = lock / 449.3`, the ratio the Dallara's row implies (`0.534 = 240 / 449.3`).
Then pass `--car ks_ferrari_sf70h` to the server and `car: ks_ferrari_sf70h` in the YAML.

**Sessions.** The console session belongs to `CSDS 2`, whose profile is `C:\Users\griss`
(Documents redirected to OneDrive). AC must render in an interactive desktop, and the
server must run **in the same session as AC**: the plugin's screen capture hands frames
over through `Local\` named events and shared memory, which Windows scopes to one session.
An SSH shell is a different session, so the server cannot be started from SSH directly.

- Start AC at the console as `CSDS 2` (Steam → Assetto Corsa → Hotlap, SF70H or Dallara,
  Silverstone GP, `sensors_par` enabled).
- Start the server in that session with a scheduled task that runs only when `CSDS 2` is
  logged on (`/IT`). As an admin over SSH:

  ```powershell
  schtasks /Create /TN fly_ac_env_server /SC ONCE /ST 00:00 /RU "CSDS 2" /IT /F `
    /TR "cmd /c cd /d C:\Users\student\formula-fly && C:\path\to\p309\python.exe scripts\ac_env_server.py --gym-root C:\src\assetto_corsa_gym --car ks_ferrari_sf70h > ac_env_server.log 2>&1"
  schtasks /Run /TN fly_ac_env_server
  ```

  `schtasks` may ask for `CSDS 2`'s password even with `/IT`; if it does, create the same
  task in Task Scheduler with "Run only when user is logged on". Whoever is logged on at the
  console sees the server's window; that is expected.
- The trainer does not need the desktop. Run it over SSH as `student` against
  `127.0.0.1:2351`; loopback TCP crosses sessions.

## One-time setup on csds-pc-2

Do the AssettoCorsaGym install from [running-the-stacks.md](running-the-stacks.md#assettocorsagym-car)
and the plugin settings from [assetto-corsa-bridge.md](assetto-corsa-bridge.md#one-time-setup-on-the-windows-box),
which together cover every **Missing** row above: Content Manager and CSP, vJoy, the
`p309` conda env, the plugin and its `system\x64` libraries, the occupancy grid, 640×480
windowed at 50 FPS, and `sensors_par/config.py` with `ego_sampling_freq = 50`,
`screen_capture_enable = True`, `screen_capture_freq = 50`, 96×96 and `bgr`. Do it in
`CSDS 2`'s session, because AC's settings live in that user's Documents.

## Run it

1. AC in a Hotlap at the console, as above.
2. The env server, by the scheduled task. Its log should show `env server for
   ks_silverstone-gp/<car> on 127.0.0.1:2351`.
3. A condition YAML for the trainer (keep it outside `fly_driver/configs/` until a
   reviewed config lands there):

   ```yaml
   name: ac_flyvis_frozen
   seeds: [0]
   total_env_steps: 1000000
   eye: {type: flyvis, frozen: true}
   env:
     type: assetto_corsa
     max_steps: 15000
     params:
       host: 127.0.0.1
       port: 2351
       track: ks_silverstone-gp
       car: ks_ferrari_sf70h      # or dallara_f317 once installed
       stall_s: 5.0
       # track_limit: 0.5, terminate_off_track: true, max_missed_ticks: 5
   ppo: {rollout_steps: 4096, minibatches: 32, epochs: 10}
   eval: {every_updates: 0}
   ```

4. The trainer, over SSH as `student`. A process started in a Windows SSH session dies
   when the session closes, so for a real run launch it detached (a `schtasks` task as
   `student`, or `Win32_Process Create`) and log to a file:

   ```powershell
   cd C:\Users\student\formula-fly
   .venv-flyvis\Scripts\python.exe scripts\train.py --config ac_flyvis_frozen.yaml
   ```

**Split host** (trainer on a Spark or a Mac): start the server with `--host 0.0.0.0`, allow
inbound TCP 2351 on csds-pc-2, and set `host:` to the box's address. The env is identical.

**Dry run without AC**, anywhere: `python scripts/ac_env_server.py --fake`, then the same
YAML with `port: 2351`. The fake reports whatever `--track` and `--car` it is given.
Verified on a Mac: `--total-env-steps 4096` with the pixel eye, 1024-step rollouts and an
evaluation every two updates completed in 114 s, with 20 training episodes, 23 resets and
both evaluations interrupting the training episode as designed. The fake also runs under
Python 3.9.

## What to check on the box

1. **Where a reset lands.** The server logs `reset N took S s: at D m, T tyres out`. Drive
   off at a known point and confirm the car comes back on the road near `D`, at rest, facing
   forwards. If it lands somewhere else (the pits, the grid), the "no fixed start" note above
   is wrong and the doc needs correcting.
2. **Reset time.** Upstream sleeps 1 s and flushes three ticks, so expect 1.1 to 1.5 s.
   Retries (`reset 2`, `reset 3`) mean the car came back off the road.
3. **Tick rate.** At 50 FPS with capture on, `info["missed_ticks"]` must be 0 on almost
   every step outside PPO updates. A steady trickle of single misses means AC is not
   holding 50 Hz or the trainer's step (eye + policy) is too slow; time it.
4. **Steer sign and scale**, as in the bridge doc: `steer=+0.5` must turn right, and full
   lock must match the car's wheel.
5. **Frames.** The first frame of each episode should be the cockpit view at 96×96 RGB.
6. **Off-track rule.** Put two, then three tyres on the grass; the episode must end at three.

## Throughput

Real time bounds everything: 50 steps/s, so **1M steps take 5.6 h** of driving. On top of
that, per episode, one reset of about 1.1 to 1.5 s, and per PPO update, the update itself
plus the episode it truncates. With 4096-step rollouts, 1M steps is 244 updates; at, say,
10 s per update on the 4090 and 1.5 s per reset, that adds about 45 min plus the resets of
ordinary episode ends. Seeds and conditions cannot run in parallel on one AC. Expect about
one condition-seed per working day.

On the Mac the fake's ticks slip by tens of milliseconds because `time.sleep` overshoots
there by about 44 ms, so the fake reports missed ticks that real AC would not. That is the
counter working, not the protocol.

## Behaviour cloning on human laps

Pure online RL here is bounded by the wall clock, and the MuJoCo-trained driver does not
steer on AC's frames. So the AC driver starts from a clone: the round-6 MuJoCo checkpoint
(frozen flyvis eye, 4,096-neuron central complex, MLP head), fine-tuned by behaviour
cloning on our own human laps, then handed to PPO here through `init_from`.

**Why our own laps.** AssettoCorsaGym's human dataset (HuggingFace `dasgringuen/assettoCorsaGym`)
has no Silverstone laps, no SF70H, and no images: 183 telemetry channels at 25 Hz from the
Dallara F317, BMW Z4 GT3 and Miata at Barcelona, Red Bull Ring, Monza and Indianapolis.
Our policy sees only 96×96 nose-camera frames, so we drive the SF70H at Silverstone GP
ourselves and render the frames from the replay.

### From a drive to a dataset

1. **Drive** (below), and save the replay.
2. **Re-render the replay** on an AC box that is not training, in the desktop session AC
   runs in (csds-pc-2's `student` console for the first drive). Copy `scripts/bc_replay_app/`
   to `<AC>\apps\lua\fly_bc_replay\` once, copy the replay to
   `Documents\Assetto Corsa\replay\`, set `race.ini`'s `[REPLAY] ACTIVE=1` and
   `FILENAME=<replay file name>`, set `video.ini`'s `SMOKE=0` (back both up first), and
   start AC as for training (`fly_ac_start`: 640×480 windowed, 50 FPS, nose camera). Smoke
   must be off: AC renders each held step several times, and its tyre-smoke emitters spawn
   particles on every one, so with training's `SMOKE=3` the replay fills with a haze that
   real driving never shows. The only frames that lose real smoke are lock-ups and spins. The Lua app keeps the camera on
   the F1 nose view (`DRIVABLE_MODE=4`), hides all of AC's UI in the replay (its playback
   bar would sit in every frame), and holds the replay wherever
   `scripts/bc_replay_capture.py` asks, so every frame is one 20 ms step of replay time
   whatever the frame rate. The grab is `sensors_par`'s own, so the frames are what the env
   server sends. Run it in AC's session (pc-2's `fly_ac_job`), in AssettoCorsaGym's env:

   ```powershell
   python scripts\bc_replay_capture.py --out C:\formula-fly-ac\bc\capture_<drive>
   ```

   It runs at about 9 frames/s, so a 4-minute drive takes 20 minutes. Afterwards stop AC
   (`fly_ac_stop`) and restore `race.ini` and `video.ini`, or AC opens the replay again
   and training runs without smoke.
3. **Pair frames with inputs**, anywhere with the repo:

   ```bash
   python scripts/bc_build_dataset.py --replay <drive>.acreplay --capture <drive>/capture \
       --out data/human_laps/<drive>.npz --strip <drive>_strip.png
   ```

   Three corrections make the frames match training's:
   - **Timing.** AC draws a held replay instant 20 ms ahead of the one asked for. The
     builder fits that shift from the car's pose (`fit_render_offset`: AC's reference point
     sits a fixed 0.58 m from the replay's, so on time the gap is constant in size) and
     reads the inputs at the instant actually drawn. Every frame must report the replay
     frame it asked for, and above 30 m/s (where the pose resolves time) a frame whose gap
     strays from the median by more than 10 ms of travel, half a step, is refused. On the first drive
     AC's frame index matched on all 11,468 frames, the fitted shift was +19 ms, and the
     worst frame was 6.7 ms off: AC blends between replay frames a little differently from
     linear interpolation, most visibly at 300 km/h.
   - **Sidebar.** Training frames carry AC's collapsed app sidebar down the right edge
     (columns 89-95 at 96×96, the scene at 0.117 × its brightness, a red marker at the
     bottom); the replay shows no UI, so `add_ac_sidebar` draws it, as measured on a
     training capture.
   - **Parked frames** (under 1 m/s with no throttle) are dropped; a launch is kept.
     Segments split at laps and gaps.

   The strip shows the hardest corners with their controls, half a second before and at the
   steering peak: check the sign by eye.

**Inputs.** `fly_driver/training/bc_data.py` reads the replay directly (version 16, the
player car): steering-wheel angle, gas and brake pedals, gear, pose, lap. On the first
drive they agreed with AC's own telemetry export (`telemetry_dump.act`, 20 Hz) to a steer
MAE of 0.47° (correlation 0.9997), throttle 1.6% and brake 0.8%, at a 10 ms lag. Steer is
the wheel angle over the SF70H's 180° lock, the env's `steer = ±1` (its `steer_map.csv`
maps ±0.4006 vJoy to ±180°); throttle and brake are the pedals, in [0, 1]. A 30 ms replay
frame is interpolated linearly to the 20 ms steps.

### Train

```bash
python scripts/bc_train.py --config fly_driver/configs/bc_flyvis_cx_ac.yaml
```

`fly_driver/training/bc.py`'s docstring has the details. The frozen eye runs once per drive
and its features are cached beside the `.npz`. The brain and the actor load from
`init_from` (round 6's best, `weights/round6_best/policy_update_00070.pt`). They learn the
human's actions by squared error of the clipped action. That error counts nothing the
env's clip would absorb anyway, so brake at -0.6 for a target of 0 is free. The brain gets
PPO's one-step gradient from membranes recomputed each epoch, in segment order. `log_std`
and the critic stay as the base checkpoint had them: human laps carry no rewards, and PPO
should start with its usual exploration. The policy's feature normaliser is refitted to the
brain's outputs on these laps. The last `holdout_laps` laps pick the epoch; the run writes
`policy_final.pt` and `brain_final.pt` for that epoch, `config.yaml` (the AC base config)
and `bc_log.csv` with the per-action error on both splits. Point a PPO config's
`init_from` at `runs/bc_flyvis_cx_ac/policy_final.pt`.

### First drive (2026-10-22)

Ryan's first drive: one valid lap (2:35.96 by AC's clock, including the pit start) and 73 s
of a second, re-rendered on csds-pc-2 in 19 minutes. After dropping the parked start,
10,295 frames: 6,626 from lap 0 (training) and 3,669 from lap 1 (held out). All 11,468
rendered frames reported the replay frame they asked for; above 30 m/s the worst was 7 ms
off. Positive steer turns right: over lap 0 the heading turns -361° (clockwise) and steer
correlates -0.87 with that sign's yaw rate.

BC on it, held-out mean absolute error of the clipped action (steer, throttle, brake):

| | Steer | Throttle | Brake |
|---|---|---|---|
| Round-6 checkpoint, no BC | 0.854 | 0.504 | 0.299 |
| BC, brain and actor (epoch 24 of 30) | 0.469 | 0.435 | 0.166 |
| BC, actor only (epoch 30, still falling) | 0.407 | 0.448 | 0.183 |
| A constant: the training lap's median action | 0.154 | 0.383 | 0.121 |

BC removes half the base checkpoint's error, but one lap is not enough: the clone does
worse on the held-out lap than a constant, steering left on average (-0.40). On the
training lap it reaches 0.13 steer error with a correlation of only 0.35. Don't use this
checkpoint as an `init_from`. Train again with 10 or more laps; the target is 20 clean laps
plus 5 recovery laps.

### Recording laps

At the console of an AC box that is not training, with a wheel or a gamepad:

- **Content Manager or AC's launcher → Hotlap**, car **Ferrari SF70H**
  (`ks_ferrari_sf70h`), track **Silverstone, GP layout**. Weather `4_mid_clear` and the
  default time of day, as in training: the frames must look like the ones the fly sees.
- **Assists:** automatic gearbox and clutch (the env auto-shifts; gear is not an action).
  Ideal line off. Any steering, throttle and brake aids off.
- **Drive.** A Hotlap starts in the pits at rest. Drive at a pace you can repeat. After
  the clean laps, drive a few **recovery laps**: on each straight, let the car drift to
  two wheels on the kerb and steer back to the middle, and once a lap stop on the track
  and launch again. The fly's own failures look like this (resets in AC leave the car at
  rest where it stopped), and clean laps never show them. Keep three tyres inside: three
  out ends an env episode.
- **Save each replay before leaving the session.** AC keeps only one temporary replay and
  overwrites it in the next session: open Replay, then Save, after every session of a few
  laps. Copy each `.acreplay` from `Documents\Assetto Corsa\replay\` to
  `D:\formula-fly-ac\human_laps\<date>_<driver>\`, then re-render it as above.
