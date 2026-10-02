# Running the simulator

How to get the MuJoCo practice track driving on a fresh machine. Pick your platform below —
the steps are the same three commands everywhere, only the paths differ.

You need **Python 3.10 or newer** and **git**. Everything else installs from pip.

---

## Windows

```powershell
git clone https://github.com/cbu-machine-and-deep-learning-26/formula-fly.git
cd formula-fly

py -3.11 -m venv .venv
.venv\Scripts\python.exe -m pip install -U pip
.venv\Scripts\python.exe -m pip install -e ".[dev]"
```

Drive:

```powershell
.venv\Scripts\python.exe scripts\drive.py
```

---

## Linux

```bash
git clone https://github.com/cbu-machine-and-deep-learning-26/formula-fly.git
cd formula-fly

python3 -m venv .venv
.venv/bin/python -m pip install -U pip
.venv/bin/python -m pip install -e ".[dev]"
```

Drive:

```bash
.venv/bin/python scripts/drive.py
```

If the viewer fails to open, you are missing OpenGL. On Debian/Ubuntu:

```bash
sudo apt install libgl1 libglu1-mesa libxrandr2 libxinerama1 libxcursor1 libxi6
```

**On Wayland** (Hyprland, GNOME on Wayland, Omarchy): keyboard driving may not work.
`pynput` captures keys globally through X11, and Wayland deliberately blocks that. A
gamepad works fine, or log into an X11/Xorg session. See [Controls](#controls).

---

## macOS

```bash
git clone https://github.com/cbu-machine-and-deep-learning-26/formula-fly.git
cd formula-fly

python3 -m venv .venv
.venv/bin/python -m pip install -U pip
.venv/bin/python -m pip install -e ".[dev]"
```

Drive — **note this says `mjpython`, not `python`**:

```bash
.venv/bin/mjpython scripts/drive.py
```

macOS requires GUI work to happen on the main thread, so MuJoCo ships its own launcher and
refuses to open the viewer otherwise. Using plain `python` gives you:

```
RuntimeError: `launch_passive` requires that the Python script be run under `mjpython` on macOS
```

`mjpython` is installed into `.venv/bin/` by the `mujoco` package. Everything that is not
the viewer — the tests, `--export` — runs under normal `python`.

> Not yet tried on a Mac by anyone on the team. The `mjpython` requirement is MuJoCo's own,
> and the rest of the stack is pure Python and numpy, so it should be uneventful. If you are
> the first to run it, please note anything that bites in issue #16.

---

## Controls

The car starts **150 m behind the start line**. Drive up to it — the lap clock starts when
you cross, so the panel shows `OUT` until then.

### Gamepad (recommended)

Plug it in before launching; it is detected automatically.

| Control | Action |
|---|---|
| Left stick, left/right | Steering (analog) |
| Right trigger | Throttle (analog) |
| Left trigger | Brake (analog) |

### Keyboard

Arrow keys only — the viewer claims the letters and digits for itself.

| Key | Action |
|---|---|
| ↑ | Throttle |
| ↓ | Brake |
| ← / → | Steer |

Keyboard input is all-or-nothing by design: the interface underneath is analog (the fly, or
a gamepad, can ask for 50% throttle), but a key is either down or up.

### Viewer

| Key | Action |
|---|---|
| `[` / `]` | Cycle cameras — press `]` to sit in the fly's head camera |
| `Esc` | Back to the free camera (orbit with the mouse) |
| `Space` | Pause / resume |
| `Backspace` | Reset to the start, and clear the lap clock, segment times and penalty |
| `F1` | MuJoCo's full shortcut list |

---

## Lap times

Every completed lap is appended to [`lap_times.md`](lap_times.md) with the date, who drove
it, and three times:

| Column | Meaning |
|---|---|
| `Lap` | The time that counts: `Raw` plus `Penalty` |
| `Penalty` | Seconds earned by leaving the circuit (0 for a clean lap) |
| `Raw` | The stopwatch time |

The `BEST` shown on the panel is simply the fastest `Lap` in that file, re-read whenever the
file changes — so you can delete your own laps and the record updates immediately, even while
the simulator is running. A lap only counts when you cross the line **on the circuit**; drift
over it through the grass and the lap is held until you rejoin.

### Leaving the circuit costs time, not the lap

Running wide no longer sends you back to the grid. You pay seconds instead, scaled by how far
off the car went (capped at one car width) and how long it stayed out: a kerb brush is about
0.57 s, a fully cut corner about 7.1 s. The penalty shows in red beside `LAP` while you are
still earning it, and is added to the lap at the line. The grass also grips at about half the
road's level, per wheel, so dropping two wheels off at speed can spin the car.
`--track-limit` brings the old restart back.

### Segment times

The lap is cut into 21 segments — corners and straights, found from the circuit's curvature —
marked by grey pillars at each boundary (the checkered pair is the start/finish line). Crossing
one prints the time you just set and how it compares with your best, e.g.
`T4  0:06.221  -0.184  NEW BEST`, and the panel shows the same delta in green or red.

The best clean time through each segment is kept in a "Best segments" table in
`lap_times.md`. A segment you went off in is timed but **never** sets a best — a corner cut is
not a corner time. As with laps, delete a row and the next car through sets it again.

---

## Options

```
--input {auto,keyboard,gamepad}   force an input device (default: auto)
--raw-steer                       full lock at any speed, as the fly gets
--no-hud                          hide the telemetry panel
--lap-log PATH                    write laps somewhere other than lap_times.md
--no-lap-log                      time laps but do not record them
--walls                           add collidable walls at the track edges
--racing-line                     paint the racing line on the road (off by default here)
--track-limit FRACTION            restart when this fraction of the car's width is on the
                                  grass; 0 (the default) charges a time penalty instead
--fovy DEGREES                    camera field of view
--export model.xml                write the MJCF instead of driving
--eye-view                        show the track through the fly's eye (needs .venv-flyvis)
```

The racing line is off by default only in this tool. The environment keeps it on by default,
because there it is a cue the fly is meant to see.

Full list: `scripts/drive.py --help`.

### Seeing the track through the fly's eye

`--eye-view` runs the flyvis optic lobe on the fly's head camera while you drive, and draws
what it made of each frame in the same window. It needs the flyvis virtualenv (see
[Connecting the fly's eye](#connecting-the-flys-eye)):

```powershell
$env:FLYVIS_ROOT_DIR = "$HOME\.cache\flyvis"
$env:CUDA_VISIBLE_DEVICES = "-1"     # see below
.\.venv-flyvis\Scripts\python.exe scripts\drive.py --eye-view
```

**F10** cycles three views:

| View | What it shows |
|---|---|
| corner (the start) | Under the telemetry panel: the fly's **retina** on the left (721 hexagonal columns of brightness, which is all the resolution a fly has), and its **motion percept** on the right. |
| big | Beside the panel: the two fused, with the retina dimmed and the motion coloured over it. This is the track as the fly's motion cells see it. |
| off | Nothing drawn. The eye keeps running, so it's the same eye when you bring it back. |

The motion colours are the webcam demo's, from the T4/T5 motion detectors. **Hue is the
direction** things are moving across the eye: right red, up yellow-green, left cyan, down
violet. **Brightness is how strongly.** "Still" means the eye left to settle on the view in
front of it, so a parked car is black and the colours are motion, not scenery. Drive and
watch the field light up; turn and watch it change colour. The view settles again whenever
you go back to the grid, so the jump there doesn't show as a burst of motion.

The terminal's status line adds the eye's cost per frame: about 9–12 ms on CPU. That fits in
the 20 ms frame here, though on a slow machine the sim drops below real time.

`CUDA_VISIBLE_DEVICES=-1` is needed on any machine with an NVIDIA GPU, for now. `import
flyvis` makes CUDA torch's default device, and the hex resampler then crashes on CPU frames.
That's a separate bug. Use `-1`, not an empty value: on Windows an empty value leaves torch
reporting CUDA with no devices.

---

## Driving it from code

`scripts/drive.py` is for humans. Anything else — a random policy, a trained one, the fly —
goes through the environment. It speaks the Gymnasium surface the evaluation harness expects,
so anything that can drive `DummyTrackEnv` can drive this:

```python
from fly_driver.envs.practice_track import PracticeTrack

with PracticeTrack(max_steps=5_000) as env:
    frame, info = env.reset(seed=0)         # (96, 96, 3) uint8, from the fly's head camera
    while True:
        action = (0.0, 1.0, 0.0)            # (steer, throttle, brake)
        frame, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:         # one step = 1/50 s of simulated time
            break
```

The action may be a `ControlVector`, a length-3 array, or a plain tuple. Out-of-range values
raise rather than being quietly clipped — use `ControlVector.clipped(...)` to squash a raw
policy output on purpose.

`info` carries the raw signals a reward is built from — `progress_m`, `speed_mps`,
`lateral_m`, `off_track_fraction` — plus `lap_complete`, `lap_time` (the **completed** lap's
time, `None` otherwise) and `lap_elapsed_s` (the running clock). Every component of the
reward is published separately in `info["reward_terms"]`.

`lap_time` is the time that counts: the stopwatch **plus** any off-track penalty, so a lap that
cut a corner cannot out-rank a clean one. The stopwatch alone is `lap_time_raw`, and that lap's
penalty is `lap_penalty_s`. While driving, `penalty_s` is what the current lap owes so far and
`excursions` how many times it has gone off. Segment timing is in `segment` (the one being
driven), `segment_elapsed_s` and `segment_dirty`, and on the step a segment finishes,
`segment_complete`, `segment_time` and `segment_clean`.

The episode **ends** when the car goes past `track_limit` (15% of its width onto the grass) —
training, the evaluation harness and the viewer all rely on that. Pass
`terminate_off_track=False` to keep driving and pay the time penalty instead, but only with a
reward built for it: the default reward charges its off-track term on every step the car is
off, so a long excursion would swamp everything else.

The default reward is distance covered, less a small penalty for running wide, plus a bonus
for a completed lap; its constants match `DummyTrackEnv`'s so scores on the two are
comparable. Pass your own with `PracticeTrack(reward=...)` — it takes `(info, dt)` and
returns `(total, terms)`.

### A note on reproducibility

The physics is bit-identical for a given seed and action sequence. The rendering is not
quite: MuJoCo hands rasterisation to the GPU, and two identical runs occasionally disagree
by one level out of 255 on a scattering of subpixels along polygon edges. That is far below
anything the eye can see, but **do not hash frames or compare them exactly** in a
determinism check — compare the physics, or compare frames with a tolerance.

### Connecting the fly's eye

The environment never imports flyvis or torch, and it should stay that way: the optic lobe
needs Python < 3.13 and platform-specific CUDA wheels, so it lives in **its own virtualenv**.
Do not `pip install torch` into `.venv` — follow
[`docs/running-the-stacks.md`](docs/running-the-stacks.md), which builds `.venv-flyvis` from
`requirements-flyvis.txt`.

What makes the two halves fit is that the frame this env emits is exactly the frame the eye
declares. Take the numbers from the env rather than typing them again:

```python
from fly_driver.eyes import FlyvisEye

eye = FlyvisEye(frame_shape=env.frame_shape, frame_rate_hz=env.frame_rate_hz)

frame, info = env.reset(seed=0)
eye.reset()                      # required at every episode start — the optic lobe keeps
                                 # state between frames
features = eye.encode(frame)     # (5768,) float32 -> brain -> policy -> action
```

Two constants in `fly_driver/interface.py` hold the agreement:

| | |
|---|---|
| `FRAME_SHAPE = (96, 96, 3)` | ours to change, as long as both sides read it from here |
| `FRAME_RATE_HZ = 50.0` | **fixed.** One frame is one Euler step of flyvis; it raises below 50 Hz |

Measured on this machine: the env runs about **400 steps/s** with rendering (≈2.5 ms/step,
of which ≈1.5 ms is the render), against the eye's 7.9 ms per frame. So the optic lobe is
the bottleneck, not the track, and one environment comfortably feeds one eye in real time.

---

## Training

Reinforcement learning on the practice track is `scripts/train.py`, driven by a YAML
condition file (`fly_driver/configs/train_default.yaml`: frozen flyvis eye, no brain,
seeds 0, 1 and 2). It needs torch, which the base `.venv` deliberately does not have, so run
it from the flyvis virtualenv, or from a `.venv-train` with CPU torch when flyvis cannot be
installed (Python 3.13 machines) -- the pixel smoke eye exercises the whole loop:

```bash
python scripts/train.py --seed 0                  # in .venv-flyvis: the default condition
python scripts/train.py --eye pixels --seed 0     # any venv with torch: the smoke eye
```

Every run writes `episodes.csv` (reward split by term), `updates.csv`, a policy checkpoint
and periodic evaluations under `runs/<name>/seed_<k>/`. Config reference, outputs and the
frozen-versus-fine-tuned rules: [`docs/training.md`](docs/training.md).

---

## Running the tests

```bash
.venv/bin/python -m pytest -q          # Windows: .venv\Scripts\python.exe -m pytest -q
```

Tests that need a GPU/OpenGL context skip themselves automatically on a headless machine.
Before opening a pull request, also run:

```bash
.venv/bin/python -m ruff format --check .
.venv/bin/python -m ruff check .
```

---

## If something goes wrong

**`keyboard driving needs pynput`** — the dev extra is not installed. Re-run the install
command with `".[dev]"` exactly as written, quotes included.

**The viewer window never opens** — you are on a machine with no display, or missing
OpenGL (see Linux above). `--export model.xml` still works and writes the track out so you
can open it elsewhere.

**Nothing happens when I press the arrow keys** — on Linux, see the Wayland note. On any
platform, check the terminal line: it prints which input device was picked at startup.

**The car is sitting still and the panel says `OUT`** — that is correct. The clock has not
started because you have not crossed the line yet.

**MuJoCo writes `MUJOCO_LOG.TXT`** — harmless; it is git-ignored.
