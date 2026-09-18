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
| `Backspace` | Reset to the start |
| `F1` | MuJoCo's full shortcut list |

---

## Lap times

Every completed lap is appended to [`lap_times.md`](lap_times.md) with the date, the time
and who drove it. The `BEST` shown on the panel is simply the fastest row in that file,
re-read whenever the file changes — so you can delete your own laps and the record updates
immediately, even while the simulator is running.

---

## Options

```
--input {auto,keyboard,gamepad}   force an input device (default: auto)
--raw-steer                       full lock at any speed, as the fly gets
--no-hud                          hide the telemetry panel
--lap-log PATH                    write laps somewhere other than lap_times.md
--no-lap-log                      time laps but do not record them
--walls                           add collidable walls at the track edges
--fovy DEGREES                    camera field of view
--export model.xml                write the MJCF instead of driving
```

Full list: `scripts/drive.py --help`.

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

`PracticeTrack` renders only the fly-head camera, at `FRAME_SHAPE`, and skips the
offscreen shadow pass (the GLFW viewer still has shadows). It also sizes MuJoCo's
offscreen FBO to that camera rather than `SceneConfig`'s 1280² ceiling — `MjrContext`
allocates the model buffer, not the 96×96 viewport, and a 2048² shadow map is a second
full scene pass. Physics is still 10× 2 ms substeps per env step. Re-time on the Mac
with the snippet in the GH-63 notes; a box that used to quote ≈2.5 ms/step was not
the Mac, and the old path was paying for a buffer the eye never reads.

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
