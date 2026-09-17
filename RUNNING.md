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
| Right stick, left/right | Steering (analog) |
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
