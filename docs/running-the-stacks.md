# Running the standalone stacks

These checks keep the eye, body, and car integrations independent.

## flyvis (optic lobe)

flyvis requires Python 3.9-3.12. Run on Linux for development and on the DGX
Sparks for experiments. Use an NGC PyTorch image on Spark rather than resolving
PyTorch wheels directly on ARM.

```bash
python3.12 -m venv .venv-flyvis
source .venv-flyvis/bin/activate
python -m pip install -r requirements-flyvis.txt
export FLYVIS_ROOT_DIR="$HOME/.cache/flyvis"
flyvis download-pretrained
python scripts/flyvis_smoke.py
```

The download supplies the `flow/0000/000` task-optimized checkpoint. The smoke
script sends a four-frame `(1, 4, 1, 721)` sequence through the complete network,
reports the full activity tensor, and exposes configurable cell-type output
readouts (`--readout T4a`, repeated as needed). T4a-d and T5a-d are the defaults.
It exits successfully with a `SKIP` message when the optional package or
checkpoint is absent, so base CI does not need flyvis.
Use `--stimulus edge --direction {ltr,rtl}` for 0.5 seconds of gray baseline, a
1-second edge sweep, and a 0.2-second hold instead of the default ramp.
`--plot PATH` samples 10 readable frames and writes `PATH_input.png` and
`PATH_responses.png` on flyvis's own hex-lattice layout.

Observed with flyvis 1.2.0: full response `(1, 4, 45669)`; each T4a-d/T5a-d
readout `(1, 4, 721)`; concatenated T4/T5 readout `(1, 4, 5768)`. The model
exposes 34 output types: T1, T2, T2a, T3, T4a-d, T5a-d, Tm1, Tm2, Tm3, Tm4,
Tm5Y, Tm5a-c, Tm9, Tm16, Tm20, Tm28, Tm30, TmY3, TmY4, TmY5a, TmY9, TmY10,
TmY13, TmY14, TmY15, and TmY18.

### Camera-to-hex geometry

`fly_driver.eyes.hex_resampler.HexResampler` converts a declared
`96 × 96 × 3` uint8 RGB frame to flyvis input `(batch, time, 1, 721)` without
requiring flyvis at runtime. It uses BT.601 luminance
(`0.299 R + 0.587 G + 0.114 B`) and the exact
`flyvis.utils.hex_utils.get_hex_coords(15)` column order: `u` changes slowest,
then `v`.

For each axial `(u, v)` coordinate, the Cartesian image offsets are
`row = 13 × (u + v / 2)` and `column = 13 × v`. Image row zero is the top, so
negative offsets are above/left of center. At fixed `u`, increasing `v` moves
down and right; this fixes chirality as well as orientation.

The declared camera frame is deliberately projected to flyvis's 391-pixel
sampling field, mean-filtered with a 13-pixel kernel, and sampled at receptor
centers. Undeclared frame sizes error instead of being silently resized. The
tests prove ordering, orientation, chirality, output parity with `BoxEye`, and
pretrained T4/T5 direction selectivity:

```bash
pytest -q tests/eyes
```

`python scripts/hex_resampler_plot.py PATH.png` renders the bright top-left
test pattern next to its 721-column resampling (matplotlib only, no flyvis).

### FlyvisEye (default visual frontend)

`fly_driver.eyes.FlyvisEye` wraps the pretrained, frozen optic lobe behind the
project interface: a `96 × 96 × 3` uint8 camera frame in, a float32 feature
vector out. `import fly_driver.eyes` works without flyvis; constructing the eye
without it raises `FlyvisNotInstalledError` with install instructions.

```python
from fly_driver.eyes import FlyvisEye

eye = FlyvisEye()                     # flow/0000/000 under $FLYVIS_ROOT_DIR/results
eye.reset()                           # 1 s grey warm-up; call at every episode start
features = eye.encode(frame)          # (5768,) float32, one call per env frame
episode = eye.encode_sequence(frames) # (T, 5768); resets first by default
```

Construction options: `checkpoint` (name under `flyvis.results_dir` or a path),
`readouts` (default `T4a-d, T5a-d`; any of the 34 output cell types listed
above, names validated), `frame_shape`, `device`, `frame_rate_hz`,
`warmup_seconds`, `warmup_luminance`. `feature_dim == len(readouts) * 721`;
the vector is the readouts concatenated in `readout_names` order, each in
flyvis hexagonal column order (the same `(u, v)` order as the resampler).
`trainable_parameters()` is `0`; `state_activity` exposes all 45,669 neurons
for diagnostics.

**Timing is the design decision.** flyvis is a dynamical system integrated at
`dt = 1/50` s, and one frame is one Euler step. `encode` keeps the network
state between calls, so an RL loop that calls it once per environment step runs
the optic lobe at the environment's frame rate. That rate must be 50 Hz, which
is Gymnasium CarRacing's `FPS`; slower rates raise (`dt > 1/50` is outside
flyvis's integration limit) and faster rates warn. Streaming frame by frame and
`encode_sequence` on the same frames are asserted equal. `reset()` warms up on
a uniform grey camera frame pushed through the resampler (so the darkened
boundary columns are already at rest); 1 s is the default because 0.5 s still
leaves a 0.27 a.u. transient versus 0.027 at 1 s.

Measured on this 4-thread x86 CPU with flyvis 1.2.0 / torch 2.14:
`encode()` median 7.9 ms per frame (p95 8.1 ms; ~1.5 ms resampler + ~6 ms
network), `encode_sequence()` 8.4 ms per frame including the reset, `reset()`
280 ms, checkpoint load 4 s. That is 40% of the 20 ms frame budget at 50 Hz, so
a single environment can run the eye in real time on CPU. Reproduce with
`python scripts/flyvis_eye_demo.py --out DIR`, which also writes
`flyvis_eye_edge_{right,left}.png` (camera frame plus T4/T5 maps over time) and
`flyvis_eye_direction_preference.png` (per-subtype grating preference), and
exits with `SKIP` when flyvis or the checkpoint is absent.

Tests: `tests/eyes/test_flyvis_eye.py` runs without flyvis (lazy import,
validation); `tests/eyes/test_flyvis_eye_pretrained.py` needs the checkpoint and
covers frozen weights, streaming/sequence equality, reset determinism, warm-up
settling, T4/T5 direction selectivity through the eye, and 10 s of noise
staying finite and below 20 a.u. The synthetic gratings and moving edges live
in `fly_driver.eyes.stimuli`.

### Live demo (webcam → eye)

`scripts/flyvis_eye_live.py` is a viewer, not a training component: it shows
the 96 × 96 frame the eye receives, the eight T4/T5 hex maps updating live
through `FlyvisEye.encode`, and a direction meter with one bar per direction
(left/right/up/down from the T4/T5 a/b/c/d subtypes, resting activity
subtracted). Wave a hand across the camera and the bar for that direction
jumps. The overlay shows display fps, per-frame eye latency, and the camera
rate next to "eye stepped at 50 Hz" because every frame, however fast it
arrives, is one 20 ms step of the optic lobe.

```bash
# in the flyvis venv (Python 3.9-3.12); uv is fastest, pip works too
uv pip install -r requirements-flyvis.txt     # adds opencv-python for the camera
export FLYVIS_ROOT_DIR="$HOME/.cache/flyvis"  # where `flyvis download-pretrained` put results/
python scripts/flyvis_eye_live.py             # webcam 0; falls back to --source synthetic
python scripts/flyvis_eye_live.py --source synthetic   # bright bar sweeping l/r/u/d, 2 s each
```

Keys: space pauses/resumes, `r` resets the eye state (1 s grey warm-up),
`q`/Esc quits. Options: `--camera-index`, `--fps-cap` (default 50),
`--meter-statistic q95|mean`, `--color-limit`, `--save-dir DIR --save-every N`
for PNG snapshots. macOS: grant camera permission to the terminal app you run
it from (System Settings → Privacy & Security → Camera); the first run prompts.
Windows and Linux work the same way through OpenCV.

Headless check, also what `tests/scripts/test_flyvis_eye_live.py` runs:

```bash
python scripts/flyvis_eye_live.py --source synthetic --frames 200 --no-display
```

It prints the meter every 10 frames, the mean display fps, eye latency
(median/p95/max against the 20 ms budget), and how often the winning meter
direction agreed with the bar's direction. The script exits `0` with `SKIP:`
when flyvis, the checkpoint, or (for the webcam) OpenCV is missing, so base CI
does not need any of them. The hex map drawing is shared with
`scripts/flyvis_eye_demo.py` via `fly_driver.analysis.hex_plots`.

## flybody (MuJoCo body)

Use a separate Linux environment. Upstream recommends Python 3.10; this x86_64
VM also passed the core check on Python 3.12 at commit
`d015e9bfe441bd90ae431bac24c55cb74bdbce26`.

```bash
python3 -m venv .venv-flybody
source .venv-flybody/bin/activate
python -m pip install \
  "flybody @ git+https://github.com/TuragaLab/flybody.git@d015e9bfe441bd90ae431bac24c55cb74bdbce26"
sudo apt-get install libosmesa6
MUJOCO_GL=osmesa python -c \
  "from flybody.fly_envs import template_task; print(template_task().action_spec())"
```

The core environment constructed, reset, and stepped here with action shape
`(59,)`, 10 observation entries, and reward `1`. On a GPU host, use
`MUJOCO_GL=egl` and install the EGL runtime instead. The optional TensorFlow/Ray
training extras were not tested.

## AssettoCorsaGym (car)

The project-supported host is the Windows x86 RTX 4090 desktop. Use the original
Assetto Corsa (2014), **not Competizione**. No AC installation was attempted on
this Linux VM.

1. Install Visual Studio C++ Build Tools and compile/copy the `sensor_par`
   plugin into Assetto Corsa's `apps/python` directory.
2. Install and select vJoy; install Content Manager plus Custom Shaders Patch
   so the plugin can restart the car.
3. Create the upstream Python 3.9.13 conda environment and use its pinned
   setuptools, Cython, wheel, pip, PyTorch 1.12.1, and CUDA 11.6 versions.
4. Download each track's occupancy-grid pickle from
   `dasgringuen/assettoCorsaGym` on Hugging Face and place it under
   `AssettoCorsaConfigs/tracks`.
5. Enable the `sensor_par` UI module, load the vJoy controls, start a Hotlap,
   and run `test_gym.ipynb`. Check `Documents/Assetto Corsa/logs/py_log.txt`
   when the plugin does not start.

Assetto Corsa embeds Python 3.3.5. Upstream screen capture starts a separate
Python 3.9+ process and requires `mss`, `pygetwindow`, and OpenCV. Upstream also
documents a Proton/Linux path, but that is not the project's supported setup
and was not tested for this ticket.

## Known failures and sharp edges

- Python 3.13+ cannot install flyvis 1.2.0 and pip reports
  `ERROR: No matching distribution found for flyvis==1.2.0`.
- `flyvis` from PyPI pulls platform-specific PyTorch/CUDA wheels. Do not assume
  those resolve on aarch64; validate the NGC Spark image first.
- The pretrained checkpoint is a separate download. The smoke script skips
  clearly when `FLYVIS_ROOT_DIR/results/flow/0000/000` is missing.
- `flybody` is not published on PyPI; install it from its Git commit. Headless
  `dm_control` import failed here until OSMesa was installed and selected.
- flybody's TensorFlow extension pins TensorFlow 2.8/CUDA 11-era packages and is
  not suitable for modern Python or ARM without a dedicated environment.
- AssettoCorsaGym needs its compiled plugin, vJoy, exact car/track selection,
  and separately downloaded occupancy files. A healthy Python client alone
  does not prove the simulator bridge works.
- AC runs in real time. Behavior-clone from the human dataset before RL
  fine-tuning; do not plan around simulator fast-forward.
