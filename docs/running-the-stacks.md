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

`scripts/flyvis_eye_live.py` is a viewer, not a training component. Its panels
split into two kinds. **Camera-derived** (pixels, no neurons): `camera`, the
96 × 96 frame the eye receives, and optionally `retina` (`--show-retina`), the
721-column hex-resampled luminance `HexResampler` feeds the network.
**Neural activity** (read from the network after every step, relative to the
resting state measured at reset): `photoreceptors R1-R6`, the first neural
stage and what the brain actually receives; `motion percept (T4/T5)`, the
eight T4/T5 direction channels fused into one hex image where hue is the
direction of the per-column motion vector (right red, up yellow-green, left
cyan, down violet; a hue wheel sits in the corner) and brightness its strength,
so a still scene is dark; the individual `T4a-d`/`T5a-d` maps from
`FlyvisEye.encode`; and any `--show R1,L1,Mi1,Tm3` cell type (all but
`Lawf1`/`Lawf2`). A direction meter (left/right/up/down from the T4/T5 a/b/c/d
subtypes) sits under the neural panels. Wave a hand across the camera and the
bar for that direction jumps. The overlay shows display fps, per-frame eye
latency, and the camera rate next to "eye stepped at 50 Hz" because every
frame, however fast it arrives, is one 20 ms step of the optic lobe.
`--hide-t5` drops the T5 row.

The layout is HiDPI-safe: a constrained-layout grid, fonts that scale with the
figure width (recomputed on resize), and a figure fitted to a 1440 × 900
logical screen. `import flyvis` restyles matplotlib for 300 dpi paper figures;
the script resets those keys, which is what made fonts swallow the panels on a
Retina Mac. `--figsize W,H`, `--scale`, and `--dpi` override the defaults.

```bash
# in the flyvis venv (Python 3.10-3.12, the project floor); uv is fastest, pip works too
uv pip install -r requirements-flyvis.txt     # adds opencv-python for the camera
export FLYVIS_ROOT_DIR="$HOME/.cache/flyvis"  # where `flyvis download-pretrained` put results/
python scripts/flyvis_eye_live.py             # webcam 0; falls back to --source synthetic
python scripts/flyvis_eye_live.py --source synthetic   # bright bar sweeping l/r/u/d, 2 s each
```

Keys: space pauses/resumes, `r` resets the eye state (1 s grey warm-up),
`q`/Esc quits. Options: `--camera-index`, `--fps-cap` (default 50),
`--show TYPE[,TYPE...]`, `--hide-t5`, `--show-retina`, `--figsize W,H`,
`--scale`, `--dpi`, `--meter-statistic q95|mean`, `--color-limit`,
`--save-dir DIR --save-every N` for PNG snapshots. macOS: grant camera permission to the terminal app you run
it from (System Settings → Privacy & Security → Camera); the first run prompts.
Windows and Linux work the same way through OpenCV.

Headless check, also what `tests/scripts/test_flyvis_eye_live.py` runs:

```bash
python scripts/flyvis_eye_live.py --source synthetic --frames 200 --no-display
```

It prints the meter and the mean motion-percept vector (angle as on the hue
wheel, 0° right / 90° up) every 10 frames, the mean display fps, eye latency
(median/p95/max against the 20 ms budget), and how often the winning meter
direction agreed with the bar's direction. The script exits `0` with `SKIP:`
when flyvis, the checkpoint, or (for the webcam) OpenCV is missing, so base CI
does not need any of them. The hex map drawing is shared with
`scripts/flyvis_eye_demo.py` via `fly_driver.analysis.hex_plots`.

## Shiu whole-brain model (the brain)

A leaky integrate-and-fire network over the whole FlyWire connectome, from Shiu et
al. It is the base for the whole-brain driver (#23) and the lesion map (#30).

Upstream is [philshiu/Drosophila_brain_model](https://github.com/philshiu/Drosophila_brain_model).
It ships the connectivity for FlyWire **v630** (the paper's) and **v783** (current
public) as parquet, plus `model.py` and two notebooks. `fly_driver.brains` wraps it:
`connectome.py` loads and subsets the data, `shiu.py` builds the published model in
Brian2, and `torch_lif.py` is the PyTorch implementation #23 asked us to compare
against.

```bash
# its own environment: brian2 is not part of the base install
python3.11 -m venv .venv-brain
.venv-brain/bin/python -m pip install brian2 "numpy<2" "pandas<3" pyarrow torch
.venv-brain/bin/python -m pip install -e .

# the data is a 370 MB download and is NOT in this repository
git clone --depth 1 https://github.com/philshiu/Drosophila_brain_model.git ~/flywire
export FLY_CONNECTOME_DIR="$HOME/flywire"

.venv-brain/bin/python scripts/brain_throughput.py
```

`v630` is 127,400 neurons and 14,687,178 synapses. The script exits `0` with a
`SKIP:` line when the data or the simulators are missing, so base CI needs neither.

### Check the code generation target before quoting a Brian2 number

Brian2's `codegen.target` defaults to `auto`: it compiles through Cython when a C++
compiler is present and falls back to plain numpy when one is not, with a warning and
nothing else. **The fallback is the number you will accidentally report.** The
benchmark prints the live target, and `fly_driver.brains.shiu.codegen_target()`
returns it. Every figure below is on the **numpy** target, so every figure below is a
floor — Brian2 can only go faster than this.

### What it costs

One environment step is 20 ms of simulated time at 50 Hz. The eye takes 7.9 ms of
that and the practice track 2.2 ms, leaving the brain about **10 ms of wall clock to
simulate 20 ms of biology**.

**Measure it the way the loop will drive it.** A brain between an eye and a policy
cannot be handed a second of biology and left alone: it advances one frame, hands its
activity over, and is advanced again. So there are two different numbers, and mixing
them up is the easiest mistake here:

| neurons | Brian2 batched | Brian2 **stepped** | PyTorch batched | PyTorch **stepped** |
|---|---|---|---|---|
| 1,000 | 3.9 ms | 35.9 ms | 4.6 ms | **5.5 ms** |
| 3,000 | 4.9 ms | 54.5 ms | 11.2 ms | **12.2 ms** |
| 5,000 | 5.8 ms | 56.2 ms | 21.7 ms | 22.6 ms |
| 10,000 | 7.8 ms | 58.6 ms | 76.3 ms | 77.7 ms |
| 127,400 (whole brain) | 101.5 ms | — | 10,768 ms | — |

*Stepped* is `measure_in_loop()`; *batched* is `shiu.measure()` / `torch_lif.measure()`,
which ask for 100 ms in one call and divide. Windows box, Python 3.11, brian2 2.9.0,
numpy target, CPU torch, 1% of neurons driven at 150 Hz, `dt = 0.5 ms`. Firing rates
stayed between 1.0 and 4.9 Hz per neuron and the two backends agreed on them, so
nothing above is fast because it stopped spiking.

Three things fall out, and none of them is the headline number:

**Brian2 charges a fixed 33-52 ms per `run()` call, independent of network size.** It
re-prepares the network every call. Batched over 1,000 ms that is invisible; stepped
20 ms at a time it is the entire cost. At 3,000 neurons: 4.5 ms per frame batched,
54.7 ms stepped. PyTorch charges nothing per call, because a step is tensor
operations and there is nothing to prepare.

**The two backends have opposite scaling.** Brian2 propagates from the neurons that
actually spiked, so its cost tracks *activity*: driving 0%, 1% and 10% of a
10,000-neuron network costs 17.1, 23.7 and 27.1 ms. The PyTorch version multiplies the
whole sparse weight matrix every step whether anything fired or not, so its cost tracks
*size* and is flat against activity: 75.7, 74.4, 78.0 ms. A fly fires at a few Hz, so
about one neuron in a thousand spikes per step — which is why Brian2 wins by 100x on
the whole brain and loses by 10x on a small one driven frame by frame.

**`dt` is a bigger lever than either backend**, since work is proportional to steps per
frame: 0.1 ms is 200 steps per frame, 0.5 ms is 40. On the whole brain that is 257.3 ms
against 101.5 ms. It is not free — the synaptic delay is 1.8 ms and the refractory
period 2.2 ms — so measured rates hold from 0.1 to 0.5 ms and drift upward by 1.0 ms
(+24% on the whole brain). **0.5 ms is the floor**, and 1.0 ms is not defensible.

### The decision

**PyTorch in the loop. Brian2 offline. The in-loop network capped at roughly
1,000-2,000 neurons at `dt = 0.5 ms`.**

Nothing fits the 10 ms budget on Brian2 when stepped, at any size, because the
per-call setup dominates. PyTorch fits at 1,000 neurons (5.5 ms) and is close at 3,000
(12.2 ms) — so the central complex is reachable and the descending neurons with it,
but not with room to spare on this uncompiled box.

Brian2 stays the backend for everything that is not in the loop: the lesion sweep
(#30), the parameter searches, anything that can be handed a second of biology at a
time. It is 3.9-7.8 ms per frame batched across the same sizes and it is the published
model, so offline results stay directly comparable with the paper.

Three honest limits:

- Every Brian2 figure is on the **uncompiled numpy target**, because this box has no
  C++ compiler. The compiled target would speed up the per-step work — but the thing
  that disqualifies Brian2 in the loop is per-*call* setup, not per-step work, so it
  probably does not move the decision. **Untested, and it is the one thing that could.**
  The Linux CI runner has a compiler and could settle it for free.
- The PyTorch version is dense-in-time by construction. An event-driven one would
  scale like Brian2 and win everywhere — but it would be reimplementing Brian2's
  scheduler to catch up with Brian2, and that is a real project, not a spike.
- CPU only. A GPU sparse product would help the large batched case and not the small
  in-loop one, where per-step kernel launches dominate.

### Cell types are a separate download

The upstream repository ships `root_id`s and a completeness flag, and nothing else —
the only named set is `sez_neurons.pickle`, for the paper's own subesophageal-zone
work. Naming the central complex and the descending neurons needs FlyWire's
annotation table from [Codex](https://codex.flywire.ai/), which is not bundled here.
Until that lands, `load_subnetwork(num_neurons=N)` takes a contiguous slice, which
measures the budget just as well: the cost of a timestep depends on how many neurons
and synapses are integrated, not which ones.

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

- The Shiu model's pins (Python 3.10, `brian2==2.5.1`) do not install on Python 3.11:
  there is no cp311 wheel for 2.5.1, so pip builds from source, which needs both
  `setuptools` and a C++ compiler. brian2 2.9.0 installs cleanly and is what the
  numbers above used.
- brian2 2.9.0 does not work with numpy 2.x — it calls `np.ndarray.ptp`, removed in
  NumPy 2.0, and fails at `import brian2`. Pin `numpy<2`.
- Brian2 silently downgrades from Cython to numpy without a C++ compiler. Check
  `codegen_target()` before believing a timing.
- Brian2 raises rather than running when a `Synapses` object was never connected, so a
  deliberately disconnected control network needs the object left out entirely.
- `unless refractory` pauses Brian2's differential equations but does **not** gate the
  synaptic pathway: `on_pre` keeps delivering during the refractory period. Any
  reimplementation that holds arriving input as well will silently drop most of the
  synapses in a busy network.
- The FlyWire connectivity is 370 MB and must never be committed. `.gitignore` covers
  the file names; point `FLY_CONNECTOME_DIR` at a clone outside the tree.
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
