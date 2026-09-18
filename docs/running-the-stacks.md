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
# its own environment: brian2 and torch are not part of the base install.
# torch FIRST, from the CUDA index, or pip hands you the CPU build and every "GPU"
# number after it is silently a CPU number.
python3.11 -m venv .venv-brain
.venv-brain/bin/python -m pip install torch --index-url https://download.pytorch.org/whl/cu128
.venv-brain/bin/python -m pip install -r requirements-brain.txt
.venv-brain/bin/python -m pip install -e .

# always check before trusting a timing
.venv-brain/bin/python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"

# the data is a 370 MB download and is NOT in this repository
git clone --depth 1 https://github.com/philshiu/Drosophila_brain_model.git ~/flywire
export FLY_CONNECTOME_DIR="$HOME/flywire"

.venv-brain/bin/python -m pytest tests/brains -q     # runs on cpu, and on cuda when present
.venv-brain/bin/python scripts/brain_throughput.py --device cpu cuda
```

`v630` is 127,400 neurons and 14,687,178 synapses. The script exits `0` with a `SKIP:`
line when the data or the simulators are missing, so base CI needs neither.

### Which device, and why it is never guessed

`--device cpu cuda` measures **both on the same machine**. That is not a convenience: a
CPU number from one box against a GPU number from another confounds the card with the
whole computer, and the only comparison worth recording is the one where the card is the
single thing that changed.

Asking for `cuda` when CUDA is unavailable **raises**. It does not fall back to the CPU.
A run that silently uses different hardware than it was told to still succeeds and still
prints a plausible number — the same failure Brian2's codegen fallback produces, one
paragraph down. The resolved device and the card's name are printed, and travel in
`ThroughputResult.notes`, so a recorded timing cannot be misattributed later.

Brian2 is CPU-only here on purpose. `brian2cuda` exists, but only in standalone mode,
which compiles an entire run ahead of time and cannot be stepped one frame at a time —
which is how a brain between an eye and a policy is driven. So the GPU work is all on the
torch side and Brian2 stays the offline backend.

### The step loop must not touch the host

Everything in `TorchLIF.step` runs 40 times per environment frame. On CUDA a single
`.item()`, `bool(...)` or `.cpu()` inside that loop drains the pipeline, so 40 steps
become 40 stalls and the card comes back slower than a laptop — while still computing the
right spikes, so nothing else catches it. The spike total therefore accumulates in a
device tensor and is read only when asked for; the spike reset is branch-free rather than
guarded by `if spiked.any()`; and `SpikeStream` stacks a whole frame's spikes on the
device and makes one transfer instead of forty. `tests/brains/test_torch_lif.py` pins this
with `torch.cuda.set_sync_debug_mode("error")`.

That work was done for CUDA and turned out to matter as much on the CPU — see the table
below.

### Check the code generation target before quoting a Brian2 number

Brian2's `codegen.target` defaults to `auto`: it compiles through Cython when a C++
compiler is present and falls back to plain numpy when one is not, with a warning and
nothing else. **The fallback is the number you will accidentally report.** The
benchmark prints the live target, and `fly_driver.brains.shiu.codegen_target()`
returns it. Every figure below is on the **numpy** target, so every figure below is a
floor — Brian2 can only go faster than this.

### What it costs

One environment step is 20 ms of simulated time at 50 Hz. The eye takes 7.9 ms of that
and the practice track 2.2 ms, leaving the brain about **10 ms of wall clock to simulate
20 ms of biology**.

**Measure it the way the loop will drive it.** A brain between an eye and a policy cannot
be handed a second of biology and left alone: it advances one frame, hands its activity
over, and is advanced again. *Stepped* is `measure_in_loop()`; *batched* is
`shiu.measure()` / `torch_lif.measure()`, which ask for the whole run in one call and
divide. Mixing them up is the easiest mistake here, and for Brian2 they differ by an order
of magnitude.

One machine, both devices — 12-core AMD (Zen 4), RTX 4060 Ti, driver 616.56, Windows,
Python 3.11, torch 2.11.0+cu128, brian2 2.9.0 on the numpy target, 1% of neurons driven at
150 Hz, `dt = 0.5 ms`, stepped:

| neurons | synapses | Brian2 | torch **CPU** | torch **CUDA** |
|---|---|---|---|---|
| 1,000 | 1,105 | 35.9 ms | **5.0 ms** | 16.3 ms |
| 3,000 | 7,944 | 53.3 ms | **5.7 ms** | 15.0 ms |
| 5,000 | 19,544 | 56.2 ms | **7.1 ms** | — |
| 10,000 | 86,842 | 58.6 ms | **9.5 ms** | 17.3 ms |
| 20,000 | 370,563 | — | 17.4 ms | **15.8 ms** |
| 50,000 | 2,284,882 | — | 47.6 ms | **14.5 ms** |

CPU figures are the median of three runs; they are stable to about ±0.5 ms at 3,000 and to
0.1 ms at 10,000. Firing rates held at 1.0–2.1 Hz per neuron across every row, so nothing
here is fast because it stopped spiking. **Do not measure the two devices concurrently** —
an early CPU sweep taken while the GPU tests were running read 10.9 ms at 10,000 against a
true 9.5 ms.

### The GPU is launch-bound, and that is the whole story

The CUDA column is flat at ~15–17 ms whatever the network size, which is not how compute
behaves. Sweeping `dt` at a fixed 3,000 neurons says why:

| dt (ms) | steps per frame | CUDA ms/frame | **ms per step** |
|---|---|---|---|
| 0.10 | 200 | 74.3 | 0.372 |
| 0.25 | 80 | 29.8 | 0.373 |
| 0.50 | 40 | 15.0 | 0.375 |
| 1.00 | 20 | 7.5 | 0.375 |
| 2.00 | 10 | 3.9 | 0.390 |

**0.375 ms per step, independent of `dt` and of network size.** Profiling a step at 3,000
neurons says exactly where it goes: **30 kernel launches, 42.8 µs of actual GPU work, 375 µs
of wall clock.** The card is busy **11%** of the time and spends the other 89% waiting to be
told what to do next, at roughly 11 µs of launch latency per kernel.

The GPU is not slow at this. Its 42.8 µs of compute beats the CPU's ~142 µs per step by more
than three times. It just pays 330 µs to get asked. The crossover where the GPU finally beats the CPU is around 20,000
neurons, which is far above anything that goes in the loop.

Two consequences worth being clear about:

- **A faster card will not fix this.** Kernel launch latency is driver- and CPU-side, not
  silicon, so a 4090 has the same floor. It will win bigger above the crossover, where the
  sparse product actually dominates, and it will not move the small-network case.
- **The two things that would fix it** are CUDA graphs — capturing a frame's ~1,200 launches
  into one replay — and Linux, whose launch latency is several times lower than Windows's
  WDDM path. Graphs are measured below and are worth 10x; Linux is untried. If the lab's
  4090s run Linux, that alone is worth re-measuring for.

### CUDA graphs would remove the idle, and by how much is measured

Not implemented, and deliberately so — see *when to build it* below. But the hard parts
were tried on the 4060 Ti so that nobody has to rediscover them, and the result is large
enough to change the decision the day it matters.

A graph records one frame's kernel sequence once and replays it as a **single** submission,
so the ~1,200 launches per frame collapse to one and the launch latency disappears. What is
left is the ~43 µs of real work per step.

```python
model = TorchLIF(subnetwork, device="cuda")
model.run(20.0)                     # warm up first; capture must not be the first run
torch.cuda.synchronize()

graph = torch.cuda.CUDAGraph()
graph.register_generator_state(model.generator)   # see the trap below -- not optional
with torch.cuda.graph(graph):
    for _ in range(40):             # one 20 ms frame at dt = 0.5 ms
        model.step()

graph.replay()                      # each replay is one frame
```

Measured on the 4060 Ti, Poisson drive on, `dt = 0.5 ms`:

| neurons | eager | graphed | speedup |
|---|---|---|---|
| 3,000 | 15.0 ms | **1.49 ms** | 10.1x |
| 10,000 | 12.3 ms | **1.37 ms** | 8.9x |
| 20,000 | 12.5 ms | **1.96 ms** | 6.4x |
| 50,000 | 11.2 ms | **5.58 ms** | 2.0x |
| 100,000 | 21.5 ms | 17.9 ms | 1.2x |

The speedup *shrinking* with size is the point: by 50,000 neurons the card is finally
compute-bound, which is what "the overhead is gone" looks like. The in-loop cap would move
from ~10,000 neurons on the CPU to ~50,000, and a central complex plus descending neurons
(~4,300) would cost about 1.5 ms of the 9.9 ms budget instead of about 7 ms.

#### The trap, which fails silently

A graph replays a **fixed** sequence of kernels. The obvious failure is that it also replays
the same random numbers, so the Poisson drive freezes and the brain receives identical input
every frame forever — at 10x the speed, with plausible firing rates, and nothing else to
show for it.

`graph.register_generator_state(model.generator)` is what prevents that: it tells the graph
to advance the generator's offset across replays. Without it the capture does not even
succeed (`Attempt to increase offset for a CUDA generator not in capture mode`), which is
lucky — but if you reach for the *default* generator to get around that error, it captures
happily and freezes the noise.

Verified rather than assumed, at 3,000 neurons:

- spikes per replayed frame vary: 63, 57, 62, 57, 59, 71, 60, 54
- the same seed reproduces that sequence exactly across separate runs
- a different seed produces a different one

Reproducibility surviving matters: `AGENTS.md` §11 asks for seed determinism in the eval
protocol, so this does not cost determinism to buy speed.

#### When to build it

Any one of these turns it from an optimisation into work worth doing:

- **The brain goes into the training loop** (#23 proper, which #17 unblocks). Training is
  not real-time-bound, so 3.8x more steps per hour is the difference between an overnight
  experiment matrix and a weekend one.
- **The in-loop network grows past ~10,000 neurons.** Then the CPU stops fitting and graphs
  are a requirement rather than a speedup.
- **The Assetto Corsa bridge** (#24), which is hard real-time with no slack.

Until one of those, the CPU fits a 2,000–4,000-neuron circuit at ~5–7 ms of a 9.9 ms budget,
the eye's 7.9 ms is the larger cost in the loop anyway, and CUDA graphs are complexity —
static shapes, a capture path, generator state — bought against a bottleneck that is not yet
the bottleneck.

What building it would involve: a capture path on :class:`TorchLIF` behind a flag, the
generator registration above, and a test that graphed and eager produce identical spikes for
the same seed. That last one is the important one, and it does not exist yet — everything
above shows the graphed path is *fast*, *varying* and *reproducible*, not that it computes
the same thing as the eager path.

### The decision

**PyTorch on the CPU in the loop. Brian2 offline. CUDA only above ~20,000 neurons, which
is not the in-loop regime.**

The in-loop network fits about **10,000 neurons at `dt = 0.5 ms`** (9.5 ms against a 9.9 ms
budget), and a central complex plus descending neurons — roughly 4,300 — costs about 7 ms
with room to spare. That is the finding that makes RQ2 reachable with a real circuit.

Nothing fits on Brian2 when stepped, at any size, because its fixed 33–52 ms of per-`run()`
setup dominates. Brian2 keeps everything that is not in the loop: the lesion sweep (#30),
parameter searches, anything that can be handed a second of biology at a time. It is
5.8–7.8 ms per frame batched at these sizes and it is the published model, so offline
results stay directly comparable with the paper.

Three honest limits:

- Every Brian2 figure is on the **uncompiled numpy target**, because this box has no C++
  compiler. The compiled target would speed up per-step work, but what disqualifies Brian2
  in the loop is per-*call* setup, so it probably does not move the decision. Untested.
- Brian2 integrates in float64 and the torch LIF in float32. That is a real difference
  between "the same model", and it is why the cross-backend checks use tolerances and
  firing rates rather than exact spike trains.
- The torch version is dense-in-time by construction. An event-driven one would scale like
  Brian2 and win everywhere — but it would be reimplementing Brian2's scheduler to catch up
  with Brian2.

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

- A CUDA graph replays a fixed kernel sequence, so it will happily replay the same random
  numbers too — freezing the Poisson drive while firing rates still look plausible. Register
  the generator with the graph; do not work around the capture error by switching to the
  default generator.
- CUDA is **slower than the CPU** for anything under ~20,000 neurons here, and it is not
  the card's fault: at 0.375 ms per step it is pure kernel-launch latency. Do not read a
  GPU row as "the GPU is bad at this" without checking the per-step figure first.
- `torch.cuda.set_sync_debug_mode("error")` passes through `TorchLIF.step`, so cuSPARSE
  does not synchronise internally for this workload. That was not obvious in advance.
- A test that calls `.numpy()` on a tensor from `TorchLIF.membrane_mv` fails on CUDA and
  passes on the CPU. The property hands back the live device tensor deliberately — copying
  it would cost the synchronisation the backend exists to avoid — so tests ask with `.cpu()`.
- `pip install torch` gives you the **CPU build**. The CUDA wheel needs
  `--index-url https://download.pytorch.org/whl/cu128` (or cu126/cu130), and installing it
  second will not fix an environment that already has the CPU build — uninstall first.
  Check `torch.cuda.is_available()` before believing any GPU timing.
- A `.item()`, `bool(...)` or `.cpu()` inside a per-step loop is close to free on the CPU
  and is a pipeline stall on CUDA. It also cost an order of magnitude on the CPU here,
  which was not expected.
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
