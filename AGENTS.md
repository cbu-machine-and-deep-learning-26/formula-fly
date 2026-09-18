# We Taught a Fruit Fly to Drive an F1 Car — Project Context

CSCI 4220 (Machine and Deep Learning), Fall 2026. Four-person team, ten-week build, proposal dated Sep 18, 2026. This file is the single source of truth for Claude Code sessions. Read it fully before writing code.

## 1. One-paragraph summary

We build a fully simulated "tethered fly rig": a virtual fruit fly sees a race track through a visual system whose wiring is copied from the real fly connectome, processes it through models of its brain circuits, moves its simulated body (wings/legs), and those movements drive a race car. It learns to drive via reinforcement learning. The headline is the demo (a fly driving a Formula car in Assetto Corsa). The science is the comparison: does biologically-copied wiring help, hurt, or not matter versus unconstrained networks of matched size? Entirely simulation. No hardware, animals, or vehicles.

## 2. Background and context

### Why now (relevant recent events)
- **MaleCNS v1.0** (Google Research + HHMI Janelia + Cambridge): complete adult *male* fruit fly CNS connectome, ~166,000 neurons, ~125M synapses, including the ventral nerve cord. Data released June 8, 2026; Cell paper published Sep 3, 2026. Largest brain map by neuron count to date.
- This triggered viral demos in Sept 2026: the full connectome simulated inside Minecraft ("NeuroCraft Fly", ~166,700 neurons, ~25.6M connections), and a Beat Saber demo (developer "lyra bubbles") where a simulated fly brain tracked ~130 motor neuron outputs at ~14ms latency. Important caveat from the developer herself: in the viral clip the motor system was overfit to a pre-recorded movement sequence; live visual-reactive play via RL was still in progress. These demos establish the cultural moment and prove the pipeline concept (game visuals → simulated sensory neurons → simulated motor neurons → game actions), but none of them are rigorous comparisons.
- **We use FlyWire (the adult female brain connectome), not MaleCNS.** Reason: the two best modeling toolchains (flyvis and the Shiu et al. whole-brain model) are both built on FlyWire. FlyWire = 139,255 neurons, >50M synapses (Dorkenwald et al., Nature 2024). MaleCNS is a good intro hook for topicality but our stack is FlyWire.

### What "training a connectome" means (core framing)
A connectome is wiring, not a trained model. The architecture (who connects to whom) is fixed and biological. What gets trained:
1. **Input encoding**: how pixels/frames become activity in sensory neurons.
2. **Output decoding**: how motor-neuron activity becomes control actions (steer, throttle, brake).
3. Optionally, **synapse strengths** (the connectome gives connection existence and count, not learned weights) and neuron time constants.
So this is standard RL / gradient training applied to a network whose topology is biologically constrained. The research question is whether that topology is a useful prior or just a constraint.

## 3. The pipeline and the one interface

```
Eye (connectome visual system) → Brain (central complex / whole-brain model) → Body (MuJoCo fly) → Car (sim racer)
```

**Single interface, agreed by all tracks:** observation (a camera frame) in → control vector (steer, throttle, brake) out. Every stage must be replaceable by its simplest version (e.g., direct-drive skips the body; the MuJoCo practice track substitutes for Assetto Corsa). No track blocks another.

Tracks: **E** (eye and brain), **B** (body and cockpit), **C** (car, training, infrastructure), **V** (validation and analysis). Current working focus for these sessions: **track E, plus the track-C baseline harness it needs.**

## 4. Research questions (the ladder — each rung is a complete result)

1. **RQ1:** Can a connectome-constrained visual system learn to drive a simulated race car, and does the constraint help/hurt/not matter vs an unconstrained network of matched size?
2. **RQ2:** Does a whole-brain connectome model (including compass/steering circuits) learn to drive with fewer trainable parameters than a blank network?
3. **RQ3:** Which neurons does driving need? Silence one cell type at a time, measure lap-time damage → a lesion map no living-fly experiment can produce.
4. **RQ4:** Does driving training keep the model biologically faithful? Show the trained brain published neuroscience stimuli and compare responses to real recordings (flyvis's own validation method).
5. **RQ5 (stretch):** Fly body with legs on wheel/pedals; multi-agent wheel-to-wheel racing.

## 5. Key resources (all public)

| Resource | What it is | What we take from it |
|---|---|---|
| FlyWire (Dorkenwald et al., Nature 2024) | Complete adult female fly brain connectome, 139,255 neurons, >50M synapses. codex.flywire.ai | All constrained wiring |
| flyvis (Lappalainen et al., Nature 2024) | Connectome-constrained network of the optic lobe (~45k neurons, 64 cell types, 721 hexagonal columns). Trains only synapse strengths + time constants; validated against real neural recordings. github.com/TuragaLab/flyvis | The eye, pretrained checkpoints, and the RQ4 validation method |
| Shiu et al. (Nature 2024) | Leaky integrate-and-fire simulation of the entire FlyWire brain (Brian2-based); predicts sensorimotor behavior from wiring alone | Base for the whole-brain driver and lesion study |
| flybody (Vaxenburg et al., Nature 2025) | MuJoCo physics model of the fruit fly body with articulated wings/legs, plus pretrained locomotion controllers (Google DeepMind + Janelia) | The body. Reuse pretrained wing controllers; only learn low-dim modulation |
| AssettoCorsaGym (Remonda et al., NeurIPS 2024) | Gym interface to Assetto Corsa: Formula car, laser-scanned tracks, RL baselines, 2.3M steps of human laps incl. a pro esports driver. github.com/dasGringuen/assetto_corsa_gym | The car, the track, the human comparison (demo) |
| MuJoCo (Google DeepMind, Apache 2.0) | General-purpose rigid-body physics with offscreen camera rendering; prebuilt wheels for x86 and aarch64 | The practice-track simulator, and the same engine flybody runs in |
| TUMFTM racetrack-database (LGPL-3.0) | Public centerlines + track widths for F1 circuits as CSV (`x_m, y_m, w_tr_right_m, w_tr_left_m`). github.com/TUMFTM/racetrack-database | Silverstone's centerline geometry for the practice track |

**The gap we fill:** flyvis never closed a loop through behavior. Shiu et al. had no learning. flybody used a conventional visual encoder. AssettoCorsaGym used conventional agents. Nobody has connected them or tested fly wiring as a prior for closed-loop control.

## 6. Locked-in experimental design decisions

These came out of proposal review. Treat as defaults unless explicitly changed.

- **Frozen pretrained flyvis eye is the DEFAULT**, not the fallback. Train only the policy readout on top. Fine-tuning the eye is a separate experimental condition (frozen vs fine-tuned column), affordable now with the GPU cluster.
- **Control conditions for RQ1 (all with matched output size / trainable param count):**
  1. Small CNN eye
  2. Random projection eye
  3. **Degree-matched shuffled connectome** — same neurons, same sparsity, same in/out degrees, scrambled connections. This is the critical control: it isolates the *pattern* of the wiring rather than mere sparsity. Do not skip it.
- **The MuJoCo practice track is the primary scientific result.** A simple car on a track surface built from Silverstone's centerline, with a first-person camera at the fly's head. Assetto Corsa is the demo and stretch goal. Chosen over Gymnasium CarRacing (which earlier drafts named) for four reasons, in order of weight:
  1. **Egocentric optic flow.** flyvis models motion vision — T4/T5 are elementary motion detectors. CarRacing's camera is top-down, so there is no expansion when accelerating and no rotational flow when turning. Training a fly retina on a bird's-eye view makes RQ1, RQ3, and RQ4 hard to interpret and the hexagonal resampler geometrically meaningless.
  2. **One physics world with the body.** flybody is MuJoCo. Legs-on-the-wheel (RQ5) is only reachable if the fly and the car share a simulator.
  3. **Transfer to the demo.** Same Silverstone geometry, so the practice-track policy is a real initialisation for the Assetto Corsa behaviour-clone-then-fine-tune, and the lap-time comparison against the track record is coherent rather than apples-to-oranges.
  4. **ARM.** MuJoCo ships clean aarch64 wheels for the Spark cluster; CarRacing needs Box2D, which is exactly the x86-only-wheel risk §7 warns about.
  The cost is that we build and maintain it, and that one fixed circuit gives a weaker generalisation story than CarRacing's procedurally generated tracks. Both tools are free and open source; the difference is engineering effort, not licensing.
- **Assetto Corsa training strategy:** behavior-clone from the 2.3M-step human dataset first, then RL fine-tune. Pure online RL is bounded by wall clock (AC runs real time only, no fast-forward). Note the AC benchmark baselines drive mostly from state features, not pixels; a hybrid observation (fly-eye output + limited telemetry) is an acceptable middle ground for the AC demo.
- **Seeds:** minimum 3 per condition ({0, 1, 2}), deterministic eval protocol, record videos of eval episodes.
- **Lesion sweep (RQ3)** is embarrassingly parallel → runs across the Spark cluster; aim to make it the most thorough result in the report (several hundred cell types × multiple eval episodes each).
- Body track: reuse flybody's pretrained wing pattern generators; learn only a low-dimensional mapping from steering signal to controller modulation. Do NOT train wing control from scratch.
- Interface spec becomes typed code + smoke tests BEFORE stages get built.

## 7. Compute and infrastructure

- **4× NVIDIA DGX Spark cluster.** Each: GB10 Grace Blackwell, 128 GB unified memory, **ARM (aarch64), DGX OS**. Use NVIDIA NGC PyTorch containers rather than raw pip — hitting an x86-only binary in week 5 is the failure mode; find out in week 1. Anything x86-only lives on the 4090 box instead. Role: the experiment matrix (eye type × brain condition × seeds in parallel), whole-brain training (unified memory matters here), lesion sweep.
- **RTX 4090 Windows desktop.** Role: Assetto Corsa host (AC is Windows x86 only, runs real time). Also the fastest single-run trainer (memory bandwidth advantage over a Spark for small nets), and its GPU is mostly idle during AC's real-time rollouts, so co-schedule practice-track training on it.
- Machines communicate over the local network when the fly and the car are on different hosts.
- AssettoCorsaGym setup gotchas (do in week 1, it is fiddly): requires **original Assetto Corsa (2014), NOT Assetto Corsa Competizione**; Windows + Visual Studio C++ build tools to compile the plugin; a Python 3.9 conda env with pinned versions; vJoy virtual controller; separately downloaded track occupancy grid files (HuggingFace `dasgringuen/assettoCorsaGym`). AC is already purchased.
- Development style: multiple agents / Claude Code sessions in parallel. Agents own glue code (env wrappers, config sweeps, plotting, job queue). Numerics get hand-verified tests (see §11).

## 8. Ten-week plan (condensed)

| Week (start) | Milestone |
|---|---|
| 1 (Sep 21) | Every component runs independently; interface spec agreed; containerized stack runs on one Spark; AC + gym installed on 4090 box with a scripted lap |
| 2 (Sep 28) | Three interchangeable eyes (flyvis, CNN, random projection); hex resampler verified (T4/T5 direction selectivity); random policy moves the practice-track car; eval harness (lap time, sample efficiency, robustness) |
| 3 (Oct 5) | Connectome eye completes a practice-track lap (frozen eye + trained policy); three-seed baselines running |
| 4 (Oct 12) | **Midterm checkpoint due Oct 16.** Baseline comparison across eyes with learning curves + robustness. Floor = direct-drive comparison; fly-body lap is ceiling |
| 5 (Oct 19) | Whole-brain model (central complex + descending neurons, Shiu base) trains; AC bridge live (fly cockpit controls the Formula car) |
| 6 (Oct 26) | Full practice-track comparison: all conditions × 3 seeds, incl. shuffled-connectome control; robustness eval |
| 7 (Nov 2) | Lesion map draft (cluster sweep); sim-to-biology validation results (RQ4) |
| 8 (Nov 9) | Stretch: AC lap by the fly; human-lap comparison; multi-agent; legs-on-controls or documented fallback |
| 9 (Nov 16) | Report through methods; record video |
| 10 (Nov 23) | Full report draft. Dec 1–14: slides, practice, final report (due Dec 14) |

## 9. Current status (as of Sep 17, 2026)

- Proposal near-final (due Sep 18).
- Repo scaffolding, git-flow templates, and CI skeleton merged to `develop`. The issue backlog (#13–#34) covers all ten weeks.
- **Landed on `develop`:** the hexagonal eye resampler (#13), the frozen flyvis eye as the default visual frontend (#14), and the standalone-stack documentation (#19, `docs/running-the-stacks.md`). The evaluation harness (#18) is in review.
- **In progress:** the MuJoCo practice track (#16) — track, car, head camera, and the env the eye plugs into.
- Assetto Corsa purchased.
- Hardware confirmed: 4× DGX Spark cluster + a Windows machine. **Note:** this file says RTX 4090 in §7, but the Windows box in use reports an RTX 4060 Ti 16 GB — confirm which is correct before sizing any run against it.
- Immediate working focus: the practice-track env (#16), then the **eye + brain track (E)** on top of it.

## 10. Immediate next steps (first Claude Code sessions, in order)

### Phase 0 — repo scaffold (first session)
1. Create repo structure: `fly_driver/` package with `eyes/`, `brains/`, `policies/`, `envs/`, `training/`, `analysis/`, `configs/`, `tests/`.
2. ~~Write the interface as typed code: `Eye.encode(frame) -> features`, `Policy.act(features) -> ControlVector(steer, throttle, brake)`, plus a `DirectDriveAgent` that composes them. Smoke tests for shapes/dtypes/ranges.~~ Done (#16 types, #20 stages): `fly_driver/interface.py` holds `ControlVector`, `validate_frame`, `validate_features` and the `Eye`/`Policy`/`Body`/`Driver` protocols; `fly_driver/drivers.py` holds `DirectDriveAgent` and `EmbodiedDriveAgent` (the fly body slots into the latter).
3. Config-driven experiments (YAML or Hydra): condition = {eye_type, brain, frozen/finetuned, seed}. Logging to CSV + optional W&B. Deterministic eval protocol + video recording utility.
4. Containerfile based on NGC PyTorch (aarch64-compatible) that installs the full stack; verify it builds and runs on one Spark. Same environment must also run on the 4090 box (x86) — keep the image multi-arch or maintain two lockfiles.

### Phase 1 — the eye (steps 5–8 done, #13/#14/#19)
5. ~~Install flyvis, download pretrained checkpoint(s) (the task-optimized ensemble from Lappalainen et al.).~~
6. ~~Push a single static frame through the optic lobe model end to end. Confirm output shapes and which cell-type readouts we expose as features (start with T4/T5 and downstream motion outputs; make the readout set configurable).~~ `FlyvisEye` exposes all 34 output cell types; T4a–d and T5a–d are the default readout, 5,768 features.
7. ~~Build the **hexagonal resampler**: practice-track camera frame (RGB; resolution and FOV are chosen in the resampler ticket and configured on the env, not fixed here) → grayscale/green channel → flyvis's 721-column hex lattice input format, with correct spatial layout and temporal handling (flyvis expects sequences; define frame-rate handling explicitly).~~ Settled at `FRAME_SHAPE = (96, 96, 3)` and `FRAME_RATE_HZ = 50.0` in `fly_driver/interface.py`; the env and the eye both read them from there.
8. ~~**Verification test (required):** feed moving-edge / drifting-grating stimuli through the resampler + eye and confirm T4/T5 direction selectivity matches known preferred directions. This test gates everything downstream.~~ Passing, in `tests/eyes/`.
9. Implement the three control eyes with matched output dimension: small CNN, fixed random projection, and the degree-matched shuffled-connectome variant of the flyvis network.

### Phase 2 — first learning
10. PPO (CleanRL-style or SB3) on the MuJoCo practice track with the frozen flyvis eye + small policy head. Get any completed lap. Then the same for all control eyes, 3 seeds each, on the Sparks.
11. Produce the first results table + learning curves automatically from logs.

### Phase 3 — brain (after Phase 2 works)
12. Integrate central-complex / descending-neuron circuitry between eye and policy using the Shiu et al. LIF model as the base. **Decided (GH-23, Sep 2026): PyTorch on the CPU in the loop, Brian2 offline, CUDA only above ~20,000 neurons.** The in-loop network fits about 10,000 neurons at `dt = 0.5 ms` (9.5 ms against a ~10 ms budget -- the 20 ms frame less the eye's 7.9 ms and the practice track's 2.2 ms), and a central complex plus descending neurons (~4,300) costs about 7 ms, so RQ2 is reachable with a real circuit. Measure *stepped*, one 20 ms frame at a time, because that is how the loop drives it; a batched number understates Brian2 by an order of magnitude. Brian2 charges a fixed 33-52 ms of setup per `run()` regardless of size, so stepped it is 36-59 ms at every size measured; batched it is 5.8-7.8 ms and scales with *activity*, which is why offline work (#30 lesions) stays on it. **CUDA measured on an RTX 4060 Ti and it loses below ~20,000 neurons**: cost is flat at 0.375 ms per step regardless of `dt` or network size, which profiling attributes to 30 kernel launches per step at ~11 us each: 42.8 us of actual GPU work inside 375 us of wall clock, so the card is busy 11% of the time. Its compute is over 3x faster than the CPU's ~142 us per step; it just pays 330 us to be asked. A faster card will not fix that -- launch latency is driver-side -- so the 4090 has the same floor and only wins bigger above the crossover. The two levers that would change it are CUDA graphs (~400 launches per frame collapsed into one replay) and Linux, whose launch path is much cheaper than Windows's WDDM; neither is tried. Also: treat any LIF timing as provisional until the hot loop has been read -- replacing sparse COO with CSR and removing per-step `.item()` calls moved the CPU 9x. Full numbers and method in `docs/running-the-stacks.md`.

## 11. Correctness checks that MUST have explicit tests (agents: do not skip)

These fail silently and look plausible when wrong. Write tests, don't eyeball:
- Hex resampler geometry (orientation, chirality, column ordering) — validated via the T4/T5 direction-selectivity test.
- LIF / network numerical stability (no NaN/exploding activity over long rollouts; spike rates in plausible ranges).
- Reward shaping (log per-term reward components; verify the agent can't farm reward without progressing).
- Interface contract (action ranges/clipping; frame timing; no silent resizing).
- Shuffled-connectome control actually preserves in/out degree distributions (test the shuffle, not just run it).
- Seed determinism of the eval protocol.

## 12. Risks and fallbacks (condensed)

- flyvis/whole-brain too slow in the loop → already mitigated: frozen pretrained eye is the default; further fallback is a hand-built sparse layer from FlyWire T4/T5 + central-complex connectivity.
- Fly body too hard to control → wings-only steering with direct throttle; direct-drive comparison stands as the primary result.
- AC training doesn't converge → the MuJoCo practice track is the primary result; AC footage with the best policy is the demo.
- Practice-track rendering too slow for the full experiment matrix → lower the camera resolution (the eye resamples to 721 hex columns anyway), render every *k*-th physics step, or batch parallel envs. Measure throughput before the training plumbing is built on top, not in week 6.
- Legs-on-controls doesn't converge → tethered cockpit fallback.
- ARM incompatibility on Sparks → that component moves to the 4090 box; discover via week-1 container milestone.

## 13. Glossary (minimal)

- **Connectome:** map of neurons + synaptic connections. Wiring, not weights, not a trained model.
- **Optic lobe:** fly's visual processing region; ~two-thirds of its neurons.
- **T4/T5:** the fly's elementary motion detectors (ON/OFF pathways), direction-selective. Our sanity-check neurons.
- **Central complex:** navigation hub (heading "compass", steering); key to RQ2.
- **Ventral nerve cord (VNC):** fly's spinal-cord analog (in MaleCNS, not FlyWire; flybody handles the body side for us).
- **LIF:** leaky integrate-and-fire neuron model (used by Shiu et al. whole-brain sim).
- **Tethered rig:** real neuroscience setup where a fixed fly steers a virtual world; our whole project is a software version of one.

## 14. Python coding standards

Applies to all Python in this repo, agent-written or human-written. Goal: any team member (or agent) can read another's code without friction, and tooling enforces most of it automatically.

### 14.1 Naming

| Element | Convention | Example |
|---|---|---|
| Variables, functions, methods | `snake_case` | `user_count`, `get_active_users()` |
| Classes, exceptions | `PascalCase` | `OrderProcessor`, `InvalidTokenError` |
| Constants | `UPPER_SNAKE_CASE` | `MAX_RETRIES`, `DEFAULT_TIMEOUT_SEC` |
| Modules, packages | short `lowercase_with_underscores` | `payment_gateway.py`, `utils/` |
| Type variables (generics) | `PascalCase`, often single letter | `T`, `KeyType` |
| "Private" / internal use | leading underscore | `_cache`, `_parse_header()` |
| Name-mangled (rare, subclass-safe) | leading double underscore | `__internal_state` |
| Test files / functions | `test_` prefix | `test_order_processor.py`, `def test_rejects_negative_amount()` |

- **Booleans** read as a yes/no question: `is_valid`, `has_permission`, `can_retry`. Avoid bare `flag`, `check`.
- **Functions are verbs**: `calculate_total()`, `send_email()`. Avoid noun-only names like `total()` for a function that computes something.
- **Collections are plural**: `users`, `order_ids`. A single item shouldn't be named like a collection.
- **No single-letter names** except conventional, short-lived ones: loop counters (`i`, `j`), comprehensions, or math where the domain uses them (`x`, `y`). Everything else gets a real name.
- **Avoid ambiguous abbreviations** (`usr`, `cfg`, `mgr`) unless the abbreviation is the domain standard (`id`, `url`, `db`, `req`, `resp` are fine).
- **Don't encode type in the name** (`user_list`, `str_name`) — type hints already do that job.
- **Match the domain vocabulary** — this project's domain is neuroscience/RL, so prefer `root_id`, `primary_type`, `synapse_count`, `control_vector` etc. over ad hoc renamings.
- **Avoid shadowing builtins**: don't name things `list`, `dict`, `id`, `type`, `input`.

### 14.2 Formatting

**Tooling (non-negotiable baseline):**
- **Formatter:** [Black](https://black.readthedocs.io/) — no manual style debates, run it in CI and pre-commit.
- **Linter:** [Ruff](https://docs.astral.sh/ruff/) (covers flake8, isort, pyupgrade, and more in one tool).
- **Type checking:** [mypy](https://mypy-lang.org/) or Ruff's type-aware rules, run in CI.
- All of the above configured in `pyproject.toml` so settings are versioned, not personal.

**Line length:** 100 characters. Don't fight the formatter with manual line breaks it will undo.

**Imports**, grouped and separated by a blank line, alphabetized within each group (Ruff/isort handles this automatically):
```python
# 1. Standard library
import os
from datetime import datetime

# 2. Third-party
import requests
from pydantic import BaseModel

# 3. Local/first-party
from fly_driver.eyes import FlyvisEye
from fly_driver.policies import ControlVector
```
- No wildcard imports (`from module import *`).
- Prefer absolute imports over relative imports across packages; relative imports (`.` / `..`) are fine within a single package's internal modules.

**Strings:** Double quotes by default (Black enforces this). Use f-strings for interpolation — not `%` formatting or `.format()`.

**Type hints:** Required on all public function signatures (parameters and return type). Encouraged elsewhere.
```python
def get_control_vector(features: torch.Tensor) -> ControlVector:
    ...
```

**Docstrings:** Required on all public modules, classes, and functions. Google style:
```python
def calculate_discount(price: float, percent: float) -> float:
    """Calculate the discounted price.

    Args:
        price: Original price before discount.
        percent: Discount percentage, e.g. 15 for 15%.

    Returns:
        The discounted price, rounded to 2 decimal places.
    """
```

**Blank lines:**
- 2 blank lines between top-level functions/classes.
- 1 blank line between methods inside a class.
- No blank line right after a `def` line before the docstring.

**Comments:** Explain *why*, not *what* — the code already says what. Delete commented-out code instead of leaving it; version control remembers it.

### 14.3 File & project structure

- **Tests mirror source structure**: `fly_driver/eyes/flyvis_eye.py` → `tests/eyes/test_flyvis_eye.py`.
- **One clear responsibility per module.** If a file is doing three unrelated things, split it. As a rough guardrail, a module pushing past ~400 lines is worth a second look.
- **`__init__.py` stays minimal** — re-export the public API of a package if it helps callers, but avoid burying real logic there.
- **No business logic in entrypoint/training-launcher scripts** — they should just wire things together (config → eye → brain → policy → env) and call into `fly_driver/`.
- **Config and secrets** live in environment variables or a config module (see `fly_driver/configs/`), never hardcoded in business logic files.
- **Filenames are `snake_case.py`**, matching the naming rules above — no `CamelCase.py` or `kebab-case.py`.

### 14.4 Enforcement

- Black, Ruff, and mypy run as **pre-commit hooks** so violations are caught before a commit lands.
- The same checks run in **CI** as a required status check — pre-commit hooks can be skipped locally, CI can't.
- Code review flags naming/structure issues the tools can't catch (a misleading name, a module doing too much) — save review time for things that need human judgment.
