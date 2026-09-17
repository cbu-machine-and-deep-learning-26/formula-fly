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

**Single interface, agreed by all tracks:** observation (a camera frame) in → control vector (steer, throttle, brake) out. Every stage must be replaceable by its simplest version (e.g., direct-drive skips the body; CarRacing substitutes for Assetto Corsa). No track blocks another.

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
| AssettoCorsaGym (Remonda et al., NeurIPS 2024) | Gym interface to Assetto Corsa: Formula car, laser-scanned tracks, RL baselines, 2.3M steps of human laps incl. a pro esports driver. github.com/dasGringuen/assetto_corsa_gym | The car, the track, the human comparison |

**The gap we fill:** flyvis never closed a loop through behavior. Shiu et al. had no learning. flybody used a conventional visual encoder. AssettoCorsaGym used conventional agents. Nobody has connected them or tested fly wiring as a prior for closed-loop control.

## 6. Locked-in experimental design decisions

These came out of proposal review. Treat as defaults unless explicitly changed.

- **Frozen pretrained flyvis eye is the DEFAULT**, not the fallback. Train only the policy readout on top. Fine-tuning the eye is a separate experimental condition (frozen vs fine-tuned column), affordable now with the GPU cluster.
- **Control conditions for RQ1 (all with matched output size / trainable param count):**
  1. Small CNN eye
  2. Random projection eye
  3. **Degree-matched shuffled connectome** — same neurons, same sparsity, same in/out degrees, scrambled connections. This is the critical control: it isolates the *pattern* of the wiring rather than mere sparsity. Do not skip it.
- **Gymnasium CarRacing is the primary scientific result** (fast, parallelizable, free). Assetto Corsa is the demo and stretch goal.
- **Assetto Corsa training strategy:** behavior-clone from the 2.3M-step human dataset first, then RL fine-tune. Pure online RL is bounded by wall clock (AC runs real time only, no fast-forward). Note the AC benchmark baselines drive mostly from state features, not pixels; a hybrid observation (fly-eye output + limited telemetry) is an acceptable middle ground for the AC demo.
- **Seeds:** minimum 3 per condition ({0, 1, 2}), deterministic eval protocol, record videos of eval episodes.
- **Lesion sweep (RQ3)** is embarrassingly parallel → runs across the Spark cluster; aim to make it the most thorough result in the report (several hundred cell types × multiple eval episodes each).
- Body track: reuse flybody's pretrained wing pattern generators; learn only a low-dimensional mapping from steering signal to controller modulation. Do NOT train wing control from scratch.
- Interface spec becomes typed code + smoke tests BEFORE stages get built.

## 7. Compute and infrastructure

- **4× NVIDIA DGX Spark cluster.** Each: GB10 Grace Blackwell, 128 GB unified memory, **ARM (aarch64), DGX OS**. Use NVIDIA NGC PyTorch containers rather than raw pip — hitting an x86-only binary in week 5 is the failure mode; find out in week 1. Anything x86-only lives on the 4090 box instead. Role: the experiment matrix (eye type × brain condition × seeds in parallel), whole-brain training (unified memory matters here), lesion sweep.
- **RTX 4090 Windows desktop.** Role: Assetto Corsa host (AC is Windows x86 only, runs real time). Also the fastest single-run trainer (memory bandwidth advantage over a Spark for small nets), and its GPU is mostly idle during AC's real-time rollouts, so co-schedule CarRacing training on it.
- Machines communicate over the local network when the fly and the car are on different hosts.
- AssettoCorsaGym setup gotchas (do in week 1, it is fiddly): requires **original Assetto Corsa (2014), NOT Assetto Corsa Competizione**; Windows + Visual Studio C++ build tools to compile the plugin; a Python 3.9 conda env with pinned versions; vJoy virtual controller; separately downloaded track occupancy grid files (HuggingFace `dasgringuen/assettoCorsaGym`). AC is already purchased.
- Development style: multiple agents / Claude Code sessions in parallel. Agents own glue code (env wrappers, config sweeps, plotting, job queue). Numerics get hand-verified tests (see §11).

## 8. Ten-week plan (condensed)

| Week (start) | Milestone |
|---|---|
| 1 (Sep 21) | Every component runs independently; interface spec agreed; containerized stack runs on one Spark; AC + gym installed on 4090 box with a scripted lap |
| 2 (Sep 28) | Three interchangeable eyes (flyvis, CNN, random projection); hex resampler verified (T4/T5 direction selectivity); random policy moves CarRacing car; eval harness (lap time, sample efficiency, robustness) |
| 3 (Oct 5) | Connectome eye completes a CarRacing lap (frozen eye + trained policy); three-seed baselines running |
| 4 (Oct 12) | **Midterm checkpoint due Oct 16.** Baseline comparison across eyes with learning curves + robustness. Floor = direct-drive comparison; fly-body lap is ceiling |
| 5 (Oct 19) | Whole-brain model (central complex + descending neurons, Shiu base) trains; AC bridge live (fly cockpit controls the Formula car) |
| 6 (Oct 26) | Full CarRacing comparison: all conditions × 3 seeds, incl. shuffled-connectome control; robustness eval |
| 7 (Nov 2) | Lesion map draft (cluster sweep); sim-to-biology validation results (RQ4) |
| 8 (Nov 9) | Stretch: AC lap by the fly; human-lap comparison; multi-agent; legs-on-controls or documented fallback |
| 9 (Nov 16) | Report through methods; record video |
| 10 (Nov 23) | Full report draft. Dec 1–14: slides, practice, final report (due Dec 14) |

## 9. Current status (as of Sep 15, 2026)

- Proposal near-final (due Sep 18). No code exists yet.
- Assetto Corsa purchased.
- Hardware confirmed: 4× DGX Spark cluster + 4090 Windows machine.
- Immediate working focus: **eye + brain track (E)** and the minimal track-C harness it needs (CarRacing + PPO baseline + logging).

## 10. Immediate next steps (first Claude Code sessions, in order)

### Phase 0 — repo scaffold (first session)
1. Create repo structure: `fly_driver/` package with `eyes/`, `brains/`, `policies/`, `envs/`, `training/`, `analysis/`, `configs/`, `tests/`.
2. Write the interface as typed code: `Eye.encode(frame) -> features`, `Policy.act(features) -> ControlVector(steer, throttle, brake)`, plus a `DirectDriveAgent` that composes them. Smoke tests for shapes/dtypes/ranges.
3. Config-driven experiments (YAML or Hydra): condition = {eye_type, brain, frozen/finetuned, seed}. Logging to CSV + optional W&B. Deterministic eval protocol + video recording utility.
4. Containerfile based on NGC PyTorch (aarch64-compatible) that installs the full stack; verify it builds and runs on one Spark. Same environment must also run on the 4090 box (x86) — keep the image multi-arch or maintain two lockfiles.

### Phase 1 — the eye
5. Install flyvis, download pretrained checkpoint(s) (the task-optimized ensemble from Lappalainen et al.).
6. Push a single static frame through the optic lobe model end to end. Confirm output shapes and which cell-type readouts we expose as features (start with T4/T5 and downstream motion outputs; make the readout set configurable).
7. Build the **hexagonal resampler**: CarRacing frame (96×96 RGB) → grayscale/green channel → flyvis's 721-column hex lattice input format, with correct spatial layout and temporal handling (flyvis expects sequences; define frame-rate handling explicitly).
8. **Verification test (required):** feed moving-edge / drifting-grating stimuli through the resampler + eye and confirm T4/T5 direction selectivity matches known preferred directions. This test gates everything downstream.
9. Implement the three control eyes with matched output dimension: small CNN, fixed random projection, and the degree-matched shuffled-connectome variant of the flyvis network.

### Phase 2 — first learning
10. PPO (CleanRL-style or SB3) on Gymnasium CarRacing with the frozen flyvis eye + small policy head. Get any completed lap. Then the same for all control eyes, 3 seeds each, on the Sparks.
11. Produce the first results table + learning curves automatically from logs.

### Phase 3 — brain (after Phase 2 works)
12. Integrate central-complex / descending-neuron circuitry between eye and policy using the Shiu et al. LIF model as the base (Brian2 or a PyTorch reimplementation of the LIF dynamics for GPU speed; decide based on throughput measurement, and record the decision here).

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
- AC training doesn't converge → CarRacing is the primary result; AC footage with the best policy is the demo.
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
