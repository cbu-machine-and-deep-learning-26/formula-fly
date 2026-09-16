# FormulaFly — We Taught a Fruit Fly to Drive an F1 Car

CSCI 4220 (Machine and Deep Learning), Fall 2026. Four-person team, ten-week build.

## What this is

A fully simulated "tethered fly rig." A virtual fruit fly watches a race track through a visual system whose wiring is copied from the real fly connectome, processes it through models of its brain circuits, moves a simulated body, and those movements drive a race car. The fly learns to drive with reinforcement learning.

The headline is the demo: a fly driving a Formula car in Assetto Corsa. The science is the comparison. Does biologically copied wiring help, hurt, or not matter compared to unconstrained networks of the same size?

Everything is simulation. No hardware, no animals, no real vehicles.

## Why a connectome, and what "training" it means

A connectome is a wiring diagram, not a trained model. It tells you which neurons connect to which, and how many synapses sit between them. It does not give you weights. So the architecture is fixed and biological, and we train the parts around it:

1. **Input encoding**: how camera pixels become activity in the fly's sensory neurons.
2. **Output decoding**: how motor-neuron activity becomes steer, throttle, and brake.
3. Optionally, **synapse strengths** and neuron time constants.

The research question is whether the fly's wiring is a useful prior for closed-loop control or just a constraint.

## The pipeline

```
Eye (connectome visual system) → Brain (central complex / whole-brain model) → Body (MuJoCo fly) → Car (sim racer)
```

One interface ties everything together: a camera frame goes in, a control vector `(steer, throttle, brake)` comes out. Every stage can be swapped for its simplest version. A direct-drive agent skips the body. Gymnasium CarRacing stands in for Assetto Corsa. No stage blocks another.

## Tools and resources

| Resource | What it is | What we use it for |
|---|---|---|
| [FlyWire](https://codex.flywire.ai) (Dorkenwald et al., Nature 2024) | Complete adult female fruit fly brain connectome. 139,255 neurons, over 50M synapses. | All biologically constrained wiring. |
| [flyvis](https://github.com/TuragaLab/flyvis) (Lappalainen et al., Nature 2024) | Connectome-constrained network of the fly optic lobe. About 45k neurons, 64 cell types, 721 hexagonal columns. Pretrained and validated against real neural recordings. | The eye. We use its pretrained checkpoints frozen by default, and borrow its validation method to check the trained model stays biologically faithful. |
| Shiu et al. (Nature 2024) | Leaky integrate-and-fire simulation of the whole FlyWire brain, built on Brian2. | The base for the whole-brain driver and the lesion study. |
| flybody (Vaxenburg et al., Nature 2025) | MuJoCo physics model of the fruit fly body with articulated wings and legs, plus pretrained locomotion controllers. | The body. We reuse the pretrained wing controllers and learn only a low-dimensional steering modulation. |
| [AssettoCorsaGym](https://github.com/dasGringuen/assetto_corsa_gym) (Remonda et al., NeurIPS 2024) | Gym interface to Assetto Corsa with a Formula car, laser-scanned tracks, RL baselines, and 2.3M steps of human driving data. | The car, the track, and the human comparison for the demo. |
| Gymnasium CarRacing | Lightweight 2D top-down racing environment. | The primary scientific benchmark. Fast, free, and parallelizable. |
| PyTorch, PPO (CleanRL or Stable-Baselines3) | Training stack. | Policy learning on top of each eye. |

None of these projects has been connected before. flyvis never closed a loop through behavior. Shiu et al. had no learning. flybody used a conventional visual encoder. AssettoCorsaGym used conventional agents. That gap is what we fill.

## Research questions

1. **RQ1.** Can a connectome-constrained visual system learn to drive, and does the constraint help, hurt, or not matter versus a matched unconstrained network?
2. **RQ2.** Does a whole-brain connectome model learn to drive with fewer trainable parameters than a blank network?
3. **RQ3.** Which neurons does driving need? Silence one cell type at a time and measure the damage to lap time. This produces a lesion map no living-fly experiment could.
4. **RQ4.** Does driving training keep the model biologically faithful? Show the trained brain published neuroscience stimuli and compare its responses to real recordings.
5. **RQ5 (stretch).** A fly body with legs on the wheel and pedals, and multi-agent wheel-to-wheel racing.

## Experimental design

The frozen pretrained flyvis eye is the default condition. Only the policy readout is trained on top. Fine-tuning the eye is a separate condition.

RQ1 compares the connectome eye against three controls with matched output size and parameter count:

- A small CNN eye.
- A fixed random projection eye.
- A degree-matched shuffled connectome. Same neurons, same sparsity, same in and out degrees, scrambled connections. This isolates the wiring pattern from mere sparsity.

Every condition runs with at least three seeds under a deterministic eval protocol with recorded videos.

For Assetto Corsa, we behavior-clone from the human dataset first and then fine-tune with RL, because the game runs in real time only.

## Compute

- **4× NVIDIA DGX Spark** (ARM, 128 GB unified memory each). Runs the experiment matrix, whole-brain training, and the lesion sweep. Uses NVIDIA NGC PyTorch containers.
- **RTX 4090 Windows desktop.** Hosts Assetto Corsa, which is Windows x86 only, and doubles as the fastest single-run trainer.

## Repository layout (planned)

```
fly_driver/
  eyes/        flyvis eye, CNN, random projection, shuffled connectome
  brains/      central complex / whole-brain LIF models
  policies/    policy heads and the DirectDriveAgent
  envs/        CarRacing and Assetto Corsa wrappers, hex resampler
  training/    PPO, behavior cloning, config-driven sweeps
  analysis/    learning curves, lesion maps, biology validation
  configs/     experiment conditions
  tests/       interface contracts and numerical correctness checks
```

## Status

Proposal stage as of September 2026. No code exists yet. Current focus is the eye and brain track plus the minimal CarRacing and PPO harness it needs.

See `CLAUDE.md` for the full project context, week-by-week plan, and required correctness tests.
