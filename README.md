# Formula Fly

CSCI 4220 (Machine and Deep Learning), Fall 2026.

A fully simulated tethered-fly rig: a virtual fruit fly sees a race track
through a visual system whose **wiring is copied from the real fly connectome**,
processes that activity through models of its brain circuits, moves a simulated
body, and those movements drive a race car. The fly learns to drive with
reinforcement learning.

The headline is the demo — a fly driving a Formula car in Assetto Corsa. The
science is the comparison: does biologically copied wiring help, hurt, or not
matter versus unconstrained networks of matched size?

Everything is simulation. No hardware, animals, or real vehicles.

## Pipeline

```
Eye (connectome visual system) → Brain (central complex / whole-brain model)
    → Body (MuJoCo fly) → Car (sim racer)
```

One interface ties the stages together: a camera frame in, a control vector
`(steer, throttle, brake)` out. Every stage can be swapped for a simpler stand-in
(direct-drive skips the body; Gymnasium CarRacing stands in for Assetto Corsa).
No track blocks another.

## Research question

A connectome is a wiring diagram, not a trained model. We keep that topology
fixed and train encoding, decoding, and optionally synapse strengths. The
question is whether fly wiring is a useful prior for closed-loop control or
only a constraint.

Primary scientific benchmark: Gymnasium CarRacing (fast, free, parallelizable).
Assetto Corsa is the demo and stretch goal.

## Stack (public resources)

| Piece | Source |
|---|---|
| Wiring | [FlyWire](https://codex.flywire.ai) adult female brain connectome (Dorkenwald et al., Nature 2024) |
| Eye | [flyvis](https://github.com/TuragaLab/flyvis) optic lobe (Lappalainen et al., Nature 2024) |
| Whole brain | Shiu et al., Nature 2024 (LIF / FlyWire) |
| Body | flybody (Vaxenburg et al., Nature 2025) |
| Car (demo) | [AssettoCorsaGym](https://github.com/dasGringuen/assetto_corsa_gym) (Remonda et al., NeurIPS 2024) |
| Car (science) | Gymnasium CarRacing |
| Learning | PyTorch, PPO |

We use **FlyWire**, not MaleCNS: flyvis and the Shiu whole-brain model are built
on FlyWire.

## Develop

See [CONTRIBUTING.md](CONTRIBUTING.md) for git-flow and branch names. Project
context for agents lives in [CLAUDE.md](CLAUDE.md).
