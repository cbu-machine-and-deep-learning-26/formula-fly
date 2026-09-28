# The fly as the Assetto Corsa driver

Issue #25: replace the driver model in Assetto Corsa with the fly, the same flybody model that
produces the controls. This page says how the model is built, how to rebuild it, and how to
put it in the game and take it out again.

## How it works

Assetto Corsa seats a car's driver with the car's `driver_base_pos.knh` and moves the arms
with the car's `steer.ksanim` and shift animations. All of them address the stock driver's
bones by name: `DRIVER:RIG_Arm_L`, `DRIVER:RIG_HAND_L` and so on (Kunos car pipeline guide,
`sdk/dev/car_pipeline_2.0rev/AC_Pipeline_PUB_Rev2.0.pdf`, "Driver position and mesh").

The fly driver keeps that skeleton, exactly as the car's `.knh` places it, and hangs the
fly's body parts on it as rigid meshes. That is the same way the stock helmet rides on the
head bone. The car's own animations then move the fly, and nothing in the car's data changes.

| Fly | Driver bone |
|---|---|
| thorax, wings, halteres | `RIG_Cest` (chest) |
| abdomen | `RIG_Hips` |
| head, mouthparts, antennae | one object named `DRIVER:HELMET` on `RIG_Head` |
| front legs: coxa and femur, tibia, tarsus and claw | `RIG_Arm`, `RIG_ForeArm`, `RIG_HAND` |
| middle and hind legs: coxa and femur, tibia, tarsus and claw | `RIG_Leg`, `RIG_Shin`, `RIG_Foot` |

The head gets the helmet's name on purpose. Cars hide `DRIVER:HELMET` in the cockpit camera,
and that keeps the camera from sitting inside the fly's head.

The fly is scaled so its thorax-to-head distance matches the driver's chest-to-head, which
makes it about 1.65 m long. It is posed in MuJoCo by inverse kinematics against the driver's
joints (`fly_driver/cockpit/seated_fly.py`). The front claws land within about 3 cm of the
stock knuckles, so the front legs hold the wheel. The front knees land within about 10 cm of
the elbows, because a fly's leg is not proportioned like an arm. The other legs point into the
footwell, and the wings are folded along the back.

## What you need

| Tool | Where | Notes |
|---|---|---|
| Assetto Corsa | Steam | The original 2014 game. ksEditor ships with it in `sdk/editor/ksEditor.exe` |
| Blender 4.5 LTS | [blender.org](https://download.blender.org/release/Blender4.5/), portable zip | Run headless, no admin |
| flybody's model files | `git clone https://github.com/TuragaLab/flybody.git ~/.cache/flybody-src` and check out `d015e9bfe441bd90ae431bac24c55cb74bdbce26` | Only the XML and OBJ assets are read; flybody is not imported |
| The project's base `.venv` | [RUNNING.md](../RUNNING.md) | numpy and mujoco |

Everything generated lands in `outputs/cockpit/`, which is git-ignored. Nothing from
Assetto Corsa is copied into the repo; Kunos's car and driver data stay local.

## Build it

**1. Pose and cut the fly** (about 10 s):

```bash
python scripts/cockpit/pose_fly_driver.py --preview outputs/cockpit/pose.png
```

This reads the SF70H's `driver_base_pos.knh` from the Assetto Corsa install and flybody's
`fruitfly.xml`. It writes one OBJ per fly part and `manifest.json` to
`outputs/cockpit/parts`, and prints how far each joint landed from its target. Use `--car` for
another car and `--ac-root` for another install.

**2. Build the FBX in Blender** (about a minute):

```bash
blender -b --factory-startup -P scripts/cockpit/build_fly_driver_fbx.py -- --parts outputs/cockpit/parts --out outputs/cockpit/fly_driver
```

This writes `fly_driver.fbx`, one colour texture per material, `fly_driver.blend` for hand
edits, and `fly_driver_preview.png`. The meshes are decimated to about 36k vertices; use
`--decimate` to keep a different fraction of the faces.

**3. Convert to KN5 in ksEditor.** This is the one step with no command line.

1. Open `sdk/editor/ksEditor.exe` in the Assetto Corsa folder.
2. Choose **File → Load FBX** and pick `outputs/cockpit/fly_driver/fly_driver.fbx`.
3. On the **Materials** tab, check every `FLY_*` material uses `ksPerPixel` with its texture
   in `txDiffuse`. Set `FLY_membrane`, the wings, to alpha blend if you want them see-through.
4. Save it as `fly_driver.kn5` in the same folder.

## Put it in the game

A car's driver model is `[MODEL] NAME` in the car's `data/driver3d.ini`. The SF70H keeps that
file inside its encrypted `data.acd`, so read the name from Content Manager. Then install:

```bash
python scripts/cockpit/install_fly_driver.py --driver <NAME> --kn5 outputs/cockpit/fly_driver/fly_driver.kn5
```

The original is kept as `content/driver/<NAME>.kn5.stock-backup` the first time, and never
overwritten after that. Every car that uses the same driver model gets the fly too. To put the
stock driver back:

```bash
python scripts/cockpit/install_fly_driver.py --driver <NAME> --restore
```

Steam's "Verify integrity of game files" also restores it.

Check it with the SF70H at Silverstone. A chase or TV camera shows the fly in the seat, and
the cockpit camera shows the front legs on the wheel. Steer to watch the car's steering
animation move them.

## Known limits

- **Rigid parts can open small gaps at the leg joints at large steering angles.** The fly's
  joints don't sit exactly on the human ones, and each part turns about its bone's pivot. If
  that shows, the next step is skinning the parts to the same bones with blended weights.
- **Pedals are not animated.** Assetto Corsa does not animate the driver's feet (Kunos car
  pipeline guide, "Driver animations"), so the middle legs stay still.
- **Moving with the physics body is #26,** a live pose from MuJoCo through Custom Shaders
  Patch. It waits for the Assetto Corsa bridge in #24.

## Tests

```bash
pytest tests/cockpit tests/scripts/test_install_fly_driver.py
```

They cover the `.knh` reader and writer on synthetic trees, the fly-to-bone map, the axis
conversion, mesh welding, the manifest and OBJ writer, and install and restore. With flybody's
model files present, they also check that the pose reaches the knuckles, that every fly body is
mapped and every mesh exported, and that the wings are folded.
