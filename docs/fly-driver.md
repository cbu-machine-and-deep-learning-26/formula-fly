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
| thorax, halteres | `RIG_Cest` (chest) |
| left and right wing | `FLY:WING_L`, `FLY:WING_R`, empty nodes on `RIG_Cest` at the wing hinges |
| abdomen | `RIG_Hips` |
| head, mouthparts | one object named `DRIVER:HELMET` on `RIG_Head` |
| left and right antenna | `FLY:ANTENNA_L`, `FLY:ANTENNA_R`, empty nodes on `RIG_Head` at the antenna hinges |
| front legs: coxa and femur, tibia, tarsus and claw | `RIG_Arm`, `RIG_ForeArm`, `RIG_HAND` |
| middle and hind legs: coxa and femur, tibia, tarsus and claw | `RIG_Leg`, `RIG_Shin`, `RIG_Foot` |

The head gets the helmet's name on purpose. Cars hide `DRIVER:HELMET` in the cockpit camera,
and that keeps the camera from sitting inside the fly's head.

The fly faces forward like a driver. Its head goes on the stock driver's head, where the
helmet is, and it is 1.3 times the driver's hips-to-head long. A fly looks along its body, so
the body points to the front of the car, tilted 30 degrees nose-up, and flybody's neck joint
(`head`, -29 to +17 degrees) bends to -16 degrees to bring the gaze level. The front legs
reach forward to the wheel, the claws 4 to 6 cm from the stock knuckles, and the abdomen goes
back into the bodywork behind the seat. In the SF70H the fly sits between 25 cm and 76 cm off
the road. `--size` and `--body-pitch` on the pose script change the fit, and the script prints
the gaze (0 is level ahead) and the neck angle.

The first two builds laid the body along the driver's reclined hips-to-head line, 30 degrees
further back. That fly looked 141 degrees above level ahead, up and back, past anything its
neck can do, and reclining less put it through the floor before the gaze came down. #88 then
turned the head bone 141 degrees in the game, past a real fly's neck; this pose replaces that.

The IK is damped least squares (`_IK_DAMPING = 1e-2` in `seated_fly.py`). At 1e-4 the many
redundant targets were ill-conditioned enough that the same pose, built on macOS and on
Windows, put the front claws 4 and 17 cm from the knuckles.

The first fit scaled the fly so its thorax-to-head matched the driver's chest-to-head. That
made it 1.65 m long, with the abdomen and wings hanging a metre through the floor. In the game
only its head showed.

The wing and antenna nodes are for #26, which turns them live. Each sits at its flybody
joint's anchor (the wing's yaw joint, the antenna's extend joint) with its local x along
that joint's axis, so turning a node about its x turns the part about its hinge. At rest the
nodes change nothing: in a Blender round trip of the FBX every vertex stays within 0.3 µm of
the GH-25 position. The nodes are not in the `.knh`, so the game keeps their transform.

Every part's shape is stored relative to its bone, with an identity transform. Assetto Corsa
resets each node named in the car's `.knh`, including `DRIVER:HELMET`, to the transform stored
there, and a part with its own offset would have it applied twice. That is what floated the
head above the car in the first build.

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
`outputs/cockpit/parts`, and prints how far each joint landed from its target and the head
pitch that levels the fly's gaze (see "How it works"). Use `--car` for
another car and `--ac-root` for another install.

**2. Build the FBX in Blender** (about a minute):

```bash
blender -b --factory-startup -P scripts/cockpit/build_fly_driver_fbx.py -- --parts outputs/cockpit/parts --out outputs/cockpit/fly_driver
```

This writes `fly_driver.fbx`, one colour texture per material, `fly_driver.blend` for hand
edits, and `fly_driver_preview.png`. It also copies the textures into a `texture` folder,
because ksEditor ignores the paths inside the FBX and looks there. The meshes are decimated to about 36k vertices; use
`--decimate` to keep a different fraction of the faces.

**3. Convert to KN5 in ksEditor.** This is the one step with no command line.

1. Open `sdk/editor/ksEditor.exe` in the Assetto Corsa folder.
2. Choose **File → Open FBX** and pick `outputs/cockpit/fly_driver/fly_driver.fbx`.
3. On the **Materials** tab, check every `FLY_*` material uses `ksPerPixel` with its texture
   in `txDiffuse`. Set `FLY_membrane`, the wings, to alpha blend if you want them see-through.
4. Choose **File → Save KN5 → Car** and save it as `fly_driver.kn5` in the same folder.

## Put it in the game

A car's driver model is `[MODEL] NAME` in the car's `data/driver3d.ini`, which the SF70H keeps
inside its encrypted `data.acd`. The game's log (`Documents\Assetto Corsa\logs\log.txt`)
names it instead: look for `LOADING MODEL content/driver/...`. The SF70H loads `driver` and
its low-detail copy `driver_B`, so install over both:

```bash
python scripts/cockpit/install_fly_driver.py --driver driver --kn5 outputs/cockpit/fly_driver/fly_driver.kn5
python scripts/cockpit/install_fly_driver.py --driver driver_B --kn5 outputs/cockpit/fly_driver/fly_driver.kn5
```

The original is kept as `content/driver/<NAME>.kn5.stock-backup` the first time, and never
overwritten after that. `driver` is Assetto Corsa's default driver, so most cars get the fly too. To put the
stock driver back:

```bash
python scripts/cockpit/install_fly_driver.py --driver driver --restore
python scripts/cockpit/install_fly_driver.py --driver driver_B --restore
```

Steam's "Verify integrity of game files" also restores it.

Check it with the SF70H at Silverstone. A chase or TV camera shows the fly in the seat, and
the cockpit camera shows the front legs on the wheel. Steer to watch the car's steering
animation move them.

## Animate it from the live brain

Issue #26: the fly in the seat moves with what the brain is doing. The loop is direct drive
(frozen flyvis eye, 4,096-neuron central complex, policy head), so there is no MuJoCo body to
copy. The pose comes from the `ControlVector` and the brain's activity instead
(`fly_driver/cockpit/live_pose.py`), goes out as one UDP datagram per frame
(`fly_driver/cockpit/pose_stream.py`), and a Custom Shaders Patch app turns the bones
(`scripts/cockpit/fly_pose/`).

| Pose | From | Bone in the game | Range |
|---|---|---|---|
| front legs on the wheel | `steer` | `RIG_Arm`, `RIG_ForeArm`, `RIG_HAND`, turned by the car's own `steer.ksanim` | ±90° wheel, sent but not applied |
| right hind leg on the throttle | `throttle` | `DRIVER:RIG_Foot_R`, about the car's side axis through `FLY:PEDAL_R` | 0 to 25° |
| left hind leg on the brake | `brake` | `DRIVER:RIG_Foot_L`, about the car's side axis through `FLY:PEDAL_L` | 0 to 25° |
| head yaw | compass ring bump, `30° × sin(heading)` | `DRIVER:RIG_Head`, about the car's up axis | ±30° |
| wings | mean \|activity\| of each half of the 32 descending neurons | `FLY:WING_L`, `FLY:WING_R`, about their x (wing yaw joint) | 0 to 10°, positive spreads the wing out |
| antennae | signed mean of each half | `FLY:ANTENNA_L`, `FLY:ANTENNA_R`, about their x (antenna extend joint) | ±15° |

The heading is the angle of the activation-weighted mean over the 812 compass neurons, cell
`i` at `2πi/812`, counting only positive activity. Successive poses are smoothed
(`alpha = 0.5`, settled in about 0.1 s at 50 Hz).

The wheel angle is not applied because Assetto Corsa already turns the arm bones from the
car's real steering, which is the policy's `steer` after the bridge. The middle legs share
the hind legs' bones (the bone table under "How it works"), so they press with the pedals too.

The foot bones turn about the fly's hind ankles, the `FLY:PEDAL_L` and `FLY:PEDAL_R` empty nodes
the build puts on them (`seated_fly.PEDAL_PIVOTS`), not about the bones' own origins. Those are
the driver's ankles, in the car's nose, 0.5 to 1.1 m from the fly leg parts on the bones, and a
press swung those parts up to 48 cm out of the car. About the fly's ankle the foot tips move
at most 9 cm and the middle legs' at most 14. A KN5 without the nodes falls back to the old
pivot.

The antennae used to be inside `DRIVER:HELMET`, which cars hide in the cockpit camera. On
their own nodes they are not, so the app hides them itself in the cockpit camera.

### How CSP poses the bones

The app runs in CSP's Lua and uses only documented scene calls, from the
[acc-lua-sdk](https://github.com/ac-custom-shaders-patch/acc-lua-sdk) at commit `7cf60f5`
(August 2026):

- `ac.findNodes('carRoot:0'):findNodes('DRIVER:RIG_Head')` finds a driver bone
  (`lib_scene.lua`, `ac.findNodes`);
- `:getTransformationRaw()` returns the bone's local matrix by reference, "if you need to
  move something often, accessing its matrix directly might be the best way"
  (`lib_scene.lua`);
- `mat4x4.rotation(angle, axis)` and `mat4x4:mul` build the turned matrix
  (`common/ac_matrices.d.lua`);
- UDP is LuaSocket, which CSP ships for every script as `require('shared/socket')`
  ([acc-lua-internal README](https://github.com/ac-custom-shaders-patch/acc-lua-internal),
  "lua-shared").

Moving driver bones live from Lua is what CSP itself does: its VR rig turns
`DRIVER:RIG_Arm_L` and the neck every frame (`lua-module/src/vr_rig_base.lua` in
acc-lua-internal). That rig also calls `ac.updateDriverModel(carIndex, true)` so the game
does not overwrite the change. That call is not in the public SDK, and it may also stop the
steering animation, so the app leaves it off (`HOLD_DRIVER_MODEL` in `fly_pose.lua`). No
minimum CSP version is documented for these calls. Install the current CSP release.

### Install

You need Custom Shaders Patch, installed through Content Manager (Settings → Custom Shaders
Patch → Install) or from [acstuff.club/patch](https://acstuff.club/patch/). Then, on the
game box:

```bash
python scripts/cockpit/install_fly_driver.py --app
```

This copies the app to `apps/lua/fly_pose`. `--app --restore` removes it. Install the fly's
KN5 first, as above.

### Run it with driving or training

Wrap whatever drives the car. Nothing else changes:

```python
from fly_driver.cockpit.pose_stream import PoseStream, PoseStreamDriver, PoseStreamEnv

stream = PoseStream(host="127.0.0.1")         # the game box's address if the fly runs elsewhere
driver = PoseStreamDriver(driver, stream)      # any Driver; reads driver.brain if it has one
env = PoseStreamEnv(env, stream, brain=brain)  # or any env with reset/step, e.g. PracticeTrack
```

Each `act` or `step` sends one pose to UDP port 9871. Open **Fly Pose** in the game's app
bar to see poses arriving. With no pose for a second, the bones go back to rest, except the
head, which keeps its fixed pitch so the fly still looks ahead.

### What to check in the game

1. The Fly Pose window counts poses and says "fly bones and hinges found". If it waits for
   the driver model forever, the installed KN5 predates the hinge nodes: rebuild it.
2. Throttle presses the right hind leg forward and brake the left. If they lift instead, set
   `PEDAL_SIGN = -1` in `fly_pose.lua`.
3. A positive head yaw (bump past a quarter of the ring) turns the head to the driver's
   right. If it turns left, set `HEAD_SIGN = -1`.
4. Pressing a pedal moves only the fly's feet, inside the car. If legs swing up out of it,
   the KN5 predates the pedal pivots: rebuild it.
5. The front legs still follow the wheel while the app runs. If the head and feet do not
   move at all, try `HOLD_DRIVER_MODEL = true`, then check the front legs again.
6. The head and antennae stay hidden in the cockpit camera, and show in the chase and TV
   cameras.
7. The wings spread out from the back with descending activity, and the antennae swing
   about their bases. If a wing folds into the body instead, set `HINGE_SIGN = -1`.

## Known limits

- **Rigid parts can open small gaps at the leg joints at large steering angles.** The fly's
  joints don't sit exactly on the human ones, and each part turns about its bone's pivot. If
  that shows, the next step is skinning the parts to the same bones with blended weights.
- **Assetto Corsa does not animate the driver's feet** (Kunos car pipeline guide, "Driver
  animations"). The CSP app above presses them.

## Tests

```bash
pytest tests/cockpit tests/scripts/test_install_fly_driver.py
```

For #26 they cover the hinge nodes (each wing and antenna on its own node at its joint,
x along the joint axis, a positive wing angle spreading the wing, the rest pose unchanged),
the pose's limits, the steer sign, the pedals and their pivots at the fly's hind ankles, a head that follows a
synthetic compass bump, the UDP round trip, one pose per step from each wrapper, and the app
install. For #25 they cover the `.knh` reader and writer on synthetic trees, the fly-to-bone map, the axis
conversion, mesh welding, the manifest and OBJ writer, and install and restore. With flybody's
model files present, they also check that the fly faces forward with a level gaze and its
neck in range, that the pose reaches the knuckles, that every fly body is
mapped and every mesh exported, and that the wings are folded.
