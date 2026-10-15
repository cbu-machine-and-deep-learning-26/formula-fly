# Assetto Corsa bridge (GH-24)

The fly stack's `ControlVector(steer, throttle, brake)` drives the Formula car
(`dallara_f317`) at Silverstone (`ks_silverstone-gp`) in the original Assetto Corsa
(2014) through AssettoCorsaGym. Two processes talk over TCP:

- `scripts/ac_fly.py` (fly side) serves a `Driver`. It listens on port 2350, checks
  every frame against the driver's `frame_shape` and never resizes it, and returns
  clipped controls. It runs anywhere: the Windows box, a Mac or a DGX Spark.
- `scripts/ac_car.py` (car side) runs next to AC on the Windows box, in AssettoCorsaGym's
  Python 3.9 env. Each tick it sends the 96x96 RGB capture, waits up to `--deadline-ms`
  for the answer, applies a control, and steps.

The car side never blocks AC. If no answer arrives by the deadline, the tick repeats the
last control and counts a miss. Only one frame is in flight at a time, so a slow fly drops
frames instead of building a queue.

Wire format: a `!I?HHH` header (seq, is_reset, height, width, channels) followed by the raw
frame, and a `!Idddd` reply (seq, steer, throttle, brake, compute_ms). See
`fly_driver/bridge/wire.py`.

## One-time setup on the Windows box

Do the AssettoCorsaGym install from [running-the-stacks.md](running-the-stacks.md#assettocorsagym-car)
first, including vJoy, the `sensor_par` plugin, the `p309` conda env and the Silverstone
occupancy grid. Then:

1. In `<AC>/apps/python/sensor_par/config.py`, set:

   ```python
   self.ego_sampling_freq = 50        # one ego tick per AC tick, flyvis's 50 Hz
   self.screen_capture_enable = True
   self.screen_capture_freq = 50
   self.final_image_width = 96
   self.final_image_height = 96
   self.color_mode = "bgr"
   ```

   `scripts/ac_car.py` sets the matching fields in `config.yml` itself, and upstream
   asserts that both sides agree on the first reset.
2. Set AC's video resolution to 640x480 windowed, as upstream's screen-capture notes ask.
3. Install upstream's screen-capture dependencies into `p309`:
   `conda activate p309` and then `pip install pygetwindow opencv-python`. The car side
   imports `fly_driver` from the checkout path, so you do not need `pip install -e` (the
   package requires Python 3.10 or newer, while the car-side modules also run on 3.9).
4. In AC, choose `dallara_f317` at `ks_silverstone-gp`, start a Hotlap and enable the
   `sensor_par` app.

## Same host (Windows box only)

Terminal 1 (fly env, `.venv` from RUNNING.md, plus torch/flyvis for a trained policy):

```powershell
.venv\Scripts\python.exe scripts\ac_fly.py --host 127.0.0.1 --throttle 0.3
```

Terminal 2 (`conda activate p309`):

```powershell
python scripts\ac_car.py --gym-root C:\path\to\assetto_corsa_gym --fly-host 127.0.0.1 --seconds 30
```

`--throttle 0.3` is the scripted driver. It holds one control and proves the bridge, but it
cannot get round Silverstone. For a trained policy, replace it with
`--checkpoint <run_dir>\policy_final.pt`. The checkpoint must come from a frozen-eye run,
and the brain is not wired in yet (GH-61).

## Split host (fly on a Mac or Spark, car on Windows)

On the fly machine:

```bash
python scripts/ac_fly.py --host 0.0.0.0 --checkpoint <run_dir>/policy_final.pt
```

On the Windows box:

```powershell
python scripts\ac_car.py --gym-root C:\path\to\assetto_corsa_gym --fly-host <fly-ip>
```

Allow inbound TCP 2350 on the fly machine's firewall. Use wired Ethernet if you can.

## What to measure on the Windows box

At the end, `ac_car.py` prints `ticks`, `misses` and round-trip p50/p95/p99/max. It also
writes `ac_bridge_timing.csv` with one row per tick (`rtt_ms`, `compute_ms`, `missed`).
`rtt_ms - compute_ms` is the network and serialisation time.

1. **Tick rate.** Confirm that AC's ego server actually delivers 50 Hz with screen capture
   on. `--seconds 30` should give about 1500 ticks. If capture cannot keep up, the frame
   rate is the problem to solve, because flyvis is fixed at 50 Hz (`FRAME_RATE_HZ`).
2. **Misses.** With the scripted driver on localhost, expect 0. Then repeat with the real
   checkpoint, and again split-host. Any misses mean that the fly's p99 does not fit the
   deadline.
3. **Deadline.** The default of 15 ms leaves 5 ms of the 20 ms tick for AC's step. If
   `ticks` falls short of `seconds x 50` while `misses` is 0, lower `--deadline-ms`.
4. **Steer sign.** Run `ac_fly.py --steer 0.5 --throttle 0.3`. The car must turn right
   (`ControlVector` uses +1 for full right). If it turns left, flip the sign in
   `to_ac_action` (`fly_driver/bridge/assetto.py`).
5. **Frame content.** With `screen_capture_save_to_disk = True` in the plugin config,
   check that the 96x96 captures show the cockpit view and not a menu. The 4:3 window is
   squashed to 1:1, so the hex resampler sees a horizontally compressed field of view.
