# Steering chatter (GH-104)

What the best checkpoint's steering does, what that costs, and how to measure it again. The full tables and the decision are on issue #104.

## What was found

Measured on `CSDS_PC_2`'s best checkpoint (1,359 m, `lab_20261103_g1` seed 1, update 1070), with noise off.

- **The policy chatters.** It flips 5.18 times a second, with mean |steer| 0.84 and full lock on 68% of frames. Noise on raises the flips to 8.12 and leaves the size alone.
- **It isn't frame-to-frame jitter.** The raw steering is smooth between frames but far too big (median 1.7, range 1), so the clip turns it into lock-to-lock steering near 2 Hz.
- **It sets a speed ceiling.** The recorded steering added to a good scripted aim crashes 12 of 12 at 155-174 km/h on the longest straights. Without it that car reaches 288-317 km/h.
- **It isn't what stops the car at Vale.** The policy speeds up into T10 and doesn't brake. With a speed governor at 60% of the profile, its own steering gets through 14 of 20 times, against 0 of 20 on its own pedals.
- **It can't be removed at run time.** Given smooth steering, the policy's pedals stall the car (37 of 41 drives). A fix has to be trained in.

## The steering dead zone

**On the default car, small steering inputs barely move the wheels.** Leaving centre, the kingpin is held still until the input is large enough: about 14% at 108 km/h, 20% at 200 km/h and 30% at 250 km/h. Coming back from a larger angle, the wheels follow accurately. Every lab run so far has trained on this car. The cause looks like the wide tyre's contact points scrubbing when it turns in place, but that isn't proven.

`steer_gain: 192000, steer_damping_nms: 800` under `env.params.car` removes it, and the car stays stable under violent inputs. The default car doesn't change, so lap times on the two cars aren't comparable. `tests/test_variable_controls.py` checks 101 levels of each control through the pipeline and records the dead zone.

## Training with variable controls

`fly_driver/configs/train_lab_variable.yaml` uses the stiff steering and starts from the best checkpoint with steering x 0.3 and pedals x 0.5 written into its weights. Make that checkpoint first:

```
python scripts/rescale_checkpoint.py --source weights/best/policy_best.pt --target weights/variable_start/policy_best.pt --steer 0.3 --pedals 0.5
```

**It hasn't been shown to work.** Two 200k-step tests, 3 seeds each, on the same 81 starts with noise off:

| | Stalled | Mean distance | Mean speed | Full lock at the wheels | Flips/s at the wheels |
|---|---|---|---|---|---|
| Baseline: default car | 7 | 1,026 m | 108 km/h | 53% | 4.92 |
| Start: stiff car, scaled | 5 | 862 m | 123 km/h | 19% | 4.83 |
| Trained, `output_range_coef: 0.1` | 5, 60, 26 | 714, 281, 778 m | 127, 78, 108 km/h | 14, 15, 11% | 4.76, 4.63, 5.00 |
| Trained, recipe as it is now | 13, 5, 4 | 631, 636, 528 m | 123, 139, 142 km/h | 19, 23, 24% | 4.92, 4.59, 4.54 |

The penalty is off in the recipe because it stalled launches from a standstill on two of three seeds. Without it, the throttle drifts back towards full and the car gets faster and crashes sooner. No seed got through T10, and flips haven't dropped. These runs are too short to say what a long run would do.

## Measuring it

Copy `policy_best.pt` and `brain_best.pt` to `weights/best/` first. These read the checkpoint and don't train.

```
# Steering flips, noise off and on, smoothing and gain, from the grid and fixed starts
python scripts/steering_chatter.py --conditions noise_off noise_on smooth_0.2 gain_0.3

# Which channel limits the car: override the steering or the pedals
python scripts/steering_channels.py --conditions own gov_0.6 gov_0.85 vale_85 script_own script_gov_0.85

# What a wobble does to a well-aimed car on a straight (no policy, no flyvis)
python scripts/steering_wobble.py --recorded runs/steering_chatter/noise_off_uniform_19.npz

# Where it comes from: closed loop, frozen frame, and a drive the policy only watches
python scripts/steering_source.py
```

## Reading the numbers

- **Measure with the eye on CPU, which is the default here.** On CPU, a drive repeats bit for bit. On CUDA the eye differs by about 1e-6 between runs, and the closed loop grows that to a full steering lock within about 4 s.
- **Same starts aren't truly paired.** The loop is that sensitive, so compare conditions with their standard errors. Two identical GPU passes of 29 drives averaged 965 m and 850 m.
- **Run-time tests are one-sided.** The policy never trained with smoothing, a lower gain or an override. If one helps, it shows a real cost. If one hurts, that clears nothing.
- **Noise can hide problems.** A stalled launch doesn't show in noisy training drives, because noise opens the throttle anyway. Judge a checkpoint with noise off.

## The steering speed limit

`CarConfig.max_steer_rate_rad_s` limits how fast the steering can move. It's off by default. A scripted driver copes down to 0.8 s lock to lock, but this policy crashes almost at once: 151 m at 0.3 s and 21 m at 0.7 s, against 1,026 m with no limit.

## The steering cost

`LapTimeReward.steer_change_m` charges for how far the steering moved since the last frame. It's off by default, and off leaves no `steering` term, no new CSV column, and the same rewards and physics as before. It shares the kerb cost's cap, so a step that moves forward is still worth at least 0.

`fly_driver/configs/train_lab_steer.yaml` is the trial recipe: `train_lab.yaml` plus `steer_change_m: 0.25` and a shorter run. It isn't what `train start` runs.

**The trial failed, so don't use this recipe.** Over 1.5M steps and 3 seeds, it missed its target on flips and on mean |steer| at every checkpoint. By the final checkpoints the car was faster (95th percentile 194-233 km/h, against 141) but crashed sooner (411-629 m, against 1,026 m), and no seed got through T10. The cost is kept so that result can be reproduced. If chatter gets another try, test a limit on the raw output's size or lower exploration noise, not a higher weight on this cost.
