# Steering chatter (GH-104)

What the best checkpoint's steering does, what that costs, and how to measure it again. The full tables and the decision are on issue #104.

## What was found

Measured on `CSDS_PC_2`'s best checkpoint (1,359 m, `lab_20261103_g1` seed 1, update 1070), with noise off.

- **The policy chatters.** It flips 5.18 times a second, with mean |steer| 0.84 and full lock on 68% of frames. Noise on raises the flips to 8.12 and leaves the size alone.
- **It isn't frame-to-frame jitter.** The raw steering is smooth between frames but far too big (median 1.7, range 1), so the clip turns it into lock-to-lock steering near 2 Hz.
- **It sets a speed ceiling.** The recorded steering added to a good scripted aim crashes 12 of 12 at 155-174 km/h on the longest straights. Without it that car reaches 288-317 km/h.
- **It isn't what stops the car at Vale.** The policy speeds up into T10 and doesn't brake. With a speed governor at 60% of the profile, its own steering gets through 14 of 20 times, against 0 of 20 on its own pedals.
- **It can't be removed at run time.** Given smooth steering, the policy's pedals stall the car (37 of 41 drives). A fix has to be trained in.

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

## The steering cost

`LapTimeReward.steer_change_m` charges for how far the steering moved since the last frame. It's off by default, and off leaves no `steering` term, no new CSV column, and the same rewards and physics as before. It shares the kerb cost's cap, so a step that moves forward is still worth at least 0.

`fly_driver/configs/train_lab_steer.yaml` is the trial recipe: `train_lab.yaml` plus `steer_change_m: 0.25` and a shorter run. It isn't what `train start` runs.
