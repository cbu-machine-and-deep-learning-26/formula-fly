# Running the standalone stacks

These checks keep the eye, body, and car integrations independent.

## flyvis (optic lobe)

Run on Linux for development and on the DGX Sparks for experiments. Use an NGC
PyTorch image on Spark rather than resolving PyTorch wheels directly on ARM.

```bash
python3 -m venv .venv-flyvis
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
