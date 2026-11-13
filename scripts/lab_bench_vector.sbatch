#!/bin/bash
# Benchmark the vectorized trainer on a CBU lab RTX 5090 (x86_64 GPU node).
# The login server is aarch64, so the venv is built here, on the GPU node, the first time.
set -euo pipefail
export HOME=/lab/home/$USER
cd "$HOME/formula-fly"
echo "== $(date) on $(hostname) ($(uname -m)), $(nproc) cores"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader

UV_DIR="$HOME/.local/uv-x86"
if [ ! -x "$UV_DIR/uv" ]; then
  curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR="$UV_DIR" INSTALLER_NO_MODIFY_PATH=1 sh
fi
export PATH="$UV_DIR:$PATH"
VENV=.venv-x86
if [ ! -x "$VENV/bin/python" ]; then
  uv venv -p 3.12 "$VENV"
  VIRTUAL_ENV=$VENV uv pip install --index-url https://download.pytorch.org/whl/cu128 torch
  VIRTUAL_ENV=$VENV uv pip install flyvis==1.2.0 -e . pyyaml
fi
PY="$VENV/bin/python"
export FLYVIS_ROOT_DIR="$HOME/formula-fly/.cache/flyvis"

# Pick a headless GL backend MuJoCo can render with (spawned workers inherit it).
for gl in egl osmesa; do
  if MUJOCO_GL=$gl "$PY" -c "import mujoco; m=mujoco.MjModel.from_xml_string('<mujoco/>'); r=mujoco.Renderer(m,32,32); r.render()" 2>/dev/null; then
    export MUJOCO_GL=$gl; break
  fi
done
echo "MUJOCO_GL=${MUJOCO_GL:-none}"
"$PY" -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda, torch.cuda.get_device_name(0))"

"$PY" scripts/bench_vector.py --config fly_driver/configs/train_lab.yaml \
  --num-envs ${NUM_ENVS:-1 4 6 8 12 16} --updates 3
