#!/bin/bash
# ---------------------------------------------------------------------------
# 5-minute preflight. Run this INSIDE an interactive GPU allocation before
# queueing the real job -- it catches the GPU-pinning and mount failures that
# otherwise waste a queue slot and produce an empty .out file.
#
#   cd ~/cmha
#   srun -N 1 --ntasks-per-node=4 --gres=gpu:a100_4g.20gb:1 --mem=16000 \
#        --time=00:20:00 --partition=gpu_faculty --pty bash
#   bash hpc/verify_dgx.sh
# ---------------------------------------------------------------------------
echo "node : $(hostname)"
echo "gpus : ${CUDA_VISIBLE_DEVICES:-NONE}"

export ENROOT_TEMP_PATH="$HOME/.enroot/tmp"
export ENROOT_CACHE_PATH="$HOME/.enroot/cache"
export ENROOT_DATA_PATH="$HOME/.enroot/data"
export ENROOT_RUNTIME_PATH="$HOME/.enroot/runtime"

set -uo pipefail
CONTAINER="pytorch_cmha"
WORKDIR="$HOME/cmha"

if [ -z "${CUDA_VISIBLE_DEVICES:-}" ]; then
    echo "FAIL: no GPU allocated -- you are not inside an srun --gres session." >&2
    exit 1
fi

echo "--- GPUs visible on the HOST ---"
nvidia-smi -L

case "$CUDA_VISIBLE_DEVICES" in
  MIG-*) GPU_SPEC="$CUDA_VISIBLE_DEVICES" ;;
  *)     GPU_SPEC="$(nvidia-smi -L | sed -n 's/.*UUID: *\([^)]*\)).*/\1/p' | head -1)" ;;
esac
echo "will pin container to: $GPU_SPEC"

enroot start \
    --env NVIDIA_VISIBLE_DEVICES="$GPU_SPEC" \
    --env CUDA_VISIBLE_DEVICES=0 \
    --mount "$HOME:$HOME" \
    "$CONTAINER" bash -lc "
    export PYTHONPATH=\$HOME/pylibs:\${PYTHONPATH:-}
    echo '--- GPUs visible INSIDE the container ---'
    echo '(must show exactly ONE device; if you see other users GPUs, pinning failed)'
    nvidia-smi -L
    echo '--- torch ---'
    python - <<'PY'
import torch
print('torch', torch.__version__, 'cuda_available', torch.cuda.is_available())
assert torch.cuda.is_available(), 'CUDA NOT AVAILABLE INSIDE CONTAINER'
print('device count', torch.cuda.device_count())
print('device name ', torch.cuda.get_device_name(0))
x = torch.randn(512, 512, device='cuda')
print('matmul ok   ', float((x @ x.T).sum()))
import torch.linalg as L
print('eigvalsh ok ', L.eigvalsh(torch.eye(3, device='cuda')).tolist())
PY
    echo '--- data mounted? ---'
    ls -la '$WORKDIR/data/processed' | head -5
    python - <<'PY'
import numpy as np, os
p = os.path.expanduser('~/cmha/data/processed/tabular.npz')
d = np.load(p, allow_pickle=True)
print('tabular.npz OK: X_clin', d['X_clin'].shape, 'y', d['y'].shape, 'pos', int(d['y'].sum()))
v = np.load(os.path.expanduser('~/cmha/data/processed/volumes.npy'), mmap_mode='r')
print('volumes.npy OK:', v.shape, v.dtype)
PY
    echo '--- one real CV fold (smoke test) ---'
    cd '$WORKDIR' && timeout 600 python src/train_cv.py --config rupture_clean_tab
"
rc=$?
echo
if [ $rc -eq 0 ]; then
    echo "PREFLIGHT PASSED -- safe to: cd ~/cmha && sbatch hpc/run_dgx.sbatch"
else
    echo "PREFLIGHT FAILED (exit $rc) -- fix before queueing." >&2
fi
exit $rc
