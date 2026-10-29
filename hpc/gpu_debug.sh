#!/bin/bash
# ---------------------------------------------------------------------------
# Find the enroot invocation that actually gives CUDA inside the container.
#
# Symptom this diagnoses:
#   "ERROR: The NVIDIA Driver is present, but CUDA failed to initialize.
#    [[ No CUDA-capable device is detected (error 100) ]]"
#   with nvidia-smi inside the container listing ALL the node's GPUs.
#
# That banner appears during container startup, BEFORE any of our code, which
# points at the enroot NVIDIA hook not exposing the device properly rather than
# at anything in train_cv.py.
#
# Run INSIDE a GPU allocation:
#   srun -N 1 --ntasks-per-node=4 --gres=gpu:a100_4g.20gb:1 --mem=16000 \
#        --time=00:20:00 --partition=gpu_faculty --pty bash
#   bash hpc/gpu_debug.sh
# ---------------------------------------------------------------------------
export ENROOT_TEMP_PATH="$HOME/.enroot/tmp"
export ENROOT_CACHE_PATH="$HOME/.enroot/cache"
export ENROOT_DATA_PATH="$HOME/.enroot/data"
export ENROOT_RUNTIME_PATH="$HOME/.enroot/runtime"
CONTAINER=pytorch_cmha

echo "=============== HOST ==============="
echo "hostname              : $(hostname)"
echo "CUDA_VISIBLE_DEVICES  : ${CUDA_VISIBLE_DEVICES:-unset}"
echo "NVIDIA_VISIBLE_DEVICES: ${NVIDIA_VISIBLE_DEVICES:-unset}"
echo "GPU_DEVICE_ORDINAL    : ${GPU_DEVICE_ORDINAL:-unset}"
echo "SLURM_JOB_GPUS        : ${SLURM_JOB_GPUS:-unset}"
echo "SLURM_STEP_GPUS       : ${SLURM_STEP_GPUS:-unset}"
echo "SLURM_GPUS_ON_NODE    : ${SLURM_GPUS_ON_NODE:-unset}"
echo "SLURM_JOB_ID          : ${SLURM_JOB_ID:-unset}"
echo
echo "--- index / uuid table ---"
nvidia-smi --query-gpu=index,uuid,name --format=csv 2>&1
echo
echo "--- MIG devices (if any) ---"
nvidia-smi -L 2>&1 | grep -i mig || echo "(none)"
echo
echo "--- device nodes we can actually open ---"
ls -l /dev/nvidia* 2>&1 | head -20
echo
echo "--- does CUDA work on the HOST (outside any container)? ---"
python3 -c "import torch;print('host torch cuda:',torch.cuda.is_available())" 2>&1 | tail -1 \
  || echo "(no host torch -- fine, not required)"
echo
echo "--- enroot hooks installed ---"
ls -l /etc/enroot/hooks.d/ 2>&1 | head -20
echo

PROBE='nvidia-smi -L 2>&1 | head -8; python -c "import torch;print(\"CUDA_OK\",torch.cuda.is_available(),torch.cuda.device_count())" 2>&1 | tail -2'

try () {
    echo "=================================================================="
    echo "VARIANT: $1"
    shift
    enroot start "$@" --mount "$HOME:$HOME" "$CONTAINER" bash -lc "$PROBE" 2>&1 \
      | grep -Ev '^(=|\s*$|Copyright|Various|This container|By pulling|https://|NVIDIA Release|PyTorch Version|== PyTorch)' \
      | head -14
    echo
}

try "A: no pin at all (simplest; often correct when SLURM already scopes the device)"
try "B: NVIDIA_VISIBLE_DEVICES=all" --env NVIDIA_VISIBLE_DEVICES=all
try "C: NVIDIA_VISIBLE_DEVICES=<index>" --env NVIDIA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
try "D: NVIDIA_VISIBLE_DEVICES=void + driver caps" \
    --env NVIDIA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" \
    --env NVIDIA_DRIVER_CAPABILITIES=compute,utility

UUID="$(nvidia-smi --query-gpu=uuid --format=csv,noheader 2>/dev/null | sed -n "$(( ${CUDA_VISIBLE_DEVICES:-0} + 1 ))p")"
if [ -n "$UUID" ]; then
    try "E: NVIDIA_VISIBLE_DEVICES=<uuid of allocated index> ($UUID)" \
        --env NVIDIA_VISIBLE_DEVICES="$UUID" \
        --env NVIDIA_DRIVER_CAPABILITIES=compute,utility
fi

echo "=================================================================="
echo "Pick the FIRST variant printing CUDA_OK True, and tell me its letter."
echo "If none work, the enroot NVIDIA hook is likely not enabled on this node."
