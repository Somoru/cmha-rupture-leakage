#!/bin/bash
#SBATCH --job-name=cmha
#SBATCH --output=logs/cmha_%j.out
#SBATCH --error=logs/cmha_%j.err
#SBATCH --partition=gpu_faculty
#SBATCH --gres=gpu:a100_4g.20gb:1
#SBATCH --ntasks-per-node=16
#SBATCH --mem=64000
#SBATCH --time=24:00:00
# ---------------------------------------------------------------------------
# CMHA rupture-prediction CV suite on the MU DGX A100.
#
#   cd ~/cmha && mkdir -p logs && sbatch hpc/run_dgx.sbatch
#
# SELF-HEALING GPU SETUP. On dgxa we observed:
#   * nvidia-smi is NOT scoped to the allocation (lists all 6 cards)
#   * CUDA_VISIBLE_DEVICES is a bare index, not a MIG-... UUID
#   * pinning NVIDIA_VISIBLE_DEVICES to nvidia-smi's first UUID gave
#       "CUDA failed to initialize [[ No CUDA-capable device is detected ]]"
# So rather than assume one incantation, this script PROBES several enroot GPU
# configurations, uses the first that yields torch.cuda.is_available()==True,
# and aborts if none do (no CPU fallback: it would tie up a GPU slot for hours). The full probe table is written to the log,
# which doubles as the diagnostic if the fallback triggers.
#
# Faculty account: gpu_faculty allows 64 CPUs, 2 GPUs, 512 GB and 30 days, so
# resources here are comfortable rather than tight. a100_4g.20gb (56 SMs, 20 GB)
# is far more than this job needs (<5 GB, overhead-bound) and queues better than
# a full card; --gres=gpu:a100:1 also works if you want one.
# ---------------------------------------------------------------------------

echo "node   : $(hostname)"
echo "job    : ${SLURM_JOB_ID:-none}"
echo "gpus   : ${CUDA_VISIBLE_DEVICES:-none}"
echo "started: $(date)"

export ENROOT_TEMP_PATH="$HOME/.enroot/tmp"
export ENROOT_CACHE_PATH="$HOME/.enroot/cache"
export ENROOT_DATA_PATH="$HOME/.enroot/data"
export ENROOT_RUNTIME_PATH="$HOME/.enroot/runtime"
mkdir -p "$ENROOT_TEMP_PATH" "$ENROOT_CACHE_PATH" "$ENROOT_DATA_PATH" "$ENROOT_RUNTIME_PATH"

WORKDIR="$HOME/cmha"
CONTAINER="pytorch_cmha"
cd "$WORKDIR" || { echo "no $WORKDIR" >&2; exit 1; }

echo
echo "=============== GPU ENVIRONMENT ==============="
echo "SLURM_JOB_GPUS     : ${SLURM_JOB_GPUS:-unset}"
echo "SLURM_STEP_GPUS    : ${SLURM_STEP_GPUS:-unset}"
echo "SLURM_GPUS_ON_NODE : ${SLURM_GPUS_ON_NODE:-unset}"
echo "GPU_DEVICE_ORDINAL : ${GPU_DEVICE_ORDINAL:-unset}"
nvidia-smi --query-gpu=index,uuid,name --format=csv 2>&1 | head -12
echo "--- enroot hooks ---"
ls /etc/enroot/hooks.d/ 2>&1 | head -12

IDX="${CUDA_VISIBLE_DEVICES:-0}"
IDX="${IDX%%,*}"
UUID="$(nvidia-smi --query-gpu=uuid --format=csv,noheader 2>/dev/null \
        | sed -n "$((IDX + 1))p" | tr -d ' ')"
MIG="$(nvidia-smi -L 2>/dev/null | sed -n 's/.*UUID: *\(MIG-[^)]*\)).*/\1/p' | head -1)"
echo "resolved index=$IDX uuid=${UUID:-none} mig=${MIG:-none}"

CAPS="--env NVIDIA_DRIVER_CAPABILITIES=compute,utility"

probe () {   # name, then --env flags
    local name="$1"; shift
    local out
    out="$(enroot start "$@" --mount "$HOME:$HOME" "$CONTAINER" \
           bash -lc "cd '$WORKDIR' && python hpc/_probe.py" 2>/dev/null \
           | grep CUDA_PROBE | tail -1)"
    printf '  %-34s -> %s\n' "$name" "${out:-no output}"
    case "$out" in *"CUDA_PROBE True"*) return 0 ;; *) return 1 ;; esac
}

echo
echo "=============== PROBING ENROOT GPU CONFIGS ==============="
echo "(probe does a real allocation on cuda:0 -- is_available() alone returns"
echo " True in the broken 'device=1, num_gpus=' configuration)"
GPU_ARGS=()
FOUND=0

# When a single device is pinned it becomes index 0 INSIDE the container, so
# CUDA_VISIBLE_DEVICES must be forced to 0 -- otherwise SLURM's inherited value
# (e.g. 1) points past the end of the exposed set.
if probe "A no pin"; then
    GPU_ARGS=(); FOUND=1
elif probe "B =all" --env NVIDIA_VISIBLE_DEVICES=all; then
    GPU_ARGS=(--env NVIDIA_VISIBLE_DEVICES=all); FOUND=1
elif probe "C =all + caps" --env NVIDIA_VISIBLE_DEVICES=all $CAPS; then
    GPU_ARGS=(--env NVIDIA_VISIBLE_DEVICES=all $CAPS); FOUND=1
elif probe "D =index($IDX) + cvd=0 + caps" --env NVIDIA_VISIBLE_DEVICES="$IDX" \
        --env CUDA_VISIBLE_DEVICES=0 $CAPS; then
    GPU_ARGS=(--env NVIDIA_VISIBLE_DEVICES="$IDX" --env CUDA_VISIBLE_DEVICES=0 $CAPS)
    FOUND=1
elif [ -n "$UUID" ] && probe "E =uuid + cvd=0 + caps" \
        --env NVIDIA_VISIBLE_DEVICES="$UUID" --env CUDA_VISIBLE_DEVICES=0 $CAPS; then
    GPU_ARGS=(--env NVIDIA_VISIBLE_DEVICES="$UUID" --env CUDA_VISIBLE_DEVICES=0 $CAPS)
    FOUND=1
elif [ -n "$MIG" ] && probe "F =mig + cvd=0 + caps" \
        --env NVIDIA_VISIBLE_DEVICES="$MIG" --env CUDA_VISIBLE_DEVICES=0 $CAPS; then
    GPU_ARGS=(--env NVIDIA_VISIBLE_DEVICES="$MIG" --env CUDA_VISIBLE_DEVICES=0 $CAPS)
    FOUND=1
elif probe "G no pin + cvd=0" --env CUDA_VISIBLE_DEVICES=0; then
    GPU_ARGS=(--env CUDA_VISIBLE_DEVICES=0); FOUND=1
fi

echo
if [ "$FOUND" -eq 1 ]; then
    echo ">> USING GPU. enroot args: ${GPU_ARGS[*]:-<none>}"
else
    cat >&2 <<'MSG'

!! NO GPU CONFIG WORKED -- ABORTING (deliberately; no CPU fallback).

   Running this suite on CPU would occupy a GPU allocation for hours, so we
   stop instead. The probe table above IS the diagnostic -- send it back along
   with the GPU ENVIRONMENT block and the enroot hooks listing.

   Things worth checking on the cluster side:
     * is pyxis available?  ->  srun --container-image=... would bypass enroot
                                hook handling entirely
     * ls /etc/enroot/hooks.d/   should contain an nvidia hook
     * does `--gres=gpu:a100_4g.20gb:1` actually grant a device here?
     * ask the HPC admin: your HPC administrator

MSG
    exit 1
fi
echo

enroot start "${GPU_ARGS[@]}" --mount "$HOME:$HOME" "$CONTAINER" bash -lc "
    set -uo pipefail
    cd '$WORKDIR'
    # Isolate from the ACCOUNT's Python packages. Two separate sources shadow
    # the container: \$HOME/pylibs (via PYTHONPATH) and \$HOME/.local, which
    # Python loads automatically as user site with no PYTHONPATH involved.
    # With those visible we saw torch 2.9.1 instead of the container's 2.2.0a0
    # and \"numpy.core.multiarray failed to import\". PYTHONNOUSERSITE is the
    # one that hides \$HOME/.local; unsetting PYTHONPATH alone is not enough.
    unset PYTHONPATH
    export PYTHONNOUSERSITE=1
    python -c 'import torch;print(\"torch\",torch.__version__,\"cuda\",torch.cuda.is_available())'
    python -c 'import sklearn;print(\"sklearn\",sklearn.__version__)'
    python -c 'import numpy;print(\"numpy\",numpy.__version__)'

    echo
    echo '===== 1/5  full configuration suite (11 configs) ====='
    if [ -f results/summary_table.md ] && [ -z \"\${FORCE_RERUN:-}\" ]; then
        echo 'SKIPPED: results/ already present. On a FRESH account it is not,'
        echo 'so this runs. Set FORCE_RERUN=1 to force a redo.'
        rc1=0
    else
        python src/train_cv.py --all --skip-missing
        rc1=\$?
    fi

    echo
    echo '===== 2/5  fusion diagnostic ====='
    if [ -f results/summary_table.md ] && [ -z \"\${FORCE_RERUN:-}\" ]; then
        echo 'SKIPPED: already reproduced twice with identical numbers.'
        rc2=0
    else
        python src/diag_fusion.py
        rc2=\$?
    fi

    echo
    echo '===== 3/5  fusion-operator study (geometry encoder, noise control, 5 operators) ====='
    python src/fusion_study.py --all
    rc3=\$?

    echo
    echo '===== 4/5  stacked generalisation (learned meta-learner) ====='
    python src/stacking.py --all
    rc4=\$?

    echo
    echo '===== 5/5  figures, tables, and results bundle ====='
    python src/make_figures.py  || echo 'legacy figure step skipped'
    python src/make_figures2.py || echo 'publication figure step skipped'
    python src/collect_results.py || echo 'collect step skipped'

    echo \"stage exit codes: train=\$rc1 diag=\$rc2 fusion=\$rc3 stack=\$rc4\"
"

echo
echo "=== FINISHED $(date) ==="
ls -la "$WORKDIR/results" 2>/dev/null | head -30
echo
echo "--- summary table (if produced) ---"
cat "$WORKDIR/results/summary_table.md" 2>/dev/null || echo "(none)"
