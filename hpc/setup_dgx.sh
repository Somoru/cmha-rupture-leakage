#!/bin/bash
# ---------------------------------------------------------------------------
# ONE-TIME DGX setup: pull the NGC PyTorch container and register it.
#
# Run as a CPU job -- enroot does not exist on the login node, and this is an
# ~18 GB download, not compute:
#
#   cd ~/cmha
#   srun -N 1 --ntasks-per-node=4 --mem=16000 --time=02:00:00 \
#        --partition=cpu_faculty --pty bash hpc/setup_dgx.sh
#
# STORAGE (faculty quota 200 GB, all in $HOME):
#   .sqsh ~18 GB; enroot layer cache ~9.5 GB (deleted at the end).
#
# Container tag 24.01-py3 (torch 2.2.0a0, numpy 1.24.4, sklearn 1.2.0). Do NOT
# drop to 23.01-py3: it ships torch 1.14.0a0, too old for some APIs.
# ---------------------------------------------------------------------------
set -uo pipefail

CONTAINER="pytorch_cmha"
IMAGE_TAG="nvidia/pytorch:24.01-py3"
SQSH="$HOME/${CONTAINER}.sqsh"

# The site-default enroot cache ($HOME is shared and NOT
# readable by users: "Found all layers in cache" then "tar: Permission denied"
# on every layer (exit 50). Always override all four.
export ENROOT_TEMP_PATH="$HOME/.enroot/tmp"
export ENROOT_CACHE_PATH="$HOME/.enroot/cache"
export ENROOT_DATA_PATH="$HOME/.enroot/data"
export ENROOT_RUNTIME_PATH="$HOME/.enroot/runtime"
mkdir -p "$ENROOT_TEMP_PATH" "$ENROOT_CACHE_PATH" "$ENROOT_DATA_PATH" \
         "$ENROOT_RUNTIME_PATH" "$HOME/cmha/logs"

echo ">> enroot paths redirected into \$HOME"
df -h "$HOME" 2>/dev/null | tail -1

if ! command -v enroot >/dev/null 2>&1; then
    cat >&2 <<'MSG'

!! enroot is not on PATH -- you are almost certainly on the LOGIN node.
   Get a CPU allocation and re-run:

     srun -N 1 --ntasks-per-node=4 --mem=16000 --time=02:00:00 \
          --partition=cpu_faculty --pty bash
     cd ~/cmha && bash hpc/setup_dgx.sh

MSG
    exit 1
fi
echo ">> enroot: $(command -v enroot)"

# An interrupted import leaves a .sqsh that is NOT a valid squashfs, and
# mksquashfs then tries to APPEND rather than overwrite:
#   "Can't find a SQUASHFS superblock ... will not overwrite - ABORTING!"
valid_sqsh () {
    [ -s "$1" ] && [ "$(head -c 4 "$1" 2>/dev/null)" = "hsqs" ]
}

if [ -f "$SQSH" ] && ! valid_sqsh "$SQSH"; then
    echo ">> removing partial/corrupt $SQSH"
    rm -f "$SQSH"
fi

if enroot list 2>/dev/null | grep -qx "$CONTAINER"; then
    echo ">> container '$CONTAINER' already exists -- skipping import/create."
else
    if ! valid_sqsh "$SQSH"; then
        rm -f "$SQSH"          # mksquashfs will not overwrite an existing file
        echo ">> importing $IMAGE_TAG (~18 GB; layers cache after the first try)"
        enroot import --output "$SQSH" "docker://nvcr.io#${IMAGE_TAG}" || {
            echo "!! import failed. If it reached 'Creating squashfs filesystem'" >&2
            echo "   and complained about a superblock, just re-run: the layer" >&2
            echo "   cache survives, so it will be quick." >&2
            exit 1
        }
    fi
    valid_sqsh "$SQSH" || { echo "!! $SQSH is not a valid image" >&2; exit 1; }
    echo ">> creating container '$CONTAINER'"
    enroot remove -f "$CONTAINER" 2>/dev/null || true
    enroot create --name "$CONTAINER" "$SQSH"
fi

echo ">> reclaiming layer cache (~9.5 GB; the .sqsh is what matters now)"
rm -rf "${ENROOT_CACHE_PATH:?}"/* 2>/dev/null || true

# ---------------------------------------------------------------------------
# Verify the container's OWN Python stack, fully isolated from the account.
#
# This account already carries packages that break the container:
#   $HOME/pylibs                 sklearn 1.7.2, numpy 2.2.6, scipy 1.15.3
#   $HOME/.local/lib/python3.10  pandas -- loaded AUTOMATICALLY as user site,
#                                with no PYTHONPATH involved
# With those visible, `import torch` returned 2.9.1+cu128 instead of the
# container's 2.2.0a0, and sklearn died with
#   "numpy.core.multiarray failed to import"
# because the container's pyarrow is built against numpy 1.x while numpy 2.2.6
# was being loaded.
#
# PYTHONNOUSERSITE=1 is the important one: clearing PYTHONPATH alone does NOT
# hide $HOME/.local. The container ships a consistent torch 2.2.0a0 /
# numpy 1.24.4 / sklearn 1.2.0, so nothing needs installing -- just as well,
# since compute nodes have no DNS and pip cannot reach PyPI.
# ---------------------------------------------------------------------------
echo ">> verifying the container's own Python stack (isolated)"
enroot start --mount "$HOME:$HOME" "$CONTAINER" bash -lc '
  unset PYTHONPATH
  export PYTHONNOUSERSITE=1
  python - <<PY
import torch, numpy, sklearn, scipy
print("torch  ", torch.__version__)
print("numpy  ", numpy.__version__)
print("sklearn", sklearn.__version__)
print("scipy  ", scipy.__version__)
assert torch.__version__.startswith("2.2"), "container torch is being shadowed"
assert numpy.__version__.startswith("1."), "container numpy is being shadowed"
print("OK: container stack is clean")
PY
'
rc=$?

echo
if [ $rc -eq 0 ]; then
    echo ">> setup complete. \$HOME usage:"
    du -sh "$HOME" 2>/dev/null || true
    echo
    echo "NEXT:  cd ~/cmha && sbatch hpc/run_dgx.sbatch"
else
    echo "!! verification failed (exit $rc)." >&2
    echo "   If it reports the stack is being shadowed, move the account's own" >&2
    echo "   packages aside and re-run:   mv ~/pylibs ~/pylibs.bak" >&2
    exit $rc
fi
