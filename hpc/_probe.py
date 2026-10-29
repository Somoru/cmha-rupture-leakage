"""GPU probe used by run_dgx.sbatch. Prints exactly one CUDA_PROBE line.

torch.cuda.is_available() is NOT a sufficient test: it only checks that the
driver initialised, and returns True in the exact configuration that later dies
with

    RuntimeError: device >= 0 && device < num_gpus INTERNAL ASSERT FAILED
    device=1, num_gpus=

which happens when enroot exposes one GPU (so it is index 0 inside the
container) while SLURM's inherited CUDA_VISIBLE_DEVICES still says 1. So we
enumerate devices AND do a real allocation and matmul on cuda:0.

Lives in a file rather than a -c string to keep quoting sane through
sbatch -> enroot start -> bash -lc.
"""
import os

try:
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("torch.cuda.is_available() is False")
    n = torch.cuda.device_count()
    if n < 1:
        raise RuntimeError("device_count() == 0")
    x = torch.randn(512, 512, device="cuda:0")
    val = float((x @ x.T).sum())
    if val != val:                      # NaN guard
        raise RuntimeError("matmul produced NaN")
    print("CUDA_PROBE True n=%d name=%s cvd=%s"
          % (n, torch.cuda.get_device_name(0),
             os.environ.get("CUDA_VISIBLE_DEVICES", "unset")))
except Exception as e:
    print("CUDA_PROBE False %s: %s" % (type(e).__name__, str(e)[:120]))
