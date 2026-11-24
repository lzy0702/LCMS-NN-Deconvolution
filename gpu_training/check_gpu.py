"""Check that PyTorch sees the GPU and that LadderNet training steps of both views run on it.

    .venv\\Scripts\\python gpu_training\\check_gpu.py      (Windows)
    .venv/bin/python gpu_training/check_gpu.py          (Linux)
"""

import sys
import time
from pathlib import Path


def main() -> int:
    try:
        import torch
    except ImportError:
        print("PyTorch is not installed in this Python. Run this with the environment's Python "
              "(.venv\\Scripts\\python on Windows), or run gpu_training\\setup_windows.bat.")
        return 1
    print(f"python {sys.version.split()[0]}, torch {torch.__version__}")
    try:
        import numba

        print(f"numba {numba.__version__}: spectra are generated at full speed")
    except ImportError:
        print("numba is NOT installed: spectrum generation will be about 20 times slower.")
        print('Install it with: pip install -e ".[train,fast]"   (or: pip install numba)')
    if not torch.cuda.is_available():
        print("CUDA is NOT available: this torch build is CPU-only or the NVIDIA driver is missing.")
        print("Reinstall PyTorch for CUDA with: gpu_training\\setup_windows.bat --reinstall-torch "
              "(Linux: bash gpu_training/setup_linux.sh --reinstall-torch)")
        return 1
    dev = torch.device("cuda")
    props = torch.cuda.get_device_properties(dev)
    print(f"GPU: {props.name}, {props.total_memory / 2**30:.1f} GiB")

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from train_all import BATCH, HEAD_CHUNK

    from lcmsdeconv.nn.losses import ChargeLoss
    from lcmsdeconv.nn.model import build_model

    loss_fn = ChargeLoss()
    length = 131072
    for view, in_ch in (("window", 7), ("comb", 8)):
        for bs in (BATCH, max(1, BATCH // 2), 1):
            model = build_model("full", z_max=100, in_ch=in_ch, arch="ladder", window=length,
                                iso=True, view=view, head_chunk=HEAD_CHUNK).to(dev)
            opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
            x = torch.rand(bs, in_ch, length, device=dev)
            x[:, 0] *= 5.0  # every bin strong: the active-bin cap binds, the most a step needs
            x[:, 4] = torch.randint(0, 30, (bs, length), device=dev).float()
            x[:, 6] = torch.randint(0, 40, (bs, length), device=dev).float()
            if view == "comb":
                x[:, -1] = 1.0
            batch = {"features": x,
                     "topk_z": torch.randint(0, 40, (bs, length, 3), device=dev),
                     "topk_w": torch.rand(bs, length, 3, device=dev) / 3,
                     "heat": torch.zeros(bs, length, device=dev)}
            torch.cuda.reset_peak_memory_stats()
            try:
                for i in range(3):
                    torch.cuda.synchronize()
                    t = time.time()
                    loss = loss_fn(model(batch["features"]), batch)["loss"]
                    opt.zero_grad()
                    loss.backward()
                    opt.step()
                    torch.cuda.synchronize()
                    print(f"{view} view, step {i}: loss {float(loss):.3f}, {time.time() - t:.3f} s "
                          f"for a batch of {bs}")
            except torch.cuda.OutOfMemoryError:
                print(f"{view} view, batch of {bs}: out of GPU memory, trying a smaller batch")
                del model, opt, x, batch
                torch.cuda.empty_cache()
                continue
            peak = torch.cuda.max_memory_reserved() / 2**20
            print(f"{view} view: peak GPU memory {peak:.0f} MiB with a batch of {bs} -- all good")
            if bs < BATCH:
                print(f"Add --batch-size {bs} to the training (train_all.bat accepts it).")
            del model, opt, x, batch
            torch.cuda.empty_cache()
            break
        else:
            print(f"{view} view: even a batch of 1 does not fit: this GPU has too little free "
                  "memory.")
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
