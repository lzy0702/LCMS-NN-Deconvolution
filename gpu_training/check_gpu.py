"""Check that PyTorch sees the GPU and that one LadderNet training step runs on it.

    python gpu_training/check_gpu.py
"""

import sys
import time


def main() -> int:
    try:
        import torch
    except ImportError:
        print("PyTorch is not installed. See gpu_training/README.md, step 3.")
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
        print("Reinstall torch with the CUDA command from https://pytorch.org/get-started/locally/")
        return 1
    dev = torch.device("cuda")
    props = torch.cuda.get_device_properties(dev)
    print(f"GPU: {props.name}, {props.total_memory / 2**30:.1f} GiB")

    from lcmsdeconv.nn.losses import ChargeLoss
    from lcmsdeconv.nn.model import build_model

    loss_fn = ChargeLoss(ignore_charges=(1,))
    for bs in (8, 4, 2):
        model = build_model("full", z_max=100, in_ch=4, arch="ladder").to(dev)
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
        x = torch.rand(bs, 4, 32768, device=dev)
        x[:, 0] *= 5.0  # every bin strong: the largest active set training can meet
        batch = {"features": x,
                 "topk_z": torch.randint(0, 40, (bs, 32768, 3), device=dev),
                 "topk_w": torch.rand(bs, 32768, 3, device=dev) / 3,
                 "heat": torch.zeros(bs, 32768, device=dev)}
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
                print(f"step {i}: loss {float(loss):.3f}, {time.time() - t:.3f} s for a batch of {bs}")
        except torch.cuda.OutOfMemoryError:
            print(f"batch of {bs}: out of GPU memory, trying a smaller batch")
            del model, opt, x, batch
            torch.cuda.empty_cache()
            continue
        peak = torch.cuda.max_memory_reserved() / 2**20
        print(f"peak GPU memory {peak:.0f} MiB with a batch of {bs} -- all good")
        if bs < 8:
            print(f"Add --batch-size {bs} to the training commands (the run scripts accept it).")
        return 0
    print("Even a batch of 2 does not fit: this GPU has too little free memory.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
