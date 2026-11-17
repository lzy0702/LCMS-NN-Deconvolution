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
    if not torch.cuda.is_available():
        print("CUDA is NOT available: this torch build is CPU-only or the NVIDIA driver is missing.")
        print("Reinstall torch with the CUDA command from https://pytorch.org/get-started/locally/")
        return 1
    dev = torch.device("cuda")
    props = torch.cuda.get_device_properties(dev)
    print(f"GPU: {props.name}, {props.total_memory / 2**30:.1f} GiB")

    from lcmsdeconv.nn.losses import ChargeLoss
    from lcmsdeconv.nn.model import build_model

    model = build_model("full", z_max=100, in_ch=4, arch="ladder").to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    x = torch.rand(8, 4, 32768, device=dev)
    x[:, 0] *= 5.0
    batch = {"features": x,
             "topk_z": torch.randint(0, 40, (8, 32768, 3), device=dev),
             "topk_w": torch.rand(8, 32768, 3, device=dev) / 3,
             "heat": torch.zeros(8, 32768, device=dev)}
    loss_fn = ChargeLoss(ignore_charges=(1,))
    for i in range(3):
        torch.cuda.synchronize()
        t = time.time()
        loss = loss_fn(model(batch["features"]), batch)["loss"]
        opt.zero_grad()
        loss.backward()
        opt.step()
        torch.cuda.synchronize()
        print(f"step {i}: loss {float(loss):.3f}, {time.time() - t:.3f} s for a batch of 8")
    print(f"peak GPU memory {torch.cuda.max_memory_allocated() / 2**20:.0f} MiB -- all good")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
