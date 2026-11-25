"""The experiments of gpu_training/profile_machine.py, which measures this computer for the
developers: each runs in a process of its own, started by the runner, and returns a JSON-ready
record.

- :mod:`.environment`: the processor (every logical processor and its kind), Windows and its
  power scheme, the memory, the GPU, its driver and PyTorch;
- :mod:`.gpu`: training steps of both views of the full LadderNet on the runs' real batches
  (step times, phases, kernels, memory, nvidia-smi during a steady loop);
- :mod:`.cpu`: training's own data loading (torch's DataLoader and the ordered loader of
  lcmsdeconv.synth.pool, the workers placed as training places them), synthesis of training
  spectra by one worker on a core of each kind, with and without power throttling;
- :mod:`.e2e`: a short real training run, exactly as train_all trains;
- :mod:`.cpusets`, :mod:`.smi`, :mod:`.common`: what they share.
"""

from .common import add_paths

add_paths()  # lcmsdeconv (src), train_charge_net (scripts), probe and train_all (gpu_training)
