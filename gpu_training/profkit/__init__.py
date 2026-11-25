"""The experiments of gpu_training/profile_machine.py, which measures this computer for the
developers: each runs in a process of its own, started by the runner, and returns a JSON-ready
record.

- :mod:`.environment`: the processor (every logical processor and its kind), Windows and its
  power scheme, the memory, the GPU, its driver and PyTorch;
- :mod:`.gpu`: training steps of both views of the full LadderNet on the runs' real batches
  (step times, phases, kernels, memory, nvidia-smi during a steady loop);
- :mod:`.cpu`: synthesis of training spectra on each core, with and without power throttling,
  and by many workers at once (PyTorch's DataLoader as training uses it, and the
  order-preserving dynamic dispatcher of :mod:`.dispatch`);
- :mod:`.e2e`: a short real training run;
- :mod:`.cpusets`, :mod:`.smi`, :mod:`.common`: what they share.
"""

from .common import add_paths

add_paths()  # lcmsdeconv (src), train_charge_net (scripts), probe and train_all (gpu_training)
