"""Shape constants shared by the network and the PyTorch-free inference path.

Kept apart from ``model.py`` so that running the software never imports PyTorch: only training
and export need it.
"""

DOWNSAMPLE = 64  # 2 ** n_levels; network input length must be a multiple of this
