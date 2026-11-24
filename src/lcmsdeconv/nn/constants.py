"""Shape constants shared by the network and the PyTorch-free inference path.

Kept apart from ``model.py`` so that running the software never imports PyTorch: only training
and export need it.
"""

DOWNSAMPLE = 64  # 2 ** n_levels; network input length must be a multiple of this

#: LadderNet's window (grid bins read at once) and the number of its strongest bins whose ladders
#: are read, as every network before the whole-spectrum recipe was built
LADDER_WINDOW = 32768
LADDER_ACTIVE_TOP = 6144


def ladder_active_top(window: int) -> int:
    """The active-bin cap of a LadderNet reading ``window`` bins: 6 144 per 32 768 bins."""
    return int(round(LADDER_ACTIVE_TOP * window / LADDER_WINDOW))
