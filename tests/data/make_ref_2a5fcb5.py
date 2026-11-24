"""Record digests of the training frames commit 2a5fcb5 made, before synthesis was made faster,
for tests/test_synth_speed.py: whole-spectrum frames of recipes v3 and v4 (131 072 bins, window
and comb view, with and without isotope-charge channels) and frames of the default mixture
(singly charged ions ignored, kept with 200 background ions, dropped; evidence channels; the
grid render path). Each frame is also made by the tests' reference generator
(tests/_synth_reference.py), which must give the same digest on that commit. Run it on a copy of
the commit, in its own process:

    git archive 2a5fcb5 | tar -x -C /tmp/old
    python -I tests/data/make_ref_2a5fcb5.py /tmp/old tests/data

It writes ref_2a5fcb5.json into the output directory.
"""

from __future__ import annotations

import json
import platform
import sys
from pathlib import Path

#: the frames recorded: (recipe, crop, training items) - see _synth_reference.mixture
FRAMES = [("default:ignore:40", 32768, range(0, 6)), ("default:keep:200", 32768, range(0, 6)),
          ("default:drop:40", 32768, range(0, 4)), ("default:ignore:40:6", 32768, range(0, 2)),
          ("grid:ignore:40", 32768, range(0, 4)),
          ("v3:window:iso", 131072, range(0, 8)), ("v3:comb:iso", 131072, range(0, 4)),
          ("v3:window:noiso", 131072, range(0, 4)),
          ("v4:window:iso", 131072, range(0, 12)), ("v4:comb:iso", 131072, range(0, 6))]


def main(old_root: Path, out: Path) -> None:
    old_root = old_root.resolve()
    sys.path.insert(0, str(old_root / "src"))
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # tests/, for the helpers
    import numba
    import numpy as np
    import scipy

    import lcmsdeconv
    from _digests import canary
    from _synth_reference import frame_digest, mixture, reference_item, sort_canary, train_script
    from lcmsdeconv.synth.dataset import FrameDataset

    assert Path(lcmsdeconv.__file__).resolve().is_relative_to(old_root), lcmsdeconv.__file__
    script = train_script(old_root)
    ref: dict = {"env": {"numpy": np.__version__, "scipy": scipy.__version__,
                         "numba": numba.__version__, "machine": platform.machine(),
                         "canary": canary(), "sort_canary": sort_canary()},
                 "frames": []}
    for recipe, crop, items in FRAMES:
        mix = mixture(recipe, crop, script)
        ds = FrameDataset(mix, 10**9, base_seed=0)
        for item in items:
            digest = frame_digest(ds.frame(item)[1])
            assert frame_digest(reference_item(mix, item)[1]) == digest, (recipe, item)
            ref["frames"].append({"recipe": recipe, "crop": crop, "item": item,
                                  "digest": digest})
            print(recipe, crop, item, digest, flush=True)
    (out / "ref_2a5fcb5.json").write_text(json.dumps(ref, indent=1) + "\n")


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]))
