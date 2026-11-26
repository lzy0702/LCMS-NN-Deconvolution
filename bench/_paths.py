"""Put this package's own code on ``sys.path``: ``<repo>/src`` (lcmsdeconv) and ``<repo>/scripts``.

Every bench script imports this first, so nothing needs to be installed but the dependencies.
"""

import sys
from pathlib import Path

BENCH = Path(__file__).resolve().parent
REPO = BENCH.parent
for _p in (REPO / "scripts", REPO / "src"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
