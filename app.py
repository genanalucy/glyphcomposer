from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SCRIPTS = ROOT / "scripts"
SRC = ROOT / "src"

for candidate in (str(SCRIPTS), str(SRC)):
    if candidate not in sys.path:
        sys.path.insert(0, candidate)

from run_demo import main


if __name__ == "__main__":
    main()
