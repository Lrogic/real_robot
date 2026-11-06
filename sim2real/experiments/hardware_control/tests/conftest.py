import sys
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import settings  # noqa: E402,F401  (adds real_robot and the experiments directory)
