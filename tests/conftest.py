"""Shared pytest fixtures.

Keep stochastic seed trees explicit. Add fixtures only when multiple tests use
them; do not hide randomness behind implicit global fixtures.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
