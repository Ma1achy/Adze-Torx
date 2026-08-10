import subprocess
import sys
from pathlib import Path


def test_public_boundary_scanner_passes():
    root = Path(__file__).resolve().parents[2]
    result = subprocess.run(
        [sys.executable, str(root / "scripts" / "check_public_boundaries.py")],
        cwd=root,
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
