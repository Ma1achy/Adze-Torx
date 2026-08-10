"""Fail if source code depends on known private dependency namespaces."""

from __future__ import annotations

from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
SEARCH_ROOTS = [ROOT / "src", ROOT / "experiments", ROOT / "tests"]

PATTERNS = {
    "private Torx import/access": re.compile(
        r"(?:from\s+torx\._|import\s+torx\._|\btorx\._[A-Za-z0-9_])"
    ),
    "private GenJAX import/access": re.compile(
        r"(?:from\s+genjax\._|import\s+genjax\._|\bgenjax\._[A-Za-z0-9_])"
    ),
}

violations: list[str] = []

for search_root in SEARCH_ROOTS:
    if not search_root.exists():
        continue
    for path in search_root.rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        for label, pattern in PATTERNS.items():
            for match in pattern.finditer(text):
                line = text.count("\n", 0, match.start()) + 1
                violations.append(f"{path.relative_to(ROOT)}:{line}: {label}")

if violations:
    print("Public dependency-boundary violations found:", file=sys.stderr)
    for violation in violations:
        print(f"  {violation}", file=sys.stderr)
    raise SystemExit(1)

print("Public dependency-boundary check passed.")
