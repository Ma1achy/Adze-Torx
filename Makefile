.PHONY: install install-torx format format-check lint typecheck test test-slow check boundaries

install:
	python -m pip install -e ".[dev]"

install-torx:
	python -m pip install -e ".[dev]"

format:
	ruff format .
	ruff check --fix .

format-check:
	ruff format --check .

lint:
	ruff check .

typecheck:
	pyright --pythonpath "$$(python -c 'import sys; print(sys.executable)')"

test:
	pytest -m "not slow"

test-slow:
	pytest -m "slow" || test $$? -eq 5

boundaries:
	python scripts/check_public_boundaries.py

check: boundaries format-check lint typecheck test
