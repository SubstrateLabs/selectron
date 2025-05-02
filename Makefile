.PHONY: format lint typecheck test install clean

cli:
	uv run scripts/cli.py

# ================================
# housekeeping
# ================================

format:
	ruff format . && ruff check --fix .

lint:
	ruff check .

typecheck:
	uv run pyright

install:
	uv pip install -e ".[dev]"

test:
	uv run pytest -s

clean:
	rm -rf .ruff_cache
	rm -rf .pytest_cache
	rm -rf .mypy_cache
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	find . -type d -name __pycache__ -exec rm -rf {} +
