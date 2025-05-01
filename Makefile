# Install the package in development mode
install:
	uv pip install -e ".[dev]"

# Run tests
test:
	uv run pytest


# Clean up cache and build artifacts
clean:
	rm -rf .ruff_cache
	rm -rf .pytest_cache
	rm -rf .mypy_cache
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	find . -type d -name __pycache__ -exec rm -rf {} +
