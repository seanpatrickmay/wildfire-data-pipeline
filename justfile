# Wildfire Data Pipeline — common tasks

# Show available recipes
default:
    @just --list

# Install all dependencies
setup:
    uv sync --dev

# Run linting checks
lint:
    uv run ruff check .
    uv run ruff format --check .

# Run type checking
typecheck:
    uv run mypy src/

# Run all tests (unit only)
test *ARGS:
    uv run pytest -v --tb=short -m "not integration" {{ ARGS }}

# Run tests with coverage
test-cov:
    uv run pytest -v --tb=short -m "not integration" --cov=wildfire_pipeline --cov-report=term-missing

# Run integration tests (requires GEE auth)
test-integration:
    uv run pytest -v -m integration

# Format code
fmt:
    uv run ruff format .
    uv run ruff check --fix .

# Run all quality checks (lint + typecheck + test)
check: lint typecheck test

# Download fire data
download FIRE *ARGS:
    uv run wildfire download {{ FIRE }} {{ ARGS }}

# Process fire labels
process INPUT *ARGS:
    uv run wildfire process {{ INPUT }} {{ ARGS }}

# Validate fire data
validate INPUT:
    uv run wildfire validate {{ INPUT }}

# List available fires
fires:
    uv run wildfire list-fires

# Run security scan
security:
    uv run bandit -r src/

# Clean generated files
clean:
    rm -rf .pytest_cache .mypy_cache .ruff_cache
    find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
