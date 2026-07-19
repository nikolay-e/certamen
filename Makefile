.PHONY: help fmt lint test clean install dev build publish discover-ollama \
	ci-deps-python ci-deps-frontend ci-lint ci-test ci-quality ci-build ci-scan

PY_SOURCES = src/ tests/

BACKEND_IMAGE ?= ghcr.io/nikolay-e/certamen-backend
FRONTEND_IMAGE ?= ghcr.io/nikolay-e/certamen-frontend
IMAGE_TAG ?= local
SARIF_DIR = .sarif

help:  ## Show this help message
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-15s\033[0m %s\n", $$1, $$2}'

fmt:  ## Format and autofix code with ruff
	@echo "Formatting code..."
	ruff check --fix $(PY_SOURCES)
	ruff format $(PY_SOURCES)

lint-structure:  ## Run structure linter
	@echo "Checking structure..."
	python scripts/lint_structure.py

lint:  ## Run all linters
	@echo "Running linters..."
	python scripts/lint_structure.py
	ruff check $(PY_SOURCES)
	ruff format --check $(PY_SOURCES)
	pyright src/
	deptry src tests

test:  ## Run tests with coverage
	@echo "Running tests..."
	python -m pytest tests/ -v --cov=src/certamen --cov-report=term-missing --cov-branch

test-quick:  ## Run tests without coverage
	python -m pytest tests/ -v

test-coverage-enforce:  ## Run tests with strict coverage requirements for core modules
	@echo "Running tests with strict coverage enforcement..."
	python -m pytest tests/ -v \
		--cov=src/certamen/domain/tournament \
		--cov=src/certamen/ports \
		--cov-report=term-missing \
		--cov-branch \
		--cov-fail-under=60
	@echo "Core modules meet 60% coverage requirement"


clean:  ## Clean build artifacts and cache
	@echo "Cleaning build artifacts..."
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	rm -rf build/ dist/ .pytest_cache/ .mypy_cache/ .coverage htmlcov/

install:  ## Install package in editable mode
	pip install -e .

dev:  ## Install package with dev + gui dependencies
	uv sync --extra dev --extra gui
	pre-commit install

build:  ## Build distribution packages
	@echo "Building distribution packages..."
	uv build

publish-test:  ## Publish to TestPyPI
	@echo "Publishing to TestPyPI..."
	uv publish --publish-url https://test.pypi.org/legacy/

publish:  ## Publish to PyPI (use with caution!)
	@echo "Publishing to PyPI..."
	@read -p "Are you sure you want to publish to PyPI? [y/N] " -n 1 -r; \
	echo; \
	if [[ $$REPLY =~ ^[Yy]$$ ]]; then \
		uv publish; \
	else \
		echo "Cancelled."; \
	fi

discover-ollama:  ## Auto-discover Ollama models and generate config.yml
	@echo "Discovering Ollama models..."
	python3 scripts/discover_ollama_models.py --output config.yml

ci-deps-python:  ## Install Python deps as CI does (uv, editable + extras)
	uv pip install -e ".[dev,gui]" --system

ci-deps-frontend:  ## Install frontend deps as CI does (npm ci)
	npm ci --prefix frontend

ci-lint: ci-deps-frontend  ## CI lint: pre-commit (ruff, pyright, ...) + frontend biome + tsc + vitest
	pip install pre-commit
	pre-commit run --show-diff-on-failure --color=always --all-files
	cd frontend && npx biome ci .
	cd frontend && npx tsc -b
	cd frontend && npx vitest run

ci-test: ci-deps-python  ## CI test: pytest with branch coverage + threshold (Python 3.12)
	deptry src tests
	pytest tests/ -v \
		--cov=src/certamen \
		--cov-report=xml \
		--cov-report=term-missing \
		--cov-branch \
		--junitxml=test-results.xml
	coverage report --fail-under=30

ci-quality: ci-deps-python  ## CI quality: mutmut + radon complexity/MI + lint-imports
	uv pip install pytest-timeout import-linter --system
	PYTHONPATH=$(CURDIR)/src mutmut run || true
	mutmut results || true
	radon cc src/certamen/ --min B --show-complexity --total-average
	radon mi src/certamen/ --min B --show
	radon cc src/certamen/ --min C --total-average || true
	lint-imports || echo "Architecture violations detected"

ci-build:  ## CI build: backend + frontend images (linux/amd64, load, no push)
	docker buildx build \
		--platform linux/amd64 \
		--load \
		--file Dockerfile.backend \
		--build-arg VITE_APP_VERSION=$(IMAGE_TAG) \
		--tag $(BACKEND_IMAGE):$(IMAGE_TAG) \
		.
	docker buildx build \
		--platform linux/amd64 \
		--load \
		--file Dockerfile.frontend \
		--build-arg VITE_APP_VERSION=$(IMAGE_TAG) \
		--tag $(FRONTEND_IMAGE):$(IMAGE_TAG) \
		.

ci-scan:  ## CI scan: bandit + trivy fs scan -> SARIF in $(SARIF_DIR)/ (CodeQL stays on GitHub)
	mkdir -p $(SARIF_DIR)
	pip install "bandit[toml,sarif]"
	bandit -r src/ -f sarif -o $(SARIF_DIR)/bandit-results.sarif || true
	bandit -r src/ -f txt || true
	trivy fs --scanners vuln,secret,misconfig \
		--severity CRITICAL,HIGH \
		--format sarif \
		--output $(SARIF_DIR)/trivy-fs-results.sarif \
		. || true

.DEFAULT_GOAL := help
