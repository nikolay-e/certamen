# Contributing to Arbitrium Core

Thank you for considering contributing to Arbitrium Core! We welcome contributions of all kinds:

- Bug reports
- Feature requests
- Code contributions
- Documentation improvements
- Example workflows

Please read our [Code of Conduct](CODE_OF_CONDUCT.md) before participating.

## Getting Started

1. **Fork** the repository on GitHub.
2. **Clone** your forked repository:

    ```sh
    git clone https://github.com/YOUR_USERNAME/arbitrium-core.git
    cd arbitrium-core
    ```

3. **Set up the environment:**

    ```sh
    python -m venv venv
    source venv/bin/activate
    pip install -e .[dev]
    ```

4. **Install pre-commit hooks:**

    ```sh
    pre-commit install
    pre-commit install --hook-type commit-msg
    ```

## Development Workflow

### Branch Naming

- `feature/short-description` for new features
- `fix/short-description` for bug fixes
- `exp/short-description` for experiments

### Making Changes

1. Create a branch from `main`:

    ```sh
    git checkout -b feature/my-feature
    ```

2. Make your changes.
3. Run quality checks:

    ```sh
    make fmt           # format code
    make lint          # lint + type check
    make test          # run tests with coverage
    pre-commit run -a  # all pre-commit hooks
    ```

4. Commit with a concise message (under 72 characters):

    ```sh
    git commit -m "add tournament timeout configuration"
    ```

### Available Make Targets

| Target | Purpose |
|--------|---------|
| `make dev` | Install dev dependencies + pre-commit hooks |
| `make fmt` | Format code (black + isort) |
| `make lint` | Lint + type check (ruff, black, isort, mypy) |
| `make test` | Run tests with coverage |
| `make test-quick` | Run tests without coverage |
| `make clean` | Remove build artifacts |

## Testing Philosophy

**Integration tests only.** We do not accept unit tests.

- Tests must exercise real components, not isolated functions
- Mock only external LLM providers (never internal logic)
- Property-based testing with Hypothesis is encouraged
- Tests go in `tests/integration/`

Do not submit:

- Unit tests that mock internal classes
- Tests in `tests/unit/` (reserved for minimal CLI flag tests)
- Tests that modify themselves to pass

## Code Standards

### Self-Documenting Code

We do not use docstrings or inline comments explaining "what" the code does. Instead:

- Use clear, descriptive function and variable names
- Keep functions small and focused
- Use type hints for all function signatures
- Let the code speak for itself

### Type Safety

- All code must pass `mypy --strict`
- Use proper type annotations on all public functions
- Avoid `Any` unless absolutely necessary

### Architecture

The project follows Clean Architecture with enforced layer boundaries:

```
interfaces -> application -> infrastructure -> domain -> ports -> shared
```

Import-linter enforces these boundaries in CI. Do not introduce cross-layer imports that violate this direction.

## Commit Messages

- Use imperative mood: "add feature" not "added feature"
- Keep under 72 characters
- No emoji prefixes
- No AI-generated footers or co-authored-by lines
- Follow [Conventional Commits](https://www.conventionalcommits.org/) format

## Submitting a Pull Request

1. Push your branch:

    ```sh
    git push origin feature/my-feature
    ```

2. Open a Pull Request against `main`.
3. Fill in the PR template.
4. Ensure CI passes (pre-commit, lint, tests on all platforms).

A maintainer will review your PR. We may request changes before merging.

## Reporting Issues

Use [GitHub Issues](https://github.com/nikolay-e/arbitrium-core/issues) with the provided templates:

- **Bug reports**: Include version, Python version, OS, and reproduction steps
- **Feature requests**: Describe the problem and proposed solution

## Security Vulnerabilities

Do **not** report security vulnerabilities through public issues. See [SECURITY.md](SECURITY.md) for responsible disclosure instructions.
