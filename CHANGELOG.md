# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.0] - 2026-02-21

### Added

- Tournament-based LLM evaluation engine with multi-round elimination
- Knowledge Bank for preserving insights from eliminated models
- Rubric-based scoring with configurable weights
- YAML workflow system for custom AI pipelines
- CLI interface with `arbitrium` command (tournament and workflow modes)
- Python API via `Arbitrium.from_settings()`
- Support for 19 LLM providers via LiteLLM (OpenAI, Anthropic, Google, Ollama, etc.)
- Provenance tracking and cost accounting
- Auto-discovery of Ollama local models
- SQLite response cache for deduplication
- Clean Architecture with import-linter enforcement
- Comprehensive pre-commit hooks (14 layers of checks)
- Cross-platform CI (Linux, macOS, Windows) with Python 3.10-3.13
- Mutation testing on domain layer
- PEP 561 type stubs (`py.typed` marker)

[Unreleased]: https://github.com/nikolay-e/arbitrium-core/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/nikolay-e/arbitrium-core/releases/tag/v0.1.0
