#!/usr/bin/env python3
"""Discover Ollama models and generate config.yml automatically."""

import argparse
import sys
from typing import Any

import httpx
import yaml

# Use centralized constant if package is available, otherwise fallback
try:
    from arbitrium_core.shared.constants import DEFAULT_OLLAMA_URL
except ImportError:
    DEFAULT_OLLAMA_URL = "http://localhost:11434"


def get_ollama_models(base_url: str) -> list[dict[str, Any]]:
    """Fetch list of available models from Ollama API."""
    try:
        response = httpx.get(f"{base_url}/api/tags", timeout=10.0)
        response.raise_for_status()
        data = response.json()
        return data.get("models", [])
    except Exception as e:
        print(f"Error fetching models from Ollama: {e}", file=sys.stderr)
        return []


def get_ollama_model_details(
    base_url: str, model_name: str
) -> dict[str, Any] | None:
    """Fetch detailed information for a specific model."""
    try:
        response = httpx.post(
            f"{base_url}/api/show",
            json={"name": model_name},
            timeout=10.0,
        )
        response.raise_for_status()
        return response.json()
    except Exception as e:
        print(
            f"Warning: Could not fetch details for {model_name}: {e}",
            file=sys.stderr,
        )
        return None


def generate_model_config(
    model: dict[str, Any], base_url: str, details: dict[str, Any] | None
) -> dict[str, Any]:
    """Generate configuration for a single model."""
    model_name = model["name"]
    display_name = model.get("details", {}).get("family", model_name)

    size_gb = model.get("size", 0) / (1024**3)
    if size_gb < 1:
        size_label = f"{model.get('size', 0) / (1024**2):.0f}MB"
    else:
        size_label = f"{size_gb:.1f}GB"

    config = {
        "provider": "ollama",
        "model_name": model_name,
        "display_name": f"{display_name} ({size_label})",
        "base_url": base_url,
    }

    return config


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Discover Ollama models and generate config"
    )
    parser.add_argument(
        "--base-url",
        default=DEFAULT_OLLAMA_URL,
        help=f"Ollama base URL (default: {DEFAULT_OLLAMA_URL})",
    )
    parser.add_argument(
        "--output",
        "-o",
        default="config.yml",
        help="Output config file (default: config.yml)",
    )
    parser.add_argument(
        "--question",
        "-q",
        default="What is the capital of France?",
        help="Question for tournament",
    )
    parser.add_argument(
        "--judge",
        "-j",
        help="Judge model (auto-select largest if not specified)",
    )

    args = parser.parse_args()

    print(f"Discovering models from {args.base_url}...", file=sys.stderr)
    models = get_ollama_models(args.base_url)

    if not models:
        print("No models found!", file=sys.stderr)
        sys.exit(1)

    print(f"Found {len(models)} models:", file=sys.stderr)

    models_config = {}
    for model in models:
        model_name = model["name"]
        print(f"\n  {model_name}", file=sys.stderr)

        details = get_ollama_model_details(args.base_url, model_name)
        safe_key = model_name.replace(":", "_").replace("/", "_")
        models_config[safe_key] = generate_model_config(
            model, args.base_url, details
        )

    judge_model = args.judge
    if not judge_model:
        judge_model = next(iter(models_config.keys()))
        print(
            f"\n  [i] Auto-selecting judge model: {judge_model}",
            file=sys.stderr,
        )

    config = {
        "question": args.question,
        "models": models_config,
        "tournament": {
            "judge_model": judge_model,
            "improvement_rounds": 1,
            "anonymize": True,
        },
        "knowledge_bank": {
            "enabled": True,
            "max_insights": 3,
            "similarity_threshold": 0.85,
            "extraction_model": judge_model,
        },
        "outputs_dir": "./outputs",
        "logging": {
            "level": "INFO",
            "file_logging": True,
            "console_logging": True,
        },
    }

    print(f"\nWriting config to {args.output}...", file=sys.stderr)
    with open(args.output, "w") as f:
        yaml.dump(
            config,
            f,
            default_flow_style=False,
            sort_keys=False,
            allow_unicode=True,
        )

    print(
        f"✓ Config generated with {len(models_config)} models", file=sys.stderr
    )


if __name__ == "__main__":
    main()
