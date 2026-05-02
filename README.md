# Certamen

Tournament-based AI decision synthesis framework where models compete and critique each other to extract maximum knowledge.

## Installation

```bash
pip install certamen-core
```

## Quick Start

```bash
# Auto-discover Ollama models
make discover-ollama

# Run tournament
certamen --config config.yml
```

## Configuration

`config.yml` is a *slim* file: question + models + workflow reference.

```yaml
question: "What is the best approach to ..."

models:
  gpt:
    provider: openai
    model_name: gpt-5.2-pro
  claude:
    provider: anthropic
    model_name: anthropic/claude-opus-4-6
  gemini:
    provider: google
    model_name: gemini/gemini-3.1-pro-preview
  grok:
    provider: xai
    model_name: xai/grok-4-1-fast-reasoning

workflow: diamond-tournament
```

The workflow defines the tournament logic; `config.yml` only declares
*what to ask* and *who to ask*. Built-in workflows ship with the
package (`diamond-tournament`); custom workflows live under
`~/.certamen/workflows/` or can be passed via `-w path.yml`.

## Architecture

See [CLAUDE.md](CLAUDE.md) for detailed architecture and development guidelines.
