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

## Python API

```python
from certamen_core import Certamen

async def main():
    arb = await Certamen.from_settings({
        "models": {
            "gpt": {"provider": "openai", "name": "gpt-4o"},
            "claude": {"provider": "anthropic", "name": "claude-sonnet-4-20250514"},
        }
    })
    result, metrics = await arb.run_tournament("What is the best approach to...")
    print(result)
```

## Architecture

See [CLAUDE.md](CLAUDE.md) for detailed architecture and development guidelines.
