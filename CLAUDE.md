# Arbitrium Core

> Extends [../CLAUDE.md](../CLAUDE.md)

## Ultimate Goal

Extract the absolute maximum of knowledge from AI — every last
insight, perspective, and connection that exists within these models
but remains inaccessible through ordinary interaction.

AI models contain vastly more knowledge than any single prompt can
unlock. They hedge, simplify, omit, and hold back. The ultimate
goal of Arbitrium is to close that gap completely: leave nothing
on the table. Every technique, every architecture, every method
that can squeeze more signal out of AI should be pursued and
integrated here.

This is the North Star — not a milestone, not a roadmap item.
It may never be fully achieved. That is the point. It exists as
a permanent direction: always extract more, always push deeper,
always advance what humanity can learn from the intelligence it
has already built.

Everything in this project serves this goal.

## Commands

### Local Development

```bash
# Auto-discover Ollama models (generates config.yml)
make discover-ollama

# Run tournament
arbitrium --config config.yml

# Run YAML workflow
arbitrium workflow execute examples/workflow.yml

# Development
source venv/bin/activate
make dev              # install dev dependencies
make fmt              # format code
make lint             # lint + type check
make test             # run tests
pre-commit run -a     # all quality checks
```

### Python API

```python
from arbitrium_core import Arbitrium

async def main():
    arb = await Arbitrium.from_settings({
        "models": {
            "gpt": {"provider": "openai", "name": "gpt-4o"},
            "claude": {"provider": "anthropic", "name": "claude-3-5-sonnet-20241022"},
        }
    })
    result, metrics = await arb.run_tournament("What is the best approach to...")
    print(result)
```

## Architecture Overview

```text
┌─────────────────────────────────┐
│  arbitrium --config config.yml  │
│  ┌───────────────────────────┐  │
│  │  Tournament Engine        │  │
│  │  (src/arbitrium/core/)    │  │
│  │  ├─ Competitors (LLMs)    │  │
│  │  ├─ Judges (LLMs)         │  │
│  │  ├─ Rubrics & Scoring     │  │
│  │  └─ Knowledge Bank        │  │
│  └───────────────────────────┘  │
│         ▼                        │
│  Console Output + JSON Reports  │
└─────────────────────────────────┘
```

### Core Components

```text
src/arbitrium/
├── core/
│   ├── tournament.py      # Tournament orchestration
│   ├── scorer.py          # Rubric-based scoring
│   ├── knowledge_bank.py  # Insight extraction
│   ├── nodes/             # Workflow node system
│   └── executor/          # Graph executor
├── models/
│   └── litellm.py         # LLM provider adapters
├── cli/
│   └── main.py            # CLI entry point
├── config/                # Configuration loading
├── serialization/         # YAML workflow loader
└── utils/                 # Utilities
```

## Configuration

```yaml
# config.yml
tournament:
  models: [claude-sonnet, gpt-4o, gemini-pro]
  judges: [claude-sonnet]
  rounds: 3

rubrics:
  - accuracy: 0.4
  - reasoning: 0.3
  - completeness: 0.3
```

## Key Concepts

- **Competitors**: LLM instances generating solutions
- **Judges**: LLM instances scoring pairwise matchups
- **Rubrics**: Weighted scoring criteria
- **Knowledge Bank**: Preserved insights from eliminated models
- **Champion**: Winning solution after tournament rounds

## Workflow System

YAML-based workflow execution for building custom AI pipelines:

```yaml
# examples/simple_workflow.yml
name: Simple LLM Pipeline
nodes:
  - id: input
    type: simple/text
    properties:
      text: "Explain quantum computing"
  - id: llm
    type: llm/completion
    properties:
      model: gpt-4o
edges:
  - source: input
    target: llm
    sourceHandle: output_text
    targetHandle: prompt
outputs: [llm]
```

Available node types:

```bash
arbitrium workflow list-nodes
```

## Testing

Integration tests only (no mocks). Tests run against real LLM providers:

```bash
pytest tests/integration/ -v
```

Requires API keys in environment or `.env`.

## Environment Variables

```bash
# LLM API Keys
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
GOOGLE_API_KEY=AI...
XAI_API_KEY=xai-...

# Ollama (local models)
OLLAMA_BASE_URL=http://localhost:11434

# LiteLLM logging
LITELLM_LOG=INFO
```

## CLI Usage

```bash
# Run tournament with config file
arbitrium --config config.yml

# Run with specific models only
arbitrium --config config.yml --models gpt,claude

# Interactive mode
arbitrium --config config.yml --interactive

# Execute YAML workflow
arbitrium workflow execute workflow.yml

# Validate workflow
arbitrium workflow validate workflow.yml

# List available node types
arbitrium workflow list-nodes
```

## PyPI Installation

```bash
pip install arbitrium-core
```
