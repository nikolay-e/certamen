# Examples

Python examples demonstrating Arbitrium Core features in order of increasing complexity.

## Python Examples

### 1. `quickstart.py` - Your First Tournament

Minimal code to run a tournament. ~2 minutes, ~$0.50.

```bash
python examples/quickstart.py
```

### 2. `single_model.py` - Query Without Tournament

Use Arbitrium for single-model queries. <30 seconds, ~$0.05-0.20.

```bash
python examples/single_model.py
```

### 3. `tournament_basic.py` - Full Tournament Flow

Phases, elimination, cost tracking. ~5-10 minutes, ~$0.50-2.00.

```bash
python examples/tournament_basic.py
```

### 4. `tournament_with_kb.py` - Knowledge Bank in Action

How eliminated models contribute to the final answer. ~5-10 minutes, ~$0.50-2.00.

```bash
python examples/tournament_with_kb.py
```

### 5. `benchmark_comparison.py` - Cost-Benefit Analysis

Compare single model vs all models vs tournament. ~10-20 minutes, ~$1-3.

```bash
python examples/benchmark_comparison.py
```

## Workflow Examples

YAML-based workflow definitions in `workflows/`:

| Workflow | Purpose |
|----------|---------|
| `hello-world.yml` | Basic LLM chain |
| `multi-model-comparison.yml` | Compare models in parallel |
| `prompt-template.yml` | Variable substitution |
| `chain-of-thought.yml` | Sequential reasoning |
| `knowledge-bank.yml` | KB extraction and injection |
| `tournament-elimination.yml` | Multi-round feedback loop |

Run a workflow:

```bash
arbitrium workflow execute examples/workflows/hello-world.yml
```

## Quick Start

```bash
pip install arbitrium-core
cp config.example.yml config.yml
# Edit config.yml with your API keys
python examples/quickstart.py
```
