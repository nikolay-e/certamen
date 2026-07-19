# Examples

Examples demonstrating Certamen Core features.

## Python Example

### `single_model.py` - Ad-hoc Model Probing

Query one model or fan out to all configured models without a tournament,
using the `Certamen` class's `run_single_model` / `run_all_models` API.
<30 seconds, ~$0.05-0.20.

```bash
python examples/single_model.py
```

> Full tournament execution is not a `Certamen` method — it runs through the
> workflow executor (see below), not in-process Python.

## Tournaments (Workflow Executor)

Tournaments run via the CLI against a config, or by executing a workflow
definition directly. The `tournament-elimination.yml` workflow is the
multi-round elimination flow (phases, elimination, knowledge bank).

```bash
# Run a tournament from a config file
certamen --config config.yml

# Or execute the elimination workflow directly
certamen workflow execute examples/workflows/tournament-elimination.yml
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
certamen workflow execute examples/workflows/hello-world.yml
```

## Quick Start

```bash
pip install certamen-core
cp config.example.yml config.yml
# Edit config.yml with your API keys
python examples/single_model.py
```
