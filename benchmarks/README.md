# Certamen Framework Benchmarks

Scientifically rigorous evaluation of Certamen Framework vs. single model baselines.

## Quick Start

### 1. Install Dependencies

```bash
# Core dependencies
pip install -e .

# Benchmark dependencies (for BBH/GPQA)
pip install -e .[bench]
```

### 2. Setup and Run

**Set API keys (for cloud models):**

```bash
export OPENAI_API_KEY="..."
export ANTHROPIC_API_KEY="..."
export VERTEX_AI_API_KEY="..."
# Or use 1Password integration (see main README)
```

**For local models (optional, requires Ollama):**

```bash
# Install Ollama: https://ollama.ai
curl -fsSL https://ollama.ai/install.sh | sh

# Pull models
ollama pull llama3 phi3 mistral
```

**Run benchmarks:**

```bash
# Edit config.example.yml to configure your models, then:
python -m benchmarks.standard_benchmarks --benchmark both --config config.example.yml
```

### 3. Benchmark Options

**Standard Benchmarks (Rigorous evaluation):**

```bash
# BBH (Big-Bench Hard)
python -m benchmarks.standard_benchmarks --benchmark bbh --config <your-config.yml>

# GPQA (Graduate-Level Questions)
python -m benchmarks.standard_benchmarks --benchmark gpqa --config <your-config.yml>

# Both
python -m benchmarks.standard_benchmarks --benchmark both --config <your-config.yml>
```

**Micro-Benchmark (Quick validation):**

```bash
python -m benchmarks.micro_benchmark --config <your-config.yml>
```

## Output Files

- `benchmarks/bbh_benchmark_results.json` - Full BBH results with statistics
- `benchmarks/gpqa_benchmark_results.json` - Full GPQA results with statistics
- `benchmarks/micro_benchmark_results.md` - Quick validation report

## Statistical Analysis

Each benchmark automatically computes:

### Primary Metrics

- **Accuracy**: Percentage of correct answers
- **Δ Accuracy**: Difference between Certamen and best baseline
- **95% CI (Bootstrap)**: Confidence interval for accuracy difference (10,000 iterations)

### Statistical Tests

- **McNemar's Test**: Paired test for significance (α = 0.05)
  - Null hypothesis: No difference between Certamen and baseline
  - Reports: b01, b10, χ², p-value

- **Cohen's h**: Effect size for proportion differences
  - Small: 0.2
  - Medium: 0.5
  - Large: 0.8

### Cost-Normalized Metrics

- **Accuracy per Dollar**: acc / total_cost
- **Accuracy per Minute**: acc / (duration_seconds / 60)

## Interpreting Results

### Statistically Significant Win

```
✅ Certamen shows STATISTICALLY SIGNIFICANT improvement (p < 0.05)
```

Requirements:

- McNemar p-value < 0.05
- Δ Accuracy > 0
- This is **publication-ready evidence**

### Numerical Improvement (Not Significant)

```
⚠️ Certamen shows improvement but not statistically significant
```

- Accuracy higher, but p-value ≥ 0.05
- Increase sample size or check if effect is domain-specific

### No Improvement

```
❌ No improvement over best baseline
```

- Investigate: wrong task type, config issues, or hypothesis invalid

## Example Output

```json
{
  "statistics": {
    "best_baseline_model": "claude",
    "best_baseline_accuracy": 68.5,
    "certamen_accuracy": 75.2,
    "delta_accuracy": 6.7,
    "bootstrap_ci_95": {
      "low": 2.3,
      "high": 11.1
    },
    "mcnemar": {
      "b01": 3,
      "b10": 8,
      "chi2": 4.55,
      "p_value": 0.033
    },
    "cohens_h": 0.42,
    "certamen_normalized": {
      "accuracy_per_dollar": 25.1,
      "accuracy_per_minute": 9.4
    }
  }
}
```

**Interpretation:**

- Certamen: 75.2% vs Baseline: 68.5% (+6.7%)
- 95% CI: [2.3%, 11.1%] - excludes zero ✓
- p = 0.033 < 0.05 - statistically significant ✓
- h = 0.42 - medium effect size ✓
- **Conclusion:** Publication-ready win

## Pre-Registration

See `reports/preregistration.md` for:

- Fixed hypotheses
- Methodology
- Analysis plan
- Exclusion criteria

**Important:** Preregister BEFORE running experiments to avoid p-hacking.

## Adding New Baselines

### Self-Consistency (n=5)

Modify `run_single_model_on_benchmark` to generate 5 samples per question and use majority vote.

### Majority Vote Across Models

Collect one answer per model, use voting.

### Certamen w/o Knowledge Bank

Set `knowledge_bank.enabled: false` in config and run separately.

## Publication Checklist

- [ ] Preregistration committed and timestamped
- [ ] Fixed seeds and configs documented
- [ ] All raw predictions saved
- [ ] Statistical tests computed and saved
- [ ] Results reviewed for integrity
- [ ] Code and data released publicly

## Citation

```bibtex
@software{certamen2025,
  author = {Eremeev, Nikolay},
  title = {Certamen: Tournament-Based Multi-Model Synthesis Framework},
  year = {2025},
  version = {0.0.4},
  url = {https://github.com/nikolay-e/certamen-framework}
}
```

## Contact

Issues: <https://github.com/nikolay-e/certamen-framework/issues>
