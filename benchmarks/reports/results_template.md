# Arbitrium Benchmark Results

**Date:** {YYYY-MM-DD}
**Preregistration:** [reports/preregistration.md](preregistration.md)
**Commit:** {git rev-parse HEAD}

---

## Executive Summary

**Main Finding:** {One sentence: Arbitrium achieved X% accuracy vs Y% for best baseline (p={p-value}, Δacc={delta}pp, h={cohens_h})}

**Key Results:**
- ✅/❌ H1 (Accuracy): {supported/not supported}
- ✅/❌ H2 (Cost-Efficiency): {supported/not supported}
- ✅/❌ H3 (Robustness): {supported/not supported}

**Practical Impact:** {1-2 sentences on when/where to use Arbitrium}

---

## 1. Primary Results (H1: Accuracy)

### 1.1 Main Comparison: Arbitrium vs Best Baseline

| Metric | Best Baseline | Arbitrium | Δ (Arb - Base) |
|--------|--------------|-----------|----------------|
| **Accuracy** | {X.X}% | {Y.Y}% | **+{Z.Z}pp** |
| **95% CI** | — | — | [{low}, {high}] |
| **McNemar χ²** | — | — | {chi2} |
| **McNemar p-value** | — | — | **{p_value}** |
| **Cohen's h** | — | — | **{h_value}** |

**Statistical Interpretation:**
- {p < 0.05 → "Statistically significant" OR p ≥ 0.05 → "Not statistically significant"}
- {CI excludes 0 → "Robust effect" OR CI includes 0 → "Effect uncertain"}
- {h interpretation: "Small/Medium/Large effect size"}

**McNemar Contingency:**
- b01 (baseline ✓, Arbitrium ✗): {count}
- b10 (baseline ✗, Arbitrium ✓): {count}
- Both correct: {count}
- Both wrong: {count}

**Decision:** {Reject H0 / Fail to reject H0}

### 1.2 Breakdown by Benchmark

| Benchmark | N | Best Baseline | Arbitrium | Δacc | p-value | h |
|-----------|---|--------------|-----------|------|---------|---|
| **BBH** | {n_bbh} | {X}% | {Y}% | +{Z}pp | {p} | {h} |
| **GPQA** | {n_gpqa} | {X}% | {Y}% | +{Z}pp | {p} | {h} |
| **Combined** | {n_total} | {X}% | {Y}% | +{Z}pp | **{p}** | **{h}** |

### 1.3 Win Rate Analysis

**Per-Question Wins:**
- Arbitrium > Baseline: {X} questions ({Y}%)
- Arbitrium = Baseline: {X} questions ({Y}%)
- Arbitrium < Baseline: {X} questions ({Y}%)

---

## 2. Cost-Efficiency Results (H2)

### 2.1 Cost-Normalized Metrics

| System | Accuracy | Cost (USD) | Time (min) | Acc/$ | Acc/min |
|--------|----------|-----------|-----------|-------|---------|
| Best Baseline | {X}% | ${Y} | {Z} | {A} | {B} |
| Arbitrium (uncapped) | {X}% | ${Y} | {Z} | {A} | {B} |
| **Arbitrium (cost-matched)** | {X}% | ${Y} | {Z} | **{A}** | **{B}** |

**H2 Result:** {Arbitrium cost-matched ≥ baseline → "Supported" / < baseline → "Not supported"}

**Interpretation:** {e.g., "Arbitrium maintains +3pp advantage even when capped to same token budget"}

### 2.2 Pareto Frontier (Accuracy vs Cost)

```
Accuracy
   │
{Y}│           ● Arb (uncapped)
   │       ● Arb (cost-matched)
   │     ● Baseline
   │   ● Other models
   │
   └─────────────────── Cost ($)
```

{Describe: "Arbitrium dominates/is competitive/trails on cost-accuracy tradeoff"}

---

## 3. Robustness (H3)

### 3.1 Consistency Across Runs (3 Seeds)

| Seed | Best Baseline Acc | Arbitrium Acc | Δacc |
|------|------------------|---------------|------|
| 42 | {X}% | {Y}% | +{Z}pp |
| 123 | {X}% | {Y}% | +{Z}pp |
| 789 | {X}% | {Y}% | +{Z}pp |
| **Mean ± SD** | **{μ ± σ}%** | **{μ ± σ}%** | **+{μ ± σ}pp** |

**H3a Result:** {Consistent → "Supported" / High variance → "Not supported"}

### 3.2 Domain Robustness (BBH vs GPQA)

**BBH (Reasoning):**
- Δacc: +{X}pp (p={p}, h={h})

**GPQA (Science):**
- Δacc: +{X}pp (p={p}, h={h})

**H3b Result:** {Both positive → "Supported" / Mixed → "Partial support"}

---

## 4. Ablation Studies

### 4.1 Component Contribution

| System | Accuracy | Δ vs Full Arbitrium |
|--------|----------|---------------------|
| **Arbitrium (Full)** | {X}% | — |
| Arbitrium -KB | {Y}% | {Z}pp |
| Self-Consistency (n=5) | {Y}% | {Z}pp |
| Majority Vote | {Y}% | {Z}pp |
| Random Model | {Y}% | {Z}pp |

**Key Findings:**
- Knowledge Bank contributes: {+/- X}pp
- Tournament > Self-Consistency by: {+/- X}pp
- Tournament > Majority Vote by: {+/- X}pp

### 4.2 Strongest Baseline Comparison

**Best Alternative Method:** {e.g., "Self-Consistency (gpt, n=5)"}
- Accuracy: {X}%
- Arbitrium vs This: Δacc = +{Y}pp (p={p}, h={h})

---

## 5. Error Analysis

### 5.1 Error Taxonomy

| Error Type | Count | % of Total Errors |
|-----------|-------|-------------------|
| Logic/reasoning flaw | {X} | {Y}% |
| Answer parsing failure | {X} | {Y}% |
| Arithmetic error | {X} | {Y}% |
| Distracted by irrelevant | {X} | {Y}% |
| Hallucination | {X} | {Y}% |
| Ambiguous question | {X} | {Y}% |

### 5.2 Top-10 Hardest Questions

**Questions where all models failed:**
1. {qid} ({benchmark}): "{question snippet...}"
2. ...

### 5.3 Biggest Arbitrium Wins

**Questions where Arbitrium ✓, all baselines ✗:**
1. {qid} ({benchmark}): "{question snippet...}"
   - Why Arbitrium won: {e.g., "Tournament caught logic flaw in early round"}

### 5.4 Biggest Arbitrium Losses

**Questions where Arbitrium ✗, baseline ✓:**
1. {qid} ({benchmark}): "{question snippet...}"
   - Why Arbitrium lost: {e.g., "Over-complicated reasoning, lost original answer"}

---

## 6. Publication-Ready Figures

### Figure 1: Main Result (Accuracy with CI)

```
Accuracy (%)
    80│              ┌──┐
      │              │██│ Arbitrium
    70│     ┌──┐     │██│
      │     │██│     └──┘
    60│     │██│ Baseline
      │     └──┘
      └──────────────────
         BBH  GPQA  Combined
```

Error bars = 95% Bootstrap CI
**Caption:** Arbitrium achieves {X}pp gain on combined benchmarks (p={p}, h={h})

### Figure 2: Pareto Frontier (Cost vs Accuracy)

```
    {Plot: Accuracy on Y, Cost on X, points for each system}
```

**Caption:** Arbitrium {dominates/competes with} baselines even under cost-matched budget

### Figure 3: McNemar Waterfall

```
    b10 (Arb wins): ████████ {count}
    b01 (Base wins): ███ {count}
                     ─────────────
                     Δ = {diff} (p={p})
```

**Caption:** Arbitrium improves on {X} questions where baseline failed, only loses {Y}

---

## 7. Interpretation & Discussion

### 7.1 Hypothesis Outcomes

**H1 (Accuracy):** {Supported/Not Supported}
- Evidence: {p-value, CI, effect size}
- Strength: {Strong/Moderate/Weak/Null}

**H2 (Cost-Efficiency):** {Supported/Not Supported}
- Evidence: {Cost-matched Δacc}
- Interpretation: {e.g., "Gains persist even with budget cap"}

**H3 (Robustness):** {Supported/Not Supported}
- Evidence: {Consistency across runs/domains}

**H4 (Components):** {Exploratory findings}
- KB contribution: {+/- X}pp
- Tournament vs SC: {+/- X}pp

### 7.2 When Does Arbitrium Help?

**Works Best On:**
- {e.g., "Multi-step reasoning requiring self-correction"}
- {e.g., "Questions where models disagree (high entropy)"}

**Less Effective On:**
- {e.g., "Simple factual recall"}
- {e.g., "Questions where all models fail (systemic knowledge gap)"}

### 7.3 Practical Recommendations

**Use Arbitrium When:**
- {Condition 1, e.g., "High-stakes decisions justify cost"}
- {Condition 2, e.g., "Multiple models available"}
- {Condition 3, e.g., "Reasoning over recall"}

**Use Single Model When:**
- {e.g., "Budget-constrained, fast responses needed"}
- {e.g., "Simple queries with low variance"}

---

## 8. Limitations & Future Work

### 8.1 Study Limitations

1. **Sample Size:** {e.g., "N=600 total, may miss small effects"}
2. **Domain:** {e.g., "Text reasoning only, no code/math"}
3. **Models:** {e.g., "4 frontier models, no open-source"}
4. **API Drift:** {e.g., "Results may vary across API versions"}

### 8.2 Future Directions

1. **Expand Domains:** {e.g., "Code (HumanEval), Math (GSM8K), Multimodal"}
2. **Ablate Components:** {e.g., "Different tournament rules, KB algorithms"}
3. **Open Models:** {e.g., "Llama 3, Mixtral for cost-sensitive settings"}
4. **Active Learning:** {e.g., "Adaptively select questions for tournament"}

---

## 9. Reproducibility

### 9.1 Artifacts Released

- ✅ Code: https://github.com/arbitrium-framework/arbitrium (commit: {hash})
- ✅ Data: Zenodo DOI {doi} (JSON, CSV, logs)
- ✅ Config: `config.example.yml` (frozen)
- ✅ Seeds: [42, 123, 789]

### 9.2 Reproduction Command

```bash
git clone https://github.com/arbitrium-framework/arbitrium.git
git checkout {commit_hash}
pip install -e .[bench]
python -m benchmarks.standard_benchmarks --benchmark both --config config.example.yml
```

**Expected Output:**
- `benchmarks/bbh_benchmark_results.json`
- `benchmarks/gpqa_benchmark_results.json`
- `benchmarks/*_per_question.csv`

### 9.3 Actual Runtime & Cost

| Benchmark | Questions | Runtime | Cost | API Calls |
|-----------|-----------|---------|------|-----------|
| BBH | {n} | {X} min | ${Y} | {Z} |
| GPQA | {n} | {X} min | ${Y} | {Z} |
| **Total** | {n} | **{X} min** | **${Y}** | **{Z}** |

**Machine:** {OS, Python version, location for API latency}

---

## 10. Conclusion

{3-4 sentence summary:}
- Main finding (Δacc, p-value, effect size)
- Cost-efficiency result
- Robustness across domains/runs
- Practical takeaway (when to use)

**Status:** {e.g., "Submitted to arXiv YYYY-MM-DD"}

---

## Appendices

### A. Full Per-Question Results

See: `benchmarks/bbh_per_question.csv`, `benchmarks/gpqa_per_question.csv`

### B. Model Versions

| Model | Provider | Version/ID | Date |
|-------|----------|-----------|------|
| grok | xAI | {version} | {date} |
| gpt | OpenAI | {version} | {date} |
| claude | Anthropic | {version} | {date} |
| gemini | Google | {version} | {date} |

### C. Exclusions Log

{If any questions excluded, list with reason:}
- BBH question {id}: {reason}
- ...

**Total excluded:** {X} / {N} ({Y}%)

---

**Authors:** Nikolay Eremeev
**Contact:** nikolay.eremeev@outlook.com
**License:** CC-BY-4.0 (report), MIT (code)
