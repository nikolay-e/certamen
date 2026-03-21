# Benchmark Results — Certamen vs Single Baselines

**Date:** {YYYY-MM-DD}
**Benchmark:** BBH / GPQA / Combined
**Config:** `config.example.yml` (commit: {hash})
**Environment:** {OS/CPU/GPU/RAM}, Python {X.Y}, datasets {Z}, providers: {list}

---

## 1. Summary Results

| Metric | Best Single | Certamen | Δ (Arb - Base) |
|--------|------------|-----------|----------------|
| **Accuracy** | {XX.X}% | {YY.Y}% | **+{Z.Z} pp** |
| **95% CI** | — | — | [{L}, {H}] |
| **McNemar χ²** | — | — | {K} |
| **McNemar p-value** | — | — | **{P}** |
| **Cohen's h** | — | — | **{H}** |
| **Acc/$** | {B1} | {A1} | {Δ1} |
| **Acc/min** | {B2} | {A2} | {Δ2} |

**McNemar Contingency:**

- b01 (baseline ✓, Certamen ✗): {B01}
- b10 (baseline ✗, Certamen ✓): {B10}
- Both correct: {X}
- Both wrong: {Y}

---

## 2. Verdict

**Statistical Significance:** {Significant / Not Significant} (p = {P}, α = 0.05)

**Effect Size:** {Small / Medium / Large} (Cohen's h = {H})

**Interpretation:** {Based on preregistered criteria from README.md:}

- ✅/❌ **Significant:** p < 0.05 and |h| > 0.2
- ✅/❌ **Improvement:** Δacc > 0 and 95% CI excludes 0
- ✅/❌ **Cost-Competitive:** Acc/$ (Certamen) ≥ Acc/$ (Baseline)

**Overall:** {Significant Improvement / Improvement / No Improvement / Regression}

---

## 3. Cost-Matched Mode (if applicable)

**Budget Cap:** {e.g., "Same total tokens as best baseline"}

| System | Accuracy | Tokens | Cost | Acc/$ |
|--------|----------|--------|------|-------|
| Best Baseline | {X}% | {T} | ${C} | {R1} |
| Certamen (uncapped) | {Y}% | {T2} | ${C2} | {R2} |
| **Certamen (capped)** | **{Z}%** | **{T}** | **${C}** | **{R3}** |

**Result:** {e.g., "Certamen maintains +3pp advantage even under token cap"}

---

## 4. Breakdown by Task Type

**Where Certamen Wins:**

- {Task type 1}: +{X}pp (e.g., "Multi-step reasoning: +5.2pp")
- {Task type 2}: +{Y}pp

**Where Certamen Loses:**

- {Task type 3}: {-X}pp (e.g., "Simple factual recall: -1.0pp")

**Key Observation:** {1-2 sentences on pattern}

---

## 5. Stability Across Runs (3 seeds)

| Seed | Best Baseline | Certamen | Δacc |
|------|--------------|-----------|------|
| 42   | {X}% | {Y}% | +{Z}pp |
| 123  | {X}% | {Y}% | +{Z}pp |
| 789  | {X}% | {Y}% | +{Z}pp |
| **Mean ± SD** | **{μ ± σ}%** | **{μ ± σ}%** | **+{μ ± σ}pp** |

**Robustness:** {Stable / Variable} (SD = {σ})

---

## 6. Artifacts & Raw Data

**GitHub Actions (Nightly Run):**

- Date: {YYYY-MM-DD}
- Workflow: [Link to nightly workflow run]
- Artifacts:
  - `bbh_benchmark_results.json`
  - `gpqa_benchmark_results.json`
  - `bbh_per_question.csv`
  - `gpqa_per_question.csv`

**Dataset Versions:**

- See [DATASET_VERSIONS.md](DATASET_VERSIONS.md)

**Reproducibility:**

```bash
git checkout {commit_hash}
pip install -e .[bench]
python -m benchmarks.standard_benchmarks --benchmark both --config config.example.yml
```

---

## 7. Figures (attach or link)

### Figure 1: Accuracy with 95% CI

{Attach bar chart image or link to artifact}

### Figure 2: Pareto Frontier (Acc vs Cost)

{Attach scatter plot image or link to artifact}

---

## 8. Deviations from Preregistration

**Any changes from [preregistration.md](preregistration.md):**

- {List any deviations with justification, or "None"}

**Post-hoc Analyses (exploratory, not confirmatory):**

- {e.g., "Stratified by question difficulty (not prespecified)"}

---

## 9. Next Steps

- [ ] Archive results to Zenodo (get DOI)
- [ ] Update README.md with key findings
- [ ] Draft technical report / arXiv preprint
- [ ] Share with community (blog post, Twitter thread)

---

**Analyst:** {Name}
**Reviewed by:** {Name, if applicable}
**Contact:** <nikolay.eremeev@outlook.com>
