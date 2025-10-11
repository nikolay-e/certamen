# Publication Readiness Checklist

**Use this checklist before running experiments and before publication.**

---

## Phase 1: Pre-Experiment (Before First Run)

### Preregistration
- [ ] Hypotheses fixed (H1-H4) with clear success criteria
- [ ] Sample size justified (power analysis for Δacc ≥ 5pp)
- [ ] Statistical plan locked (McNemar, Bootstrap CI, Cohen's h, α=0.05)
- [ ] Exclusion criteria defined (duplicates, API errors, ambiguous)
- [ ] Ablations specified (Arbitrium -KB, Self-Consistency, Majority Vote, Cost-Matched)
- [ ] Multiple comparisons plan (primary: no correction; exploratory: FDR q=0.10)
- [ ] Preregistration committed to git with tag: `git tag v1.0-preregistration`

### Configuration Lock
- [ ] Models and versions documented (`config.example.yml` frozen)
- [ ] Seeds fixed: [42, 123, 789]
- [ ] Temperature/max_tokens/retry logic documented
- [ ] Answer parsing rules defined (regex: `\b([A-D])\b`, case-insensitive)
- [ ] Compute parity plan (cost-matched mode token caps)
- [ ] Commit hash recorded: `git rev-parse HEAD`

### Infrastructure Ready
- [ ] Benchmarks run without errors: `python -m benchmarks.standard_benchmarks --help`
- [ ] Stats module tested: `python -c "from benchmarks.stats import mcnemar; print(mcnemar(3,8))"`
- [ ] CSV export working (per-question predictions)
- [ ] API keys configured (1Password or env vars)
- [ ] Budget limit set (stop if exceeded: ${fill})

---

## Phase 2: During Experiment

### Data Collection
- [ ] Run 3 seeds sequentially: 42 → 123 → 789
- [ ] Log model versions/timestamps in JSON
- [ ] Track token counts (input + output) per question
- [ ] Record API errors/retries (separate log)
- [ ] **NO PEEKING** at accuracy until all N complete

### Quality Control
- [ ] Verify answer parsing (spot-check 10 random questions)
- [ ] Check for duplicates (flag any exact matches)
- [ ] Monitor API error rate (<20% threshold)
- [ ] Ensure same question order for all systems (shuffled with seed=42)

---

## Phase 3: Analysis

### Statistical Tests (Confirmatory)
- [ ] Compute accuracy for all systems
- [ ] McNemar exact test: Arbitrium vs Best Baseline (p-value, χ²)
- [ ] Bootstrap 95% CI for Δacc (10,000 iterations, seed=42)
- [ ] Cohen's h effect size (interpret: 0.2/0.5/0.8)
- [ ] Decision: Reject H0 if p<0.05 AND CI excludes 0 AND h≥0.2

### Secondary Analyses (Exploratory)
- [ ] Cost-normalized metrics (Acc/$, Acc/min)
- [ ] Win rate (% questions where Arbitrium > baseline)
- [ ] Stratified by benchmark (BBH vs GPQA)
- [ ] Apply Benjamini-Hochberg FDR (q=0.10) for multiple exploratory tests

### Ablations
- [ ] Arbitrium -KB (isolate Knowledge Bank contribution)
- [ ] Self-Consistency (n=5) for strongest baseline
- [ ] Majority Vote ensemble
- [ ] Cost-Matched Arbitrium (token cap = best_baseline)

### Error Analysis
- [ ] Taxonomy: Logic, Parsing, Arithmetic, Distraction, Hallucination, Ambiguous
- [ ] Top-10 hardest questions (all models fail)
- [ ] Top-5 Arbitrium wins (correct when all baselines fail)
- [ ] Top-5 Arbitrium losses (wrong when baseline correct)

---

## Phase 4: Reporting

### Figures & Tables (Publication-Ready)
- [ ] **Figure 1:** Accuracy with 95% CI error bars (BBH, GPQA, Combined)
- [ ] **Figure 2:** Pareto frontier (Cost vs Accuracy scatter plot)
- [ ] **Figure 3:** McNemar waterfall (b10 vs b01 bars)
- [ ] **Table 1:** Main results (Accuracy, Δacc, CI, p, h)
- [ ] **Table 2:** Cost-normalized metrics (Acc/$, Acc/min)
- [ ] **Table 3:** Ablations (component contributions)
- [ ] **Table 4:** Error taxonomy (counts and percentages)

### Results Document
- [ ] Fill `reports/results_template.md` with actual numbers
- [ ] Executive summary (1-sentence main finding)
- [ ] Hypothesis outcomes (H1-H4: Supported/Not Supported)
- [ ] Interpretation (when does Arbitrium help?)
- [ ] Practical recommendations (use cases)
- [ ] Limitations (sample size, domain, API drift)
- [ ] Future work (expand domains, open models, active learning)

### Reproducibility Artifacts
- [ ] Code: Commit hash in results
- [ ] Data: `{benchmark}_benchmark_results.json` + `{benchmark}_per_question.csv`
- [ ] Config: `config.example.yml` (exact version used)
- [ ] Seeds: [42, 123, 789] documented
- [ ] Model versions: Log from API responses (timestamps, IDs)
- [ ] Exclusion log: List any dropped questions with reasons
- [ ] Runtime/cost table: Actual minutes and dollars spent

---

## Phase 5: Publication

### Open Release
- [ ] GitHub repo public: https://github.com/arbitrium-framework/arbitrium
- [ ] Tag release: `git tag v1.0-results && git push --tags`
- [ ] Zenodo upload: JSON + CSV + logs → get DOI
- [ ] Licenses: MIT (code), CC-BY-4.0 (data/report)

### Paper Submission
- [ ] arXiv preprint: "Arbitrium: Tournament-Based Multi-Model Synthesis Improves Reasoning Under Fixed Compute Budgets"
- [ ] Abstract: Hypotheses, methods, results (p/CI/h), conclusions
- [ ] Introduction: Gap (single-model variance, cost-accuracy tradeoff)
- [ ] Methods: Tournament design, KB, datasets, controls
- [ ] Results: Tables/figures from Phase 4
- [ ] Discussion: When Arbitrium helps, limitations, future work
- [ ] Appendices: Full per-question data, model versions, exclusions

### Ethics & Transparency
- [ ] Dataset licenses acknowledged (BBH: Apache 2.0, GPQA: MIT)
- [ ] API ToS compliance confirmed
- [ ] No selective reporting (publish positive/negative/null)
- [ ] Preregistration deviations (if any) clearly marked
- [ ] No HARKing (hypotheses were fixed before data)

---

## Phase 6: Communication (Optional)

### One-Pager
- [ ] Key figure: Accuracy bars with CI + cost-matched result
- [ ] One-sentence finding: "+X pp on BBH+GPQA, p<0.05, h=Y, persists under cost cap"
- [ ] Practical takeaway: "Use Arbitrium when {condition}; use single model when {condition}"

### Social Media / Blog Post
- [ ] Thread: Problem → Method → Results → Impact
- [ ] Visual: Pareto frontier or McNemar waterfall
- [ ] Link: arXiv preprint + GitHub repo

### Demo / Notebook
- [ ] Jupyter notebook: Run micro-benchmark, visualize results
- [ ] Colab link for easy replication

---

## Checklist Summary (Before Each Phase)

### ✅ Before Experiment:
Preregistration locked, config frozen, infrastructure tested, budget set, commit tagged

### ✅ During Experiment:
3 seeds run, logs captured, no peeking, quality checks passed

### ✅ Before Analysis:
All N complete, exclusions logged, data files verified

### ✅ Before Publication:
Figures done, results template filled, artifacts released, paper drafted

### ✅ Before Submission:
Ethics reviewed, licenses correct, reproducibility verified, transparency confirmed

---

## Quick Validation Commands

```bash
# Pre-experiment checks
python -c "from benchmarks.stats import mcnemar; chi2, p = mcnemar(3,8); print(f'McNemar test: χ²={chi2:.3f}, p={p:.4f}')"
python -m benchmarks.standard_benchmarks --help
git rev-parse HEAD  # Record commit

# During experiment
python -m benchmarks.standard_benchmarks --benchmark bbh --config config.example.yml
python -m benchmarks.standard_benchmarks --benchmark gpqa --config config.example.yml

# Post-experiment verification
ls benchmarks/*_benchmark_results.json benchmarks/*_per_question.csv
python -c "import json; print(json.load(open('benchmarks/bbh_benchmark_results.json'))['statistics'])"

# Release
git tag v1.0-results
git push --tags
# Upload to Zenodo, get DOI
```

---

**Use this checklist to ensure rigorous, reproducible, publication-ready results.**
