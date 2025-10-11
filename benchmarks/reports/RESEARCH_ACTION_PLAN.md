# Arbitrium Framework: Research Action Plan

**Goal:** Transform benchmarks into publication-ready scientific evidence

**Timeline:** 7 days (preregistration to paper draft)

---

## ✅ Phase 1: Infrastructure Setup (COMPLETED)

### Statistical Analysis
- [x] Created `benchmarks/stats.py` with McNemar, Bootstrap CI, Cohen's h
- [x] Updated `standard_benchmarks.py` to collect paired predictions
- [x] Added cost and time tracking
- [x] Implemented automatic statistical significance testing

### Documentation
- [x] Created `reports/preregistration.md` (hypothesis, methodology, controls)
- [x] Created `benchmarks/README.md` (usage, interpretation)
- [x] Created `reports/paper_outline.md` (full paper structure)

### Code Quality
- [x] Fixed benchmark imports
- [x] Fixed integration tests
- [x] Added network-free smoke tests
- [x] Moved datasets to optional dependencies

---

## 🔄 Phase 2: Preregistration & Data Collection (NEXT 48 HOURS)

### Day 1: Preregistration

**Morning:**
1. [ ] Review `reports/preregistration.md`
2. [ ] Fix any remaining config issues (API keys, model access)
3. [ ] Commit preregistration with timestamp
4. [ ] Tag commit: `v1.0-preregistration`

**Afternoon:**
5. [ ] Test run BBH with 2 questions (smoke test)
6. [ ] Test run GPQA with 2 questions (smoke test)
7. [ ] Verify JSON output format
8. [ ] Check statistical calculations

### Day 2: Data Collection

**BBH Benchmark (40 questions):**
```bash
python -m benchmarks.standard_benchmarks \
  --benchmark bbh \
  --config config.example.yml
```

**GPQA Benchmark (20 questions):**
```bash
python -m benchmarks.standard_benchmarks \
  --benchmark gpqa \
  --config config.example.yml
```

**Checklist:**
- [ ] BBH results saved to `benchmarks/bbh_benchmark_results.json`
- [ ] GPQA results saved to `benchmarks/gpqa_benchmark_results.json`
- [ ] All predictions logged
- [ ] Cost and time tracked
- [ ] No manual intervention during runs

---

## 📊 Phase 3: Analysis & Validation (DAYS 3-4)

### Day 3: Statistical Analysis

**Load Results:**
```python
import json

with open('benchmarks/bbh_benchmark_results.json') as f:
    bbh = json.load(f)

with open('benchmarks/gpqa_benchmark_results.json') as f:
    gpqa = json.load(f)
```

**Verify:**
1. [ ] McNemar p-values calculated correctly
2. [ ] Bootstrap CIs exclude zero (if significant)
3. [ ] Cohen's h indicates effect size
4. [ ] Cost-normalized metrics computed

**Quality Checks:**
1. [ ] No excluded questions (or documented if any)
2. [ ] All models completed all questions
3. [ ] Predictions are valid (not None/error)
4. [ ] Ground truth matches dataset

### Day 4: Ablation Studies

**Required Ablations:**

**1. Arbitrium w/o Knowledge Bank:**
```bash
# Edit config: knowledge_bank.enabled: false
python -m benchmarks.standard_benchmarks \
  --benchmark bbh \
  --config config.no_kb.yml
```

**2. Self-Consistency (Optional but Recommended):**
- Modify `run_single_model_on_benchmark` to sample n=5 times
- Implement majority vote
- Run on BBH subset (20 questions for speed)

**Checklist:**
- [ ] Ablation results saved
- [ ] Comparisons computed (Arbitrium vs Arbitrium-no-KB)
- [ ] Statistical tests run for ablations

---

## 📝 Phase 4: Report Writing (DAYS 5-6)

### Day 5: Technical Report

**Create `reports/TECHNICAL_REPORT.md`:**

```markdown
# Arbitrium Framework: Technical Report

## Executive Summary
- [Results summary: accuracy gains, p-values, effect sizes]

## Key Findings
1. **BBH Results:**
   - Arbitrium: XX.X% vs Best Single: XX.X%
   - Δ Accuracy: +X.X% (p = 0.0XX, 95% CI [X.X%, X.X%])
   - Cohen's h = 0.XX (medium/large effect)

2. **GPQA Results:**
   - [Same format]

3. **Cost-Normalized:**
   - Arbitrium: XX.X acc/$
   - Best Baseline: XX.X acc/$
   - Gain: +XX%

## Statistical Significance
[Tables with McNemar results, CIs, effect sizes]

## Qualitative Analysis
[Pick 2-3 examples where Arbitrium won and explain WHY]
[Show Knowledge Bank contribution in specific cases]

## Recommendations
[When to use Arbitrium vs single model]
```

**Checklist:**
- [ ] All numbers from JSON outputs
- [ ] Tables formatted
- [ ] Visualizations created (accuracy comparison, cost analysis)
- [ ] Qualitative examples selected

### Day 6: Paper Draft

**Use `reports/paper_outline.md` as template:**

1. [ ] Fill in Abstract with actual numbers
2. [ ] Complete Results section (Tables 1-2)
3. [ ] Write Analysis section with examples
4. [ ] Draft Discussion/Limitations
5. [ ] Compile Appendices (configs, prompts, raw data)

**Target Length:**
- Main paper: 8-10 pages
- Appendices: 5-8 pages
- Total: ~15 pages (arXiv format)

---

## 🚀 Phase 5: Publication (DAY 7)

### Morning: Finalization

1. [ ] Proofread paper draft
2. [ ] Check all references
3. [ ] Verify reproducibility claims
4. [ ] Prepare supplementary materials

### Afternoon: Release

**Code Release:**
1. [ ] Tag release: `v1.0-paper`
2. [ ] Create GitHub Release with:
   - Paper PDF
   - All benchmark results (JSON)
   - Reproduction instructions
   - License (MIT for code, CC-BY for data)

**arXiv Submission:**
1. [ ] Format paper in arXiv template
2. [ ] Upload to arXiv
3. [ ] Get arXiv ID

**Social Announcement:**
```
🎉 New Paper: "Arbitrium: Tournament-Based Multi-Model Synthesis"

📊 Key Results:
- BBH: +X.X% accuracy (p < 0.05)
- GPQA: +X.X% accuracy (p < 0.05)
- Cost-competitive: +XX% acc/$

🔬 Statistically rigorous:
- McNemar tests
- Bootstrap 95% CIs
- Cohen's h effect sizes

📂 Open source + full data release
🔗 arXiv: [link]
💻 Code: github.com/arbitrium-framework/arbitrium

#MachineLearning #NLP #LLMs
```

---

## 🎯 Success Criteria

### Minimum Viable Publication

**Must Have:**
- [ ] p < 0.05 on at least one benchmark
- [ ] Positive Cohen's h > 0.2
- [ ] Cost-normalized improvement
- [ ] Preregistered analysis plan followed
- [ ] Reproducible with documented configs

### Stretch Goals

- [ ] Significant on both BBH and GPQA
- [ ] Large effect size (h > 0.5)
- [ ] Ablation shows KB contributes significantly
- [ ] Conference submission ready

---

## 📋 Critical Path Items

### Blockers to Watch

1. **API Access:** Ensure all models (GPT-5, Claude, Gemini, Grok) are accessible
   - **Mitigation:** Have fallback models ready

2. **API Costs:** Budget $50-100 for full runs
   - **Mitigation:** Start with smaller samples if needed

3. **Statistical Power:** 60 questions may be borderline
   - **Mitigation:** Focus on effect size, qualitative analysis

4. **P-hacking Risk:** Follow preregistration strictly
   - **Mitigation:** Document any deviations, run sensitivity analyses

### Daily Standups

**Each day answer:**
1. What did I complete yesterday?
2. What am I working on today?
3. What blockers do I have?

**Log in:** `reports/progress_log.md`

---

## 🔬 Scientific Integrity Checklist

- [ ] Preregistration committed before data collection
- [ ] Hypotheses not changed post-hoc
- [ ] All exclusions documented with justification
- [ ] Seeds fixed and logged
- [ ] Model versions documented
- [ ] Raw predictions saved
- [ ] No selective reporting (publish all results)
- [ ] Negative results disclosed
- [ ] Limitations clearly stated
- [ ] Code and data released

---

## 📚 References for Methods

**Statistical Methods:**
- McNemar (1947): Note on the sampling error
- Efron (1979): Bootstrap methods
- Cohen (1988): Statistical power analysis

**Related Work:**
- Wang et al. (2022): Self-Consistency
- Irving et al. (2018): AI Safety via Debate
- Du et al. (2023): Multi-agent debate

**Benchmarks:**
- Suzgun et al. (2022): Big-Bench Hard
- Rein et al. (2023): GPQA

---

## 🛠️ Emergency Troubleshooting

### If Results Are Negative (p > 0.05)

**Don't panic. Options:**

1. **Honest reporting:**
   - Report null result
   - Discuss why (task mismatch, insufficient power, etc.)
   - Valuable negative result for community

2. **Exploratory analysis (clearly labeled):**
   - Check if specific task types show gains
   - Look for qualitative improvements
   - Investigate cost-effectiveness even if accuracy similar

3. **Pivot to case study:**
   - Deep dive on 3-5 complex questions
   - Expert evaluation (blind)
   - Focus on "when and why" Arbitrium helps

### If Costs Explode

**Mitigation:**

1. Use smaller subsets (10-20 questions)
2. Focus on one benchmark (BBH or GPQA)
3. Run ablations on subset only
4. Document resource constraints

### If Time Runs Out

**Minimum viable:**

1. Complete one benchmark (BBH preferred)
2. Basic statistical analysis
3. Technical report only (skip full paper)
4. Release results + plan for full study

---

## 🎓 Learning from This Process

**Meta-goal:** Establish repeatable scientific workflow for LLM research

**Reusable artifacts:**
- Statistical analysis pipeline
- Preregistration template
- Benchmark harness
- Paper outline

**For next iteration:**
- Larger sample sizes
- More diverse tasks
- Human evaluation protocol
- Real-world deployment study

---

## ✅ Final Deliverables

1. **Preregistration:** `reports/preregistration.md` (timestamped commit)
2. **Results:** JSON files with full data + statistics
3. **Technical Report:** `reports/TECHNICAL_REPORT.md`
4. **Paper Draft:** `reports/paper_draft.pdf`
5. **Code Release:** Tagged GitHub release with reproduction instructions
6. **arXiv Preprint:** Submitted and ID obtained

**Timeline:** 7 days from preregistration to publication

**Budget:** $50-100 for API costs

**Risk:** Medium (depends on API access and statistical power)

**Upside:** High-impact evidence for Arbitrium Framework value

---

## 🚦 Go/No-Go Decision Points

### End of Day 1 (Preregistration)
- ✅ **GO:** Preregistration committed, configs tested
- ❌ **NO-GO:** API access issues, pivot to local models

### End of Day 2 (Data Collection)
- ✅ **GO:** Results collected, no major failures
- ❌ **NO-GO:** Extend collection or reduce scope

### End of Day 4 (Analysis)
- ✅ **GO:** Statistical significance achieved
- ⚠️ **CAUTION:** Null result → pivot to qualitative
- ❌ **NO-GO:** Data quality issues → re-run

---

**Start Date:** [Today]
**Target Completion:** [Today + 7 days]
**Lead:** [Your Name]
**Status:** Phase 1 Complete ✅, Ready for Phase 2
