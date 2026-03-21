# Preregistration: Certamen Tournament-Based Multi-Model Synthesis

**Date:** 2025-10-05
**Version:** 2.0
**Status:** 🔒 Locked (do not modify after first run)
**Study Lead:** Nikolay Eremeev

---

## Note on Model Availability

**⚠️ Important:** This research uses advanced model versions that may not be publicly available:

- `gpt-5` - Advanced OpenAI model (beta/research access)
- `claude-sonnet-4-5-20250929` - Advanced Anthropic model (beta/research access)
- `gemini-2.5-pro` - Advanced Google model (beta/research access)
- `xai/grok-4-latest` - xAI Grok model

**For Public Reproduction:**
To replicate this study with publicly available models, use `config.public.yml` with free local models via Ollama:

- `phi3`, `phi4-mini`, `gemma3:4b`, `qwen3:4b`

While results may differ due to model capability differences, the tournament methodology and statistical analysis remain identical.

---

## 1. Research Question

Does a tournament-based multi-model synthesis system (Certamen) improve reasoning accuracy compared to single-model baselines under fixed compute budgets?

---

## 2. Hypotheses

### H1 (Primary): Accuracy Improvement

**Hypothesis:** Certamen achieves higher accuracy than the best single model baseline on BBH and GPQA benchmarks.

**Success Criteria:**

- Δaccuracy > 0 (Certamen - best_baseline)
- p < 0.05 (McNemar's exact test, two-sided)
- 95% Bootstrap CI excludes 0
- Cohen's h ≥ 0.2 (small-to-medium effect size)

**Null Hypothesis (H0):** No difference between Certamen and best baseline (b01 = b10 in McNemar)

### H2: Cost-Efficiency

**Hypothesis:** Certamen maintains accuracy advantage under equal compute budget (cost-matched mode).

**Metrics:**

- Accuracy/$ (accuracy points per dollar spent)
- Accuracy/min (accuracy points per minute)
- Cost-matched accuracy (same total tokens/calls as baseline)

### H3: Robustness

**Hypothesis:** Performance gain persists across:

- Different benchmark domains (BBH reasoning tasks vs GPQA science)
- Multiple independent runs (3 seeds: 42, 123, 789)
- Different question difficulty levels

### H4 (Exploratory): Component Contribution

- Knowledge Bank adds value (Certamen > Certamen -KB)
- Self-Consistency (n=5) is strong baseline to beat
- Majority Vote across models is weaker than tournament

---

## 3. Datasets

### 3.1 Primary Benchmarks

**BBH (Big-Bench Hard):**

- Challenging reasoning tasks (full set or subset)
- Closed-choice format
- Target: ≥300 questions for adequate power

**GPQA (Graduate-Level Questions):**

- Diamond subset (hardest tier)
- Expert-level science questions
- Target: ≥100 questions

**Power Analysis:**

- Minimum detectable effect: Δacc ≥ 5 percentage points
- Target power: 80% at α=0.05
- Required n ≈ 300-600 per benchmark (paired test)

### 3.2 Exclusion Criteria (A Priori)

Questions excluded if:

1. Duplicate/near-duplicate in dataset (exact string match)
2. Parsing error (malformed JSON/invalid format)
3. API failures after 3 retries → log separately, exclude from accuracy calculation
4. Ground truth ambiguous (flagged by manual review if applicable)

**Pre-commitment:** All exclusions logged with reason before analysis

---

## 4. Models & Configuration

### 4.1 Models Under Test

**Active Models (config.example.yml):**

- **grok:** xai/grok-beta
- **gpt:** gpt-4o
- **claude:** claude-sonnet-4-5-20250929
- **gemini:** vertex_ai/gemini-2.0-flash-exp

**Version Lock (fill before run):**

- API versions: {check litellm.get_model_info()}
- Snapshot date: {YYYY-MM-DD}
- Model hashes/IDs: {log from API responses}

### 4.2 Fixed Parameters

**From config.example.yml:**

```yaml
features:
  deterministic_mode: true
  judge_model: null
  knowledge_bank_model: "leader"
  llm_compression: false
  compression_model: "claude"
  max_insights_to_inject: 5

certamen:
  max_rounds: 50
  min_rounds_before_stopping: 3
  entropy_threshold: 0.01
  similarity_threshold: 0.8
  disagreement_threshold: 0.3
```

**Temperature:** 0.0 (deterministic) or fixed seed if temp>0
**Max tokens:** Model default (no artificial cap in uncapped mode)
**Retry logic:** 3 attempts with exponential backoff

---

## 5. Experimental Conditions

### 5.1 Baselines

1. **Best Single Model:** Highest accuracy individual model (identified post-hoc)
2. **Self-Consistency (n=5):** Each model samples 5 times, majority vote
3. **Majority Vote Ensemble:** One response per model, vote across all
4. **Random Model Selection:** Random choice per question (control)
5. **Certamen -KB:** Tournament without Knowledge Bank (ablation)
6. **Certamen (Full):** Tournament + Knowledge Bank

### 5.2 Compute Parity Controls

**Cost-Matched Mode:**

- Track total tokens (input + output) for best baseline
- Cap Certamen tokens to same budget per question
- Report both uncapped and capped results
- Token accounting: log every API call, sum across tournament

**Fair Comparison:**

- Same prompts/instructions for all systems
- No cherry-picking of hyperparameters post-hoc
- Baselines use strong CoT prompts (same quality as Certamen)

### 5.3 Stability & Reproducibility

**Multiple Runs:**

- 3 independent runs: seeds [42, 123, 789]
- For deterministic models (temp=0): results should be identical
- For stochastic: average metrics with ±SD/CI

**Answer Parsing Protocol:**

- Extract from "Answer:" marker or last line
- Case-insensitive letter matching (A/B/C/D)
- Regex: `\b([A-D])\b` (word boundary)
- Unit tests: `tests/test_answer_parser.py` (validates edge cases)
- No manual fixes post-run

**Question Order:**

- Shuffle with fixed seed (42)
- Same order for all systems (paired comparison)
- Log original indices in CSV

---

## 6. Statistical Analysis Plan

### 6.1 Primary Analysis (H1)

**Paired Tests (Certamen vs Best Baseline):**

1. **McNemar's Exact Test:**
   - b01 = baseline correct, Certamen wrong
   - b10 = baseline wrong, Certamen correct
   - Null: b01 = b10 (no difference)
   - Two-sided p-value via exact binomial (see `benchmarks/stats.py`)
   - α = 0.05

2. **Bootstrap 95% CI for Δaccuracy:**
   - Paired bootstrap (10,000 iterations, seed=42)
   - Resample indices with replacement
   - Calculate Δacc for each resample
   - CI = [2.5th percentile, 97.5th percentile]

3. **Cohen's h Effect Size:**
   - Formula: `2 * (arcsin(√p1) - arcsin(√p2))`
   - Interpretation: 0.2=small, 0.5=medium, 0.8=large
   - Report even if p ≥ 0.05 (magnitude matters)

**Decision Rule:**

- Reject H0 if: p < 0.05 AND 95% CI excludes 0 AND h > 0.2

### 6.2 Secondary Analyses (Exploratory)

- **Win Rate:** % questions where Certamen > baseline
- **Cost-Normalized:**
  - Accuracy/$ (acc points per dollar)
  - Accuracy/min (acc points per minute)
- **Stratified by Benchmark:** BBH vs GPQA (domain differences)
- **Difficulty Stratification:** If metadata available (easy/medium/hard)

**Multiple Comparisons Correction:**

- Primary (H1): No correction (single prespecified hypothesis)
- Exploratory: Benjamini-Hochberg FDR at q=0.10
- Ablations: No correction (mechanistic, not confirmatory)

### 6.3 Error Analysis (Qualitative)

**Taxonomy (code each error):**

1. Logic/reasoning flaw
2. Answer parsing failure
3. Arithmetic/calculation error
4. Distracted by irrelevant info
5. Hallucination/unfounded claim
6. Ambiguous question (dataset issue)

**Top-10 Analyses:**

- Hardest questions (most models fail)
- Biggest Certamen wins (correct when all baselines wrong)
- Biggest Certamen losses (wrong when baseline correct)

---

## 7. Ablation Studies

**Required Ablations:**

1. **Certamen -KB** (knowledge_bank_model: null)
   - Isolate contribution of Knowledge Bank
   - Compare: Certamen vs Certamen -KB

2. **Self-Consistency n=5** (strong baseline)
   - Each model samples 5 times, majority vote
   - Shows value vs simple ensembling

3. **Majority Vote Ensemble**
   - One response per model, vote
   - Cheaper than tournament, fair comparison

4. **Cost-Matched Certamen**
   - Cap tokens to match best_baseline
   - Most conservative test of H2

**Comparison Metrics:**

- Same as primary (accuracy, McNemar, CI, h)
- Identify which components drive gains

---

## 8. Reproducibility & Release

### 8.1 Artifacts to Release

**Code & Config:**

- Repository: <https://github.com/nikolay-e/certamen-framework>
- Commit hash: {fill before run}
- `config.example.yml` (exact version used)
- Seeds: [42, 123, 789]

**Data:**

- `{benchmark}_benchmark_results.json` (full statistics)
- `{benchmark}_per_question.csv` (raw predictions)
- Model versions/dates (in JSON metadata)
- Exclusion log (if any questions dropped)

**Reproduction Command:**

```bash
git clone https://github.com/nikolay-e/certamen-framework.git
git checkout {commit_hash}
pip install -e .[bench]
python -m benchmarks.standard_benchmarks --benchmark bbh --config config.example.yml
```

### 8.2 Expected Runtime & Cost

**Estimates (fill before run):**

- BBH: ~{X} minutes, ~${Y}
- GPQA: ~{X} minutes, ~${Y}
- Total: ~{X} hours, ~${Y}

**Machine Info:**

- OS: {uname -a}
- Python: {python --version}
- CPU/RAM: {fill}
- Network: {location for API latency}

### 8.3 Logging Requirements

**Per-Question Logs:**

- Model versions and timestamps
- Full prompts and responses
- Intermediate tournament states (Certamen)
- Knowledge Bank retrievals/injections
- Token counts (input/output per call)
- Cost and duration per call

**Aggregate Logs:**

- Configuration files (frozen)
- Exclusion decisions with reasons
- Random seeds used
- API errors/retries

---

## 9. Success Criteria & Stopping Rules

### 9.1 Strong Evidence (Publish as Success)

**H1 satisfied if:**

- p < 0.05 (McNemar)
- 95% CI excludes 0
- Cohen's h ≥ 0.2

**H2 satisfied if:**

- Cost-matched Δacc ≥ 0 (maintains advantage under budget cap)

**H3 satisfied if:**

- Consistent across runs (3 seeds)
- Positive in both BBH and GPQA

→ **Action:** Prepare paper with title "Certamen Improves Reasoning..."

### 9.2 Weak/Null Result (Report Honestly)

**If p ≥ 0.05 or CI includes 0:**

- Report full results transparently (no p-hacking)
- Analyze subgroups where Certamen wins (exploratory)
- Identify task characteristics that benefit from tournament
- Strengthen mechanism (better elimination, KB tuning)

→ **Action:** Technical report "Conditions Under Which Tournament-Based Synthesis Helps"

### 9.3 Stopping Rules

**Stop experiment if:**

- ✅ Planned sample size reached (N ≥ 300 per benchmark)
- ❌ API budget exceeded (set limit: ${fill} before run)
- ❌ >20% questions fail API (systemic issue, not dataset problem)

**No peeking:** Do not inspect accuracy until full dataset complete

---

## 10. Limitations & Ethics

### 10.1 Disclosed Limitations (Upfront)

1. **Model Scope:** 4 frontier models only (no open-source comparisons)
2. **Domain:** Text reasoning (no multimodal, code, math with tools)
3. **Compute Conservatism:** Cost-matched mode intentionally handicaps Certamen
4. **Prompt Engineering:** Zero-shot (no few-shot tuning per task)
5. **API Variability:** Model behavior may drift across versions
6. **Sample Size:** Power calculated for Δacc ≥ 5pp; smaller effects may be missed

### 10.2 Ethical Considerations

- **API ToS:** All models used within provider terms of service
- **No Human Data:** Benchmarks only, no private/sensitive information
- **Cost Transparency:** Report full costs to enable independent replication
- **No Selective Reporting:** Publish all results (positive, negative, null)
- **No HARKing:** Hypotheses fixed before data collection

---

## 11. Publication Plan

### 11.1 Planned Outputs

1. **Preregistration (This Document):** Locked before first run
2. **Technical Report:** Full results in this repository
3. **arXiv Preprint:** "Certamen: Tournament-Based Multi-Model Synthesis Improves Reasoning Under Fixed Compute Budgets"
4. **Open Release:**
   - Code: <https://github.com/nikolay-e/certamen-framework>
   - Data: Zenodo DOI (JSON, CSV, logs)

### 11.2 Timeline (Tentative)

- **2025-10-05:** Preregistration locked (this document)
- **2025-10-06:** Data collection (BBH + GPQA runs)
- **2025-10-07:** Ablations and cost-matched runs
- **2025-10-08:** Analysis and error taxonomy
- **2025-10-09:** Draft results and figures
- **2025-10-10:** Submit to arXiv

---

## 12. Sign-Off & Commitment

**This preregistration commits the research team to:**

- Fixed hypotheses (H1-H4 as stated)
- Prespecified analysis plan (§6)
- Transparent reporting (positive/negative/null)
- No data peeking before N complete
- Documentation of any deviations

**Any changes to this plan after lock will be:**

- Clearly marked as post-hoc
- Justified in final report
- Separated from confirmatory analyses

---

**Preregistered by:** Nikolay Eremeev
**Contact:** <nikolay.eremeev@outlook.com>
**Date:** 2025-10-05
**Status:** 🔒 LOCKED — Do not modify after first benchmark run
**Commit:** {fill: git rev-parse HEAD}

---

## Changelog

- **2025-10-05 v2.0:** Updated with compute parity, 3-run stability, exact McNemar, error taxonomy, release plan
- **2025-01-05 v1.0:** Initial preregistration
