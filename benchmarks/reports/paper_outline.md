# Paper Outline: Arbitrium Framework

**Title:** *Arbitrium: Tournament-Based Multi-Model Synthesis Achieves Higher Accuracy than Single Models at Comparable Cost*

**Target:** arXiv preprint → ML/NLP conference (ICLR, NeurIPS, ACL, EMNLP)

---

## Abstract (200-250 words)

Large language models (LLMs) demonstrate impressive capabilities but struggle with complex reasoning tasks requiring synthesis of diverse perspectives. Existing ensemble methods like self-consistency and majority voting show limited gains. We introduce **Arbitrium**, a tournament-based framework that combines competitive elimination with a Knowledge Bank to synthesize superior responses from multiple models.

**Key contributions:**
1. Tournament architecture with iterative elimination and cross-model learning
2. Knowledge Bank mechanism for preserving insights from eliminated models
3. Statistically significant accuracy improvements on reasoning benchmarks

**Results:**
- BBH (Big-Bench Hard): X.X% improvement (p < 0.05, Cohen's h = 0.XX)
- GPQA (Graduate Questions): X.X% improvement (p < 0.05, Cohen's h = 0.XX)
- Cost-competitive: XX% higher accuracy/$ vs. best single model

We demonstrate that tournament-based synthesis outperforms strong baselines (Self-Consistency n=5, Majority Vote, Judge Re-rank) while maintaining reasonable computational costs. Analysis reveals Arbitrium excels on tasks requiring multi-perspective reasoning and bias mitigation.

---

## 1. Introduction

### Problem Statement
- Single LLMs exhibit blind spots, biases, and inconsistencies
- Complex decisions require synthesizing multiple perspectives
- Existing ensemble methods: limited theoretical grounding, marginal gains

### Related Approaches
- Self-Consistency: sample multiple times, majority vote [Wang et al., 2022]
- Majority Vote: different models, simple voting [unclear gains]
- Debate: adversarial back-and-forth [Irving et al., 2018; Du et al., 2023]
- Re-ranking: judge selects best from k candidates [limited synthesis]

### Our Contribution
1. **Novel architecture**: Tournament eliminates weak responses iteratively
2. **Knowledge preservation**: Bank captures insights from eliminated models
3. **Empirical validation**: Statistically significant gains on reasoning benchmarks
4. **Cost analysis**: Normalized metrics show efficiency vs. baselines

### Research Questions
- **RQ1:** Does tournament + KB outperform single best model?
- **RQ2:** How does Arbitrium compare to ensemble baselines?
- **RQ3:** What is the cost-effectiveness trade-off?
- **RQ4:** Which task characteristics favor Arbitrium?

---

## 2. Method

### 2.1 Tournament Architecture

**Input:** Question Q, set of models M = {m₁, m₂, ..., mₙ}

**Round r:**
1. All active models generate response to Q (with feedback from round r-1)
2. Rotating evaluator judges responses (excluding self)
3. Lowest-scored model eliminated
4. Knowledge Bank updated with eliminated model's insights

**Output:** Champion model's final response

**Key design choices:**
- Rotating evaluation prevents single-judge bias
- Iterative elimination focuses on strongest candidates
- Feedback loop enables cross-model learning

### 2.2 Knowledge Bank

**Purpose:** Preserve valuable insights from eliminated models

**Process:**
1. Extract key insights from eliminated response (vectorized)
2. Store in semantic vector database (sklearn cosine similarity)
3. Before next round: retrieve top-k relevant insights (threshold = 0.75)
4. Inject into improvement prompt for remaining models

**Rationale:** Weak models may contribute valid partial insights

### 2.3 Prompting Strategy

**Initial Prompt:**
```
Analyze from multiple perspectives. Challenge assumptions.
Identify core dilemma. Ground in evidence. Be concise.
```

**Feedback Prompt:**
```
Identify most insightful, evidence-based ideas.
Note bias and speculation. Value independent thinking.
```

**Improvement Prompt:**
```
Improve using feedback. Mitigate biases.
Make verifiable insights central. Remove speculation.
[KB insights injected here if relevant]
```

**Evaluation Prompt:**
```
Judge analytical depth and rigor. Does it rely on
proven methodology or speculation? Score 0-10.
```

### 2.4 Implementation Details
- Models: GPT-5, Claude 4.5 Sonnet, Gemini 2.5 Pro, Grok-4
- Temperature: 0.7
- Context window: Model-specific (auto-detected)
- KB similarity threshold: 0.75
- KB max injections: 5
- Cost tracking: Token-level via LiteLLM

---

## 3. Experimental Setup

### 3.1 Datasets

**BBH (Big-Bench Hard)** [Suzgun et al., 2022]
- 8 tasks: causal judgement, formal fallacies, navigation, disambiguation, deduction, tracking, web of lies, recommendations
- 5 questions per task = 40 total
- Rationale: Tests multi-step reasoning

**GPQA (Graduate-Level Questions)** [Rein et al., 2023]
- Diamond subset: 20 expert-level science questions
- Requires synthesis of domain knowledge
- Rationale: High difficulty, PhD-level reasoning

**Task Characteristics:**
- Closed-choice (MCQ)
- Multi-perspective reasoning required
- No single obvious answer
- Synthesis benefits expected

### 3.2 Baselines

1. **Single (Best)**: Claude 4.5 Sonnet, 1 sample
2. **Self-Consistency (n=5)**: Claude, 5 samples, majority vote
3. **Majority Vote**: Each model votes once
4. **Judge Re-rank**: Generate k=3, judge ranks
5. **Arbitrium w/o KB**: Tournament, no knowledge bank
6. **Arbitrium (Full)**: Tournament + KB

**Cost Normalization:**
- Track tokens, time, $/question
- Report accuracy/$, accuracy/minute

### 3.3 Metrics

**Primary:**
- Accuracy (% correct)
- McNemar test (paired, α=0.05)
- Bootstrap 95% CI (10k iterations, seed=123)
- Cohen's h (effect size)

**Secondary:**
- Cost per correct answer
- Time per correct answer
- Accuracy/$ and accuracy/min

### 3.4 Controls
- Fixed seeds: 42 (experiments), 123 (bootstrap)
- Fixed model versions (logged)
- Fixed prompts and temperature
- Preregistered analysis plan (see Appendix A)

---

## 4. Results

### 4.1 Main Results

**Table 1: Accuracy Comparison**

| Method | BBH Acc (%) | GPQA Acc (%) | Combined (%) | Δ vs Best Single |
|--------|-------------|--------------|--------------|-------------------|
| Single (Claude) | XX.X | XX.X | XX.X | - |
| Self-Consistency (n=5) | XX.X | XX.X | XX.X | +X.X |
| Majority Vote | XX.X | XX.X | XX.X | +X.X |
| Judge Re-rank | XX.X | XX.X | XX.X | +X.X |
| Arbitrium w/o KB | XX.X | XX.X | XX.X | +X.X |
| **Arbitrium (Full)** | **XX.X** | **XX.X** | **XX.X** | **+X.X*** |

\*p < 0.05 (McNemar), Cohen's h = 0.XX

**Key Finding:** Arbitrium achieves statistically significant improvement (p = 0.0XX, 95% CI [X.X%, X.X%])

### 4.2 Statistical Significance

**McNemar Test Results:**
- BBH: b01=X, b10=X, χ²=X.XX, p=0.0XX
- GPQA: b01=X, b10=X, χ²=X.XX, p=0.0XX
- Combined: Significant at α=0.05 ✓

**Effect Size (Cohen's h):**
- BBH: h = 0.XX (medium/large)
- GPQA: h = 0.XX (medium/large)

### 4.3 Cost-Normalized Performance

**Table 2: Efficiency Metrics**

| Method | Acc/$ | Acc/min | Total Cost ($) | Time (min) |
|--------|-------|---------|----------------|------------|
| Single | XX.X | XX.X | X.XX | X.X |
| Self-Consistency | XX.X | XX.X | X.XX | X.X |
| Majority Vote | XX.X | XX.X | X.XX | X.X |
| **Arbitrium** | **XX.X** | **XX.X** | X.XX | X.X |

**Finding:** Arbitrium achieves XX% higher accuracy/$ than best baseline

### 4.4 Ablation Studies

**Knowledge Bank Contribution:**
- Arbitrium (Full): XX.X%
- Arbitrium w/o KB: XX.X%
- **KB gain: +X.X%** (p = 0.0XX)

**Sensitivity Analysis:**
- KB threshold [0.6, 0.75, 0.9]: Optimal at 0.75
- KB max insights [3, 5, 10]: Diminishing returns beyond 5
- Number of models [3, 4, 5]: Gains plateau at 4 models

---

## 5. Analysis

### 5.1 When Does Arbitrium Win?

**Task characteristics favoring Arbitrium:**
1. **Multi-perspective problems**: Causal judgement (+XX%), formal fallacies (+XX%)
2. **Ambiguous questions**: Disambiguation (+XX%)
3. **Complex reasoning chains**: Deduction (+XX%)

**Tasks with marginal gains:**
- Simple factual recall
- Single-step inference
- Unambiguous answers

### 5.2 Knowledge Bank Impact

**Qualitative Analysis:**
- Case Study 1: Eliminated model identifies critical risk; KB preserves it; champion integrates
- Case Study 2: Weak model has domain insight; KB retrieves for specialist model

**Quantitative:**
- XX% of champion responses cite KB-retrieved insights
- KB retrievals correlate with accuracy gains (r = 0.XX)

### 5.3 Cost-Benefit Analysis

**Break-even scenarios:**
- High-stakes decisions (>$XXk value): Arbitrium justified
- Low-stakes tasks (<$X value): Single model sufficient
- Critical decisions requiring bias mitigation: Arbitrium preferred

**Recommendations:**
- Use Arbitrium for: complex reasoning, multi-stakeholder decisions, bias-sensitive tasks
- Use single model for: simple QA, speed-critical tasks, low-value decisions

---

## 6. Limitations

1. **Computational cost**: 3-5x tokens vs. single model (but cost-competitive per accuracy)
2. **Latency**: Sequential rounds increase wall-time
3. **Model availability**: Requires access to multiple frontier models
4. **Closed-choice focus**: Current study limited to MCQ tasks
5. **Dataset size**: 60 questions (BBH+GPQA); larger validation needed
6. **Prompt sensitivity**: Performance may vary with prompt engineering

---

## 7. Ethics and Broader Impact

**Positive:**
- Bias mitigation through diverse model perspectives
- Transparency via tournament provenance
- Reduced single-model failure modes

**Concerns:**
- Computational cost environmental impact
- Potential to amplify shared model biases
- Access inequality (requires multiple paid APIs)

**Mitigation:**
- Cost-normalized metrics guide deployment
- Bias analysis in model selection
- Open-source framework democratizes access

---

## 8. Related Work

**Ensemble Methods:**
- Self-Consistency [Wang et al., 2022]: Single model, sampling
- Majority Vote: Multiple models, simple aggregation
- Mixture-of-Experts [Shazeer et al., 2017]: Routing, not synthesis

**Multi-Agent Systems:**
- Debate [Irving et al., 2018]: Adversarial, truth-seeking
- Society of Mind [Singh et al., 2023]: Cooperative agents
- MAD [Liang et al., 2023]: Multi-agent debate

**Differences from Arbitrium:**
- Competitive elimination (vs. cooperative/adversarial)
- Knowledge preservation (vs. discarding losers)
- Rotating evaluation (vs. fixed judge)

---

## 9. Conclusion

We introduced Arbitrium, a tournament-based framework for multi-model synthesis. Key contributions:

1. **Novel architecture**: Competitive elimination + Knowledge Bank
2. **Empirical validation**: Statistically significant gains on BBH and GPQA
3. **Cost analysis**: Competitive accuracy/$ vs. baselines
4. **Open-source release**: Reproducible framework and datasets

**Future work:**
- Open-ended tasks (human evaluation)
- Larger-scale datasets (MMLU, ARC-Challenge)
- Real-world deployment studies
- Theoretical analysis of tournament dynamics

**Code:** https://github.com/arbitrium-framework/arbitrium
**Data:** Available under CC-BY-4.0

---

## Appendices

### A. Preregistration
[Full preregistration document]

### B. Prompts
[Complete prompt templates]

### C. Model Configurations
[Exact configs, versions, parameters]

### D. Raw Results
[Full JSON outputs, per-question analysis]

### E. Error Analysis
[Failure modes, edge cases]

### F. Computational Resources
[Hardware, costs, carbon footprint]

---

## References

[To be filled with actual citations]

- Wang et al. (2022). Self-Consistency Improves Chain of Thought Reasoning
- Irving et al. (2018). AI Safety via Debate
- Du et al. (2023). Improving Factuality and Reasoning in Language Models
- Suzgun et al. (2022). Challenging BIG-Bench Tasks
- Rein et al. (2023). GPQA: A Graduate-Level Google-Proof Q&A Benchmark
