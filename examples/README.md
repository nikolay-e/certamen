# Arbitrium Framework - Examples

Learn by doing! These examples demonstrate Arbitrium Framework's core features in order of increasing complexity.

## 🌐 Try in Your Browser (No Installation!)

### 🏆 **Interactive Demo** - Run Real Tournaments in Browser

**Perfect for:** First-time users, demos, workshops, quick validation
**What you'll learn:** Full tournament execution with real AI models
**Runtime:** 5-10 minutes | **Cost:** ~$0.50-2.00

**Zero installation required** - runs directly in your browser!

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/nikolay-e/arbitrium-framework/blob/main/examples/interactive_demo.ipynb)

**Features:**

- ✅ Real API calls to GPT, Claude, Grok
- ✅ Live tournament execution
- ✅ Interactive API key entry
- ✅ Cost tracking and metrics
- ✅ Downloadable reports
- ✅ Compare tournament vs. single model

📖 **[Full Documentation](./INTERACTIVE_DEMO.md)** | 📓 **[View Notebook](./interactive_demo.ipynb)**

**Alternative platforms:**

- [JupyterLite](https://jupyterlite.github.io/demo) (100% browser-based)
- [MyBinder](https://mybinder.org) (free cloud environment)
- Local Jupyter: `jupyter notebook examples/interactive_demo.ipynb`

---

## 📚 Python Examples (Local Installation)

### 1. `quickstart.py` - Your First Tournament (2 minutes)

**Perfect for:** First-time users, quick validation
**What you'll learn:** Run a tournament with minimal code
**Runtime:** ~2 minutes | **Cost:** ~$0.50

```bash
python examples/quickstart.py
```

### 2. `single_model.py` - Query Without Tournament (30 seconds)

**Perfect for:** Simple queries, cost-sensitive scenarios
**What you'll learn:** Use Arbitrium for single-model queries, compare models
**Runtime:** <30 seconds | **Cost:** ~$0.05-0.20

```bash
python examples/single_model.py
```

### 3. `tournament_basic.py` - Full Tournament Flow (5-10 minutes)

**Perfect for:** Understanding the tournament process
**What you'll learn:** Phases, elimination, cost tracking
**Runtime:** ~5-10 minutes | **Cost:** ~$0.50-2.00

```bash
python examples/tournament_basic.py
```

### 4. `tournament_with_kb.py` - Knowledge Bank in Action (5-10 minutes)

**Perfect for:** Understanding Arbitrium's core innovation
**What you'll learn:** How eliminated models contribute to the final answer
**Runtime:** ~5-10 minutes | **Cost:** ~$0.50-2.00

```bash
python examples/tournament_with_kb.py
```

### 5. `benchmark_comparison.py` - Cost-Benefit Analysis (10-20 minutes)

**Perfect for:** Deciding if tournament is worth it for your use case
**What you'll learn:** Compare single model vs all models vs tournament
**Runtime:** ~10-20 minutes | **Cost:** ~$1-3

```bash
python examples/benchmark_comparison.py
```

---

## 🚀 Quick Start

1. **Install Arbitrium:**

   ```bash
   pip install arbitrium-framework
   ```

2. **Copy example config:**

   ```bash
   cp config.example.yml config.yml
   # Edit config.yml with your API keys
   ```

3. **Run your first example:**

   ```bash
   python examples/quickstart.py
   ```

---

## 📊 When to Use What?

| Scenario | Example to Run | Why |
|----------|---------------|-----|
| **First time / No install** | **[Interactive Demo](./INTERACTIVE_DEMO.md)** 🌐 | Try in browser, zero setup |
| **Just starting** | `quickstart.py` | Fastest local setup |
| **Simple query** | `single_model.py` | No tournament overhead |
| **High-stakes decision** | `tournament_basic.py` | Full synthesis power |
| **Understand KB** | `tournament_with_kb.py` | See innovation in action |
| **Evaluate ROI** | `benchmark_comparison.py` | Cost-benefit proof |

---

## 💡 Tips

### Modify for Your Use Case

All examples use placeholder questions. Replace with your actual questions:

```python
# Change this:
question = "What is the best strategy for..."

# To your question:
question = "Should we migrate to Kubernetes or stay on EC2?"
```

### Adjust Model Selection

Edit `config.yml` to:

- Add/remove models
- Change temperature settings
- Enable/disable Knowledge Bank
- Adjust retry logic

### Monitor Costs

Every example prints cost breakdown. Start with cheap models (`grok`, `gemini-2.5-flash`) for testing.

### Enable Debug Logging

Set in your config:

```yaml
logging:
  level: DEBUG
  file: arbitrium_debug.log
```

---

## 🆘 Troubleshooting

### "No models configured"

- Check `config.yml` exists
- Verify API keys are set correctly
- Run health check: `arbitrium health`

### "Model timeout"

- Increase timeout in config: `retry.max_delay`
- Check network connection
- Try simpler question first

### High costs

- Start with 2-3 models for testing
- Use cheaper models: `grok-4`, `gemini-2.5-flash`
- Set `features.max_cost` in config

---

## 📖 Next Steps

- Read the [full documentation](../README.md)
- Check out [GitHub Discussions](https://github.com/nikolay-e/arbitrium-framework/discussions) for Q&A
- Contribute your own examples via PR!

## 🔬 For Developers: Internal Benchmarks

Looking for performance benchmarking and statistical evaluation tools?
See [`src/arbitrium/benchmarks/`](../src/arbitrium/benchmarks/) for internal testing frameworks.

**Examples vs Benchmarks:**

- **`examples/`** (this folder): User-facing tutorials showing how to use Arbitrium
- **`src/arbitrium/benchmarks/`**: Developer tools for framework validation and performance testing

---

**Questions?** Open a [GitHub Discussion](https://github.com/nikolay-e/arbitrium-framework/discussions) or [Issue](https://github.com/nikolay-e/arbitrium-framework/issues)!
