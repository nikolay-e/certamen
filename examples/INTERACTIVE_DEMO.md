# 🏆 Arbitrium Framework - Interactive Browser Demo

## Run Real AI Tournaments in Your Browser

This interactive demo lets you run **actual** Arbitrium tournaments with **real AI models** directly in your browser - no installation required!

---

## 🚀 Quick Start Options

### Option 1: Google Colab (Recommended)

**Easiest way to get started - just click and run!**

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/nikolay-e/arbitrium-framework/blob/main/examples/interactive_demo.ipynb)

**Pros:**

- ✅ Free tier available
- ✅ No setup required
- ✅ Fast execution
- ✅ Easy to share

**Setup time:** < 1 minute

---

### Option 2: JupyterLite (Fully Browser-Based)

**Runs 100% in your browser with Pyodide - no backend needed!**

🔗 **[Launch JupyterLite Demo](https://jupyterlite.github.io/demo/lab/index.html?path=arbitrium-demo.ipynb)**

**Pros:**

- ✅ Completely client-side
- ✅ No server required
- ✅ Works offline after initial load
- ✅ Maximum privacy

**Setup time:** < 2 minutes (initial load)

**Steps:**

1. Click link above
2. Upload `interactive_demo.ipynb` when JupyterLite loads
3. Run cells in order

---

### Option 3: MyBinder

**Free cloud Jupyter environment**

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/nikolay-e/arbitrium-framework/main?labpath=examples/interactive_demo.ipynb)

**Pros:**

- ✅ Free
- ✅ Full Python environment
- ✅ No account needed

**Setup time:** 2-5 minutes (first build)

---

### Option 4: Local Jupyter

**Run on your own machine**

```bash
# Install Jupyter
pip install jupyter arbitrium

# Start Jupyter
jupyter notebook examples/interactive_demo.ipynb
```

**Pros:**

- ✅ Full control
- ✅ Fastest execution
- ✅ No quotas

**Setup time:** 5 minutes

---

## 📋 Requirements

### API Keys (Required)

You'll need API keys for the AI models you want to use:

**Minimum:** At least 2 of the following:

- 🔑 **OpenAI API Key** (for GPT)
  - Get it: <https://platform.openai.com/api-keys>

- 🔑 **Anthropic API Key** (for Claude)
  - Get it: <https://console.anthropic.com/>

- 🔑 **XAI API Key** (for Grok)
  - Get it: <https://console.x.ai/>

### Budget

- **Estimated cost per tournament:** $0.50 - $2.00
- **Runtime:** 5-10 minutes
- **Tip:** Start with 2-3 cheaper models (Gemini + GPT-4) to minimize cost

---

## 🎯 What You'll Experience

### Real Tournament Execution

1. **Phase 1: Initial Responses**
   - Each AI model independently analyzes your question
   - Real API calls to OpenAI, Anthropic, Google, etc.

2. **Phase 2: Improvement**
   - Models see each other's responses
   - Generate improved versions based on peer feedback

3. **Phase 3: Evaluation & Elimination**
   - Cross-evaluation scores all responses
   - Weakest model eliminated
   - **Knowledge Bank** preserves its best insights

4. **Repeat**
   - Process continues until one champion remains
   - Champion incorporates insights from all eliminated models

### Real Metrics

- 💰 **Exact cost tracking** (down to $0.0001)
- ⏱️ **Real-time execution monitoring**
- 📊 **Per-model cost breakdown**
- 📁 **Downloadable tournament reports**

---

## 📖 Demo Walkthrough

### Step 1: Install Framework

```python
# Try PyPI first (once published)
import micropip
await micropip.install('arbitrium')

# Or from GitHub:
# !pip install git+https://github.com/nikolay-e/arbitrium-framework.git
```

### Step 2: Configure API Keys

Interactive prompt to securely enter your API keys:

```python
import getpass
openai_key = getpass.getpass("OpenAI API Key: ")
```

**Security:** Keys stay in your browser session only!

### Step 3: Initialize Tournament

```python
from arbitrium_core import Arbitrium

arbitrium = await Arbitrium.from_settings(config)
# Health checks all models
```

### Step 4: Define Your Question

```python
question = """
Should our startup focus on PLG or SLG?
Context: B2B SaaS, $500k funding...
"""
```

### Step 5: Run Tournament

```python
result, metrics = await arbitrium.run_tournament(question)

print(f"Champion: {metrics['champion_model']}")
print(f"Cost: ${metrics['total_cost']:.4f}")
print(result)
```

### Step 6: Analyze Results

- View champion solution
- Download detailed reports
- Compare with single-model baseline

---

## 💡 Tips for Best Results

### Choose Good Questions

✅ **Good:**

- Strategic business decisions
- Technical architecture choices
- Complex trade-off analysis
- Multi-stakeholder synthesis

❌ **Poor:**

- Simple factual queries
- Time-sensitive decisions
- Questions with objective answers

### Optimize Costs

1. Start with 2-3 models (not 4+)
2. Use cheaper models for testing:
   - Gemini 1.5 Flash (~$0.10)
   - GPT-4 Turbo Mini (~$0.15)
3. Test with shorter questions first
4. Monitor costs in real-time

### Maximize Value

- Use for decisions worth $5,000+
- Questions where diverse perspectives help
- When you need audit trails
- For compliance-sensitive decisions

---

## 🔒 Security & Privacy

### Your API Keys

- ✅ Entered via secure `getpass` prompt
- ✅ Stored only in session memory
- ✅ Never sent to any server (except AI providers)
- ✅ Cleared when notebook closes

### Your Data

- ✅ Questions sent only to AI providers you choose
- ✅ Results stay in your browser
- ✅ No analytics or tracking
- ✅ Open source - verify yourself!

---

## 🆘 Troubleshooting

### "No module named 'arbitrium'"

**Solution:** Run the installation cell:

```python
import micropip
await micropip.install('arbitrium')
```

### "Need at least 2 healthy models"

**Solution:**

- Check API keys are correct
- Ensure you have sufficient API quota
- Try different model providers

### "Rate limit exceeded"

**Solution:**

- Wait a few minutes
- Use different API key
- Reduce number of concurrent models

### High costs

**Solution:**

- Start with 2 models instead of 4
- Use cheaper models (Gemini Flash)
- Test with shorter questions

---

## 📚 Example Questions to Try

### Business Strategy

```
Should we pivot from B2B to B2C, given our current runway
and market feedback?
```

### Technical Architecture

```
What is the best data pipeline architecture for processing
1TB/day of event data with <1 hour latency?
```

### Product Roadmap

```
Which 3 features should we prioritize for Q2 to maximize
user retention and revenue?
```

### Market Analysis

```
Should we enter the European market now or wait 12 months
to build more features?
```

---

## 🎓 Understanding the Results

### What Makes a Good Champion?

The winning solution typically:

- ✅ Comprehensive analysis
- ✅ Concrete recommendations
- ✅ Risk assessment
- ✅ Implementation roadmap
- ✅ Success metrics
- ✅ Incorporates insights from eliminated models

### Reading Tournament Reports

Generated reports include:

- **Complete History**: All responses, scores, feedback
- **Provenance**: How ideas evolved across rounds
- **Knowledge Bank**: Preserved insights from eliminated models
- **Cost Breakdown**: Per-phase spending

### Comparing Tournament vs. Single Model

The notebook includes a comparison showing:

- Side-by-side responses
- Cost multiplier
- Quality assessment framework

---

## 🚀 Deploy Your Own Version

### Host on GitHub Pages

```bash
# 1. Fork the repository
# 2. Enable GitHub Pages
# 3. JupyterLite will auto-deploy

# Your demo URL will be:
# https://yourusername.github.io/arbitrium/lab/index.html
```

### Embed in Documentation

```html
<iframe
  src="https://colab.research.google.com/github/your-org/arbitrium/blob/main/examples/interactive_demo.ipynb"
  width="100%"
  height="800px">
</iframe>
```

### Custom Branding

Edit the notebook markdown cells to:

- Add your logo
- Customize colors
- Include company context
- Pre-fill example questions

---

## 📊 What's Next?

### After Running the Demo

**If you liked it:**

1. ⭐ Star the [GitHub repo](https://github.com/nikolay-e/arbitrium-framework)
2. 📦 Install locally: `pip install arbitrium-framework`
3. 📖 Read the [full documentation](https://github.com/nikolay-e/arbitrium-framework)
4. 💬 Join our [Discord community](https://discord.gg/arbitrium)

**If you have feedback:**

1. 🐛 [Report issues](https://github.com/nikolay-e/arbitrium-framework/issues)
2. 💡 [Request features](https://github.com/nikolay-e/arbitrium-framework/issues/new)
3. 🤝 [Contribute](https://github.com/nikolay-e/arbitrium-framework/blob/main/CONTRIBUTING.md)

### Real-World Usage

```python
# Install for production use
pip install arbitrium
# Or from GitHub: pip install git+https://github.com/nikolay-e/arbitrium-framework.git

# Create config file
cp config.example.yml config.yml

# Run from CLI
arbitrium

# Or use programmatically
from arbitrium_core import Arbitrium

async def main():
    arbitrium = await Arbitrium.from_config("config.yml")
    result, metrics = await arbitrium.run_tournament(
        "Your strategic question"
    )
    return result
```

---

## 🙋 FAQ

### Q: Is this really free?

**A:** The demo is free to run. You only pay for the AI API calls to OpenAI, Anthropic, etc. (~$0.50-2 per tournament).

### Q: Do I need to install anything?

**A:** No! Works directly in browser with Google Colab or JupyterLite.

### Q: How long does it take?

**A:** 5-10 minutes for a typical tournament with 3-4 models.

### Q: Can I use my own models?

**A:** Yes! The framework supports any LiteLLM-compatible model.

### Q: Is my data private?

**A:** Your questions go only to the AI providers you choose. The framework doesn't send data anywhere else.

### Q: Can I download the results?

**A:** Yes! Complete tournament reports are generated and can be downloaded.

### Q: What if I only have one API key?

**A:** You need at least 2 healthy models to run a tournament. Get keys from different providers.

---

**Ready to run your first AI tournament?**

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/nikolay-e/arbitrium-framework/blob/main/examples/interactive_demo.ipynb)

*Arbitrium Framework - Tournament-Based AI Decision Synthesis*
