# Dataset Versions

**Last Updated:** 2025-10-05

This file tracks the exact versions of datasets used in benchmarks to ensure reproducibility.

---

## BBH (Big-Bench Hard)

- **Source:** `lukaemon/bbh` (Hugging Face Datasets)
- **Version/Commit:** {fill: check datasets library version and dataset commit}
- **Access Date:** {YYYY-MM-DD}
- **Subset Used:** {list specific tasks or "all"}

**Verification:**

```python
from datasets import load_dataset
ds = load_dataset("lukaemon/bbh", split="test")
print(f"Total examples: {len(ds)}")
print(f"Dataset info: {ds.info}")
```

---

## GPQA (Graduate-Level Questions)

- **Source:** `Idavidrein/gpqa` (Hugging Face Datasets)
- **Subset:** `gpqa_diamond` (hardest tier)
- **Version/Commit:** {fill: check datasets library version and dataset commit}
- **Access Date:** {YYYY-MM-DD}
- **Total Questions:** {N}

**Verification:**

```python
from datasets import load_dataset
ds = load_dataset("Idavidrein/gpqa", "gpqa_diamond", split="train")
print(f"Total examples: {len(ds)}")
print(f"Columns: {ds.column_names}")
```

---

## Update Protocol

**When updating dataset versions:**

1. Record new version/commit hash in this file
2. Note update date and reason (e.g., "bug fix in dataset", "new questions added")
3. Update `reports/RESULTS.md` to reference new version
4. Re-run benchmarks and compare results (note any significant changes)
5. If results differ substantially, document delta in RESULTS.md

**Versioning Strategy:**

- **Lock versions** for preregistered experiments (use specific commits)
- **Document any deviations** from preregistered dataset in final report
- **Archive old results** when upgrading datasets (keep JSON/CSV in git history)

---

## Environment Info

**Python Packages (used for dataset loading):**

```
datasets==2.14.0  # or actual version
huggingface-hub==0.X.Y
```

**Cache Location:**

- Hugging Face cache: `~/.cache/huggingface/datasets/`
- To ensure reproducibility, cache can be committed to CI artifacts

---

## Citation

**BBH:**

```bibtex
@article{suzgun2022challenging,
  title={Challenging BIG-Bench tasks and whether chain-of-thought can solve them},
  author={Suzgun, Mirac and Scales, Nathan and Sch{\"a}rli, Nathanael and Gehrmann, Sebastian and Tay, Yi and Chung, Hyung Won and Chowdhery, Aakanksha and Le, Quoc V and Chi, Ed H and Zhou, Denny and others},
  journal={arXiv preprint arXiv:2210.09261},
  year={2022}
}
```

**GPQA:**

```bibtex
@article{rein2023gpqa,
  title={GPQA: A Graduate-Level Google-Proof Q\&A Benchmark},
  author={Rein, David and Hou, Betty Li and Stickland, Asa Cooper and Petty, Jackson and Pang, Richard Yuanzhe and Dirani, Julien and Michael, Julian and Bowman, Samuel R},
  journal={arXiv preprint arXiv:2311.12022},
  year={2023}
}
```
