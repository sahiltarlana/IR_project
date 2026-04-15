# Adaptive-RAG: Learning to Adapt Retrieval-Augmented Large Language Models through Question Complexity

**IT367 Information Retrieval — Course Project [Jan–Apr 2026]**

**Base Paper:** Jeong et al., NAACL 2024 ([arXiv:2403.14403](https://arxiv.org/abs/2403.14403))

**Team:** Pranav Moothedath (221AI030) | Tarlana Sahil (221AI040)

---

## What This Project Does

We implement the **Adaptive-RAG** framework that dynamically selects the best retrieval strategy for each query based on its complexity:

- **Level A (No Retrieval):** Simple factual questions → LLM answers directly
- **Level B (Single-step):** Moderate questions → Retrieve documents once, then answer
- **Level C (Multi-step):** Complex multi-hop questions → Iteratively retrieve and reason

We also implement **3 proposed enhancements** over the base paper:
1. **Dense Retrieval** (sentence-transformers) replacing BM25
2. **Improved Classifier** with linguistic features + Gradient Boosting
3. **Confidence-based Fallback** — escalates to a harder strategy when classifier is uncertain

---

## Project Structure

```
├── demo.py                    # Main entry point — runs everything
├── requirements.txt           # Python dependencies
├── generate_report_pdf.py     # Generates the PDF report
├── report.pdf                 # Final project report (7 pages)
├── report.md                  # Report in Markdown format
├── src/
│   ├── run_experiments.py     # Runs all 6 experiments
│   ├── data_loader.py         # Loads SQuAD, NQ, TriviaQA, HotpotQA
│   ├── retrieval.py           # BM25 + Dense retrieval
│   ├── classifier.py          # Base + Enhanced query-complexity classifiers
│   ├── strategies.py          # No-retrieval, Single-step, Multi-step, Adaptive-RAG
│   ├── metrics.py             # EM, F1, Accuracy evaluation
│   └── generate_plots.py      # Generates all result plots
├── results/                   # Experiment results (JSON)
├── plots/                     # Generated figures (PNG)
└── 2403.14403v2.pdf           # Original research paper
```

---

## Quick Start

### 1. Setup (one-time, ~2 minutes)

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate        # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Run Full Pipeline (training + experiments + plots + report)

```bash
python demo.py
```

This will:
- Download datasets from HuggingFace (SQuAD, NQ, TriviaQA, HotpotQA)
- Download FLAN-T5-Small model (~300MB, first run only)
- Download sentence-transformer model (~90MB, first run only)
- Train both classifiers (Base + Enhanced)
- Run all 6 experiments
- Generate 5 plots in `plots/`
- Generate `report.pdf`

**Total time:** ~5-10 minutes (depending on internet speed for first download)

### 3. Interactive Demo (quick test)

```bash
python demo.py --quick
```

This loads the models, shows example queries with Adaptive-RAG routing decisions, then lets you type your own questions interactively.

---

## Experiments Run

| # | Method | Description |
|---|--------|-------------|
| 1 | No Retrieval | LLM answers from parametric knowledge only |
| 2 | Single-step (BM25) | Retrieve once with BM25, then answer |
| 3 | Multi-step (BM25) | Iterative retrieval + reasoning (up to 3 steps) |
| 4 | Adaptive-RAG (Base) | Base classifier + BM25 |
| 5 | Adaptive-RAG (Enhanced) | Enhanced classifier + Dense retrieval |
| 6 | Adaptive-RAG (Enhanced+Fallback) | Enhanced + confidence-based fallback |

---

## Results Summary

| Method | EM (%) | F1 (%) | Acc (%) | Avg Steps | Avg Time (s) |
|--------|--------|--------|---------|-----------|---------------|
| No Retrieval | 7.89 | 12.33 | 7.89 | 0.00 | 0.437 |
| Single-step (BM25) | 15.79 | 20.02 | 23.68 | 1.00 | 0.440 |
| Multi-step (BM25) | 21.05 | 22.73 | 23.68 | 2.39 | 1.168 |
| Adaptive-RAG (Base) | 15.79 | 20.02 | 23.68 | 0.95 | 0.465 |
| Adaptive-RAG (Enhanced) | 15.79 | 17.44 | 23.68 | 0.76 | 0.568 |
| Adaptive-RAG (Enhanced+Fallback) | 15.79 | 17.44 | 23.68 | 0.76 | 0.576 |

**Note:** We use FLAN-T5-Small (80M params) instead of the paper's FLAN-T5-XL (3B) due to laptop constraints. Absolute scores are lower but relative patterns match the paper.

---

## Running Individual Components

```bash
# Activate venv first
source venv/bin/activate

# Run only experiments
python src/run_experiments.py

# Generate only plots (requires results/ to exist)
python src/generate_plots.py

# Generate only the PDF report (requires results/ and plots/ to exist)
python generate_report_pdf.py
```

---

## Cleanup

```bash
# Remove everything except source code and report
rm -rf venv data_cache __pycache__ src/__pycache__
```

---

## References

- Jeong et al. (2024). Adaptive-RAG. NAACL 2024. [Paper](https://arxiv.org/abs/2403.14403) | [Code](https://github.com/starsuzi/Adaptive-RAG)
