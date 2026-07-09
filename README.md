# Explainable Insurance Claims Fraud Detection

A fraud-triage system that doesn't just score claims — it **explains** them. For each
suspicious transaction it produces a fraud probability *and* a human-readable reason
(via SHAP) that an investigator or regulator can act on.

> **Status:** 🚧 in progress — baseline model and imbalance analysis complete;
> explainability layer, calibration, and demo app in development.
> This is project 1 of a 3-part insurance-ML portfolio.

---

## The problem

An insurer receives thousands of claims a week. Only ~1–3.5% are fraudulent, and there
are only a handful of human investigators. A useful model must therefore deliver a
**ranked, explained shortlist** — not a black-box yes/no. Under UK/EU rules on automated
decisioning, an unexplained denial is increasingly not legally actionable, so the
explanation layer is the point, not a nice-to-have.

Two coupled problems:
1. **Prediction** — severely imbalanced binary classification (find the needles).
2. **Explanation** — per-claim justification via Shapley values (TreeSHAP).

## Why it's hard

- **Extreme class imbalance (~3.5% fraud).** Accuracy is meaningless — a model that
  predicts "legit" for everyone already scores 96.5%. The primary metric here is
  **PR-AUC**, not accuracy or ROC-AUC.
- **Asymmetric costs.** A missed fraud (full payout lost) is far more expensive than a
  false alarm (wasted investigator time), so the decision threshold is a *cost* decision.
- **Trust before explanation.** A confident explanation of a badly-calibrated score is
  just a convincing lie — hence explicit probability calibration.

## Results so far

Baseline: **default XGBoost, numeric features only, no imbalance handling** — deliberately
naive, to establish a floor.

| Metric | Value | Note |
|---|---|---|
| Accuracy | 97.96% | vs **96.50%** for "always legit" → +1.5pt for all that ML |
| ROC-AUC | 0.936 | flattered by the easy majority class |
| **PR-AUC** | **0.693** | the honest metric |
| Recall @ 0.5 threshold | 47.8% | **misses over half the fraud** |
| Precision @ 0.5 | 88.6% | cautious: only flags when very sure |

Threshold-moving + cost-sensitive analysis shows recall of 48% was a *choice* baked into
the 0.5 cutoff: within a realistic review budget (~3,000 claims) the same model reaches
**~56% recall at ~80% precision**. Lifting the curve further requires better training —
which is the next step.

## Roadmap

- [x] Data pipeline: IEEE-CIS transaction/identity merge (left join, missingness kept)
- [x] Naive XGBoost baseline + honest evaluation (PR-AUC, confusion matrix)
- [x] Threshold-moving + cost-sensitive operating-point analysis
- [ ] Imbalance handling — compare class weighting, SMOTE (incl. the leakage bug), threshold-moving
- [ ] Probability calibration (Platt / isotonic) + reliability diagram
- [ ] SHAP explainability — global summary + per-claim waterfall plots (TreeSHAP)
- [ ] Written comparison report (PR curves, cost-based eval)
- [ ] Streamlit demo — pick a claim, see its SHAP-explained score
- [ ] Port pipeline to a second, insurance-shaped dataset

## Tech stack

`Python 3.12` · `uv` · `pandas` · `scikit-learn` · `XGBoost` · `imbalanced-learn` ·
`SHAP` · `matplotlib` · `Streamlit`

## Setup

```bash
# Install dependencies into a managed virtual environment
uv sync

# Launch the analysis notebook
uv run jupyter lab
```

## Data

The dataset (Kaggle **IEEE-CIS Fraud Detection**) is **not committed** — it's gitignored
(~1.2 GB). To reproduce, download it into `data/raw/`:

```bash
uv run kaggle competitions download -c ieee-fraud-detection -p data/raw
# then unzip the four train_/test_ CSVs into data/raw/
```

## Project structure

```
data/          # gitignored — raw + processed Kaggle data
notebooks/     # exploration + modelling narrative
src/           # pipeline modules (grown as code solidifies)
README.md
```

## License

MIT
