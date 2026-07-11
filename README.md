# Explainable Insurance Claims Fraud Detection

A fraud-triage system that doesn't just score claims — it **explains** them. For each
suspicious transaction it produces a fraud probability *and* a human-readable reason
(via SHAP) that an investigator or regulator can act on.

> **Status:** 🚧 in progress — baseline model and imbalance analysis complete;
> next up: leakage-aware temporal evaluation, imbalance comparison, SHAP, calibration,
> and a FastAPI inference service.
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

## Results so far

**Naive baseline** (default XGBoost, numeric features only, no imbalance handling),
evaluated two ways — and the difference is the first finding:

| Evaluation | PR-AUC | Recall @ 0.5 | Note |
|---|---|---|---|
| Random 80/20 split | 0.693 | 47.8% | **inflated — leaks future information** |
| Temporal split (near future / val) | **0.557** | 38.5% | the honest number |
| Temporal split (far future / test) | 0.448 | 32.7% | **concept drift: the model decays with time** |
| Imbalance comparison, 3k budget (val) | 0.557 / 0.525 / 0.560 | 44.9% / 42.0% / 44.6% @ budget | control / weighted / SMOTE — rebalancing added no skill (nb 03) |
| Feature engineering (val) | 0.577 | 39% @ 0.5 | categoricals +0.017; time +0.003; per-card amounts −0.007 (rejected) — nb 04 |
| Calibration check (val) | 0.577 | — | raw scores near-calibrated (Brier 0.0237 vs 0.0375 naive); Platt/isotonic ≤1% gain — shipped raw (nb 05) |
| SHAP layer (val) | 0.577 | — | exact per-claim waterfalls; top global feature is raw time → drift mechanism found (nb 06) |


Key findings so far:

- **~0.14 of the baseline PR-AUC was leakage.** A random split lets the model train on
  the future; a time-ordered split removes the subsidy. (ROC-AUC hides most of this:
  0.936 → 0.905.)
- **Fraud models age.** Two months further into the future costs another ~0.11 PR-AUC —
  in production this model needs rolling retraining and drift monitoring.
- **Operating points are perishable.** A 3,000-claim review-budget threshold tuned on
  the validation window catches 45% of fraud there, but only 37% in the test window.
- Accuracy is meaningless here (a "flag nothing" model scores 96.5%), and a flat
  100:1 cost model produces a degenerate "flag everything" optimum — the threshold is
  a capacity/cost decision, not a statistical constant.
- **Rebalancing ≠ knowledge.** Class weighting and (correctly applied) SMOTE failed to beat
  the untreated baseline on PR-AUC; they mostly relocate the decision threshold. The same
  SMOTE applied *before* the split scores a fake 0.998 — the classic leakage bug, reproduced
  and documented in notebook 03.
- **Features, not rebalancing, moved the needle** — restoring the 31 discarded categorical
  columns beat every imbalance technique combined (+0.017 PR-AUC). A rejected feature
  family (noisy per-card statistics) is documented in notebook 04.
- **Scores are verified probabilities.** Reliability diagram is near-diagonal; calibration
  (Platt/isotonic) was tested and rejected as unnecessary — raw log-loss training on the
  true distribution already yields honest probabilities (unlike the 5×-inflated weighted
  model from the imbalance comparison).
- **SHAP found the drift mechanism**: the model's strongest global feature was raw calendar
  time — in-window memorization, not fraud knowledge. Explainability doubled as model
  debugging.
- **Explanations surfaced a linked fraud burst**: near-identical SHAP signatures (same
  device, amount, email domain) flagged separate claims as one actor.



Current benchmark to beat: **PR-AUC 0.577** (temporal validation).

## Roadmap

- [x] Data pipeline: IEEE-CIS transaction/identity merge (left join, missingness kept)
- [x] Naive XGBoost baseline + honest evaluation (PR-AUC, confusion matrix)
- [x] Threshold-moving + cost-sensitive operating-point analysis
- [x] Temporal evaluation harness (train/val/test split by time — no future leakage)
- [x] Imbalance handling — compare class weighting, SMOTE (incl. the leakage bug), threshold-moving
- [x] Feature engineering — categoricals + per-card aggregates (lift the PR curve)
- [x] Probability calibration (Platt / isotonic) + reliability diagram
- [X] SHAP explainability — global summary + per-claim waterfall plots (TreeSHAP)
- [ ] Written comparison report (PR curves, cost-based eval)
- [ ] **FastAPI inference service** — POST a claim → score + decision + SHAP reasons
- [ ] Optional Streamlit UI over the API
- [ ] Port pipeline to a second, insurance-shaped dataset

## Tech stack

`Python 3.12` · `uv` · `pandas` · `scikit-learn` · `XGBoost` · `imbalanced-learn` ·
`SHAP` · `matplotlib` · `FastAPI` · `Streamlit (optional UI)`

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
api/           # FastAPI inference service (planned)
models/        # persisted model artifacts (gitignored)
reports/       # PR curves, comparison report
README.md
```

## License

MIT
