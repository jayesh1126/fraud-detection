"""Fit the full pipeline on the train window and persist serving artifacts.

Run from repo root:  uv run python src/pipeline.py
"""
import json
from pathlib import Path

import joblib
from xgboost import XGBClassifier

from data_prep import load_merged, temporal_split
from eval_harness import evaluate, pick_threshold_for_budget
from features import (categorical_columns, fit_categories,
                      apply_categories, add_time_features, feature_list)

BUDGET = 3000          # investigator capacity per val-window (nb 02 decision)
EXPECTED_PR_AUC = 0.577   # nb 04/05/06 benchmark — the repro check


def main(data_dir="data/raw", out_dir="models"):
    df = load_merged(data_dir)
    train, val, _test = temporal_split(df)

    vocab = fit_categories(train, categorical_columns(train))
    train = add_time_features(apply_categories(train, vocab))
    val = add_time_features(apply_categories(val, vocab))
    feats = feature_list(train)

    model = XGBClassifier(tree_method="hist", eval_metric="logloss", n_jobs=-1,
                          random_state=42, enable_categorical=True)
    model.fit(train[feats], train["isFraud"])

    proba_val = model.predict_proba(val[feats])[:, 1]
    r = evaluate(val["isFraud"], proba_val, name="pipeline val check")
    assert abs(r["pr_auc"] - EXPECTED_PR_AUC) < 0.005, f"repro check failed: {r}"
    threshold = pick_threshold_for_budget(proba_val, BUDGET)

    out = Path(out_dir)
    out.mkdir(exist_ok=True)
    model.save_model(out / "model.json")            # xgboost native format
    joblib.dump(vocab, out / "vocab.joblib")        # category vocabulary
    (out / "meta.json").write_text(json.dumps({
        "features": feats,
        "threshold": threshold,
        "val_pr_auc": r["pr_auc"],
    }, indent=2))
    # 500 real val claims for the demo UI (gitignored with models/)
    val.sample(500, random_state=0).to_parquet(out / "demo_claims.parquet")

    print(f"artifacts written to {out}/  |  val PR-AUC {r['pr_auc']}  "
          f"threshold {threshold:.4f}")


if __name__ == "__main__":
    main()
