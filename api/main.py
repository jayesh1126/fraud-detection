"""Fraud scoring service.  Run from repo root:
   uv run uvicorn api.main:app --reload
"""
import json
import sys
from contextlib import asynccontextmanager
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT / "src"))
from features import apply_categories, add_time_features

ART = {}  # loaded artifacts


@asynccontextmanager
async def lifespan(app):
    models = ROOT / "models"
    booster = xgb.Booster()
    booster.load_model(models / "model.json")
    meta = json.loads((models / "meta.json").read_text())
    ART.update(
        booster=booster,
        vocab=joblib.load(models / "vocab.joblib"),
        features=meta["features"],
        threshold=meta["threshold"],
        demo=pd.read_parquet(models / "demo_claims.parquet"),
    )
    yield


app = FastAPI(title="Fraud Triage API", lifespan=lifespan)


class Claim(BaseModel):
    features: dict[str, float | int | str | None]


def _score_frame(frame: pd.DataFrame) -> dict:
    """Score ONE claim row: probability, decision, top SHAP reasons."""
    frame = frame.reindex(columns=ART["features"])       # full schema FIRST: missing -> NaN
    frame = add_time_features(apply_categories(frame, ART["vocab"]))
    frame = frame[ART["features"]]                        # restore exact column order

    dm = xgb.DMatrix(frame, enable_categorical=True)

    contribs = ART["booster"].predict(dm, pred_contribs=True)[0]
    phi, base = contribs[:-1], float(contribs[-1])       # last slot = baseline
    margin = base + phi.sum()
    proba = float(1 / (1 + np.exp(-margin)))             # sigmoid(log-odds)

    top = np.argsort(-np.abs(phi))[:8]
    reasons = [{
        "feature": ART["features"][i],
        "value": None if pd.isna(v := frame.iloc[0, i]) else str(v),
        "contribution_logodds": round(float(phi[i]), 4),
        "direction": "toward fraud" if phi[i] > 0 else "toward legit",
    } for i in top if abs(phi[i]) > 1e-6]

    return {
        "fraud_score": round(proba, 6),
        "decision": "investigate" if proba >= ART["threshold"] else "clear",
        "threshold": round(ART["threshold"], 6),
        "baseline_logodds": round(base, 4),
        "top_reasons": reasons,
    }


@app.get("/health")
def health():
    return {"status": "ok", "n_features": len(ART["features"])}


@app.post("/score")
def score(claim: Claim):
    known = {k: v for k, v in claim.features.items() if k in ART["features"]}
    if not known:
        raise HTTPException(422, "no recognized feature names in payload")
    return _score_frame(pd.DataFrame([known]))


@app.get("/demo_claims")
def demo_claims():
    """IDs + labels of stored demo claims (for the UI's picker)."""
    d = ART["demo"]
    return d[["TransactionID", "isFraud"]].to_dict(orient="records")


@app.get("/demo_score/{transaction_id}")
def demo_score(transaction_id: int):
    d = ART["demo"]
    row = d[d["TransactionID"] == transaction_id]
    if row.empty:
        raise HTTPException(404, "unknown demo claim")
    out = _score_frame(row.drop(columns=["isFraud"]))
    out["actually_fraud"] = bool(row["isFraud"].iloc[0])   # demo only!
    return out
