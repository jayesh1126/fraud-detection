"""Load + temporally split the IEEE-CIS fraud data. Single source of truth."""
from pathlib import Path
import numpy as np
import pandas as pd


def load_merged(data_dir="data/raw"):
    """All transactions, left-joined with identity (missingness kept as signal)."""
    d = Path(data_dir)
    txn = pd.read_csv(d / "train_transaction.csv")
    idn = pd.read_csv(d / "train_identity.csv")
    return txn.merge(idn, on="TransactionID", how="left")


def temporal_split(df, frac_train=0.60, frac_val=0.20):
    """Time-ordered train/val/test. val = decisions, test = touch once."""
    df = df.sort_values("TransactionDT").reset_index(drop=True)
    n = len(df)
    i_val, i_test = int(n * frac_train), int(n * (frac_train + frac_val))
    return df.iloc[:i_val], df.iloc[i_val:i_test], df.iloc[i_test:]


def numeric_features(df):
    """Baseline feature set: numeric columns minus label and id."""
    return df.select_dtypes(include=[np.number]).columns.drop(
        ["isFraud", "TransactionID"]
    )
