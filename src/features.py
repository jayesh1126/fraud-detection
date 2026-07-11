"""Feature pipeline: fit on train, transform anything. Single source of truth."""
import numpy as np
import pandas as pd


def categorical_columns(df):
    return df.select_dtypes(include=["object", "str", "category"]).columns.tolist()


def fit_categories(train_df, cols):
    """Learn category vocabulary FROM TRAIN ONLY."""
    return {c: pd.CategoricalDtype(train_df[c].dropna().unique()) for c in cols}


def apply_categories(df_, vocab):
    """Cast with train's vocabulary; unseen values -> NaN (missing)."""
    out = df_.copy()
    for c, dtype in vocab.items():
        out[c] = pd.Categorical(out[c], categories=dtype.categories)
    return out


def add_time_features(df_):
    out = df_.copy()
    out["hour"] = (out["TransactionDT"] // 3600) % 24
    out["dow"] = (out["TransactionDT"] // 86400) % 7
    return out


def feature_list(df_):
    """Final model features: numeric + categoricals + time (nb 04 decision)."""
    num = df_.select_dtypes(include=[np.number]).columns.drop(
        ["isFraud", "TransactionID"]
    ).tolist()
    return num + categorical_columns(df_)
