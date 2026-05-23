"""
utils.py  —  Shared helpers for all model scripts.
Includes proper train/val/test split and cross-validation to detect overfitting.
"""
import os, time, json
import numpy as np
import pandas as pd
from sqlalchemy import create_engine, text
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score,
    recall_score, log_loss, roc_auc_score,
)

BASE_DIR     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH      = os.path.join(BASE_DIR, "sephora.db")
ENGINE       = create_engine(f"sqlite:///{DB_PATH}")
TARGET       = "is_recommended"
SAMPLE_N     = 80_000   # reduced to avoid overfitting risk from huge samples
RANDOM_STATE = 42


def load_features(sample_n=SAMPLE_N):
    print(f"📊 Loading {sample_n:,} rows from DB ...")
    df = pd.read_sql(
        f"SELECT * FROM processed_reviews LIMIT {sample_n}", ENGINE
    )
    y = df[TARGET]
    X = df.drop(columns=[TARGET]).select_dtypes(include=[np.number])
    print(f"   Features: {X.shape[1]}  |  Rows: {len(X):,}")
    print(f"   Class balance: {y.mean():.1%} positive")
    return X, y


def split(X, y, test_size=0.2, val_size=0.1):
    """
    3-way split: train / validation / test
    val_size is fraction of total data (not of remaining after test split)
    """
    # First split off test set
    X_temp, X_te, y_temp, y_te = train_test_split(
        X, y, test_size=test_size, random_state=RANDOM_STATE, stratify=y
    )
    # Then split validation from remaining
    val_fraction = val_size / (1.0 - test_size)
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_temp, y_temp, test_size=val_fraction,
        random_state=RANDOM_STATE, stratify=y_temp
    )
    print(f"   Train: {len(X_tr):,}  |  Val: {len(X_val):,}  |  Test: {len(X_te):,}")
    return X_tr, X_val, X_te, y_tr, y_val, y_te


def simple_split(X, y, test_size=0.2):
    """Simple 80/20 split for scratch implementations."""
    return train_test_split(
        X, y, test_size=test_size, random_state=RANDOM_STATE, stratify=y
    )


def evaluate(model_name: str, y_test, y_pred,
             y_train=None, y_pred_train=None,
             y_prob=None, train_scores=None) -> dict:
    """
    Compute all 6 metrics + overfit gap (train acc - test acc).
    Prints a clear summary.
    """
    y_test = np.array(y_test)
    y_pred = np.array(y_pred)

    acc      = accuracy_score(y_test, y_pred)
    f1_mac   = f1_score(y_test, y_pred, average="macro", zero_division=0)
    prec     = precision_score(y_test, y_pred, average="macro", zero_division=0)
    rec      = recall_score(y_test, y_pred, average="macro", zero_division=0)
    ll       = None
    auc      = None

    if y_prob is not None:
        y_prob_arr = np.array(y_prob)
        y_prob_arr = np.clip(y_prob_arr, 1e-7, 1 - 1e-7)
        ll  = log_loss(y_test, y_prob_arr)
        auc = roc_auc_score(y_test, y_prob_arr)

    # Overfitting gap
    train_acc = None
    if y_train is not None and y_pred_train is not None:
        train_acc = accuracy_score(np.array(y_train), np.array(y_pred_train))

    gap_str = ""
    if train_acc is not None:
        gap = train_acc - acc
        gap_str = f"  Overfit Gap : {gap:+.4f}  (train={train_acc:.4f}, test={acc:.4f})"

    print(f"\n{'='*56}")
    print(f"  {model_name}")
    print(f"  Accuracy     : {acc:.4f}  ({acc*100:.2f}%)")
    print(f"  Macro F1     : {f1_mac:.4f}")
    print(f"  Macro Prec   : {prec:.4f}")
    print(f"  Macro Recall : {rec:.4f}")
    if ll  is not None: print(f"  Log-Loss     : {ll:.4f}  ← lower=better")
    if auc is not None: print(f"  ROC-AUC      : {auc:.4f}")
    if gap_str:         print(gap_str)
    print(f"{'='*56}")

    row = dict(
        model=model_name,
        accuracy=round(acc, 4),
        f1_macro=round(f1_mac, 4),
        precision_macro=round(prec, 4),
        recall_macro=round(rec, 4),
        log_loss=round(ll, 4)    if ll  is not None else None,
        roc_auc=round(auc, 4)   if auc is not None else None,
        train_accuracy=round(train_acc, 4) if train_acc is not None else None,
        overfit_gap=round(train_acc - acc, 4) if train_acc is not None else None,
    )
    if train_scores:
        row["train_scores_json"] = json.dumps(train_scores)
    else:
        row["train_scores_json"] = None
    return row


def cv_score(model, X, y, cv=5, metric="accuracy"):
    """Quick cross-validation to check generalisation."""
    from sklearn.model_selection import StratifiedKFold
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=RANDOM_STATE)
    scores = cross_val_score(model, X, y, cv=skf, scoring=metric, n_jobs=-1)
    return scores.mean(), scores.std()


def save_results(results: list, table="model_results"):
    df = pd.DataFrame(results)
    df.to_sql(table, ENGINE, if_exists="append", index=False)
    print(f"\n💾 Saved {len(df)} result(s) → '{table}'")


def clear_results(table="model_results"):
    with ENGINE.connect() as conn:
        conn.execute(text(f"DROP TABLE IF EXISTS {table}"))
        conn.commit()
    print(f"🗑  Cleared '{table}'")
