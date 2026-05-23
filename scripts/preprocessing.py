"""
preprocessing.py
================
Loads all Sephora CSVs, fixes the DATA LEAKAGE problem by dropping 'rating'
(which has 0.87 correlation with the target and artificially inflates accuracy),
engineers proper features, and saves to SQLite.

WHY DROP RATING?
  rating vs is_recommended correlation = 0.87
  rating=5 → 99.9% recommend,  rating=1 → 0.6% recommend
  Keeping rating means the model is NOT learning from customer data —
  it's just reading the star rating. That's not ML, that's a lookup table.

Run:  python scripts/preprocessing.py data/
"""

import os, sys
import pandas as pd
import numpy as np
from sqlalchemy import create_engine

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH  = os.path.join(BASE_DIR, "sephora.db")
ENGINE   = create_engine(f"sqlite:///{DB_PATH}")

REVIEW_FILES = [
    "reviews_0-250.csv",
    "reviews_250-500.csv",
    "reviews_500-750.csv",
    "reviews_750-1250.csv",
    "reviews_1250-end.csv",
]


def load(data_dir):
    print("📂 Loading reviews ...")
    dfs = []
    for f in REVIEW_FILES:
        path = os.path.join(data_dir, f)
        df   = pd.read_csv(path, low_memory=False)
        dfs.append(df)
        print(f"   {f}: {len(df):,} rows")
    reviews = pd.concat(dfs, ignore_index=True)
    print(f"   Total: {len(reviews):,} rows\n")

    print("📂 Loading product info ...")
    prod = pd.read_csv(os.path.join(data_dir, "product_info.csv"), low_memory=False)
    print(f"   product_info: {len(prod):,} rows\n")
    return reviews, prod


def merge(reviews, prod):
    print("🔗 Merging reviews with product info ...")
    keep_cols = [
        "product_id", "loves_count", "limited_edition",
        "new", "online_only", "out_of_stock",
        "sephora_exclusive", "primary_category",
    ]
    # Only keep columns that exist in product info
    keep_cols = [c for c in keep_cols if c in prod.columns]
    slim = prod[keep_cols].drop_duplicates("product_id")
    df   = reviews.merge(slim, on="product_id", how="left")
    print(f"   Merged shape: {df.shape}\n")
    return df


def preprocess(df):
    print("🧹 Preprocessing ...")

    # ── 1. Drop rows with missing target ─────────────────────────────────────
    before = len(df)
    df = df.dropna(subset=["is_recommended"]).copy()
    df["is_recommended"] = df["is_recommended"].astype(int)
    print(f"   Dropped {before - len(df):,} rows with missing target")
    print(f"   Remaining: {len(df):,} rows")

    # ── 2. DROP RATING — this is the key fix ─────────────────────────────────
    # rating has correlation 0.87 with is_recommended.
    # Keeping it = data leakage = fake high accuracy.
    # We want models to learn from actual review metadata.
    print("\n   ⚠️  DROPPING 'rating' — correlation with target = 0.87 (data leakage)")
    if "rating" in df.columns:
        df.drop(columns=["rating"], inplace=True)

    # ── 3. Drop irrelevant / leaky columns ───────────────────────────────────
    drop_cols = [
        "Unnamed: 0",
        "author_id",
        "submission_time",
        "review_text",        # free text, needs NLP pipeline
        "review_title",       # free text
        "product_id",         # high-cardinality ID
        "product_name",       # high-cardinality text
        "brand_name",         # high-cardinality text
    ]
    df.drop(columns=[c for c in drop_cols if c in df.columns], inplace=True)

    # ── 4. Feature engineering ────────────────────────────────────────────────
    # helpfulness_ratio: what fraction of voters found review helpful?
    df["helpfulness_ratio"] = np.where(
        df["total_feedback_count"] > 0,
        df["total_pos_feedback_count"] / df["total_feedback_count"],
        0.0
    )

    # log_feedback_count: log-transform to reduce right skew
    df["log_feedback_count"] = np.log1p(df["total_feedback_count"])

    # Fill helpfulness NaN with median
    df["helpfulness"] = df["helpfulness"].fillna(df["helpfulness"].median())

    # ── 5. Encode categorical features ───────────────────────────────────────
    skin_tone_map = {
        "fair": 1, "light": 2, "medium": 3, "tan": 4,
        "olive": 5, "deep": 6, "rich": 7, "ebony": 8,
    }
    skin_type_map = {"dry": 0, "normal": 1, "combination": 2, "oily": 3}

    df["skin_tone_enc"] = (
        df["skin_tone"].str.lower().str.strip()
        .map(skin_tone_map).fillna(0).astype(int)
    )
    df["skin_type_enc"] = (
        df["skin_type"].str.lower().str.strip()
        .map(skin_type_map).fillna(-1).astype(int)
    )

    # Drop original string categoricals
    df.drop(columns=["skin_tone", "skin_type", "eye_color", "hair_color"],
            inplace=True, errors="ignore")

    # ── 6. Product category one-hot (top 10 + Other) ─────────────────────────
    if "primary_category" in df.columns:
        top_cats = df["primary_category"].value_counts().nlargest(10).index
        df["primary_category"] = df["primary_category"].where(
            df["primary_category"].isin(top_cats), other="Other"
        )
        dummies = pd.get_dummies(df["primary_category"], prefix="cat", dtype=int)
        df = pd.concat([df, dummies], axis=1)
        df.drop(columns=["primary_category"], inplace=True)

    # ── 7. Numeric imputation ─────────────────────────────────────────────────
    if "price_usd" in df.columns:
        df["price_usd"] = df["price_usd"].fillna(df["price_usd"].median())

    if "loves_count" in df.columns:
        df["loves_count"] = df["loves_count"].fillna(0)

    for col in ["limited_edition", "new", "online_only",
                "out_of_stock", "sephora_exclusive"]:
        if col in df.columns:
            df[col] = df[col].fillna(0).astype(int)

    # ── 8. Final numeric only ─────────────────────────────────────────────────
    df = df.select_dtypes(include=[np.number]).copy()

    print(f"\n   ✅ Final shape: {df.shape}")
    print(f"   Features: {[c for c in df.columns if c != 'is_recommended']}")
    print(f"\n   Target distribution:")
    vc = df["is_recommended"].value_counts(normalize=True)
    print(f"     Class 1 (Recommend)    : {vc.get(1, 0):.1%}")
    print(f"     Class 0 (Not Recommend): {vc.get(0, 0):.1%}")
    return df


def save(df):
    print("\n💾 Saving to sephora.db ...")

    # Drop existing table and recreate
    with ENGINE.connect() as conn:
        conn.execute(
            __import__("sqlalchemy").text("DROP TABLE IF EXISTS processed_reviews")
        )
        conn.execute(
            __import__("sqlalchemy").text("DROP TABLE IF EXISTS column_meta")
        )
        conn.commit()

    # Save in chunks to avoid SQLite parameter limit (999 params max)
    # SQLite limit = 999 params, so chunk_size = floor(999 / n_columns)
    n_cols = len(df.columns)
    chunk_size = max(1, 900 // n_cols)  # safe chunk size
    print(f"   Writing {len(df):,} rows in chunks of {chunk_size:,} ...")

    df.to_sql(
        "processed_reviews", ENGINE,
        if_exists="replace",
        index=False,
        chunksize=chunk_size,
        method="multi"
    )

    # Save column metadata
    pd.DataFrame({
        "column": df.columns.tolist(),
        "dtype":  df.dtypes.astype(str).values,
    }).to_sql("column_meta", ENGINE, if_exists="replace", index=False)

    print(f"   ✅ Saved {len(df):,} rows to sephora.db")
    print(f"   Table: processed_reviews  |  Columns: {n_cols}")


if __name__ == "__main__":
    data_dir = sys.argv[1] if len(sys.argv) > 1 else os.path.join(BASE_DIR, "data")

    if not os.path.isdir(data_dir):
        print(f"❌  Data directory not found: {data_dir}")
        print(f"    Usage: python scripts/preprocessing.py path/to/data/")
        sys.exit(1)

    reviews, prod = load(data_dir)
    df = merge(reviews, prod)
    df = preprocess(df)
    save(df)

    print("\n" + "="*55)
    print("  ✅ Preprocessing complete!")
    print("  Next: python scripts/model_01_linear_logistic.py")
    print("="*55)
