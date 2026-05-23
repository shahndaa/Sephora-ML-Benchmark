# Sephora ML Benchmark — Scratch vs Library

A production‑grade benchmark comparing **8 machine learning algorithms** implemented **from scratch** (pure NumPy) against their **library equivalents** (scikit‑learn, XGBoost). The target is `is_recommended` – a binary classification task using real Sephora product and review data.

## 🔍 Why this project matters

- **No data leakage** – The high‑correlation `rating` column is **dropped** so models must learn from actual review metadata.
- **Fair comparison** – Every algorithm uses the same train/val/test split and the same 6 evaluation metrics.
- **Educational** – See exactly how each algorithm works under the hood, and understand why library versions sometimes perform better (speed, regularisation, hyperparameters).
- **Complete pipeline** – From raw CSV ingestion → feature engineering → model training → interactive dashboard.

## 🧠 Algorithms covered

| Family | Models |
|---------|--------|
| Linear | Linear Regression (thresholded), Logistic Regression |
| Tree‑based | Decision Tree, Random Forest |
| Distance / Prob. | K‑Nearest Neighbours, Gaussian Naive Bayes |
| Advanced | XGBoost / Gradient Boosting, SVM (Linear) |

Each model appears twice: **scratch** and **library** (sklearn / XGBoost).

## 📊 Evaluation metrics

- Accuracy
- Macro F1‑score
- Macro Precision / Recall
- Log‑Loss (cross‑entropy)
- ROC‑AUC
- Training time & overfitting gap (train vs test accuracy)

## 🗂️ Data pipeline (fully automated)

1. **Load** – 5 review CSVs + product info (total ~1.09M reviews)
2. **Merge** – on `product_id`
3. **Clean** – drop missing targets, remove text & ID columns
4. **Engineer** – helpfulness ratio, log‑feedback count, skin tone/type encodings, category dummies
5. **Leakage fix** – `rating` is **removed** (its correlation with `is_recommended` is 0.87 → would otherwise give fake high accuracy)
6. **Save** – processed features into SQLite (`sephora.db`)

## 🚀 How to run the full project

### 1. Clone the repository

```bash
git clone https://github.com/yourusername/sephora-ml-benchmark.git
cd sephora-ml-benchmark
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Download the Sephora dataset

You need the following CSV files (available from the Sephora product review dataset on Kaggle or similar):

- `product_info.csv`
- `reviews_0-250.csv`
- `reviews_250-500.csv`
- `reviews_500-750.csv`
- `reviews_750-1250.csv`
- `reviews_1250-end.csv`

Place them inside the `data/` folder.

### 4. Run the complete pipeline

```bash
python run_all.py data/
```

This will:

- Preprocess the data and create `sephora.db`
- Train all 8 algorithms (both scratch and library)
- Save all metrics and learning curves into the database
- Automatically start the Flask web dashboard at `http://localhost:5000`

### 5. Explore the dashboard

The dashboard shows:

- Live comparison cards (accuracy bars, log‑loss, learning curves)
- Leaderboard ranked by test accuracy
- Accuracy / Log‑Loss bar charts
- Winner analysis (XGBoost library) with detailed explanation
- Algorithm‑by‑algorithm educational breakdown

## 📈 Key results (example)

| Model | Accuracy | Macro F1 | Log‑Loss | Train time (scratch vs lib) |
|--------|----------|----------|----------|------------------------------|
| XGBoost (library) | 0.926 | 0.842 | 0.261 | 2.1s / 0.8s |
| Random Forest (lib) | 0.915 | 0.815 | 0.289 | 18s / 0.5s |
| Logistic Regression | 0.886 | 0.778 | 0.353 | 0.4s / 0.06s |
| KNN (scratch) | 0.875 | 0.751 | 0.304 | 92s / 0.3s |

Full results are generated on your machine after training.

## 🛠️ Technologies used

- **Backend** – Flask, SQLAlchemy, NumPy, scikit‑learn, XGBoost
- **Frontend** – HTML5 / CSS3, vanilla JavaScript (no external chart libs, all custom SVG)
- **Database** – SQLite
- **Visualisation** – Animated accuracy bars, learning curves (train/val), leaderboard

## 📁 Project structure

```text
.
├── app.py                     # Flask server + API endpoints
├── run_all.py                 # Orchestrates the entire pipeline
├── requirements.txt
├── .gitignore
├── README.md
├── scripts/                   # All ML training modules
│   ├── preprocessing.py       # Data ingestion, cleaning, feature engineering
│   ├── model_01_linear_logistic.py
│   ├── model_02_tree_forest.py
│   ├── model_03_knn_naivebayes.py
│   ├── model_04_xgboost_svm.py
│   └── utils.py               # Shared helpers (split, evaluate, DB save)
├── templates/
│   └── index.html             # Interactive dashboard
└── data/                      # Place your CSVs here (not tracked by git)
```

## 📜 License

MIT – you are free to use, modify, and distribute this project for educational or commercial purposes.

## 🙌 Acknowledgements

- Sephora product review dataset (publicly available)
- Inspired by the need to demonstrate true ML understanding – implementing algorithms from scratch while comparing against battle‑tested libraries.
