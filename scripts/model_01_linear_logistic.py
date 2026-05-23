"""
model_01_linear_logistic.py
============================
LINEAR REGRESSION  +  LOGISTIC REGRESSION
  • Both from scratch AND sklearn
  • Proper 3-way split (train/val/test) — val used for learning curves
  • Reports overfit gap (train acc - test acc)
  • All 6 metrics: Accuracy, Macro-F1, Precision, Recall, Log-Loss, ROC-AUC
"""
import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from utils import load_features, split, simple_split, evaluate, save_results
from sklearn.linear_model import LinearRegression as SKLinear
from sklearn.linear_model import LogisticRegression as SKLogistic
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score


# ══════════════════════════════════════════════════════════════
#  SCRATCH: Linear Regression (Normal Equation)
# ══════════════════════════════════════════════════════════════
class LinearRegressionScratch:
    """
    OLS via Normal Equation:  θ = (XᵀX)⁻¹ Xᵀy
    Thresholded at 0.5 for binary classification.
    Uses pseudoinverse for numerical stability.
    """
    def __init__(self):
        self.theta = None

    def fit(self, X, y):
        X_b = np.c_[np.ones(len(X)), X]
        self.theta = np.linalg.pinv(X_b.T @ X_b) @ X_b.T @ y

    def predict_raw(self, X):
        X_b = np.c_[np.ones(len(X)), X]
        return X_b @ self.theta

    def predict(self, X, threshold=0.5):
        return (self.predict_raw(X) >= threshold).astype(int)

    def predict_proba(self, X):
        return np.clip(self.predict_raw(X), 0, 1)


# ══════════════════════════════════════════════════════════════
#  SCRATCH: Logistic Regression (Gradient Descent with L2)
# ══════════════════════════════════════════════════════════════
class LogisticRegressionScratch:
    """
    Binary Logistic Regression via batch gradient descent.
    Includes L2 regularization (lambda_) to prevent overfitting.
    Records train and val accuracy every 'record_every' epochs
    for the learning curve.
    """
    def __init__(self, lr=0.1, n_iter=300, lambda_=0.01):
        self.lr      = lr
        self.n_iter  = n_iter
        self.lambda_ = lambda_   # L2 regularization strength
        self.w       = None
        self.b       = 0.0

    @staticmethod
    def _sigmoid(z):
        return 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))

    def fit(self, X_tr, y_tr, X_val=None, y_val=None, record_every=30):
        n, m = X_tr.shape
        self.w = np.zeros(m)
        self.b = 0.0
        y_tr   = np.array(y_tr)
        self.train_curve = []
        self.val_curve   = []

        for i in range(self.n_iter):
            z   = X_tr @ self.w + self.b
            p   = self._sigmoid(z)
            err = p - y_tr

            # Gradient with L2 regularization on weights (not bias)
            dw = (X_tr.T @ err) / n + (self.lambda_ / n) * self.w
            db = err.mean()

            self.w -= self.lr * dw
            self.b -= self.lr * db

            if (i + 1) % record_every == 0:
                tr_pred = (self._sigmoid(X_tr @ self.w + self.b) >= .5).astype(int)
                self.train_curve.append(round(accuracy_score(y_tr, tr_pred), 4))
                if X_val is not None:
                    self.val_curve.append(round(
                        accuracy_score(y_val, self.predict(X_val)), 4
                    ))

    def predict_proba(self, X):
        return self._sigmoid(np.array(X) @ self.w + self.b)

    def predict(self, X, threshold=0.5):
        return (self.predict_proba(X) >= threshold).astype(int)


# ══════════════════════════════════════════════════════════════
#  Learning curve helper for sklearn models
# ══════════════════════════════════════════════════════════════
def lc_sklearn(model_cls, X_tr, y_tr, X_val, y_val, n_pts=8, linear=False, **kw):
    train_s, val_s = [], []
    for frac in np.linspace(0.1, 1.0, n_pts):
        n = max(10, int(len(X_tr) * frac))
        m = model_cls(**kw)
        m.fit(X_tr[:n], y_tr[:n])
        if linear:
            tr_p = (m.predict(X_tr[:n]) >= .5).astype(int)
            va_p = (m.predict(X_val) >= .5).astype(int)
        else:
            tr_p = m.predict(X_tr[:n])
            va_p = m.predict(X_val)
        train_s.append(round(accuracy_score(y_tr[:n], tr_p), 4))
        val_s.append(round(accuracy_score(y_val, va_p), 4))
    return train_s, val_s


# ══════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════
def main():
    print("\n" + "█"*58)
    print("  MODEL 01 — Linear Regression + Logistic Regression")
    print("█"*58)

    X, y = load_features()
    X_tr, X_val, X_te, y_tr, y_val, y_te = split(X, y)

    # Standardize (required for gradient descent)
    scaler  = StandardScaler()
    X_tr_s  = scaler.fit_transform(X_tr)
    X_val_s = scaler.transform(X_val)
    X_te_s  = scaler.transform(X_te)

    y_tr_np, y_val_np, y_te_np = y_tr.values, y_val.values, y_te.values

    results = []

    # ── 1. Linear Regression — Scratch ──────────────────────
    print("\n[1/4] Linear Regression — Scratch (Normal Equation) ...")
    t0 = time.time()
    m = LinearRegressionScratch()
    m.fit(X_tr_s, y_tr_np)
    y_pred_tr  = m.predict(X_tr_s)
    y_pred_te  = m.predict(X_te_s)
    y_prob_te  = m.predict_proba(X_te_s)
    row = evaluate("Linear Regression (Scratch)", y_te_np, y_pred_te,
                   y_train=y_tr_np, y_pred_train=y_pred_tr, y_prob=y_prob_te,
                   train_scores={"train":[round(accuracy_score(y_tr_np,y_pred_tr),4)],
                                 "val":[round(accuracy_score(y_val_np,m.predict(X_val_s)),4)]})
    row["implementation"] = "scratch"
    row["training_time"]  = round(time.time() - t0, 3)
    results.append(row)

    # ── 2. Linear Regression — sklearn ──────────────────────
    print("\n[2/4] Linear Regression — sklearn ...")
    t0 = time.time()
    tr_s, va_s = lc_sklearn(SKLinear, X_tr_s, y_tr_np, X_val_s, y_val_np, linear=True)
    m2 = SKLinear(); m2.fit(X_tr_s, y_tr_np)
    y_raw      = m2.predict(X_te_s)
    y_pred_te  = (y_raw >= .5).astype(int)
    y_pred_tr2 = (m2.predict(X_tr_s) >= .5).astype(int)
    y_prob_te  = np.clip(y_raw, 0, 1)
    row = evaluate("Linear Regression (sklearn)", y_te_np, y_pred_te,
                   y_train=y_tr_np, y_pred_train=y_pred_tr2, y_prob=y_prob_te,
                   train_scores={"train": tr_s, "val": va_s})
    row["implementation"] = "sklearn"
    row["training_time"]  = round(time.time() - t0, 3)
    results.append(row)

    # ── 3. Logistic Regression — Scratch ────────────────────
    print("\n[3/4] Logistic Regression — Scratch (GD + L2) ...")
    t0 = time.time()
    m3 = LogisticRegressionScratch(lr=0.1, n_iter=300, lambda_=0.01)
    m3.fit(X_tr_s, y_tr_np, X_val=X_val_s, y_val=y_val_np, record_every=30)
    y_pred_te = m3.predict(X_te_s)
    y_pred_tr = m3.predict(X_tr_s)
    y_prob_te = m3.predict_proba(X_te_s)
    row = evaluate("Logistic Regression (Scratch)", y_te_np, y_pred_te,
                   y_train=y_tr_np, y_pred_train=y_pred_tr, y_prob=y_prob_te,
                   train_scores={"train": m3.train_curve, "val": m3.val_curve})
    row["implementation"] = "scratch"
    row["training_time"]  = round(time.time() - t0, 3)
    results.append(row)

    # ── 4. Logistic Regression — sklearn ────────────────────
    print("\n[4/4] Logistic Regression — sklearn (L2 regularized) ...")
    t0 = time.time()
    tr_s, va_s = lc_sklearn(SKLogistic, X_tr_s, y_tr_np, X_val_s, y_val_np,
                              max_iter=300, C=1.0, random_state=42)
    m4 = SKLogistic(max_iter=300, C=1.0, random_state=42)
    m4.fit(X_tr_s, y_tr_np)
    y_pred_te = m4.predict(X_te_s)
    y_pred_tr = m4.predict(X_tr_s)
    y_prob_te = m4.predict_proba(X_te_s)[:, 1]
    row = evaluate("Logistic Regression (sklearn)", y_te_np, y_pred_te,
                   y_train=y_tr_np, y_pred_train=y_pred_tr, y_prob=y_prob_te,
                   train_scores={"train": tr_s, "val": va_s})
    row["implementation"] = "sklearn"
    row["training_time"]  = round(time.time() - t0, 3)
    results.append(row)

    save_results(results)
    print("\n✅ Model 01 complete.")


if __name__ == "__main__":
    main()
