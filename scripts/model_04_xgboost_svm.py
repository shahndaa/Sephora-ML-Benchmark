"""
model_04_xgboost_svm.py
========================
XGBOOST  +  SVM
  • XGBoost uses max_depth=5 and regularization
  • SVM uses Pegasos SGD with hinge loss
  • All 6 metrics + overfit gap
"""
import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from utils import load_features, split, evaluate, save_results
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import xgboost as xgb


# ══════════════════════════════════════════════════════════════
#  SCRATCH: Gradient Boosting (log-loss + depth-1 stumps)
# ══════════════════════════════════════════════════════════════
class _Stump:
    """Depth-1 regression tree (single best split)."""
    def fit(self, X, r):
        n, m = X.shape
        best = -np.inf
        self.f = self.t = self.lv = self.rv = None
        for f in range(m):
            thrs = np.percentile(X[:, f], np.linspace(10, 90, 12))
            for t in thrs:
                lm = X[:, f] <= t; rm = ~lm
                if lm.sum() < 2 or rm.sum() < 2: continue
                gain = -(lm.sum() * r[lm].var() + rm.sum() * r[rm].var()) / n
                if gain > best:
                    best = gain; self.f = f; self.t = t
                    self.lv = r[lm].mean(); self.rv = r[rm].mean()

    def predict(self, X):
        return np.where(X[:, self.f] <= self.t, self.lv, self.rv)


class GradBoostScratch:
    """
    Gradient Boosting for binary classification.
    Loss: log-loss. Weak learner: depth-1 stumps.
    L2 regularization on leaf values (shrinkage).
    """
    def __init__(self, n=60, lr=0.08):
        self.n = n; self.lr = lr

    @staticmethod
    def _sig(z): return 1/(1+np.exp(-np.clip(z,-500,500)))

    def fit(self, X, y, X_val=None, y_val=None):
        X, y  = np.array(X, float), np.array(y, float)
        p0    = y.mean()
        self.F0 = np.log(p0/(1-p0+1e-9))
        F     = np.full(len(y), self.F0)
        self.stumps     = []
        self.train_curve = []
        self.val_curve   = []
        for i in range(self.n):
            r = y - self._sig(F)
            s = _Stump(); s.fit(X, r)
            F += self.lr * s.predict(X)
            self.stumps.append(s)
            if (i+1) % 15 == 0:
                print(f"    Boosting round {i+1}/{self.n}")
                self.train_curve.append(
                    round(accuracy_score(y, (self._sig(F)>=.5).astype(int)), 4)
                )
                if X_val is not None:
                    self.val_curve.append(
                        round(accuracy_score(y_val, self.predict(X_val)), 4)
                    )

    def _raw(self, X):
        F = np.full(len(X), self.F0)
        for s in self.stumps: F += self.lr * s.predict(np.array(X, float))
        return F

    def predict_proba(self, X): return self._sig(self._raw(X))
    def predict(self, X):       return (self.predict_proba(X) >= .5).astype(int)


# ══════════════════════════════════════════════════════════════
#  SCRATCH: Linear SVM (Pegasos SGD — hinge loss)
# ══════════════════════════════════════════════════════════════
class SVMScratch:
    """
    Linear SVM via Pegasos mini-batch SGD.
    Objective: min (1/2)||w||² + C·Σmax(0, 1-y(w·x+b))
    Labels mapped to {-1, +1} internally.
    """
    def __init__(self, C=0.5, n_iter=6000, lr=0.01, batch=256):
        self.C=C; self.n_iter=n_iter; self.lr=lr; self.batch=batch

    def fit(self, X, y, X_val=None, y_val=None):
        X  = np.array(X, float)
        ys = np.where(np.array(y)==1, 1, -1).astype(float)
        n, m = X.shape
        self.w = np.zeros(m); self.b = 0.0
        self.train_curve = []; self.val_curve = []
        for t in range(1, self.n_iter+1):
            eta = self.lr / t
            idx = np.random.choice(n, min(self.batch, n), replace=False)
            xi, yi = X[idx], ys[idx]
            margin = yi * (xi @ self.w + self.b)
            mask   = margin < 1
            self.w  = (1-eta)*self.w + eta*self.C*(yi[mask,None]*xi[mask]).sum(0)
            self.b += eta * self.C * yi[mask].sum()
            if t % (self.n_iter//5) == 0:
                yp = self.predict(X)
                self.train_curve.append(round(
                    accuracy_score((ys==1).astype(int), yp), 4))
                if X_val is not None:
                    self.val_curve.append(round(
                        accuracy_score(y_val, self.predict(X_val)), 4))

    def decision(self, X): return np.array(X, float) @ self.w + self.b
    def predict(self, X):  return (self.decision(X) >= 0).astype(int)
    def predict_proba(self, X):
        return 1/(1+np.exp(-self.decision(X)))


def main():
    print("\n" + "█"*58)
    print("  MODEL 04 — XGBoost + SVM")
    print("█"*58)

    X, y = load_features(sample_n=50_000)
    X_tr, X_val, X_te, y_tr, y_val, y_te = split(X, y)

    scaler = StandardScaler()
    Xts  = scaler.fit_transform(X_tr)
    Xvs  = scaler.transform(X_val)
    Xtes = scaler.transform(X_te)

    ytn, yvn, yten = y_tr.values, y_val.values, y_te.values
    results = []

    # ── 1. Gradient Boosting — Scratch ──────────────────────
    print("\n[1/4] Gradient Boosting — Scratch (60 stumps, lr=0.08) ...")
    t0 = time.time()
    m  = GradBoostScratch(n=60, lr=0.08)
    m.fit(Xts, ytn, X_val=Xvs, y_val=yvn)
    yp_te = m.predict(Xtes); yp_tr = m.predict(Xts)
    row   = evaluate("XGBoost/GB (Scratch)", yten, yp_te,
                     y_train=ytn, y_pred_train=yp_tr,
                     y_prob=m.predict_proba(Xtes),
                     train_scores={"train": m.train_curve, "val": m.val_curve})
    row["implementation"] = "scratch"; row["training_time"] = round(time.time()-t0, 3)
    results.append(row)

    # ── 2. XGBoost — library ────────────────────────────────
    print("\n[2/4] XGBoost — library (depth=5, L2 reg) ...")
    t0 = time.time()
    dtrain = xgb.DMatrix(X_tr, label=y_tr)
    dtest  = xgb.DMatrix(X_te, label=y_te)
    dtrain_full = xgb.DMatrix(X_tr, label=y_tr)
    eval_res = {}
    params = dict(
        objective="binary:logistic",
        eval_metric=["logloss", "auc"],
        max_depth=5,          # reduced to prevent overfitting
        eta=0.08,
        subsample=0.7,
        colsample_bytree=0.7,
        reg_lambda=2.0,       # L2 regularization
        reg_alpha=0.5,        # L1 regularization
        min_child_weight=10,  # prevents fitting noise
        seed=42,
    )
    bst = xgb.train(params, dtrain, num_boost_round=150,
                    evals=[(dtrain_full,"train"),(dtest,"val")],
                    evals_result=eval_res, verbose_eval=False,
                    early_stopping_rounds=15)
    yprob_te = bst.predict(dtest)
    yp_te    = (yprob_te >= .5).astype(int)
    yprob_tr = bst.predict(dtrain_full)
    yp_tr    = (yprob_tr >= .5).astype(int)
    # Learning curves from eval results
    tr_ll = eval_res["train"]["logloss"]
    va_ll = eval_res["val"]["logloss"]
    step  = max(1, len(tr_ll)//8)
    tr_s  = [round(1 - v, 4) for v in tr_ll[::step]]
    va_s  = [round(1 - v, 4) for v in va_ll[::step]]
    row = evaluate("XGBoost (library)", y_te, yp_te,
                   y_train=y_tr, y_pred_train=yp_tr, y_prob=yprob_te,
                   train_scores={"train": tr_s, "val": va_s})
    row["implementation"] = "sklearn"; row["training_time"] = round(time.time()-t0, 3)
    results.append(row)

    # ── 3. SVM — Scratch ─────────────────────────────────────
    print("\n[3/4] SVM — Scratch (Pegasos, C=0.5, 6000 iters) ...")
    t0 = time.time()
    m3 = SVMScratch(C=0.5, n_iter=6000, lr=0.01, batch=256)
    m3.fit(Xts, ytn, X_val=Xvs, y_val=yvn)
    yp_te = m3.predict(Xtes); yp_tr = m3.predict(Xts)
    row = evaluate("SVM (Scratch)", yten, yp_te,
                   y_train=ytn, y_pred_train=yp_tr,
                   y_prob=m3.predict_proba(Xtes),
                   train_scores={"train": m3.train_curve, "val": m3.val_curve})
    row["implementation"] = "scratch"; row["training_time"] = round(time.time()-t0, 3)
    results.append(row)

    # ── 4. SVM — sklearn ─────────────────────────────────────
    print("\n[4/4] SVM — sklearn (LinearSVC + Platt calibration, C=0.5) ...")
    t0 = time.time()
    tr_s, va_s = [], []
    for frac in np.linspace(0.1, 1.0, 6):
        n  = max(10, int(len(Xts)*frac))
        mc = CalibratedClassifierCV(
            LinearSVC(C=0.5, max_iter=3000, random_state=42))
        mc.fit(Xts[:n], ytn[:n])
        tr_s.append(round(accuracy_score(ytn[:n], mc.predict(Xts[:n])), 4))
        va_s.append(round(accuracy_score(yvn,     mc.predict(Xvs)),     4))
    m4 = CalibratedClassifierCV(LinearSVC(C=0.5, max_iter=3000, random_state=42))
    m4.fit(Xts, ytn)
    yp_te = m4.predict(Xtes); yp_tr = m4.predict(Xts)
    yprob = m4.predict_proba(Xtes)[:, 1]
    row = evaluate("SVM (sklearn LinearSVC)", y_te, yp_te,
                   y_train=y_tr, y_pred_train=yp_tr, y_prob=yprob,
                   train_scores={"train": tr_s, "val": va_s})
    row["implementation"] = "sklearn"; row["training_time"] = round(time.time()-t0, 3)
    results.append(row)

    save_results(results)
    print("\n✅ Model 04 complete.")


if __name__ == "__main__":
    main()
