"""
model_02_tree_forest.py
========================
DECISION TREE  +  RANDOM FOREST
  • max_depth limited to prevent overfitting
  • Reports train vs test accuracy gap
  • All 6 metrics + learning curves
"""
import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from collections import Counter
from utils import load_features, split, evaluate, save_results
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


# ══════════════════════════════════════════════════════════════
#  SCRATCH: Decision Tree (CART — Gini)
# ══════════════════════════════════════════════════════════════
def _gini(y):
    if len(y) == 0: return 0
    c = np.bincount(y, minlength=2)
    p = c / len(y)
    return 1 - np.sum(p ** 2)


def _best_split(X, y, n_feat):
    best_gain, feat, thr = -1.0, None, None
    pg = _gini(y)
    n  = len(y)
    idxs = np.random.choice(X.shape[1], min(n_feat, X.shape[1]), replace=False)
    for f in idxs:
        vals = np.percentile(X[:, f], np.linspace(10, 90, 10))
        for t in vals:
            lm = X[:, f] <= t
            nl, nr = lm.sum(), (~lm).sum()
            if nl < 2 or nr < 2:
                continue
            gain = pg - (nl / n * _gini(y[lm]) + nr / n * _gini(y[~lm]))
            if gain > best_gain:
                best_gain, feat, thr = gain, f, t
    return feat, thr


class _Node:
    __slots__ = ("feat", "thr", "left", "right", "val")
    def __init__(self, **kw): [setattr(self, k, v) for k, v in kw.items()]
    @property
    def is_leaf(self): return self.val is not None


class DecisionTreeScratch:
    """
    CART classifier using Gini impurity.
    max_depth and min_samples_leaf control overfitting.
    """
    def __init__(self, max_depth=8, min_samples_leaf=15, n_feat=None):
        self.max_depth        = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.n_feat           = n_feat

    def fit(self, X, y):
        X, y = np.array(X), np.array(y)
        nf   = self.n_feat or X.shape[1]
        self.root = self._grow(X, y, 0, nf)

    def _grow(self, X, y, depth, nf):
        if (depth >= self.max_depth
                or len(y) < self.min_samples_leaf * 2
                or len(np.unique(y)) == 1):
            return _Node(feat=None, thr=None, left=None, right=None,
                         val=Counter(y).most_common(1)[0][0])
        f, t = _best_split(X, y, nf)
        if f is None:
            return _Node(feat=None, thr=None, left=None, right=None,
                         val=Counter(y).most_common(1)[0][0])
        m = X[:, f] <= t
        return _Node(feat=f, thr=t,
                     left=self._grow(X[m],  y[m],  depth+1, nf),
                     right=self._grow(X[~m], y[~m], depth+1, nf),
                     val=None)

    def _pred1(self, x, node):
        if node.is_leaf: return node.val
        return self._pred1(x, node.left if x[node.feat] <= node.thr else node.right)

    def predict(self, X):
        return np.array([self._pred1(x, self.root) for x in np.array(X)])

    def predict_proba(self, X):
        return self.predict(X).astype(float)


# ══════════════════════════════════════════════════════════════
#  SCRATCH: Random Forest (Bagging + feature subsampling)
# ══════════════════════════════════════════════════════════════
class RandomForestScratch:
    """
    Ensemble of CART trees on bootstrap samples.
    max_depth=7 limits individual tree complexity.
    """
    def __init__(self, n_trees=30, max_depth=7, min_samples_leaf=20):
        self.n_trees          = n_trees
        self.max_depth        = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.trees            = []
        self.train_curve      = []
        self.val_curve        = []

    def fit(self, X, y, X_val=None, y_val=None):
        X, y = np.array(X), np.array(y)
        nf   = max(1, int(np.sqrt(X.shape[1])))
        self.trees = []
        for i in range(self.n_trees):
            idx  = np.random.choice(len(y), len(y), replace=True)
            tree = DecisionTreeScratch(
                max_depth=self.max_depth,
                min_samples_leaf=self.min_samples_leaf,
                n_feat=nf,
            )
            tree.fit(X[idx], y[idx])
            self.trees.append(tree)
            if (i + 1) % 10 == 0:
                print(f"    Tree {i+1}/{self.n_trees} grown ...")
                tr_acc = accuracy_score(y, self.predict(X))
                self.train_curve.append(round(tr_acc, 4))
                if X_val is not None:
                    self.val_curve.append(round(
                        accuracy_score(y_val, self.predict(X_val)), 4
                    ))

    def predict(self, X):
        X     = np.array(X)
        votes = np.array([t.predict(X) for t in self.trees])
        return np.apply_along_axis(
            lambda c: Counter(c).most_common(1)[0][0], 0, votes
        )

    def predict_proba(self, X):
        votes = np.array([t.predict(np.array(X)) for t in self.trees])
        return votes.mean(axis=0)


# ── sklearn learning curve helper ────────────────────────────
def lc_sklearn(cls, X_tr, y_tr, X_val, y_val, n_pts=6, **kw):
    tr_s, va_s = [], []
    for frac in np.linspace(0.1, 1.0, n_pts):
        n = max(10, int(len(X_tr) * frac))
        m = cls(**kw); m.fit(X_tr[:n], y_tr[:n])
        tr_s.append(round(accuracy_score(y_tr[:n], m.predict(X_tr[:n])), 4))
        va_s.append(round(accuracy_score(y_val,    m.predict(X_val)),    4))
    return tr_s, va_s


def main():
    print("\n" + "█"*58)
    print("  MODEL 02 — Decision Tree + Random Forest")
    print("█"*58)

    X, y = load_features()
    X_tr, X_val, X_te, y_tr, y_val, y_te = split(X, y)
    Xtn, Xvn, Xten = X_tr.values, X_val.values, X_te.values
    ytn, yvn, yten = y_tr.values, y_val.values, y_te.values
    results = []

    # ── 1. Decision Tree — Scratch ───────────────────────────
    print("\n[1/4] Decision Tree — Scratch (max_depth=8, min_leaf=15) ...")
    t0 = time.time()
    m  = DecisionTreeScratch(max_depth=8, min_samples_leaf=15)
    m.fit(Xtn, ytn)
    y_pred_te = m.predict(Xten)
    y_pred_tr = m.predict(Xtn)
    row = evaluate("Decision Tree (Scratch)", yten, y_pred_te,
                   y_train=ytn, y_pred_train=y_pred_tr,
                   y_prob=m.predict_proba(Xten),
                   train_scores={"train":[round(accuracy_score(ytn, y_pred_tr), 4)],
                                 "val":  [round(accuracy_score(yvn, m.predict(Xvn)), 4)]})
    row["implementation"] = "scratch"; row["training_time"] = round(time.time()-t0, 3)
    results.append(row)

    # ── 2. Decision Tree — sklearn ───────────────────────────
    print("\n[2/4] Decision Tree — sklearn (max_depth=8, min_leaf=15) ...")
    t0 = time.time()
    tr_s, va_s = lc_sklearn(DecisionTreeClassifier, Xtn, ytn, Xvn, yvn,
                              max_depth=8, min_samples_leaf=15, random_state=42)
    m2 = DecisionTreeClassifier(max_depth=8, min_samples_leaf=15, random_state=42)
    m2.fit(X_tr, y_tr)
    y_pred_te = m2.predict(X_te); y_pred_tr = m2.predict(X_tr)
    y_prob_te = m2.predict_proba(X_te)[:, 1]
    row = evaluate("Decision Tree (sklearn)", y_te, y_pred_te,
                   y_train=y_tr, y_pred_train=y_pred_tr, y_prob=y_prob_te,
                   train_scores={"train": tr_s, "val": va_s})
    row["implementation"] = "sklearn"; row["training_time"] = round(time.time()-t0, 3)
    results.append(row)

    # ── 3. Random Forest — Scratch ───────────────────────────
    print("\n[3/4] Random Forest — Scratch (30 trees, max_depth=7) ...")
    t0 = time.time()
    m3 = RandomForestScratch(n_trees=30, max_depth=7, min_samples_leaf=20)
    m3.fit(Xtn, ytn, X_val=Xvn, y_val=yvn)
    y_pred_te = m3.predict(Xten); y_pred_tr = m3.predict(Xtn)
    row = evaluate("Random Forest (Scratch)", yten, y_pred_te,
                   y_train=ytn, y_pred_train=y_pred_tr,
                   y_prob=m3.predict_proba(Xten),
                   train_scores={"train": m3.train_curve, "val": m3.val_curve})
    row["implementation"] = "scratch"; row["training_time"] = round(time.time()-t0, 3)
    results.append(row)

    # ── 4. Random Forest — sklearn ───────────────────────────
    print("\n[4/4] Random Forest — sklearn (100 trees, max_depth=10, min_leaf=10) ...")
    t0 = time.time()
    tr_s, va_s = lc_sklearn(RandomForestClassifier, Xtn, ytn, Xvn, yvn, n_pts=5,
                              n_estimators=50, max_depth=10, min_samples_leaf=10,
                              random_state=42, n_jobs=-1)
    m4 = RandomForestClassifier(n_estimators=100, max_depth=10, min_samples_leaf=10,
                                 random_state=42, n_jobs=-1)
    m4.fit(X_tr, y_tr)
    y_pred_te = m4.predict(X_te); y_pred_tr = m4.predict(X_tr)
    y_prob_te = m4.predict_proba(X_te)[:, 1]
    row = evaluate("Random Forest (sklearn)", y_te, y_pred_te,
                   y_train=y_tr, y_pred_train=y_pred_tr, y_prob=y_prob_te,
                   train_scores={"train": tr_s, "val": va_s})
    row["implementation"] = "sklearn"; row["training_time"] = round(time.time()-t0, 3)
    results.append(row)

    save_results(results)
    print("\n✅ Model 02 complete.")


if __name__ == "__main__":
    main()
