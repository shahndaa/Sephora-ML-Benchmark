"""
model_03_knn_naivebayes.py
===========================
KNN  +  GAUSSIAN NAIVE BAYES
  • KNN uses StandardScaler (required for distance-based methods)
  • Naive Bayes is fast but assumes independence
  • All 6 metrics + overfit gap
"""
import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from collections import Counter
from utils import load_features, split, evaluate, save_results
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score


# ══════════════════════════════════════════════════════════════
#  SCRATCH: KNN (Euclidean distance)
# ══════════════════════════════════════════════════════════════
class KNNScratch:
    """
    K-Nearest Neighbours: no training, O(n·d) inference.
    Uses Euclidean (L2) distance, majority vote.
    """
    def __init__(self, k=7):
        self.k = k

    def fit(self, X, y):
        self.Xtr = np.array(X)
        self.ytr = np.array(y)

    def _distances(self, x):
        return np.sqrt(((self.Xtr - x) ** 2).sum(axis=1))

    def predict_one(self, x):
        d   = self._distances(x)
        idx = np.argpartition(d, self.k)[:self.k]
        return Counter(self.ytr[idx]).most_common(1)[0][0]

    def predict(self, X):
        return np.array([self.predict_one(x) for x in np.array(X)])

    def predict_proba(self, X):
        out = []
        for x in np.array(X):
            d   = self._distances(x)
            idx = np.argpartition(d, self.k)[:self.k]
            out.append(self.ytr[idx].mean())
        return np.array(out)


# ══════════════════════════════════════════════════════════════
#  SCRATCH: Gaussian Naive Bayes
# ══════════════════════════════════════════════════════════════
class GaussianNBScratch:
    """
    Bayes theorem with Gaussian likelihood and feature independence assumption.
    log-posterior = log P(c) + Σ log N(xj; μcj, σ²cj)
    """
    def fit(self, X, y):
        X, y = np.array(X), np.array(y)
        self.classes = np.unique(y)
        self.prior, self.mu, self.var = {}, {}, {}
        for c in self.classes:
            Xc = X[y == c]
            self.prior[c] = len(Xc) / len(y)
            self.mu[c]    = Xc.mean(axis=0)
            self.var[c]   = Xc.var(axis=0) + 1e-9

    def _log_likelihood(self, x, c):
        return -0.5 * np.sum(
            np.log(2 * np.pi * self.var[c]) + (x - self.mu[c]) ** 2 / self.var[c]
        )

    def _log_posterior(self, x):
        return {c: np.log(self.prior[c]) + self._log_likelihood(x, c)
                for c in self.classes}

    def predict(self, X):
        return np.array([
            max(self._log_posterior(x), key=self._log_posterior(x).get)
            for x in np.array(X)
        ])

    def predict_proba(self, X):
        out = []
        for x in np.array(X):
            lp   = self._log_posterior(x)
            vals = np.array([lp[c] for c in self.classes])
            vals -= vals.max()
            exp_v = np.exp(vals)
            p     = exp_v / exp_v.sum()
            out.append(p[list(self.classes).index(1)])
        return np.array(out)


# ── sklearn learning curve helpers ───────────────────────────
def lc_knn(Xts, ytn, Xvn, yvn, k=7, n_pts=5):
    tr_s, va_s = [], []
    for frac in np.linspace(0.1, 1.0, n_pts):
        n = max(k+1, int(len(Xts)*frac))
        m = KNeighborsClassifier(n_neighbors=k, n_jobs=-1)
        m.fit(Xts[:n], ytn[:n])
        tr_s.append(round(accuracy_score(ytn[:n], m.predict(Xts[:n])), 4))
        va_s.append(round(accuracy_score(yvn,     m.predict(Xvn)),     4))
    return tr_s, va_s

def lc_nb(Xts, ytn, Xvn, yvn, n_pts=6):
    tr_s, va_s = [], []
    for frac in np.linspace(0.1, 1.0, n_pts):
        n = max(10, int(len(Xts)*frac))
        m = GaussianNB(); m.fit(Xts[:n], ytn[:n])
        tr_s.append(round(accuracy_score(ytn[:n], m.predict(Xts[:n])), 4))
        va_s.append(round(accuracy_score(yvn,     m.predict(Xvn)),     4))
    return tr_s, va_s


def main():
    print("\n" + "█"*58)
    print("  MODEL 03 — KNN + Naive Bayes")
    print("█"*58)

    # Use smaller sample for KNN (O(n·d) per query)
    X, y = load_features(sample_n=25_000)
    X_tr, X_val, X_te, y_tr, y_val, y_te = split(X, y)

    scaler = StandardScaler()
    Xts  = scaler.fit_transform(X_tr)
    Xvs  = scaler.transform(X_val)
    Xtes = scaler.transform(X_te)

    ytn, yvn, yten = y_tr.values, y_val.values, y_te.values
    results = []

    # ── 1. KNN — Scratch (k=7) ──────────────────────────────
    print("\n[1/4] KNN — Scratch (k=7) ... evaluating on 3,000 test points")
    t0    = time.time()
    m     = KNNScratch(k=7)
    m.fit(Xts, ytn)
    # Limit test to 3,000 for speed (scratch is O(n·d) per query)
    n_t   = min(3000, len(Xtes))
    yp_te = m.predict(Xtes[:n_t])
    yp_tr = m.predict(Xts[:1000])   # small train sample for gap
    row   = evaluate("KNN (Scratch, k=7)", yten[:n_t], yp_te,
                     y_train=ytn[:1000], y_pred_train=yp_tr,
                     y_prob=m.predict_proba(Xtes[:n_t]))
    row["implementation"] = "scratch"
    row["training_time"]  = round(time.time()-t0, 3)
    results.append(row)

    # ── 2. KNN — sklearn (k=7) ──────────────────────────────
    print("\n[2/4] KNN — sklearn (k=7) ...")
    t0 = time.time()
    tr_s, va_s = lc_knn(Xts, ytn, Xvs, yvn, k=7)
    m2 = KNeighborsClassifier(n_neighbors=7, n_jobs=-1)
    m2.fit(Xts, ytn)
    yp_te = m2.predict(Xtes); yp_tr = m2.predict(Xts)
    yprob = m2.predict_proba(Xtes)[:, 1]
    row = evaluate("KNN (sklearn, k=7)", y_te, yp_te,
                   y_train=y_tr, y_pred_train=yp_tr, y_prob=yprob,
                   train_scores={"train": tr_s, "val": va_s})
    row["implementation"] = "sklearn"
    row["training_time"]  = round(time.time()-t0, 3)
    results.append(row)

    # ── 3. Naive Bayes — Scratch ─────────────────────────────
    print("\n[3/4] Gaussian Naive Bayes — Scratch ...")
    t0 = time.time()
    m3 = GaussianNBScratch()
    m3.fit(Xts, ytn)
    yp_te = m3.predict(Xtes); yp_tr = m3.predict(Xts)
    row = evaluate("Naive Bayes (Scratch)", y_te, yp_te,
                   y_train=y_tr, y_pred_train=yp_tr,
                   y_prob=m3.predict_proba(Xtes))
    row["implementation"] = "scratch"
    row["training_time"]  = round(time.time()-t0, 3)
    results.append(row)

    # ── 4. Naive Bayes — sklearn ─────────────────────────────
    print("\n[4/4] Gaussian Naive Bayes — sklearn ...")
    t0 = time.time()
    tr_s, va_s = lc_nb(Xts, ytn, Xvs, yvn)
    m4 = GaussianNB()
    m4.fit(Xts, ytn)
    yp_te = m4.predict(Xtes); yp_tr = m4.predict(Xts)
    yprob = m4.predict_proba(Xtes)[:, 1]
    row = evaluate("Naive Bayes (sklearn)", y_te, yp_te,
                   y_train=y_tr, y_pred_train=yp_tr, y_prob=yprob,
                   train_scores={"train": tr_s, "val": va_s})
    row["implementation"] = "sklearn"
    row["training_time"]  = round(time.time()-t0, 3)
    results.append(row)

    save_results(results)
    print("\n✅ Model 03 complete.")


if __name__ == "__main__":
    main()
