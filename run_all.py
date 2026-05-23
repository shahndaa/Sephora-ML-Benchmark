"""
run_all.py  —  Master pipeline script.

Usage:
    python run_all.py data/

Where data/ contains:
    product_info.csv
    reviews_0-250.csv
    reviews_250-500.csv
    reviews_500-750.csv
    reviews_750-1250.csv
    reviews_1250-end.csv
"""
import sys, os, subprocess

BASE    = os.path.dirname(os.path.abspath(__file__))
SCRIPTS = os.path.join(BASE, "scripts")
DATA    = sys.argv[1] if len(sys.argv)>1 else os.path.join(BASE,"data")

STEPS = [
    ("Preprocessing",              "preprocessing.py",           [DATA]),
    ("Model 01 — Linear+Logistic", "model_01_linear_logistic.py",[]),
    ("Model 02 — Tree+Forest",     "model_02_tree_forest.py",    []),
    ("Model 03 — KNN+NaiveBayes",  "model_03_knn_naivebayes.py", []),
    ("Model 04 — XGBoost+SVM",     "model_04_xgboost_svm.py",   []),
]

def run(label, script, args):
    print(f"\n{'█'*60}\n  STEP: {label}\n{'█'*60}")
    res = subprocess.run([sys.executable, os.path.join(SCRIPTS,script)]+args)
    if res.returncode!=0:
        print(f"\n❌  '{label}' failed (code {res.returncode})"); sys.exit(1)

if __name__=="__main__":
    print("\n"+"="*60+"\n  SEPHORA ML — FULL PIPELINE\n"+"="*60)
    print(f"Data : {DATA}\nDB   : {os.path.join(BASE,'sephora.db')}")
    for label, script, args in STEPS:
        run(label, script, args)
    print("\n"+"="*60+"\n  ALL DONE — launching web app …\n  Open http://localhost:5000\n"+"="*60)
    subprocess.run([sys.executable, os.path.join(BASE,"app.py")])
