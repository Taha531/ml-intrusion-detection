import os
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, precision_score, recall_score
from scripts.preprocess import load_and_preprocess

SEED = 42
OUT_DIR = os.path.join(os.getcwd())
MODELS_DIR = os.path.join(OUT_DIR, 'models')
RESULTS_DIR = os.path.join(OUT_DIR, 'results')

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)


def compute_tpr_fpr(cm):
    # cm for binary: [[tn, fp], [fn, tp]]
    tn, fp, fn, tp = cm.ravel()
    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    return tpr, fpr


if __name__ == '__main__':
    X, y, feature_names = load_and_preprocess()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=SEED, stratify=y)

    models = {}
    metrics = []

    # 1) Gaussian Naive Bayes
    print('Training GaussianNB...')
    gnb = GaussianNB()
    gnb.fit(X_train, y_train)
    y_pred = gnb.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    tpr, fpr = compute_tpr_fpr(cm)
    metrics.append({'model': 'GaussianNB', 'precision': precision_score(y_test, y_pred), 'recall': recall_score(y_test, y_pred), 'tpr': tpr, 'fpr': fpr})
    joblib.dump(gnb, os.path.join(MODELS_DIR, 'gnb.joblib'))

    # 2) Decision Tree (max_depth from paper ~9)
    print('Training DecisionTree (max_depth=9)...')
    dt = DecisionTreeClassifier(max_depth=9, random_state=SEED)
    dt.fit(X_train, y_train)
    y_pred = dt.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    tpr, fpr = compute_tpr_fpr(cm)
    metrics.append({'model': 'DecisionTree', 'precision': precision_score(y_test, y_pred), 'recall': recall_score(y_test, y_pred), 'tpr': tpr, 'fpr': fpr})
    joblib.dump(dt, os.path.join(MODELS_DIR, 'decision_tree.joblib'))

    # 3) Random Forest
    print('Training RandomForest (n_estimators=16, max_depth=9)...')
    rf = RandomForestClassifier(n_estimators=16, max_depth=9, random_state=SEED, n_jobs=-1)
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    tpr, fpr = compute_tpr_fpr(cm)
    metrics.append({'model': 'RandomForest', 'precision': precision_score(y_test, y_pred), 'recall': recall_score(y_test, y_pred), 'tpr': tpr, 'fpr': fpr})
    joblib.dump(rf, os.path.join(MODELS_DIR, 'random_forest.joblib'))

    # Save metrics
    dfm = pd.DataFrame(metrics)
    dfm.to_csv(os.path.join(RESULTS_DIR, 'metrics.csv'), index=False)
    print('\nTraining complete. Models saved to `models/` and metrics to `results/metrics.csv`')