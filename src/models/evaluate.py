"""
Evaluasi model: Precision, Recall, F1-score, ROC-AUC.
Jalankan setelah train.py, pakai data test terpisah (holdout).
"""
from pathlib import Path
import joblib
import pandas as pd
from sklearn.metrics import (
    precision_score, recall_score, f1_score, roc_auc_score, classification_report
)
from src.models.train import FEATURE_COLUMNS, TARGET_COLUMN, DATASET_PATH

MODEL_DIR = Path("models/saved")


def evaluate_model(model_path: Path, X_test, y_test, name: str):
    model = joblib.load(model_path)
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    print(f"\n=== {name} ===")
    print(f"Precision: {precision_score(y_test, y_pred):.3f}")
    print(f"Recall   : {recall_score(y_test, y_pred):.3f}")
    print(f"F1-score : {f1_score(y_test, y_pred):.3f}")
    print(f"ROC-AUC  : {roc_auc_score(y_test, y_proba):.3f}")
    print(classification_report(y_test, y_pred))


def main():
    df = pd.read_csv(DATASET_PATH)
    X_test = df[FEATURE_COLUMNS]
    y_test = df[TARGET_COLUMN]

    evaluate_model(MODEL_DIR / "random_forest.pkl", X_test, y_test, "Random Forest")
    evaluate_model(MODEL_DIR / "xgboost.pkl", X_test, y_test, "XGBoost")


if __name__ == "__main__":
    main()
