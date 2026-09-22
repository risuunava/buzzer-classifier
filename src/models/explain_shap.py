"""
Interpretasi model menggunakan SHAP -- untuk menjawab tujuan penelitian
"menganalisis kontribusi tiap fitur terhadap hasil klasifikasi model".
"""
from pathlib import Path
import joblib
import pandas as pd
import shap
import matplotlib.pyplot as plt
from models.train import FEATURE_COLUMNS, DATASET_PATH

MODEL_DIR = Path("models/saved")
FIGURES_DIR = Path("reports/figures")


def explain(model_path: Path, X, name: str):
    model = joblib.load(model_path)
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)

    shap.summary_plot(shap_values, X, show=False)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    out_path = FIGURES_DIR / f"shap_summary_{name}.png"
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    print(f"[{name}] SHAP summary plot disimpan di {out_path}")


def main():
    df = pd.read_csv(DATASET_PATH)
    X = df[FEATURE_COLUMNS]

    explain(MODEL_DIR / "random_forest.pkl", X, "random_forest")
    explain(MODEL_DIR / "xgboost.pkl", X, "xgboost")


if __name__ == "__main__":
    main()
