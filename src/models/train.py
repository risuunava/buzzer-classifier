"""
Training Random Forest & XGBoost dengan k-fold cross validation.

Input : data/processed/dataset.csv (hasil gabungan fitur + kolom 'label')
Output: models/saved/random_forest.pkl, models/saved/xgboost.pkl
"""
from pathlib import Path
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from xgboost import XGBClassifier

DATASET_PATH = Path("data/processed/dataset.csv")
MODEL_DIR = Path("models/saved")

FEATURE_COLUMNS = [
    # isi sesuai kolom fitur yang sudah dihasilkan di src/features/*
    "following_follower_ratio",
    "profile_completeness",
    "comment_delay_minutes",
    "comment_hour",
    "comment_length",
    "emoji_hashtag_ratio",
    "avg_similarity_to_others",
]
TARGET_COLUMN = "label"


def load_dataset():
    df = pd.read_csv(DATASET_PATH)
    X = df[FEATURE_COLUMNS]
    y = df[TARGET_COLUMN]
    return X, y


def train_and_evaluate(model, X, y, name: str, n_splits: int | None = 5):
    if n_splits is None:
        # Tidak cukup data untuk CV — langsung fit saja (hanya untuk tes pipeline)
        print(f"[{name}] CV dilewati. Langsung fit tanpa evaluasi fold.")
        model.fit(X, y)
        return model

    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    scores = cross_val_score(model, X, y, cv=cv, scoring="f1")
    print(f"[{name}] F1 tiap fold: {scores}")
    print(f"[{name}] F1 rata-rata: {scores.mean():.3f} (+/- {scores.std():.3f})")

    model.fit(X, y)
    return model


def main():
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    X, y = load_dataset()

    # ── Guard rail: cek jumlah sampel per kelas ──────────────────────────
    min_class_count = y.value_counts().min()

    if min_class_count < 2:
        # Dengan 1 sampel per kelas, CV tidak bisa dilakukan sama sekali.
        # Langsung fit tanpa evaluasi fold.
        print(
            f"[PERINGATAN] Kelas minoritas cuma {min_class_count} sampel. "
            "Hasil training TIDAK VALID untuk skripsi, hanya untuk tes pipeline. "
            "Kumpulkan lebih banyak data sebelum training final."
        )
        print("[INFO] Cross-validation dilewati (butuh minimal 2 sampel per kelas).")
        n_splits = None
    elif min_class_count < 5:
        n_splits = min_class_count  # clamp ke jumlah sampel minimal (>=2)
        print(
            f"[PERINGATAN] Kelas minoritas cuma {min_class_count} sampel. "
            "Hasil training TIDAK VALID untuk skripsi, hanya untuk tes pipeline. "
            "Kumpulkan lebih banyak data sebelum training final."
        )
        print(
            f"[INFO] n_splits otomatis diturunkan dari 5 menjadi {n_splits} "
            "agar tidak crash."
        )
    else:
        n_splits = 5
        print(f"[INFO] {min_class_count} sampel kelas minoritas — n_splits=5 dipakai.")

    # ── Training ─────────────────────────────────────────────────────────
    rf = RandomForestClassifier(n_estimators=200, random_state=42)
    rf = train_and_evaluate(rf, X, y, "Random Forest", n_splits=n_splits)
    joblib.dump(rf, MODEL_DIR / "random_forest.pkl")

    xgb = XGBClassifier(eval_metric="logloss", random_state=42)
    xgb = train_and_evaluate(xgb, X, y, "XGBoost", n_splits=n_splits)
    joblib.dump(xgb, MODEL_DIR / "xgboost.pkl")

    print("Model tersimpan di", MODEL_DIR)


if __name__ == "__main__":
    main()
