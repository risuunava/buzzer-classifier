"""
Gabungkan comments_structured.csv + profiles_structured.csv, terapkan
feature engineering (behavioral, temporal, linguistic), lalu simpan
jadi satu dataset siap training di data/processed/dataset.csv.

Jalankan setelah:
1. src/data/preprocess.py sudah dijalankan (hasil comments_structured.csv
   & profiles_structured.csv ada di data/interim/)
2. Kolom-kolom di kedua CSV itu sudah diisi manual (username, following_count,
   follower_count, bio_filled, comment_time, post_time, dst)

Catatan: 'label' WAJIB sudah diisi (0/1) di comments_structured.csv sebelum
menjalankan ini, karena tanpa label, dataset ini hanya bisa dipakai untuk
prediksi, bukan training.
"""
from pathlib import Path
import pandas as pd
import sys

sys.path.append(str(Path(__file__).resolve().parent.parent))

from features.behavioral import add_behavioral_features
from features.temporal import add_temporal_features
from features.linguistic import add_linguistic_features

COMMENTS_PATH = Path("data/interim/comments_structured.csv")
PROFILES_PATH = Path("data/interim/profiles_structured.csv")
OUTPUT_PATH = Path("data/processed/dataset.csv")


def main():
    comments_df = pd.read_csv(COMMENTS_PATH)
    profiles_df = pd.read_csv(PROFILES_PATH)

    missing_username_comments = comments_df["username"].isna().sum()
    missing_username_profiles = profiles_df["username"].isna().sum()
    if missing_username_comments or missing_username_profiles:
        print(
            f"[WARNING] Ada {missing_username_comments} baris comments dan "
            f"{missing_username_profiles} baris profiles tanpa username. "
            "Baris ini tidak akan bisa digabung -- isi dulu kolom username."
        )

    # Terapkan feature engineering di masing-masing tabel dulu
    comments_df = add_temporal_features(comments_df)
    comments_df = add_linguistic_features(comments_df)
    profiles_df = add_behavioral_features(profiles_df)

    # Gabungkan berdasarkan username (inner join: hanya akun yang punya
    # data komentar DAN profil yang ikut masuk dataset)
    merged = comments_df.merge(
        profiles_df, on="username", how="inner", suffixes=("_comment", "_profile")
    )

    dropped = len(comments_df) - len(merged)
    if dropped > 0:
        print(
            f"[INFO] {dropped} baris komentar tidak punya pasangan data profil "
            "(username tidak cocok) dan tidak ikut masuk dataset."
        )

    if merged["label"].isna().any():
        n_missing_label = merged["label"].isna().sum()
        print(
            f"[WARNING] {n_missing_label} baris belum punya label. "
            "Baris ini akan tetap disimpan, tapi HARUS dilabeli manual "
            "sebelum dipakai untuk training model."
        )

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(OUTPUT_PATH, index=False)
    print(f"Dataset final: {len(merged)} baris -> {OUTPUT_PATH}")


if __name__ == "__main__":
    main()