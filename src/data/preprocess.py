"""
Membersihkan hasil OCR mentah (comments/profiles) menjadi struktur tabel
(CSV) yang siap dipakai untuk feature engineering.

Input : data/raw/comments/*.txt, data/raw/profiles/*.txt
Output: data/interim/comments_structured.csv, data/interim/profiles_structured.csv

CATATAN: kolom di bawah ini contoh -- sesuaikan dengan variabel penelitian
kamu (lihat matriks penelitian: fitur perilaku, temporal, linguistik).
"""
from pathlib import Path
import pandas as pd
import re

COMMENTS_DIR = Path("data/raw/comments")
PROFILES_DIR = Path("data/raw/profiles")
OUTPUT_DIR = Path("data/interim")


def clean_text(raw: str) -> str:
    text = raw.strip()
    text = re.sub(r"\s+", " ", text)
    return text


def build_comments_table() -> pd.DataFrame:
    rows = []
    for txt_file in COMMENTS_DIR.glob("*.txt"):
        raw = txt_file.read_text(encoding="utf-8")
        rows.append({
            "source_file": txt_file.stem,
            "username": None,          # isi manual/parsing
            "comment_text": clean_text(raw),
            "comment_time": None,      # isi manual dari screenshot
            "post_time": None,         # isi manual dari screenshot
            "label": None,             # diisi saat proses labeling (buzzer/bukan)
        })
    return pd.DataFrame(rows)


def build_profiles_table() -> pd.DataFrame:
    rows = []
    for txt_file in PROFILES_DIR.glob("*.txt"):
        raw = txt_file.read_text(encoding="utf-8")
        rows.append({
            "source_file": txt_file.stem,
            "username": None,
            "following_count": None,
            "follower_count": None,
            "post_count": None,
            "bio_filled": None,        # True/False kelengkapan profil
        })
    return pd.DataFrame(rows)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    comments_df = build_comments_table()
    profiles_df = build_profiles_table()

    comments_df.to_csv(OUTPUT_DIR / "comments_structured.csv", index=False)
    profiles_df.to_csv(OUTPUT_DIR / "profiles_structured.csv", index=False)

    print(f"comments: {len(comments_df)} baris -> {OUTPUT_DIR/'comments_structured.csv'}")
    print(f"profiles: {len(profiles_df)} baris -> {OUTPUT_DIR/'profiles_structured.csv'}")


if __name__ == "__main__":
    main()
