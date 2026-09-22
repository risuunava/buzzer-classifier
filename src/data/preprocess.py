"""
Membersihkan hasil OCR mentah (comments/profiles) menjadi struktur tabel
(CSV) yang siap dipakai untuk feature engineering.

Input : data/raw/comments/*.txt, data/raw/profiles/*.txt
Output: data/interim/comments_structured.csv, data/interim/profiles_structured.csv

AMAN DIJALANKAN ULANG: kalau CSV outputnya sudah ada dan sudah kamu isi manual
(username, waktu, label, dll), data yang SUDAH ADA tidak akan ditimpa/dihapus.
Script ini hanya menambahkan baris BARU untuk file .txt yang belum pernah
tercatat sebelumnya (dicocokkan lewat kolom source_file). Kalau kamu
menghapus sebuah baris di CSV secara manual lalu jalankan script ini lagi,
baris itu akan muncul lagi dalam keadaan kosong (karena file .txt sumbernya
masih ada) -- itu wajar dan disengaja.
"""
from pathlib import Path
import pandas as pd
import re

COMMENTS_DIR = Path("data/raw/comments")
PROFILES_DIR = Path("data/raw/profiles")
OUTPUT_DIR = Path("data/interim")

COMMENTS_CSV = OUTPUT_DIR / "comments_structured.csv"
PROFILES_CSV = OUTPUT_DIR / "profiles_structured.csv"

COMMENTS_COLUMNS = ["source_file", "username", "comment_text", "comment_time", "post_time", "label"]
PROFILES_COLUMNS = ["source_file", "username", "following_count", "follower_count", "post_count", "bio_filled"]


def clean_text(raw: str) -> str:
    text = raw.strip()
    text = re.sub(r"\s+", " ", text)
    return text


def load_existing(csv_path: Path, columns: list) -> pd.DataFrame:
    if csv_path.exists():
        return pd.read_csv(csv_path)
    return pd.DataFrame(columns=columns)


def build_comments_table() -> pd.DataFrame:
    existing_df = load_existing(COMMENTS_CSV, COMMENTS_COLUMNS)
    existing_files = set(existing_df["source_file"]) if not existing_df.empty else set()

    new_rows = []
    for txt_file in sorted(COMMENTS_DIR.glob("*.txt")):
        if txt_file.stem in existing_files:
            continue  # sudah ada (mungkin sudah diisi manual) -- jangan disentuh
        raw = txt_file.read_text(encoding="utf-8")
        new_rows.append({
            "source_file": txt_file.stem,
            "username": None,
            "comment_text": clean_text(raw),
            "comment_time": None,
            "post_time": None,
            "label": None,
        })

    if new_rows:
        print(f"[comments] {len(new_rows)} baris BARU ditambahkan: {[r['source_file'] for r in new_rows]}")
    else:
        print("[comments] tidak ada file baru, CSV yang sudah ada tidak diubah.")

    combined = pd.concat([existing_df, pd.DataFrame(new_rows)], ignore_index=True)
    return combined


def build_profiles_table() -> pd.DataFrame:
    existing_df = load_existing(PROFILES_CSV, PROFILES_COLUMNS)
    existing_files = set(existing_df["source_file"]) if not existing_df.empty else set()

    new_rows = []
    for txt_file in sorted(PROFILES_DIR.glob("*.txt")):
        if txt_file.stem in existing_files:
            continue
        new_rows.append({
            "source_file": txt_file.stem,
            "username": None,
            "following_count": None,
            "follower_count": None,
            "post_count": None,
            "bio_filled": None,
        })

    if new_rows:
        print(f"[profiles] {len(new_rows)} baris BARU ditambahkan: {[r['source_file'] for r in new_rows]}")
    else:
        print("[profiles] tidak ada file baru, CSV yang sudah ada tidak diubah.")

    combined = pd.concat([existing_df, pd.DataFrame(new_rows)], ignore_index=True)
    return combined


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    comments_df = build_comments_table()
    profiles_df = build_profiles_table()

    comments_df.to_csv(COMMENTS_CSV, index=False)
    profiles_df.to_csv(PROFILES_CSV, index=False)

    print(f"\ncomments: {len(comments_df)} baris total -> {COMMENTS_CSV}")
    print(f"profiles: {len(profiles_df)} baris total -> {PROFILES_CSV}")


if __name__ == "__main__":
    main()