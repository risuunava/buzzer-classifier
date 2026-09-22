"""
Unit tests untuk src/data/preprocess.py.
Memastikan fungsi merge-safe tidak menimpa baris yang sudah ada di CSV.
"""
import io
import sys
from pathlib import Path

import pandas as pd
import pytest

# Tambahkan src ke sys.path supaya import relatif bisa dipakai dari tests/
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from data.preprocess import build_comments_table, build_profiles_table, COMMENTS_CSV, PROFILES_CSV, OUTPUT_DIR


# ─── Fixture: reset CSV sebelum dan sesudah setiap tes ─────────────────────
@pytest.fixture(autouse=True)
def bersihkan_csv_sementara(tmp_path, monkeypatch):
    """
    Monkeypatch semua path CSV dan direktori output ke tmp_path
    agar tes tidak menyentuh file proyek asli.
    """
    import data.preprocess as mod

    tmp_comments_csv = tmp_path / "comments_structured.csv"
    tmp_profiles_csv = tmp_path / "profiles_structured.csv"
    tmp_comments_dir = tmp_path / "comments"
    tmp_profiles_dir = tmp_path / "profiles"
    tmp_comments_dir.mkdir()
    tmp_profiles_dir.mkdir()

    monkeypatch.setattr(mod, "COMMENTS_CSV", tmp_comments_csv)
    monkeypatch.setattr(mod, "PROFILES_CSV", tmp_profiles_csv)
    monkeypatch.setattr(mod, "COMMENTS_DIR", tmp_comments_dir)
    monkeypatch.setattr(mod, "PROFILES_DIR", tmp_profiles_dir)
    monkeypatch.setattr(mod, "OUTPUT_DIR", tmp_path)

    yield {
        "comments_csv": tmp_comments_csv,
        "profiles_csv": tmp_profiles_csv,
        "comments_dir": tmp_comments_dir,
        "profiles_dir": tmp_profiles_dir,
    }


class TestBuildCommentsTable:
    def test_baris_baru_ditambahkan_kalau_file_txt_ada(self, bersihkan_csv_sementara):
        """Kalau ada file .txt baru, baris baru harus muncul di DataFrame hasil."""
        txt = bersihkan_csv_sementara["comments_dir"] / "comment_akun1.txt"
        txt.write_text("komentar dari akun1", encoding="utf-8")

        import data.preprocess as mod
        df = mod.build_comments_table()

        assert len(df) == 1
        assert df.iloc[0]["source_file"] == "comment_akun1"
        assert "akun1" not in str(df.iloc[0]["username"])  # username masih None/NaN

    def test_merge_safe_tidak_timpa_baris_lama(self, bersihkan_csv_sementara):
        """
        Kalau CSV sudah ada dan sudah ada baris yang diisi manual,
        menjalankan ulang build_comments_table() TIDAK boleh mengubah baris itu.
        """
        # Simulasi: baris sudah ada di CSV dengan data diisi manual
        csv_path = bersihkan_csv_sementara["comments_csv"]
        existing_data = pd.DataFrame([{
            "source_file": "comment_akun1",
            "username": "akun1_manual",
            "comment_text": "sudah diisi manual",
            "comment_time": "2026-09-20 10:00",
            "post_time": "2026-09-19 08:00",
            "label": 1,
        }])
        existing_data.to_csv(csv_path, index=False)

        # File .txt yang sudah terdaftar di CSV
        txt = bersihkan_csv_sementara["comments_dir"] / "comment_akun1.txt"
        txt.write_text("teks OCR mentah yang berbeda", encoding="utf-8")

        import data.preprocess as mod
        df = mod.build_comments_table()

        # Baris lama harus TIDAK berubah
        assert len(df) == 1
        assert df.iloc[0]["username"] == "akun1_manual"
        assert df.iloc[0]["comment_text"] == "sudah diisi manual"
        assert int(df.iloc[0]["label"]) == 1

    def test_file_baru_ditambahkan_tanpa_hapus_yang_lama(self, bersihkan_csv_sementara):
        """Kalau ada file .txt baru DAN CSV sudah punya baris lama, keduanya harus ada."""
        csv_path = bersihkan_csv_sementara["comments_csv"]
        existing_data = pd.DataFrame([{
            "source_file": "comment_lama",
            "username": "user_lama",
            "comment_text": "teks lama",
            "comment_time": None,
            "post_time": None,
            "label": 0,
        }])
        existing_data.to_csv(csv_path, index=False)

        # File baru yang belum ada di CSV
        txt_baru = bersihkan_csv_sementara["comments_dir"] / "comment_baru.txt"
        txt_baru.write_text("teks komentar baru", encoding="utf-8")

        import data.preprocess as mod
        df = mod.build_comments_table()

        assert len(df) == 2
        source_files = list(df["source_file"])
        assert "comment_lama" in source_files
        assert "comment_baru" in source_files

    def test_clean_text_hapus_whitespace_berlebih(self, bersihkan_csv_sementara):
        """Fungsi clean_text harus menggabungkan spasi/newline berlebih."""
        txt = bersihkan_csv_sementara["comments_dir"] / "comment_spasi.txt"
        txt.write_text("  banyak   spasi\n\n  dan newline  ", encoding="utf-8")

        import data.preprocess as mod
        df = mod.build_comments_table()

        assert df.iloc[0]["comment_text"] == "banyak spasi dan newline"


class TestBuildProfilesTable:
    def test_baris_baru_profil_ditambahkan(self, bersihkan_csv_sementara):
        """Kalau ada file .txt profil baru, baris baru dengan kolom kosong harus muncul."""
        txt = bersihkan_csv_sementara["profiles_dir"] / "profile_akun1.txt"
        txt.write_text("profil akun1", encoding="utf-8")

        import data.preprocess as mod
        df = mod.build_profiles_table()

        assert len(df) == 1
        assert df.iloc[0]["source_file"] == "profile_akun1"
        assert pd.isna(df.iloc[0]["username"])

    def test_merge_safe_tidak_timpa_profil_lama(self, bersihkan_csv_sementara):
        """Baris profil yang sudah diisi manual tidak boleh ditimpa."""
        csv_path = bersihkan_csv_sementara["profiles_csv"]
        existing_data = pd.DataFrame([{
            "source_file": "profile_akun1",
            "username": "akun1",
            "following_count": 500,
            "follower_count": 200,
            "post_count": 30,
            "bio_filled": True,
        }])
        existing_data.to_csv(csv_path, index=False)

        txt = bersihkan_csv_sementara["profiles_dir"] / "profile_akun1.txt"
        txt.write_text("data profil OCR", encoding="utf-8")

        import data.preprocess as mod
        df = mod.build_profiles_table()

        assert len(df) == 1
        assert df.iloc[0]["username"] == "akun1"
        assert int(df.iloc[0]["following_count"]) == 500
