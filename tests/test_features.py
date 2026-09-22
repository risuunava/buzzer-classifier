"""
Unit tests untuk src/features/behavioral.py, temporal.py, linguistic.py.
Memastikan fungsi add_*_features menghasilkan kolom yang diharapkan
dengan tipe data yang benar.
"""
import sys
from pathlib import Path
import pandas as pd
import pytest
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from features.behavioral import add_behavioral_features
from features.temporal import add_temporal_features
from features.linguistic import add_linguistic_features


# ─── Fixtures data dummy ────────────────────────────────────────────────────
@pytest.fixture
def profiles_df_normal():
    return pd.DataFrame([
        {"source_file": "profile_a", "username": "userA", "following_count": 500,
         "follower_count": 200, "post_count": 30, "bio_filled": "True"},
        {"source_file": "profile_b", "username": "userB", "following_count": 10,
         "follower_count": 5000, "post_count": 100, "bio_filled": "False"},
    ])


@pytest.fixture
def profiles_df_dengan_nan():
    return pd.DataFrame([
        {"source_file": "profile_c", "username": "userC", "following_count": None,
         "follower_count": None, "post_count": None, "bio_filled": "True"},
    ])


@pytest.fixture
def comments_df_normal():
    return pd.DataFrame([
        {"source_file": "c1", "username": "userA",
         "comment_text": "pemerintah terus mendorong ekonomi",
         "comment_time": "2026-09-21 16:34", "post_time": "2026-09-19 15:58", "label": 1},
        {"source_file": "c2", "username": "userB",
         "comment_text": "pemerintah terus mendorong ekonomi",
         "comment_time": "2026-09-22 16:34", "post_time": "2026-09-22 08:24", "label": 0},
        {"source_file": "c3", "username": "userC",
         "comment_text": "komentar berbeda sekali",
         "comment_time": "2026-09-22 10:00", "post_time": "2026-09-22 08:24", "label": 1},
    ])


@pytest.fixture
def comments_df_satu_baris():
    return pd.DataFrame([
        {"source_file": "c1", "username": "userA",
         "comment_text": "hanya satu komentar",
         "comment_time": "2026-09-21 16:34", "post_time": "2026-09-19 15:58", "label": 1},
    ])


# ─── Test behavioral ────────────────────────────────────────────────────────
class TestBehavioral:
    def test_kolom_baru_ada(self, profiles_df_normal):
        df = add_behavioral_features(profiles_df_normal)
        for kolom in ["following_follower_ratio", "profile_completeness", "post_count_norm"]:
            assert kolom in df.columns, f"Kolom '{kolom}' tidak ditemukan"

    def test_tipe_data_benar(self, profiles_df_normal):
        df = add_behavioral_features(profiles_df_normal)
        assert pd.api.types.is_float_dtype(df["following_follower_ratio"]), \
            "following_follower_ratio harus float"
        assert pd.api.types.is_integer_dtype(df["profile_completeness"]), \
            "profile_completeness harus integer (0 atau 1)"

    def test_bio_filled_string_true_dikonversi(self, profiles_df_normal):
        """String 'True' harus menghasilkan profile_completeness = 1."""
        df = add_behavioral_features(profiles_df_normal)
        assert df.loc[df["username"] == "userA", "profile_completeness"].values[0] == 1

    def test_bio_filled_string_false_dikonversi(self, profiles_df_normal):
        """String 'False' harus menghasilkan profile_completeness = 0."""
        df = add_behavioral_features(profiles_df_normal)
        assert df.loc[df["username"] == "userB", "profile_completeness"].values[0] == 0

    def test_follower_nol_tidak_menyebabkan_divisi_nol(self):
        """Follower = 0 tidak boleh menghasilkan ZeroDivisionError atau inf."""
        df_input = pd.DataFrame([{
            "source_file": "x", "username": "x",
            "following_count": 100, "follower_count": 0,
            "post_count": 0, "bio_filled": "False"
        }])
        df = add_behavioral_features(df_input)
        rasio = df["following_follower_ratio"].values[0]
        assert np.isfinite(rasio), "Rasio harus finite (tidak inf/NaN) saat follower = 0"

    def test_nan_numerik_diimputasi_tanpa_crash(self, profiles_df_dengan_nan):
        """Nilai NaN pada kolom numerik harus diimputasi 0 tanpa crash."""
        df = add_behavioral_features(profiles_df_dengan_nan)
        assert not df["following_follower_ratio"].isna().any(), \
            "following_follower_ratio tidak boleh NaN setelah imputasi"
        assert df["profile_completeness"].values[0] == 1

    def test_jumlah_baris_tidak_berubah(self, profiles_df_normal):
        df = add_behavioral_features(profiles_df_normal)
        assert len(df) == len(profiles_df_normal)


# ─── Test temporal ──────────────────────────────────────────────────────────
class TestTemporal:
    def test_kolom_baru_ada(self, comments_df_normal):
        df = add_temporal_features(comments_df_normal)
        for kolom in ["comment_delay_minutes", "comment_hour"]:
            assert kolom in df.columns, f"Kolom '{kolom}' tidak ditemukan"

    def test_delay_menit_benar(self, comments_df_normal):
        """comment_time 2026-09-21 16:34 - post_time 2026-09-19 15:58 = 2916 menit."""
        df = add_temporal_features(comments_df_normal)
        delay = df.loc[df["source_file"] == "c1", "comment_delay_minutes"].values[0]
        assert pytest.approx(delay, abs=1) == 2916, \
            f"Delay menit tidak sesuai: {delay} (diharapkan 2916)"

    def test_comment_hour_benar(self, comments_df_normal):
        """Jam 16:34 harus menghasilkan comment_hour = 16."""
        df = add_temporal_features(comments_df_normal)
        jam = df.loc[df["source_file"] == "c1", "comment_hour"].values[0]
        assert jam == 16, f"comment_hour tidak sesuai: {jam}"

    def test_timestamp_kosong_tidak_crash(self):
        """Baris dengan comment_time atau post_time kosong tidak boleh crash."""
        df_input = pd.DataFrame([{
            "source_file": "c_kosong", "username": "x",
            "comment_text": "teks", "comment_time": None,
            "post_time": "2026-09-19 10:00", "label": 1
        }])
        df = add_temporal_features(df_input)
        assert pd.isna(df["comment_delay_minutes"].values[0]), \
            "Delay harus NaN kalau comment_time kosong"

    def test_jumlah_baris_tidak_berubah(self, comments_df_normal):
        df = add_temporal_features(comments_df_normal)
        assert len(df) == len(comments_df_normal)


# ─── Test linguistic ────────────────────────────────────────────────────────
class TestLinguistic:
    def test_kolom_baru_ada(self, comments_df_normal):
        df = add_linguistic_features(comments_df_normal)
        for kolom in ["comment_length", "emoji_count", "hashtag_count",
                      "emoji_hashtag_ratio", "clean_text", "avg_similarity_to_others"]:
            assert kolom in df.columns, f"Kolom '{kolom}' tidak ditemukan"

    def test_comment_length_benar(self, comments_df_normal):
        """Panjang karakter teks komentar asli harus dihitung dengan benar."""
        df = add_linguistic_features(comments_df_normal)
        panjang = df.loc[df["source_file"] == "c1", "comment_length"].values[0]
        teks = comments_df_normal.loc[
            comments_df_normal["source_file"] == "c1", "comment_text"
        ].values[0]
        assert panjang == len(teks)

    def test_hashtag_count_benar(self):
        df_input = pd.DataFrame([{
            "source_file": "h1", "username": "u",
            "comment_text": "dua #hashtag #disini", "label": 1
        }])
        df = add_linguistic_features(df_input)
        assert df["hashtag_count"].values[0] == 2

    def test_satu_baris_tidak_crash(self, comments_df_satu_baris):
        """Kalau hanya ada 1 baris, avg_similarity_to_others harus 0.0."""
        df = add_linguistic_features(comments_df_satu_baris)
        assert df["avg_similarity_to_others"].values[0] == pytest.approx(0.0)

    def test_avg_similarity_komentar_identik_tinggi(self):
        """Dua komentar yang identik harus punya avg_similarity mendekati 1."""
        df_input = pd.DataFrame([
            {"source_file": "s1", "username": "u1",
             "comment_text": "pemerintah terus mendorong ekonomi", "label": 1},
            {"source_file": "s2", "username": "u2",
             "comment_text": "pemerintah terus mendorong ekonomi", "label": 1},
        ])
        df = add_linguistic_features(df_input)
        # Dua komentar sama persis harus punya similarity mendekati 1.0
        assert df["avg_similarity_to_others"].mean() > 0.8, \
            "Komentar identik harus punya similarity tinggi"

    def test_teks_hanya_stopword_tidak_crash(self):
        """Kalau setelah preprocessing teks jadi kosong, tidak boleh crash."""
        df_input = pd.DataFrame([
            {"source_file": "sw1", "username": "u1",
             "comment_text": "dan atau dengan", "label": 0},
            {"source_file": "sw2", "username": "u2",
             "comment_text": "ini itu mereka", "label": 0},
        ])
        # Test ini memastikan tidak ada ValueError "empty vocabulary"
        try:
            df = add_linguistic_features(df_input)
            # Tidak crash adalah kondisi lulus
        except ValueError as e:
            pytest.fail(f"Teks stopword menyebabkan crash: {e}")

    def test_jumlah_baris_tidak_berubah(self, comments_df_normal):
        df = add_linguistic_features(comments_df_normal)
        assert len(df) == len(comments_df_normal)
