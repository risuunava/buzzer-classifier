"""
Fitur temporal: jeda waktu komentar terhadap unggahan, jam aktif komentar.
Asumsi comment_time & post_time berformat datetime (isi manual saat cleaning).
"""
import pandas as pd


def add_temporal_features(comments_df: pd.DataFrame) -> pd.DataFrame:
    df = comments_df.copy()

    df["comment_time"] = pd.to_datetime(df["comment_time"], errors="coerce")
    df["post_time"] = pd.to_datetime(df["post_time"], errors="coerce")

    df["comment_delay_minutes"] = (
        (df["comment_time"] - df["post_time"]).dt.total_seconds() / 60
    )
    df["comment_hour"] = df["comment_time"].dt.hour

    return df
