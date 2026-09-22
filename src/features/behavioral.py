"""
Fitur perilaku: rasio following/follower, kelengkapan profil, jumlah post.
"""
import pandas as pd


def add_behavioral_features(profiles_df: pd.DataFrame) -> pd.DataFrame:
    df = profiles_df.copy()

    df["following_follower_ratio"] = df["following_count"] / df["follower_count"].replace(0, 1)
    df["profile_completeness"] = df["bio_filled"].astype(int)
    df["post_count_norm"] = df["post_count"]  # bisa dinormalisasi lebih lanjut jika perlu

    return df
