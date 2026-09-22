"""
Fitur perilaku: rasio following/follower, kelengkapan profil, jumlah post.
"""
import pandas as pd


def add_behavioral_features(profiles_df: pd.DataFrame) -> pd.DataFrame:
    df = profiles_df.copy()

    # Konversi bio_filled: handle baik bool asli maupun string "True"/"False"
    # (pandas membaca CSV sebagai string, bukan bool)
    bio = df["bio_filled"].astype(str).str.strip().str.lower()
    df["bio_filled"] = bio.map({"true": True, "false": False, "1": True, "0": False})

    # Imputasi nilai kosong pada kolom numerik profil dengan 0
    # (keputusan: tidak drop baris agar tidak kehilangan sampel langka)
    for kolom in ["following_count", "follower_count", "post_count"]:
        n_kosong = df[kolom].isna().sum()
        if n_kosong:
            print(f"[behavioral] '{kolom}' punya {n_kosong} nilai kosong — diisi 0 (imputasi).")
        df[kolom] = pd.to_numeric(df[kolom], errors="coerce").fillna(0).astype(int)

    # Hitung rasio; hindari pembagian dengan nol
    df["following_follower_ratio"] = (
        df["following_count"] / df["follower_count"].replace(0, 1)
    )
    df["profile_completeness"] = df["bio_filled"].astype(int)
    df["post_count_norm"] = df["post_count"]  # bisa dinormalisasi lebih lanjut jika perlu

    return df
