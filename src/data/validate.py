"""
Validasi otomatis data CSV sebelum build_dataset.py dijalankan.

Cek yang dilakukan:
  1. Username match antara comments ↔ profiles (bidirectional)
  2. Anomali temporal: comment_time < post_time untuk pasangan username+post_time
  3. Label kosong (warning, bukan fatal)
  4. Nilai negatif pada follower_count, following_count, post_count
  5. Duplikat source_file dalam masing-masing CSV

Keluar dengan exit code 0 kalau tidak ada isu FATAL, exit code 1 kalau ada.
"""
import sys
from pathlib import Path
import pandas as pd

COMMENTS_PATH = Path("data/interim/comments_structured.csv")
PROFILES_PATH = Path("data/interim/profiles_structured.csv")

# ─── Warna terminal sederhana ───────────────────────────────────────────────
GREEN  = "\033[92m"
YELLOW = "\033[93m"
RED    = "\033[91m"
RESET  = "\033[0m"
BOLD   = "\033[1m"


def ok(msg: str):
    print(f"  {GREEN}[OK]{RESET} {msg}")


def warn(msg: str):
    print(f"  {YELLOW}[WARNING]{RESET} {msg}")


def fatal(msg: str):
    print(f"  {RED}[FATAL]{RESET} {msg}")


def load_csv(path: Path, nama: str) -> pd.DataFrame | None:
    if not path.exists():
        fatal(f"File tidak ditemukan: {path}  (jalankan preprocess.py dulu)")
        return None
    df = pd.read_csv(path)
    return df


# ─── Cek 1: username match ───────────────────────────────────────────────────
def cek_username_match(comments_df: pd.DataFrame, profiles_df: pd.DataFrame) -> int:
    """
    Mengembalikan jumlah isu fatal (username tidak punya pasangan).
    Ini dianggap fatal karena inner join akan membuang baris tersebut
    secara diam-diam, dan peneliti mungkin tidak sadar.
    """
    print(f"\n{BOLD}[1] Cek kecocokan username comments <-> profiles{RESET}")
    fatal_count = 0

    # username yang terdaftar di comments
    comments_usernames = set(comments_df["username"].dropna())
    # username yang terdaftar di profiles
    profiles_usernames = set(profiles_df["username"].dropna())

    tidak_ada_profil = comments_usernames - profiles_usernames
    tidak_ada_komentar = profiles_usernames - comments_usernames

    if tidak_ada_profil:
        fatal(
            f"{len(tidak_ada_profil)} username di comments TIDAK punya profil "
            f"→ akan terbuang saat inner join: {sorted(tidak_ada_profil)}"
        )
        fatal_count += len(tidak_ada_profil)
    else:
        ok("Semua username di comments punya data profil.")

    if tidak_ada_komentar:
        warn(
            f"{len(tidak_ada_komentar)} username di profiles TIDAK punya komentar "
            f"→ tidak memengaruhi dataset (hanya info): {sorted(tidak_ada_komentar)}"
        )
    else:
        ok("Semua username di profiles punya data komentar.")

    return fatal_count


# ─── Cek 2: anomali temporal ─────────────────────────────────────────────────
def cek_temporal(comments_df: pd.DataFrame) -> int:
    """
    Komentar yang comment_time-nya lebih awal dari post_time adalah anomali
    (mustahil secara fisik).  Dianggap fatal karena akan menghasilkan
    comment_delay_minutes negatif → bias fitur.
    """
    print(f"\n{BOLD}[2] Cek anomali temporal (comment_time < post_time){RESET}")
    fatal_count = 0

    df = comments_df.copy()
    df["comment_time"] = pd.to_datetime(df["comment_time"], errors="coerce")
    df["post_time"]    = pd.to_datetime(df["post_time"],    errors="coerce")

    # Baris yang punya kedua nilai (tidak NaN)
    lengkap = df.dropna(subset=["comment_time", "post_time"])
    anomali = lengkap[lengkap["comment_time"] < lengkap["post_time"]]

    if anomali.empty:
        ok("Tidak ada anomali temporal.")
    else:
        fatal(
            f"{len(anomali)} baris comment_time lebih awal dari post_time "
            f"→ source_file: {list(anomali['source_file'])}"
        )
        fatal_count += len(anomali)

    # Info tambahan: berapa baris yang timestamp-nya masih kosong
    n_kosong_comment = df["comment_time"].isna().sum()
    n_kosong_post    = df["post_time"].isna().sum()
    if n_kosong_comment or n_kosong_post:
        warn(
            f"{n_kosong_comment} baris comment_time kosong, "
            f"{n_kosong_post} baris post_time kosong — tidak bisa diperiksa temporalnya."
        )

    return fatal_count


# ─── Cek 3: label kosong ─────────────────────────────────────────────────────
def cek_label(comments_df: pd.DataFrame) -> int:
    """Warning saja, bukan fatal (dataset parsial masih boleh untuk testing)."""
    print(f"\n{BOLD}[3] Cek kelengkapan kolom label{RESET}")
    n_kosong = comments_df["label"].isna().sum()

    if n_kosong == 0:
        ok("Semua baris sudah punya label.")
    else:
        warn(
            f"{n_kosong} baris belum punya label (0/1). "
            "Baris ini tidak bisa dipakai untuk training model. "
            "Isi dulu kolom label di comments_structured.csv."
        )
    # Cek nilai label di luar 0/1
    label_valid = comments_df["label"].dropna()
    label_aneh = label_valid[~label_valid.isin([0, 1, 0.0, 1.0])]
    if not label_aneh.empty:
        fatal_count = len(label_aneh)
        fatal(
            f"{fatal_count} baris punya label selain 0/1: "
            f"{list(label_aneh.values)}"
        )
        return fatal_count
    return 0


# ─── Cek 4: nilai negatif ────────────────────────────────────────────────────
def cek_nilai_negatif(profiles_df: pd.DataFrame) -> int:
    """Nilai negatif pada kolom numerik profil adalah data entry error."""
    print(f"\n{BOLD}[4] Cek nilai negatif (follower/following/post_count){RESET}")
    fatal_count = 0
    kolom_numerik = ["following_count", "follower_count", "post_count"]

    for kolom in kolom_numerik:
        if kolom not in profiles_df.columns:
            warn(f"Kolom '{kolom}' tidak ditemukan di profiles CSV — skip.")
            continue
        seri = pd.to_numeric(profiles_df[kolom], errors="coerce")
        negatif = seri[seri < 0]
        if negatif.empty:
            ok(f"'{kolom}' tidak ada nilai negatif.")
        else:
            fatal(
                f"'{kolom}' punya {len(negatif)} nilai negatif di baris: "
                f"{list(profiles_df.loc[negatif.index, 'source_file'])}"
            )
            fatal_count += len(negatif)

    return fatal_count


# ─── Cek 5: duplikat source_file ─────────────────────────────────────────────
def cek_duplikat(comments_df: pd.DataFrame, profiles_df: pd.DataFrame) -> int:
    """Duplikat source_file menyebabkan baris ganda saat join."""
    print(f"\n{BOLD}[5] Cek duplikat source_file{RESET}")
    fatal_count = 0

    for nama, df in [("comments", comments_df), ("profiles", profiles_df)]:
        duplikat = df[df.duplicated("source_file", keep=False)]
        if duplikat.empty:
            ok(f"Tidak ada duplikat source_file di {nama}.")
        else:
            files = list(duplikat["source_file"].unique())
            fatal(f"{len(files)} source_file duplikat di {nama}: {files}")
            fatal_count += len(files)

    return fatal_count


# ─── Main ────────────────────────────────────────────────────────────────────
def main():
    print(f"\n{BOLD}{'='*55}")
    print("  VALIDASI DATA — Buzzer Classifier")
    print(f"{'='*55}{RESET}")

    # Muat CSV
    comments_df = load_csv(COMMENTS_PATH, "comments")
    profiles_df = load_csv(PROFILES_PATH, "profiles")

    if comments_df is None or profiles_df is None:
        print(f"\n{RED}Validasi dibatalkan karena file tidak ditemukan.{RESET}")
        sys.exit(1)

    print(f"\nData dimuat: {len(comments_df)} baris comments, {len(profiles_df)} baris profiles.")

    # Jalankan semua pengecekan
    total_fatal = 0
    total_fatal += cek_username_match(comments_df, profiles_df)
    total_fatal += cek_temporal(comments_df)
    total_fatal += cek_label(comments_df)
    total_fatal += cek_nilai_negatif(profiles_df)
    total_fatal += cek_duplikat(comments_df, profiles_df)

    # Ringkasan
    print(f"\n{BOLD}{'='*55}")
    if total_fatal == 0:
        print(f"  {GREEN}HASIL: LULUS — Tidak ada isu fatal.{RESET}")
        print(f"{BOLD}{'='*55}{RESET}\n")
        sys.exit(0)
    else:
        print(f"  {RED}HASIL: GAGAL — {total_fatal} isu fatal ditemukan.{RESET}")
        print(f"  Perbaiki isu di atas sebelum menjalankan build_dataset.py.")
        print(f"{BOLD}{'='*55}{RESET}\n")
        sys.exit(1)


if __name__ == "__main__":
    main()
