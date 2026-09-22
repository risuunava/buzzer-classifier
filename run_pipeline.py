"""
Orkestrasi pipeline Buzzer Classifier via CLI.

Penggunaan:
    python run_pipeline.py ocr
    python run_pipeline.py preprocess
    python run_pipeline.py validate
    python run_pipeline.py build-dataset
    python run_pipeline.py train
    python run_pipeline.py evaluate
    python run_pipeline.py explain
    python run_pipeline.py all        # jalankan semua berurutan, berhenti jika ada yang gagal
"""
import argparse
import sys
import importlib.util
from pathlib import Path


# ─── Pastikan src/ ada di sys.path ──────────────────────────────────────────
SRC_DIR = Path(__file__).resolve().parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


# ─── Helper: cetak header tahap ─────────────────────────────────────────────
BOLD  = "\033[1m"
GREEN = "\033[92m"
RED   = "\033[91m"
CYAN  = "\033[96m"
RESET = "\033[0m"


def cetak_tahap(nama: str):
    print(f"\n{BOLD}{CYAN}{'='*55}")
    print(f"  PIPELINE: {nama.upper()}")
    print(f"{'='*55}{RESET}")


def cetak_ok(nama: str):
    print(f"{GREEN}[OK]{RESET} {nama} selesai.")


def cetak_gagal(nama: str, error: Exception):
    print(f"{RED}[GAGAL]{RESET} {nama} error: {error}")


# ─── Subcommand: ocr ────────────────────────────────────────────────────────
def jalankan_ocr():
    cetak_tahap("OCR")
    from data.ocr_extract import run_batch
    run_batch()


# ─── Subcommand: preprocess ─────────────────────────────────────────────────
def jalankan_preprocess():
    cetak_tahap("Preprocess")
    from data.preprocess import main
    main()


# ─── Subcommand: validate ───────────────────────────────────────────────────
def jalankan_validate():
    """
    validate.py menggunakan sys.exit() sendiri.
    Kita tangkap SystemExit supaya pipeline 'all' bisa mendeteksi kegagalan.
    """
    cetak_tahap("Validasi Data")
    from data.validate import main
    main()


# ─── Subcommand: build-dataset ──────────────────────────────────────────────
def jalankan_build_dataset():
    cetak_tahap("Build Dataset")
    from features.build_dataset import main
    main()


# ─── Subcommand: train ──────────────────────────────────────────────────────
def jalankan_train():
    cetak_tahap("Training Model")
    from models.train import main
    main()


# ─── Subcommand: evaluate ───────────────────────────────────────────────────
def jalankan_evaluate():
    cetak_tahap("Evaluasi Model")
    from models.evaluate import main
    main()


# ─── Subcommand: explain ────────────────────────────────────────────────────
def jalankan_explain():
    cetak_tahap("SHAP Explain")
    from models.explain_shap import main
    main()


# ─── Subcommand: all ────────────────────────────────────────────────────────
URUTAN_PIPELINE = [
    ("preprocess",    jalankan_preprocess,    "Preprocess"),
    ("validate",      jalankan_validate,      "Validasi Data"),
    ("build-dataset", jalankan_build_dataset, "Build Dataset"),
    ("train",         jalankan_train,         "Training Model"),
    ("evaluate",      jalankan_evaluate,      "Evaluasi Model"),
    ("explain",       jalankan_explain,       "SHAP Explain"),
]


def jalankan_all():
    """
    Jalankan semua tahap pipeline secara berurutan.
    Berhenti langsung jika ada tahap yang gagal.
    Note: tahap OCR tidak dimasukkan ke 'all' karena memerlukan
          screenshot fisik yang sudah disiapkan manual terlebih dahulu.
    """
    print(f"\n{BOLD}{'='*55}")
    print("  MENJALANKAN PIPELINE PENUH")
    print(f"  (OCR dilewati — jalankan manual kalau ada screenshot baru)")
    print(f"{'='*55}{RESET}")

    for kunci, fungsi, nama in URUTAN_PIPELINE:
        try:
            fungsi()
            cetak_ok(nama)
        except SystemExit as e:
            # validate.py pakai sys.exit(1) saat ada isu fatal
            if e.code != 0:
                print(f"\n{RED}[PIPELINE BERHENTI]{RESET} Tahap '{nama}' "
                      f"keluar dengan kode {e.code}. Perbaiki isu tersebut dulu.")
                sys.exit(1)
        except Exception as e:
            cetak_gagal(nama, e)
            print(f"\n{RED}[PIPELINE BERHENTI]{RESET} Perbaiki error di atas, "
                  "lalu jalankan ulang.")
            sys.exit(1)

    print(f"\n{BOLD}{GREEN}{'='*55}")
    print("  PIPELINE SELESAI TANPA ERROR")
    print(f"{'='*55}{RESET}\n")


# ─── Main / argparse ────────────────────────────────────────────────────────
SUBCOMMAND_MAP = {
    "ocr":           jalankan_ocr,
    "preprocess":    jalankan_preprocess,
    "validate":      jalankan_validate,
    "build-dataset": jalankan_build_dataset,
    "train":         jalankan_train,
    "evaluate":      jalankan_evaluate,
    "explain":       jalankan_explain,
    "all":           jalankan_all,
}


def main():
    parser = argparse.ArgumentParser(
        prog="run_pipeline.py",
        description=(
            "CLI orkestrasi pipeline Buzzer Classifier.\n"
            "Kerjakan berurutan: preprocess -> validate -> build-dataset "
            "-> train -> evaluate -> explain\n"
            "Atau jalankan semuanya sekaligus dengan: python run_pipeline.py all"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "perintah",
        choices=list(SUBCOMMAND_MAP.keys()),
        metavar="PERINTAH",
        help=(
            "Pilihan: "
            + ", ".join(SUBCOMMAND_MAP.keys())
        ),
    )

    args = parser.parse_args()
    fungsi = SUBCOMMAND_MAP[args.perintah]

    try:
        fungsi()
    except SystemExit:
        raise  # biarkan sys.exit dari validate meneruskan exit code
    except Exception as e:
        print(f"\n{RED}[ERROR]{RESET} {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
