"""
OCR untuk mengubah screenshot komentar/profil Instagram menjadi teks.
Input : data/raw/screenshots/*.png
Output: data/raw/comments/*.txt  atau  data/raw/profiles/*.txt

Butuh Tesseract OCR terinstall di sistem (bukan cuma pip install pytesseract):
- Windows: https://github.com/UB-Mannheim/tesseract/wiki
- Mac: brew install tesseract
- Linux: sudo apt install tesseract-ocr tesseract-ocr-ind
"""
import platform
from pathlib import Path
import pytesseract
from PIL import Image

# Windows tidak selalu otomatis menambahkan Tesseract ke PATH saat instalasi,
# jadi kita set manual kalau file-nya ada di lokasi default.
if platform.system() == "Windows":
    default_win_path = Path(r"C:\Program Files\Tesseract-OCR\tesseract.exe")
    if default_win_path.exists():
        pytesseract.pytesseract.tesseract_cmd = str(default_win_path)

RAW_SCREENSHOT_DIR = Path("data/raw/screenshots")
COMMENTS_OUTPUT_DIR = Path("data/raw/comments")
PROFILES_OUTPUT_DIR = Path("data/raw/profiles")

# Kalau perlu bahasa Indonesia: pastikan file tesseract-ocr-ind sudah terinstall
OCR_LANG = "ind+eng"

# Nama file yang diawali salah satu prefix ini akan dianggap screenshot PROFIL.
# Selain itu (default) dianggap screenshot KOMENTAR.
# Rename file screenshot profil dulu di File Explorer, contoh:
#   profile_haris_s3245.png
PROFILE_PREFIXES = ("profile_", "profil_")


def extract_text_from_image(image_path: Path) -> str:
    image = Image.open(image_path)
    text = pytesseract.image_to_string(image, lang=OCR_LANG)
    return text


def is_profile_screenshot(img_path: Path) -> bool:
    name_lower = img_path.stem.lower()
    return name_lower.startswith(PROFILE_PREFIXES)


def run_batch():
    COMMENTS_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    PROFILES_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    images = list(RAW_SCREENSHOT_DIR.glob("*.png")) + list(RAW_SCREENSHOT_DIR.glob("*.jpg"))

    if not images:
        print(f"Tidak ada gambar di {RAW_SCREENSHOT_DIR}. Taruh screenshot di sana dulu.")
        return

    for img_path in images:
        text = extract_text_from_image(img_path)

        if is_profile_screenshot(img_path):
            out_dir = PROFILES_OUTPUT_DIR
            kind = "PROFIL"
        else:
            out_dir = COMMENTS_OUTPUT_DIR
            kind = "KOMENTAR"

        out_path = out_dir / f"{img_path.stem}.txt"
        out_path.write_text(text, encoding="utf-8")
        print(f"[OK - {kind}] {img_path.name} -> {out_path}")


if __name__ == "__main__":
    run_batch()