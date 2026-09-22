# Klasifikasi Akun Terindikasi Buzzer Politik di Instagram

Skripsi: Klasifikasi Akun Terindikasi Buzzer Politik di Instagram Berdasarkan Fitur
Perilaku, Temporal, dan Linguistik Menggunakan Random Forest dan XGBoost
(Studi Kasus: Komentar pada Isu Kebijakan Publik Periode 2024–2029)

## Struktur Proyek

```
buzzer-classifier/
├── data/
│   ├── raw/                # data mentah, TIDAK diubah setelah masuk
│   │   ├── screenshots/    # hasil GoFullPage/Fireshot (.png/.jpg)
│   │   ├── comments/       # hasil OCR komentar (belum dibersihkan)
│   │   └── profiles/       # hasil OCR/manual data profil akun
│   ├── interim/            # data hasil cleaning tapi belum final
│   └── processed/          # dataset siap training (fitur + label)
├── notebooks/              # eksplorasi & eksperimen (Jupyter)
├── src/
│   ├── data/               # script OCR, clean data, validasi
│   │   ├── ocr_extract.py
│   │   ├── preprocess.py
│   │   ├── validate.py         # validasi data otomatis (BARU)
│   │   └── labeling_agreement.py
│   ├── features/           # ekstraksi fitur perilaku/temporal/linguistik
│   └── models/             # training, evaluasi, SHAP
├── tests/                  # unit tests (pytest)
├── models/saved/           # model .pkl hasil training
├── reports/figures/        # grafik untuk laporan skripsi
├── codebook/               # kriteria labeling buzzer + hasil Cohen's Kappa
├── run_pipeline.py         # CLI orkestrasi pipeline (BARU)
├── requirements.txt
└── .gitignore
```

---

## Setup

```bash
# 1. Buat virtual environment
python -m venv venv

# 2. Aktifkan venv
venv\Scripts\activate          # Windows PowerShell
# source venv/bin/activate     # Mac/Linux

# 3. Install dependensi (termasuk pytest)
pip install -r requirements.txt
```

> **Prasyarat sistem:** Tesseract OCR harus diinstall secara terpisah (bukan lewat pip).
> - Windows: https://github.com/UB-Mannheim/tesseract/wiki
> - Pastikan bahasa Indonesia tersedia: `tesseract-ocr-ind`

---

## Alur Kerja Cepat — via `run_pipeline.py`

Setelah setup, semua tahap pipeline bisa dijalankan dengan satu perintah:

```bash
# Jalankan seluruh pipeline (preprocess → validate → build-dataset → train → evaluate → explain)
python run_pipeline.py all
```

Atau jalankan tahap individual:

```bash
python run_pipeline.py ocr           # OCR screenshot → .txt (jalankan manual kalau ada screenshot baru)
python run_pipeline.py preprocess    # .txt → comments_structured.csv & profiles_structured.csv
python run_pipeline.py validate      # validasi data otomatis (cek anomali, missing, duplikat)
python run_pipeline.py build-dataset # gabung CSV → dataset.csv + feature engineering
python run_pipeline.py train         # training Random Forest & XGBoost
python run_pipeline.py evaluate      # cetak Precision, Recall, F1, ROC-AUC
python run_pipeline.py explain       # SHAP summary plot → reports/figures/
```

---

## Alur Kerja Lengkap (Manual + Otomatis)

1. **Data collection** — screenshot manual → `data/raw/screenshots/`
   - Nama file wajib berawalan `comment_` atau `profile_`

2. **OCR** — `python run_pipeline.py ocr`
   - Hasil → `data/raw/comments/*.txt` & `data/raw/profiles/*.txt`

3. **Cleaning & structuring** — `python run_pipeline.py preprocess`
   - Hasil → `data/interim/comments_structured.csv` & `profiles_structured.csv`
   - **AMAN DIULANG:** tidak menimpa baris yang sudah diisi manual

4. **Isi data manual** — buka kedua CSV, isi kolom yang masih kosong:
   - `comments_structured.csv`: `username`, `comment_time`, `post_time`, `label`
   - `profiles_structured.csv`: `username`, `following_count`, `follower_count`, `post_count`, `bio_filled`

5. **Labeling manual** — pakai `codebook/codebook.md`, isi kolom `label` (0=bukan buzzer, 1=buzzer)
   - Hitung reliabilitas antar-annotator: `python src/data/labeling_agreement.py`

6. **Validasi data** — `python run_pipeline.py validate`
   - Cek username match, anomali temporal, label kosong, nilai negatif, duplikat
   - Exit code 0 = lulus, exit code 1 = ada isu fatal (perbaiki dulu sebelum lanjut)

7. **Feature engineering** — `python run_pipeline.py build-dataset`
   - Hasil → `data/processed/dataset.csv`

8. **Training model** — `python run_pipeline.py train`
   - Random Forest & XGBoost, k-fold Stratified CV (n_splits otomatis disesuaikan jika data sedikit)
   - Hasil → `models/saved/random_forest.pkl` & `xgboost.pkl`

9. **Evaluasi** — `python run_pipeline.py evaluate`
   - Precision, Recall, F1, ROC-AUC

10. **Interpretasi SHAP** — `python run_pipeline.py explain`
    - Grafik kontribusi fitur → `reports/figures/`

---

## Menjalankan Unit Tests

```bash
# Jalankan semua test
python -m pytest tests/ -v

# Dengan ringkasan singkat
python -m pytest tests/
```

Test mencakup:
- `tests/test_preprocess.py` — sifat merge-safe (baris lama tidak ditimpa)
- `tests/test_features.py` — fungsi fitur behavioral, temporal, linguistic + edge cases
- `tests/test_labeling_agreement.py` — logika Cohen's Kappa

---

## Catatan Penting

- **Tidak pakai API Instagram** (tidak tersedia gratis) → data diambil manual via screenshot.
- **Bahasa teks:** Indonesia → gunakan Sastrawi untuk stemming/stopword.
- **Data sensitif:** `data/raw/`, `data/interim/`, `data/processed/`, `models/saved/` di-gitignore.
  Jangan commit data yang mengandung informasi pribadi akun tanpa anonimisasi.
- **Warning kelas minoritas:** Jika jumlah sampel per kelas < 5, training tetap berjalan
  tapi hasilnya **tidak valid untuk skripsi** — kumpulkan lebih banyak data terlebih dahulu.
