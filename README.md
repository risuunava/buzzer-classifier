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
├── notebooks/               # eksplorasi & eksperimen (Jupyter)
├── src/
│   ├── data/                # script OCR & load/clean data
│   ├── features/            # ekstraksi fitur perilaku/temporal/linguistik
│   ├── models/               # training, evaluasi, SHAP
│   └── visualization/        # plotting hasil
├── models/saved/             # model .pkl hasil training
├── reports/figures/          # grafik untuk laporan skripsi
├── codebook/                 # kriteria labeling buzzer + hasil Cohen's Kappa
├── tests/                    # unit test kecil untuk fungsi fitur
├── requirements.txt
└── .gitignore
```

## Setup

```bash
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Alur Kerja (sesuai matriks penelitian)

1. **Data collection** — screenshot manual → `data/raw/screenshots/`
2. **OCR** — `src/data/ocr_extract.py` → hasil ke `data/raw/comments/` & `data/raw/profiles/`
3. **Cleaning & structuring** — `src/data/preprocess.py` → `data/interim/`
4. **Labeling manual** — pakai `codebook/codebook.md`, hitung Cohen's Kappa via `src/data/labeling_agreement.py`
5. **Feature engineering** — `src/features/behavioral.py`, `temporal.py`, `linguistic.py` → gabung ke `data/processed/dataset.csv`
6. **Modeling** — `src/models/train.py` (Random Forest & XGBoost, k-fold CV)
7. **Evaluasi** — `src/models/evaluate.py` (Precision, Recall, F1, ROC-AUC)
8. **Interpretasi** — `src/models/explain_shap.py` → simpan grafik ke `reports/figures/`

## Catatan
- Tidak pakai API Instagram (tidak tersedia gratis) → data diambil manual via screenshot.
- Bahasa teks: Indonesia → gunakan Sastrawi untuk stemming/stopword.
