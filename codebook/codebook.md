# Codebook Pelabelan Akun Terindikasi Buzzer

## Definisi Operasional
Akun dilabeli **1 (terindikasi buzzer)** jika memenuhi mayoritas kriteria berikut,
**0 (bukan buzzer)** jika sebaliknya.

## Kriteria
| No | Kriteria | Indikator |
|----|----------|-----------|
| 1 | Kelengkapan profil rendah | Tidak ada foto profil asli, bio kosong/generik |
| 2 | Rasio following/follower tidak wajar | Following jauh lebih tinggi dari follower, atau sebaliknya secara ekstrem |
| 3 | Waktu komentar tersinkronisasi | Muncul dalam rentang waktu singkat bersama akun lain di postingan yang sama |
| 4 | Kemiripan konten tinggi | Narasi/kalimat mirip dengan komentar akun lain |
| 5 | Pola pembelaan berulang | Nada defensif terhadap kebijakan tanpa argumentasi substantif |
| 6 | Usia akun/aktivitas mencurigakan | Post minim tapi aktif berkomentar secara masif |

## Instruksi Annotator
1. Baca komentar & profil akun secara independen (tanpa diskusi dengan annotator lain)
2. Isi kolom `label` = 1 atau 0 berdasarkan kriteria di atas
3. Simpan hasil masing-masing sebagai `labels_annotator1.csv` / `labels_annotator2.csv`
   di `data/interim/`, dengan kolom minimal: `source_file, label`
4. Jalankan `src/data/labeling_agreement.py` untuk cek Cohen's Kappa
5. Jika Kappa < 0.6, diskusikan perbedaan & revisi kriteria sebelum lanjut ke fase modeling
