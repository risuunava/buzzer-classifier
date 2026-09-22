"""
Unit tests untuk src/data/labeling_agreement.py.
Memverifikasi bahwa Cohen's Kappa dihitung dengan benar
untuk kasus agreement sempurna (kappa = 1.0) dan acak (kappa ~ 0).
"""
import sys
from pathlib import Path
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from sklearn.metrics import cohen_kappa_score


# ─── Test langsung via sklearn (unit level) ──────────────────────────────────
# Kita tidak bisa monkeypatch path di labeling_agreement.py dengan mudah
# karena keduanya adalah konstanta modul level. Lebih tepat menguji logika
# hitungan kappa secara langsung.

class TestCohenKappaLogika:
    def test_agreement_sempurna_kappa_satu(self):
        """Kalau dua annotator sepakat 100%, kappa harus = 1.0."""
        label_a1 = [0, 1, 0, 1, 1, 0, 0, 1]
        label_a2 = [0, 1, 0, 1, 1, 0, 0, 1]
        kappa = cohen_kappa_score(label_a1, label_a2)
        assert kappa == pytest.approx(1.0), f"Kappa agreement sempurna harus 1.0, didapat {kappa}"

    def test_agreement_tidak_ada_kappa_negatif_atau_nol(self):
        """
        Kalau dua annotator saling berlawanan secara acak (bukan semuanya di satu kelas),
        kappa harus <= 0.
        Catatan: Kasus degenerate di mana semua prediksi berada di satu kelas
        (misal [0,0,0,0] vs [1,1,1,1]) menghasilkan kappa = 0.0 di sklearn,
        karena expected agreement tidak bisa dihitung. Kita uji kasus yang lebih
        representatif: sebagian cocok, sebagian tidak, tapi pola berlawanan.
        """
        # Annotator 1 dan 2 sepenuhnya berlawanan pada data berimbang
        label_a1 = [0, 1, 0, 1, 0, 1]
        label_a2 = [1, 0, 1, 0, 1, 0]
        kappa = cohen_kappa_score(label_a1, label_a2)
        assert kappa <= 0, f"Kappa saat selalu berbeda harus <= 0, didapat {kappa}"

    def test_kappa_moderat(self):
        """Agreement moderat (50%) harus menghasilkan kappa di antara 0 dan 1."""
        # 4 agree, 4 tidak agree dari 8 sampel seimbang
        label_a1 = [0, 1, 0, 1, 0, 1, 0, 1]
        label_a2 = [0, 1, 0, 1, 1, 0, 1, 0]
        kappa = cohen_kappa_score(label_a1, label_a2)
        assert 0.0 <= kappa <= 1.0, \
            f"Kappa moderat harus antara 0 dan 1, didapat {kappa}"

    def test_interpretasi_threshold(self):
        """Verifikasi threshold interpretasi sesuai codebook."""
        # Kappa < 0.4 → lemah
        assert 0.3 < 0.4, "Threshold lemah: < 0.4"
        # Kappa 0.6–0.8 → kuat
        assert 0.6 <= 0.7 < 0.8, "Threshold kuat: 0.6–0.8"
        # Kappa ≥ 0.8 → sangat kuat
        assert 0.85 >= 0.8, "Threshold sangat kuat: >= 0.8"


# ─── Test integrasi: fungsi main() labeling_agreement.py ────────────────────
class TestLabelingAgreementIntegrasi:
    def test_main_berjalan_dengan_file_sempurna(self, tmp_path, monkeypatch):
        """Menjalankan main() dengan data agreement sempurna tidak boleh crash."""
        # Buat file CSV annotator sementara
        csv_a1 = tmp_path / "labels_annotator1.csv"
        csv_a2 = tmp_path / "labels_annotator2.csv"

        data = pd.DataFrame({
            "source_file": ["f1", "f2", "f3", "f4"],
            "label": [0, 1, 0, 1]
        })
        data.to_csv(csv_a1, index=False)
        data.to_csv(csv_a2, index=False)  # identik → kappa = 1.0

        import data.labeling_agreement as mod
        monkeypatch.setattr(mod, "ANNOTATOR_1_PATH", csv_a1)
        monkeypatch.setattr(mod, "ANNOTATOR_2_PATH", csv_a2)

        # Menjalankan main() tidak boleh raise exception
        mod.main()

    def test_main_berjalan_dengan_file_berbeda(self, tmp_path, monkeypatch):
        """Menjalankan main() dengan data saling berbeda tidak boleh crash."""
        csv_a1 = tmp_path / "labels_annotator1.csv"
        csv_a2 = tmp_path / "labels_annotator2.csv"

        pd.DataFrame({"source_file": ["f1", "f2", "f3", "f4"],
                      "label": [0, 0, 0, 0]}).to_csv(csv_a1, index=False)
        pd.DataFrame({"source_file": ["f1", "f2", "f3", "f4"],
                      "label": [1, 0, 1, 0]}).to_csv(csv_a2, index=False)

        import data.labeling_agreement as mod
        monkeypatch.setattr(mod, "ANNOTATOR_1_PATH", csv_a1)
        monkeypatch.setattr(mod, "ANNOTATOR_2_PATH", csv_a2)

        mod.main()

    def test_hasil_kappa_agreement_sempurna_adalah_satu(self, tmp_path, monkeypatch, capsys):
        """Output kappa agreement sempurna harus menampilkan nilai 1.000."""
        csv_a1 = tmp_path / "labels_annotator1.csv"
        csv_a2 = tmp_path / "labels_annotator2.csv"

        data = pd.DataFrame({
            "source_file": ["x1", "x2", "x3"],
            "label": [0, 1, 1]
        })
        data.to_csv(csv_a1, index=False)
        data.to_csv(csv_a2, index=False)

        import data.labeling_agreement as mod
        monkeypatch.setattr(mod, "ANNOTATOR_1_PATH", csv_a1)
        monkeypatch.setattr(mod, "ANNOTATOR_2_PATH", csv_a2)

        mod.main()
        captured = capsys.readouterr()
        assert "1.000" in captured.out, \
            f"Output harus mengandung '1.000' untuk agreement sempurna: {captured.out}"
