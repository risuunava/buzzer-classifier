"""
Hitung reliabilitas antar-annotator (Cohen's Kappa) untuk labeling manual
buzzer / bukan-buzzer.

Input: dua file CSV hasil labeling dari 2 annotator berbeda, dengan kolom
       'source_file' dan 'label' (label: 1 = buzzer, 0 = bukan buzzer)
"""
from pathlib import Path
import pandas as pd
from sklearn.metrics import cohen_kappa_score

ANNOTATOR_1_PATH = Path("data/interim/labels_annotator1.csv")
ANNOTATOR_2_PATH = Path("data/interim/labels_annotator2.csv")


def main():
    df1 = pd.read_csv(ANNOTATOR_1_PATH)
    df2 = pd.read_csv(ANNOTATOR_2_PATH)

    merged = df1.merge(df2, on="source_file", suffixes=("_a1", "_a2"))

    kappa = cohen_kappa_score(merged["label_a1"], merged["label_a2"])
    print(f"Jumlah data yang dibandingkan: {len(merged)}")
    print(f"Cohen's Kappa: {kappa:.3f}")

    if kappa < 0.4:
        print("Interpretasi: agreement lemah, codebook perlu direvisi")
    elif kappa < 0.6:
        print("Interpretasi: agreement moderat")
    elif kappa < 0.8:
        print("Interpretasi: agreement kuat")
    else:
        print("Interpretasi: agreement sangat kuat")


if __name__ == "__main__":
    main()
