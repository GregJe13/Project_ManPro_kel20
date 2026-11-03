"""
Latih HistGradientBoosting pada dataset hasil preprocess_dataprep.py.

Jalankan:  python train_hgb_dataprep.py
Pilihan:   --labels 16 4  --seed 42
Hasil:     output/hybrid/hasil_hgb.csv

Model belajar dari contoh kejadian, jadi baru aktif setelah ada kejadian yang selesai sebelum bulan uji.
Label utama adalah 16 kejadian paper. Skenario 4 label resmi hanya untuk uji ketahanan: 12 kejadian lainnya tidak
diketahui model, kecuali yang tersaring otomatis sebagai episode pola kebocoran.
Setelan model dan ambang 0,5 sama dengan model_supervised.py, tanpa tuning.

Fitur: 51 fitur menghasilkan 6 alarm palsu, melewati batas 5 yang ditetapkan sebelum pengujian. Sesuai rencana cadangan,
HGB memakai 9 fitur: 4 alarm palsu dengan deteksi sama (12/12 sesudah April). Pilihan ini sudah melihat hasil uji.
"""
import argparse

import numpy as np
import pandas as pd

from evaluation import SUMMARY_COLUMNS, summarize
from model_supervised import DEFAULT_THRESHOLD, build_model
from preprocess_dataprep import load_tables
from preprocess_iforest import FOLDS, OUTPUT_DIR, SEED, event_masks, select_events
from preprocess_supervised import LABEL_COVERAGE
from train_iforest_dataprep import test_rows

HYBRID_DIR = OUTPUT_DIR / "hybrid"
HGB_FEATURES = ["duty_15", "duty_60", "load_run_min", "cycles_60", "oil_mean_15", "oil_trend_60", "curr_mean_60", "tp2_mean_60", "h1_mean_60"]


def labeled_rows(train_table, label_set, cut):
    """Positif: kejadian yang selesai sebelum 'cut'. Negatif: baris usable_train di luar zona kejadian.

    Negatif hanya dari periode berlabel. Di luar April-Juli, "tidak berlabel" bukan berarti normal.
    """
    rows = train_table[train_table.index < cut]
    y, outside_zone, n_events = event_masks(rows.index, select_events(label_set), cut=cut)
    labeled_period = (rows.index >= LABEL_COVERAGE[0]) & (rows.index < LABEL_COVERAGE[1])
    positive = (y == 1) & rows["data_ok"].to_numpy()
    negative = (y == 0) & outside_zone & rows["usable_train"].to_numpy() & labeled_period
    keep = positive | negative
    return rows.loc[keep, HGB_FEATURES].to_numpy("float32"), y[keep], n_events


def fit_hgb(train_table, label_set, cut, seed=SEED):
    """Mengembalikan model dan jumlah kejadian yang diketahui. Tanpa kejadian, belum ada model."""
    X, y, n_events = labeled_rows(train_table, label_set, cut)
    if n_events == 0:
        return None, 0
    return build_model("hgb", len(X), seed).fit(X, y), n_events


def hgb_score(model, rows):
    """HGB memproses NaN langsung, jadi fitur tidak perlu diisi."""
    if model is None:
        return np.zeros(len(rows))
    return model.predict_proba(rows[HGB_FEATURES].to_numpy("float32"))[:, 1]


def walk_forward(train_table, test, label_set="16", seed=SEED, threshold=DEFAULT_THRESHOLD):
    flags, scores = [], []
    for start, end in FOLDS:
        model, n_events = fit_hgb(train_table, label_set, start, seed)
        fold_test = test[(test.index >= start) & (test.index < end)]
        proba = hgb_score(model, fold_test)
        scores.append(pd.Series(proba, index=fold_test.index))
        flags.append(pd.Series(proba > threshold, index=fold_test.index))
        print(f"  {label_set} label, uji {start:%b}: {n_events:>2} kejadian diketahui, {'aktif' if model else 'belum aktif'}", flush=True)
    return flags, scores


def main():
    parser = argparse.ArgumentParser(description="Latih HGB pada dataset preprocess_dataprep.py")
    parser.add_argument("--labels", nargs="+", default=["16", "4"], choices=["16", "4"])
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    train_table, online_table = load_tables()
    test = test_rows(online_table)
    del online_table

    rows = []
    for label_set in args.labels:
        flags, scores = walk_forward(train_table, test, label_set, args.seed)
        rows.append(summarize("HistGradientBoosting", "walk", label_set, flags, scores))

    results = pd.DataFrame(rows)
    HYBRID_DIR.mkdir(parents=True, exist_ok=True)
    results.to_csv(HYBRID_DIR / "hasil_hgb.csv", index=False)
    print("\n" + results[SUMMARY_COLUMNS + ["meleset"]].to_string(index=False))


if __name__ == "__main__":
    main()
