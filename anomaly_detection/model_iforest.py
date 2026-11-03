"""
Model IsolationForest: mempelajari kondisi normal, lalu memberi alarm jika data berbeda dari itu.

Jalankan:  python model_iforest.py
Pilihan:   --labels 4 16  --schemes walk block  --seed 42  --train-days 60
Hasil:     output/hasil_iforest.csv dan output/iforest_final.joblib

Catatan jendela latih: 60 hari lebih baik di Apr-Mei (10-11 alarm palsu, melawan 14-15 jika memakai semua data
sebelumnya), tetapi kalah di Jun-Agu (44 melawan 28-29). Bukti Apr-Mei lemah karena putaran April identik.
Jika beralih ke semua data, tulis di laporan bahwa pilihan itu didasarkan pada hasil Jun-Agu.
"""
import argparse

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

from evaluation import SUMMARY_COLUMNS, summarize
from preprocess_iforest import (
    FEATURES,
    FOLDS,
    N_JOBS,
    OUTPUT_DIR,
    SEED,
    TRAIN_DAYS,
    boxplot_threshold,
    load_tables,
    prepare_final_iforest,
    prepare_fold_iforest,
    self_clean,
    valid_rows,
)


def anomaly_score(model, X):
    """Makin tinggi makin aneh. score_samples sklearn bernilai sebaliknya."""
    return -model.score_samples(X)


def fit_iforest(X_train, seed=SEED):
    """Bersihkan data latih, latih model, lalu hitung threshold (langkah 10) dari skor data latih saja."""
    X_clean = self_clean(X_train, seed)
    model = IsolationForest(n_estimators=200, max_samples=4096, random_state=seed, n_jobs=N_JOBS).fit(X_clean)
    return model, boxplot_threshold(anomaly_score(model, X_clean))


def run_scenario(train_valid, online_valid, label_set, scheme, seed=SEED, train_days=TRAIN_DAYS):
    flags, scores = [], []
    for start, end in FOLDS:
        fold = prepare_fold_iforest(train_valid, online_valid, start, end, label_set, scheme, train_days)
        model, threshold = fit_iforest(fold.X_train, seed)
        test_scores = anomaly_score(model, fold.X_test)
        scores.append(pd.Series(test_scores, index=fold.test_index))
        flags.append(pd.Series(test_scores > threshold, index=fold.test_index))
    return summarize("IsolationForest", scheme, label_set, flags, scores)


def train_final(train_valid, seed=SEED, train_days=TRAIN_DAYS):
    """Model akhir dari 60 hari terakhir. Model ini tidak punya data uji, jadi kinerjanya belum terukur langsung."""
    X_train, medians = prepare_final_iforest(train_valid, train_days)
    model, threshold = fit_iforest(X_train, seed)
    path = OUTPUT_DIR / "iforest_final.joblib"
    joblib.dump({"model": model, "threshold": float(threshold), "medians": medians, "features": FEATURES}, path)
    return path


def main():
    parser = argparse.ArgumentParser(description="Latih dan nilai IsolationForest")
    parser.add_argument("--labels", nargs="+", default=["4", "16"], choices=["4", "16"])
    parser.add_argument("--schemes", nargs="+", default=["walk", "block"], choices=["walk", "block"])
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--train-days", type=int, default=TRAIN_DAYS)
    args = parser.parse_args()

    train_table, online_table = load_tables()
    train_valid, online_valid = valid_rows(train_table), valid_rows(online_table)

    rows = []
    for scheme in args.schemes:
        for label_set in args.labels:
            rows.append(run_scenario(train_valid, online_valid, label_set, scheme, args.seed, args.train_days))
            print(f"selesai: IsolationForest, {scheme}, {label_set} label", flush=True)

    results = pd.DataFrame(rows)
    results.to_csv(OUTPUT_DIR / "hasil_iforest.csv", index=False)
    print("\n" + results[SUMMARY_COLUMNS].to_string(index=False))
    print(f"\nModel akhir: {train_final(train_valid, args.seed, args.train_days)}")


if __name__ == "__main__":
    main()
