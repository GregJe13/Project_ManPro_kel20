"""
Model supervised: mempelajari perbedaan kondisi kejadian dan normal dari label.

Jalankan:  python model_supervised.py
Pilihan:   --models hgb rf lr  --labels 4 16  --schemes walk block  --threshold 0.5  --pre-hours 0  --seed 42
Hasil:     output/hasil_supervised.csv dan output/supervised_hgb_final.joblib

Threshold berupa probabilitas tetap 0,5, ditetapkan sebelum melihat data uji. Kejadian terlalu sedikit untuk
menyetelnya, dan class_weight="balanced" mengimbangi jumlah baris kejadian yang jauh lebih sedikit.
SMOTE tidak dipakai: baris sintetis dari kejadian yang sama tidak menambah kejadian baru.
"""
import argparse

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from evaluation import SUMMARY_COLUMNS, summarize
from preprocess_iforest import FEATURES, FOLDS, N_JOBS, OUTPUT_DIR, SEED, load_tables, valid_rows
from preprocess_supervised import prepare_final_supervised, prepare_fold_supervised, prepare_for_model

MODEL_NAMES = {"hgb": "HistGradientBoosting", "rf": "RandomForest", "lr": "LogisticRegression"}
DEFAULT_THRESHOLD = 0.5


def build_model(kind, n_rows, seed=SEED):
    if kind == "hgb":
        return HistGradientBoostingClassifier(max_iter=200, class_weight="balanced", random_state=seed, early_stopping=False)
    if kind == "rf":
        return RandomForestClassifier(
            n_estimators=100,
            max_depth=12,
            min_samples_leaf=5,
            class_weight="balanced_subsample",
            max_samples=min(50_000, n_rows),
            n_jobs=N_JOBS,
            random_state=seed,
        )
    return LogisticRegression(class_weight="balanced", max_iter=500)


def run_scenario(train_valid, online_valid, kind, label_set, scheme, threshold=DEFAULT_THRESHOLD, pre_hours=0, seed=SEED):
    flags, scores = [], []
    for start, end in FOLDS:
        fold = prepare_fold_supervised(train_valid, online_valid, start, end, label_set, scheme, pre_hours)
        if fold.y_train.sum() == 0:
            # Belum ada label kejadian, jadi belum ada model. Kejadian di bulan ini dihitung terlewat, bukan dibuang.
            probabilities = np.zeros(len(fold.X_test))
        else:
            X_train, X_test = prepare_for_model(kind, fold.X_train, fold.X_test)
            model = build_model(kind, len(X_train), seed).fit(X_train, fold.y_train)
            probabilities = model.predict_proba(X_test)[:, 1]
        scores.append(pd.Series(probabilities, index=fold.test_index))
        flags.append(pd.Series(probabilities > threshold, index=fold.test_index))
    return summarize(MODEL_NAMES[kind], scheme, label_set, flags, scores)


def train_final(train_valid, threshold=DEFAULT_THRESHOLD, seed=SEED):
    """Model akhir HGB dari semua periode berlabel. Model ini tidak punya data uji, jadi kinerjanya belum terukur langsung."""
    X_train, y_train = prepare_final_supervised(train_valid)
    model = build_model("hgb", len(X_train), seed).fit(X_train, y_train)
    path = OUTPUT_DIR / "supervised_hgb_final.joblib"
    joblib.dump({"model": model, "threshold": threshold, "features": FEATURES}, path)
    return path


def main():
    parser = argparse.ArgumentParser(description="Latih dan nilai model supervised")
    parser.add_argument("--models", nargs="+", default=["hgb", "rf", "lr"], choices=list(MODEL_NAMES))
    parser.add_argument("--labels", nargs="+", default=["4", "16"], choices=["4", "16"])
    parser.add_argument("--schemes", nargs="+", default=["walk", "block"], choices=["walk", "block"])
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    parser.add_argument("--pre-hours", type=float, default=0)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    train_table, online_table = load_tables()
    train_valid, online_valid = valid_rows(train_table), valid_rows(online_table)

    rows = []
    for kind in args.models:
        for scheme in args.schemes:
            for label_set in args.labels:
                rows.append(run_scenario(train_valid, online_valid, kind, label_set, scheme, args.threshold, args.pre_hours, args.seed))
                print(f"selesai: {MODEL_NAMES[kind]}, {scheme}, {label_set} label", flush=True)

    results = pd.DataFrame(rows)
    results.to_csv(OUTPUT_DIR / "hasil_supervised.csv", index=False)
    print("\n" + results[SUMMARY_COLUMNS].to_string(index=False))
    print(f"\nModel akhir: {train_final(train_valid, args.threshold, args.seed)}")


if __name__ == "__main__":
    main()
