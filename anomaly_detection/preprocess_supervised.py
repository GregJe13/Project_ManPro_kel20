"""
Preprocessing MetroPT-3 untuk metode supervised.

Jalankan:  python preprocess_supervised.py

Tahap A (langkah 1-5) sengaja diimpor dari preprocess_iforest.py, bukan ditulis ulang.
Kedua metode harus memakai fitur dan baris uji yang persis sama agar perbandingannya adil.

Yang berbeda dari IsolationForest:
- Data latih berisi normal DAN kejadian, dari semua periode lampau (kejadian sedikit, jadi semuanya dipakai).
- Data normal hanya diambil dari periode berlabel. Ini langkah paling berpengaruh.
- Label: 1 di dalam kejadian, 0 di luar. Masa sebelum kegagalan tidak dilabeli positif.
- Tidak ada self-cleaning, dan threshold berupa probabilitas tetap (lihat model_supervised.py).
"""
from typing import NamedTuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from preprocess_iforest import (
    FEATURES,
    FOLDS,
    event_masks,
    load_tables,
    select_events,
    test_rows_for_fold,
    train_rows_for_fold,
    valid_rows,
)

# Label paper hanya mencakup April-Juli. Di luar itu, "tidak berlabel" bukan berarti "normal".
# Maret punya 8 episode panas tanpa label; jika dianggap normal, HGB 16 label turun dari 12/16 ke 8/16.
LABEL_COVERAGE = (pd.Timestamp("2020-04-01"), pd.Timestamp("2020-08-01"))


class SupervisedFold(NamedTuple):
    X_train: np.ndarray
    y_train: np.ndarray
    X_test: np.ndarray
    test_index: pd.DatetimeIndex
    n_events: int


def labeled_training_data(rows, label_set, cut, pre_hours=0):
    """Buat y dan pilih baris yang boleh dipakai.

    Pada skenario 4 label, 12 kejadian lainnya SENGAJA tetap berlabel 0. Tanpa label, kejadian itu memang tidak
    diketahui, jadi menyingkirkannya memberi supervised keuntungan yang tidak ada pada kondisi nyata.

    pre_hours=0 adalah setelan utama. Menambahkan 2 jam sebelum kejadian membuat alarm palsu naik dari 0 ke 10
    tanpa membuat alarm lebih cepat.
    """
    y, usable, n_events = event_masks(rows.index, select_events(label_set), cut=cut, pre_hours=pre_hours)
    labeled_period = (rows.index >= LABEL_COVERAGE[0]) & (rows.index < LABEL_COVERAGE[1])
    keep = usable & ((y == 1) | labeled_period)
    return rows.loc[keep, FEATURES].to_numpy(np.float32), y[keep], n_events


def prepare_fold_supervised(train_valid, online_valid, start, end, label_set="16", scheme="walk", pre_hours=0):
    """Data satu putaran. Pada walk-forward, putaran April tidak punya label positif, jadi model belum bisa dilatih."""
    rows = train_rows_for_fold(train_valid, start, end, scheme)
    X_train, y_train, n_events = labeled_training_data(rows, label_set, start if scheme == "walk" else None, pre_hours)
    test = test_rows_for_fold(online_valid, start, end)
    return SupervisedFold(X_train, y_train, test[FEATURES].to_numpy(np.float32), test.index, n_events)


def prepare_final_supervised(train_valid, label_set="16"):
    """Data latih untuk model akhir: semua periode berlabel dan semua kejadian."""
    cut = train_valid.index.max() + pd.Timedelta(minutes=1)
    X_train, y_train, _ = labeled_training_data(train_valid, label_set, cut)
    return X_train, y_train


def prepare_for_model(kind, X_train, X_test):
    """Penyesuaian per jenis model, dihitung dari data latih saja.

    hgb: bisa memproses NaN langsung dan tidak terpengaruh skala.
    rf : perlu angka lengkap, tetapi tidak terpengaruh skala.
    lr : perlu angka lengkap dan scaling.
    """
    if kind == "hgb":
        return X_train, X_test
    medians = np.nanmedian(X_train, axis=0)
    X_train = np.where(np.isnan(X_train), medians, X_train)
    X_test = np.where(np.isnan(X_test), medians, X_test)
    if kind == "lr":
        scaler = StandardScaler().fit(X_train)
        return scaler.transform(X_train), scaler.transform(X_test)
    return X_train, X_test


if __name__ == "__main__":
    train_table, online_table = load_tables()
    train_valid, online_valid = valid_rows(train_table), valid_rows(online_table)
    for label_set in ("4", "16"):
        print(f"\nData per putaran ({label_set} label, walk-forward):")
        for start, end in FOLDS:
            fold = prepare_fold_supervised(train_valid, online_valid, start, end, label_set)
            print(
                f"  uji {start:%b}: {fold.n_events:>2} kejadian, {int(fold.y_train.sum()):>6,} baris positif, "
                f"{int((fold.y_train == 0).sum()):>7,} baris negatif, {len(fold.X_test):>6,} baris uji"
            )
