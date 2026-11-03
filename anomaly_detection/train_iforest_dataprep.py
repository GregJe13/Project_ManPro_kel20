"""
Latih IsolationForest pada dataset hasil preprocess_dataprep.py.

Jalankan:  python train_iforest_dataprep.py
Pilihan:   --seed 42  --train-days 60
Hasil:     output/dataprep/hasil_iforest.csv, alarm_iforest.csv, kejadian_iforest.csv, iforest_final.joblib

Model belajar kondisi normal dan diuji secara walk-forward: tiap bulan April-Agustus dinilai oleh model yang hanya
dilatih dari 60 hari sebelumnya. Fungsi latih dan penilaiannya sama dengan model_iforest.py, supaya hasilnya sebanding.

Data latih : usable_train, tanpa 16 kejadian dan zona sekitarnya.
Data uji   : tabel online, semua menit data_ok di luar zona grace. Episode pola kebocoran tetap dinilai.
"""
import argparse

import joblib
import pandas as pd

from evaluation import SUMMARY_COLUMNS, evaluate, persist_alarm, summarize, to_episodes
from model_iforest import anomaly_score, fit_iforest
from preprocess_dataprep import FEATURES, OUT_DIR, load_tables
from preprocess_iforest import EVENTS, FOLDS, SEED, TRAIN_DAYS, neutral_mask


def normal_rows(train_table):
    """Baris yang boleh dipelajari sebagai normal. Jika data bocor ikut dipelajari, bocor dianggap normal."""
    keep = train_table["usable_train"].to_numpy() & ~neutral_mask(train_table.index)
    return train_table.loc[keep, FEATURES]


def test_rows(online_table):
    keep = online_table["data_ok"] & ~online_table["in_grace"]
    return online_table.loc[keep, FEATURES]


def prepare_fold(normal, test, start, end, train_days):
    """NaN sisa diisi median data latih, supaya tidak memakai informasi data uji."""
    window = normal[(normal.index < start) & (normal.index >= start - pd.Timedelta(days=train_days))]
    fold_test = test[(test.index >= start) & (test.index < end)]
    medians = window.median()
    return window.fillna(medians).to_numpy("float32"), fold_test.fillna(medians).to_numpy("float32"), fold_test.index


def walk_forward(normal, test, seed=SEED, train_days=TRAIN_DAYS):
    flags, scores = [], []
    for start, end in FOLDS:
        X_train, X_test, test_index = prepare_fold(normal, test, start, end, train_days)
        model, threshold = fit_iforest(X_train, seed)
        test_scores = anomaly_score(model, X_test)
        scores.append(pd.Series(test_scores, index=test_index))
        flags.append(pd.Series(test_scores > threshold, index=test_index))
        print(f"  uji {start:%b}: {len(X_train):>6,} baris latih, {len(X_test):>6,} baris uji, threshold {threshold:.4f}", flush=True)
    return flags, scores


def event_report(flags):
    """Hasil per kejadian dan daftar episode alarm, untuk laporan dan pemeriksaan teknisi."""
    episodes = to_episodes(persist_alarm(pd.concat(flags).sort_index()))
    result = evaluate(episodes)

    first_alarm = pd.Series(result["first_alarm"])
    events = EVENTS.set_index("id")[["start", "end", "official"]].copy()
    events["alarm_pertama"] = pd.to_datetime(first_alarm)
    events["delay_menit"] = (events["alarm_pertama"] - events["start"]).dt.total_seconds() / 60

    false_alarms = set(result["false_alarms"])
    alarms = pd.DataFrame(episodes, columns=["mulai", "selesai"])
    alarms["menit"] = ((alarms["selesai"] - alarms["mulai"]).dt.total_seconds() / 60 + 1).astype(int)
    alarms["status"] = ["alarm palsu" if pair in false_alarms else "menyentuh kejadian atau perbaikan" for pair in episodes]
    return events, alarms


def train_final(normal, seed=SEED, train_days=TRAIN_DAYS):
    """Model akhir dari 60 hari terakhir. Model ini tidak punya data uji, jadi kinerjanya belum terukur langsung."""
    window = normal[normal.index >= normal.index.max() - pd.Timedelta(days=train_days)]
    medians = window.median()
    model, threshold = fit_iforest(window.fillna(medians).to_numpy("float32"), seed)
    path = OUT_DIR / "iforest_final.joblib"
    joblib.dump({"model": model, "threshold": float(threshold), "medians": medians, "features": FEATURES}, path)
    return path


def main():
    parser = argparse.ArgumentParser(description="Latih IsolationForest pada dataset preprocess_dataprep.py")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--train-days", type=int, default=TRAIN_DAYS)
    args = parser.parse_args()

    train_table, online_table = load_tables()
    normal, test = normal_rows(train_table), test_rows(online_table)
    del train_table, online_table
    print(f"{len(FEATURES)} fitur | {len(normal):,} baris normal | {len(test):,} baris uji\n")

    flags, scores = walk_forward(normal, test, args.seed, args.train_days)
    summary = pd.DataFrame([summarize("IsolationForest", "walk", "16", flags, scores)])
    events, alarms = event_report(flags)

    summary.to_csv(OUT_DIR / "hasil_iforest.csv", index=False)
    alarms.to_csv(OUT_DIR / "alarm_iforest.csv", index=False)
    events.to_csv(OUT_DIR / "kejadian_iforest.csv")

    print("\n" + summary[SUMMARY_COLUMNS].to_string(index=False))
    print("\nPer kejadian:")
    print(events[["official", "alarm_pertama", "delay_menit"]].to_string())
    print(f"\nModel akhir: {train_final(normal, args.seed, args.train_days)}")


if __name__ == "__main__":
    main()
