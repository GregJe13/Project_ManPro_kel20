"""
Sistem hybrid: IsolationForest untuk Kuning dan cold start, HistGradientBoosting untuk Merah.

Jalankan:  python train_hybrid.py
Pilihan:   --seeds 42 7  --labels 16 4  --train-days 60
Hasil:     output/hybrid/
             hasil_hybrid.csv            ringkasan per metode, label, dan seed
             kejadian_hybrid.csv         Kuning dan Merah pertama per kejadian
             alarm_hybrid.csv            log episode alarm beserta nilai sensor
             tradeoff_hybrid.csv         variasi ambang, hanya untuk kurva trade-off
             status_walkforward.parquet  status per menit untuk replay dashboard
             hybrid_final.joblib         kedua model akhir

Walk-forward: tiap bulan April-Agustus dinilai oleh model yang hanya dilatih dari data sebelum bulan itu.
Berkas selain hasil_hybrid.csv memakai skenario utama: seed pertama dan label pertama.
Ambang dikunci sebelum melihat hasil: IsolationForest Q3 + 3*IQR, HGB 0,5, cold start 60 menit.
"""
import argparse

import joblib
import numpy as np
import pandas as pd

from evaluation import evaluate, to_episodes
from hybrid import STATUS_COLUMNS, combine_status, red_alarm, to_grid, warning_alarm
from model_iforest import anomaly_score, fit_iforest
from model_supervised import DEFAULT_THRESHOLD
from preprocess_dataprep import FEATURES, load_tables
from preprocess_iforest import EVENTS, FOLDS, SEED, TRAIN_DAYS, event_masks, select_events, self_clean
from train_hgb_dataprep import HGB_FEATURES, HYBRID_DIR, fit_hgb, hgb_score
from train_iforest_dataprep import prepare_fold

MAY = pd.Timestamp("2020-05-01")
AUG = pd.Timestamp("2020-08-01")
RULE_RUN_MIN, RULE_OIL = 15, 60
SENSOR_COLUMNS = ["TP3", "Oil_temperature", "Motor_current", "duty_60", "load_run_min"]
TRADEOFF_K = (1.5, 3, 4.5, 6)
TRADEOFF_HGB = (0.3, 0.5, 0.7, 0.9)


def normal_rows(train_table, label_set):
    """Baris normal untuk IsolationForest. Pada skenario 4 label, kejadian lainnya memang tidak diketahui."""
    y, outside_zone, _ = event_masks(train_table.index, select_events(label_set))
    keep = train_table["usable_train"].to_numpy() & outside_zone & (y == 0)
    return train_table.loc[keep, FEATURES]


def score_folds(train_table, online_rows, label_set, seed, train_days, with_quartiles=False):
    """Skor kedua model untuk semua menit data_ok pada April-Agustus."""
    normal = normal_rows(train_table, label_set)
    parts = []
    for start, end in FOLDS:
        X_train, X_test, index = prepare_fold(normal, online_rows[FEATURES], start, end, train_days)
        iforest, if_threshold = fit_iforest(X_train, seed)
        hgb, n_events = fit_hgb(train_table, label_set, start, seed)
        part = pd.DataFrame(
            {
                "if_score": anomaly_score(iforest, X_test),
                "if_threshold": if_threshold,
                "hgb_score": hgb_score(hgb, online_rows.loc[index]),
                "hgb_threshold": DEFAULT_THRESHOLD,
                "hgb_aktif": hgb is not None,
            },
            index=index,
        )
        if with_quartiles:
            # self_clean memakai seed yang sama, jadi datanya identik dengan yang dipakai fit_iforest.
            part["if_q1"], part["if_q3"] = np.quantile(anomaly_score(iforest, self_clean(X_train, seed)), [0.25, 0.75])
        parts.append(part)
        print(f"  {label_set} label, seed {seed}, uji {start:%b}: {len(X_train):>6,} baris normal, {n_events:>2} kejadian, HGB {'aktif' if hgb else 'belum aktif'}", flush=True)
    return pd.concat(parts).sort_index()


def summary_row(method, label_set, seed, alarm):
    result = evaluate(to_episodes(alarm))
    first_alarm, false_alarms = result["first_alarm"], result["false_alarms"]

    def hits(ids):
        return f"{sum(first_alarm[i] is not None for i in ids)}/{len(ids)}"

    return {
        "metode": method,
        "label": label_set,
        "seed": seed,
        "kejadian": hits(EVENTS["id"]),
        "resmi": hits(EVENTS.loc[EVENTS["official"], "id"]),
        "april": hits(EVENTS.loc[EVENTS["start"] < MAY, "id"]),
        "mei_agu": hits(EVENTS.loc[EVENTS["start"] >= MAY, "id"]),
        "alarm_palsu": len(false_alarms),
        "palsu_april": sum(a < MAY for a, _ in false_alarms),
        "palsu_mei_jul": sum(MAY <= a < AUG for a, _ in false_alarms),
        "palsu_agustus": sum(a >= AUG for a, _ in false_alarms),
        "delay_median_menit": result["median_delay"],
        "meleset": ",".join(k for k, t in first_alarm.items() if t is None),
    }


def rule_alarm(rows):
    """Pembanding tanpa model. Ambangnya dipilih setelah melihat kejadian, jadi hasilnya bukan bukti yang adil."""
    return (rows["load_run_min"] >= RULE_RUN_MIN) & (rows["oil_mean_15"].fillna(0) >= RULE_OIL)


def event_table(frame):
    """Kuning dan Merah pertama per kejadian. Delay dibatasi 2 jam sebelum mulai, sesuai aturan evaluation.py."""
    warning = pd.to_datetime(pd.Series(evaluate(to_episodes(warning_alarm(frame)))["first_alarm"]))
    red = pd.to_datetime(pd.Series(evaluate(to_episodes(red_alarm(frame)))["first_alarm"]))
    table = EVENTS.set_index("id")[["start", "end", "official"]].copy()
    table["peringatan_pertama"] = warning
    table["merah_pertama"] = red
    table["delay_peringatan_menit"] = (table["peringatan_pertama"] - table["start"]).dt.total_seconds() / 60
    table["delay_merah_menit"] = (table["merah_pertama"] - table["start"]).dt.total_seconds() / 60
    table["selisih_ke_merah_menit"] = (table["merah_pertama"] - table["peringatan_pertama"]).dt.total_seconds() / 60
    table["tertinggi"] = np.select(
        [table["merah_pertama"].notna(), table["peringatan_pertama"].notna()], ["Merah", "Kuning"], "Tidak terdeteksi"
    )
    return table


def alarm_log(frame, rows):
    """Satu baris per episode peringatan, dengan nilai sensor saat alarm mulai."""
    warning = warning_alarm(frame)
    red = red_alarm(frame)
    episodes = to_episodes(warning)
    false_warning = set(evaluate(episodes)["false_alarms"])
    false_red = evaluate(to_episodes(red))["false_alarms"]
    records = []
    for a, b in episodes:
        red_minutes = red.loc[a:b]
        red_minutes = red_minutes.index[red_minutes.to_numpy()]
        first_red = red_minutes[0] if len(red_minutes) else pd.NaT
        record = {
            "mulai": a,
            "selesai": b,
            "menit": int((b - a).total_seconds() / 60) + 1,
            "tertinggi": "Merah" if len(red_minutes) else "Kuning",
            "merah_mulai": first_red,
            "pemicu_merah": frame.loc[first_red, "pemicu"] if len(red_minutes) else "-",
            "penilaian": "alarm palsu" if (a, b) in false_warning else "menyentuh kejadian atau perbaikan",
            "merah_palsu": any(x <= b and y >= a for x, y in false_red),
        }
        record.update({column: round(float(rows.loc[a, column]), 3) for column in SENSOR_COLUMNS})
        records.append(record)
    return pd.DataFrame(records)


def tradeoff(scored):
    """Hanya untuk kurva Tujuan 3. Jangan memilih ambang terbaik dari tabel ini: kejadiannya sama dengan data uji."""
    rows = []
    for k in TRADEOFF_K:
        for hgb_threshold in TRADEOFF_HGB:
            trial = scored.assign(if_threshold=scored["if_q3"] + k * (scored["if_q3"] - scored["if_q1"]), hgb_threshold=hgb_threshold)
            frame = combine_status(trial[["if_score", "if_threshold", "hgb_score", "hgb_threshold", "hgb_aktif"]])
            warning = evaluate(to_episodes(warning_alarm(frame)))
            red = evaluate(to_episodes(red_alarm(frame)))
            rows.append(
                {
                    "if_k": k,
                    "hgb_ambang": hgb_threshold,
                    "peringatan_kejadian": warning["hits"],
                    "peringatan_palsu": len(warning["false_alarms"]),
                    "peringatan_delay": warning["median_delay"],
                    "merah_kejadian": red["hits"],
                    "merah_palsu": len(red["false_alarms"]),
                    "merah_delay": red["median_delay"],
                }
            )
    return pd.DataFrame(rows)


def train_final(train_table, seed, train_days):
    """Model akhir untuk data sesudah Agustus. Model ini tidak punya data uji, jadi kinerjanya belum terukur langsung."""
    normal = normal_rows(train_table, "16")
    window = normal[normal.index >= normal.index.max() - pd.Timedelta(days=train_days)]
    medians = window.median()
    iforest, if_threshold = fit_iforest(window.fillna(medians).to_numpy("float32"), seed)
    hgb, n_events = fit_hgb(train_table, "16", train_table.index.max() + pd.Timedelta(minutes=1), seed)
    bundle = {
        "iforest": iforest,
        "if_threshold": float(if_threshold),
        "if_features": FEATURES,
        "medians": medians,
        "hgb": hgb,
        "hgb_threshold": DEFAULT_THRESHOLD,
        "hgb_features": HGB_FEATURES,
        "n_events": n_events,
        "seed": seed,
        "train_days": train_days,
    }
    path = HYBRID_DIR / "hybrid_final.joblib"
    joblib.dump(bundle, path)
    return path


def main():
    parser = argparse.ArgumentParser(description="Latih dan nilai sistem hybrid")
    parser.add_argument("--seeds", nargs="+", type=int, default=[SEED, 7])
    parser.add_argument("--labels", nargs="+", default=["16", "4"], choices=["16", "4"])
    parser.add_argument("--train-days", type=int, default=TRAIN_DAYS)
    args = parser.parse_args()

    train_table, online_table = load_tables()
    grid = pd.date_range(FOLDS[0][0], FOLDS[-1][1], freq="1min", inclusive="left", name="timestamp")
    online_rows = online_table.loc[online_table["data_ok"], FEATURES]
    in_grace = online_table.loc[online_table["data_ok"], "in_grace"]
    grace_full = online_table["in_grace"].reindex(grid)
    del online_table
    HYBRID_DIR.mkdir(parents=True, exist_ok=True)

    summaries = []
    main_label, main_seed = args.labels[0], args.seeds[0]
    for label_set in args.labels:
        for seed in args.seeds:
            is_main = label_set == main_label and seed == main_seed
            scored = score_folds(train_table, online_rows, label_set, seed, args.train_days, with_quartiles=is_main)
            # Zona grace tidak dinilai, sama seperti train_iforest_dataprep.py. Status per menit tetap mencakupnya.
            evaluated = scored[~in_grace.reindex(scored.index).to_numpy()]
            frame = combine_status(evaluated[["if_score", "if_threshold", "hgb_score", "hgb_threshold", "hgb_aktif"]])
            rule_rows = online_rows.loc[frame.index]
            summaries += [
                summary_row("IF saja", label_set, seed, frame["if_alarm"]),
                summary_row("HGB saja", label_set, seed, frame["hgb_alarm"]),
                summary_row("Aturan run>=15 & oli>=60", label_set, seed, rule_alarm(rule_rows)),
                summary_row("Hybrid Kuning atau Merah", label_set, seed, warning_alarm(frame)),
                summary_row("Hybrid Merah", label_set, seed, red_alarm(frame)),
            ]
            if is_main:
                events = event_table(frame)
                alarms = alarm_log(frame, rule_rows)
                tradeoffs = tradeoff(evaluated)
                status = to_grid(combine_status(scored[["if_score", "if_threshold", "hgb_score", "hgb_threshold", "hgb_aktif"]]), grid)[STATUS_COLUMNS]
                status["in_grace"] = grace_full.to_numpy()

    results = pd.DataFrame(summaries)
    results.to_csv(HYBRID_DIR / "hasil_hybrid.csv", index=False)
    events.to_csv(HYBRID_DIR / "kejadian_hybrid.csv")
    alarms.to_csv(HYBRID_DIR / "alarm_hybrid.csv", index=False)
    tradeoffs.to_csv(HYBRID_DIR / "tradeoff_hybrid.csv", index=False)
    status.to_parquet(HYBRID_DIR / "status_walkforward.parquet")

    pd.set_option("display.width", 220)
    print("\n" + results.to_string(index=False))
    print("\nPer kejadian (skenario utama):")
    print(events[["official", "tertinggi", "delay_peringatan_menit", "delay_merah_menit", "selisih_ke_merah_menit"]].to_string())
    print("\nStatus per menit:", status["status"].value_counts().to_dict())
    print("\nTrade-off (skenario utama):")
    print(tradeoffs.to_string(index=False))
    print(f"\nModel akhir: {train_final(train_table, main_seed, args.train_days)}")


if __name__ == "__main__":
    main()
