"""
Penilaian yang dipakai bersama oleh model_iforest.py dan model_supervised.py.

Kedua metode harus dinilai dengan kode yang persis sama agar perbandingannya adil:
filter alarm yang sama, penggabungan episode yang sama, dan penilaian per kejadian yang sama.
"""
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from preprocess_iforest import EVENTS, FOLDS, MAINTENANCE_GAPS, MARGIN, neutral_mask

TEST_START, TEST_END = FOLDS[0][0], FOLDS[-1][1]
TEST_WEEKS = (TEST_END - TEST_START).total_seconds() / (7 * 86400)
APRIL_END = pd.Timestamp("2020-05-01")  # Sebelum tanggal ini, supervised walk-forward belum punya label.

SUMMARY_COLUMNS = ["metode", "skema", "label", "kejadian", "resmi", "april", "mei_agu", "alarm_palsu", "delay_median_menit", "ap_baris"]


def persist_alarm(flags, window_min=15, need=2 / 3, min_valid=8):
    """Alarm menyala jika minimal 2/3 menit valid dalam 15 menit terakhir abnormal, agar lonjakan sesaat tidak jadi alarm."""
    share = flags.astype(float).rolling(f"{window_min}min", min_periods=min_valid).mean()
    return share >= need


def to_episodes(alarm, merge_min=30):
    """Gabungkan menit alarm yang berjarak paling lama 30 menit menjadi satu episode (mulai, selesai)."""
    times = alarm.index[alarm.to_numpy()]
    if len(times) == 0:
        return []
    episodes = []
    first = last = times[0]
    for t in times[1:]:
        if t - last > pd.Timedelta(minutes=merge_min):
            episodes.append((first, last))
            first = t
        last = t
    episodes.append((first, last))
    return episodes


def evaluate(episodes):
    """Nilai per kejadian, bukan per baris, karena F3 saja mencakup lebih dari separuh baris kegagalan.

    Selalu dinilai terhadap 16 kejadian, walaupun model hanya dilatih dengan 4 label.
    Kejadian terdeteksi jika ada alarm antara 2 jam sebelum mulai dan akhir kejadian.
    Alarm palsu adalah episode yang tidak menyentuh kejadian, margin, atau jeda sampai perbaikan.
    """
    episodes = [(a, b) for a, b in episodes if b >= TEST_START]

    first_alarm = {}
    for e in EVENTS.itertuples():
        low, high = e.start - MARGIN, e.end
        overlaps = [max(a, low) for a, b in episodes if a <= high and b >= low]
        first_alarm[e.id] = min(overlaps) if overlaps else None

    def touches_known_period(a, b):
        near_event = any(a <= e.end + MARGIN and b >= e.start - MARGIN for e in EVENTS.itertuples())
        near_gap = any(a <= gap_end and b >= gap_start for gap_start, gap_end in MAINTENANCE_GAPS)
        return near_event or near_gap

    false_alarms = [(a, b) for a, b in episodes if not touches_known_period(a, b)]
    starts = EVENTS.set_index("id")["start"]
    delays = [(t - starts[k]).total_seconds() / 60 for k, t in first_alarm.items() if t is not None]

    hits = sum(t is not None for t in first_alarm.values())
    recall = hits / len(EVENTS)
    precision = (len(episodes) - len(false_alarms)) / len(episodes) if episodes else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "first_alarm": first_alarm,
        "false_alarms": false_alarms,
        "hits": hits,
        "recall": recall,
        "precision": precision,
        "f1": f1,
        "median_delay": float(np.median(delays)) if delays else np.nan,
    }


def average_precision(scores):
    """Seberapa konsisten menit rusak mendapat skor teratas (0-1). Zona abu-abu tidak dihitung."""
    y = np.zeros(len(scores), dtype=int)
    for e in EVENTS.itertuples():
        y[(scores.index >= e.start) & (scores.index <= e.end)] = 1
    used = ~neutral_mask(scores.index) | (y == 1)
    return average_precision_score(y[used], scores.to_numpy()[used])


def summarize(method, scheme, label_set, flags, scores):
    """Satu baris hasil. flags dan scores adalah daftar Series per putaran."""
    flags = pd.concat(flags).sort_index()
    scores = pd.concat(scores).sort_index()
    result = evaluate(to_episodes(persist_alarm(flags)))

    first_alarm = result["first_alarm"]
    april_ids = EVENTS.loc[EVENTS["start"] < APRIL_END, "id"]
    later_ids = EVENTS.loc[EVENTS["start"] >= APRIL_END, "id"]
    official_ids = EVENTS.loc[EVENTS["official"], "id"]

    def count_hits(ids):
        return sum(first_alarm[i] is not None for i in ids)

    false_april = sum(a < APRIL_END for a, _ in result["false_alarms"])
    return {
        "metode": method,
        "skema": scheme,
        "label": label_set,
        "kejadian": f"{result['hits']}/{len(EVENTS)}",
        "resmi": f"{count_hits(official_ids)}/{len(official_ids)}",
        "april": f"{count_hits(april_ids)}/{len(april_ids)}",
        "mei_agu": f"{count_hits(later_ids)}/{len(later_ids)}",
        "alarm_palsu": len(result["false_alarms"]),
        "alarm_palsu_april": false_april,
        "alarm_palsu_mei_agu": len(result["false_alarms"]) - false_april,
        "alarm_palsu_per_minggu": round(len(result["false_alarms"]) / TEST_WEEKS, 2),
        "delay_median_menit": result["median_delay"],
        "precision": round(result["precision"], 3),
        "recall": round(result["recall"], 3),
        "skor_f1": round(result["f1"], 3),
        "ap_baris": round(average_precision(scores), 3),
        "meleset": ",".join(k for k, t in first_alarm.items() if t is None),
    }
