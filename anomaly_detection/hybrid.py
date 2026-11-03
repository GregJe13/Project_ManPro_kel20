"""
Aturan status sistem hybrid. Tidak ada training di sini, hanya mengubah skor kedua model menjadi status per menit.

Aturan diperiksa berurutan, dan yang pertama cocok dipakai:
  1. Abu-abu  data kosong, sensor beku, atau kurang dari 8 menit valid dalam 15 menit terakhir
  2. Merah    HGB aktif dan alarm HGB menyala
  3. Merah    HGB belum aktif (cold start) dan alarm IsolationForest sudah berjalan minimal 60 menit
  4. Kuning   alarm IsolationForest menyala
  5. Hijau    selain itu

Alarm menyala jika minimal 2/3 menit valid dalam 15 menit terakhir abnormal, sama seperti evaluation.py.
Semua aturan hanya melihat masa lalu, jadi status tidak berubah saat data baru masuk.
"""
import numpy as np
import pandas as pd

from evaluation import persist_alarm

GREY, GREEN, YELLOW, RED = "Abu-abu", "Hijau", "Kuning", "Merah"
TRIGGER_NONE, TRIGGER_HGB, TRIGGER_IF_LONG, TRIGGER_IF = "-", "HGB", "IF bertahan 60 menit", "IF"

ALARM_WINDOW_MIN = 15
ALARM_SHARE = 2 / 3
ALARM_MIN_VALID = 8 
ESCALATION_MIN = 60  # Ditetapkan sebelum melihat hasil. Jangan disetel dari kejadian yang diuji.
MERGE_MIN = 30  # Sama dengan to_episodes di evaluation.py.

STATUS_COLUMNS = ["status", "pemicu", "if_alarm", "hgb_alarm", "if_score", "if_threshold", "hgb_score", "hgb_threshold", "hgb_aktif"]


def alarm_from_flags(flags):
    return persist_alarm(flags, ALARM_WINDOW_MIN, ALARM_SHARE, ALARM_MIN_VALID)


def escalated(alarm, hold_min=ESCALATION_MIN, merge_min=MERGE_MIN):
    """True jika episode alarm sudah berjalan minimal hold_min menit dan alarm sedang menyala.

    Episode tetap dianggap berjalan selama jeda antar-alarm paling lama merge_min menit.
    """
    times = alarm.index.to_numpy()
    hold, merge = np.timedelta64(hold_min, "m"), np.timedelta64(merge_min, "m")
    result = np.zeros(len(alarm), bool)
    first = last = None
    for i in np.flatnonzero(alarm.to_numpy()):
        t = times[i]
        if first is None or t - last > merge:
            first = t
        last = t
        result[i] = t - first >= hold
    return pd.Series(result, index=alarm.index)


def combine_status(scored):
    """scored: baris valid dengan kolom if_score, if_threshold, hgb_score, hgb_threshold, hgb_aktif."""
    if_flags = scored["if_score"] > scored["if_threshold"]
    hgb_flags = scored["hgb_score"] > scored["hgb_threshold"]
    active = scored["hgb_aktif"].astype(bool)

    if_alarm = alarm_from_flags(if_flags)
    hgb_alarm = alarm_from_flags(hgb_flags) & active
    cold_start_red = escalated(if_alarm) & ~active
    enough_data = if_flags.astype(float).rolling(f"{ALARM_WINDOW_MIN}min").count() >= ALARM_MIN_VALID

    conditions = [~enough_data.to_numpy(), hgb_alarm.to_numpy(), cold_start_red.to_numpy(), if_alarm.to_numpy()]
    frame = scored.copy()
    frame["status"] = np.select(conditions, [GREY, RED, RED, YELLOW], GREEN)
    frame["pemicu"] = np.select(conditions, [TRIGGER_NONE, TRIGGER_HGB, TRIGGER_IF_LONG, TRIGGER_IF], TRIGGER_NONE)
    frame["if_alarm"] = if_alarm
    frame["hgb_alarm"] = hgb_alarm
    return frame[STATUS_COLUMNS]


def to_grid(frame, grid):
    """Lengkapi ke grid menit penuh. Menit tanpa data berstatus Abu-abu, bukan Hijau."""
    full = frame.reindex(grid)
    full["status"] = full["status"].fillna(GREY)
    full["pemicu"] = full["pemicu"].fillna(TRIGGER_NONE)
    for column in ("if_alarm", "hgb_alarm", "hgb_aktif"):
        full[column] = full[column].astype("boolean").fillna(False).astype(bool)
    full.index.name = "timestamp"
    return full


def warning_alarm(frame):
    return frame["status"].isin([YELLOW, RED])


def red_alarm(frame):
    return frame["status"] == RED


def score_table(bundle, table):
    """Beri status pada tabel online baru memakai model akhir (hybrid_final.joblib).

    Jangan dipakai untuk memutar ulang Februari-Agustus 2020: model ini sudah melihat bulan-bulan itu, jadi hasilnya
    terlihat sempurna padahal tidak jujur. Untuk replay dashboard, pakai status_walkforward.parquet.
    """
    rows = table[table["data_ok"]]
    if_input = rows[bundle["if_features"]].fillna(bundle["medians"]).to_numpy("float32")
    scored = pd.DataFrame(
        {
            "if_score": -bundle["iforest"].score_samples(if_input),
            "if_threshold": bundle["if_threshold"],
            "hgb_score": bundle["hgb"].predict_proba(rows[bundle["hgb_features"]].to_numpy("float32"))[:, 1],
            "hgb_threshold": bundle["hgb_threshold"],
            "hgb_aktif": True,
        },
        index=rows.index,
    )
    return to_grid(combine_status(scored), table.index)
