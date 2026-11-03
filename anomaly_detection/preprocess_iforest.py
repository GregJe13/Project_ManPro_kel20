"""
Preprocessing MetroPT-3 untuk IsolationForest.

Jalankan:  python preprocess_iforest.py
Hasil:     output/fitur_latih.parquet dan output/fitur_online.parquet

Tahap A (sekali jalan, dipakai juga oleh preprocess_supervised.py)
  1. load_raw             muat CSV, ambil 6 sensor, urutkan menurut waktu
  2. frozen_masks         tandai logger macet
  3. minute_table         agregasi per menit tanpa mengisi gap
  4. rolling_features     hitung 9 fitur yang hanya melihat masa lalu
  5. build_feature_table  tambahkan kolom status dan label kejadian

Tahap B (per putaran latih-uji, khusus IsolationForest)
  6. train_rows_for_fold  bagi data menurut waktu
  7. event_masks          keluarkan kejadian yang sudah diketahui
  8. prepare_fold_iforest isi NaN sisa dengan median data latih
  9. self_clean           buang baris paling aneh dari data latih

Langkah 10 (threshold dari skor latih) ada di model_iforest.py.
"""
import os
from pathlib import Path
from typing import NamedTuple

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

N_JOBS = 4

# Mematikan peringatan joblib di Windows 11 saat menghitung jumlah core.
# Nilainya harus lebih kecil dari jumlah core agar diikuti joblib.
os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(N_JOBS))

BASE_DIR = Path(__file__).resolve().parent
CSV_PATH = BASE_DIR.parent / "MetroPT3(AirCompressor).csv"
OUTPUT_DIR = BASE_DIR / "output"
TABLE_TRAIN = OUTPUT_DIR / "fitur_latih.parquet"
TABLE_ONLINE = OUTPUT_DIR / "fitur_online.parquet"

# TP3 hanya dipakai untuk mendeteksi logger macet, bukan sebagai fitur model.
SENSORS = ["DV_eletric", "Oil_temperature", "Motor_current", "TP2", "H1", "TP3"]
FEATURES = ["duty_10", "duty_60", "run_min", "cyc_60", "oil_10", "oil_rise_60", "mc_30", "tp2_30", "h1_30"]

SEED = 42

# Tabel II Davari dkk. (2021). Entri yang berdempetan digabung menjadi 16 kejadian.
# official=True berarti termasuk 4 kegagalan di deskripsi dataset.
EVENTS = pd.DataFrame(
    [
        ("E01", "2020-04-12 11:50", "2020-04-12 23:30", False),
        ("E02_F1", "2020-04-18 00:00", "2020-04-19 01:30", True),
        ("E03", "2020-04-29 03:20", "2020-04-29 04:00", False),
        ("E04", "2020-04-29 22:00", "2020-04-29 22:20", False),
        ("E05", "2020-05-13 14:00", "2020-05-13 23:59", False),
        ("E06", "2020-05-18 05:00", "2020-05-18 05:30", False),
        ("E07", "2020-05-19 10:10", "2020-05-19 11:00", False),
        ("E08", "2020-05-19 22:10", "2020-05-20 20:00", False),
        ("E09", "2020-05-23 09:50", "2020-05-23 10:10", False),
        ("E10_F2", "2020-05-29 23:30", "2020-05-30 06:00", True),
        ("E11", "2020-06-01 15:00", "2020-06-01 15:40", False),
        ("E12", "2020-06-03 10:00", "2020-06-03 11:00", False),
        ("E13_F3", "2020-06-05 10:00", "2020-06-07 14:30", True),
        ("E14", "2020-07-08 17:30", "2020-07-08 19:00", False),
        ("E15_F4", "2020-07-15 14:30", "2020-07-15 19:00", True),
        ("E16", "2020-07-17 04:30", "2020-07-17 05:30", False),
    ],
    columns=["id", "start", "end", "official"],
)
EVENTS["start"] = pd.to_datetime(EVENTS["start"])
EVENTS["end"] = pd.to_datetime(EVENTS["end"])

# Waktu label dibulatkan manusia (contoh: F3 mulai berbeban 09:48, labelnya 10:00), jadi batasnya diberi margin.
MARGIN = pd.Timedelta(hours=2)

# Jeda dari akhir kegagalan sampai perbaikan: mesin belum tentu normal.
# Jeda pertama memakai asumsi 30 Mei, karena deskripsi dataset menulis 30 April pada baris kegagalan 29-30 Mei.
MAINTENANCE_GAPS = [
    (pd.Timestamp("2020-05-30 06:00"), pd.Timestamp("2020-05-30 12:00")),
    (pd.Timestamp("2020-06-07 14:30"), pd.Timestamp("2020-06-08 16:00")),
    (pd.Timestamp("2020-07-15 19:00"), pd.Timestamp("2020-07-16 00:00")),
]

# Lima putaran: tiap bulan diuji oleh model yang belum pernah melihat bulan itu.
FOLDS = [
    (pd.Timestamp("2020-04-01"), pd.Timestamp("2020-05-01")),
    (pd.Timestamp("2020-05-01"), pd.Timestamp("2020-06-01")),
    (pd.Timestamp("2020-06-01"), pd.Timestamp("2020-07-01")),
    (pd.Timestamp("2020-07-01"), pd.Timestamp("2020-08-01")),
    (pd.Timestamp("2020-08-01"), pd.Timestamp("2020-09-01")),
]
TRAIN_DAYS = 60  # Parameter: bukti pemilihan 60 hari lemah, lihat catatan di model_iforest.py.
BLOCK_BUFFER = pd.Timedelta(days=3)

# Angka acuan untuk memeriksa apakah pipeline menghasilkan data yang sama.
EXPECTED = {
    "baris mentah": 1_516_948,
    "baris frozen_train": 50_868,
    "baris frozen_online": 49_582,
    "grid menit": 306_960,
    "baris valid versi latih": 236_943,
    "baris valid versi online": 237_191,
}


# ---------------------------------------------------------------------------
# Tahap A
# ---------------------------------------------------------------------------
def load_raw(csv_path=CSV_PATH):
    """Langkah 1. Jendela waktu pada langkah berikutnya mengandalkan urutan waktu yang benar."""
    df = pd.read_csv(csv_path, usecols=["timestamp"] + SENSORS, parse_dates=["timestamp"])
    return df.sort_values("timestamp").reset_index(drop=True)


def frozen_masks(df, window_min=30):
    """Langkah 2. Logger macet: baris tetap tertulis, tetapi nilai analog tidak berubah.

    Mengembalikan dua penanda:
    - frozen_online: aktif sejak syarat terpenuhi. Hanya ini yang tersedia saat data berjalan.
    - frozen_train : seluruh segmen, termasuk 30 menit pertamanya. Hanya boleh untuk data latih.
    """
    x = df.set_index("timestamp")
    window = f"{window_min}min"

    def value_range(column):
        rolling = x[column].rolling(window)
        return rolling.max() - rolling.min()

    frozen_online = (
        (value_range("TP3") <= 0.002)
        & (value_range("Oil_temperature") <= 0)
        & (value_range("Motor_current") <= 0)
        & (x["TP3"].rolling(window).count() >= 60)
    ).to_numpy()

    frozen_train = frozen_online.copy()
    flagged = np.where(frozen_online)[0]
    if len(flagged):
        times = df["timestamp"].to_numpy()
        reach_back = np.timedelta64(window_min, "m")
        segment_start = previous = flagged[0]
        segments = []
        for i in flagged[1:]:
            if times[i] - times[previous] > reach_back:
                segments.append((segment_start, previous))
                segment_start = i
            previous = i
        segments.append((segment_start, previous))
        for first, last in segments:
            frozen_train[np.searchsorted(times, times[first] - reach_back) : last + 1] = True
    return frozen_train, frozen_online


def minute_grid(df):
    """Grid menit penuh, supaya '60 menit lalu' benar-benar 60 menit lalu, bukan 60 baris lalu."""
    grid = pd.date_range(df["timestamp"].min().floor("min"), df["timestamp"].max().floor("min"), freq="1min")
    return grid.rename("timestamp")


def minute_table(df, grid):
    """Langkah 3. Agregasi per menit TANPA imputasi: menit kosong tetap NaN.

    Ffill tidak dipakai karena mengarang 17,7% menit, termasuk kondisi rusak yang diulang setelah F3.
    Start dihitung dari data 10 detik, karena rata-rata per menit menyembunyikan perubahan nyala-mati.
    Syarat jarak 8-15 detik mencegah perubahan di seberang gap terhitung sebagai start.
    """
    d = df.sort_values("timestamp")
    consecutive = d["timestamp"].diff().dt.total_seconds().between(8, 15).to_numpy()
    d = d.set_index("timestamp")
    load = d["DV_eletric"]
    is_start = pd.Series(((load.diff() == 1).to_numpy() & consecutive).astype(float), index=d.index)

    table = pd.DataFrame(
        {
            "n": load.resample("1min").count(),
            "duty": load.resample("1min").mean(),
            "Oil": d["Oil_temperature"].resample("1min").mean(),
            "MC": d["Motor_current"].resample("1min").mean(),
            "TP2": d["TP2"].resample("1min").mean(),
            "H1": d["H1"].resample("1min").mean(),
            "start": is_start.resample("1min").sum(),
        }
    ).reindex(grid)

    # Rata-rata dari 1-2 sampel (seharusnya sekitar 6) tidak mewakili menit itu.
    too_few = table["n"].fillna(0) < 3
    table.loc[too_few, table.columns.difference(["n"])] = np.nan
    return table


def run_length(duty, pause_tolerance=3, cap=360):
    """Menit berbeban beruntun. Jeda singkat ditoleransi; gap data mereset ke 0 karena kondisinya tidak diketahui."""
    loaded = (duty >= 0.5).to_numpy()
    valid = duty.notna().to_numpy()
    out = np.zeros(len(duty))
    current = pause = 0
    for i in range(len(duty)):
        if valid[i] and loaded[i]:
            current += 1 + pause
            pause = 0
        elif current > 0 and valid[i] and pause < pause_tolerance:
            pause += 1
        else:
            current = pause = 0
        out[i] = current
    return np.minimum(out, cap)


def rolling_features(minutes):
    """Langkah 4. IsolationForest menilai tiap baris tanpa ingatan, jadi riwayat harus dimasukkan ke dalam baris.

    Semua jendela hanya melihat ke belakang. min_periods mencegah rata-rata dari terlalu sedikit menit di sekitar gap.
    """

    def roll(column, minutes_back, min_valid):
        return minutes[column].rolling(f"{minutes_back}min", min_periods=min_valid)

    f = pd.DataFrame(index=minutes.index)
    f["duty_10"] = roll("duty", 10, 5).mean()
    f["duty_60"] = roll("duty", 60, 30).mean()
    f["run_min"] = run_length(minutes["duty"])
    f["cyc_60"] = roll("start", 60, 30).sum()
    f["oil_10"] = roll("Oil", 10, 5).mean()
    f["oil_rise_60"] = f["oil_10"] - f["oil_10"].shift(60)  # Kenaikan relatif, karena suhu dasar ikut musim.
    f["mc_30"] = roll("MC", 30, 15).mean()
    f["tp2_30"] = roll("TP2", 30, 15).mean()
    f["h1_30"] = roll("H1", 30, 15).mean()
    return f[FEATURES]


def add_event_labels(table):
    """Label hanya untuk evaluasi dan menyaring data latih. Jangan pernah dijadikan fitur."""
    table["event_id"] = pd.Series(index=table.index, dtype=object)
    table["official"] = False
    for e in EVENTS.itertuples():
        inside = (table.index >= e.start) & (table.index <= e.end)
        table.loc[inside, "event_id"] = e.id
        table.loc[inside, "official"] = e.official
    return table


def build_feature_table(df, exclude_mask, grid):
    """Langkah 3-5. Satu baris per menit, berisi 9 fitur, status (valid, gap, macet), dan label kejadian."""
    minutes = minute_table(df[~exclude_mask], grid)
    table = rolling_features(minutes).astype(np.float32)
    valid = minutes["duty"].notna() & table["duty_60"].notna() & table["oil_10"].notna()
    table.loc[~valid, FEATURES] = np.nan

    # Gap berarti "tidak tahu", jadi dashboard tidak boleh menampilkannya sebagai hijau.
    frozen_minutes = df.loc[exclude_mask, "timestamp"].dt.floor("min").unique()
    status = np.full(len(table), "gap", dtype=object)
    status[table.index.isin(frozen_minutes)] = "macet"
    status[valid.to_numpy()] = "valid"
    table["status"] = status
    return add_event_labels(table)


def build_tables():
    """Jalankan Tahap A dua kali: versi latih tanpa frozen_train, versi online tanpa frozen_online."""
    df = load_raw()
    frozen_train, frozen_online = frozen_masks(df)
    grid = minute_grid(df)
    train_table = build_feature_table(df, frozen_train, grid)
    online_table = build_feature_table(df, frozen_online, grid)

    OUTPUT_DIR.mkdir(exist_ok=True)
    train_table.to_parquet(TABLE_TRAIN)
    online_table.to_parquet(TABLE_ONLINE)

    print_checks(
        {
            "baris mentah": len(df),
            "baris frozen_train": int(frozen_train.sum()),
            "baris frozen_online": int(frozen_online.sum()),
            "grid menit": len(grid),
            "baris valid versi latih": int((train_table["status"] == "valid").sum()),
            "baris valid versi online": int((online_table["status"] == "valid").sum()),
        }
    )
    check_no_future_leak(df, online_table)
    return train_table, online_table


def load_tables():
    """Baca tabel yang sudah tersimpan. Jika belum ada, jalankan Tahap A."""
    if TABLE_TRAIN.exists() and TABLE_ONLINE.exists():
        return pd.read_parquet(TABLE_TRAIN), pd.read_parquet(TABLE_ONLINE)
    return build_tables()


def valid_rows(table):
    return table[table["status"] == "valid"]


def print_checks(actual):
    print("\nCek hasil Tahap A:")
    for name, expected in EXPECTED.items():
        verdict = "OK" if actual[name] == expected else f"BEDA (seharusnya {expected:,})"
        print(f"  {name:<26} {actual[name]:>10,}  {verdict}")


def check_no_future_leak(df, online_table, cut=pd.Timestamp("2020-05-15 12:00")):
    """Hitung ulang fitur tanpa data setelah 'cut'. Fitur sebelum 'cut' harus tetap sama persis."""
    past = df[df["timestamp"] < cut]
    _, frozen_online = frozen_masks(past)
    grid = minute_grid(past)
    grid = grid[grid < cut]
    recomputed = build_feature_table(past, frozen_online, grid)
    same = np.allclose(
        recomputed[FEATURES].to_numpy(), online_table.loc[grid, FEATURES].to_numpy(), equal_nan=True
    )
    print(f"  {'uji bocoran masa depan':<26} {'':>10}  {'OK' if same else 'GAGAL: fitur memakai data masa depan'}")


# ---------------------------------------------------------------------------
# Bagian yang dipakai bersama pada Tahap B
# ---------------------------------------------------------------------------
def select_events(label_set):
    """'16' = semua kejadian paper, '4' = hanya kegagalan resmi di deskripsi dataset."""
    return EVENTS if str(label_set) == "16" else EVENTS[EVENTS["official"]]


def train_rows_for_fold(train_valid, start, end, scheme="walk", train_days=None):
    """Langkah 6. Split acak dilarang: menit berdekatan hampir identik dan model akan melihat masa depan.

    walk : hanya data sebelum bulan uji, seperti kondisi nyata.
    block: semua bulan lain termasuk masa depan, dengan jeda 3 hari. Hanya untuk melihat kondisi terbaik.
    """
    index = train_valid.index
    if scheme == "walk":
        rows = train_valid[index < start]
        return rows if train_days is None else rows[rows.index >= start - pd.Timedelta(days=train_days)]
    return train_valid[(index < start - BLOCK_BUFFER) | (index >= end + BLOCK_BUFFER)]


def test_rows_for_fold(online_valid, start, end):
    return online_valid[(online_valid.index >= start) & (online_valid.index < end)]


def event_masks(index, events, cut=None, pre_hours=0):
    """Langkah 7. Mengembalikan y (1 di dalam kejadian), usable (baris boleh dipakai), dan jumlah kejadian.

    cut: hanya kejadian yang selesai sebelum waktu ini yang dianggap diketahui.
    Zona abu-abu (margin sekitar kejadian dan jeda sampai perbaikan) tidak dipakai sebagai data normal.
    """
    y = np.zeros(len(index), dtype=np.int8)
    neutral = np.zeros(len(index), dtype=bool)
    pre = pd.Timedelta(hours=pre_hours)
    n_events = 0
    for e in events.itertuples():
        if cut is not None and e.end >= cut:
            continue
        inside = (index >= e.start - pre) & (index <= e.end)
        y[inside] = 1
        neutral |= (index >= e.start - pre - MARGIN) & (index <= e.end + MARGIN)
        n_events += int(inside.any())
    for gap_start, gap_end in MAINTENANCE_GAPS:
        if cut is None or gap_end < cut:
            neutral |= (index >= gap_start) & (index <= gap_end)
    return y, ~neutral | (y == 1), n_events


def neutral_mask(index):
    """Zona abu-abu dari semua kejadian. Dipakai saat menilai, bukan saat melatih."""
    mask = np.zeros(len(index), dtype=bool)
    for e in EVENTS.itertuples():
        mask |= (index >= e.start - MARGIN) & (index <= e.end + MARGIN)
    for gap_start, gap_end in MAINTENANCE_GAPS:
        mask |= (index >= gap_start) & (index <= gap_end)
    return mask


def boxplot_threshold(scores):
    """Q3 + 3*IQR, aturan yang dipakai Davari dkk. (2021)."""
    q1, q3 = np.quantile(scores, [0.25, 0.75])
    return q3 + 3 * (q3 - q1)


# ---------------------------------------------------------------------------
# Tahap B khusus IsolationForest
# ---------------------------------------------------------------------------
class IForestFold(NamedTuple):
    X_train: np.ndarray
    X_test: np.ndarray
    test_index: pd.DatetimeIndex
    medians: pd.Series


def normal_rows(rows, label_set, cut):
    """Langkah 7. Model belajar kondisi normal; jika data bocor ikut dipelajari, bocor dianggap normal."""
    y, usable, _ = event_masks(rows.index, select_events(label_set), cut=cut)
    return rows.loc[usable & (y == 0), FEATURES]


def prepare_fold_iforest(train_valid, online_valid, start, end, label_set="16", scheme="walk", train_days=TRAIN_DAYS):
    """Langkah 6-8 untuk satu putaran."""
    days = train_days if scheme == "walk" else None
    rows = train_rows_for_fold(train_valid, start, end, scheme, days)
    normal = normal_rows(rows, label_set, cut=start if scheme == "walk" else None)
    test = test_rows_for_fold(online_valid, start, end)

    # Langkah 8. Berbeda dari ffill: tidak menciptakan menit baru, hanya melengkapi fitur yang jendelanya kurang.
    # Median diambil dari data latih agar tidak memakai informasi data uji.
    medians = normal.median()
    return IForestFold(
        normal.fillna(medians).to_numpy(np.float32),
        test[FEATURES].fillna(medians).to_numpy(np.float32),
        test.index,
        medians,
    )


def prepare_final_iforest(train_valid, train_days=TRAIN_DAYS):
    """Data latih untuk model akhir: 60 hari terakhir, dengan semua kejadian dikeluarkan."""
    cut = train_valid.index.max() + pd.Timedelta(minutes=1)
    rows = train_rows_for_fold(train_valid, cut, cut, "walk", train_days)
    normal = normal_rows(rows, "16", cut=cut)
    medians = normal.median()
    return normal.fillna(medians).to_numpy(np.float32), medians


def self_clean(X_train, seed=SEED):
    """Langkah 9. Label hanya mencakup April-Juli, padahal Maret punya 8 episode panas tanpa label.

    Latih model sementara, lalu buang baris yang skornya melewati Q3 + 3*IQR.
    """
    forest = IsolationForest(n_estimators=100, max_samples=4096, random_state=seed, n_jobs=N_JOBS).fit(X_train)
    scores = -forest.score_samples(X_train)
    return X_train[scores <= boxplot_threshold(scores)]


if __name__ == "__main__":
    train_table, online_table = build_tables()
    train_valid, online_valid = valid_rows(train_table), valid_rows(online_table)

    print("\nData per putaran (16 label, walk-forward):")
    for start, end in FOLDS:
        fold = prepare_fold_iforest(train_valid, online_valid, start, end)
        print(f"  uji {start:%b}: {len(fold.X_train):>7,} baris normal untuk latih, {len(fold.X_test):>6,} baris uji")
    print(f"\nTabel tersimpan di {OUTPUT_DIR}")
