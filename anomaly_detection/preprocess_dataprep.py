"""
Preprocessing MetroPT-3: terjemahan DataPrep.ipynb ke Python, dengan perbaikan hasil review.

Jalankan:  python preprocess_dataprep.py
Hasil:     output/dataprep/ (dua tabel 1 menit dan tiga file CSV pendukung)

Alur
  1. load_raw            muat CSV dan urutkan menurut waktu
  2. clean_values        duplikat, tekanan negatif, sinyal digital, nilai mustahil
  3. frozen_mask         tandai sensor beku (versi latih dan versi real-time)
  4. minute_table        agregasi 1 menit, hitung start dan slope dari data 10 detik, ffill maksimal 3 menit
  5. add_labels          label kegagalan resmi, 16 kejadian, zona grace, segmen, RUL
  6. rolling_features    51 fitur yang hanya melihat ke belakang
  7. find_episodes       episode pola kebocoran, untuk menyaring data latih

Yang berbeda dari notebook
  a. usable_train hanya untuk data latih. Notebook juga membuang baris itu dari penilaian, sehingga PR-AUC menjadi 0,99.
  b. 8 fitur dibuang (lihat EXCLUDED_FEATURES). DV_pressure membuat F4 terlewat oleh XGBoost.
  c. Ada label 16 kejadian (event_id, Event_Flag). Dengan 4 label saja, 8 kejadian paper dianggap normal.
  d. Dua tabel: versi latih dan versi online. Mask beku notebook menandai mundur, jadi tidak tersedia saat real-time.
  e. Episode dengan motor mati tidak dianggap pola kebocoran (arusnya 0,02 A pada 19 Mei dan 31 Juli).
  f. Maintenance F2 memakai 30 Mei, bukan 30 April (lihat MAINTENANCE_GAPS di preprocess_iforest.py).

Tidak ikut diterjemahkan: grafik, Cohen's d, fold LOFO, dan baseline XGBoost, karena bukan bagian preprocessing.
"""
import numpy as np
import pandas as pd

from preprocess_iforest import CSV_PATH, EVENTS, FOLDS, MAINTENANCE_GAPS, OUTPUT_DIR

OUT_DIR = OUTPUT_DIR / "dataprep"
TABLE_TRAIN = OUT_DIR / "dataset_latih_1min.parquet"
TABLE_ONLINE = OUT_DIR / "dataset_online_1min.parquet"

ANALOG = ["TP2", "TP3", "H1", "DV_pressure", "Reservoirs", "Oil_temperature", "Motor_current"]
DIGITAL = ["COMP", "DV_eletric", "Towers", "MPG", "LPS", "Pressure_switch", "Oil_level", "Caudal_impulses"]
SENSORS = ANALOG + DIGITAL
PRESSURE = ["TP2", "TP3", "H1", "DV_pressure", "Reservoirs"]

# Laporan resmi di deskripsi dataset. Tabel II paper memperpanjang F1 sampai 19 April 01:30, dan itu ada di EVENTS.
FAILURES = pd.DataFrame(
    {
        "id": [1, 2, 3, 4],
        "start": pd.to_datetime(["2020-04-18 00:00", "2020-05-29 23:30", "2020-06-05 10:00", "2020-07-15 14:30"]),
        "end": pd.to_datetime(["2020-04-18 23:59", "2020-05-30 06:00", "2020-06-07 14:30", "2020-07-15 19:00"]),
    }
)
MAINTENANCE = [gap_end for _, gap_end in MAINTENANCE_GAPS]

FROZEN_MIN = 30
FFILL_LIMIT = 3
RUL_CAP = 72.0
GRACE_H = 48.0

EP_DUTY = 0.9
EP_MIN = 30
EP_MERGE = 60
EP_MIN_CURRENT = 2.0  # Di bawah ini motor mati, walaupun DV_eletric = 1.
SUSPECT_CATEGORIES = ("Kejadian Tabel II", "Tak tercatat")

WINDOWS = (15, 60, 360)

# DV_pressure: 1,9-2,0 bar pada F1-F3, tetapi 0,000 pada F4.
# Oil_level bernilai 0 pada 121.917 baris Agustus; Caudal_impulses bernilai 0 pada 78.829 baris Maret.
# tp3_minus_res hanya berisi noise karena TP3 dan Reservoirs praktis sama.
EXCLUDED_FEATURES = [
    "DV_pressure", "Oil_level", "Caudal_impulses", "dvp_mean_15", "dvp_mean_60", "dvp_mean_360", "air_ratio_60", "tp3_minus_res",
]

BASE_FEATURES = [c for c in SENSORS if c not in ("Reservoirs", "MPG", "DV_pressure", "Oil_level", "Caudal_impulses")]
ROLLING_NAMES = [
    "duty", "oil_mean", "oil_std", "curr_mean", "tp2_mean", "h1_mean", "tp3_min", "tp3_mean",
    "cycles", "tower_sw", "charge_slope", "discharge_slope",
]
OTHER_FEATURES = ["duty_1440", "duty_dev_60", "load_run_min", "oil_trend_60", "cooling_deficit"]
FEATURES = BASE_FEATURES + [f"{name}_{w}" for w in WINDOWS for name in ROLLING_NAMES] + OTHER_FEATURES

LABELS = ["data_ok", "in_grace", "segment", "failure_id", "Anomaly_Flag", "event_id", "official", "Event_Flag", "RUL_Hours", "RUL_Stage"]

# Angka acuan untuk memeriksa apakah pipeline menghasilkan data yang sama.
EXPECTED = {
    "baris mentah": 1_516_948,
    "baris beku versi latih": 50_716,
    "grid menit": 306_960,
}


# ---------------------------------------------------------------------------
# Data mentah
# ---------------------------------------------------------------------------
def load_raw(csv_path=CSV_PATH):
    dtype = {c: "float32" for c in SENSORS}
    df = pd.read_csv(csv_path, usecols=["timestamp"] + SENSORS, dtype=dtype, parse_dates=["timestamp"])
    return df.sort_values("timestamp").reset_index(drop=True)


def find_gaps(df, min_hours=1):
    step = df["timestamp"].diff().dt.total_seconds()
    gap = step > min_hours * 3600
    return pd.DataFrame(
        {
            "mulai": df["timestamp"].shift(1)[gap].to_numpy(),
            "selesai": df["timestamp"][gap].to_numpy(),
            "durasi_jam": (step[gap] / 3600).round(2).to_numpy(),
        }
    )


def clean_values(raw):
    """Pembersihan yang tidak bergantung pada waktu. Tekanan negatif kecil adalah offset sensor."""
    df = raw.drop_duplicates("timestamp").reset_index(drop=True)
    bad_oil = df["Oil_temperature"].lt(0) | df["Oil_temperature"].gt(150)
    bad_current = df["Motor_current"].lt(0)
    log = {
        "baris_awal": len(raw),
        "duplikat_dibuang": len(raw) - len(df),
        "tekanan_negatif_diclip": int((df[PRESSURE] < 0).sum().sum()),
        "nilai_mustahil": int((bad_oil | bad_current).sum()),
    }
    df[PRESSURE] = df[PRESSURE].clip(lower=0)
    df[DIGITAL] = df[DIGITAL].round().clip(0, 1)
    df.loc[bad_oil, "Oil_temperature"] = np.nan
    df.loc[bad_current, "Motor_current"] = np.nan
    return df, log


def frozen_mask(df, min_minutes=FROZEN_MIN):
    """Sensor analog asli selalu punya noise. Jika ketujuhnya identik selama 30 menit, logger-nya macet.

    Mengembalikan dua penanda:
    - train : seluruh segmen, termasuk 30 menit pertamanya. Baru diketahui belakangan, jadi hanya untuk data latih.
    - online: aktif sejak syarat 30 menit terpenuhi. Hanya ini yang tersedia saat data berjalan.
    """
    same = (df[ANALOG].diff().abs().sum(axis=1) == 0).to_numpy()
    step = df["timestamp"].diff().dt.total_seconds().fillna(10).to_numpy()
    edges = np.diff(np.concatenate([[0], same.astype(np.int8), [0]]))
    train, online = np.zeros(len(df), bool), np.zeros(len(df), bool)
    for first, stop in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)):
        elapsed = np.cumsum(step[first:stop])
        if elapsed[-1] >= min_minutes * 60:
            train[max(first - 1, 0) : stop] = True
            online[first + np.searchsorted(elapsed, min_minutes * 60) : stop] = True
    return train, online


def runs(mask, min_len=1, merge_gap=0):
    """Ubah penanda boolean menjadi pasangan indeks (awal, akhir). Run pendek dibuang, jeda pendek digabung."""
    edges = np.diff(np.concatenate([[0], mask.astype(np.int8), [0]]))
    starts, ends = np.flatnonzero(edges == 1), np.flatnonzero(edges == -1) - 1
    keep = (ends - starts + 1) >= min_len
    merged = []
    for a, b in zip(starts[keep], ends[keep]):
        if merged and a - merged[-1][1] <= merge_gap:
            merged[-1][1] = b
        else:
            merged.append([a, b])
    return merged


def frozen_periods(df, mask):
    rows = [(df["timestamp"].iloc[a], df["timestamp"].iloc[b]) for a, b in runs(mask)]
    table = pd.DataFrame(rows, columns=["mulai", "selesai"])
    table["durasi_jam"] = ((table["selesai"] - table["mulai"]).dt.total_seconds() / 3600).round(1)
    return table


# ---------------------------------------------------------------------------
# Tabel per menit
# ---------------------------------------------------------------------------
def minute_table(df, frozen):
    """Rata-rata per menit. Start dan slope dihitung dulu, karena rata-rata per menit menyembunyikannya.

    Gap sampai 3 menit diisi ffill. Gap yang lebih panjang tetap kosong (data_ok = False).
    """
    d = df.copy()
    d.loc[frozen, SENSORS] = np.nan
    step = d["timestamp"].diff().dt.total_seconds()
    pressure_rate = (d["TP3"].diff() / step).where(step.between(8, 15))
    loaded = d["DV_eletric"] >= 0.5
    events = pd.DataFrame(
        {
            "timestamp": d["timestamp"],
            "load_starts": (d["DV_eletric"].diff() == 1).astype("float32"),
            "tower_switches": (d["Towers"].diff().abs() > 0).astype("float32"),
            "charge_slope": pressure_rate.where(loaded),
            "discharge_slope": pressure_rate.where(~loaded),
        }
    ).set_index("timestamp")

    s = d.set_index("timestamp")
    table = pd.concat(
        [
            s[SENSORS].resample("1min").mean(),
            events[["load_starts", "tower_switches"]].resample("1min").sum(),
            events[["charge_slope", "discharge_slope"]].resample("1min").mean(),
            s["TP3"].resample("1min").count().rename("n_raw"),
        ],
        axis=1,
    )
    has_raw = table["n_raw"] > 0
    table[SENSORS] = table[SENSORS].ffill(limit=FFILL_LIMIT)
    table["data_ok"] = table["TP3"].notna()
    table.loc[~has_raw, ["load_starts", "tower_switches"]] = np.nan
    return table


def grace_windows():
    """Dari akhir kegagalan sampai perbaikan (maksimal 48 jam) mesin mungkin masih rusak."""
    windows = []
    for f in FAILURES.itertuples():
        limit = f.end + pd.Timedelta(hours=GRACE_H)
        later = [m for m in MAINTENANCE if m > f.end]
        windows.append((f.end, min(later[0], limit) if later else limit))
    return pd.DataFrame(windows, columns=["start", "end"])


def add_labels(table):
    """Label hanya untuk evaluasi dan menyaring data latih. Jangan pernah dijadikan fitur."""
    ts = table.index
    n = len(table)

    failure_id = np.zeros(n, np.int8)
    for f in FAILURES.itertuples():
        failure_id[(ts >= f.start) & (ts <= f.end)] = f.id
    table["failure_id"] = failure_id
    table["Anomaly_Flag"] = (failure_id > 0).astype(np.int8)

    table["event_id"] = pd.Series(index=ts, dtype=object)
    table["official"] = False
    for e in EVENTS.itertuples():
        inside = (ts >= e.start) & (ts <= e.end)
        table.loc[inside, "event_id"] = e.id
        table.loc[inside, "official"] = e.official
    table["Event_Flag"] = table["event_id"].notna().astype(np.int8)

    grace = grace_windows()
    in_grace = np.zeros(n, bool)
    for g in grace.itertuples():
        in_grace |= (ts > g.start) & (ts <= g.end)
    table["in_grace"] = in_grace

    # Segmen k = satu siklus "normal lalu failure k". Segmen 5 = sesudah failure terakhir.
    segment = np.full(n, len(FAILURES) + 1, np.int8)
    previous = ts.min() - pd.Timedelta(seconds=1)
    for k, boundary in enumerate(grace["end"], 1):
        segment[(ts > previous) & (ts <= boundary)] = k
        previous = boundary
    table["segment"] = segment

    # RUL menuju awal failure berikutnya, dibatasi 72 jam. Di atas itu mesin dianggap sehat.
    starts = FAILURES["start"].to_numpy()
    t = ts.to_numpy()
    upcoming = np.searchsorted(starts, t, side="right")
    has_next = upcoming < len(starts)
    time_to_failure = np.where(has_next, (starts[np.minimum(upcoming, len(starts) - 1)] - t) / np.timedelta64(1, "h"), np.nan)
    rul = np.where(has_next, np.minimum(time_to_failure, RUL_CAP), np.nan)
    hours_to_end = (ts.max() - ts).total_seconds().to_numpy() / 3600
    rul[~has_next & (hours_to_end >= RUL_CAP)] = RUL_CAP  # Tersensor kanan: yang diketahui hanya RUL minimal 72 jam.
    rul[failure_id > 0] = 0.0
    rul[in_grace] = np.nan
    rul = rul.astype("float32")
    table["RUL_Hours"] = rul
    table["RUL_Stage"] = np.select(
        [failure_id > 0, rul <= 6, rul <= 24, rul > 24],
        ["Failure", "Critical (<6h)", "Warning (6-24h)", "Normal (>24h)"],
        "Tidak diketahui",
    )
    return table


def rolling_features(t):
    """Semua jendela hanya melihat ke belakang. Grid 1 menit reguler, jadi 60 baris lalu benar-benar 60 menit lalu."""
    f = {}
    for w in WINDOWS:
        min_valid = max(3, w // 3)

        def roll(column):
            return t[column].rolling(w, min_periods=min_valid)

        f[f"duty_{w}"] = roll("DV_eletric").mean()
        f[f"oil_mean_{w}"] = roll("Oil_temperature").mean()
        f[f"oil_std_{w}"] = roll("Oil_temperature").std()
        f[f"curr_mean_{w}"] = roll("Motor_current").mean()
        f[f"tp2_mean_{w}"] = roll("TP2").mean()
        f[f"h1_mean_{w}"] = roll("H1").mean()
        f[f"tp3_min_{w}"] = roll("TP3").min()
        f[f"tp3_mean_{w}"] = roll("TP3").mean()
        f[f"cycles_{w}"] = roll("load_starts").sum()
        f[f"tower_sw_{w}"] = roll("tower_switches").sum()
        f[f"charge_slope_{w}"] = t["charge_slope"].rolling(w, min_periods=1).mean()
        f[f"discharge_slope_{w}"] = t["discharge_slope"].rolling(w, min_periods=1).mean()

    duty_day = t["DV_eletric"].rolling(1440, min_periods=360).mean()
    f["duty_1440"] = duty_day
    f["duty_dev_60"] = f["duty_60"] - duty_day  # Deviasi dari baseline harian mesin sendiri.

    run, accumulated = 0, np.zeros(len(t), np.float32)
    for i, loaded in enumerate((t["DV_eletric"] >= 0.9).to_numpy()):
        run = run + 1 if loaded else 0
        accumulated[i] = run
    f["load_run_min"] = pd.Series(accumulated, index=t.index)

    f["oil_trend_60"] = t["Oil_temperature"] - t["Oil_temperature"].shift(60)
    f["cooling_deficit"] = t["Oil_temperature"] - t["Oil_temperature"].rolling(720, min_periods=60).min()
    return pd.DataFrame(f).astype("float32")[[c for c in FEATURES if c not in BASE_FEATURES]]


def build_table(df, frozen):
    minutes = minute_table(df, frozen)
    labeled = add_labels(minutes)
    table = pd.concat([labeled[SENSORS + LABELS], rolling_features(labeled)], axis=1)
    table.index.name = "timestamp"
    return table, minutes


# ---------------------------------------------------------------------------
# Episode pola kebocoran
# ---------------------------------------------------------------------------
def duty_60(table):
    return table["DV_eletric"].rolling(60, min_periods=30).mean()


def episode_category(window):
    failure = int(window["failure_id"].max())
    if failure > 0:
        return f"Failure resmi #{failure}"
    if window["in_grace"].any():
        return "Zona grace"
    if window["Motor_current"].mean() < EP_MIN_CURRENT:
        return "Motor mati"
    return "Kejadian Tabel II" if window["event_id"].notna().any() else "Tak tercatat"


def next_failure(t):
    upcoming = FAILURES[FAILURES["start"] > t]
    if upcoming.empty:
        return "-"
    return f"#{upcoming['id'].iloc[0]} (+{(upcoming['start'].iloc[0] - t).total_seconds() / 86400:.1f} hari)"


def find_episodes(table):
    """Episode duty 60 menit di atas 0,9 selama minimal 30 menit. Ini kandidat anomali, bukan label.

    Jangan disamakan dengan 16 kejadian paper: jumlah episodenya kebetulan juga 16, tetapi hanya 8 yang sama.
    """
    rows = []
    for a, b in runs((duty_60(table) > EP_DUTY).to_numpy(), EP_MIN, EP_MERGE):
        first = max(a - 60, 0)  # Duty 60 menit melihat ke belakang, jadi episode dimulai 60 menit lebih awal.
        window = table.iloc[first : b + 1]
        start = table.index[first]
        rows.append(
            {
                "start": start,
                "end": table.index[b],
                "menit": b - a + 1,
                "first": first,
                "last": b,
                "kategori": episode_category(window),
                "kejadian_tabel_ii": ",".join(window["event_id"].dropna().unique()),
                "oil_mean": round(float(window["Oil_temperature"].mean()), 2),
                "arus_mean": round(float(window["Motor_current"].mean()), 2),
                "DV_pressure_mean": round(float(window["DV_pressure"].mean()), 2),
                "failure_berikutnya": next_failure(start),
            }
        )
    return pd.DataFrame(rows)


def episode_sensitivity(table):
    """Jumlah episode bergantung definisi, jadi angkanya tidak tunggal."""
    duty = duty_60(table).to_numpy()
    rows = [
        {"ambang_duty": limit, "durasi_min_menit": minutes, "jumlah_episode": len(runs(duty > limit, minutes, EP_MERGE))}
        for limit in (0.9, 0.8, 0.7)
        for minutes in (30, 60, 120)
    ]
    return pd.DataFrame(rows).pivot(index="ambang_duty", columns="durasi_min_menit", values="jumlah_episode")


def mark_training_rows(table, episodes):
    """Episode pola kebocoran tidak boleh menjadi contoh normal, karena labelnya tidak jelas.

    usable_train hanya untuk memilih data latih. Saat menilai model, baris ini harus tetap dihitung.
    """
    suspect = np.zeros(len(table), bool)
    for e in episodes[episodes["kategori"].isin(SUSPECT_CATEGORIES)].itertuples():
        suspect[e.first : e.last + 1] = True
    table["suspect"] = suspect
    table["usable_train"] = table["data_ok"] & ~table["in_grace"] & ~table["suspect"]
    return table


# ---------------------------------------------------------------------------
# Pemeriksaan
# ---------------------------------------------------------------------------
def check_no_future_leak(df, online_table, cut=pd.Timestamp("2020-05-15 12:00")):
    """Hitung ulang tabel online tanpa data setelah 'cut'. Fitur sebelum 'cut' harus tetap sama persis."""
    past = df[df["timestamp"] < cut].reset_index(drop=True)
    _, online = frozen_mask(past)
    recomputed, _ = build_table(past, online)
    return bool(np.allclose(recomputed[FEATURES].to_numpy(), online_table.loc[recomputed.index, FEATURES].to_numpy(), equal_nan=True))


def validate(df, frozen_train, train, online):
    fold_starts = [start for start, _ in FOLDS]
    one_minute = pd.Timedelta(minutes=1)
    checks = {
        "grid reguler 1 menit": bool((train.index.to_series().diff().dropna() == one_minute).all()),
        "timestamp unik": bool(train.index.is_unique),
        "4 failure terlabel": sorted(train.loc[train["Anomaly_Flag"] == 1, "failure_id"].unique().tolist()) == [1, 2, 3, 4],
        "16 kejadian terlabel": train["event_id"].nunique() == len(EVENTS),
        "RUL = 0 saat failure": bool((train.loc[train["Anomaly_Flag"] == 1, "RUL_Hours"] == 0).all()),
        "RUL dalam 0..72": bool(train["RUL_Hours"].dropna().between(0, RUL_CAP).all()),
        "tidak ada sensor beku tersisa": int(frozen_mask(df[~frozen_train].reset_index(drop=True))[0].sum()) == 0,
        "8 fitur bermasalah tidak dipakai": not set(EXCLUDED_FEATURES) & set(FEATURES),
        "semua fitur ada di kedua tabel": set(FEATURES) <= set(train.columns) and set(FEATURES) <= set(online.columns),
        "episode tidak melintasi batas putaran": not any(train["suspect"].loc[c - one_minute] and train["suspect"].loc[c] for c in fold_starts),
        "fitur online tidak memakai masa depan": check_no_future_leak(df, online),
    }
    print("\nPemeriksaan:")
    for name, passed in checks.items():
        print(f"  {name:<40} {'OK' if passed else 'GAGAL'}")
    return all(checks.values())


def print_expected(actual):
    print("\nAngka acuan:")
    for name, expected in EXPECTED.items():
        verdict = "OK" if actual[name] == expected else f"BEDA (seharusnya {expected:,})"
        print(f"  {name:<26} {actual[name]:>10,}  {verdict}")


# ---------------------------------------------------------------------------
# Jalankan semua
# ---------------------------------------------------------------------------
def build_tables():
    raw = load_raw()
    gaps = find_gaps(raw)
    print(f"{len(raw):,} baris | {raw['timestamp'].min()} s/d {raw['timestamp'].max()}")
    print(f"Gap > 1 jam: {len(gaps)} | > 6 jam: {(gaps['durasi_jam'] > 6).sum()} | > 24 jam: {(gaps['durasi_jam'] > 24).sum()} | total {gaps['durasi_jam'].sum():.0f} jam")

    df, log = clean_values(raw)
    del raw
    print(f"Cleaning: {log}")

    frozen_train, frozen_online = frozen_mask(df)
    periods = frozen_periods(df, frozen_train)
    print(f"\nSensor beku: {len(periods)} periode, {periods['durasi_jam'].sum():.0f} jam")
    print(f"  {int(frozen_train.sum()):,} baris versi latih | {int(frozen_online.sum()):,} baris versi online")
    print(periods.to_string(index=False))

    train, minutes = build_table(df, frozen_train)
    has_raw = minutes["n_raw"] > 0
    print(f"\nGrid 1 menit: {len(train):,} | berdata asli: {int(has_raw.sum()):,} | diisi ffill: {int((train['data_ok'] & ~has_raw).sum()):,} | gap atau beku: {int((~train['data_ok']).sum()):,}")
    del minutes

    online, _ = build_table(df, frozen_online)

    episodes = find_episodes(train)
    train = mark_training_rows(train, episodes)
    print(f"\nEpisode pola kebocoran: {len(episodes)}")
    print(episodes.drop(columns=["first", "last"]).to_string(index=False))
    print("\nJumlah episode per definisi:")
    print(episode_sensitivity(train).to_string())

    print(f"\n{len(FEATURES)} fitur | menit usable_train: {int(train['usable_train'].sum()):,} dari {len(train):,}")
    print(f"  dikeluarkan: gap atau beku {int((~train['data_ok']).sum()):,}, grace {int(train['in_grace'].sum()):,}, episode {int(train['suspect'].sum()):,}")
    print(f"  menit berlabel failure (usable_train): {train.loc[train['usable_train'], 'Anomaly_Flag'].mean() * 100:.2f}%")

    print_expected({"baris mentah": log["baris_awal"], "baris beku versi latih": int(frozen_train.sum()), "grid menit": len(train)})
    passed = validate(df, frozen_train, train, online)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    train.to_parquet(TABLE_TRAIN)
    online.to_parquet(TABLE_ONLINE)
    episodes.drop(columns=["first", "last"]).to_csv(OUT_DIR / "episode_pola_air_leak.csv", index=False)
    periods.to_csv(OUT_DIR / "periode_sensor_beku.csv", index=False)
    EVENTS.to_csv(OUT_DIR / "kejadian_expert_16.csv", index=False)
    print(f"\n{'SEMUA LULUS' if passed else 'ADA YANG GAGAL'} | hasil tersimpan di {OUT_DIR}")
    return train, online


def load_tables():
    """Baca tabel yang sudah tersimpan. Jika belum ada, jalankan preprocessing."""
    if TABLE_TRAIN.exists() and TABLE_ONLINE.exists():
        return pd.read_parquet(TABLE_TRAIN), pd.read_parquet(TABLE_ONLINE)
    return build_tables()


if __name__ == "__main__":
    build_tables()
