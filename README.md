# Anomaly Detection on MetroPT-3

Anomaly detection for the air compressor (APU) of a Metro do Porto train, using the
[MetroPT-3 dataset](https://archive.ics.uci.edu/dataset/791/metropt+3+dataset).
The main model is a **hybrid system**:

- **IsolationForest** (unsupervised, 51 features) raises the early warning (Kuning / yellow).
- **HistGradientBoosting** (supervised, 9 features) raises the alarm (Merah / red) after it has seen past failures.
- In April, before any failure has been seen, an IsolationForest alarm that lasts 60 minutes turns red instead (the cold-start rule).

All code is in [`anomaly_detection/`](anomaly_detection/).

---

## 1. Get the dataset (not included in this repo)

The raw CSV is about **218 MB**. GitHub rejects files over 100 MB, so the CSV is not in the repo
(`.gitignore` excludes every `*.csv`). Download it yourself:

1. Go to <https://archive.ics.uci.edu/dataset/791/metropt+3+dataset> and click **Download**.
2. Extract the zip.
3. Put **`MetroPT3(AirCompressor).csv`** in the **repo root**, next to this README. Do not rename it.

```
Project-ManPro-kel-20/
├── MetroPT3(AirCompressor).csv     <- put the dataset here
├── README.md
└── anomaly_detection/
    ├── preprocess_dataprep.py
    ├── train_hybrid.py
    ├── ...
    └── output/
```

The code finds the file through this line in `anomaly_detection/preprocess_iforest.py`:

```python
CSV_PATH = BASE_DIR.parent / "MetroPT3(AirCompressor).csv"
```

If you want to keep the CSV somewhere else, change this line. Don't commit the CSV.

> **Check:** the file should have **1,516,948 rows**. `preprocess_dataprep.py` prints
> `baris mentah 1,516,948 OK` when the file is complete. Don't open and save it in Excel:
> Excel cuts it off at about 1,048,576 rows (26 June 2020) and you lose failure F4.

### Do I need the CSV?

The preprocessed 1-minute tables are already in the repo, so you can train the hybrid **without** the CSV.

| What you want to do | Need the CSV? |
|---|---|
| Train and evaluate the hybrid (`train_hybrid.py`) | No. It reads `output/dataprep/*.parquet` (already in the repo) |
| Run the unit tests (`test_hybrid.py`) | No. They use made-up data |
| Make the charts (`visualize_hybrid.py`) | No, but run `train_hybrid.py` first (see step 3) |
| Rebuild the preprocessing (`preprocess_dataprep.py`) | **Yes** |
| Run the older pipeline (`preprocess_iforest.py`, `model_iforest.py`, `model_supervised.py`) | **Yes** |

---

## 2. Setup

Tested with Python 3.12.

```bash
python -m venv .venv
# Windows (PowerShell):  .venv\Scripts\Activate.ps1
# macOS / Linux:         source .venv/bin/activate

pip install numpy pandas scikit-learn joblib matplotlib pyarrow
```

Versions used during development: numpy 1.26.4, pandas 2.2.3, scikit-learn 1.6.0, joblib 1.4.2,
matplotlib 3.10.0, pyarrow 20.0.0.

---

## 3. Run the hybrid

Run everything from inside `anomaly_detection/`. The scripts import each other
(`evaluation.py`, `preprocess_iforest.py`, `model_iforest.py`, ...), so keep all `.py` files in that one folder.

```bash
cd anomaly_detection

python preprocess_dataprep.py   # optional: only if you want to rebuild output/dataprep/ from the CSV
python train_hybrid.py          # train + walk-forward evaluation, writes output/hybrid/
python test_hybrid.py           # 13 tests of the status rules (should end with "OK")
python visualize_hybrid.py      # 8 PNG charts for the report -> output/hybrid/grafik/
```

**After a fresh clone, run `train_hybrid.py` before `visualize_hybrid.py`.** The charts read
`hasil_hybrid.csv`, `kejadian_hybrid.csv`, `alarm_hybrid.csv` and `tradeoff_hybrid.csv`.
These are not in the repo (they match `*.csv` in `.gitignore`), so `train_hybrid.py` has to create them first.

### Options for `train_hybrid.py`

```bash
python train_hybrid.py --seeds 42 7 --labels 16 4 --train-days 60   # these are the defaults
```

| Option | Meaning |
|---|---|
| `--seeds` | Random seeds. Results are reported per seed |
| `--labels` | `16` = the 16 events from Davari et al. (2021) Table II (main labels), `4` = only the 4 official failures |
| `--train-days` | Training window before each test month (default 60 days) |

The first seed and the first label set make up the main scenario. The per-event files, the alarm log,
the per-minute status and the final model all come from that scenario.

### How it is evaluated

Walk-forward: each month from April to August 2020 is scored by a model trained only on data from before that month.
The thresholds were fixed before looking at the results: IsolationForest Q3 + 3×IQR on the training scores,
HGB 0.5, cold start 60 minutes. An alarm fires when at least 2/3 of the valid minutes in the last 15 minutes are abnormal.

---

## 4. Output

Everything goes to `anomaly_detection/output/hybrid/`:

| File | Content | In repo? |
|---|---|---|
| `status_walkforward.parquet` | Status for every minute, Apr–Aug 2020. **Use this for dashboard replay** | Yes |
| `hybrid_final.joblib` | Both final models, for data after August 2020 | Yes |
| `grafik/*.png` | 8 charts for the report | Yes |
| `hasil_hybrid.csv` | Summary per method, label set and seed | No, created by `train_hybrid.py` |
| `kejadian_hybrid.csv` | First warning and first red per event | No, created by `train_hybrid.py` |
| `alarm_hybrid.csv` | Log of alarm episodes with sensor values | No, created by `train_hybrid.py` |
| `tradeoff_hybrid.csv` | Other thresholds, only for the trade-off curve | No, created by `train_hybrid.py` |

### Status rules

The rules are checked in order and the first one that matches is used:

| # | Condition | Status |
|---|---|---|
| 1 | No data, frozen sensor, or fewer than 8 valid minutes in the last 15 | Abu-abu (grey) |
| 2 | HGB is active and the HGB alarm is on | Merah (red) |
| 3 | HGB not active yet and the IF alarm has been on for 60+ minutes | Merah (red) |
| 4 | IF alarm is on | Kuning (yellow) |
| 5 | Otherwise | Hijau (green) |

The full column descriptions, in Indonesian, are in
[`anomaly_detection/HYBRID_KELUARAN.md`](anomaly_detection/HYBRID_KELUARAN.md).

---

## 5. Using the final model on new data

```python
import joblib
from hybrid import score_table

bundle = joblib.load("output/hybrid/hybrid_final.joblib")
status = score_table(bundle, new_online_table)  # table must have the columns made by preprocess_dataprep.py
```

**Don't use `hybrid_final.joblib` to replay February–August 2020.** That model was trained on those months,
so red looks like 16/16 events with 0 false alarms. The honest walk-forward result is 14/16 with 5 false alarms.
For a replay, read `status_walkforward.parquet`.

---

## 6. Troubleshooting

| Problem | Fix |
|---|---|
| `FileNotFoundError: ... MetroPT3(AirCompressor).csv` | The CSV is missing or in the wrong place. Put it in the repo root with exactly that name (step 1) |
| `FileNotFoundError: ... hasil_hybrid.csv` when making charts | Run `python train_hybrid.py` first |
| `ModuleNotFoundError: No module named 'evaluation'` | Run the scripts from inside `anomaly_detection/`, and keep all `.py` files together |
| `ImportError: Unable to find a usable engine ... parquet` | `pip install pyarrow` |
| Typing the filename in PowerShell fails | Put the name in quotes, because of the parentheses: `"MetroPT3(AirCompressor).csv"` |
| A row-count check prints `BEDA` | The CSV is incomplete or was edited. Download it again |

---

## Reference

Davari, N., Veloso, B., Ribeiro, R. P., Pereira, P. M., & Gama, J. (2021).
*Predictive maintenance based on anomaly detection using deep learning for air production unit in the railway industry.*
IEEE DSAA 2021. The 16 evaluation events come from Table II of this paper.
