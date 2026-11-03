# Keluaran sistem hybrid

Panduan untuk tim dashboard. Semua file ada di `output/hybrid/`.

## Cara membuat ulang

```
cd anomaly_detection
python preprocess_dataprep.py
python train_hybrid.py
python test_hybrid.py
python visualize_hybrid.py   # Grafik untuk laporan, disimpan di output/hybrid/grafik/
```

Python akan membaca `preprocess_iforest.py`, `model_iforest.py`, `model_supervised.py`, `preprocess_supervised.py`, dan `evaluation.py`, jadi semuanya harus tetap satu folder.

## Arti status

Aturan diperiksa berurutan, dan yang pertama cocok dipakai.

| Urutan | Kondisi | Status | `pemicu` |
|---|---|---|---|
| 1 | Data kosong, sensor beku, atau kurang dari 8 menit valid dalam 15 menit terakhir | Abu-abu | `-` |
| 2 | HGB aktif dan alarm HGB menyala | Merah | `HGB` |
| 3 | HGB belum aktif dan alarm IF sudah berjalan minimal 60 menit | Merah | `IF bertahan 60 menit` |
| 4 | Alarm IF menyala | Kuning | `IF` |
| 5 | Selain itu | Hijau | `-` |

Notifikasi hanya untuk Merah. Kuning cukup tampil di dashboard dan log, karena alarm palsunya sekitar 3 per minggu.

## Tiga hal yang wajib diikuti

1. **Replay Februari-Agustus 2020 harus memakai `status_walkforward.parquet`, bukan `hybrid_final.joblib`.** Model akhir sudah melihat bulan-bulan itu, sehingga Merah terlihat 16/16 tanpa alarm palsu. Hasil jujurnya 14/16 dengan 5 alarm palsu.
2. **`hgb_score` bukan peluang kegagalan.** Tampilkan sebagai "skor keyakinan model". `class_weight="balanced"` menggeser nilainya, dan 16 kejadian terlalu sedikit untuk kalibrasi.
3. **Abu-abu tidak boleh ditampilkan sebagai Hijau.** Sekitar 23,5% menit April-Agustus berstatus Abu-abu.

## `status_walkforward.parquet`

Satu baris per menit, 1 April sampai 31 Agustus 2020. Indeks: `timestamp`. Nilai sensor untuk grafik ada di `output/dataprep/dataset_online_1min.parquet` dengan indeks yang sama.

| Kolom | Isi |
|---|---|
| `status` | Hijau, Kuning, Merah, atau Abu-abu |
| `pemicu` | Penyebab status, lihat tabel di atas |
| `if_alarm` | Alarm IF menyala (2/3 menit valid dalam 15 menit terakhir abnormal) |
| `hgb_alarm` | Alarm HGB menyala. Selalu `False` selama HGB belum aktif |
| `if_score` | Skor keanehan IF, makin tinggi makin aneh. Kosong jika tidak ada data |
| `if_threshold` | Ambang IF pada bulan itu |
| `hgb_score` | Skor keyakinan HGB, 0 sampai 1. Bernilai 0 selama HGB belum aktif |
| `hgb_threshold` | Ambang HGB, 0,5 |
| `hgb_aktif` | HGB sudah aktif. `False` untuk April dan menit tanpa data |
| `in_grace` | Menit sesudah kegagalan dan sebelum perbaikan. Statusnya tetap ada, tetapi tidak ikut dinilai |

## `alarm_hybrid.csv`

Satu baris per episode peringatan, untuk halaman log. Zona grace tidak termasuk.

| Kolom | Isi |
|---|---|
| `mulai`, `selesai`, `menit` | Waktu episode. Jeda sampai 30 menit digabung |
| `tertinggi` | Level tertinggi selama episode: Kuning atau Merah |
| `merah_mulai`, `pemicu_merah` | Kapan menjadi Merah dan penyebabnya |
| `TP3`, `Oil_temperature`, `Motor_current`, `duty_60`, `load_run_min` | Nilai saat alarm mulai. `duty_60` bisa kosong tepat sesudah gap |
| `penilaian`, `merah_palsu` | Hasil penilaian terhadap label 2020. Hanya untuk laporan, jangan tampilkan ke teknisi |

## `kejadian_hybrid.csv`

Satu baris per kejadian, dari 16 kejadian paper. `official` menandai 4 kegagalan resmi.

| Kolom | Isi |
|---|---|
| `peringatan_pertama`, `merah_pertama` | Kuning atau Merah pertama, dan Merah pertama |
| `delay_peringatan_menit`, `delay_merah_menit` | Menit dari awal kejadian. Negatif berarti sebelum label |
| `selisih_ke_merah_menit` | Jarak dari peringatan pertama ke Merah pertama |
| `tertinggi` | Merah, Kuning, atau Tidak terdeteksi |

Delay dibatasi 2 jam sebelum mulai. Angka -120 berarti alarm sudah menyala minimal 2 jam sebelumnya.

## `hasil_hybrid.csv` dan `tradeoff_hybrid.csv`

- `hasil_hybrid.csv`: ringkasan per metode, label, dan seed.
- `tradeoff_hybrid.csv`: variasi ambang IF (`if_k`) dan HGB (`hgb_ambang`). Hanya untuk kurva Tujuan 3. Jangan memilih ambang terbaik dari tabel ini tanpa data uji baru, karena kejadiannya sama.

## `hybrid_final.joblib`

Model untuk data sesudah Agustus 2020. Kinerjanya belum terukur langsung, karena tidak punya data uji sendiri.

| Kunci | Isi |
|---|---|
| `iforest`, `if_threshold`, `if_features`, `medians` | IsolationForest dengan 51 fitur |
| `hgb`, `hgb_threshold`, `hgb_features` | HistGradientBoosting dengan 9 fitur |
| `n_events`, `seed`, `train_days` | Informasi pelatihan |

```python
import joblib
from hybrid import score_table

bundle = joblib.load("output/hybrid/hybrid_final.joblib")
status = score_table(bundle, tabel_online_baru)  # Tabel harus berisi kolom hasil preprocess_dataprep.py.
```
