"""
Grafik hasil sistem hybrid, untuk laporan. Bukan dashboard.

Jalankan:  python visualize_hybrid.py   (sesudah train_hybrid.py)
Hasil:     output/hybrid/grafik/*.png

Tidak ada training di sini. Semua grafik dibaca dari file hasil walk-forward, dan judulnya dihitung dari data,
jadi angkanya mengikuti hasil terbaru. Data tiap grafik ada di CSV dan parquet pada output/hybrid/.
"""
import matplotlib

matplotlib.use("Agg")

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle
from matplotlib.ticker import MultipleLocator

from evaluation import to_episodes
from hybrid import GREEN, GREY, RED, YELLOW
from preprocess_dataprep import OUT_DIR as DATAPREP_DIR
from preprocess_iforest import EVENTS
from train_hgb_dataprep import HYBRID_DIR

CHART_DIR = HYBRID_DIR / "grafik"
DPI = 200

SURFACE, INK, INK_2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
BLUE, ORANGE, WASH = "#2a78d6", "#eb6834", "#f0efec"
STATUS_COLOR = {GREEN: "#dff0df", YELLOW: "#fab219", RED: "#d03b3b", GREY: AXIS}
STATUS_ORDER = [GREEN, YELLOW, RED, GREY]
STATUS_CMAP = ListedColormap([STATUS_COLOR[s] for s in STATUS_ORDER])
STATUS_CODE = {s: i for i, s in enumerate(STATUS_ORDER)}
BAR_PT = 8.5
MONTHS = {4: "Apr", 5: "Mei", 6: "Jun", 7: "Jul", 8: "Agu"}

plt.rcParams.update(
    {
        "font.family": ["Segoe UI", "DejaVu Sans"],
        "font.size": 9,
        "text.color": INK,
        "axes.edgecolor": AXIS,
        "axes.labelcolor": INK_2,
        "axes.facecolor": SURFACE,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.axisbelow": True,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "xtick.labelcolor": INK_2,
        "ytick.labelcolor": INK_2,
        "figure.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
    }
)


# ---------------------------------------------------------------------------
# Pembantu
# ---------------------------------------------------------------------------
def num(value):
    text = f"{value:.1f}".rstrip("0").rstrip(".")
    return text.replace(".", ",")


def value_range(a, b):
    return num(a) if a == b else f"{num(min(a, b))}–{num(max(a, b))}"


def clock(t):
    return f"{t:%H:%M}"


def header(fig, title, subtitle, note=None):
    fig.text(0.02, 0.975, title, fontsize=12.5, fontweight=600, va="top", color=INK)
    fig.text(0.02, 0.975 - 0.34 / fig.get_figheight(), subtitle, fontsize=9, va="top", color=INK_2)
    if note:
        fig.text(0.02, 0.02, note, fontsize=8, va="bottom", color=MUTED)


def save(fig, name):
    fig.savefig(CHART_DIR / name, dpi=DPI)
    plt.close(fig)
    print("  ", name, flush=True)


def tip_cap(ax, thick, axis="x"):
    """Ujung bulat menambah panjang garis, jadi garis dipendekkan supaya ujungnya tepat di nilai data."""
    box = ax.get_window_extent()
    low, high = ax.get_xlim() if axis == "x" else ax.get_ylim()
    size = box.width if axis == "x" else box.height
    return (thick / 72 * ax.figure.dpi / 2) * (high - low) / size


def hbar(ax, y, value, color, thick=BAR_PT):
    if value > 0:
        low, high = ax.get_xlim()
        # Nilai yang lebih kecil dari ujung bulat tetap digambar sebagai batang pendek, bukan hilang.
        end = max(value - tip_cap(ax, thick, "x"), 0.002 * (high - low))
        ax.plot([0, end], [y, y], lw=thick, color=color, solid_capstyle="round", zorder=3)


def vbar(ax, x, value, color, thick=BAR_PT):
    if value > 0:
        low, high = ax.get_ylim()
        end = max(value - tip_cap(ax, thick, "y"), 0.002 * (high - low))
        ax.plot([x, x], [0, end], lw=thick, color=color, solid_capstyle="round", zorder=3)


def hide_value_axis(ax):
    ax.set_xticks([])
    ax.spines["bottom"].set_visible(False)
    ax.tick_params(axis="y", length=0)


def status_legend(fig, y, extra=()):
    handles = [Patch(facecolor=STATUS_COLOR[s], edgecolor=AXIS if s == GREEN else "none", label=s) for s in STATUS_ORDER]
    fig.legend(handles=handles + list(extra), loc="upper left", bbox_to_anchor=(0.015, y), ncol=len(handles) + len(extra), frameon=False, fontsize=8.5, handlelength=1.1, columnspacing=1.4)


# ---------------------------------------------------------------------------
# Grafik
# ---------------------------------------------------------------------------
METHOD_LABEL = {
    "Hybrid Merah": "Hybrid: Merah",
    "Hybrid Kuning atau Merah": "Hybrid: Kuning atau Merah",
    "HGB saja": "HGB saja",
    "IF saja": "IF saja",
    "Aturan run>=15 & oli>=60": "Aturan: run ≥ 15 menit & oli ≥ 60 °C",
}


def method_rows(results, label_set, seeds=(42, 7)):
    rows = []
    for method in METHOD_LABEL:
        data = results[(results["metode"] == method) & (results["label"] == label_set)].set_index("seed")
        first, second = data.loc[seeds[0]], data.loc[seeds[1]]
        rows.append(
            {
                "metode": method,
                "kejadian": int(first["kejadian"].split("/")[0]),
                "resmi": first["resmi"],
                "palsu": first["alarm_palsu"],
                "palsu_teks": value_range(first["alarm_palsu"], second["alarm_palsu"]),
                "delay": first["delay_median_menit"],
                "delay_teks": value_range(first["delay_median_menit"], second["delay_median_menit"]),
            }
        )
    return pd.DataFrame(rows).set_index("metode")


def chart_methods(results):
    rows = method_rows(results, 16)
    red, rule = rows.loc["Hybrid Merah"], rows.loc["Aturan run>=15 & oli>=60"]
    fig, axes = plt.subplots(1, 3, figsize=(11, 4.2), sharey=True, gridspec_kw={"width_ratios": [1.45, 1, 1], "wspace": 0.1})
    fig.subplots_adjust(left=0.235, right=0.985, top=0.74, bottom=0.13)
    header(
        fig,
        f"Merah: {red['kejadian']}/16 kejadian, {red['resmi']} resmi, {red['palsu']} alarm palsu. Aturan sederhana: {rule['kejadian']}/16, {rule['resmi']}, {rule['palsu']} alarm palsu",
        "Walk-forward April–Agustus 2020, 16 label. Rentang menunjukkan hasil seed 42 dan 7.",
        "Ambang aturan dipilih setelah melihat kejadian yang sama, jadi perbandingannya tidak sepenuhnya adil. IF = IsolationForest, HGB = HistGradientBoosting.",
    )
    y = np.arange(len(rows))[::-1]
    panels = [
        ("kejadian", "Kejadian terdeteksi (dari 16)", 29, lambda r: f"{r['kejadian']}/16, resmi {r['resmi']}"),
        ("palsu", "Alarm palsu (22 minggu)", 105, lambda r: r["palsu_teks"]),
        ("delay", "Delay median (menit)", 15, lambda r: r["delay_teks"]),
    ]
    for ax, (column, title, limit, text) in zip(axes, panels):
        ax.set_xlim(0, limit)
        ax.set_ylim(-0.6, len(rows) - 0.4)
        ax.set_title(title, loc="left", fontsize=9.5, fontweight=600, color=INK, pad=8)
        for yi, (method, row) in zip(y, rows.iterrows()):
            hbar(ax, yi, row[column], BLUE if method.startswith("Hybrid") else MUTED)
            ax.text(row[column] + limit * 0.015, yi, text(row), va="center", fontsize=8.5, color=INK)
        hide_value_axis(ax)
    axes[0].set_yticks(y, [METHOD_LABEL[m] for m in rows.index], fontsize=9, color=INK)
    save(fig, "01_perbandingan_metode.png")


def chart_status(status):
    days = len(status) // 1440
    codes = status["status"].map(STATUS_CODE).to_numpy().reshape(days, 1440)
    inside = np.zeros(len(status), bool)
    for e in EVENTS.itertuples():
        inside |= (status.index >= e.start) & (status.index <= e.end)
    red = (status["status"] == RED).to_numpy()
    grey_share = (status["status"] == GREY).mean() * 100

    fig, ax = plt.subplots(figsize=(10, 12.5))
    fig.subplots_adjust(left=0.075, right=0.885, top=0.905, bottom=0.04)
    header(
        fig,
        f"{inside[red].mean() * 100:.0f}% menit Merah jatuh di dalam kejadian berlabel",
        f"Status per menit, walk-forward. Tiap baris satu hari, kotak hitam adalah label kejadian. Abu-abu (tidak ada data) mencakup {num(round(grey_share, 1))}% waktu.",
    )
    ax.imshow(codes, aspect="auto", cmap=STATUS_CMAP, vmin=0, vmax=3, interpolation="nearest", extent=[0, 24, days, 0])

    start_day = status.index[0].normalize()
    labels = {}
    for e in EVENTS.itertuples():
        name = e.id.split("_")[1] if e.official else e.id
        day = e.start.normalize()
        labels.setdefault((day - start_day).days, []).append((name, bool(e.official)))
        while day <= e.end:
            low = max(e.start, day)
            high = min(e.end, day + pd.Timedelta(days=1))
            row = (day - start_day).days
            ax.add_patch(Rectangle(((low - day).total_seconds() / 3600, row), (high - low).total_seconds() / 3600, 1, fill=False, edgecolor=INK, lw=0.8, zorder=4))
            day += pd.Timedelta(days=1)

    # Baris sehari terlalu rapat untuk dua label, jadi label hari berurutan digeser ke samping.
    previous_row, shift = None, 0.0
    for row, names in sorted(labels.items()):
        shift = 1.7 if previous_row is not None and row - previous_row == 1 and shift == 0 else 0.0
        text = ", ".join(name for name, _ in names)
        weight = 700 if any(official for _, official in names) else 400
        ax.text(24.2 + shift, row + 0.5, text, va="center", fontsize=6.8, fontweight=weight, color=INK, clip_on=False)
        previous_row = row

    ticks, tick_labels = [], []
    for row in range(days):
        day = start_day + pd.Timedelta(days=row)
        if day.day in (1, 8, 15, 22):
            ticks.append(row + 0.5)
            tick_labels.append(f"{day.day} {MONTHS[day.month]}")
        if day.day == 1 and row > 0:
            ax.axhline(row, color=SURFACE, lw=1.6, zorder=3)
    ax.set_yticks(ticks, tick_labels, fontsize=7.5)
    ax.set_xticks(range(0, 25, 3), [f"{h:02d}:00" for h in range(0, 25, 3)], fontsize=8)
    ax.tick_params(length=0)
    for side in ax.spines.values():
        side.set_visible(False)
    status_legend(fig, 0.932, [Line2D([], [], color=INK, lw=0.9, label="Label kejadian (F = kegagalan resmi)")])
    save(fig, "02_status_per_menit.png")


def chart_delay(events):
    after_april = events[events["start"] >= "2020-05-01"]["selisih_ke_merah_menit"]
    fig, ax = plt.subplots(figsize=(10.5, 6.2))
    fig.subplots_adjust(left=0.09, right=0.98, top=0.82, bottom=0.09)
    header(
        fig,
        f"Sesudah April, Kuning hanya mendahului Merah sekitar {num(after_april.median())} menit (rentang {num(after_april.min())}–{num(after_april.max())})",
        "Peringatan dan Merah pertama per kejadian, dihitung dari awal label. Negatif berarti sebelum label, dan dibatasi −120 menit. Seed 42, 16 label.",
    )
    y = np.arange(len(events))
    ax.set_xlim(-135, 108)
    ax.set_ylim(len(events) - 0.4, -1.3)
    ax.set_xticks([-120, -90, -60, -30, 0, 30, 60, 90])
    ax.xaxis.grid(True, color=GRID, lw=0.7)
    ax.axvline(0, color=INK_2, lw=1, zorder=2)
    ax.text(0, -0.95, "awal label", ha="center", va="center", fontsize=8, color=INK_2, bbox={"facecolor": SURFACE, "edgecolor": "none", "pad": 1.5}, zorder=6)
    ax.set_xlabel("Menit dari awal label")
    ax.tick_params(length=0)
    ax.spines["left"].set_visible(False)

    names = []
    for yi, e in zip(y, events.itertuples()):
        names.append(f"{e.Index.split('_')[1]} ({e.Index.split('_')[0]})" if e.official else e.Index)
        warning, red = e.delay_peringatan_menit, e.delay_merah_menit
        if not np.isnan(red):
            ax.plot([warning, red], [yi, yi], color=AXIS, lw=1.5, zorder=2)
            ax.scatter(red, yi, marker="D", s=46, color=STATUS_COLOR[RED], edgecolor=SURFACE, lw=1.2, zorder=5)
        ax.scatter(warning, yi, marker="o", s=62, color=STATUS_COLOR[YELLOW], edgecolor=SURFACE, lw=1.2, zorder=4)
    ax.set_yticks(y, names, fontsize=9)
    for tick, official in zip(ax.get_yticklabels(), events["official"]):
        tick.set_color(INK)
        tick.set_fontweight(700 if official else 400)

    position = {e: i for i, e in enumerate(events.index)}
    ax.text(21, position["E03"], "hanya Kuning: Merah menyala 37 menit sesudah kejadian selesai", va="center", fontsize=8, color=INK_2)
    ax.text(13, position["E04"], "hanya Kuning: episode terlalu singkat", va="center", fontsize=8, color=INK_2)
    ax.text(-112, position["E15_F4"], "Kuning sejak 00:27, Merah terus sejak 08:31, label 14:30", va="center", fontsize=8, color=INK_2, bbox={"facecolor": SURFACE, "edgecolor": "none", "pad": 1.5}, zorder=6)
    separator = position["E05"] - 0.5
    ax.axhline(separator, color=AXIS, lw=0.8)
    ax.text(107, position["E01"] - 0.75, "Cold start: Merah dari IF yang bertahan 60 menit", ha="right", fontsize=8, color=INK_2)
    ax.text(107, separator + 0.42, "HGB aktif: Merah dari HGB", ha="right", fontsize=8, color=INK_2)
    fig.legend(
        handles=[
            Line2D([], [], marker="o", ls="", color=STATUS_COLOR[YELLOW], markersize=8, label="Peringatan pertama (Kuning atau Merah)"),
            Line2D([], [], marker="D", ls="", color=STATUS_COLOR[RED], markersize=7, label="Merah pertama"),
        ],
        loc="upper left", bbox_to_anchor=(0.015, 0.905), ncol=2, frameon=False, fontsize=8.5,
    )
    save(fig, "03_waktu_per_kejadian.png")


def chart_false_alarms(alarms):
    alarms = alarms.assign(month=alarms["mulai"].dt.month)
    months = sorted(MONTHS)
    warning = alarms[alarms["penilaian"] == "alarm palsu"].groupby("month").size().reindex(months, fill_value=0)
    red = alarms[alarms["merah_palsu"]].groupby("month").size().reindex(months, fill_value=0)
    fig, ax = plt.subplots(figsize=(8.8, 4.6))
    fig.subplots_adjust(left=0.06, right=0.98, top=0.76, bottom=0.17)
    header(
        fig,
        f"Peringatan palsu {warning.min()}–{warning.max()} per bulan, Merah palsu paling banyak {red.max()} per bulan",
        f"Jumlah episode alarm palsu per bulan, seed 42, 16 label. Total {warning.sum()} peringatan dan {red.sum()} Merah.",
        "Agustus tidak punya label, jadi alarm bulan itu belum tentu palsu. Merah palsu April berasal dari IF yang bertahan 60 menit.",
    )
    ax.set_ylim(0, warning.max() * 1.18)
    ax.set_xlim(-0.6, len(months) - 0.4)
    ax.yaxis.set_major_locator(MultipleLocator(5))
    ax.yaxis.grid(True, color=GRID, lw=0.7)
    ax.tick_params(length=0)
    ax.spines["left"].set_visible(False)
    gap = 0.14
    for x, month in enumerate(months):
        vbar(ax, x - gap, warning[month], STATUS_COLOR[YELLOW], thick=11)
        vbar(ax, x + gap, red[month], STATUS_COLOR[RED], thick=11)
        ax.text(x - gap, warning[month] + 0.4, str(warning[month]), ha="center", fontsize=8.5)
        ax.text(x + gap, red[month] + 0.4, str(red[month]), ha="center", fontsize=8.5)
    ax.set_xticks(range(len(months)), [MONTHS[m] for m in months], fontsize=9, color=INK)
    fig.legend(
        handles=[Patch(facecolor=STATUS_COLOR[YELLOW], label="Peringatan palsu (Kuning atau Merah)"), Patch(facecolor=STATUS_COLOR[RED], label="Merah palsu")],
        loc="upper left", bbox_to_anchor=(0.015, 0.875), ncol=2, frameon=False, fontsize=8.5, handlelength=1.1,
    )
    save(fig, "04_alarm_palsu_per_bulan.png")


def chart_tradeoff(tradeoff):
    data = tradeoff[tradeoff["hgb_ambang"] == 0.5].set_index("if_k")
    first, last = data.iloc[0], data.iloc[-1]
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.9), gridspec_kw={"wspace": 0.2})
    fig.subplots_adjust(left=0.045, right=0.98, top=0.72, bottom=0.18)
    header(
        fig,
        f"Menaikkan ambang IF: alarm palsu turun dari {first['peringatan_palsu']:.0f} ke {last['peringatan_palsu']:.0f}, deteksi dari {first['peringatan_kejadian']:.0f}/16 ke {last['peringatan_kejadian']:.0f}/16",
        "Level peringatan (Kuning atau Merah) untuk ambang Q3 + k·IQR. Garis abu-abu menandai k = 3, ambang yang dipakai sistem.",
        "Seed 42, pada kejadian yang sama dengan data uji. Hanya untuk kurva Tujuan 3, bukan dasar memilih ambang.",
    )
    panels = [
        ("peringatan_kejadian", "Kejadian terdeteksi (dari 16)", (0, 19), 4),
        ("peringatan_palsu", "Alarm palsu (22 minggu)", (0, 160), 40),
        ("peringatan_delay", "Delay median (menit)", (0, 16), 4),
    ]
    for ax, (column, title, limits, step) in zip(axes, panels):
        ax.set_title(title, loc="left", fontsize=9.5, fontweight=600, color=INK, pad=8)
        ax.set_ylim(*limits)
        ax.yaxis.set_major_locator(MultipleLocator(step))
        ax.set_xlim(1, 6.6)
        ax.yaxis.grid(True, color=GRID, lw=0.7)
        ax.axvline(3, color=AXIS, lw=1.2, zorder=1)
        ax.plot(data.index, data[column], color=BLUE, lw=1.5, marker="o", markersize=6, markeredgecolor=SURFACE, markeredgewidth=1.2, zorder=3)
        for k, value in data[column].items():
            ax.annotate(num(value), (k, value), xytext=(0, 7), textcoords="offset points", ha="center", fontsize=8.5, bbox={"facecolor": SURFACE, "edgecolor": "none", "pad": 0.8}, zorder=5)
        ax.set_xticks(data.index, [num(k) for k in data.index], fontsize=8.5)
        ax.set_xlabel("k")
        ax.tick_params(length=0)
        ax.spines["left"].set_visible(False)
    save(fig, "05_tradeoff_ambang_if.png")


def chart_labels(results):
    rows_16, rows_4 = method_rows(results, 16), method_rows(results, 4)
    methods = ["Hybrid Kuning atau Merah", "Hybrid Merah", "HGB saja"]
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.9), sharey=True, gridspec_kw={"wspace": 0.08})
    fig.subplots_adjust(left=0.2, right=0.98, top=0.68, bottom=0.05)
    header(
        fig,
        f"Dengan hanya 4 label, Merah turun dari {rows_16.loc['Hybrid Merah', 'kejadian']}/16 ke {rows_4.loc['Hybrid Merah', 'kejadian']}/16, tetapi peringatan tetap {rows_4.loc['Hybrid Kuning atau Merah', 'kejadian']}/16",
        "Uji ketahanan: 16 label paper dibanding hanya 4 kegagalan resmi. Walk-forward April–Agustus, rentang dari seed 42 dan 7.",
    )
    y = np.arange(len(methods))[::-1]
    gap = 0.17
    panels = [("kejadian", "Kejadian terdeteksi (dari 16)", 19, lambda r: f"{r['kejadian']}/16"), ("palsu", "Alarm palsu (22 minggu)", 95, lambda r: r["palsu_teks"])]
    for ax, (column, title, limit, text) in zip(axes, panels):
        ax.set_xlim(0, limit)
        ax.set_ylim(-0.6, len(methods) - 0.4)
        ax.set_title(title, loc="left", fontsize=9.5, fontweight=600, color=INK, pad=8)
        for yi, method in zip(y, methods):
            for rows, color, shift in ((rows_16, BLUE, gap), (rows_4, ORANGE, -gap)):
                hbar(ax, yi + shift, rows.loc[method, column], color, thick=7)
                ax.text(rows.loc[method, column] + limit * 0.015, yi + shift, text(rows.loc[method]), va="center", fontsize=8.5)
        hide_value_axis(ax)
    axes[0].set_yticks(y, [METHOD_LABEL[m] for m in methods], fontsize=9, color=INK)
    fig.legend(
        handles=[Patch(facecolor=BLUE, label="16 label paper"), Patch(facecolor=ORANGE, label="4 label resmi")],
        loc="upper left", bbox_to_anchor=(0.015, 0.875), ncol=2, frameon=False, fontsize=8.5, handlelength=1.1,
    )
    save(fig, "06_16_vs_4_label.png")


def time_label(value, _position):
    t = mdates.num2date(value)
    return f"{t:%H:%M}\n{t.day} {MONTHS[t.month]}" if t.hour == 0 else f"{t:%H:%M}"


def chart_zoom(status, online, event_id, window, name, title, subtitle, panels):
    event = EVENTS.set_index("id").loc[event_id]
    low, high = pd.Timestamp(window[0]), pd.Timestamp(window[1])
    part = status.loc[low:high]
    sensors = online.loc[low:high]
    fig, axes = plt.subplots(
        len(panels) + 1, 1, figsize=(10.5, 1.45 * len(panels) + 2.6), sharex=True,
        gridspec_kw={"height_ratios": [0.32] + [1] * len(panels), "hspace": 0.32},
    )
    fig.subplots_adjust(left=0.075, right=0.975, top=1 - 1.25 / fig.get_figheight(), bottom=0.55 / fig.get_figheight())
    header(fig, title, subtitle)
    status_legend(fig, 1 - 0.62 / fig.get_figheight(), [Patch(facecolor=WASH, edgecolor=AXIS, label="Label kejadian")])

    strip = axes[0]
    codes = part["status"].map(STATUS_CODE).to_numpy()[np.newaxis, :]
    strip.imshow(codes, aspect="auto", cmap=STATUS_CMAP, vmin=0, vmax=3, interpolation="nearest", extent=[mdates.date2num(low), mdates.date2num(high), 0, 1])
    strip.set_yticks([0.5], ["Status"], fontsize=8.5)
    strip.tick_params(length=0)
    for side in strip.spines.values():
        side.set_visible(False)
    strip.plot([event["start"], min(event["end"], high)], [1.28, 1.28], color=INK, lw=1.2, clip_on=False, solid_capstyle="butt")
    strip.text(event["start"], 1.42, f"Label {event_id.split('_')[1]}", fontsize=8, color=INK_2, va="bottom", clip_on=False)

    for ax, (column, label, threshold) in zip(axes[1:], panels):
        source = part if column in part.columns else sensors
        scale = 100 if column == "duty_60" else 1
        ax.axvspan(event["start"], min(event["end"], high), color=WASH, lw=0, zorder=0)
        ax.plot(source.index, source[column] * scale, color=BLUE, lw=1.2, zorder=3, solid_joinstyle="round")
        if threshold:
            ax.plot(part.index, part[threshold], color=INK_2, lw=0.9, zorder=2)
            ax.text(1.0, 1.0, "garis gelap = ambang", transform=ax.transAxes, ha="right", va="bottom", fontsize=7.5, color=INK_2)
        ax.set_title(label, loc="left", fontsize=8.8, fontweight=600, color=INK, pad=3)
        ax.yaxis.grid(True, color=GRID, lw=0.7)
        ax.tick_params(length=0)
        ax.spines["left"].set_visible(False)
    axes[-1].xaxis.set_major_locator(mdates.HourLocator(byhour=range(0, 24, 3)))
    axes[-1].xaxis.set_major_formatter(plt.FuncFormatter(time_label))
    axes[-1].set_xlim(low, high)
    save(fig, name)


def first_time(mask):
    return mask.index[mask.to_numpy()][0]


def main():
    CHART_DIR.mkdir(parents=True, exist_ok=True)
    results = pd.read_csv(HYBRID_DIR / "hasil_hybrid.csv")
    events = pd.read_csv(HYBRID_DIR / "kejadian_hybrid.csv", index_col="id", parse_dates=["start", "end"])
    alarms = pd.read_csv(HYBRID_DIR / "alarm_hybrid.csv", parse_dates=["mulai", "selesai", "merah_mulai"])
    tradeoff = pd.read_csv(HYBRID_DIR / "tradeoff_hybrid.csv")
    status = pd.read_parquet(HYBRID_DIR / "status_walkforward.parquet")
    online = pd.read_parquet(DATAPREP_DIR / "dataset_online_1min.parquet", columns=["duty_60", "Oil_temperature", "TP3", "data_ok"])
    online.loc[~online["data_ok"], ["duty_60", "Oil_temperature", "TP3"]] = np.nan

    print("Membuat grafik:")
    chart_methods(results)
    chart_status(status)
    chart_delay(events)
    chart_false_alarms(alarms)
    chart_tradeoff(tradeoff)
    chart_labels(results)

    sensor_panels = [("duty_60", "Beban 60 menit terakhir (%)", None), ("Oil_temperature", "Suhu oli (°C)", None), ("TP3", "Tekanan TP3 (bar)", None)]

    f1 = EVENTS.set_index("id").loc["E02_F1"]
    around = status.loc[f1["start"] - pd.Timedelta(hours=1) : f1["end"]]
    warning_f1 = first_time(around["status"].isin([YELLOW, RED]))
    red_f1 = first_time(around["status"] == RED)
    chart_zoom(
        status, online, "E02_F1", ("2020-04-17 18:00", "2020-04-19 06:00"), "07_zoom_f1_cold_start.png",
        f"F1 pada masa cold start: Kuning {clock(warning_f1)}, lalu Merah {clock(red_f1)} dari IF yang bertahan 60 menit",
        "HGB belum aktif di April. Abu-abu sebelum label berasal dari sensor beku. Walk-forward, seed 42.",
        [("if_score", "Skor IsolationForest", "if_threshold")] + sensor_panels,
    )

    f4 = EVENTS.set_index("id").loc["E15_F4"]
    before = status.loc[f4["start"] - pd.Timedelta(hours=18) : f4["end"]]
    warning_f4 = first_time(before["status"].isin([YELLOW, RED]))
    red_episodes = to_episodes(before["status"] == RED)
    chart_zoom(
        status, online, "E15_F4", ("2020-07-14 21:00", "2020-07-15 21:00"), "08_zoom_f4_lebih_awal.png",
        f"F4: Kuning {clock(warning_f4)} dan Merah terus sejak {clock(red_episodes[-1][0])}, padahal label mulai {clock(f4['start'])}",
        "Alarm menyala beberapa jam sebelum label, tetapi ini baru satu kejadian, jadi belum bisa disebut bukti prediksi. Walk-forward, seed 42.",
        [("if_score", "Skor IsolationForest", "if_threshold"), ("hgb_score", "Skor keyakinan HGB", "hgb_threshold")] + sensor_panels,
    )
    print(f"Selesai: {CHART_DIR}")


if __name__ == "__main__":
    main()
