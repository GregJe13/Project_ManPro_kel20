"""
Uji aturan status di hybrid.py dengan data buatan, tanpa dataset dan tanpa training.

Jalankan:  python test_hybrid.py
"""
import unittest

import numpy as np
import pandas as pd

from hybrid import GREEN, GREY, RED, TRIGGER_HGB, TRIGGER_IF, TRIGGER_IF_LONG, YELLOW, combine_status, escalated, score_table, to_grid

START = pd.Timestamp("2020-06-01 00:00")


def scored(minutes, if_on=(), hgb_on=(), active=True, drop=()):
    """if_on dan hgb_on berisi pasangan (awal, akhir) dalam menit. drop = menit tanpa data."""
    index = pd.date_range(START, periods=minutes, freq="1min")
    if_score, hgb_score = np.zeros(minutes), np.zeros(minutes)
    for a, b in if_on:
        if_score[a:b] = 1
    for a, b in hgb_on:
        hgb_score[a:b] = 1
    frame = pd.DataFrame(
        {"if_score": if_score, "if_threshold": 0.5, "hgb_score": hgb_score, "hgb_threshold": 0.5, "hgb_aktif": active},
        index=index,
    )
    keep = np.ones(minutes, bool)
    for a, b in drop:
        keep[a:b] = False
    return frame[keep]


def at(frame, minute):
    return frame.loc[START + pd.Timedelta(minutes=minute)]


class TestStatus(unittest.TestCase):
    def test_tanpa_alarm_hijau(self):
        frame = combine_status(scored(60))
        self.assertEqual(at(frame, 30)["status"], GREEN)

    def test_awal_data_abu_abu_sampai_8_menit(self):
        frame = combine_status(scored(30, if_on=[(0, 30)]))
        self.assertEqual(at(frame, 6)["status"], GREY)
        self.assertEqual(at(frame, 7)["status"], YELLOW)

    def test_alarm_if_kuning(self):
        frame = combine_status(scored(120, if_on=[(30, 60)]))
        self.assertEqual(at(frame, 50)["status"], YELLOW)
        self.assertEqual(at(frame, 50)["pemicu"], TRIGGER_IF)

    def test_lonjakan_singkat_tidak_alarm(self):
        frame = combine_status(scored(120, if_on=[(30, 33)]))
        self.assertEqual(at(frame, 33)["status"], GREEN)

    def test_hgb_aktif_merah_mengalahkan_kuning(self):
        frame = combine_status(scored(120, if_on=[(30, 90)], hgb_on=[(30, 90)]))
        self.assertEqual(at(frame, 50)["status"], RED)
        self.assertEqual(at(frame, 50)["pemicu"], TRIGGER_HGB)

    def test_hgb_belum_aktif_tidak_bisa_merah_sendiri(self):
        frame = combine_status(scored(120, hgb_on=[(30, 90)], active=False))
        self.assertEqual(at(frame, 50)["status"], GREEN)

    def test_cold_start_naik_merah_setelah_60_menit(self):
        frame = combine_status(scored(240, if_on=[(30, 200)], active=False))
        alarm_on = frame.index[frame["if_alarm"]][0]
        before = frame.loc[alarm_on + pd.Timedelta(minutes=59)]
        after = frame.loc[alarm_on + pd.Timedelta(minutes=60)]
        self.assertEqual(before["status"], YELLOW)
        self.assertEqual(after["status"], RED)
        self.assertEqual(after["pemicu"], TRIGGER_IF_LONG)

    def test_setelah_hgb_aktif_if_lama_tetap_kuning(self):
        frame = combine_status(scored(240, if_on=[(30, 200)], active=True))
        self.assertEqual(at(frame, 190)["status"], YELLOW)

    def test_abu_abu_mengalahkan_alarm(self):
        frame = combine_status(scored(120, if_on=[(0, 120)], hgb_on=[(0, 120)], drop=[(40, 90)]))
        self.assertEqual(at(frame, 92)["status"], GREY)
        self.assertEqual(at(frame, 97)["status"], RED)

    def test_menit_tanpa_data_abu_abu_di_grid(self):
        frame = combine_status(scored(120, drop=[(40, 90)]))
        grid = pd.date_range(START, periods=120, freq="1min")
        full = to_grid(frame, grid)
        self.assertEqual(len(full), 120)
        self.assertEqual(at(full, 60)["status"], GREY)
        self.assertFalse(at(full, 60)["if_alarm"])

    def test_jeda_pendek_episode_berlanjut_jeda_panjang_mulai_ulang(self):
        index = pd.date_range(START, periods=300, freq="1min")
        short = np.zeros(300, bool)
        short[0:40] = True
        short[60:100] = True  # Jeda 20 menit.
        result = escalated(pd.Series(short, index=index))
        self.assertTrue(result.iloc[60])

        long_gap = np.zeros(300, bool)
        long_gap[0:40] = True
        long_gap[100:150] = True  # Jeda 60 menit.
        result = escalated(pd.Series(long_gap, index=index))
        self.assertFalse(result.iloc[149])

    def test_status_tidak_memakai_masa_depan(self):
        rng = np.random.default_rng(0)
        data = scored(2000, active=False)
        data["if_score"] = rng.random(2000) < 0.6
        data["if_score"] = data["if_score"].astype(float)
        data["hgb_score"] = (rng.random(2000) < 0.3).astype(float)
        full = combine_status(data)
        for cut in (300, 1000, 1700):
            partial = combine_status(data.iloc[:cut])
            pd.testing.assert_series_equal(partial["status"], full["status"].iloc[:cut])
            pd.testing.assert_series_equal(partial["pemicu"], full["pemicu"].iloc[:cut])

    def test_score_table(self):
        class FakeForest:
            def score_samples(self, X):
                return -X[:, 0]

        class FakeBoost:
            def predict_proba(self, X):
                return np.column_stack([1 - X[:, 0], X[:, 0]])

        index = pd.date_range(START, periods=100, freq="1min")
        table = pd.DataFrame({"a": np.zeros(100), "b": np.zeros(100), "data_ok": True}, index=index)
        table.iloc[50:70, table.columns.get_loc("a")] = 1.0
        table.iloc[80:90, table.columns.get_loc("data_ok")] = False
        table.iloc[20, table.columns.get_loc("a")] = np.nan
        bundle = {
            "iforest": FakeForest(), "if_features": ["a"], "if_threshold": 0.5, "medians": pd.Series({"a": 0.0}),
            "hgb": FakeBoost(), "hgb_features": ["b"], "hgb_threshold": 0.5,
        }
        out = score_table(bundle, table)
        self.assertEqual(len(out), 100)
        self.assertEqual(at(out, 25)["status"], GREEN)
        self.assertEqual(at(out, 65)["status"], YELLOW)
        self.assertEqual(at(out, 85)["status"], GREY)


if __name__ == "__main__":
    unittest.main()
