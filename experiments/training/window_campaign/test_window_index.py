"""B6 — unit tests for window slicing (WINDOW_CAMPAIGN_FINAL_PLAN)."""
from __future__ import annotations

import unittest

from window_index import iter_window_slices


def _windows(t: int, *, wl: int = 96, stride: int = 48, min_tail: int = 24):
    return list(
        iter_window_slices(
            t,
            window_len=wl,
            stride=stride,
            include_short=True,
            min_tail_for_final_window=min_tail,
        )
    )


class TestWindowSlices(unittest.TestCase):
    def test_48_padded_short(self):
        self.assertEqual(_windows(48), [(0, 48)])

    def test_72(self):
        self.assertEqual(_windows(72), [(0, 72)])

    def test_95(self):
        self.assertEqual(_windows(95), [(0, 95)])

    def test_96(self):
        self.assertEqual(_windows(96), [(0, 96)])

    def test_97_only_first(self):
        self.assertEqual(_windows(97), [(0, 96)])

    def test_144(self):
        self.assertEqual(_windows(144), [(0, 96), (48, 144)])

    def test_168_with_final(self):
        self.assertEqual(_windows(168), [(0, 96), (48, 144), (72, 168)])

    def test_192(self):
        self.assertEqual(_windows(192), [(0, 96), (48, 144), (96, 192)])

    def test_240(self):
        self.assertEqual(_windows(240), [(0, 96), (48, 144), (96, 192), (144, 240)])

    def test_no_duplicates(self):
        for t in range(1, 250):
            ws = _windows(t)
            self.assertEqual(len(ws), len(set(ws)))


if __name__ == "__main__":
    unittest.main()