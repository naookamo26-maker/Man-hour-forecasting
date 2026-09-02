"""
時間軸ユーティリティ

このシステムには2つの時間軸がある。

  実時間軸 u : ある案件の経過期間比(0=開始, 1=終了)。案件ごとに実在する軸。
  正準時間軸 s: 学習カーブが住む共通の軸。マイルストーンが常に同じ位置に来る。

両者を結ぶのが Warp(区分線形写像)。
これが設計書 5章 Step3「マイルストーンによる位置合わせ」の実体である。

  学習時: 実時間軸の実績density -> 正準時間軸へ (to_canonical)
  予測時: 正準時間軸の学習カーブ -> 実時間軸へ (to_actual)

位置合わせをしない場合は Warp が恒等写像になるだけで、
呼び出し側のコードは一切変わらない(素朴版と位置合わせ版が同じ経路を通る)。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# 月と経過期間比
# ---------------------------------------------------------------------------
def month_list(start: str, end: str) -> list[str]:
    """'2021-04' 〜 '2022-06' を月文字列 ['2021-04', ...] に展開する。"""
    return [str(p) for p in pd.period_range(pd.Period(start, freq="M"),
                                            pd.Period(end, freq="M"), freq="M")]


def month_edges(months: list[str]) -> np.ndarray:
    """各月の境界を経過期間比 t で返す(日数ベース)。長さ = len(months)+1。"""
    starts = [pd.Period(m, freq="M").start_time for m in months]
    end = pd.Period(months[-1], freq="M").end_time
    days = np.array([(d - starts[0]).days for d in starts] + [(end - starts[0]).days],
                    dtype=float)
    return days / days[-1]


def date_to_t(months: list[str], date) -> float:
    """日付を経過期間比 t に変換する。範囲外は 0〜1 にクリップ。"""
    start = pd.Period(months[0], freq="M").start_time
    end = pd.Period(months[-1], freq="M").end_time
    total = (end - start).days
    v = (pd.Timestamp(date) - start).days / total if total else 0.0
    return float(np.clip(v, 0.0, 1.0))


def t_to_date(months: list[str], t: float) -> pd.Timestamp:
    """経過期間比 t を日付に変換する。"""
    start = pd.Period(months[0], freq="M").start_time
    end = pd.Period(months[-1], freq="M").end_time
    return start + pd.Timedelta(days=float(np.clip(t, 0.0, 1.0)) * (end - start).days)


# ---------------------------------------------------------------------------
# ワープ
# ---------------------------------------------------------------------------
@dataclass
class Warp:
    """正準時間軸 s と実時間軸 u の間の区分線形写像。

    anchors_canonical[i] <-> anchors_actual[i] が対応する。
    両端の 0.0 / 1.0 は常に含める(案件の開始と終了は必ず一致する)。
    アンカーが両端だけなら恒等写像 = 位置合わせなし。
    """

    anchors_canonical: np.ndarray
    anchors_actual: np.ndarray
    names: list[str]

    @classmethod
    def identity(cls) -> "Warp":
        return cls(np.array([0.0, 1.0]), np.array([0.0, 1.0]), [])

    @classmethod
    def build(cls, pairs: list[tuple[str, float, float]], min_gap: float = 0.02) -> "Warp":
        """pairs = [(名前, 正準位置, 実位置), ...] からワープを作る。

        単調増加が崩れる指定(β版がα版より前 等)は順序を保つよう補正する。
        補正しないと逆変換が壊れ、工数が時間軸上で折り返してしまう。
        """
        pairs = sorted(pairs, key=lambda p: p[1])
        names, canon, actual = [], [0.0], [0.0]
        for nm, c, a in pairs:
            if c <= canon[-1] + min_gap or c >= 1.0 - min_gap:
                continue
            a = float(np.clip(a, actual[-1] + min_gap, 1.0 - min_gap))
            names.append(nm)
            canon.append(float(c))
            actual.append(a)
        canon.append(1.0)
        actual.append(1.0)
        return cls(np.array(canon), np.array(actual), names)

    @property
    def is_identity(self) -> bool:
        return len(self.names) == 0

    def to_actual(self, s):
        """正準時間軸 -> 実時間軸(予測で使う)。"""
        return np.interp(s, self.anchors_canonical, self.anchors_actual)

    def to_canonical(self, u):
        """実時間軸 -> 正準時間軸(学習で使う)。"""
        return np.interp(u, self.anchors_actual, self.anchors_canonical)

    def describe(self) -> str:
        if self.is_identity:
            return "位置合わせなし(恒等写像)"
        parts = [f"{nm}: {c:.3f}->{a:.3f}" for nm, c, a
                 in zip(self.names, self.anchors_canonical[1:-1], self.anchors_actual[1:-1])]
        return " / ".join(parts)


# ---------------------------------------------------------------------------
# 質量(工数)の移送
# ---------------------------------------------------------------------------
FINE_N = 4000  # 移送に使う細分グリッドの点数


def monthly_to_canonical(monthly: np.ndarray, edges: np.ndarray,
                         warp: Warp, n_bin: int) -> np.ndarray:
    """月次工数(実時間軸)を正準時間軸のビンに移送する。

    細かいグリッド上に工数を薄く撒き、各点をワープで正準軸へ写して数え直す。
    合計値は厳密に保存される。
    """
    fine_u = (np.arange(FINE_N) + 0.5) / FINE_N
    m_idx = np.clip(np.searchsorted(edges, fine_u, side="right") - 1, 0, len(monthly) - 1)
    counts = np.bincount(m_idx, minlength=len(monthly)).astype(float)
    counts[counts == 0] = 1.0
    mass = np.asarray(monthly, dtype=float)[m_idx] / counts[m_idx]

    s = warp.to_canonical(fine_u)
    c_idx = np.clip((s * n_bin).astype(int), 0, n_bin - 1)
    return np.bincount(c_idx, weights=mass, minlength=n_bin)


def canonical_to_monthly(shape: np.ndarray, edges: np.ndarray, warp: Warp) -> np.ndarray:
    """正準時間軸のカーブを実時間軸の月次に貼り付ける(設計書 6 Step3・Step4)。

    monthly_to_canonical と逆向きの、同じ移送処理。
    """
    n_bin = len(shape)
    n_month = len(edges) - 1
    fine_s = (np.arange(FINE_N) + 0.5) / FINE_N
    c_idx = np.clip((fine_s * n_bin).astype(int), 0, n_bin - 1)
    counts = np.bincount(c_idx, minlength=n_bin).astype(float)
    counts[counts == 0] = 1.0
    mass = np.asarray(shape, dtype=float)[c_idx] / counts[c_idx]

    u = warp.to_actual(fine_s)
    m_idx = np.clip(np.searchsorted(edges, u, side="right") - 1, 0, n_month - 1)
    return np.bincount(m_idx, weights=mass, minlength=n_month)
