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

RECON_BOX = "月内均等"      # 従来方式
RECON_SMOOTH = "単調補間"   # 累積を単調補間して微分する方式
RECON_MODES = (RECON_BOX, RECON_SMOOTH)


def _pchip_tangents(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """単調保存の3次補間(Fritsch-Carlson)の各節点における傾き。

    scipy.interpolate.PchipInterpolator と同じ式。
    scipyを依存に加えないため、必要な部分だけ numpy で持つ。
    """
    h = np.diff(x)
    delta = np.diff(y) / h
    d = np.zeros_like(y, dtype=float)

    if len(x) == 2:
        d[:] = delta[0]
        return d

    # 内部の節点: 隣り合う傾きの符号が違えば 0(=行き過ぎを作らない)、
    # 同符号なら区間長で重み付けした調和平均。
    m0, m1 = delta[:-1], delta[1:]
    h0, h1 = h[:-1], h[1:]
    same = np.sign(m0) * np.sign(m1) > 0
    w1, w2 = 2 * h1 + h0, h1 + 2 * h0
    with np.errstate(divide="ignore", invalid="ignore"):
        harmonic = (w1 + w2) / (w1 / np.where(m0 == 0, np.nan, m0)
                                + w2 / np.where(m1 == 0, np.nan, m1))
    d[1:-1] = np.where(same, np.nan_to_num(harmonic), 0.0)

    # 端点は片側3点式。母数の傾きと符号が違えば 0、行き過ぎなら 3倍で頭打ち。
    def _edge(h0_, h1_, m0_, m1_):
        v = ((2 * h0_ + h1_) * m0_ - h0_ * m1_) / (h0_ + h1_)
        if np.sign(v) != np.sign(m0_):
            return 0.0
        if np.sign(m0_) != np.sign(m1_) and abs(v) > 3 * abs(m0_):
            return 3 * m0_
        return float(v)

    d[0] = _edge(h[0], h[1], delta[0], delta[1])
    d[-1] = _edge(h[-1], h[-2], delta[-1], delta[-2])
    return d


def _pchip_eval(x: np.ndarray, y: np.ndarray, xi: np.ndarray) -> np.ndarray:
    """単調保存3次補間の評価。x は昇順、xi は x の範囲内。"""
    d = _pchip_tangents(x, y)
    k = np.clip(np.searchsorted(x, xi, side="right") - 1, 0, len(x) - 2)
    h = (x[k + 1] - x[k])
    t = (xi - x[k]) / h
    t2, t3 = t * t, t * t * t
    # エルミート基底
    h00 = 2 * t3 - 3 * t2 + 1
    h10 = t3 - 2 * t2 + t
    h01 = -2 * t3 + 3 * t2
    h11 = t3 - t2
    return h00 * y[k] + h10 * h * d[k] + h01 * y[k + 1] + h11 * h * d[k + 1]


def monthly_to_canonical(monthly: np.ndarray, edges: np.ndarray,
                         warp: Warp, n_bin: int,
                         recon: str = RECON_BOX) -> np.ndarray:
    """月次工数(実時間軸)を正準時間軸のビンに移送する。

    recon が決めるのは「月次の値から、月の内側の分布をどう復元するか」。

      月内均等  月次工数を月の中に一様に撒く。実装は素直だが、
                幅 1/月数 の箱型ぼかしを全案件に掛けるのと等価で、
                期間の短い案件ほど山がなまる(4ヶ月案件は真の山の約5割)。
      単調補間  月境界での累積値だけは厳密に既知であることを使う。
                累積を単調保存3次補間し、正準ビンの境界での差をとる。
                月内均等の仮定を外すぶん山が保たれ、長期案件でも改善する。

    どちらも合計値は厳密に保存される。
    """
    monthly = np.asarray(monthly, dtype=float)
    if recon == RECON_SMOOTH and len(monthly) >= 3:
        total = float(monthly.sum())
        if total <= 0:
            return np.zeros(n_bin)
        cum = np.concatenate([[0.0], np.cumsum(monthly)])
        s_edges = np.linspace(0.0, 1.0, n_bin + 1)
        u_edges = warp.to_actual(s_edges)          # 正準ビンの境界を実時間軸へ戻す
        out = np.diff(_pchip_eval(np.asarray(edges, dtype=float), cum, u_edges))
        out = np.clip(out, 0.0, None)
        s = out.sum()
        return out * (total / s) if s > 0 else np.zeros(n_bin)

    fine_u = (np.arange(FINE_N) + 0.5) / FINE_N
    m_idx = np.clip(np.searchsorted(edges, fine_u, side="right") - 1, 0, len(monthly) - 1)
    counts = np.bincount(m_idx, minlength=len(monthly)).astype(float)
    counts[counts == 0] = 1.0
    mass = monthly[m_idx] / counts[m_idx]

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
