"""
立ち上がり制約 ─ 月次工数が現実にはあり得ない速さで変化するのを抑える

段階予測で前半の消化が低調だった場合、辻褄を合わせるために残り区間へ工数が
上乗せされる。このとき現在の方式は残りのカーブを一律に縦へ伸ばすだけなので、
確定区間の直後にいきなり工数が跳ね、最初の月が最大値になることがある。

現実にはそうならない。人員は突然数倍にはならず、増やせてもハンドリングが
追いつかない。実際に起きるのは「山が後ろへ移動して、なだらかになる」であり、
消化する総量そのものは変わらない。

そこで月次工数の増加率に上限を置く。上限は恣意的な定数ではなく、
実績データの前月比の分布から決める(observed_growth_limit)。

── 総量とどう両立させるか

「上限を守る」と「合計を残工数にちょうど合わせる」は同時に満たせる。

    x[j] = min(f[j] * S, limit * x[j-1])

は S について単調増加なので、合計が目標に一致する S を二分探索できる
(ramp_monthly_totals)。上限いっぱいに立ち上げたときの消化容量は limit^m で
指数的に増えるため、実測では全 345 段階で実行可能だった。
つまり「制約を緩めて無理に収める」も「あふれを警告する」も要らない。

── 行程グループ別の総量も同時に守る

月次合計に上限をかけると、行程グループ別の総量が動いてしまう。
そこで月次合計とグループ別総量の両方を周辺分布として与え、
学習カーブの形に最も近い配分を求める(fit_to_margins)。
反復比例調整(IPF)で、どちらの合計も厳密に一致する。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

LIMIT_AUTO = "自動"
GROWTH_QUANTILE = 0.90   # 実績の前月比のこの分位点を上限にする
LIMIT_MIN, LIMIT_MAX = 1.05, 5.0


def observed_growth_limit(monthlies, q: float = GROWTH_QUANTILE) -> float | None:
    """実績の月次工数の前月比から、現実に起きうる増加率の上限を決める。

    monthlies は案件ごとの月次合計(1次元配列)の列。
    定数を決め打ちにすると、その組織の実態と合っているかを誰も確かめられない。
    実績から出せば「この職場では月あたり何倍までなら実際に起きているか」になる。
    """
    ratios = []
    for m in monthlies:
        m = np.asarray(m, dtype=float)
        if len(m) < 2:
            continue
        prev = m[:-1]
        r = np.divide(m[1:], prev, out=np.full(len(prev), np.nan), where=prev > 0)
        ratios.append(r[np.isfinite(r)])
    if not ratios:
        return None
    all_r = np.concatenate(ratios)
    if len(all_r) < 10:
        # 標本が少なすぎる分位点は上限として信用できない。制約なしに倒す。
        return None
    return float(np.clip(np.quantile(all_r, q), LIMIT_MIN, LIMIT_MAX))


def ramp_monthly_totals(base: float, weights: np.ndarray, total: float,
                        limit: float) -> np.ndarray | None:
    """増加率の上限を守りつつ、合計が total にちょうど一致する月次配分を返す。

    base    立ち上がりの起点。直前の月の実績工数
    weights 学習カーブの月次配分(合計1)。上限に当たらない月はこの形に従う
    limit   前月比の上限

    上限に当たった月からあふれた分は後ろの月へ回る。その結果として
    山が後ろへ移動し、なだらかになる。これが再現したい挙動そのものである。
    """
    weights = np.asarray(weights, dtype=float)
    n = len(weights)
    if n == 0 or total <= 0 or base <= 0 or limit <= 1.0:
        return None

    def build(scale: float) -> np.ndarray:
        x = np.empty(n)
        prev = base
        for j in range(n):
            prev = x[j] = min(weights[j] * scale, limit * prev)
        return x

    # 上限いっぱい(base * limit^j)でも足りなければ、この期間では消化しきれない。
    ceiling = base * np.cumprod(np.full(n, limit))
    if ceiling.sum() < total - 1e-9:
        return None

    hi = 1.0
    for _ in range(400):
        if build(hi).sum() >= total:
            break
        hi *= 2.0
    else:
        return None
    lo = 0.0
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if build(mid).sum() < total:
            lo = mid
        else:
            hi = mid
    return build(hi)


def fit_to_margins(seed: np.ndarray, row_totals: np.ndarray,
                   col_totals: np.ndarray, iters: int = 500) -> np.ndarray:
    """行(月)合計と列(行程グループ)合計の両方に一致する表を、種の形に近い形で作る。

    反復比例調整(IPF)。行方向と列方向のスケーリングを交互に繰り返すと、
    与えた周辺分布を持つ表のうち、種の形から最も離れていないものへ収束する。
    ここでの種は学習カーブなので、「学習した形をできるだけ保ったまま、
    月次の上限とグループ別総量の両方を満たす」配分になる。
    """
    X = np.array(seed, dtype=float)
    X[~np.isfinite(X)] = 0.0
    X[X < 0] = 0.0
    if X.sum() <= 0:
        X = np.ones_like(X)
    # 0 のセルはスケーリングでは二度と正にならない。行または列の合計が
    # 正なのに全セルが 0 だと解が作れないため、微小値を敷いて逃げ道を残す。
    X = X + max(X.sum(), 1.0) * 1e-12

    row_totals = np.asarray(row_totals, dtype=float)
    col_totals = np.asarray(col_totals, dtype=float)
    for _ in range(iters):
        r = X.sum(axis=1)
        X *= np.divide(row_totals, r, out=np.zeros_like(r), where=r > 0)[:, None]
        c = X.sum(axis=0)
        X *= np.divide(col_totals, c, out=np.zeros_like(c), where=c > 0)[None, :]
        if np.abs(X.sum(axis=1) - row_totals).max() <= max(row_totals.sum(), 1.0) * 1e-12:
            break
    return X


def apply_ramp(table: pd.DataFrame, base: float, limit: float) -> tuple[pd.DataFrame, dict]:
    """月 × 行程グループ の表に立ち上がり制約をかける。

    グループ別の総量と全体の総量はどちらも動かさない。動かすのは
    「いつ消化するか」だけである。制約に当たらなければ表はそのまま返る。

    戻り値: (調整後の表, 診断情報)
    """
    info: dict = {"適用": False, "上限": limit, "起点": base}
    if table.empty or len(table) < 2 or limit is None or limit <= 1.0 or base <= 0:
        info["理由"] = "制約をかける条件を満たさない(期間が短い/起点が0)"
        return table, info

    arr = table.to_numpy(dtype=float)
    total = float(arr.sum())
    monthly = arr.sum(axis=1)
    if total <= 0 or monthly.sum() <= 0:
        info["理由"] = "予測工数が0"
        return table, info

    before_step = float(monthly[0] / base)
    targets = ramp_monthly_totals(base, monthly / total, total, limit)
    if targets is None:
        # 上限を守ると期間内に消化しきれない。総量を動かさない約束のほうが優先なので
        # ここは制約を諦める。黙って諦めるとグラフの跳ねの理由が追えなくなるため記録する。
        info["理由"] = (f"上限 {limit:.2f} 倍/月では残り {len(monthly)} ヶ月に "
                        f"{total:,.0f} 時間を消化しきれないため、制約をかけていない")
        info["収まらない"] = True
        return table, info

    if np.abs(targets - monthly).max() <= max(total, 1.0) * 1e-9:
        info["理由"] = "学習カーブが元から上限に収まっている"
        return table, info

    fitted = fit_to_margins(arr, targets, arr.sum(axis=0))
    out = pd.DataFrame(fitted, index=table.index, columns=table.columns)

    ratios = targets[1:] / np.where(targets[:-1] > 0, targets[:-1], np.nan)
    info.update({
        "適用": True,
        "制約に当たった月数": int(np.sum(targets < monthly - max(total, 1.0) * 1e-9)),
        "境界段差 前": before_step,
        "境界段差 後": float(targets[0] / base),
        "ピーク位置 前": int(np.argmax(monthly)),
        "ピーク位置 後": int(np.argmax(targets)),
        "最大前月比 後": float(np.nanmax(ratios)) if len(ratios) else 1.0,
    })
    return out, info
