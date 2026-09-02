"""
検証(設計書 8章)

leave-one-out ─ 1件を除いた9件で学習し、除いた1件を予測して実績と比較する。

比較するモードは3つ。
  素朴版              位置合わせ OFF。全案件のカーブを単純平均して期間に貼るだけ
  位置合わせ(MS学習値)  位置合わせ ON。ただし対象案件のマイルストーンは学習した平均位置を使う
  位置合わせ(MS指定)    位置合わせ ON。対象案件のマイルストーン日付を既知として与える

「素朴版 vs 位置合わせ(MS学習値)」は
  平均カーブを鋭くしただけで当たるようになるか を測る。
「位置合わせ(MS学習値) vs 位置合わせ(MS指定)」は
  マイルストーンを手入力する価値がどれだけあるか を測る。
設計書 5 Step3 の「入力の手間を正当化する根拠」がここで数字になる。

工数の「量」ではなく「形」を評価するため、
予測値は実績合計に合わせて正規化してから比較する(設計書 1章 前提)。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .forecast import forecast
from .learning import ProjectCurve, learn

MODES = [
    ("素朴版", False, False),
    ("位置合わせ(MS学習値)", True, False),
    ("位置合わせ(MS指定)", True, True),
]


def _metrics(pred: pd.DataFrame, act: pd.DataFrame) -> dict:
    """予測表と実績表(同じ index/columns)から誤差指標を出す。"""
    p = pred.to_numpy(dtype=float)
    a = act.to_numpy(dtype=float)
    total = a.sum()

    p_m, a_m = p.sum(axis=1), a.sum(axis=1)
    wape_month = np.abs(p_m - a_m).sum() / total
    wape_cell = np.abs(p - a).sum() / total

    fp = np.cumsum(p_m) / p_m.sum()
    fa = np.cumsum(a_m) / a_m.sum()
    ks = float(np.abs(fp - fa).max())

    peak_gap = int(np.argmax(p_m) - np.argmax(a_m))
    peak_err = float(p_m.max() / a_m.max() - 1.0)

    return {
        "月次誤差WAPE": round(float(wape_month), 4),
        "月×グループ誤差WAPE": round(float(wape_cell), 4),
        "累積カーブ最大乖離": round(ks, 4),
        "ピーク月ズレ": peak_gap,
        "ピーク高さ誤差": round(peak_err, 4),
    }


def leave_one_out(ds, curves: dict[str, ProjectCurve], groups: list[str], *,
                  n_bin: int = 100, backbone_spec: str = "自動",
                  backbone_coverage: float = 0.6, hours_per_mm: float = 160.0,
                  modes=MODES) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """leave-one-out を実行する。

    戻り値: (案件別の誤差, 月次の予測vs実績, モード別サマリ)
    """
    detail_rows, monthly_rows = [], []
    pids = sorted(curves.keys())

    if len(pids) < 2:
        raise ValueError("leave-one-out には最低2案件の実績が必要です。")

    for held in pids:
        rest = {p: c for p, c in curves.items() if p != held}
        act = curves[held].monthly
        act_total = float(act.to_numpy().sum())

        for label, align, use_given in modes:
            model = learn(rest, groups, align=align, n_bin=n_bin,
                          backbone_spec=backbone_spec,
                          backbone_coverage=backbone_coverage,
                          hours_per_mm=hours_per_mm)
            fc = forecast(model, ds, held, use_given_milestones=use_given,
                          months_override=curves[held].months)

            # 形の比較に集中するため、量は実績合計に合わせる
            pred = fc.table * (act_total / fc.table.to_numpy().sum())

            m = _metrics(pred, act)
            detail_rows.append({
                "案件ID": held, "名称": curves[held].name,
                "種別": curves[held].ptype, "モード": label,
                "契約人月": curves[held].contract_mm,
                "期間(月)": len(curves[held].months),
                "学習件数": len(rest),
                **m,
            })

            pm, am = pred.sum(axis=1), act.sum(axis=1)
            cum_p = np.cumsum(pm.to_numpy()) / pm.sum()
            cum_a = np.cumsum(am.to_numpy()) / am.sum()
            for i, mon in enumerate(act.index):
                monthly_rows.append({
                    "案件ID": held, "モード": label, "月": mon, "月次番号": i + 1,
                    "予測(時間)": round(float(pm.iloc[i]), 1),
                    "実績(時間)": round(float(am.iloc[i]), 1),
                    "差分(時間)": round(float(pm.iloc[i] - am.iloc[i]), 1),
                    "予測累積率": round(float(cum_p[i]), 4),
                    "実績累積率": round(float(cum_a[i]), 4),
                })

    detail = pd.DataFrame(detail_rows)
    monthly = pd.DataFrame(monthly_rows)

    order = [m[0] for m in modes]
    summary = (detail.groupby("モード", observed=True)
               .agg(**{
                   "案件数": ("案件ID", "count"),
                   "月次誤差WAPE_平均": ("月次誤差WAPE", "mean"),
                   "月次誤差WAPE_最悪": ("月次誤差WAPE", "max"),
                   "月×グループ誤差WAPE_平均": ("月×グループ誤差WAPE", "mean"),
                   "累積カーブ最大乖離_平均": ("累積カーブ最大乖離", "mean"),
                   "ピーク月ズレ_絶対値平均": ("ピーク月ズレ", lambda s: np.abs(s).mean()),
               })
               .reindex(order).reset_index().round(4))

    base = summary.loc[summary["モード"] == order[0], "月次誤差WAPE_平均"]
    if not base.empty and base.iloc[0] > 0:
        summary["素朴版比_改善率"] = (1 - summary["月次誤差WAPE_平均"] / base.iloc[0]).round(4)

    return detail, monthly, summary
