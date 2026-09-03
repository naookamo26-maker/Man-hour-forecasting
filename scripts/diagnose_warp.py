"""マイルストーン位置合わせによる「工数の跳ね」の診断。

位置合わせを入れると月次工数に大きな段差が出ることがある。
原因はほぼ必ず「特定の案件のマイルストーン位置が、学習データの平均から離れていること」で、
どの案件・どのマイルストーンが効いているかは出力Excelを眺めても分からない。
このスクリプトはそれを案件ごとの数字にする。

    python scripts/diagnose_warp.py
    python scripts/diagnose_warp.py --strengths 1.0 0.7 0.5 --max-stretch 1.5

見方
    段差比        隣り合う区間で時間軸の伸縮率が何倍変わるか。
                  これがそのまま、その境目の月に出る工数の跳ねの倍率になる。
                  1.0 = 跳ねなし。3 を超えたら月次グラフに明らかな段差が出る。
    密度倍率      最も圧縮された区間で、学習カーブの山が縦に何倍に伸びるか。
    跳ね度        leave-one-out 予測の月次変化幅 ÷ 実績の月次変化幅。
                  1 を大きく超えていれば、実績には無い跳ねを作っている。
    WAPE          月次の予測誤差。位置合わせが効いているかどうかの本体。

跳ねている案件を特定したら、対処は3つ。
    1. その案件のマイルストーン日付が実態と合っているか確認する(入力ミスが一番多い)
    2. settings の 位置合わせ強度 を下げる(1.0 → 0.7 など)。全案件に効く
    3. settings の 伸縮率上限 を設定する(例 1.5)。跳ねの高さに直接上限をかける
2・3 はマイルストーンちょうどに山を合わせるのを諦める代わりに跳ねを抑える取引なので、
このスクリプトが出す WAPE を見て、精度が悪化しないことを確かめてから採用すること。
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_loader import load_all
from src.forecast import forecast
from src.learning import aggregate_actuals, build_project_curves, group_names, learn

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="位置合わせによる工数の跳ねを診断する")
    p.add_argument("--master", default=os.path.join(ROOT, "data", "master.xlsx"))
    p.add_argument("--actuals", default=os.path.join(ROOT, "data", "actuals.csv"))
    p.add_argument("--group-col", default=None, help="集約軸(既定は settings の値)")
    p.add_argument("--backbone-coverage", type=float, default=None)
    p.add_argument("--strengths", type=float, nargs="+", default=[1.0, 0.7, 0.5, 0.0],
                   help="比較する 位置合わせ強度 の一覧")
    p.add_argument("--max-stretch", type=float, default=None,
                   help="伸縮率上限も併せて試す場合に指定(例 1.5)")
    p.add_argument("--top", type=int, default=10, help="表示する案件数")
    return p.parse_args(argv)


def _run_one(ds, curves, groups, *, strength, max_stretch, n_bin, backbone_spec,
             coverage, hpm) -> pd.DataFrame:
    """leave-one-out で全案件を予測し、ワープの急峻さと跳ねを案件ごとに測る。"""
    rows = []
    for held in sorted(curves):
        rest = {p: c for p, c in curves.items() if p != held}
        if not rest:
            continue
        model = learn(rest, groups, align=True, n_bin=n_bin,
                      backbone_spec=backbone_spec, backbone_coverage=coverage,
                      hours_per_mm=hpm, warp_strength=strength, max_stretch=max_stretch)
        fc = forecast(model, ds, held, use_given_milestones=True,
                      months_override=curves[held].months)
        act = curves[held].monthly
        total = float(act.to_numpy().sum())
        pred = fc.table * (total / fc.table.to_numpy().sum())
        pm = pred.sum(axis=1).to_numpy()
        am = act.sum(axis=1).to_numpy()
        # 月次の最大変化幅どうしを比べる。跳ねは「隣の月との差」に一番はっきり出る。
        step_a = float(np.abs(np.diff(am)).max()) if len(am) > 1 else 0.0
        step_p = float(np.abs(np.diff(pm)).max()) if len(pm) > 1 else 0.0
        rows.append({
            "案件ID": held,
            "名称": curves[held].name[:16],
            "期間(月)": len(curves[held].months),
            "実績人月": round(total / hpm),
            "MS数": len(curves[held].ms_t),
            "段差比": round(fc.warp.max_step_ratio, 2),
            "密度倍率": round(fc.warp.max_density_gain, 2),
            "補正MS": ", ".join(fc.warp.clipped),
            "予測ピーク倍率": round(float(pm.max() / am.max()), 2) if am.max() else np.nan,
            "跳ね度": round(step_p / step_a, 2) if step_a else np.nan,
            "WAPE": round(float(np.abs(pm - am).sum() / total), 3),
        })
    return pd.DataFrame(rows)


def main(argv=None) -> int:
    a = parse_args(argv)
    overrides = {}
    if a.group_col:
        overrides["集約軸"] = a.group_col
    if a.backbone_coverage is not None:
        overrides["背骨最小カバー率"] = a.backbone_coverage

    ds = load_all(a.master, a.actuals, overrides=overrides)
    hpm = ds.hours_per_mm
    n_bin = int(ds.settings["カーブ解像度"])
    backbone_spec = str(ds.settings["背骨マイルストーン"])
    coverage = float(ds.settings["背骨最小カバー率"])

    agg = aggregate_actuals(ds)
    groups = group_names(agg, ds.phase_map, ds.group_col)
    curves = build_project_curves(ds, agg, groups)

    probe = learn(curves, groups, align=True, n_bin=n_bin, backbone_spec=backbone_spec,
                  backbone_coverage=coverage, hours_per_mm=hpm)
    print()
    print("=" * 78)
    print(f"学習案件 {len(curves)} 件 / 背骨: {', '.join(probe.backbone) or '(なし)'}")
    if not probe.backbone:
        print("背骨マイルストーンが選ばれていないため、位置合わせは働いていません。")
        print("背骨最小カバー率 を下げるか、背骨マイルストーン を明示指定してください。")
        return 0
    print("正準位置(= そのマイルストーンを持つ案件だけの平均位置)と、その母数:")
    for nm in probe.backbone:
        row = probe.ms_stats[probe.ms_stats["マイルストーン名"] == nm].iloc[0]
        sd = row["標準偏差"]
        print(f"    {nm:<16} 平均位置 {probe.canonical_anchor[nm]:.3f} "
              f"/ 標準偏差 {sd:.3f} / 記入 {int(row['件数'])} 件 "
              f"({int(row['件数']) / len(curves):.0%})")
    print("  ※ 記入件数が少ないほど平均位置そのものが不安定で、")
    print("     個々の案件をそこへ合わせる伸縮が極端になりやすい。")

    combos = [(s, None) for s in a.strengths]
    if a.max_stretch:
        combos += [(s, a.max_stretch) for s in a.strengths if s > 0]

    summary = []
    first = None
    for strength, cap in combos:
        d = _run_one(ds, curves, groups, strength=strength, max_stretch=cap,
                     n_bin=n_bin, backbone_spec=backbone_spec, coverage=coverage, hpm=hpm)
        label = f"強度 {strength:g}" + (f" / 上限 {cap:g}倍" if cap else "")
        if first is None:
            first = (label, d)
        summary.append({
            "設定": label,
            "段差比_最悪": d["段差比"].max(),
            "跳ね度_平均": round(d["跳ね度"].mean(), 2),
            "跳ね度_最悪": d["跳ね度"].max(),
            "WAPE_平均": round(d["WAPE"].mean(), 3),
            "WAPE_最悪": round(d["WAPE"].max(), 3),
        })

    label, d = first
    print()
    print("=" * 78)
    print(f"案件別(設定: {label})  段差比の大きい順")
    print("=" * 78)
    cols = ["案件ID", "名称", "期間(月)", "実績人月", "MS数", "段差比", "密度倍率",
            "補正MS", "予測ピーク倍率", "跳ね度", "WAPE"]
    print(d.sort_values("段差比", ascending=False).head(a.top)[cols].to_string(index=False))

    worst = d.sort_values("段差比", ascending=False).iloc[0]
    if worst["段差比"] >= 2.0:
        print()
        print(f"→ 跳ねの主因は {worst['案件ID']}({worst['名称']})。"
              f"アンカーの前後で時間軸の伸縮率が {worst['段差比']:.1f} 倍変わっており、")
        print(f"   その境目の月で工数が同じ倍率で跳ねる。"
              f"この案件のマイルストーン日付を消すと跳ねが収まるのはこのため。")

    print()
    print("=" * 78)
    print("設定別のまとめ(leave-one-out 全案件)")
    print("=" * 78)
    print(pd.DataFrame(summary).to_string(index=False))
    print()
    print("WAPE を悪化させずに 跳ね度・段差比 が下がる設定があれば、")
    print("それを settings の 位置合わせ強度 / 伸縮率上限 に書き込むこと。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
