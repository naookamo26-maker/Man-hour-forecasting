"""工数カーブの形ごとに、予測が当たっているかを測る。

実データの消化カーブは「中央に大きな山が1つ」ばかりではない。
前半が緩やかで後半に一気に消化する案件、早めに片付いて終盤が落ち着く案件がある。
平均の WAPE だけを見ていると、この差は見えない。

このスクリプトは leave-one-out を回し、案件を実績から測った形で分類して
種類ごとに誤差を出す。分類は名称や種別ではなく実績そのものから決めるので、
サンプルデータでも実データでも同じように使える。

    重心      Σ(位置 × 工数) / Σ工数。0.5 なら前後均等。
              0.45 未満 = 前倒し型 / 0.55 超 = 後半集中型 / 間 = 標準型
    山の数    月次工数の極大の数(平滑化後)。2 以上なら二山型

同じマイルストーンを複数回通過した案件(やり直しのあった案件)についても、
回数ごとに誤差を出し、「名前(初回)」を背骨に入れた場合と入れない場合を比べる。

    python scripts/validate_patterns.py
    python scripts/validate_patterns.py --master data_large/master.xlsx \
        --actuals data_large/actuals.csv
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_loader import FIRST_ATTEMPT_SUFFIX, load_all
from src.forecast import forecast
from src.learning import aggregate_actuals, build_project_curves, group_names, learn

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def classify(monthly: np.ndarray) -> tuple[str, float, int]:
    """月次工数の並びから、カーブの形を分類する。"""
    w = np.asarray(monthly, dtype=float)
    t = (np.arange(len(w)) + 0.5) / len(w)
    centroid = float((t * w).sum() / w.sum()) if w.sum() > 0 else 0.5

    # 山を数えるには、まず月次のノイズを落とす必要がある。
    # 月次には ±5% 程度のばらつきが常に乗っており、生の並びでは
    # 隣より高いだけの点がいくらでも見つかって山の数にならない。
    # 窓は期間に比例させる(36ヶ月案件と12ヶ月案件で同じ窓幅は使えない)。
    win = max(3, len(w) // 6 | 1)
    pad = win // 2
    k = np.convolve(np.pad(w, pad, mode="edge"), np.ones(win) / win, mode="valid")

    # 高さが最大の半分以上ある極大だけを候補にし、
    # さらに「間の谷が、両側の山の低いほうの 75% より深い」ものだけを別の山と数える。
    # この谷の条件が無いと、なだらかな肩がすべて山になってしまう。
    cand = [i for i in range(1, len(k) - 1)
            if k[i] > k[i - 1] and k[i] >= k[i + 1] and k[i] >= 0.5 * k.max()]
    peaks = 1 if cand else 0
    last = cand[0] if cand else 0
    for i in cand[1:]:
        valley = k[last:i + 1].min()
        if valley < 0.75 * min(k[last], k[i]):
            peaks += 1
            last = i

    if peaks >= 2:
        shape = "二山型"
    elif centroid < 0.45:
        shape = "前倒し型"
    elif centroid > 0.55:
        shape = "後半集中型"
    else:
        shape = "標準型"
    return shape, centroid, peaks


def loo(ds, curves, groups, *, align, backbone_spec, coverage, n_bin, hpm,
        warp_strength, max_stretch) -> dict[str, float]:
    """案件ID -> 月次WAPE。"""
    out = {}
    for held in sorted(curves):
        rest = {p: c for p, c in curves.items() if p != held}
        model = learn(rest, groups, align=align, n_bin=n_bin,
                      backbone_spec=backbone_spec, backbone_coverage=coverage,
                      hours_per_mm=hpm, warp_strength=warp_strength,
                      max_stretch=max_stretch)
        fc = forecast(model, ds, held, use_given_milestones=True,
                      months_override=curves[held].months)
        act = curves[held].monthly
        total = float(act.to_numpy().sum())
        pred = fc.table * (total / fc.table.to_numpy().sum())
        out[held] = float(np.abs(pred.sum(axis=1).to_numpy()
                                 - act.sum(axis=1).to_numpy()).sum() / total)
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="カーブの形ごとの予測精度を測る")
    ap.add_argument("--master", default=os.path.join(ROOT, "data", "master.xlsx"))
    ap.add_argument("--actuals", default=os.path.join(ROOT, "data", "actuals.csv"))
    ap.add_argument("--backbone-coverage", type=float, default=None)
    ap.add_argument("--warp-strength", type=float, default=None)
    ap.add_argument("--max-stretch", type=float, default=None)
    a = ap.parse_args(argv)

    overrides = {}
    if a.backbone_coverage is not None:
        overrides["背骨最小カバー率"] = a.backbone_coverage
    ds = load_all(a.master, a.actuals, overrides=overrides)
    hpm = ds.hours_per_mm
    n_bin = int(ds.settings["カーブ解像度"])
    coverage = float(ds.settings["背骨最小カバー率"])
    strength = a.warp_strength if a.warp_strength is not None else float(
        ds.settings["位置合わせ強度"])
    cap_raw = a.max_stretch if a.max_stretch is not None else float(ds.settings["伸縮率上限"])
    cap = cap_raw if cap_raw and cap_raw > 1.0 else None

    agg = aggregate_actuals(ds)
    groups = group_names(agg, ds.phase_map, ds.group_col)
    curves = build_project_curves(ds, agg, groups)

    shape = {pid: classify(c.monthly.sum(axis=1).to_numpy()) for pid, c in curves.items()}
    naive = loo(ds, curves, groups, align=False, backbone_spec="自動", coverage=coverage,
                n_bin=n_bin, hpm=hpm, warp_strength=strength, max_stretch=cap)
    aligned = loo(ds, curves, groups, align=True, backbone_spec="自動", coverage=coverage,
                  n_bin=n_bin, hpm=hpm, warp_strength=strength, max_stretch=cap)

    att = ds.ms_attempts
    tries = ({} if att.empty else
             att.groupby("案件ID")["回数"].max().to_dict())

    rows = []
    for pid, c in curves.items():
        sh, centroid, peaks = shape[pid]
        rows.append({
            "案件ID": pid, "形": sh, "重心": round(centroid, 3), "山": peaks,
            "期間": len(c.months), "実績人月": round(c.total_hours / hpm),
            "やり直し": int(tries.get(pid, 1)),
            "素朴版": naive[pid], "位置合わせ": aligned[pid],
        })
    d = pd.DataFrame(rows)
    d["改善"] = d["素朴版"] - d["位置合わせ"]

    print()
    print("=" * 74)
    print(f"leave-one-out  学習案件 {len(curves)} 件 / 月次誤差WAPE(小さいほど良い)")
    print("=" * 74)

    def _agg(by: str):
        g = (d.groupby(by)
             .agg(件数=("案件ID", "count"),
                  素朴版=("素朴版", "mean"),
                  位置合わせ=("位置合わせ", "mean"),
                  改善=("改善", "mean"),
                  最悪=("位置合わせ", "max"))
             .round(3))
        return g.sort_values("件数", ascending=False)

    print("\n■ カーブの形ごと")
    print(_agg("形").to_string())
    print("\n■ 期間の長さごと")
    d["期間帯"] = pd.cut(d["期間"], [0, 15, 24, 99],
                        labels=["〜15ヶ月", "16〜24ヶ月", "25ヶ月〜"])
    print(_agg("期間帯").to_string())

    if not att.empty:
        print("\n■ 同じマイルストーンのやり直し回数ごと")
        print(_agg("やり直し").to_string())

        # 「(初回)」を背骨に入れると、手戻り期間を時間軸として表現できる。
        # 実際に効くかどうかはデータ次第なので、必ず数字で確かめる。
        names = sorted({nm for c in curves.values() for nm in c.ms_t})
        with_first = ";".join(names)
        without = ";".join(n for n in names if not n.endswith(FIRST_ATTEMPT_SUFFIX))
        w1 = loo(ds, curves, groups, align=True, backbone_spec=without,
                 coverage=coverage, n_bin=n_bin, hpm=hpm,
                 warp_strength=strength, max_stretch=cap)
        w2 = loo(ds, curves, groups, align=True, backbone_spec=with_first,
                 coverage=coverage, n_bin=n_bin, hpm=hpm,
                 warp_strength=strength, max_stretch=cap)
        sub = [p for p in curves if tries.get(p, 1) > 1]
        print(f"\n■ 「{FIRST_ATTEMPT_SUFFIX}」を背骨に入れる効果"
              f"(やり直しのあった {len(sub)} 件)")
        print(f"    入れない : {np.mean([w1[p] for p in sub]):.3f}")
        print(f"    入れる   : {np.mean([w2[p] for p in sub]):.3f}")
        print(f"  全案件 {len(curves)} 件")
        print(f"    入れない : {np.mean(list(w1.values())):.3f}")
        print(f"    入れる   : {np.mean(list(w2.values())):.3f}")

    print("\n■ 位置合わせで悪化した案件(上位5件)")
    print(d.nsmallest(5, "改善")[
        ["案件ID", "形", "重心", "期間", "実績人月", "やり直し", "素朴版", "位置合わせ", "改善"]
    ].round(3).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
