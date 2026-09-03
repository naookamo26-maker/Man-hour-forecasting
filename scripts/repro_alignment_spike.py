"""マイルストーン位置合わせで「大きすぎる工数の跳ね」が出る現象の最小再現。

実データで観測された状況を合成データで再現する。

    完了案件 30 件 / マイルストーン記入 22 行 / 主なマイルストーン 3 種
    うち1件(PJ-BIG)だけ様子が違う
        - 期間が最長(36ヶ月)で工数も最大(3000人月)
        - α版までの期間が長く、その直前に工数を大きく消費
        - α版〜β版の間は低調で、β版の直前から再び活発

このスクリプトは PJ-BIG 自身を leave-one-out で予測し、
    (1) マイルストーンあり  → 月次に大きな跳ねが出る
    (2) PJ-BIG のマイルストーンを消す → 跳ねが収まる
を並べて出す。跳ねの倍率は、その案件のワープの「アンカー段差」と一致する。

    python scripts/repro_alignment_spike.py
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_loader import Dataset
from src.forecast import forecast
from src.learning import aggregate_actuals, build_project_curves, group_names, learn
from src.timeaxis import t_to_date

HPM = 160
GROUPS = ["開発", "QA"]
PHASE_MAP = pd.DataFrame({"実績行程名": ["実装", "デバッグ"],
                          "行程グループ": GROUPS, "大分類": ["制作", "検証"]})
SETTINGS = {"人月換算係数": HPM, "集約軸": "行程グループ", "位置合わせ": "ON",
            "背骨マイルストーン": "自動", "背骨最小カバー率": 0.2,
            "カーブ解像度": 100, "カーブ復元": "月内均等",
            "マイルストーン最小件数": 3, "行程グループ最小件数": 3,
            "位置合わせ強度": 1.0, "伸縮率上限": 0.0, "k": 3.0, "タグ重み係数": 0.5}
MS_NAMES = {"α版": 0.30, "β版": 0.65, "マスターアップ": 0.93}


def _bump(u, center, width, height):
    return height * np.exp(-0.5 * ((u - center) / width) ** 2)


def density_typical(u):
    """典型案件: α版(0.30)前に軽い山、β版(0.65)前に山、終盤にQAの山。"""
    return (0.6 + _bump(u, 0.28, 0.05, 1.0) + _bump(u, 0.62, 0.06, 1.4)
            + _bump(u, 0.90, 0.05, 1.2))


def density_outlier(u):
    """異質案件: α版(0.55)が遅く、その直前に巨大な山。
    α版〜β版(0.72)は低調で、β版の直前から再び活発になる。"""
    return (0.25 + _bump(u, 0.50, 0.055, 3.0) + _bump(u, 0.70, 0.05, 1.8)
            + _bump(u, 0.88, 0.06, 1.5))


def build_dataset(rng: np.random.Generator):
    proj, ms, act = [], [], []

    def add(pid, n_month, mm, density, ms_t, start):
        months = [str(pd.Period(start, freq="M") + i) for i in range(n_month)]
        starts = [pd.Period(m, freq="M").start_time for m in months]
        end = pd.Period(months[-1], freq="M").end_time
        days = np.array([(d - starts[0]).days for d in starts] + [(end - starts[0]).days],
                        dtype=float)
        edges = days / days[-1]
        mid, width = (edges[:-1] + edges[1:]) / 2, np.diff(edges)
        base = density(mid) * width
        for g, share in (("開発", 0.7), ("QA", 0.3)):
            d = base * (mid ** 1.8 if g == "QA" else 1.0)   # QA は後ろ寄せ
            d = d / d.sum()
            for m, v in zip(months, d * mm * HPM * share):
                act.append({"案件ID": pid, "行程": "実装" if g == "開発" else "デバッグ",
                            "月": m, "時間": float(v)})
        proj.append({"案件ID": pid, "名称": pid, "種別": "新規", "契約人月": mm,
                     "開始": months[0], "終了": months[-1], "タグ": "", "ステータス": "完了"})
        for nm, t in ms_t.items():
            ms.append({"案件ID": pid, "マイルストーン名": nm,
                       "日付": t_to_date(months, t).strftime("%Y-%m-%d")})

    # 通常案件 29 件。マイルストーンを記入するのは先頭 7 件だけ(= 21 行)。
    for i in range(29):
        jitter = rng.normal(0, 0.03, 3)
        ms_t = ({nm: t + j for (nm, t), j in zip(MS_NAMES.items(), jitter)}
                if i < 7 else {})
        add(f"PJ-{i:02d}", int(rng.integers(12, 30)), float(rng.integers(200, 900)),
            density_typical, ms_t, f"{2016 + i % 6}-{1 + i % 9:02d}")

    # 異質な大型長期案件(これで計 24 行、実データの 22 行とほぼ同じ密度)
    add("PJ-BIG", 36, 3000.0, density_outlier,
        {"α版": 0.55, "β版": 0.72, "マスターアップ": 0.93}, "2017-01")

    return (pd.DataFrame(proj), pd.DataFrame(ms), pd.DataFrame(act))


def loo(projects, milestones, actuals, pid, label, *, strength=1.0, max_stretch=None):
    """pid を学習から外して予測し、実績と並べる。"""
    prj = projects.copy()
    prj.loc[prj["案件ID"] == pid, "ステータス"] = "予測対象"
    ds = Dataset(projects=prj, milestones=milestones, phase_map=PHASE_MAP,
                 settings=dict(SETTINGS), actuals=actuals)
    agg = aggregate_actuals(ds)
    groups = group_names(agg, PHASE_MAP, "行程グループ")
    curves = build_project_curves(ds, agg, groups)
    model = learn(curves, groups, align=True, n_bin=100, backbone_spec="自動",
                  backbone_coverage=0.2, hours_per_mm=HPM,
                  warp_strength=strength, max_stretch=max_stretch)
    fc = forecast(model, ds, pid)
    pred = fc.table.sum(axis=1).to_numpy() / HPM
    act = (actuals[actuals["案件ID"] == pid].groupby("月")["時間"].sum()
           .reindex(fc.months).fillna(0.0).to_numpy() / HPM)
    pred = pred * (act.sum() / pred.sum())      # 形だけを比べる

    print(f"\n----- {label} -----")
    print(f"  正準位置: " + " / ".join(f"{k} {v:.3f}" for k, v in model.canonical_anchor.items()))
    print(f"  ワープ:   {fc.warp.describe()}")
    print("  予測 " + " ".join(f"{v:5.1f}" for v in pred))
    print("  実績 " + " ".join(f"{v:5.1f}" for v in act))
    print(f"  月次の最大変化幅  予測 {np.abs(np.diff(pred)).max():5.1f} 人月 / "
          f"実績 {np.abs(np.diff(act)).max():5.1f} 人月 "
          f"→ 跳ね度 {np.abs(np.diff(pred)).max() / np.abs(np.diff(act)).max():.2f}")
    print(f"  WAPE {np.abs(pred - act).sum() / act.sum():.3f}")


def main() -> int:
    rng = np.random.default_rng(7)
    projects, milestones, actuals = build_dataset(rng)
    print(f"完了案件 {len(projects)} 件 / マイルストーン {len(milestones)} 行 / "
          f"種類 {milestones['マイルストーン名'].nunique()} 個")
    print("PJ-BIG: 36ヶ月 / 3000人月 / α版が u=0.55(平均は約0.30)と極端に遅い")

    loo(projects, milestones, actuals, "PJ-BIG", "現状: マイルストーンあり(跳ねる)")
    loo(projects, milestones[milestones["案件ID"] != "PJ-BIG"], actuals, "PJ-BIG",
        "PJ-BIG のマイルストーンを削除(跳ねが収まる = 位置合わせが効かなくなる)")
    loo(projects, milestones, actuals, "PJ-BIG",
        "対策: 位置合わせ強度 0.5", strength=0.5)
    loo(projects, milestones, actuals, "PJ-BIG",
        "対策: 伸縮率上限 1.5 倍", max_stretch=1.5)
    print("\nマイルストーンを消すと跳ねは消えるが、位置合わせの効果も丸ごと失われる。")
    print("強度・上限は、位置合わせを残したまま跳ねだけを抑えるための調整口。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
