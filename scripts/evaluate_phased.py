"""段階予測が「情報を足すほど当たるようになるか」を全完了案件で測る。

Excel の 段階予測 シートは1案件を見るためのもので、
その1件が良かったか悪かったかは運でも決まる。
仕組みとして機能しているかは、全案件を横に並べないと分からない。

このスクリプトは完了案件すべてに段階予測を掛け、

    段階0(着手前)から段階が進むにつれて、残り区間の誤差が下がるか
    残工数の決め方(契約総量固定 / 実績スケール)でどちらが当たるか

を数字にする。段階の番号は案件ごとにマイルストーン数が違って揃わないので、
「確定した期間の割合」でも束ねて出す。

    python scripts/evaluate_phased.py
    python scripts/evaluate_phased.py --align off
    python scripts/evaluate_phased.py --master data_large/master.xlsx \
        --actuals data_large/actuals.csv
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_loader import load_all
from src.learning import aggregate_actuals, build_project_curves, group_names
from src.phased import BASIS_MODES, phased_forecast

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="段階予測の精度を全完了案件で評価する")
    p.add_argument("--master", default=os.path.join(ROOT, "data", "master.xlsx"))
    p.add_argument("--actuals", default=os.path.join(ROOT, "data", "actuals.csv"))
    p.add_argument("--align", choices=["on", "off"], default=None)
    p.add_argument("--basis", choices=["fixed", "scaled", "both"], default="both",
                   help="残工数の決め方。既定は両方を並べる")
    p.add_argument("--ramp-limit", default=None,
                   help="立ち上がり上限。自動(既定) / 数値 / off。"
                        "compare を指定すると 制約あり・なし を並べて比較する")
    p.add_argument("--csv", default=None, help="案件×段階の明細をCSVに書き出す")
    return p.parse_args(argv)


def _bucket(share: float) -> str:
    """確定した期間の割合を段階の目安に丸める。案件ごとの段階数の違いを吸収する。"""
    for hi, lab in [(0.001, "0% (着手前)"), (0.25, "〜25%"), (0.5, "〜50%"),
                    (0.75, "〜75%"), (1.01, "75%超")]:
        if share < hi:
            return lab
    return "75%超"


def main(argv=None) -> int:
    a = parse_args(argv)
    overrides = {"位置合わせ": a.align.upper()} if a.align else {}
    ds = load_all(a.master, a.actuals, overrides=overrides)
    align = str(ds.settings["位置合わせ"]).upper() == "ON"

    agg = aggregate_actuals(ds)
    groups = group_names(agg, ds.phase_map, ds.group_col)
    curves = build_project_curves(ds, agg, groups)

    bases = BASIS_MODES if a.basis == "both" else \
        (BASIS_MODES[0],) if a.basis == "fixed" else (BASIS_MODES[1],)

    raw = a.ramp_limit
    if raw == "compare":
        ramps = [("制約なし", None), ("立ち上がり上限", "自動")]
    elif raw is None:
        ramps = [("立ち上がり上限", "自動")]
    elif str(raw).lower() in ("off", "なし"):
        ramps = [("制約なし", None)]
    else:
        ramps = [(f"上限 {raw}", float(raw))]

    rows = []
    for rlab, rlim in ramps:
      for basis in bases:
        for pid in sorted(curves):
            ph = phased_forecast(
                ds, curves, groups, agg, pid, align=align, ramp_limit=rlim,
                n_bin=int(ds.settings["カーブ解像度"]),
                backbone_spec=str(ds.settings["背骨マイルストーン"]),
                backbone_coverage=float(ds.settings["背骨最小カバー率"]),
                hours_per_mm=ds.hours_per_mm,
                recon=str(ds.settings["カーブ復元"]), basis=basis)
            n_month = len(ph.months)
            for _, r in ph.metrics.iterrows():
                rows.append({
                    "立ち上がり": rlab,
                    "残工数の決め方": basis,
                    "案件ID": pid,
                    "名称": ph.name,
                    "段階": int(r["段階"]),
                    "区切り": r["区切り"],
                    "確定割合": round(r["確定月数"] / n_month, 3),
                    "確定区分": _bucket(r["確定月数"] / n_month),
                    "残り月次WAPE": r["残り月次WAPE"],
                    "残り総量誤差(%)": r["残り総量誤差(%)"],
                    "累積カーブ最大乖離": r["累積カーブ最大乖離"],
                    "残りが全体に占める割合(%)": r["残りが全体に占める割合(%)"],
                    "立上り制約に当たった月数": r["立上り制約に当たった月数"],
                })
    d = pd.DataFrame(rows)

    print("=" * 78)
    print(f"段階予測の評価  完了案件 {len(curves)} 件 / 位置合わせ {'ON' if align else 'OFF'}"
          f" / 集約軸 {ds.group_col}")
    print("=" * 78)
    print("評価は残り区間だけで行う(確定分は実績そのもので誤差0のため)。")
    print("段階が進む = 確定が増える につれて誤差が下がっていれば、")
    print("予測は追加された実績を正しく使えている。")

    if len(ramps) > 1:
        print("\n■ 立ち上がり上限の効果(残り月次WAPE の平均)")
        cmp_r = (d.pivot_table(index="確定区分", columns="立ち上がり", values="残り月次WAPE",
                               aggfunc="mean", observed=True)
                   .reindex(["0% (着手前)", "〜25%", "〜50%", "〜75%", "75%超"])
                   .dropna(how="all").round(3))
        print(cmp_r.to_string())
        hit = d[d["立ち上がり"] != "制約なし"]["立上り制約に当たった月数"]
        print(f"  制約が効いた段階: {int((hit > 0).sum())}/{len(hit)}")
        d = d[d["立ち上がり"] == ramps[-1][0]]

    order = [b for b in BASIS_MODES if b in set(d["残工数の決め方"])]
    for basis in order:
        sub = d[d["残工数の決め方"] == basis]
        print(f"\n■ {basis}  確定した期間の割合で束ねた誤差")
        piv = (sub.groupby("確定区分", observed=True)
                  .agg(件数=("案件ID", "count"),
                       月次WAPE平均=("残り月次WAPE", "mean"),
                       月次WAPE中央=("残り月次WAPE", "median"),
                       総量誤差絶対値平均=("残り総量誤差(%)", lambda s: s.abs().mean()),
                       残り量平均=("残りが全体に占める割合(%)", "mean"))
                  .reindex(["0% (着手前)", "〜25%", "〜50%", "〜75%", "75%超"])
                  .dropna(how="all").round(3))
        print(piv.to_string())

    if len(order) > 1:
        print("\n■ 残工数の決め方の比較(残り月次WAPE の平均)")
        cmp = (d.pivot_table(index="確定区分", columns="残工数の決め方",
                             values="残り月次WAPE", aggfunc="mean", observed=True)
                 .reindex(["0% (着手前)", "〜25%", "〜50%", "〜75%", "75%超"])
                 .dropna(how="all").round(3))
        print(cmp.to_string())
        print("\n  終盤(75%超)は残り区間が数ヶ月しかなく、量も全体の数%しかない。")
        print("  分母が小さいぶん誤差率は跳ねるので、中盤までの数字で判断すること。")

    print("\n■ 案件ごと(残り月次WAPE / 段階順)")
    for basis in order:
        sub = d[d["残工数の決め方"] == basis]
        piv = (sub.pivot_table(index="案件ID", columns="段階",
                               values="残り月次WAPE", aggfunc="first", observed=True)
                  .round(3))
        print(f"\n  [{basis}]")
        print(piv.to_string())
        first, last = piv[piv.columns[0]], piv.max(axis=1)
        better = int((piv.iloc[:, 1:].min(axis=1) < first).sum())
        print(f"  着手前より当たる段階が1つ以上ある案件: {better}/{len(piv)} 件"
              f"(最悪段階が着手前より悪い案件: {int((last > first).sum())} 件)")

    if a.csv:
        d.to_csv(a.csv, index=False, encoding="utf-8-sig")
        print(f"\n明細を書き出しました: {a.csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
