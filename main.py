"""
工数予測システム ─ エントリポイント

使い方
    python main.py                                # 既定設定で実行
    python main.py --target PJ-2026-K             # 予測対象を指定
    python main.py --align off                    # 素朴版(位置合わせなし)
    python main.py --group-col 大分類             # 学習粒度を変える
    python main.py --no-validate                  # leave-one-out を省略して高速に

出力ファイル名には条件が埋め込まれる(設計書 7章)。
    output/forecast_PJ-2026-K_align-on_行程グループ.xlsx
"""

from __future__ import annotations

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.data_loader import load_all
from src.excel_writer import write_workbook
from src.forecast import forecast
from src.learning import (aggregate_actuals, build_project_curves, group_names,
                          learn, project_monthly)
from src.timeaxis import RECON_BOX, RECON_MODES, RECON_SMOOTH
from src.validation import leave_one_out

ROOT = os.path.dirname(os.path.abspath(__file__))


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="工数予測システム(第1版:素朴版+マイルストーン位置合わせ)")
    p.add_argument("--master", default=os.path.join(ROOT, "data", "master.xlsx"),
                   help="マスタExcelのパス")
    p.add_argument("--actuals", default=os.path.join(ROOT, "data", "actuals.csv"),
                   help="実績CSVのパス")
    p.add_argument("--target", default=None,
                   help="予測対象の案件ID。省略時は ステータス=予測対象 の先頭案件")
    p.add_argument("--align", choices=["on", "off"], default=None,
                   help="マイルストーン位置合わせ。省略時は settings の値")
    p.add_argument("--group-col", default=None,
                   help="集約軸(phase_map の列名)。例: 行程グループ / 大分類")
    p.add_argument("--n-bin", type=int, default=None, help="正準時間軸の分割数")
    p.add_argument("--backbone", default=None,
                   help="背骨マイルストーン名を ; 区切りで指定。既定は 自動")
    p.add_argument("--backbone-coverage", type=float, default=None,
                   help="背骨に採用するマイルストーンの最小カバー率(0〜1)")
    p.add_argument("--ms-precision", choices=["自動", "月"], default=None,
                   help="マイルストーン日付の精度。月 にすると日付を月央に丸める。"
                        "月まで表記のつもりが Excel に日付化されている場合に使う")
    p.add_argument("--warp-strength", type=float, default=None,
                   help="位置合わせの強さ 0〜1。1=マイルストーン日付ちょうどに合わせる(既定)。"
                        "下げるほど時間軸の伸縮がゆるくなり、工数の跳ねが小さくなる")
    p.add_argument("--max-stretch", type=float, default=None,
                   help="時間軸の伸縮率の上限(倍)。例 1.5 なら区間の伸縮を 1/1.5〜1.5 倍に抑える。"
                        "0 または未指定で無制限")
    p.add_argument("--exclude-group", default=None,
                   help="この案件が行わない行程グループを ; 区切りで指定し、予測から外す。"
                        "estimates シートに見積もりがある場合は、そちらが優先される")
    p.add_argument("--ignore-estimates", action="store_true",
                   help="estimates シートの見積もりを使わず、契約人月と学習比率だけで予測する")
    p.add_argument("--reconstruct", choices=["box", "smooth"], default=None,
                   help="月次から月内分布を復元する方式。box=月内均等(既定) / "
                        "smooth=累積の単調補間。省略時は settings の カーブ復元")
    p.add_argument("--no-compare-recon", action="store_true",
                   help="検証で復元方式を比較しない(既定は両方式を並べて出す)")
    p.add_argument("--out", default=None, help="出力Excelのパス")
    p.add_argument("--no-validate", action="store_true", help="leave-one-out を実行しない")
    p.add_argument("--no-cache", action="store_true", help="実績のparquetキャッシュを使わない")
    return p.parse_args(argv)


def main(argv=None) -> int:
    try:
        return _run(argv)
    except (ValueError, FileNotFoundError, KeyError) as e:
        # データ不備は想定内の失敗。トレースバックではなく、直せる形で伝える。
        print()
        print(f"[中断] {e}")
        return 1


def _run(argv=None) -> int:
    a = parse_args(argv)
    t0 = time.time()

    overrides = {}
    if a.align:
        overrides["位置合わせ"] = a.align.upper()
    if a.group_col:
        overrides["集約軸"] = a.group_col
    if a.n_bin:
        overrides["カーブ解像度"] = a.n_bin
    if a.backbone:
        overrides["背骨マイルストーン"] = a.backbone
    if a.backbone_coverage is not None:
        overrides["背骨最小カバー率"] = a.backbone_coverage
    if a.ms_precision is not None:
        overrides["マイルストーン精度"] = a.ms_precision
    if a.warp_strength is not None:
        overrides["位置合わせ強度"] = a.warp_strength
    if a.max_stretch is not None:
        overrides["伸縮率上限"] = a.max_stretch
    if a.reconstruct:
        overrides["カーブ復元"] = RECON_BOX if a.reconstruct == "box" else RECON_SMOOTH

    print("=" * 72)
    print("工数予測システム(第1版:素朴版 + マイルストーン位置合わせ)")
    print("=" * 72)

    # --- 読み込み ---
    ds = load_all(a.master, a.actuals, overrides=overrides, use_cache=not a.no_cache)
    align = str(ds.settings["位置合わせ"]).upper() == "ON"
    n_bin = int(ds.settings["カーブ解像度"])
    backbone_spec = str(ds.settings["背骨マイルストーン"])
    coverage = float(ds.settings["背骨最小カバー率"])
    recon = str(ds.settings["カーブ復元"])
    warp_strength = float(ds.settings["位置合わせ強度"])
    if not 0.0 <= warp_strength <= 1.0:
        raise ValueError(f"位置合わせ強度 は 0〜1 の範囲で指定してください(指定値: {warp_strength})")
    max_stretch_raw = float(ds.settings["伸縮率上限"])
    max_stretch = max_stretch_raw if max_stretch_raw > 1.0 else None
    if 0.0 < max_stretch_raw <= 1.0:
        raise ValueError(
            f"伸縮率上限 は 1 より大きい値(例 1.5)か、無制限を表す 0 を指定してください"
            f"(指定値: {max_stretch_raw})")
    if recon not in RECON_MODES:
        raise ValueError(f"カーブ復元 は {' / '.join(RECON_MODES)} のいずれか(指定値: {recon!r})")
    hpm = ds.hours_per_mm

    print(f"[1/5] 読み込み  案件 {len(ds.projects)} 件 / 実績 {len(ds.actuals):,} 行 "
          f"/ マイルストーン {len(ds.milestones)} 行")

    # --- 集約と案件カーブ ---
    agg = aggregate_actuals(ds)
    groups = group_names(agg, ds.phase_map, ds.group_col)
    curves = build_project_curves(ds, agg, groups)
    print(f"[2/5] 集約      {len(ds.actuals):,} 行 -> {len(agg):,} 行 "
          f"(集約軸: {ds.group_col}、{len(groups)} グループ)")
    print(f"                学習に使える案件: {len(curves)} 件"
          f" / 実績に現れる案件 {agg['案件ID'].nunique()} 件")

    # --- 学習 ---
    model = learn(curves, groups, align=align, n_bin=n_bin,
                  backbone_spec=backbone_spec, backbone_coverage=coverage,
                  hours_per_mm=hpm, recon=recon,
                  warp_strength=warp_strength, max_stretch=max_stretch)
    model_naive = (learn(curves, groups, align=False, n_bin=n_bin, hours_per_mm=hpm,
                         recon=recon)
                   if align else None)
    print(f"[3/5] 学習      位置合わせ {'ON' if align else 'OFF'} / "
          f"背骨: {', '.join(model.backbone) or '(なし)'} / カーブ復元: {recon}")
    if not model.backbone and align:
        print("                マイルストーンが不足のため素朴版と同じ挙動になります。")

    # 読み込み層・学習部が記録した警告を引き継ぐ(未登録案件、マイルストーン0件 など)
    warnings: list[str] = list(ds.warnings)
    min_n = int(ds.settings["マイルストーン最小件数"])
    for nm in model.low_sample_milestones(min_n):
        n = int(model.ms_stats.loc[model.ms_stats["マイルストーン名"] == nm, "件数"].iloc[0])
        warnings.append(f"マイルストーン「{nm}」は n={n} 件のみ。参考値として扱うこと(閾値 {min_n})。")
    # 行程グループの形状カーブは、そのグループの業務がある案件だけで平均される。
    # 一部の案件にしか無い業務は少数サンプルのカーブになるが、出力上は
    # 全案件から学習したカーブと見分けがつかないため、件数を明示する。
    min_g = int(ds.settings["行程グループ最小件数"])
    for g in model.low_sample_groups(min_g):
        n = model.group_sample_n[g]
        warnings.append(
            f"行程グループ「{g}」の工数カーブは {n}/{len(curves)} 案件の実績しか"
            f"根拠にしていない(閾値 {min_g})。その業務が無い案件は形の平均に参加しないため、"
            "カーブの形は参考値として扱うこと。なお総量の比率は業務が無い案件も"
            "0として平均に含めるため、その分だけ薄まっている。")

    for nm in model.ms_stats["マイルストーン名"]:
        if nm not in model.backbone and align:
            n = int(model.ms_stats.loc[model.ms_stats["マイルストーン名"] == nm, "件数"].iloc[0])
            warnings.append(
                f"マイルストーン「{nm}」はカバー率 {n}/{len(curves)} が閾値 {coverage:.0%} 未満のため"
                "位置合わせには使っていない。")
    if not align:
        warnings.append("位置合わせ OFF で実行。マイルストーン直前の工数の山は平均によってならされている。")

    # 位置合わせは時間軸を区間ごとに伸縮させる。伸縮率はアンカーの前後で不連続に変わり、
    # その比がそのまま月次工数の跳ねの倍率になる。極端な案件は学習カーブを歪め、
    # その案件自身の予測では月次グラフに大きな段差として現れる。
    # どの案件のマイルストーンが原因かは、ここを見ないと特定できない。
    if align and model.backbone:
        clipped = [(p_, c.warp.clipped) for p_, c in curves.items() if c.warp.clipped]
        for p_, nms in clipped:
            warnings.append(
                f"{p_} はマイルストーン {', '.join(nms)} の日付が近すぎる/順序が逆のため、"
                "時間軸上の位置を補正した。補正で潰れた区間に工数が集中し、"
                "学習カーブを歪める。milestones シートの日付を確認すること。")
        steep = sorted(((c.warp.max_step_ratio, p_, c) for p_, c in curves.items()
                        if not c.warp.is_identity), reverse=True)
        for ratio, p_, c in steep:
            if ratio < 3.0:
                break
            warnings.append(
                f"{p_}({c.name}, {len(c.months)}ヶ月, 実績 {c.total_hours / hpm:,.0f} 人月)は"
                f"マイルストーン位置が学習データの平均から離れており、時間軸の伸縮率が"
                f"アンカーの前後で {ratio:.1f} 倍変わる(最大密度倍率 {c.warp.max_density_gain:.1f} 倍)。"
                "この案件は学習カーブを尖らせる側にも、自身の予測が跳ねる側にも効く。"
                "scripts/diagnose_warp.py で影響を確認し、必要なら 位置合わせ強度 を下げること。")

    # 契約人月と実績合計の乖離。学習結果には影響しないが、実績側の欠落を疑う手がかりになる。
    gap = model.contributors.dropna(subset=["乖離率(%)"])
    for _, r in gap[gap["乖離率(%)"].abs() >= 20.0].iterrows():
        warnings.append(
            f"{r['案件ID']} は契約 {r['契約人月']:,.0f} 人月に対し実績 {r['実績人月']:,.1f} 人月"
            f"({r['乖離率(%)']:+.1f}%)。カーブの形は正規化して学習するため予測には影響しないが、"
            "実績の記録漏れが無いか確認すること。")

    # --- 予測 ---
    target = a.target
    if target is None:
        tp = ds.target_projects()
        if tp.empty:
            print("[エラー] ステータス=予測対象 の案件がありません。--target で指定してください。")
            return 1
        target = str(tp.iloc[0]["案件ID"])

    # 着手前の案件は実績が1行も無く、分かっているのは要素別の見積もりだけになる。
    # 見積もりがあればそれをグループ別総量の正とし、無ければ契約人月を学習比率で割る。
    est = {} if a.ignore_estimates else ds.estimates_of(target)
    excl = [s.strip() for s in (a.exclude_group or "").split(";") if s.strip()]
    fc = forecast(model, ds, target, group_totals=est or None, exclude_groups=excl)

    src = "見積もり(estimates)" if est else "契約人月 + 学習比率"
    print(f"[4/5] 予測      {target} {fc.name}  "
          f"{fc.months[0]}〜{fc.months[-1]} ({len(fc.months)}ヶ月) / "
          f"{fc.total_hours:,.0f} 時間 = {fc.total_hours/hpm:,.0f} 人月")
    print(f"                総量の根拠: {src} / 行程グループ {len(fc.groups)}/{len(groups)}")
    dropped = [g for g in groups if g not in fc.groups]
    if dropped:
        print(f"                予測から除外: {', '.join(dropped)}")

    # 予測対象が進行中なら、同じ月軸・同じ集計で実績を並べて比較できるようにする。
    actual_table = project_monthly(agg, target, fc.months, fc.groups)

    # --- 検証 ---
    if a.no_validate:
        import pandas as pd
        detail = monthly = summary = pd.DataFrame()
        print("[5/5] 検証      スキップ")
    else:
        recons = (recon,) if a.no_compare_recon else RECON_MODES
        detail, monthly, summary = leave_one_out(
            ds, curves, groups, n_bin=n_bin, backbone_spec=backbone_spec,
            backbone_coverage=coverage, hours_per_mm=hpm, recons=recons,
            warp_strength=warp_strength, max_stretch=max_stretch)
        print("[5/5] 検証      leave-one-out 完了")
        print()
        for _, r in summary.iterrows():
            imp = r.get("素朴版比_改善率", 0.0)
            print(f"    {r['モード']:<22} 月次誤差WAPE {r['月次誤差WAPE_平均']:.3f}"
                  f"   累積乖離 {r['累積カーブ最大乖離_平均']:.3f}"
                  f"   素朴版比 {imp:+.1%}")
        print()

    # --- 出力 ---
    out = a.out or os.path.join(
        ROOT, "output",
        f"forecast_{target}_align-{'on' if align else 'off'}_{ds.group_col}.xlsx")
    os.makedirs(os.path.dirname(out), exist_ok=True)

    params = {
        "実行日時": time.strftime("%Y-%m-%d %H:%M:%S"),
        "予測対象": f"{target}  {fc.name}",
        "マスタ": ds.source.get("master", ""),
        "実績データ": ds.source.get("actuals", ""),
        "集約軸": ds.group_col,
        "位置合わせ": "ON" if align else "OFF",
        "背骨マイルストーン": ", ".join(model.backbone) or "(なし)",
        "背骨最小カバー率": coverage,
        "マイルストーン精度": str(ds.settings["マイルストーン精度"]),
        "位置合わせ強度": warp_strength,
        "伸縮率上限": max_stretch or "無制限",
        "カーブ復元": recon,
        "カーブ解像度": n_bin,
        "人月換算係数": hpm,
        "学習案件数": len(curves),
        "総量の根拠": src,
        "予測した行程グループ": ", ".join(fc.groups),
        "除外した行程グループ": ", ".join(dropped) or "(なし)",
        "実装範囲": "実装順序 1〜2(素朴版 + マイルストーン位置合わせ) + 見積もりによる総量指定",
        "未使用パラメータ": f"k={ds.settings['k']}, タグ重み係数={ds.settings['タグ重み係数']} "
                            f"(実装順序 3・4 で使用予定)",
    }

    write_workbook(out, fc=fc, model=model, model_naive=model_naive, curves=curves,
                   ds=ds, detail=detail, monthly=monthly, summary=summary,
                   params=params, warnings=warnings, actual_table=actual_table)

    print(f"出力: {out}")
    print(f"所要: {time.time() - t0:.1f} 秒")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
