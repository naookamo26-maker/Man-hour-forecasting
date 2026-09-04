"""
工数予測システム ─ エントリポイント

使い方
    python main.py                                # ステータス=予測対象 の案件をすべて出力
    python main.py --target PJ-2026-K             # 予測対象を指定
    python main.py --target PJ-2026-K,PJ-2027-L   # 複数指定(, または ; 区切り)
    python main.py --align off                    # 素朴版(位置合わせなし)
    python main.py --group-col 大分類             # 学習粒度を変える
    python main.py --no-validate                  # leave-one-out を省略して高速に
    python main.py --target PJ-2021-E --phased on # 完了案件で段階予測を確かめる

出力ファイル名は 案件の名称 + 条件 になる(設計書 7章)。
    output/蒼穹のレガリア II_align-on_行程グループ.xlsx
"""

from __future__ import annotations

import argparse
import os
import re
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.data_loader import load_all
from src.excel_writer import write_workbook
from src.forecast import forecast
from src.learning import (aggregate_actuals, build_project_curves, group_names,
                          learn, project_monthly)
from src.phased import (BASIS_FIXED, BASIS_MODES, BASIS_SCALED, MS_ALL,
                        MS_MODES, MS_PASSED, phased_forecast)
from src.elasticity import ELASTICITY_MAX
from src.ramp import LIMIT_AUTO, observed_growth_limit
from src.timeaxis import RECON_BOX, RECON_MODES, RECON_SMOOTH
from src.validation import leave_one_out

ROOT = os.path.dirname(os.path.abspath(__file__))

# Windows で使えない文字と、末尾に置けない文字
_BAD_CHARS = re.compile(r'[\\/:*?"<>|\x00-\x1f]')
_WS = re.compile(r"\s+")


def safe_filename(name: str, fallback: str, limit: int = 80) -> str:
    """案件の名称をファイル名に使える形にする。

    名称には「クロノスギア:蒼き遺産」のようにファイル名に使えない文字が入りうる。
    そのまま渡すと OS 依存で失敗するので、ここで落とす。
    空になった場合や、そもそも名称が未記入の場合は案件IDに戻す。
    """
    s = _BAD_CHARS.sub("", str(name or ""))
    s = _WS.sub(" ", s).strip().strip(".")
    if len(s) > limit:
        s = s[:limit].rstrip()
    return s or str(fallback)


# コマンドラインの短い綴りと、settings シートの日本語表記の対応。
# 引数は settings への上書きとして働くので、ここで表記を揃えてから overrides に入れる。
CLI_PHASED = {"auto": "自動", "on": "ON", "off": "OFF"}
CLI_BASIS = {"fixed": "固定", "scaled": "引き直す"}
CLI_SCOPE = {"phased": "段階予測のみ", "all": "全体", "off": "なし"}

BASIS_BY_NAME = {"固定": BASIS_FIXED, "引き直す": BASIS_SCALED}


def _flag(settings: dict, key: str) -> bool:
    """settings の ON/OFF を真偽値にする。"""
    v = str(settings.get(key, "ON")).strip()
    if v.upper() in ("ON", "TRUE", "1", "する", "はい"):
        return True
    if v.upper() in ("OFF", "FALSE", "0", "しない", "いいえ", "なし"):
        return False
    raise ValueError(
        f"settings の {key} は ON / OFF のどちらかで書いてください(記入値: {v!r})")


def _one_of(settings: dict, key: str, allowed: tuple[str, ...]) -> str:
    """settings の値が選択肢のどれかであることを確かめる。ON/OFF は大小を吸収する。"""
    v = str(settings.get(key, "")).strip()
    if v.upper() in ("ON", "OFF") and v.upper() in allowed:
        return v.upper()
    if v not in allowed:
        raise ValueError(
            f"settings の {key} は {' / '.join(allowed)} のいずれかで書いてください"
            f"(記入値: {v!r})")
    return v


def _id_list(raw) -> list[str]:
    """', ' や ';' 区切りの案件ID・行程グループ名を配列にする。"""
    return [x.strip() for x in re.split(r"[,;]", str(raw or "")) if x.strip()]


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="工数予測システム(第1版:素朴版+マイルストーン位置合わせ)")
    p.add_argument("--master", default=os.path.join(ROOT, "data", "master.xlsx"),
                   help="マスタExcelのパス")
    p.add_argument("--actuals", default=None,
                   help="実績CSVのパス。省略時は settings の 実績CSV、"
                        "それも空ならマスタと同じフォルダの actuals.csv")
    p.add_argument("--target", default=None,
                   help="予測対象の案件ID。, または ; 区切りで複数指定できる。"
                        "省略時は settings の 予測対象の案件ID、"
                        "それも空なら ステータス=予測対象 の案件をすべて出力する")
    p.add_argument("--align", choices=["on", "off"], default=None,
                   help="マイルストーン位置合わせ。省略時は settings の値")
    p.add_argument("--group-col", default=None,
                   help="集約軸(phase_map の列名)。例: 行程グループ / 大分類")
    p.add_argument("--n-bin", type=int, default=None, help="正準時間軸の分割数")
    p.add_argument("--backbone", default=None,
                   help="背骨マイルストーン名を ; 区切りで指定。既定は 自動")
    p.add_argument("--backbone-coverage", type=float, default=None,
                   help="背骨に採用するマイルストーンの最小カバー率(0〜1)")
    p.add_argument("--ms-precision", choices=["月", "自動"], default=None,
                   help="マイルストーン日付の精度。既定の 月 は日付を月央に丸める。"
                        "自動 にすると日まで書いた日付をその精度のまま使う")
    p.add_argument("--warp-strength", type=float, default=None,
                   help="位置合わせの強さ 0〜1。1=マイルストーン日付ちょうどに合わせる(既定)。"
                        "下げるほど時間軸の伸縮がゆるくなり、工数の跳ねが小さくなる")
    p.add_argument("--max-stretch", type=float, default=None,
                   help="時間軸の伸縮率の上限(倍)。例 1.5 なら区間の伸縮を 1/1.5〜1.5 倍に抑える。"
                        "0 または未指定で無制限")
    p.add_argument("--interval-elasticity", type=float, default=None,
                   help="学習データより狭く潰れたマイルストーン区間への工数配分を減らす度合い 0〜1。"
                        "0=区間の工数比率を学習値のまま使う(既定・設計書 3-1 の案A) / "
                        "1=区間が潰れたことによる工数密度の上昇を完全に打ち消す。"
                        "マイルストーンの日付は動かない")
    p.add_argument("--exclude-group", default=None,  # settings: 除外する行程グループ
                   help="この案件が行わない行程グループを ; 区切りで指定し、予測から外す。"
                        "estimates シートに見積もりがある場合は、そちらが優先される")
    p.add_argument("--ignore-estimates", action="store_true",
                   help="estimates シートの見積もりを使わず、契約人月と学習比率だけで予測する"
                        "(settings の 見積もりを使う を OFF にするのと同じ)")
    p.add_argument("--reconstruct", choices=["box", "smooth"], default=None,
                   help="月次から月内分布を復元する方式。box=月内均等(既定) / "
                        "smooth=累積の単調補間。省略時は settings の カーブ復元")
    p.add_argument("--no-compare-recon", action="store_true",
                   help="検証で復元方式を比較しない(既定は両方式を並べて出す。"
                        "settings の カーブ復元の比較 を OFF にするのと同じ)")
    p.add_argument("--phased", choices=["auto", "on", "off"], default=None,
                   help="段階予測(マイルストーンごとに実績を確定させ、残りを予測)。"
                        "auto=実績とマイルストーンがある案件では自動で出す。"
                        "省略時は settings の 段階予測")
    p.add_argument("--remain-basis", choices=["fixed", "scaled"], default=None,
                   help="段階予測の残工数の決め方。fixed=契約・見積もりの総量を固定して"
                        "残り=総量-確定分(既定) / scaled=確定分の実績から総量を引き直す。"
                        "省略時は settings の 段階予測の残工数")
    p.add_argument("--phased-milestones", choices=["すべて", "通過済み"], default=None,
                   help="段階予測でマイルストーンをどこまで既知とするか。"
                        "すべて=記入済みの日付を全部使う(既定・「予測」シートと同じ条件) / "
                        "通過済み=その段階までに通過したものだけ(未来を覗かない)")
    p.add_argument("--ramp-limit", default=None,
                   help="月次工数の増加率の上限(前月比)。自動=実績の前月比90%点から決める(既定) / "
                        "数値(例 1.3) / off=制約なし")
    p.add_argument("--ramp-scope", choices=["phased", "all", "off"], default=None,
                   help="立ち上がり上限をどこに効かせるか。phased=段階予測のみ(既定) / "
                        "all=全期間予測と検証にも効かせる / off=使わない。"
                        "省略時は settings の 立ち上がり上限の適用範囲")
    p.add_argument("--out", default=None,
                   help="出力Excelのパス。予測対象が1件のときだけ指定できる")
    p.add_argument("--out-dir", default=None,
                   help="出力先ディレクトリ。省略時は settings の 出力先フォルダ、"
                        "それも空なら output/")
    p.add_argument("--no-validate", action="store_true",
                   help="leave-one-out を実行しない(settings の 検証 を OFF にするのと同じ)")
    p.add_argument("--no-cache", action="store_true",
                   help="実績のparquetキャッシュを使わない"
                        "(settings の 実績キャッシュ を OFF にするのと同じ)")
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
    if a.interval_elasticity is not None:
        overrides["区間弾力性"] = a.interval_elasticity
    if a.reconstruct:
        overrides["カーブ復元"] = RECON_BOX if a.reconstruct == "box" else RECON_SMOOTH
    if a.ramp_limit is not None:
        overrides["立ち上がり上限"] = a.ramp_limit
    if a.phased_milestones is not None:
        overrides["段階予測のマイルストーン"] = a.phased_milestones
    if a.actuals:
        overrides["実績CSV"] = os.path.abspath(a.actuals)
    if a.target:
        overrides["予測対象の案件ID"] = a.target
    if a.out_dir:
        overrides["出力先フォルダ"] = a.out_dir
    if a.exclude_group is not None:
        overrides["除外する行程グループ"] = a.exclude_group
    if a.phased is not None:
        overrides["段階予測"] = CLI_PHASED[a.phased]
    if a.remain_basis is not None:
        overrides["段階予測の残工数"] = CLI_BASIS[a.remain_basis]
    if a.ramp_scope is not None:
        overrides["立ち上がり上限の適用範囲"] = CLI_SCOPE[a.ramp_scope]
    # --no-* は「無効化」の一方向。settings で OFF にしたものを引数で ON には戻さない。
    if a.no_validate:
        overrides["検証"] = "OFF"
    if a.no_compare_recon:
        overrides["カーブ復元の比較"] = "OFF"
    if a.ignore_estimates:
        overrides["見積もりを使う"] = "OFF"
    if a.no_cache:
        overrides["実績キャッシュ"] = "OFF"

    print("=" * 72)
    print("工数予測システム(第1版:素朴版 + マイルストーン位置合わせ)")
    print("=" * 72)

    # --- 読み込み ---
    # 実績CSVの場所とキャッシュの可否は settings 側で解決される(引数があればそちらが勝つ)。
    ds = load_all(a.master, overrides=overrides)
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
    elasticity = float(ds.settings["区間弾力性"])
    if not 0.0 <= elasticity <= ELASTICITY_MAX:
        raise ValueError(
            f"区間弾力性 は 0〜{ELASTICITY_MAX:g} の範囲で指定してください"
            f"(指定値: {elasticity})。1 で「区間が潰れたことによる密度上昇を"
            "完全に打ち消す」ところまで届くため、それ以上に振る意味はありません。")
    if recon not in RECON_MODES:
        raise ValueError(f"カーブ復元 は {' / '.join(RECON_MODES)} のいずれか(指定値: {recon!r})")

    # 残りの実行設定。ここまでと同じく settings が既定で、引数が上書きしている。
    phased_mode = _one_of(ds.settings, "段階予測", ("自動", "ON", "OFF"))
    basis = BASIS_BY_NAME[_one_of(ds.settings, "段階予測の残工数", ("固定", "引き直す"))]
    ramp_scope = _one_of(ds.settings, "立ち上がり上限の適用範囲",
                         ("段階予測のみ", "全体", "なし"))
    use_estimates = _flag(ds.settings, "見積もりを使う")
    do_validate = _flag(ds.settings, "検証")
    compare_recon = _flag(ds.settings, "カーブ復元の比較")
    excl = _id_list(ds.settings["除外する行程グループ"])
    out_dir = str(ds.settings["出力先フォルダ"] or "").strip() or os.path.join(ROOT, "output")

    # 立ち上がり上限。「自動」は学習データの前月比から決めるので、集約後でないと出せない。
    # ここでは指定値の解釈だけ行い、自動の解決は集約のあとで行う。
    raw_ramp = str(ds.settings["立ち上がり上限"]).strip()
    if ramp_scope == "なし" or raw_ramp.lower() in ("off", "なし", ""):
        ramp_spec = None
    elif raw_ramp == LIMIT_AUTO:
        ramp_spec = LIMIT_AUTO
    else:
        try:
            ramp_spec = float(raw_ramp)
        except ValueError:
            raise ValueError(
                f"立ち上がり上限 は 自動 / 数値(例 1.3) / off のいずれか(指定値: {raw_ramp!r})") from None
        if ramp_spec <= 1.0:
            raise ValueError(
                f"立ち上がり上限 は 1 より大きい値を指定してください(指定値: {ramp_spec})。"
                "制約を外すなら off と書きます。")
    ms_mode = str(ds.settings["段階予測のマイルストーン"]).strip()
    if ms_mode not in MS_MODES:
        raise ValueError(
            f"段階予測のマイルストーン は {' / '.join(MS_MODES)} のいずれか(指定値: {ms_mode!r})")

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

    # 全期間予測・検証に効かせる上限。段階予測は対象案件を学習から外した上で
    # 自前に決め直すため、ここで決めるのは scope=all のときに使う値になる。
    ramp_all = None
    if ramp_spec is not None and ramp_scope == "全体":
        ramp_all = (observed_growth_limit([c.monthly.sum(axis=1).to_numpy()
                                           for c in curves.values()])
                    if ramp_spec == LIMIT_AUTO else float(ramp_spec))
        if ramp_all is None:
            print("                立ち上がり上限は実績から決められないため使いません。")

    # --- 学習 ---
    # 予測対象が完了案件の場合(予測の妥当性を測るときがこれ)、その案件を学習に
    # 入れたままにすると自分の実績で学習したカーブで自分を予測することになる。
    # 案件ごとに、その案件だけを外して学習し直す。
    #
    # 予測対象をまとめて外して1回だけ学習するほうが速いが、それをすると
    # 対象の数だけ学習データが減る。1件ずつ外して学習し直すほうが、
    # どの案件についても使える学習データが最大になる。
    def learn_pair(subset):
        m = learn(subset, groups, align=align, n_bin=n_bin,
                  backbone_spec=backbone_spec, backbone_coverage=coverage,
                  hours_per_mm=hpm, recon=recon,
                  warp_strength=warp_strength, max_stretch=max_stretch,
                  interval_elasticity=elasticity)
        mn = (learn(subset, groups, align=False, n_bin=n_bin, hours_per_mm=hpm,
                    recon=recon) if align else None)
        return m, mn

    model, model_naive = learn_pair(curves)
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

    # --- 予測対象の決定 ---
    # 予測対象が複数あるのは普通のことなので、1回の実行で全部出す。
    # 学習と検証は対象案件に依存しないため、ここまでの結果をそのまま使い回す。
    targets_spec = _id_list(ds.settings["予測対象の案件ID"])
    if targets_spec:
        targets = targets_spec
        unknown = [t for t in targets if t not in ds.known_ids]
        if unknown:
            raise ValueError(
                f"指定された案件が projects シートにありません: {', '.join(unknown)}")
    else:
        tp = ds.target_projects()
        if tp.empty:
            print("[エラー] ステータス=予測対象 の案件がありません。"
                  "settings の 予測対象の案件ID か --target で指定してください。")
            return 1
        targets = [str(v) for v in tp["案件ID"]]
    if a.out and len(targets) > 1:
        raise ValueError(
            f"--out は予測対象が1件のときだけ指定できます(対象 {len(targets)} 件: "
            f"{', '.join(targets)})。出力先を変えるなら --out-dir を使ってください。")

    # --- 検証(対象案件に依存しないので1回だけ) ---
    if not do_validate:
        import pandas as pd
        detail = monthly = summary = pd.DataFrame()
        print("[4/5] 検証      スキップ")
    else:
        recons = (recon,) if not compare_recon else RECON_MODES
        detail, monthly, summary = leave_one_out(
            ds, curves, groups, n_bin=n_bin, backbone_spec=backbone_spec,
            backbone_coverage=coverage, hours_per_mm=hpm, recons=recons,
            warp_strength=warp_strength, max_stretch=max_stretch,
            interval_elasticity=elasticity, ramp_limit=ramp_all)
        print("[4/5] 検証      leave-one-out 完了")
        for _, r in summary.iterrows():
            imp = r.get("素朴版比_改善率", 0.0)
            print(f"    {r['モード']:<22} 月次誤差WAPE {r['月次誤差WAPE_平均']:.3f}"
                  f"   累積乖離 {r['累積カーブ最大乖離_平均']:.3f}"
                  f"   素朴版比 {imp:+.1%}")

    # --- 予測と出力(予測対象ごと) ---
    print(f"[5/5] 予測      対象 {len(targets)} 件: {', '.join(targets)}")
    base_params = {
        "実行日時": time.strftime("%Y-%m-%d %H:%M:%S"),
        "マスタ": ds.source.get("master", ""),
        "実績データ": ds.source.get("actuals", ""),
        "集約軸": ds.group_col,
        "位置合わせ": "ON" if align else "OFF",
        "背骨マイルストーン": ", ".join(model.backbone) or "(なし)",
        "背骨最小カバー率": coverage,
        "マイルストーン精度": str(ds.settings["マイルストーン精度"]),
        "位置合わせ強度": warp_strength,
        "伸縮率上限": max_stretch or "無制限",
        "区間弾力性": elasticity,
        "カーブ復元": recon,
        "カーブ解像度": n_bin,
        "人月換算係数": hpm,
        "見積もりを使う": "ON" if use_estimates else "OFF",
        "除外する行程グループ": ", ".join(excl) or "(なし)",
        "立ち上がり上限": str(ds.settings["立ち上がり上限"]),
        "立ち上がり上限の適用範囲": ramp_scope,
        "段階予測の動作": phased_mode,
        "段階予測の残工数": str(ds.settings["段階予測の残工数"]),
        "段階予測のマイルストーン": ms_mode,
        "検証": "ON" if do_validate else "OFF",
        "カーブ復元の比較": "ON" if compare_recon else "OFF",
        "出力先フォルダ": out_dir,
        "実装範囲": "実装順序 1〜2(素朴版 + マイルストーン位置合わせ) + 見積もりによる総量指定",
        "未使用パラメータ": f"k={ds.settings['k']}, タグ重み係数={ds.settings['タグ重み係数']} "
                            f"(実装順序 3・4 で使用予定)",
    }

    # 同名の案件があるとファイルが上書きされ、片方が黙って消える。
    # 名称が衝突する案件だけ案件IDを添える。
    stems: dict[str, list[str]] = {}
    for t in targets:
        stems.setdefault(safe_filename(ds.project(t)["名称"], t), []).append(t)

    written = []
    for target in targets:
        try:
            # 対象が学習データに入っていれば、その案件だけを外して学習し直す。
            if target in curves:
                sub = {p_: c for p_, c in curves.items() if p_ != target}
                if len(sub) < 2:
                    raise ValueError(
                        f"{target} を学習から外すと学習案件が {len(sub)} 件しか残りません。")
                t_model, t_naive = learn_pair(sub)
                t_curves = sub
                print(f"    ({target} は完了案件のため、自分を除いた {len(sub)} 件で学習し直し)")
            else:
                t_model, t_naive, t_curves = model, model_naive, curves

            path = _run_one(
                target, ds=ds, agg=agg, model=t_model, model_naive=t_naive,
                curves=t_curves, groups=groups, align=align, n_bin=n_bin,
                backbone_spec=backbone_spec, coverage=coverage, hpm=hpm,
                recon=recon, warp_strength=warp_strength, max_stretch=max_stretch,
                elasticity=elasticity,
                ramp_spec=ramp_spec, ramp_all=ramp_all, ms_mode=ms_mode,
                basis=basis, use_estimates=use_estimates, phased_mode=phased_mode,
                excl=excl, args=a, base_warnings=warnings, base_params=base_params,
                detail=detail, monthly=monthly, summary=summary,
                out_dir=out_dir, stems=stems)
        except (ValueError, KeyError) as e:
            # 1件の不備で残りの案件まで出せなくなるのは困る。落として続ける。
            print(f"    [中断] {target}: {e}")
            continue
        written.append(path)

    print()
    for path in written:
        print(f"出力: {path}")
    if not written:
        print("[エラー] 出力できた案件がありません。")
        return 1
    if len(written) < len(targets):
        print(f"({len(targets) - len(written)} 件は上記の理由で出力できませんでした)")
    print(f"所要: {time.time() - t0:.1f} 秒")
    return 0


def _run_one(target, *, ds, agg, model, model_naive, curves, groups, align, n_bin,
             backbone_spec, coverage, hpm, recon, warp_strength, max_stretch,
             elasticity, ramp_spec, ramp_all, ms_mode, basis, use_estimates,
             phased_mode, excl, args, base_warnings,
             base_params, detail, monthly, summary, out_dir, stems) -> str:
    """1案件分の予測・段階予測・Excel出力。戻り値は書き出したパス。

    学習・検証は呼び出し側で1回だけ行い、ここでは対象案件に依存する部分だけを行う。
    警告は案件ごとに独立させる(前の案件の警告が次の案件のシートに混ざらないように)。
    """
    warnings = list(base_warnings)
    if target not in curves and target in {str(p_) for p_ in ds.projects["案件ID"]} \
            and str(ds.project(target).get("ステータス", "")).strip() == "完了":
        warnings.append(
            f"{target} は完了案件のため、自分自身を学習から外した {len(curves)} 件で"
            "学習し直している(自分の実績で学習したカーブで自分を予測すると評価にならない)。"
            "「予測」シートの数字は、全案件で学習した場合とは一致しない。")

    # 着手前の案件は実績が1行も無く、分かっているのは要素別の見積もりだけになる。
    # 見積もりがあればそれをグループ別総量の正とし、無ければ契約人月を学習比率で割る。
    est = ds.estimates_of(target) if use_estimates else {}
    fc = forecast(model, ds, target, group_totals=est or None, exclude_groups=excl,
                  ramp_limit=ramp_all)

    src = "見積もり(estimates)" if est else "契約人月 + 学習比率"
    print(f"    {target} {fc.name}  {fc.months[0]}〜{fc.months[-1]} "
          f"({len(fc.months)}ヶ月) / {fc.total_hours:,.0f} 時間 "
          f"= {fc.total_hours / hpm:,.0f} 人月  [{src}]")
    dropped = [g for g in groups if g not in fc.groups]
    if dropped:
        print(f"        予測から除外: {', '.join(dropped)}")

    # 予測対象が進行中なら、同じ月軸・同じ集計で実績を並べて比較できるようにする。
    actual_table = project_monthly(agg, target, fc.months, fc.groups)

    # --- 段階予測 ---
    # 案件は普通、途中まで進んだ状態で「残りはどうなるか」を問われる。
    # 完了案件でそれを再現すれば、予測が当たっているかを案件の完了を待たずに測れる。
    has_actual = actual_table is not None and not actual_table.empty \
        and float(actual_table.to_numpy().sum()) > 0
    has_ms = not ds.milestones_of(target).empty
    want_phased = (phased_mode == "ON"
                   or (phased_mode == "自動" and has_actual and has_ms))
    phased = None
    if want_phased:
        try:
            phased = phased_forecast(
                ds, curves, groups, agg, target, align=align, n_bin=n_bin,
                backbone_spec=backbone_spec, backbone_coverage=coverage,
                hours_per_mm=hpm, recon=recon, warp_strength=warp_strength,
                max_stretch=max_stretch, interval_elasticity=elasticity,
                basis=basis, milestone_mode=ms_mode,
                ramp_limit=ramp_spec, group_totals=est or None, exclude_groups=excl)
        except ValueError as e:
            # 段階予測はあくまで検証用の付録。ここで落ちても本体の予測は出す。
            msg = f"段階予測は作れませんでした: {e}"
            print(f"        [警告] {msg}")
            warnings.append(msg)
    elif phased_mode == "自動" and not (has_actual and has_ms):
        lack = "実績" if not has_actual else "マイルストーン"
        print(f"        段階予測なし({lack}が無い。--phased on で強制)")

    if phased is not None:
        lim = f"{phased.ramp_limit:.2f} 倍/月" if phased.ramp_limit else "なし"
        n_al = sum(1 for st in phased.stages if st.aligned_milestones)
        print(f"        段階予測 {len(phased.stages)} 段階 / 残工数: {basis} / "
              f"立ち上がり上限: {lim} / MS: {phased.milestone_mode} / "
              f"位置合わせが効いた段階 {n_al}/{len(phased.stages)}")
        for _, r in phased.metrics.iterrows():
            w = r["残り月次WAPE"]
            print(f"          段階{int(r['段階'])} {str(r['区切り']):<12} "
                  f"確定 {int(r['確定月数']):>2}ヶ月 → 残り {int(r['残り月数']):>2}ヶ月  "
                  f"月次WAPE {w if w is None else f'{w:.3f}'}   "
                  f"残り総量誤差 {r['残り総量誤差(%)']:+.1f}%")
        warnings.extend(phased.warnings)

    # --- 出力 ---
    # ファイル名は案件IDではなく名称にする。関係者に渡すのは名称であって
    # IDではないため、受け取った側がどの案件か分かる形にしておく。
    stem = safe_filename(ds.project(target)["名称"], target)
    if len(stems.get(stem, [])) > 1:
        stem = f"{stem}_{target}"     # 同名の案件があるので案件IDで区別する
    out = args.out or os.path.join(
        out_dir, f"{stem}_align-{'on' if align else 'off'}_{ds.group_col}.xlsx")
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)

    params = dict(base_params)
    params["予測対象"] = f"{target}  {fc.name}"
    params["学習案件数"] = len(curves)
    params["総量の根拠"] = src
    params["予測した行程グループ"] = ", ".join(fc.groups)
    params["除外した行程グループ"] = ", ".join(dropped) or "(なし)"
    params["段階予測"] = (f"{len(phased.stages)} 段階 / 残工数の決め方: {phased.basis}"
                          if phased is not None else "(出力なし)")
    params["立ち上がり上限"] = (
        f"{ramp_all:.2f} 倍/月(全期間予測・検証にも適用)" if ramp_all
        else (f"{phased.ramp_limit:.2f} 倍/月(段階予測のみ)"
              if phased is not None and phased.ramp_limit else "使用しない"))

    write_workbook(out, fc=fc, model=model, model_naive=model_naive, curves=curves,
                   ds=ds, detail=detail, monthly=monthly, summary=summary,
                   params=params, warnings=warnings, actual_table=actual_table,
                   phased=phased)
    return out


if __name__ == "__main__":
    raise SystemExit(main())
