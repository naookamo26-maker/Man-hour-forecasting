"""期間とカーブの形の関係を測る(実装前の判断材料を出すだけのスクリプト)。

予測は現在、全案件の平均カーブを1本だけ持っている。案件ごとの形の違いは
マイルストーン位置合わせでしか表現されず、「この案件は長いからスロースタート」
といった条件づけは一切していない(settings の k・タグ重み係数 は未実装)。

もし 期間 が形を予測するなら、それは着手前に分かる情報なので、
実績が1行も無い時点の予測(「予測」シート・段階0)を改善できる。
情報を足す場所として一番価値が高い。このスクリプトはその可否を数字にする。

    python scripts/analyze_duration_shape.py
    python scripts/analyze_duration_shape.py --master path/to/master.xlsx
    python scripts/analyze_duration_shape.py --csv out.csv   # 案件別の表も書き出す

── 何を測っているか

  重心    位置合わせ後の正準軸([0,1])上で、工数の重心がどこにあるか。
          0.40 なら「工数の重心が期間の40%地点」= 前半型、0.60 なら後半型。
          4分類(前半/中央/後半/平均的)より連続量のほうが同じ件数で検出力が高い。
  集中度  同じ軸上の標準偏差。小さいほど山が尖っている(短期集中)。
          期間は「山の位置」より「平坦さ」に効いている可能性があるので併せて見る。

重心は必ず「位置合わせ後」で測る。生の月次で測ると、マイルストーン位置合わせが
既に吸収している分まで数えてしまい、条件づけを足したときに二重にかかる。
参考として位置合わせ前の値も出すので、差が大きければ位置合わせが効いている。

── 読み方(先に決めておくこと)

有意にならなかった場合、それは「関係が無い」ではなく「強い関係は無い」としか
言えない。合成データでの検証では、完了30件のとき

    重心の差 0.23(12ヶ月 0.41 → 36ヶ月 0.63) … 94% の確率で検出
    重心の差 0.15(        0.45 → 0.60)      … 68%
    重心の差 0.08(        0.49 → 0.57)      … 27%   ← 30件では見えない
    関係なし                                 …  5% (偽陽性)

弱い関係は30件では拾えない。件数が増えたら測り直すこと。

── 最後の LOO が本命

相関が出ても、それが予測精度の改善につながるとは限らない。
このスクリプトは最後に「期間の近い案件を重く見て平均カーブを作る」方式を
leave-one-out で実測する。採否はそこの数字で決めること。
合成データでの検証では、関係が無いときの副作用は −0.4% と小さく、
関係があるときの改善は +19〜51% だった。
"""

from __future__ import annotations

import argparse
import os
import sys
from math import erf, sqrt

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_loader import load_all
from src.elasticity import deflate
from src.learning import (aggregate_actuals, build_project_curves,
                          choose_backbone, group_names, monthly_to_canonical)
from src.timeaxis import Warp, canonical_to_monthly

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# ---------------------------------------------------------------------------
# 統計の小道具。scipy を足したくないので正規近似で済ませる(df >= 20 で十分)。
# ---------------------------------------------------------------------------
def _norm_sf(z: float) -> float:
    return 1.0 - 0.5 * (1.0 + erf(abs(z) / sqrt(2.0)))


def partial_corr(x: np.ndarray, y: np.ndarray,
                 controls: np.ndarray | None = None) -> dict:
    """x と y の相関。controls を渡すとその影響を除いた偏相関を返す。

    期間・規模・種別は互いに相関している(大きい案件は長い)。
    単純相関だけを見ると「期間が効いている」のか「規模が効いている」のか
    区別できないため、両方を出す。
    """
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    n = len(x)
    if controls is not None and controls.size:
        D = np.column_stack([np.ones(n), controls])
        # 特異な計画行列(種別が1種類しかない等)でも落ちないように lstsq を使う
        resid = lambda v: v - D @ np.linalg.lstsq(D, v, rcond=None)[0]
        x, y = resid(x), resid(y)
        df = n - D.shape[1] - 1
    else:
        x, y = x - x.mean(), y - y.mean()
        df = n - 2
    sx, sy = x.std(), y.std()
    if df < 3 or sx < 1e-12 or sy < 1e-12:
        return {"r": float("nan"), "p": float("nan"), "n": n, "df": max(df, 0),
                "lo": float("nan"), "hi": float("nan")}
    r = float(np.clip((x * y).mean() / (sx * sy), -0.999999, 0.999999))
    t = r * sqrt(df / (1 - r * r))
    p = 2 * _norm_sf(t)
    # Fisher z による 95% 信頼区間
    z = np.arctanh(r)
    se = 1.0 / sqrt(max(n - (0 if controls is None else controls.shape[1]) - 3, 1))
    lo, hi = np.tanh(z - 1.96 * se), np.tanh(z + 1.96 * se)
    return {"r": r, "p": p, "n": n, "df": df, "lo": float(lo), "hi": float(hi)}


def _stars(p: float) -> str:
    if not np.isfinite(p):
        return ""
    return "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""


# ---------------------------------------------------------------------------
# 案件ごとの形の指標
# ---------------------------------------------------------------------------
def build_table(ds, curves, n_bin: int, recon: str, warp_strength: float,
                max_stretch: float | None, elasticity: float,
                backbone_spec: str, coverage: float) -> tuple[pd.DataFrame, list[str]]:
    """完了案件ごとに 期間・種別・規模・形の指標 を並べた表を作る。

    位置合わせは learn() と同じ手順で組む。ここがずれると
    「学習が見ている形」と別のものを測ることになる。
    """
    backbone = choose_backbone(curves, backbone_spec, coverage)
    anchor = {nm: float(np.mean([c.ms_t[nm] for c in curves.values() if nm in c.ms_t]))
              for nm in backbone}

    x = (np.arange(n_bin) + 0.5) / n_bin
    rows = []
    for pid, c in curves.items():
        pairs = [(nm, anchor[nm], c.ms_t[nm]) for nm in backbone if nm in c.ms_t]
        c.warp = Warp.build(pairs, strength=warp_strength,
                            max_stretch=max_stretch) if pairs else Warp.identity()

        monthly = c.monthly.sum(axis=1).to_numpy(dtype=float)
        if monthly.sum() <= 0:
            continue
        can = monthly_to_canonical(monthly, c.edges, c.warp, n_bin, recon=recon)
        can = deflate(can / can.sum(), c.warp, elasticity)
        can = can / can.sum()

        cen = float((x * can).sum())
        spread = float(np.sqrt(((x - cen) ** 2 * can).sum()))

        # 位置合わせ前(生の経過期間比)。位置合わせがどれだけ吸収したかを見る用。
        m = monthly / monthly.sum()
        xm = (np.arange(len(m)) + 0.5) / len(m)
        raw_cen = float((xm * m).sum())

        rows.append({
            "案件ID": pid,
            "名称": c.name,
            "種別": c.ptype,
            "期間(月)": len(c.months),
            "実績人月": round(c.total_hours / ds.hours_per_mm, 1),
            "重心": round(cen, 4),
            "集中度": round(spread, 4),
            "重心(位置合わせ前)": round(raw_cen, 4),
            "アンカー数": len(pairs),
            "型": ("前半型" if cen < 0.47 else "後半型" if cen > 0.55 else "中央型"),
        })
    return pd.DataFrame(rows).sort_values("期間(月)").reset_index(drop=True), backbone


def _controls(df: pd.DataFrame, use_type: bool, use_size: bool) -> np.ndarray:
    """統制変数の計画行列。種別はダミー化、規模は対数。"""
    cols = []
    if use_type:
        types = sorted(df["種別"].astype(str).unique())
        for t in types[1:]:                   # 1つを基準カテゴリにする
            cols.append((df["種別"].astype(str) == t).astype(float).to_numpy())
    if use_size:
        mm = df["実績人月"].to_numpy(dtype=float)
        if (mm > 0).all():
            cols.append(np.log(mm))
    return np.column_stack(cols) if cols else np.empty((len(df), 0))


# ---------------------------------------------------------------------------
# 本命: 期間で条件づけすると予測が良くなるか
# ---------------------------------------------------------------------------
def loo_conditioning(curves, n_bin: int, recon: str, elasticity: float,
                     taus: list[float]) -> pd.DataFrame:
    """期間の近い案件を重く見て平均カーブを作り、leave-one-out で比較する。

    重みは w = exp(-(log期間の差)^2 / 2τ^2)。τ が小さいほど似た期間だけを見る。
    τ=None(現行)は単純平均。総量は評価から外し、形だけを比べる。
    """
    items = []
    for pid, c in curves.items():
        monthly = c.monthly.sum(axis=1).to_numpy(dtype=float)
        if monthly.sum() <= 0:
            continue
        can = monthly_to_canonical(monthly, c.edges, c.warp, n_bin, recon=recon)
        can = deflate(can / can.sum(), c.warp, elasticity)
        items.append({"pid": pid, "dur": len(c.months), "edges": c.edges,
                      "warp": c.warp, "act": monthly / monthly.sum(),
                      "can": can / can.sum()})
    if len(items) < 3:
        return pd.DataFrame()

    def run(tau):
        out = []
        for i, t in enumerate(items):
            cs, ws = [], []
            for j, o in enumerate(items):
                if i == j:
                    continue
                cs.append(o["can"])
                ws.append(1.0 if tau is None else
                          float(np.exp(-((np.log(o["dur"]) - np.log(t["dur"])) ** 2)
                                       / (2 * tau * tau))))
            w = np.array(ws)
            if w.sum() <= 0:
                continue
            w = w / w.sum()
            shape = np.average(np.array(cs), axis=0, weights=w)
            pred = canonical_to_monthly(shape, t["edges"], t["warp"])
            s = pred.sum()
            if s <= 0:
                continue
            pred = pred / s
            out.append({
                "WAPE": float(np.abs(pred - t["act"]).sum()),
                "ピーク月ズレ": abs(int(np.argmax(pred) - np.argmax(t["act"]))),
                # 実効的に何件を参照したか。1 に近いと1案件に頼っていて危険。
                "実効参照件数": float(1.0 / np.sum(w ** 2)),
            })
        return pd.DataFrame(out)

    rows = []
    base = run(None)
    rows.append({"方式": "現行(単純平均)", "τ": None,
                 "月次WAPE": base["WAPE"].mean(),
                 "ピーク月ズレ": base["ピーク月ズレ"].mean(),
                 "実効参照件数": base["実効参照件数"].mean(), "改善率": 0.0})
    for tau in taus:
        r = run(tau)
        if r.empty:
            continue
        rows.append({"方式": f"期間で条件づけ", "τ": tau,
                     "月次WAPE": r["WAPE"].mean(),
                     "ピーク月ズレ": r["ピーク月ズレ"].mean(),
                     "実効参照件数": r["実効参照件数"].mean(),
                     "改善率": 1 - r["WAPE"].mean() / base["WAPE"].mean()})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
def main() -> int:
    p = argparse.ArgumentParser(
        description="期間とカーブの形の関係を測る(判断材料を出すだけ。予測は変えない)")
    p.add_argument("--master", default=os.path.join(ROOT, "data", "master.xlsx"))
    p.add_argument("--actuals", default=None, help="省略時は settings / master と同じ場所")
    p.add_argument("--csv", default=None, help="案件別の表をCSVに書き出す")
    p.add_argument("--no-cache", action="store_true")
    p.add_argument("--taus", type=float, nargs="*", default=[0.15, 0.25, 0.40],
                   help="条件づけの幅。小さいほど似た期間だけを見る")
    a = p.parse_args()

    ds = load_all(a.master, actuals_path=a.actuals,
                  use_cache=not a.no_cache)
    agg = aggregate_actuals(ds)
    groups = group_names(agg, ds.phase_map, ds.group_col)
    curves = build_project_curves(ds, agg, groups)

    n_bin = int(ds.settings["カーブ解像度"])
    recon = str(ds.settings["カーブ復元"])
    ws = float(ds.settings["位置合わせ強度"])
    ms = float(ds.settings["伸縮率上限"] or 0) or None
    el = float(ds.settings["区間弾力性"] or 0)
    cov = float(ds.settings["背骨最小カバー率"])
    bspec = str(ds.settings["背骨マイルストーン"])

    df, backbone = build_table(ds, curves, n_bin, recon, ws, ms, el, bspec, cov)
    n = len(df)

    print()
    print("=" * 78)
    print("期間とカーブの形の関係")
    print("=" * 78)
    print(f"完了案件 {n} 件 / 背骨: {', '.join(backbone) or '(なし)'} / "
          f"集約軸 {ds.group_col} / 位置合わせ強度 {ws:g}")
    if n < 10:
        print("[警告] 件数が少なすぎます。以下の数字は参考になりません。")
        return 1

    print()
    print("--- 案件別 ---")
    print(df.drop(columns=["名称"]).to_string(index=False))
    if a.csv:
        df.to_csv(a.csv, index=False, encoding="utf-8-sig")
        print(f"\n案件別の表を書き出しました: {a.csv}")

    # ---- 1. 期間と重心 ----
    dur = np.log(df["期間(月)"].to_numpy(dtype=float))
    cen = df["重心"].to_numpy(dtype=float)
    spread = df["集中度"].to_numpy(dtype=float)
    raw = df["重心(位置合わせ前)"].to_numpy(dtype=float)

    print()
    print("--- 1. 期間は形を予測するか ---")
    print("  対象                     |    r    |  95%区間        |   p     |")
    combos = [
        ("重心   ~ 期間(単純)",        cen, None),
        ("重心   ~ 期間(種別を統制)",  cen, _controls(df, True, False)),
        ("重心   ~ 期間(種別+規模)",   cen, _controls(df, True, True)),
        ("集中度 ~ 期間(種別+規模)",   spread, _controls(df, True, True)),
        ("重心(位置合わせ前) ~ 期間",  raw, None),
    ]
    for label, y, ctl in combos:
        s = partial_corr(dur, y, ctl)
        print(f"  {label:<24} | {s['r']:+.3f}  | [{s['lo']:+.2f}, {s['hi']:+.2f}]  | "
              f"{s['p']:.4f} {_stars(s['p'])}")

    full = partial_corr(dur, cen, _controls(df, True, True))
    raw_s = partial_corr(dur, raw, None)
    if np.isfinite(full["r"]) and np.isfinite(raw_s["r"]) and abs(raw_s["r"]) > 0.05:
        absorbed = 1 - abs(full["r"]) / abs(raw_s["r"])
        print(f"\n  位置合わせの前後: 生 r={raw_s['r']:+.2f} → 位置合わせ後 r={full['r']:+.2f}")
        if absorbed > 0.05:
            print(f"  → 位置合わせが関係の約 {absorbed:.0%} を吸収している。"
                  "残りが、条件づけで新たに使える情報にあたる。")
        elif absorbed < -0.05:
            print("  → 位置合わせをしても関係は弱まらない(むしろはっきりする)。"
                  "マイルストーンでは説明できない形の差なので、条件づけで拾える余地がある。")
        else:
            print("  → 位置合わせは関係をほとんど吸収していない。"
                  "条件づけで拾える余地がそのまま残っている。")

    # ---- 2. 期間帯別の姿(直感で確かめる用) ----
    print()
    print("--- 2. 期間帯別の形 ---")
    q = df["期間(月)"].quantile([1/3, 2/3]).to_numpy()
    band = pd.cut(df["期間(月)"], [-np.inf, q[0], q[1], np.inf],
                  labels=[f"短期 〜{q[0]:.0f}ヶ月", f"中期 {q[0]:.0f}〜{q[1]:.0f}ヶ月",
                          f"長期 {q[1]:.0f}ヶ月〜"])
    tab = df.groupby(band, observed=True).agg(
        件数=("案件ID", "count"), 平均期間=("期間(月)", "mean"),
        重心=("重心", "mean"), 集中度=("集中度", "mean"))
    tab["平均期間"] = tab["平均期間"].round(1)
    print(tab.round(3).to_string())
    print()
    print(pd.crosstab(band, df["型"]).to_string())
    lo_c = df.loc[band == band.cat.categories[0], "重心"].mean()
    hi_c = df.loc[band == band.cat.categories[-1], "重心"].mean()
    print(f"\n  短期 {lo_c:.3f} → 長期 {hi_c:.3f}(差 {hi_c - lo_c:+.3f})")
    print("  ※ 差 0.23 なら30件で94%、0.15 なら68%、0.08 なら27% の確率でしか検出できない。")
    print("     有意でなくても『関係が無い』ではなく『強い関係は無い』としか言えない。")

    # ---- 3. 種別のほうが効いていないか ----
    print()
    print("--- 3. 期間と種別、どちらが効いているか ---")
    print(df.groupby("種別", observed=True).agg(
        件数=("案件ID", "count"), 平均期間=("期間(月)", "mean"),
        重心=("重心", "mean"), 集中度=("集中度", "mean")).round(3).to_string())
    only_type = partial_corr(dur, cen, _controls(df, True, False))
    no_ctl = partial_corr(dur, cen, None)
    print(f"\n  期間の効果: 統制なし r={no_ctl['r']:+.3f} → 種別を統制 r={only_type['r']:+.3f}")
    if np.isfinite(no_ctl["r"]) and abs(no_ctl["r"]) > 0.05:
        shrink = 1 - abs(only_type["r"]) / abs(no_ctl["r"])
        if shrink > 0.5:
            print("  → 種別を統制すると相関が半分以下に落ちる。"
                  "効いているのは種別で、期間はその代理変数の可能性が高い。")
        else:
            print("  → 種別を統制しても相関が残る。期間そのものが情報を持っている。")

    # ---- 4. 本命: 条件づけで予測が良くなるか ----
    print()
    print("--- 4. 期間で条件づけすると予測は良くなるか(leave-one-out・形のみ) ---")
    res = loo_conditioning(curves, n_bin, recon, el, list(a.taus))
    if res.empty:
        print("  案件が少なすぎて評価できません。")
    else:
        out = res.copy()
        out["月次WAPE"] = out["月次WAPE"].round(4)
        out["ピーク月ズレ"] = out["ピーク月ズレ"].round(2)
        out["実効参照件数"] = out["実効参照件数"].round(1)
        out["改善率"] = (out["改善率"] * 100).round(1).astype(str) + "%"
        out["τ"] = out["τ"].apply(lambda v: "-" if v is None or pd.isna(v) else f"{v:g}")
        print(out.to_string(index=False))
        best = res.iloc[1:]["改善率"].max() if len(res) > 1 else 0.0
        print()
        if best >= 0.05:
            b = res.iloc[1:].loc[res.iloc[1:]["改善率"].idxmax()]
            print(f"→ τ={b['τ']:g} で {best:.1%} の改善。実装を検討する価値がある。")
            print("   実効参照件数が 3 を下回っていないか確認すること。"
                  "少数の案件に頼った改善は、案件が入れ替わると消える。")
        elif best > 0:
            print(f"→ 改善は {best:.1%} にとどまる。実装コストに見合わない。")
        else:
            print("→ 改善なし。期間で条件づけしても予測は良くならない。")
        print("   採否はこの数字で決めること。1 の相関が有意でも、ここが改善しなければ意味がない。")

    print()
    print("=" * 78)
    print("このスクリプトは測るだけで、予測の出力は一切変えていない。")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
