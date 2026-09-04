"""マイルストーン区間への工数配分は、区間の長さにどれだけ引きずられるか。

設計書 3-1(案A)は「区間の工数比率は学習値のまま、期間だけを伸縮する」と決めた。
その帰結として、区間が短いほど工数密度が上がり、極端に短い区間では鋭い山になる。

これに対する疑いはこうである。

    区間が2ヶ月しかないなら、そこに全体の30%を投入することは人員的にできない。
    短い区間は「判断のための版を作っただけ」で、そもそも工数が軽いのではないか。
    予算消化は時間軸に、カーブの形はマイルストーン区間に従うのではないか。

これは検証できる。完了案件について、区間ごとに次の3つが測れるからである。

    時間比率 t  区間の実時間の長さ ÷ 全期間
    正準比率 c  学習カーブがその区間に置く工数の比率(= 案Aが割り当てる量)
    実績比率 a  実際にその区間で消化された工数の比率

案Aが正しいなら a ≒ c。疑いが正しいなら、c が t より大きい区間(=時間の割に
工数が濃い区間)では a は c より小さくなり、t の側へ引き戻されているはずである。

── 推定するもの

    モデルA(現行)   w = c
    モデルB(両側)   w ∝ c^(1-λ) · t^λ      λ=0 で案A、λ=1 で完全な時間比例
    モデルC(片側)   濃い区間(c>t)だけ B と同じに緩め、薄い区間は案Aのまま

B は両辺の対数をとると

    log(a/t) = 定数 + (1-λ)·log(c/t)

という直線になる。つまり **log-log 回帰の傾きがそのまま 1-λ** で、λ が
回帰係数として直接推定できる。C はグリッド探索で誤差最小の λ を選ぶ。

なぜ B と C を分けるか。設計書が案B・案C(工数を期間に連動)を却下した理由は
「期間が延びるのは承認待ちが理由で、作業量が増えるわけではない」だった。
これは区間が **伸びる** 側の話で、縮む側には当てはまらない。
片側だけ緩める C なら、却下理由と衝突しない。どちらが実態に合うかを数字で見る。

── 数字を信じてよいかの確認(--selftest)

推定の前に、この手法自体が当てになるかを確かめる。区間配分だけを既知の λ で
作り直した疑似実績を通し、その λ が戻ってくるかを見る。戻らなければ、
実データに対して出た数字も信用できない。

    python scripts/estimate_interval_elasticity.py
    python scripts/estimate_interval_elasticity.py --selftest
    python scripts/estimate_interval_elasticity.py --master 実データ.xlsx --actuals 実績.csv
"""

from __future__ import annotations

import argparse
import os
import sys

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_loader import load_all  # noqa: E402
from src.learning import (aggregate_actuals, build_project_curves, group_names,  # noqa: E402
                          learn)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e6e5e0", "#fcfcfb"
S1, S3, S7, S8 = "#2a78d6", "#1baf7a", "#4a3aa7", "#e34948"
CYCLE = [S1, S3, S7, S8]
MARKERS = ["o", "s", "^", "D"]   # 色だけに頼らないための第二の手がかり

LAMBDAS = np.round(np.arange(0.0, 1.001, 0.02), 3)


# ---------------------------------------------------------------------------
# 下ごしらえ
# ---------------------------------------------------------------------------
def setup_style() -> None:
    from matplotlib import font_manager
    have = {f.name for f in font_manager.fontManager.ttflist}
    for name in ("IPAGothic", "Noto Sans CJK JP", "Yu Gothic", "Meiryo",
                 "Hiragino Sans", "MS Gothic", "TakaoGothic"):
        if name in have:
            plt.rcParams["font.family"] = name
            break
    else:
        print("[注意] 日本語フォントが見つかりません。図の文字が崩れます。")
    plt.rcParams.update({
        "axes.unicode_minus": False, "figure.facecolor": SURF, "axes.facecolor": SURF,
        "axes.edgecolor": "#c9c8c3", "axes.labelcolor": INK2, "xtick.color": INK2,
        "ytick.color": INK2, "text.color": INK, "grid.color": GRID,
    })


def load(master: str, actuals: str):
    ds = load_all(master, actuals)
    agg = aggregate_actuals(ds)
    groups = group_names(agg, ds.phase_map, ds.group_col)
    curves = build_project_curves(ds, agg, groups)
    cfg = dict(n_bin=int(ds.settings["カーブ解像度"]),
               backbone_spec=str(ds.settings["背骨マイルストーン"]),
               backbone_coverage=float(ds.settings["背骨最小カバー率"]),
               hours_per_mm=ds.hours_per_mm)
    return ds, groups, curves, cfg


def build_model(curves, groups, cfg, *, exclude=None):
    rest = {p: c for p, c in curves.items() if p != exclude} if exclude else curves
    return learn(rest, groups, align=True, n_bin=cfg["n_bin"],
                 backbone_spec=cfg["backbone_spec"],
                 backbone_coverage=cfg["backbone_coverage"],
                 hours_per_mm=cfg["hours_per_mm"])


# ---------------------------------------------------------------------------
# 区間ごとの 時間比率 / 正準比率 / 実績比率
# ---------------------------------------------------------------------------
def interval_bounds(curve, model):
    """区間の境界を、実時間軸 u と正準時間軸 s の対で返す。

    背骨マイルストーンのうち、その案件が実際に持っているものだけを使う。
    区間が1つしか作れない(=マイルストーンが無い)案件は対象外。
    """
    names = [nm for nm in model.backbone if nm in curve.ms_t]
    if not names:
        return None
    names.sort(key=lambda nm: curve.ms_t[nm])
    u = np.array([0.0] + [curve.ms_t[nm] for nm in names] + [1.0])
    s = np.array([0.0] + [model.canonical_anchor[nm] for nm in names] + [1.0])
    # 日付の近接・逆順で潰れた区間は比率が発散するので使わない
    if np.any(np.diff(u) <= 1e-6) or np.any(np.diff(s) <= 1e-6):
        return None
    return names, u, s


def interval_rows(pid: str, monthly_total: np.ndarray, edges: np.ndarray,
                  n_month: int, names, u, s, shape: np.ndarray) -> pd.DataFrame:
    """1案件ぶんの区間表を作る。

    実績比率は月境界の累積を区間境界へ線形補間して求める(月内は一様と見なす)。
    マイルストーンは月精度で丸められているため、月内の分布まで踏み込む意味は薄い。
    """
    cum = np.concatenate([[0.0], np.cumsum(monthly_total)])
    cum = cum / cum[-1]
    a = np.diff(np.interp(u, edges, cum))

    s_edges = np.linspace(0.0, 1.0, len(shape) + 1)
    s_cum = np.concatenate([[0.0], np.cumsum(shape)])
    c = np.diff(np.interp(s, s_edges, s_cum))

    t = np.diff(u)
    lab = ["開始"] + list(names) + ["終了"]
    return pd.DataFrame({
        "案件ID": pid,
        "区間": [f"{lab[i]}→{lab[i + 1]}" for i in range(len(u) - 1)],
        "月数": np.round(t * n_month, 1),
        "時間比率": t, "正準比率": c, "実績比率": a,
    })


def collect(ds, groups, curves, cfg, monthly_override: dict | None = None) -> pd.DataFrame:
    """全案件の区間表を leave-one-out で集める。

    正準比率は「その案件を除いて学習したカーブ」から取る。自分自身が入った
    平均に自分を合わせると、区間配分の当たり具合が実際より良く見えてしまう。
    """
    out = []
    for pid in sorted(curves):
        c = curves[pid]
        model = build_model(curves, groups, cfg, exclude=pid)
        b = interval_bounds(c, model)
        if b is None:
            continue
        names, u, s = b
        m = (monthly_override[pid] if monthly_override
             else c.monthly.sum(axis=1).to_numpy())
        out.append(interval_rows(pid, m, c.edges, len(c.months), names, u, s,
                                 model.total_shape))
    if not out:
        raise SystemExit("背骨マイルストーンを持つ案件がありません。位置合わせが働いていないため、"
                         "この検証はできません(背骨最小カバー率 を確認してください)。")
    d = pd.concat(out, ignore_index=True)
    d["モデル密度"] = d["正準比率"] / d["時間比率"]   # 案Aが要求する密度倍率
    d["実績密度"] = d["実績比率"] / d["時間比率"]     # 実際に起きた密度倍率
    return d


# ---------------------------------------------------------------------------
# λ の推定
# ---------------------------------------------------------------------------
def weights_of(c: np.ndarray, t: np.ndarray, lam: float, one_sided: bool) -> np.ndarray:
    """モデル B / C が区間に置く工数比率。

    one_sided=True が C。濃い区間(c>t)だけを時間比率の側へ緩め、
    薄い区間は案Aのまま動かさない。
    """
    r = c / t
    r2 = r ** (1.0 - lam)
    if one_sided:
        r2 = np.where(r > 1.0, r2, r)
    w = r2 * t
    return w / w.sum()


def fit_loglog(d: pd.DataFrame, mask=None) -> tuple[float, float, float, int]:
    """log-log 回帰の傾きから λ を出す。戻り値 (λ, 傾きの標準誤差, 相関, 標本数)。"""
    sub = d if mask is None else d[mask]
    x = np.log(sub["モデル密度"].to_numpy())
    y = np.log(sub["実績密度"].to_numpy())
    n = len(x)
    if n < 3 or np.ptp(x) < 1e-9:
        return np.nan, np.nan, np.nan, n
    beta, a0 = np.polyfit(x, y, 1)
    resid = y - (beta * x + a0)
    sxx = float(np.sum((x - x.mean()) ** 2))
    se = float(np.sqrt(np.sum(resid ** 2) / (n - 2) / sxx)) if n > 2 else np.nan
    return float(1.0 - beta), se, float(np.corrcoef(x, y)[0, 1]), n


def fit_grid(d: pd.DataFrame, one_sided: bool) -> tuple[float, np.ndarray]:
    """区間比率の予測誤差(案件ごとのL1を平均)が最小になる λ を探す。

    回帰と違い、これは「その λ を使ったら配分がどれだけ当たるか」を直接測る。
    採否はこちらの数字で決めるべきである。
    """
    errs = []
    for lam in LAMBDAS:
        e = []
        for _, g in d.groupby("案件ID"):
            w = weights_of(g["正準比率"].to_numpy(), g["時間比率"].to_numpy(),
                           lam, one_sided)
            a = g["実績比率"].to_numpy()
            e.append(float(np.abs(a / a.sum() - w).sum()))
        errs.append(float(np.mean(e)))
    errs = np.array(errs)
    return float(LAMBDAS[int(np.argmin(errs))]), errs


def bootstrap_lambda(d: pd.DataFrame, one_sided: bool, n_iter: int = 2000,
                     seed: int = 0) -> tuple[float, float]:
    """案件単位の復元抽出で λ の 95% 区間を出す。

    区間は同じ案件の中で強く相関するので、抜き出す単位は区間ではなく案件にする。
    区間単位で抜くと標本数を実際より多く見積もり、区間が不当に狭くなる。
    """
    rng = np.random.default_rng(seed)
    pids = d["案件ID"].unique()
    vals = []
    for _ in range(n_iter):
        pick = rng.choice(pids, size=len(pids), replace=True)
        sub = pd.concat([d[d["案件ID"] == p] for p in pick], ignore_index=True)
        lam, _ = fit_grid(sub, one_sided)
        vals.append(lam)
    return float(np.quantile(vals, 0.025)), float(np.quantile(vals, 0.975))


# ---------------------------------------------------------------------------
# 自己テスト ─ 既知の λ を復元できるか
# ---------------------------------------------------------------------------
def synth_monthly(curve, model, lam_true: float, one_sided: bool) -> np.ndarray | None:
    """区間配分だけを λ_true に従って作り直した疑似月次を返す。

    区間内の形は正準カーブのまま、区間の総量だけを w(λ_true) に置き換える。
    「予算消化は時間軸に、形は区間に従う」を、そのままデータにしたもの。
    """
    b = interval_bounds(curve, model)
    if b is None:
        return None
    names, u, s = b
    shape = model.total_shape
    s_edges = np.linspace(0.0, 1.0, len(shape) + 1)
    s_cum = np.concatenate([[0.0], np.cumsum(shape)])
    c = np.diff(np.interp(s, s_edges, s_cum))
    t = np.diff(u)
    w = weights_of(c, t, lam_true, one_sided)

    # 細分グリッドで、区間ごとに正準カーブの形を実区間へ線形に貼り、w に正規化する
    fine = 8000
    x = (np.arange(fine) + 0.5) / fine
    mass = np.zeros(fine)
    pos = np.zeros(fine)
    for j in range(len(u) - 1):
        lo, hi = u[j], u[j + 1]
        sel = (x >= lo) & (x < hi)
        if not sel.any():
            continue
        frac = (x[sel] - lo) / (hi - lo)
        s_at = s[j] + frac * (s[j + 1] - s[j])
        dens = np.interp(s_at, (s_edges[:-1] + s_edges[1:]) / 2, shape)
        dens = dens / dens.sum() if dens.sum() > 0 else np.full(sel.sum(), 1.0 / sel.sum())
        mass[sel] = dens * w[j]
        pos[sel] = x[sel]
    idx = np.clip(np.searchsorted(curve.edges, pos, side="right") - 1,
                  0, len(curve.months) - 1)
    out = np.bincount(idx, weights=mass, minlength=len(curve.months))
    return out * (curve.total_hours / out.sum())


def selftest(ds, groups, curves, cfg, one_sided: bool) -> pd.DataFrame:
    """λ を 0.0〜0.8 で仕込み、推定が戻すかを見る。

    正準位置は全案件から学習したものを固定して使う(復元性だけを見るため)。
    実データの推定はこれより厳しい条件(leave-one-out)なので、
    ここで戻らなければ実データでは尚のこと戻らない。
    """
    base = build_model(curves, groups, cfg)
    rows = []
    for lam_true in (0.0, 0.2, 0.4, 0.6, 0.8):
        override = {}
        for pid, c in curves.items():
            m = synth_monthly(c, base, lam_true, one_sided)
            if m is not None:
                override[pid] = m
        d = collect(ds, groups, curves, cfg, monthly_override=override)
        lam_grid, _ = fit_grid(d, one_sided)
        lam_reg, se, r, n = fit_loglog(d)
        rows.append({"仕込んだλ": lam_true, "推定λ(誤差最小)": lam_grid,
                     "推定λ(回帰)": round(lam_reg, 3), "回帰の相関": round(r, 3),
                     "区間数": n})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 図
# ---------------------------------------------------------------------------
def draw(d: pd.DataFrame, errs_one: np.ndarray, errs_both: np.ndarray,
         lam_one: float, lam_both: float, out: str,
         st: pd.DataFrame | None = None, floor: float | None = None) -> str:
    fig, axes_2d = plt.subplots(2, 2, figsize=(14.5, 10))
    axes = np.ravel(axes_2d)

    # (1) 案Aが要求する密度 と 実際に起きた密度
    ax = axes[0]
    kinds = list(dict.fromkeys(d["区間"]))
    for i, k in enumerate(kinds):
        sub = d[d["区間"] == k]
        ax.scatter(sub["モデル密度"], sub["実績密度"], s=46,
                   color=CYCLE[i % len(CYCLE)], marker=MARKERS[i % len(MARKERS)],
                   edgecolor=SURF, linewidth=1.2, label=k, zorder=3)
    lo = float(min(d["モデル密度"].min(), d["実績密度"].min())) * 0.85
    hi = float(max(d["モデル密度"].max(), d["実績密度"].max())) * 1.15
    ax.plot([lo, hi], [lo, hi], color="#8f8e88", lw=1.4, ls="--", zorder=2,
            label="案A(現行)が仮定する線")
    x = np.log(d["モデル密度"].to_numpy())
    y = np.log(d["実績密度"].to_numpy())
    beta, a0 = np.polyfit(x, y, 1)
    xs = np.linspace(np.log(lo), np.log(hi), 50)
    ax.plot(np.exp(xs), np.exp(beta * xs + a0), color=S8, lw=2.0, zorder=4,
            label=f"実データの回帰(傾き {beta:.2f})")
    ax.axvline(1.0, color=GRID, lw=1.0, zorder=1)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("案Aが割り当てる密度  正準比率 ÷ 時間比率", fontsize=9)
    ax.set_ylabel("実際に消化された密度  実績比率 ÷ 時間比率", fontsize=9)
    ax.set_title("区間の濃さ ─ モデルの要求 と 実態", fontsize=11, color=INK, loc="left")
    ax.legend(fontsize=7.5, frameon=False, loc="upper left")

    # (2) λ ごとの配分誤差
    ax = axes[1]
    ax.plot(LAMBDAS, errs_both, color=S1, lw=2.2, label="モデルB(両側)")
    ax.plot(LAMBDAS, errs_one, color=S3, lw=2.2, label="モデルC(濃い側だけ)")
    if floor is not None:
        ax.axvspan(0.0, floor, color=GRID, zorder=1)
        ax.annotate(f"ノイズ床 λ≦{floor:.2f}\nここに落ちたら「効果なし」",
                    (floor, ax.get_ylim()[1]), xycoords=("data", "axes fraction"),
                    xytext=(4, -14), textcoords="offset points",
                    fontsize=8, color=INK2, ha="left", va="top")
    k = int(np.argmin(np.abs(LAMBDAS - lam_one)))
    ax.plot([lam_one], [errs_one[k]], "o", color=S3, ms=9, mec=SURF, mew=1.6, zorder=4)
    ax.annotate(f"誤差最小 λ={lam_one:.2f}", (lam_one, errs_one[k]), fontsize=9,
                color=S3, ha="left", va="bottom", xytext=(8, 10),
                textcoords="offset points")
    ax.set_xlabel("λ(0 = 現行 / 1 = 時間比例)", fontsize=9)
    ax.set_ylabel("区間配分の誤差(案件あたりL1・小さいほど良い)", fontsize=9)
    ax.set_title("λ を入れると配分は当たるようになるか", fontsize=11, color=INK, loc="left")
    ax.legend(fontsize=9, frameon=False, loc="upper left")

    # (3) 区間種別ごとの 月数 と 実績密度
    ax = axes[2]
    for i, k in enumerate(kinds):
        sub = d[d["区間"] == k]
        ax.scatter(sub["月数"], sub["実績密度"], s=46, color=CYCLE[i % len(CYCLE)],
                   marker=MARKERS[i % len(MARKERS)], edgecolor=SURF, linewidth=1.2,
                   label=k, zorder=3)
    ax.axhline(1.0, color="#8f8e88", lw=1.4, ls="--", zorder=2)
    ax.annotate("時間比例(密度1.0)", (ax.get_xlim()[1], 1.0), fontsize=8, color=INK2,
                ha="right", va="bottom")
    ax.set_xlabel("区間の長さ(ヶ月)", fontsize=9)
    ax.set_ylabel("実際に消化された密度", fontsize=9)
    ax.set_yscale("log")
    ax.set_title("短い区間は本当に軽いか", fontsize=11, color=INK, loc="left")
    ax.legend(fontsize=7.5, frameon=False)

    # (4) 自己テスト ─ 仕込んだ λ を戻せるか
    ax = axes[3]
    if st is None:
        ax.axis("off")
    else:
        xt = st["仕込んだλ"].to_numpy()
        ax.plot([0, 0.85], [0, 0.85], color="#8f8e88", lw=1.4, ls="--", zorder=2,
                label="完全に戻せた場合")
        ax.plot(xt, st["推定λ(誤差最小)"].to_numpy(), "-o", color=S3, lw=2.2, ms=7,
                mec=SURF, mew=1.4, zorder=4, label="誤差最小による推定(採用)")
        ax.plot(xt, st["推定λ(回帰)"].to_numpy(), "-s", color=S8, lw=2.0, ms=7,
                mec=SURF, mew=1.4, zorder=3, label="log-log回帰による推定(不採用)")
        if floor is not None:
            ax.axhline(floor, color=INK2, lw=1.0, ls=":", zorder=1)
            ax.annotate(f"ノイズ床 {floor:.2f}", (0.85, floor), fontsize=8, color=INK2,
                        ha="right", va="bottom")
        ax.set_xlabel("仕込んだ λ(真値)", fontsize=9)
        ax.set_ylabel("推定された λ", fontsize=9)
        ax.set_title("推定手法そのものは当てになるか", fontsize=11, color=INK, loc="left")
        ax.legend(fontsize=8.5, frameon=False, loc="upper left")

    for ax in axes:
        ax.grid(lw=0.6)
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    fig.suptitle("マイルストーン区間への工数配分は、区間の長さにどれだけ引きずられるか",
                 fontsize=12, color=INK)
    fig.tight_layout(rect=[0, 0, 1, 0.955])
    path = os.path.join(out, "interval_elasticity.png")
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="区間配分が区間長にどれだけ従うかを推定する")
    p.add_argument("--master", default=os.path.join(ROOT, "data", "master.xlsx"))
    p.add_argument("--actuals", default=os.path.join(ROOT, "data", "actuals.csv"))
    p.add_argument("--out", default=os.path.join(ROOT, "output"))
    p.add_argument("--no-selftest", action="store_true",
                   help="自己テストを省く。ノイズ床が出ないので判定はできなくなる")
    p.add_argument("--boot", type=int, default=2000, help="ブートストラップ反復数。0で省略")
    a = p.parse_args(argv)

    setup_style()
    os.makedirs(a.out, exist_ok=True)
    ds, groups, curves, cfg = load(a.master, a.actuals)
    probe = build_model(curves, groups, cfg)
    print()
    print("=" * 78)
    print(f"学習案件 {len(curves)} 件 / 背骨: {', '.join(probe.backbone) or '(なし)'}")
    if not probe.backbone:
        print("背骨マイルストーンが選ばれていません。区間が作れないため検証できません。")
        return 0

    st, floor = None, None
    if not a.no_selftest:
        print()
        print("=" * 78)
        print("自己テスト ─ 既知の λ で作った疑似実績から、その λ が戻るか")
        print("=" * 78)
        st = selftest(ds, groups, curves, cfg, one_sided=True)
        print(st.to_string(index=False))
        floor = float(st.loc[st["仕込んだλ"] == 0.0, "推定λ(誤差最小)"].iloc[0])
        print(f"  λ=0 を仕込んでも {floor:.2f} が返る。これがこのデータでの推定の下駄"
              f"(ノイズ床)で、")
        print(f"  実データの推定値がこれを超えない限り『区間長に引きずられている』とは言えない。")
        print("  ※ 回帰による推定は仕込んだ値の半分以下しか返さない。回帰は参考値に留めること。")

    d = collect(ds, groups, curves, cfg)
    pd.set_option("display.width", 200)
    print()
    print("=" * 78)
    print("区間ごとの比率(leave-one-out)")
    print("=" * 78)
    print(d.round(3).to_string(index=False))

    lam_reg, se, r, n = fit_loglog(d)
    lam_one, errs_one = fit_grid(d, one_sided=True)
    lam_both, errs_both = fit_grid(d, one_sided=False)
    k = int(np.argmin(np.abs(LAMBDAS - lam_one)))

    print()
    print("=" * 78)
    print("推定")
    print("=" * 78)
    print(f"  log-log 回帰(モデルB)   λ = {lam_reg:.3f}  "
          f"(傾き {1 - lam_reg:.3f} ± {se:.3f} / 相関 {r:.3f} / 区間 {n} 個)")
    print(f"  配分誤差の最小化 モデルB λ = {lam_both:.2f}  "
          f"誤差 {errs_both.min():.4f}(λ=0 は {errs_both[0]:.4f})")
    print(f"  配分誤差の最小化 モデルC λ = {lam_one:.2f}  "
          f"誤差 {errs_one[k]:.4f}(λ=0 は {errs_one[0]:.4f})")
    gain = (errs_one[0] - errs_one[k]) / errs_one[0] * 100 if errs_one[0] else 0.0
    print(f"  → モデルC を採ると、区間配分の誤差が {gain:.1f}% 減る")

    lo = hi = None
    if a.boot:
        lo, hi = bootstrap_lambda(d, one_sided=True, n_iter=a.boot)
        print(f"  ブートストラップ95%区間(モデルC / 案件単位 {a.boot}回): "
              f"λ ∈ [{lo:.2f}, {hi:.2f}]")

    print()
    print("=" * 78)
    print("判定")
    print("=" * 78)
    if floor is None:
        print("  自己テストを省いたためノイズ床が不明。判定できない。")
    elif lam_one <= floor + 1e-9:
        print(f"  推定 λ {lam_one:.2f} ≦ ノイズ床 {floor:.2f}")
        print("  → 区間配分が区間長に引きずられている証拠は無い。このデータでは案A(現行)が妥当。")
    elif lo is not None and lo <= 0.0:
        print(f"  推定 λ {lam_one:.2f} > ノイズ床 {floor:.2f} だが、95%区間が 0 を含む")
        print("  → 傾向はあるが、案件数が足りず言い切れない。案件を増やして再測定すること。")
    else:
        print(f"  推定 λ {lam_one:.2f} > ノイズ床 {floor:.2f}、95%区間も 0 を含まない")
        print(f"  → 区間配分は区間長に引きずられている。λ={lam_one:.2f} で実装する価値がある。")

    print()
    print("区間種別ごとの内訳(実績密度 = 実績比率 ÷ 時間比率。1.0 が時間比例)")
    print(d.groupby("区間").agg(件数=("案件ID", "count"), 月数_中央=("月数", "median"),
                                モデル密度=("モデル密度", "mean"),
                                実績密度=("実績密度", "mean")).round(3).to_string())

    path = draw(d, errs_one, errs_both, lam_one, lam_both, a.out, st, floor)
    print()
    print("図:", path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
