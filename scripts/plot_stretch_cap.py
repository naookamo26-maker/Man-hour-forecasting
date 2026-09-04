"""伸縮率上限(および位置合わせ強度)の効き具合を、数字ではなく絵で確かめる。

`diagnose_warp.py` は跳ねと精度を数字で出す。だが「山がどう変わったか」は
数字の列を眺めても掴めない。特に

    ・マイルストーンの区間が狭いと、なぜ鋭い山ができるのか
    ・上限をかけると、その山はどちらへ流れるのか(前か、後ろか)
    ・そのとき、マイルストーンは指定した日付からどれだけ動いてしまうのか

は、月次カーブとアンカー位置を重ねて描かないと判断できない。
このスクリプトはその4枚を出す。

    python scripts/plot_stretch_cap.py            # 4枚とも
    python scripts/plot_stretch_cap.py --figs 2 3 # 2枚だけ

出力(既定 output/)
    warp_cap_1_実データ.png    完了案件の leave-one-out。実績と重ねた月次カーブ
    warp_cap_2_狭い区間.png    区間の狭さを人工的に作った3ケース。上限の効きが一番はっきり出る
    warp_cap_3_トレードオフ.png 上限 k を振ったときの「山の高さ」と「日付のずれ」
    warp_cap_4_精度.png        上限 k を振ったときの跳ねと WAPE。採用可否はここで決める

図2 は data/master.xlsx の予測対象案件のマイルストーン日付を **メモリ上でだけ**
差し替えて描く。ファイルは書き換えない。
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
from matplotlib import font_manager  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.forecast import forecast  # noqa: E402
from src.data_loader import load_all  # noqa: E402
from src.learning import (aggregate_actuals, build_project_curves, group_names,  # noqa: E402
                          learn)
from src.timeaxis import month_edges, month_list, t_to_date  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 色。実績は比較の基準なのでニュートラル、設定違いは識別色で分ける。
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e6e5e0", "#fcfcfb"
ACT = "#8f8e88"
S1, S2, S3, S4 = "#2a78d6", "#eb6834", "#1baf7a", "#4a3aa7"

# 図1・図2で比べる設定。(伸縮率上限, 凡例名, 色)
CAPS = [(None, "上限なし(現状)", S1), (1.5, "上限 1.5倍", S2), (1.3, "上限 1.3倍", S3)]
# 図3・図4で振る上限。None = 無制限
SWEEP = [1.1, 1.2, 1.3, 1.4, 1.5, 1.75, 2.0, 2.5, 3.0, None]

# 図2で人工的に作るマイルストーン配置。(見出し, {名前: 経過期間比}, 色)
NARROW_CASES = [
    ("(a) 標準的な配置 ─ α版 0.50 / β版 0.74 / マスターアップ 0.91",
     {"α版": 0.50, "β版": 0.74, "マスターアップ": 0.91}, S3),
    ("(b) 中間の区間が狭い ─ α版→β版 が 2ヶ月",
     {"α版": 0.62, "β版": 0.70, "マスターアップ": 0.91}, S2),
    ("(c) 終盤の区間が狭い ─ マスターアップが終了の1ヶ月前",
     {"α版": 0.50, "β版": 0.88, "マスターアップ": 0.965}, S4),
]


# ---------------------------------------------------------------------------
# 下ごしらえ
# ---------------------------------------------------------------------------
def setup_style() -> None:
    """日本語が出るフォントを選び、図全体の見た目を決める。

    フォント名は OS ごとに違う。見つからないと文字が全部豆腐になって
    図の意味が消えるため、候補を順に当たって最初に在るものを使う。
    """
    have = {f.name for f in font_manager.fontManager.ttflist}
    for name in ("IPAGothic", "Noto Sans CJK JP", "Yu Gothic", "Meiryo",
                 "Hiragino Sans", "MS Gothic", "TakaoGothic"):
        if name in have:
            plt.rcParams["font.family"] = name
            break
    else:
        print("[注意] 日本語フォントが見つかりません。図の文字が崩れます。")
    plt.rcParams.update({
        "axes.unicode_minus": False,
        "figure.facecolor": SURF, "axes.facecolor": SURF,
        "axes.edgecolor": "#c9c8c3", "axes.labelcolor": INK2,
        "xtick.color": INK2, "ytick.color": INK2, "text.color": INK,
        "grid.color": GRID,
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


def build_model(curves, groups, cfg, *, cap=None, strength=1.0, exclude=None):
    """exclude を除いて学習する(leave-one-out)。cap = 伸縮率上限。"""
    rest = {p: c for p, c in curves.items() if p != exclude} if exclude else curves
    return learn(rest, groups, align=True, n_bin=cfg["n_bin"],
                 backbone_spec=cfg["backbone_spec"],
                 backbone_coverage=cfg["backbone_coverage"],
                 hours_per_mm=cfg["hours_per_mm"],
                 warp_strength=strength, max_stretch=cap)


def anchor_positions(warp, months: list[str]) -> list[float]:
    """アンカーの実位置(経過期間比)を、月インデックスの座標に直す。

    縦線を月次グラフの上に重ねるために要る。月の幅は日数で決まるので、
    month_edges を通してから内挿する(単純な t × 月数 では最大半月ずれる)。
    """
    edges = month_edges(months)
    xs = np.arange(len(months) + 1) - 0.5
    return [float(np.interp(a, edges, xs)) for a in warp.anchors_actual[1:-1]]


def _finish(ax, months=None, every=3, ylabel="月次工数(人月)"):
    ax.set_ylabel(ylabel, fontsize=9)
    ax.grid(axis="y", lw=0.6)
    ax.set_axisbelow(True)
    ax.set_ylim(bottom=0)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    if months is not None:
        x = np.arange(len(months))
        ax.set_xticks(x[::every])
        ax.set_xticklabels(months[::every], fontsize=7.5, rotation=45, ha="right")


def _panel_note(ax, lines: list[str]) -> None:
    """パネル左上に、その図の数字を小さく置く。"""
    ax.text(0.015, 0.97, "\n".join(lines), transform=ax.transAxes, fontsize=8,
            color=INK2, va="top", ha="left",
            bbox=dict(fc=SURF, ec=GRID, lw=0.8, pad=4))


# ---------------------------------------------------------------------------
# 図1 実データ leave-one-out
# ---------------------------------------------------------------------------
def fig_real(ds, groups, curves, cfg, out: str, top: int = 4) -> str:
    """段差比の大きい案件について、実績と設定別の予測を重ねる。

    総量は実績に合わせて正規化する。ここで見たいのは総量の当たり外れではなく
    「山の形が上限でどう変わるか」だからである。
    """
    hpm = cfg["hours_per_mm"]
    order = []
    for pid in sorted(curves):
        m = build_model(curves, groups, cfg, cap=None, exclude=pid)
        fc = forecast(m, ds, pid, use_given_milestones=True,
                      months_override=curves[pid].months)
        order.append((fc.warp.max_step_ratio, pid))
    picks = [p for _, p in sorted(order, reverse=True)[:top]]

    ncol = 2
    nrow = int(np.ceil(len(picks) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(7.5 * ncol, 4.5 * nrow))
    for ax, pid in zip(np.ravel(axes), picks):
        months = curves[pid].months
        am = curves[pid].monthly.sum(axis=1).to_numpy() / hpm
        x = np.arange(len(months))
        ax.fill_between(x, am, color=ACT, alpha=0.22, zorder=1)
        ax.plot(x, am, color=ACT, lw=2.6, label="実績", zorder=2)
        notes = []
        for cap, lbl, col in CAPS:
            m = build_model(curves, groups, cfg, cap=cap, exclude=pid)
            fc = forecast(m, ds, pid, use_given_milestones=True, months_override=months)
            pm = fc.table.sum(axis=1).to_numpy() / hpm
            pm = pm * (am.sum() / pm.sum())
            ax.plot(x, pm, color=col, lw=2.0, label=lbl, zorder=3)
            notes.append(f"{lbl} ─ 段差比 {fc.warp.max_step_ratio:.2f} / "
                         f"WAPE {float(np.abs(pm - am).sum() / am.sum()):.3f}")
            if cap is None:
                for a, nm in zip(anchor_positions(fc.warp, months), fc.warp.names):
                    ax.axvline(a, color="#c9c8c3", lw=1.0, ls=":", zorder=1)
                    ax.annotate(nm, (a, 1.0), xycoords=("data", "axes fraction"),
                                fontsize=7.5, color=INK2, ha="center", va="bottom",
                                xytext=(0, 2), textcoords="offset points")
        ax.set_title(f"{pid}  {curves[pid].name[:22]}({len(months)}ヶ月)",
                     fontsize=10, color=INK, loc="left", pad=16)
        _panel_note(ax, notes)
        _finish(ax, months)
    h, l = np.ravel(axes)[0].get_legend_handles_labels()
    fig.legend(h, l, fontsize=10, frameon=False, ncol=len(l), loc="upper center",
               bbox_to_anchor=(0.5, 0.945))
    fig.suptitle("サンプルデータ leave-one-out ─ 段差比の大きい案件"
                 "(総量は実績に合わせて正規化)", fontsize=12, color=INK)
    fig.tight_layout(rect=[0, 0, 1, 0.915])
    path = os.path.join(out, "warp_cap_1_実データ.png")
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# 図2 区間の狭さを人工的に作る
# ---------------------------------------------------------------------------
def _target_pid(ds) -> str:
    t = ds.target_projects()
    if t.empty:
        raise SystemExit("ステータス=予測対象 の案件がありません(図2・図3が描けません)")
    return str(t.iloc[0]["案件ID"])


def _swap_milestones(ds, pid: str, months: list[str], ms_t: dict) -> None:
    """対象案件のマイルストーン日付をメモリ上で差し替える(ファイルは触らない)。"""
    keep = ds.milestones[ds.milestones["案件ID"] != pid]
    rows = [{"案件ID": pid, "マイルストーン名": nm, "日付": t_to_date(months, t)}
            for nm, t in ms_t.items()]
    ds.milestones = pd.concat([keep, pd.DataFrame(rows)], ignore_index=True)


def fig_narrow(ds, groups, curves, cfg, out: str) -> str:
    """区間の狭さが作る山と、上限がそれをどちらへ流すかを見る。"""
    hpm = cfg["hours_per_mm"]
    pid = _target_pid(ds)
    pr = ds.project(pid)
    months = month_list(str(pr["開始"]), str(pr["終了"]))
    mean_mm = float(pr["契約人月"]) / len(months)
    saved = ds.milestones.copy()
    models = {cap: build_model(curves, groups, cfg, cap=cap) for cap, _, _ in CAPS}

    fig, axes = plt.subplots(len(NARROW_CASES), 1, figsize=(13, 4.3 * len(NARROW_CASES)),
                             sharex=True)
    for ax, (label, ms_t, _) in zip(np.ravel(axes), NARROW_CASES):
        _swap_milestones(ds, pid, months, ms_t)
        x = np.arange(len(months))
        ax.axhline(mean_mm, color="#c9c8c3", lw=1.2, ls="--", zorder=1)
        ax.annotate(f"月平均 {mean_mm:.0f}人月", (len(months) - 0.3, mean_mm),
                    fontsize=8, color=INK2, ha="right", va="bottom")
        notes = []
        for cap, lbl, col in CAPS:
            fc = forecast(models[cap], ds, pid, use_given_milestones=True)
            pm = fc.table.sum(axis=1).to_numpy() / hpm
            k = int(np.argmax(pm))
            ax.plot(x, pm, color=col, lw=2.2, label=lbl, zorder=3)
            ax.plot([k], [pm[k]], "o", color=col, ms=7, mec=SURF, mew=1.6, zorder=4)
            ax.annotate(f"{pm[k]:.0f}", (k, pm[k]), fontsize=9, color=col, ha="center",
                        va="bottom", xytext=(0, 6), textcoords="offset points")
            notes.append(f"{lbl} ─ ピーク {pm[k]:.0f}人月(平均の {pm[k] / mean_mm:.1f}倍) / "
                         f"段差比 {fc.warp.max_step_ratio:.2f} / "
                         f"密度倍率 {fc.warp.max_density_gain:.2f}")
            # アンカーは両端の設定だけ引く。3本とも引くと線が飽和して読めない。
            if cap in (CAPS[0][0], CAPS[-1][0]):
                for a, nm in zip(anchor_positions(fc.warp, months), fc.warp.names):
                    ax.axvline(a, color=col, lw=1.1, ls=":", alpha=0.75, zorder=2)
                    if cap is CAPS[0][0]:
                        ax.annotate(nm, (a, 1.0), xycoords=("data", "axes fraction"),
                                    fontsize=8, color=INK2, ha="center", va="bottom",
                                    xytext=(0, 2), textcoords="offset points")
        ax.set_title(label, fontsize=11, color=INK, loc="left", pad=18)
        _panel_note(ax, notes)
        _finish(ax)
    ds.milestones = saved
    np.ravel(axes)[-1].set_xticks(np.arange(len(months))[::2])
    np.ravel(axes)[-1].set_xticklabels(months[::2], fontsize=8, rotation=45, ha="right")
    h, l = np.ravel(axes)[0].get_legend_handles_labels()
    fig.legend(h, l, fontsize=10, frameon=False, ncol=len(l), loc="upper center",
               bbox_to_anchor=(0.5, 0.945))
    fig.suptitle(
        f"マイルストーン区間の狭さが作る山と、伸縮率上限の効き ─ {pid}"
        f"({len(months)}ヶ月 / {float(pr['契約人月']):.0f}人月)\n"
        f"縦の点線 = 実際に貼られたアンカー位置({CAPS[0][1]} / {CAPS[-1][1]})。"
        "上限は山を削るが、アンカーも指定日付から動かす",
        fontsize=12, color=INK)
    fig.tight_layout(rect=[0, 0, 1, 0.925])
    path = os.path.join(out, "warp_cap_2_狭い区間.png")
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# 図3 トレードオフ
# ---------------------------------------------------------------------------
def fig_tradeoff(ds, groups, curves, cfg, out: str) -> str:
    """上限を絞るほど山は下がるが、マイルストーンは指定日付から離れていく。

    その2つを並べて、どこで折り合うかを目で決められるようにする。
    """
    hpm = cfg["hours_per_mm"]
    pid = _target_pid(ds)
    pr = ds.project(pid)
    months = month_list(str(pr["開始"]), str(pr["終了"]))
    n = len(months)
    mean_mm = float(pr["契約人月"]) / n
    saved = ds.milestones.copy()
    models = {cap: build_model(curves, groups, cfg, cap=cap) for cap in SWEEP}

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.2))
    xs = np.arange(len(SWEEP))
    for label, ms_t, col in NARROW_CASES:
        _swap_milestones(ds, pid, months, ms_t)
        peak, shift = [], []
        for cap in SWEEP:
            fc = forecast(models[cap], ds, pid, use_given_milestones=True)
            pm = fc.table.sum(axis=1).to_numpy() / hpm
            given = dict(zip(fc.milestones["マイルストーン名"], fc.milestones["位置t"]))
            # 指定した位置と、実際に貼られた位置の差。月に直すと大きさが実感できる。
            d = [abs(a - given[nm]) * n for nm, a
                 in zip(fc.warp.names, fc.warp.anchors_actual[1:-1]) if nm in given]
            peak.append(pm.max() / mean_mm)
            shift.append(max(d) if d else 0.0)
        short = label.split(" ─ ")[0]
        axes[0].plot(xs, peak, "-o", color=col, lw=2.0, ms=6, mec=SURF, mew=1.4, label=short)
        axes[1].plot(xs, shift, "-o", color=col, lw=2.0, ms=6, mec=SURF, mew=1.4, label=short)
    ds.milestones = saved

    axes[0].axhline(1.0, color="#c9c8c3", lw=1.0, ls="--")
    axes[0].set_title("上限を絞るほど山は下がる", fontsize=11, color=INK, loc="left")
    axes[1].set_title("同時に、マイルストーンが指定日付から離れる", fontsize=11,
                      color=INK, loc="left")
    for ax, ylab in zip(axes, ["ピーク月次工数 ÷ 月平均(倍)",
                               "指定日付からのずれ(ヶ月・最大)"]):
        ax.set_xticks(xs)
        ax.set_xticklabels([f"{k:g}" if k else "なし" for k in SWEEP], fontsize=9)
        ax.set_xlabel("伸縮率上限 k(倍)", fontsize=9)
        _finish(ax, ylabel=ylab)
        ax.legend(fontsize=9, frameon=False)
    fig.suptitle("伸縮率上限のトレードオフ ─ 山の高さ と マイルストーン位置の忠実さ",
                 fontsize=12, color=INK)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    path = os.path.join(out, "warp_cap_3_トレードオフ.png")
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# 図4 精度
# ---------------------------------------------------------------------------
def fig_accuracy(ds, groups, curves, cfg, out: str) -> str:
    """上限を絞ったとき、跳ねと予測精度が実データでどう動くか。

    図2・図3 は人工的な配置なので「効くこと」しか言えない。
    採用してよいかどうかは、実データ leave-one-out の WAPE で決める。
    """
    hpm = cfg["hours_per_mm"]
    rows = []
    for cap in SWEEP:
        for pid in sorted(curves):
            m = build_model(curves, groups, cfg, cap=cap, exclude=pid)
            fc = forecast(m, ds, pid, use_given_milestones=True,
                          months_override=curves[pid].months)
            am = curves[pid].monthly.sum(axis=1).to_numpy()
            pm = fc.table.sum(axis=1).to_numpy()
            pm = pm * (am.sum() / pm.sum())
            sa = float(np.abs(np.diff(am)).max())
            sp = float(np.abs(np.diff(pm)).max())
            rows.append({"k": cap if cap else np.inf, "段差比": fc.warp.max_step_ratio,
                         "跳ね度": sp / sa if sa else np.nan,
                         "WAPE": float(np.abs(pm - am).sum() / am.sum())})
    g = (pd.DataFrame(rows).groupby("k")
         .agg(段差比_最悪=("段差比", "max"), 跳ね度_平均=("跳ね度", "mean"),
              跳ね度_最悪=("跳ね度", "max"), WAPE_平均=("WAPE", "mean"),
              WAPE_最悪=("WAPE", "max")).reset_index().sort_values("k"))
    print()
    print(g.to_string(index=False))

    xs = np.arange(len(SWEEP))
    panels = [(["段差比_最悪"], "跳ねの大きさ(構造)", "アンカー段差比(最悪案件)"),
              (["跳ね度_平均", "跳ね度_最悪"], "跳ねの大きさ(結果)",
               "跳ね度 = 予測の月次変化幅 ÷ 実績"),
              (["WAPE_平均", "WAPE_最悪"], "予測精度(小さいほど良い)", "WAPE")]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
    for ax, (cols, title, ylab) in zip(axes, panels):
        for c, col in zip(cols, [S1, S2]):
            ax.plot(xs, g[c].to_numpy(), "-o", color=col, lw=2.0, ms=6, mec=SURF,
                    mew=1.4, label=c.replace("_", " "))
        ax.set_xticks(xs)
        ax.set_xticklabels([f"{k:g}" if k else "なし" for k in SWEEP], fontsize=9)
        ax.set_xlabel("伸縮率上限 k(倍)", fontsize=9)
        ax.set_title(title, fontsize=11, color=INK, loc="left")
        _finish(ax, ylabel=ylab)
        ax.legend(fontsize=9, frameon=False)
    axes[1].axhline(1.0, color="#c9c8c3", lw=1.0, ls="--")
    fig.suptitle(f"サンプルデータ(完了{len(curves)}案件)leave-one-out ─ "
                 "上限を絞ったときの跳ねと精度", fontsize=12, color=INK)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    path = os.path.join(out, "warp_cap_4_精度.png")
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="伸縮率上限の効き具合を図にする")
    p.add_argument("--master", default=os.path.join(ROOT, "data", "master.xlsx"))
    p.add_argument("--actuals", default=os.path.join(ROOT, "data", "actuals.csv"))
    p.add_argument("--out", default=os.path.join(ROOT, "output"))
    p.add_argument("--figs", type=int, nargs="+", choices=[1, 2, 3, 4],
                   default=[1, 2, 3, 4], help="描く図の番号")
    a = p.parse_args(argv)

    setup_style()
    os.makedirs(a.out, exist_ok=True)
    ds, groups, curves, cfg = load(a.master, a.actuals)
    builders = {1: fig_real, 2: fig_narrow, 3: fig_tradeoff, 4: fig_accuracy}
    for n in a.figs:
        print(f"図{n} を作成中...")
        print("  ->", builders[n](ds, groups, curves, cfg, a.out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
