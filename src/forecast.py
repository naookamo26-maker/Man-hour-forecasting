"""
予測ロジック(設計書 6章)

Step 1. 総時間の決定       契約人月 × 換算係数
Step 2. マイルストーン日付の予測  学習した位置分布 + 人による上書き
Step 3. カーブの貼り付け    正準軸の学習カーブを実日付軸へ逆変換
Step 4. 月次への離散化      暦月で切り出し、合計が総時間に一致するよう正規化

案A(設計書 3-1)の帰結として、どのモードでも
「消化率そのものは動かさず、時間軸だけが伸縮する」。
期間が延びても比率は学習値のまま、密度が薄くなるだけである。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .learning import Model
from .timeaxis import (Warp, canonical_to_monthly, date_to_t, month_edges,
                       month_list, t_to_date)


@dataclass
class Forecast:
    pid: str
    name: str
    months: list[str]
    table: pd.DataFrame        # index=月, columns=行程グループ, 値=時間
    milestones: pd.DataFrame   # マイルストーン名 / 予測日付 / 位置t / 根拠
    total_hours: float
    hours_per_mm: float
    warp: Warp
    notes: list[str]

    @property
    def groups(self) -> list[str]:
        """実際に予測した行程グループ。除外されたものは含まない。"""
        return list(self.table.columns)

    def as_manmonth(self) -> pd.DataFrame:
        return self.table / self.hours_per_mm

    def cumulative_ratio(self) -> np.ndarray:
        tot = self.table.sum(axis=1).to_numpy()
        return np.cumsum(tot) / tot.sum()


def _safe_positive(v) -> float | None:
    """正の数として読めれば返す。空欄・文字列・0以下は None。"""
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if np.isfinite(f) and f > 0 else None


def resolve_milestone_positions(model: Model, ds, pid: str, months: list[str],
                                use_given: bool = True) -> tuple[dict, dict]:
    """対象案件のマイルストーン位置(経過期間比)を決める(設計書 6 Step2)。

    人が指定した日付があればそれを正とし、無いものは学習した平均位置を使う。
    「マイルストーンが決まっている / いない」が同じ仕組みの入力有無だけで切り替わる。
    """
    given = {}
    if use_given:
        ms = ds.milestones_of(pid)
        for _, r in ms.iterrows():
            given[r["マイルストーン名"]] = date_to_t(months, r["日付"])

    positions, source = {}, {}
    for _, r in model.ms_stats.iterrows():
        nm = r["マイルストーン名"]
        if nm in given:
            positions[nm] = given[nm]
            source[nm] = "指定"
        else:
            positions[nm] = float(r["平均位置"])
            source[nm] = f"学習(n={int(r['件数'])})"
    # 学習側に存在しないマイルストーンを人が指定した場合も尊重する
    for nm, t in given.items():
        if nm not in positions:
            positions[nm] = t
            source[nm] = "指定(学習データに無し)"
    return positions, source


def forecast(model: Model, ds, pid: str, *,
             use_given_milestones: bool = True,
             group_totals: dict[str, float] | None = None,
             months_override: list[str] | None = None,
             exclude_groups: list[str] | None = None) -> Forecast:
    """1案件の 月 × 行程グループ 工数を予測する。

    exclude_groups にその案件が行わない行程グループを渡すと、
    その列を予測から外す。総工数(契約人月 × 換算係数)は動かさず、
    残ったグループへ学習比率のまま配分し直す。
    案件によってサウンドが外注、ローカライズ無しといった差があるため、
    全案件の平均構成をそのまま当てると、やらない業務に工数が乗り、
    やる業務が過小になる(設計書 6 Step1 の見積もり分類指定と同じ狙い)。
    """
    prow = ds.project(pid)
    try:
        months = months_override or month_list(prow["開始"], prow["終了"])
    except Exception as e:
        raise ValueError(
            f"{pid} の期間を解釈できません(開始={prow['開始']!r} 終了={prow['終了']!r}): {e}"
        ) from e
    if not months:
        raise ValueError(f"{pid} の期間が空です。projects シートの 開始・終了 を確認してください。")
    edges = month_edges(months)
    notes: list[str] = []

    # --- Step 1: 総時間とグループ配分(設計書 6 Step1 の制約の階層) ---
    #   分類ごとの見積もりあり → その値をそのままグループ別総量にする
    #   全体の契約人月のみ     → 学習した工数比率で配分する
    # 見積もりに現れないグループは「この案件では行わない業務」として列ごと落とす。
    exclude = {str(g) for g in (exclude_groups or [])}
    unknown = sorted(exclude - set(model.groups))
    if unknown:
        notes.append(f"除外指定のうち、学習側に存在しない行程グループは無視しました: {', '.join(unknown)}")

    if group_totals:
        est = pd.Series({g: float(v) for g, v in group_totals.items()
                         if str(g) in model.groups}, dtype=float)
        est = est[est > 0]
        if est.empty:
            raise ValueError(
                f"{pid} の見積もりが空、または行程グループ名が学習側と一致しません。"
                f"利用できる行程グループ: {', '.join(model.groups)}")
        # 見積もりが出ているグループだけを対象にする(未記入 = 行わない業務)
        groups = [g for g in model.groups if g in est.index and g not in exclude]
        if not groups:
            raise ValueError(f"{pid} は見積もりのある行程グループがすべて除外されました。")
        gt = est.reindex(groups)
        total_hours = float(gt.sum())
        notes.append("行程グループ別の総量は見積もりの指定値をそのまま使っています"
                     "(契約人月ではなく見積もりの合計が総工数になります)。")
        # 契約人月が入っていれば突き合わせる。どちらが正かは人が決めることなので、
        # 黙ってどちらかに寄せず、差だけを見せる。
        mm = _safe_positive(prow.get("契約人月"))
        if mm is not None:
            gap = total_hours / (mm * model.hours_per_mm) - 1.0
            if abs(gap) >= 0.01:
                notes.append(
                    f"見積もり合計 {total_hours / model.hours_per_mm:,.1f} 人月 は "
                    f"契約人月 {mm:,.1f} 人月 と {gap:+.1%} ずれています。"
                    "総量は見積もり側を採用しました。")
    else:
        # 契約人月が空欄でも例外にならず、表全体が空白のExcelが出来上がってしまう。
        # 黙って空の成果物を出すのが一番たちが悪いので、ここで止める。
        mm = _safe_positive(prow.get("契約人月"))
        if mm is None:
            raise ValueError(
                f"{pid} の契約人月が未設定または不正です(値: {prow['契約人月']!r})。"
                "予測対象の総工数を決められないため、projects シートの 契約人月 を入力するか、"
                "estimates シートに行程グループ別の見積もりを入力してください。")
        total_hours = mm * model.hours_per_mm
        groups = [g for g in model.groups if g not in exclude]
        if not groups:
            raise ValueError(f"{pid} は行程グループがすべて除外され、予測できません。")
        ratio = model.group_ratio.reindex(groups).fillna(0.0)
        if ratio.sum() <= 0:
            raise ValueError(
                f"{pid} は残った行程グループの学習比率がすべて 0 です: {', '.join(groups)}")
        # 除外した分は残りのグループへ、学習比率の割合を保ったまま配分し直す。
        gt = ratio / ratio.sum() * total_hours
        notes.append("行程グループ間の総量は学習した工数比率で配分しています。")

    dropped = [g for g in model.groups if g not in groups]
    if dropped:
        notes.append(f"この案件が行わない行程グループとして除外: {', '.join(dropped)}")

    # --- Step 2: マイルストーン位置 ---
    positions, source = resolve_milestone_positions(model, ds, pid, months,
                                                    use_given=use_given_milestones)

    # --- Step 3: ワープを組み、カーブを貼る ---
    if model.align and model.backbone:
        pairs = [(nm, model.canonical_anchor[nm], positions[nm])
                 for nm in model.backbone if nm in positions]
        warp = Warp.build(pairs)
        given_names = [nm for nm in model.backbone if source.get(nm) == "指定"]
        if given_names:
            notes.append(f"指定されたマイルストーンで時間軸を固定: {', '.join(given_names)}")
        else:
            notes.append("マイルストーンの指定が無いため、学習カーブをそのまま期間に貼っています。")
    else:
        warp = Warp.identity()
        notes.append("位置合わせ OFF(素朴版)。学習カーブをそのまま期間に貼っています。")

    # --- Step 4: 月次への離散化と正規化 ---
    cols = {}
    for g in groups:
        monthly = canonical_to_monthly(model.shape[g], edges, warp)
        s = monthly.sum()
        cols[g] = monthly / s * float(gt[g]) if s > 0 else np.zeros(len(months))
    table = pd.DataFrame(cols, index=pd.Index(months, name="月"))[groups]

    # 丸め誤差の吸収。合計は必ず総時間に一致させる。
    diff = total_hours - table.to_numpy().sum()
    if abs(diff) > 1e-6:
        biggest = table.sum(axis=1).idxmax()
        table.loc[biggest, groups[int(np.argmax(gt.to_numpy()))]] += diff

    # --- マイルストーン予測日付 ---
    ms_rows = []
    for nm, t_can in positions.items():
        t_act = float(warp.to_actual(t_can)) if source[nm] != "指定" else float(t_can)
        ms_rows.append({
            "マイルストーン名": nm,
            "予測日付": t_to_date(months, t_act).strftime("%Y-%m-%d"),
            "位置t": round(t_act, 3),
            "根拠": source[nm],
        })
    # マイルストーンが1件も無くても動く必要がある(設計書 3-2)
    ms_cols = ["マイルストーン名", "予測日付", "位置t", "根拠"]
    ms_df = pd.DataFrame(ms_rows, columns=ms_cols)
    if not ms_df.empty:
        ms_df = ms_df.sort_values("位置t").reset_index(drop=True)
    else:
        notes.append("マイルストーンの記録が1件も無いため、位置合わせは行っていません。")

    return Forecast(pid=pid, name=str(prow["名称"]), months=months, table=table,
                    milestones=ms_df, total_hours=total_hours,
                    hours_per_mm=model.hours_per_mm, warp=warp, notes=notes)
