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

from .elasticity import describe as describe_elasticity, inflate
from .learning import Model
from .ramp import apply_ramp
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
                                use_given: bool = True,
                                known_milestones: set[str] | None = None
                                ) -> tuple[dict, dict]:
    """対象案件のマイルストーン位置(経過期間比)を決める(設計書 6 Step2)。

    人が指定した日付があればそれを正とし、無いものは学習した平均位置を使う。
    「マイルストーンが決まっている / いない」が同じ仕組みの入力有無だけで切り替わる。

    known_milestones を渡すと、そこに載っている名前の日付だけを使う。
    途中まで実績が確定している時点を再現するときに要る。
    その時点ではまだ通過していないマイルストーンの実日付は分かっておらず、
    渡してしまうと未来の情報で位置合わせすることになる(src/phased.py)。
    """
    given = {}
    if use_given:
        ms = ds.milestones_of(pid)
        for _, r in ms.iterrows():
            nm = r["マイルストーン名"]
            if known_milestones is not None and nm not in known_milestones:
                continue
            given[nm] = date_to_t(months, r["日付"])

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
             known_milestones: set[str] | None = None,
             group_totals: dict[str, float] | None = None,
             months_override: list[str] | None = None,
             exclude_groups: list[str] | None = None,
             ramp_limit: float | None = None) -> Forecast:
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
    positions, source = resolve_milestone_positions(
        model, ds, pid, months, use_given=use_given_milestones,
        known_milestones=known_milestones)

    # --- Step 3: ワープを組み、カーブを貼る ---
    if model.align and model.backbone:
        pairs = [(nm, model.canonical_anchor[nm], positions[nm])
                 for nm in model.backbone if nm in positions]
        warp = Warp.build(pairs, strength=model.warp_strength,
                          max_stretch=model.max_stretch)
        if warp.clipped:
            notes.append(
                "マイルストーンの日付が近すぎる/順序が逆のため、時間軸上の位置を"
                f"補正しました: {', '.join(warp.clipped)}。"
                "補正で潰れた区間には工数が集中し、月次に大きな跳ねが出ます。"
                "milestones シートの日付を確認してください。")
        if not warp.is_identity and warp.max_step_ratio >= 2.0:
            notes.append(
                f"マイルストーンの間隔が学習データの平均から離れているため、"
                f"時間軸の伸縮率がアンカーの前後で最大 {warp.max_step_ratio:.1f} 倍変わります"
                f"(最大密度倍率 {warp.max_density_gain:.1f} 倍)。"
                "その境目の月に工数の跳ねが出ます。"
                "settings の 区間弾力性 を上げると、マイルストーンの日付を動かさないまま"
                "狭い区間への配分を減らせます。位置合わせ強度 を下げる / 伸縮率上限 を"
                "設定する方法もありますが、そちらは山の位置が日付からずれます。")
        if model.warp_strength < 1.0 or (model.max_stretch or 0) > 1.0:
            notes.append(
                f"位置合わせを弱めて適用しています(位置合わせ強度 {model.warp_strength:g}"
                + (f" / 伸縮率上限 {model.max_stretch:g} 倍" if (model.max_stretch or 0) > 1.0 else "")
                + ")。跳ねを抑える代わりに、カーブの山はマイルストーンの日付ちょうどには来ません。")
        given_names = [nm for nm in model.backbone if source.get(nm) == "指定"]
        if given_names:
            notes.append(f"指定されたマイルストーンで時間軸を固定: {', '.join(given_names)}")
        else:
            notes.append("マイルストーンの指定が無いため、学習カーブをそのまま期間に貼っています。")
    else:
        warp = Warp.identity()
        notes.append("位置合わせ OFF(素朴版)。学習カーブをそのまま期間に貼っています。")

    # --- Step 3b: 潰れた区間への配分を減らす(設計書 3-1 の弾力化) ---
    # 学習データより極端に狭い区間は、案Aのままだと実在しない要員数を要求する。
    # その区間の総量だけを削り、あふれた分を他の区間へ回す。
    # 既定(0)では何も起きず、アンカーの位置は最後まで一切動かない。
    lam = float(getattr(model, "interval_elasticity", 0.0) or 0.0)
    if lam > 0 and not warp.is_identity:
        blend = sum(model.shape[g] * float(gt[g]) for g in groups)
        if blend.sum() > 0:
            desc = describe_elasticity(blend / blend.sum(), warp, lam)
            if desc:
                notes.append(
                    f"区間弾力性 {lam:g} により、学習データより狭く潰れた区間への"
                    f"配分を減らしました({desc})。"
                    "マイルストーンの日付とグループ別の総量は変えていません。")

    # --- Step 4: 月次への離散化と正規化 ---
    cols = {}
    for g in groups:
        monthly = canonical_to_monthly(inflate(model.shape[g], warp, lam), edges, warp)
        s = monthly.sum()
        cols[g] = monthly / s * float(gt[g]) if s > 0 else np.zeros(len(months))
    table = pd.DataFrame(cols, index=pd.Index(months, name="月"))[groups]

    # 立ち上がり制約(任意)。位置合わせは伸縮率がアンカーの前後で不連続に変わるため、
    # その境目に月次工数の段差や鋭い突起が出る。月次の増加率に上限を置くと、
    # 総量と行程グループ別の総量を動かさないまま、その尖りだけを均せる。
    # 既定では効かせない。全期間予測に効かせると位置合わせの評価そのものが変わるため、
    # 効果を実データで確かめてから採用すること(設計書 15章)。
    if ramp_limit and ramp_limit > 1.0 and len(table) >= 2:
        # 先頭月には縛る相手がいない。自分自身を起点にして2ヶ月目以降だけを縛る。
        base = float(table.iloc[0].sum()) / ramp_limit
        fitted, info = apply_ramp(table, base, ramp_limit)
        if info.get("適用"):
            table = fitted
            notes.append(
                f"立ち上がり上限 {ramp_limit:.2f} 倍/月 を適用しました"
                f"({info['制約に当たった月数']} ヶ月が上限に当たり、工数を後ろの月へ送っています)。"
                "総工数と行程グループ別の総量は変わりません。")

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
