"""
段階予測(途中まで実績が確定している場合の予測)

「着手前に全期間を予測する」だけでは、予測が当たっているかどうかを
案件が終わるまで判定できない。実運用では案件は途中まで進んでおり、
そこまでの実績は確定値として分かっている。知りたいのは常に
「ここまでは実績で確定した。では残りはどうなるか」である。

このモジュールは、その予測を完了案件の上で再現する。
マイルストーンを区切りにして、

    段階0  着手前            実績を一切使わず全期間を予測(従来の予測と同じ)
    段階1  プロトタイプ後    その日付が属する月までを実績で確定し、残りを予測
    段階2  α版後            さらに先まで確定し、残りを予測
    ...

を並べて出す。完了案件なら全期間の実績が分かっているので、
どの段階の予測がどれだけ当たったかをそのまま数字にできる。
段階が進むほど誤差が小さくなっていれば、予測は情報を正しく使えている。
逆に情報を足しても改善しないなら、その予測は形を当てていない。

── 残工数の決め方(2通り。どちらが正しいかはデータが決める)

  契約総量固定  総量は契約人月(または見積もり)のまま動かさない。
                残り = 総量 − 確定分。現場の運用に一番近い。
                実績が総量を超えていれば残りは 0 になる。
  実績スケール  確定分の実績と、学習カーブが「その時点までに消化するはず」と
                言う比率から総量そのものを引き直す。
                総量 = 確定分 ÷ 学習カーブの確定分比率。
                契約人月が実態とずれている案件で効く。

── 情報漏れの防止

学習データに予測対象そのものが入っていると、自分の実績で学習した
カーブで自分を予測することになり、評価にならない。
このモジュールは対象案件を必ず学習から外した(leave-one-out)モデルを使う。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .forecast import forecast
from .learning import ProjectCurve, learn
from .timeaxis import RECON_BOX

BASIS_FIXED = "契約総量固定"
BASIS_SCALED = "実績スケール"
BASIS_MODES = (BASIS_FIXED, BASIS_SCALED)


@dataclass
class Stage:
    """1つの区切り(= 段階)の予測結果。"""

    idx: int                    # 0 = 着手前
    label: str                  # グラフの系列名
    milestone: str              # 区切りにしたマイルストーン名("" = 着手前)
    cut_month: str | None       # ここまでを実績で確定。None = 確定なし
    n_fixed: int                # 確定した月数
    known_milestones: list[str]  # その時点で日付が分かっているマイルストーン
    table: pd.DataFrame         # 月 × 行程グループ。確定分は実績、残りは予測
    pred_only: pd.DataFrame     # 残り区間の予測だけ(確定分は 0)
    total_hours: float          # 段階時点で見込んだ案件総工数
    notes: list[str] = field(default_factory=list)

    @property
    def monthly(self) -> pd.Series:
        return self.table.sum(axis=1)


@dataclass
class PhasedResult:
    pid: str
    name: str
    months: list[str]
    groups: list[str]
    actual: pd.DataFrame        # 月 × 行程グループ の実績(全期間)
    eval_last: int              # 実績が存在する最後の月の位置。ここまでしか答え合わせできない
    stages: list[Stage]
    basis: str
    series: pd.DataFrame        # 月 × 系列(実績 + 各段階)の合計時間
    metrics: pd.DataFrame       # 段階ごとの誤差
    warnings: list[str] = field(default_factory=list)

    @property
    def actual_total(self) -> float:
        return float(self.actual.to_numpy().sum())


def _cut_index(months: list[str], date) -> int:
    """日付が属する月の位置を返す。期間より前なら -1、期間より後なら最終月。"""
    key = pd.Period(pd.Timestamp(date), freq="M")
    keys = [pd.Period(m, freq="M") for m in months]
    if key < keys[0]:
        return -1
    for i, k in enumerate(keys):
        if k >= key:
            return i
    return len(months) - 1


def _stage_points(ds, pid: str, months: list[str]) -> list[tuple[str, pd.Timestamp, int]]:
    """区切りに使うマイルストーンを (名前, 日付, 確定月index) で返す。

    同じ月に落ちるマイルストーンは確定範囲が同じ = 同じ段階になるため、
    先に来たものだけを残す。最終月まで確定してしまう区切り(予測する残りが無い)も外す。
    """
    ms = ds.milestones_of(pid)
    if ms.empty:
        return []
    rows = []
    seen: set[int] = set()
    for _, r in ms.sort_values("日付").iterrows():
        ci = _cut_index(months, r["日付"])
        if ci < 0 or ci >= len(months) - 1:
            continue
        if ci in seen:
            continue
        seen.add(ci)
        rows.append((str(r["マイルストーン名"]), pd.Timestamp(r["日付"]), ci))
    return rows


def _known_names(ds, pid: str, cut_date) -> list[str]:
    """その時点で日付が確定しているマイルストーン名。

    段階kの時点で人が知っているのは、通過済みのマイルストーンの実日付だけ。
    先のマイルストーンの日付を渡してしまうと未来を覗くことになるので外す。
    """
    ms = ds.milestones_of(pid)
    if ms.empty:
        return []
    hit = ms[ms["日付"] <= pd.Timestamp(cut_date)]
    return sorted({str(v) for v in hit["マイルストーン名"]})


def _metrics(pred: pd.Series, act: pd.Series, n_fixed: int, eval_last: int) -> dict:
    """段階の予測を実績と突き合わせる。

    評価するのは「まだ確定していない区間」。確定分は実績そのものなので
    誤差ゼロで当たり前であり、そこを混ぜると段階が進むほど自動的に
    誤差が下がってしまい、予測が良くなったのか確定が増えただけなのか区別できない。

    eval_last は実績が存在する最後の月の位置。進行中案件では期間の末尾に
    まだ実績が無く、そこを 0 として突き合わせると「予測が過大」と誤って出る。
    実績が無い月は正解が無いだけであって、正解が 0 なのではない。
    """
    p = pred.to_numpy(dtype=float)
    a = act.to_numpy(dtype=float)
    hi = eval_last + 1                       # 評価に使える範囲 [0, hi)
    rp, ra = p[n_fixed:hi], a[n_fixed:hi]
    out: dict = {}

    ra_sum = float(ra.sum())
    a_sum = float(a[:hi].sum())
    out["残り月数"] = int(len(p) - n_fixed)
    out["評価した月数"] = int(len(ra))
    out["残り実績(時間)"] = round(ra_sum, 1)
    # 段階が進むほど残り区間は短く、量も小さくなる。誤差率はその小さい分母で割った値なので、
    # 終盤の段階の % は数字が跳ねやすい。どれだけの量に対する誤差なのかを併記する。
    out["残りが全体に占める割合(%)"] = round(ra_sum / a_sum * 100, 1) if a_sum > 0 else None
    out["残り予測(時間)"] = round(float(rp.sum()), 1)
    out["残り総量誤差(%)"] = (round((rp.sum() / ra_sum - 1.0) * 100, 1)
                              if ra_sum > 0 else None)
    out["残り月次WAPE"] = (round(float(np.abs(rp - ra).sum() / ra_sum), 4)
                           if ra_sum > 0 else None)
    out["残りピーク月ズレ"] = (int(np.argmax(rp) - np.argmax(ra))
                              if len(ra) and ra_sum > 0 else None)

    # 累積カーブは評価できる範囲の全体で見る。確定分を含めた
    # 「案件全体の消化の形」がどれだけ実績に寄ったかを表す。
    # 量ではなく形の指標なので各々の合計で正規化する。
    pe, ae = p[:hi], a[:hi]
    if pe.sum() > 0 and ae.sum() > 0:
        fp = np.cumsum(pe) / pe.sum()
        fa = np.cumsum(ae) / ae.sum()
        out["累積カーブ最大乖離"] = round(float(np.abs(fp - fa).max()), 4)
    else:
        out["累積カーブ最大乖離"] = None
    out["総量誤差(%)"] = (round((pe.sum() / ae.sum() - 1.0) * 100, 1)
                          if ae.sum() > 0 else None)
    return out


def phased_forecast(ds, curves: dict[str, ProjectCurve], groups: list[str],
                    agg: pd.DataFrame, pid: str, *,
                    align: bool = True, n_bin: int = 100,
                    backbone_spec: str = "自動", backbone_coverage: float = 0.6,
                    hours_per_mm: float = 160.0, recon: str = RECON_BOX,
                    warp_strength: float = 1.0, max_stretch: float | None = None,
                    basis: str = BASIS_FIXED,
                    group_totals: dict[str, float] | None = None,
                    exclude_groups: list[str] | None = None) -> PhasedResult:
    """1案件について、マイルストーンごとに実績を確定させた段階予測を作る。

    curves は build_project_curves の結果(= 完了案件のみ)。
    対象案件がそこに含まれていれば必ず学習から外す。
    """
    if basis not in BASIS_MODES:
        raise ValueError(f"残工数の決め方 は {' / '.join(BASIS_MODES)} のいずれか(指定値: {basis!r})")

    pid = str(pid)
    warnings: list[str] = []

    rest = {p: c for p, c in curves.items() if p != pid}
    if len(rest) < 2:
        raise ValueError(
            f"段階予測には、対象 {pid} を除いた学習用の完了案件が2件以上必要です"
            f"(現在 {len(rest)} 件)。")
    if pid in curves:
        warnings.append(
            f"{pid} は完了案件として学習データに入っているため、"
            f"段階予測では自分自身を学習から外した {len(rest)} 件で学習し直している"
            "(自分の実績で学習したカーブで自分を予測すると評価にならない)。")

    model = learn(rest, groups, align=align, n_bin=n_bin,
                  backbone_spec=backbone_spec, backbone_coverage=backbone_coverage,
                  hours_per_mm=hours_per_mm, recon=recon,
                  warp_strength=warp_strength, max_stretch=max_stretch)

    # 月軸は「予測に使う軸」に揃える。実績側もこの軸に載せる。
    base = forecast(model, ds, pid, group_totals=group_totals,
                    exclude_groups=exclude_groups)
    months, fgroups = base.months, base.groups

    actual = (curves[pid].monthly.reindex(index=months, columns=fgroups).fillna(0.0)
              if pid in curves else
              _actual_table(agg, pid, months, fgroups))
    if float(actual.to_numpy().sum()) <= 0:
        raise ValueError(
            f"{pid} には実績が1行もないため、段階予測できません。"
            "実績のある案件(完了案件・進行中案件)を --target に指定してください。")

    # 実績が存在する最後の月。進行中案件では期間の末尾に実績が無い。
    # そこから先は「確定させる実績」も「答え合わせに使う実績」も無いので、
    # 区切りにも評価にも使わない。
    act_m_all = actual.sum(axis=1).to_numpy()
    nz = np.nonzero(act_m_all > 0)[0]
    eval_last = int(nz[-1])
    partial = eval_last < len(months) - 1
    if partial:
        warnings.append(
            f"{pid} は実績が {months[eval_last]} までしかない(契約期間は {months[-1]} まで)。"
            f"それ以降は答え合わせに使える実績が無いため、区切りにも誤差の計算にも使っていない。"
            "進行中案件では、この先の予測が当たっているかはまだ判定できない。")

    points = [pt for pt in _stage_points(ds, pid, months) if pt[2] <= eval_last]
    if not points:
        warnings.append(
            f"{pid} には区切りに使えるマイルストーンがない"
            "(未記入、期間外、または最終月に集中している)ため、着手前の予測だけを出している。")

    stages: list[Stage] = [
        _build_stage(model, ds, pid, idx=0, label="段階0 着手前", milestone="",
                     cut_idx=-1, cut_month=None, known=[], months=months,
                     groups=fgroups, actual=actual, basis=basis,
                     group_totals=group_totals, exclude_groups=exclude_groups)
    ]
    for i, (nm, date, ci) in enumerate(points, start=1):
        known = _known_names(ds, pid, date)
        stages.append(
            _build_stage(model, ds, pid, idx=i,
                         label=f"段階{i} {nm}後({months[ci]}まで確定)",
                         milestone=nm, cut_idx=ci, cut_month=months[ci],
                         known=known, months=months, groups=fgroups,
                         actual=actual, basis=basis, group_totals=group_totals,
                         exclude_groups=exclude_groups))

    act_m = actual.sum(axis=1)
    # 実績が無い月を 0 として描くと、グラフ上は「実績が急に落ちた」ように見える。
    # 実績が無いだけなので、線を途切れさせて空欄にする。
    act_plot = act_m.round(1)
    if partial:
        act_plot = act_plot.astype(float).copy()
        act_plot.iloc[eval_last + 1:] = np.nan
    series = pd.DataFrame({"実績": act_plot}, index=pd.Index(months, name="月"))
    for st in stages:
        series[st.label] = st.monthly.round(1)

    rows = []
    for st in stages:
        rows.append({
            "段階": st.idx,
            "区切り": st.milestone or "(なし)",
            "確定した月": st.cut_month or "(なし)",
            "確定月数": st.n_fixed,
            "見込んだ総工数(時間)": round(st.total_hours, 1),
            **_metrics(st.monthly, act_m, st.n_fixed, eval_last),
        })
    metrics = pd.DataFrame(rows)

    return PhasedResult(pid=pid, name=base.name, months=months, groups=fgroups,
                        actual=actual, eval_last=eval_last, stages=stages, basis=basis,
                        series=series, metrics=metrics, warnings=warnings)


def _actual_table(agg: pd.DataFrame, pid: str, months: list[str],
                  groups: list[str]) -> pd.DataFrame:
    sub = agg[agg["案件ID"].astype(str) == str(pid)]
    if sub.empty:
        return pd.DataFrame(0.0, index=pd.Index(months, name="月"), columns=groups)
    piv = (sub.pivot_table(index="月", columns="行程グループ",
                           values="時間", aggfunc="sum", observed=True)
              .reindex(index=months, columns=groups).fillna(0.0))
    piv.index = pd.Index(months, name="月")
    return piv


def _build_stage(model, ds, pid: str, *, idx: int, label: str, milestone: str,
                 cut_idx: int, cut_month: str | None, known: list[str],
                 months: list[str], groups: list[str], actual: pd.DataFrame,
                 basis: str, group_totals, exclude_groups) -> Stage:
    """1段階分を組み立てる。

    予測そのものは通常の forecast() をそのまま使う。
    違うのは「その時点で分かっているマイルストーンだけを渡す」ことと、
    出てきた月次カーブを 確定分 / 残り に切って、残りだけを使うこと。
    """
    n_fixed = cut_idx + 1
    fc = forecast(model, ds, pid, use_given_milestones=True,
                  known_milestones=set(known) if idx > 0 else set(),
                  group_totals=group_totals, months_override=months,
                  exclude_groups=exclude_groups)
    notes = list(fc.notes)

    fixed = actual.iloc[:n_fixed]
    table = pd.DataFrame(0.0, index=pd.Index(months, name="月"), columns=groups)
    if n_fixed:
        table.iloc[:n_fixed] = fixed.to_numpy()

    pred_only = pd.DataFrame(0.0, index=pd.Index(months, name="月"), columns=groups)
    scaled_groups, kept_groups = [], []

    for g in groups:
        curve = fc.table[g].to_numpy(dtype=float)
        s = curve.sum()
        if s <= 0:
            continue
        f = curve / s                       # 学習カーブの月次配分(合計1)
        done = float(fixed[g].sum()) if n_fixed else 0.0
        p_done = float(f[:n_fixed].sum())   # その時点までに消化するはずの比率

        if basis == BASIS_SCALED and n_fixed and done > 0 and p_done > 1e-6:
            total_g = done / p_done
            scaled_groups.append(g)
        else:
            total_g = s                     # 契約人月・見積もりから決まった総量
            if basis == BASIS_SCALED and n_fixed:
                kept_groups.append(g)
        remain = max(total_g - done, 0.0)

        rest_w = f[n_fixed:]
        if rest_w.sum() > 0:
            pred_only.iloc[n_fixed:, pred_only.columns.get_loc(g)] = \
                remain * rest_w / rest_w.sum()
        elif len(rest_w):
            # 学習カーブがこの区間に何も置いていない(すべて確定分に入っている)。
            # 残工数を消せば総量が合わなくなるので、残り月へ均等に置く。
            pred_only.iloc[n_fixed:, pred_only.columns.get_loc(g)] = remain / len(rest_w)

    table = table + pred_only
    total = float(table.to_numpy().sum())

    if idx == 0:
        notes.insert(0, "実績を一切使わない全期間予測。マイルストーンの日付も使っていない"
                        "(着手前に通過済みのマイルストーンは無いため)。"
                        "「予測」シートは記入済みの日付をすべて使うので、その分だけ条件が違う。")
    else:
        notes.insert(0, f"{months[0]}〜{cut_month} の {n_fixed} ヶ月を実績で確定し、"
                        f"残り {len(months) - n_fixed} ヶ月を予測している。")
        notes.insert(1, "使うマイルストーンは通過済みのものだけ: "
                        + (", ".join(known) if known else "(なし)")
                        + "。先のマイルストーンの日付はその時点では分かっていないため渡さず、"
                        "学習した平均位置を使う(渡すと未来を覗いた予測になり評価にならない)。")
    if scaled_groups:
        notes.append(f"総量を実績から引き直した行程グループ: {', '.join(scaled_groups)}")
    if kept_groups:
        notes.append(
            f"確定分の実績が 0 のため総量を引き直せず、契約・見積もりの値のままにした"
            f"行程グループ: {', '.join(kept_groups)}")

    return Stage(idx=idx, label=label, milestone=milestone, cut_month=cut_month,
                 n_fixed=n_fixed, known_milestones=known, table=table,
                 pred_only=pred_only, total_hours=total, notes=notes)
