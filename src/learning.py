"""
学習ロジック(設計書 5章)

Step 1. 実績の集約           aggregate_actuals()
Step 2. 累積消化カーブの作成  build_project_curves()
Step 3. マイルストーン位置合わせ  choose_backbone() + Warp
Step 4. 行程グループ別の工数比率  learn()
Step 5. マイルストーン位置分布    learn()

第1版のスコープは 実装順序 1〜2(素朴版 + 位置合わせ)。
Step 6(種別・タグの重み付け)は構造だけ用意し、重みは全案件 1.0 固定とする。
重みを差し替える口は Model.contributors と learn(weights=...) に開けてある。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .elasticity import deflate
from .timeaxis import (RECON_BOX, Warp, date_to_t, month_edges, month_list,
                       monthly_to_canonical, t_to_date)


# ===========================================================================
# Step 1. 実績の集約
# ===========================================================================
def aggregate_actuals(ds, group_col: str | None = None) -> pd.DataFrame:
    """実績を 案件ID × 行程グループ × 月 に集約する(設計書 5 Step1)。

    数十万行の実績もここで数千行に落ちる。以降の計算はすべて小さい表の上で行う。

    分析対象は phase_map に載っている行程だけとする。
    実績CSVには別部門の作業・間接業務・廃止された行程など、
    このシステムの対象外の行程が混ざっている前提なので、集約する前に落とす。

    残したまま「(未分類)」として1グループに集めると、
    行程グループ比率にも学習カーブにも対象外の工数が混ざり込み、
    予測が対象外業務の形に引っ張られる。
    ただし黙って捨てると総量が合わない原因を追えなくなるため、量は必ず報告する。
    """
    group_col = group_col or ds.group_col
    if group_col not in ds.phase_map.columns:
        raise ValueError(
            f"集約軸 '{group_col}' が phase_map シートにありません。"
            f"利用可能な列: {list(ds.phase_map.columns)}")

    mapping = dict(zip(ds.phase_map["実績行程名"], ds.phase_map[group_col]))
    df = ds.actuals[["案件ID", "行程", "月", "時間"]].copy()
    df["行程"] = df["行程"].astype(str)
    df[group_col] = df["行程"].map(mapping)

    known = df[group_col].notna()
    if not known.all():
        drop = df.loc[~known]
        names = sorted(drop["行程"].unique())
        hpm = float(ds.settings["人月換算係数"])
        shown = ", ".join(names[:8]) + (" ほか" if len(names) > 8 else "")
        msg = (f"phase_map に無い行程 {len(names)} 種 / {len(drop):,} 行 "
               f"/ 計 {drop['時間'].sum() / hpm:,.0f} 人月 を分析対象から除外しました: {shown}")
        print(f"[警告] {msg}")
        ds.warnings.append(msg)
        ds.warnings.append(
            "上記を除外したため、各案件の実績人月は実績CSVの総量より小さくなり、"
            "契約人月との乖離率もその分マイナス側に振れる。"
            "対象に含めたい行程があれば phase_map シートに行を追加すること。")
        df = df.loc[known]

    if df.empty:
        raise ValueError(
            "phase_map に載っている行程の実績が1行もありません。"
            "phase_map シートの 実績行程名 が実績CSVの 行程 と一致しているか確認してください。")

    agg = (df.groupby(["案件ID", group_col, "月"], observed=True)["時間"]
             .sum().reset_index().rename(columns={group_col: "行程グループ"}))
    return agg


# ===========================================================================
# Step 2. 案件ごとのカーブ
# ===========================================================================
@dataclass
class ProjectCurve:
    pid: str
    name: str
    ptype: str
    contract_mm: float
    months: list[str]
    edges: np.ndarray
    monthly: pd.DataFrame      # index=月, columns=行程グループ, 値=時間
    ms_t: dict[str, float]     # マイルストーン名 -> 経過期間比
    warp: Warp = field(default_factory=Warp.identity)

    @property
    def total_hours(self) -> float:
        return float(self.monthly.to_numpy().sum())

    @property
    def group_ratio(self) -> pd.Series:
        s = self.monthly.sum(axis=0)
        return s / s.sum()

    def cumulative(self) -> np.ndarray:
        """累積消化率 F(t)。月末時点の値(長さ = 月数)。"""
        tot = self.monthly.sum(axis=1).to_numpy()
        return np.cumsum(tot) / tot.sum()


def build_project_curves(ds, agg: pd.DataFrame, groups: list[str]) -> dict[str, ProjectCurve]:
    """案件ごとに 月 × 行程グループ の実績表とマイルストーン位置を組み立てる。

    必要なのは月次実績と開始・終了月だけ。
    マイルストーン情報がゼロの案件も必ずここに現れ、学習に貢献できる(設計書 5 Step2)。
    """
    known = ds.known_ids
    hpm = float(ds.settings["人月換算係数"])
    curves: dict[str, ProjectCurve] = {}
    skipped: dict[str, list[str]] = {"未登録": [], "期間不正": [], "実績ゼロ": []}
    imputed: list[str] = []
    out_of_range: list[dict] = []

    for pid, sub in agg.groupby("案件ID", observed=True):
        pid = str(pid)

        # 実績CSVには projects に無い案件が混ざりうる。止めずに除外する。
        if pid not in known:
            skipped["未登録"].append(pid)
            continue

        prow = ds.project(pid)

        # 学習に使うのは完了案件だけ。
        #
        # 進行中・予測対象の案件は実績が途中までしか無い。カーブは合計1に
        # 正規化してから平均するため、途中までの実績をそのまま混ぜると
        # 「前半にすべて消化する案件」として学習され、平均カーブが前に倒れる。
        # 予測対象については、自分自身を学習データにする情報漏れでもある。
        #
        # 実測(完了30件+進行中5件の規模)では、混ぜると月次誤差WAPEが 0.7% 悪化し、
        # 完了案件が少ないほど影響が大きい(完了5件のとき 9.1% 悪化)。
        # 途中までの実績を正しく扱う仕組み自体は作れるが、完了案件が十分あれば
        # 得られるものはほぼ無い(同条件で +0.1%)ため、第1版では単純に外す。
        status = str(prow.get("ステータス", "完了")).strip() or "完了"
        if status != "完了":
            skipped.setdefault(f"ステータス={status}", []).append(pid)
            continue
        try:
            months = month_list(prow["開始"], prow["終了"])
            if not months:
                raise ValueError("期間が空")
        except Exception as e:
            skipped["期間不正"].append(f"{pid}({e})")
            continue

        piv = (sub.pivot_table(index="月", columns="行程グループ",
                               values="時間", aggfunc="sum", observed=True)
                  .reindex(index=months, columns=groups).fillna(0.0))

        # 契約期間の外に記録された実績は集計しない。
        #
        # かつては端月(開始月・終了月)に加算していたが、これは実データを歪める。
        # 遅延・検収対応・保守の続きで実績が終了月をはみ出すのは普通に起きるため、
        # そのはみ出し分がすべて最終月1つに積み上がり、
        # 元データには存在しない工数の山を最後に作ってしまう。
        # 学習カーブの末尾が跳ね、予測にもその山が転写される。
        #
        # reindex(index=months) の時点で期間外の行は既に落ちている。
        # ここでやるのは「どれだけ落としたか」を残すことだけ。
        outside = sub[~sub["月"].astype(str).isin(months)]
        if not outside.empty:
            mons = sorted(outside["月"].astype(str).unique())
            before = int((outside["月"].astype(str) < months[0]).sum())
            after = len(outside) - before
            out_hours = float(outside["時間"].sum())
            share = out_hours / (out_hours + float(piv.to_numpy().sum()) or 1.0)
            out_of_range.append({
                "案件ID": pid, "行数": len(outside), "人月": out_hours / hpm,
                "割合": share, "前": before, "後": after,
                "範囲": f"{mons[0]}〜{mons[-1]}",
            })

        # 総工数0の案件を残すと group_ratio が 0/0 になり、平均全体が NaN で汚染される
        if piv.to_numpy().sum() <= 0:
            skipped["実績ゼロ"].append(pid)
            continue

        # 過去案件の契約人月が未記入でも学習は続けたい(形の学習には使わないため)。
        # 実績合計を契約人月とみなして補完する。予測対象案件では補完せず、forecast() で止める。
        contract = _safe_float(prow["契約人月"])
        actual_mm = piv.to_numpy().sum() / hpm
        if not (contract > 0):
            contract = actual_mm
            imputed.append(pid)

        ms = ds.milestones_of(pid)
        ms_t = {r["マイルストーン名"]: date_to_t(months, r["日付"]) for _, r in ms.iterrows()}

        curves[pid] = ProjectCurve(
            pid=pid, name=str(prow["名称"]), ptype=str(prow["種別"]),
            contract_mm=contract, months=months,
            edges=month_edges(months), monthly=piv, ms_t=ms_t)

    for reason, ids in skipped.items():
        if not ids:
            continue
        bare = [i.split("(")[0] for i in ids]
        mm = agg[agg["案件ID"].astype(str).isin(bare)]["時間"].sum() / hpm
        msg = (f"学習から除外({reason}): {len(ids)} 件 / 計 {mm:,.0f} 人月 — "
               + ", ".join(ids[:8]) + (" ほか" if len(ids) > 8 else ""))
        print(f"[警告] {msg}")
        ds.warnings.append(msg)

    if out_of_range:
        oor = sorted(out_of_range, key=lambda d: -d["人月"])
        total = sum(d["人月"] for d in oor)
        head = (f"契約期間外に記録された実績を除外しました: {len(oor)} 案件 / "
                f"計 {total:,.0f} 人月")
        print(f"[警告] {head}")
        ds.warnings.append(head)
        for d in oor[:10]:
            line = (f"  {d['案件ID']}: {d['行数']:,} 行 / {d['人月']:,.0f} 人月 "
                    f"(その案件の実績の {d['割合']:.0%}) 期間外の月 {d['範囲']} "
                    f"[開始前 {d['前']} 行 / 終了後 {d['後']} 行]")
            print(f"[警告] {line}")
            ds.warnings.append(line)
        if len(oor) > 10:
            ds.warnings.append(f"  ほか {len(oor) - 10} 案件")
        worst = oor[0]
        if worst["割合"] >= 0.10:
            ds.warnings.append(
                f"{worst['案件ID']} は実績の {worst['割合']:.0%} が契約期間外にある。"
                "projects シートの 開始・終了 が実態とずれていないか確認すること。"
                "期間の指定を直せばその工数も学習に使える。")

    if imputed:
        msg = (f"契約人月が未記入のため実績合計で補完した案件 {len(imputed)} 件: "
               + ", ".join(imputed[:8]) + (" ほか" if len(imputed) > 8 else ""))
        print(f"[警告] {msg}")
        ds.warnings.append(msg)

    if not curves:
        raise ValueError(
            "学習に使える案件が1件もありません。"
            "実績の案件IDが projects シートの案件IDと一致しているか確認してください。")
    return curves


def project_monthly(agg: pd.DataFrame, pid: str, months: list[str],
                    groups: list[str]) -> pd.DataFrame:
    """1案件の 月 × 行程グループ 実績表を作る(実績が無ければ空のDataFrame)。

    予測対象が進行中の場合に「予測 vs 実績」を並べるために使う。
    build_project_curves の中の pivot と同じ変換なので、
    予測シートに出る実績と、学習が見ている実績は必ず一致する。
    """
    sub = agg[agg["案件ID"].astype(str) == str(pid)]
    if sub.empty:
        return pd.DataFrame()
    piv = (sub.pivot_table(index="月", columns="行程グループ",
                           values="時間", aggfunc="sum", observed=True)
              .reindex(index=months, columns=groups).fillna(0.0))
    piv.index = pd.Index(months, name="月")
    return piv


def _safe_float(v, default: float = float("nan")) -> float:
    try:
        f = float(v)
        return default if np.isnan(f) else f
    except (TypeError, ValueError):
        return default


# ===========================================================================
# Step 3. 背骨マイルストーンの選定
# ===========================================================================
def choose_backbone(curves: dict[str, ProjectCurve], spec: str = "自動",
                    coverage: float = 0.6) -> list[str]:
    """位置合わせに使うマイルストーンを選ぶ。

    設計書 5 Step3:「案件間で共通するマイルストーンだけを使って揃える」

    ただし「全案件に100%存在するもの」を条件にすると、1案件で記録が漏れた
    だけで背骨が消え、他の9案件で稼げたはずの精度まで失う。
    手入力データでは「存在しなかった」と「記録されていない」の区別がつかない
    (設計書 11)ため、この全か無かは実運用で頻発する。

    そこでカバー率の閾値で選ぶ。持っていない案件はその点を単に使わないだけで、
    残りのアンカーで位置合わせされる。マイルストーンが1つも無い案件は
    恒等写像になり、素朴版と同じ扱いで平均に参加する。
    情報が増えるほど滑らかに精度が上がる(設計書 3-2)。
    """
    if not curves:
        return []
    n = len(curves)
    count: dict[str, int] = {}
    for c in curves.values():
        for name in c.ms_t:
            count[name] = count.get(name, 0) + 1

    if spec and str(spec) != "自動":
        want = [s.strip() for s in str(spec).split(";") if s.strip()]
        names = [nm for nm in want if count.get(nm, 0) >= 2]
    else:
        names = [nm for nm, k in count.items() if k >= coverage * n and k >= 2]

    # 平均位置の順に並べる(持っている案件だけで平均する)
    return sorted(names, key=lambda nm: float(
        np.mean([c.ms_t[nm] for c in curves.values() if nm in c.ms_t])))


# ===========================================================================
# 学習結果
# ===========================================================================
@dataclass
class Model:
    groups: list[str]
    n_bin: int
    shape: dict[str, np.ndarray]      # 行程グループ -> 正準軸上の形状(合計1)
    group_ratio: pd.Series            # 行程グループ -> 総工数に占める比率
    ms_stats: pd.DataFrame            # マイルストーン名 / 平均位置 / 標準偏差 / 件数
    backbone: list[str]
    canonical_anchor: dict[str, float]
    contributors: pd.DataFrame        # 案件ID / 名称 / 種別 / 重み / 実績人月
    align: bool
    project_shapes: dict[str, np.ndarray] = field(default_factory=dict)  # カーブシート用
    hours_per_mm: float = 160.0
    recon: str = RECON_BOX     # 月次から月内分布を復元する方式
    warp_strength: float = 1.0     # 位置合わせの強さ(0=なし / 1=実位置ちょうど)
    max_stretch: float | None = None   # 区間ごとの伸縮率の上限。跳ねの高さの上限になる
    interval_elasticity: float = 0.0
    # 区間の工数配分を区間長へ歩み寄らせる度合い(0=案A・既定 / 1=時間比例)。
    # ここが 0 より大きいとき、この形状カーブは各案件から区間長の影響を
    # 抜いたうえで平均されている(src/elasticity.py)。予測側で入れ直す。
    group_sample_n: dict[str, int] = field(default_factory=dict)
    # 行程グループ -> その形状カーブの学習に参加した案件数。
    # そのグループの業務が一切ない案件は形状の平均に参加しないため、
    # グループごとに実際のサンプル数が違う。カーブの信頼度はここでしか分からない。

    @property
    def total_shape(self) -> np.ndarray:
        """全行程グループ合計の形状(合計1)。"""
        s = sum(self.shape[g] * self.group_ratio.get(g, 0.0) for g in self.groups)
        tot = s.sum()
        return s / tot if tot else s

    def milestone_t(self, name: str) -> float | None:
        hit = self.ms_stats[self.ms_stats["マイルストーン名"] == name]
        return None if hit.empty else float(hit.iloc[0]["平均位置"])

    def low_sample_milestones(self, min_n: int) -> list[str]:
        return self.ms_stats.loc[self.ms_stats["件数"] < min_n, "マイルストーン名"].tolist()

    def low_sample_groups(self, min_n: int) -> list[str]:
        """形状カーブが少数の案件からしか学習できていない行程グループ。

        比率が 0 のグループ(どの案件にも実績が無い)は予測に一切配分されず、
        誤った数字が出るわけではないので対象外とする。
        """
        return [g for g in self.groups
                if 0 < self.group_sample_n.get(g, 0) < min_n
                and self.group_ratio.get(g, 0.0) > 0]


# ===========================================================================
# Step 4・5・学習本体
# ===========================================================================
def learn(curves: dict[str, ProjectCurve], groups: list[str], *,
          align: bool = True, n_bin: int = 100, backbone_spec: str = "自動",
          backbone_coverage: float = 0.6,
          weights: dict[str, float] | None = None,
          hours_per_mm: float = 160.0,
          recon: str = RECON_BOX,
          warp_strength: float = 1.0,
          max_stretch: float | None = None,
          interval_elasticity: float = 0.0) -> Model:
    """案件カーブ群から予測モデルを学習する。

    align=False なら素朴版(経過期間比のまま単純平均)、
    align=True ならマイルストーンで位置を揃えてから平均する。

    warp_strength / max_stretch は位置合わせの効かせ具合。
    学習と予測で同じ値を使わないと、貼り付ける座標系が学習時とずれるため、
    ここで受け取った値を Model に載せて forecast() へ引き渡す。

    interval_elasticity は「学習データより狭く潰れた区間への配分を減らす」度合い
    (設計書 3-1-2)。0 が既定で、そのときこの関数の挙動は従来と1ビットも変わらない。
    0 より大きいときは、各案件の形状から潰れの影響を抜いてから平均する
    (src/elasticity.py の deflate)。予測側の inflate と対になっていて、
    どちらか片方だけを効かせると補正が二重にかかる。
    """
    if not curves:
        raise ValueError("学習に使える案件がありません。")

    weights = weights or {pid: 1.0 for pid in curves}

    backbone = choose_backbone(curves, backbone_spec, backbone_coverage) if align else []
    # 正準位置 = そのマイルストーンを持つ案件だけの平均位置。
    # ここに各案件の実位置を引き寄せる。
    canonical_anchor = {
        nm: float(np.mean([c.ms_t[nm] for c in curves.values() if nm in c.ms_t]))
        for nm in backbone}

    # 各案件のワープを確定させる
    for c in curves.values():
        if backbone:
            pairs = [(nm, canonical_anchor[nm], c.ms_t[nm]) for nm in backbone if nm in c.ms_t]
            c.warp = Warp.build(pairs, strength=warp_strength, max_stretch=max_stretch)
        else:
            c.warp = Warp.identity()

    # --- Step 4: 行程グループ別の工数比率 ---
    ratio_rows, wlist = [], []
    for pid, c in curves.items():
        ratio_rows.append(c.group_ratio.reindex(groups).fillna(0.0))
        wlist.append(weights.get(pid, 1.0))
    W = np.array(wlist, dtype=float)
    R = np.vstack([r.to_numpy() for r in ratio_rows])
    group_ratio = pd.Series((R * W[:, None]).sum(axis=0) / W.sum(), index=groups)
    group_ratio = group_ratio / group_ratio.sum()

    # --- 形状の平均(位置合わせ済みの正準軸上で) ---
    shape: dict[str, np.ndarray] = {}
    group_sample_n: dict[str, int] = {}
    for g in groups:
        acc = np.zeros(n_bin)
        wsum = 0.0
        n_used = 0
        for pid, c in curves.items():
            monthly = c.monthly[g].to_numpy()
            if monthly.sum() <= 0:
                continue  # その案件に存在しないグループは平均に参加しない
            can = monthly_to_canonical(monthly, c.edges, c.warp, n_bin, recon=recon)
            can = can / can.sum()          # 案件規模で重み付けしないよう正規化
            # 区間弾力性を使うときは、その案件の区間長による偏りをここで抜く。
            # 抜かずに平均すると、予測側で区間長を効かせたときに二重にかかる。
            can = deflate(can, c.warp, interval_elasticity)
            w = weights.get(pid, 1.0)
            acc += can * w
            wsum += w
            n_used += 1
        # 全案件に無いグループは一様分布で埋める。比率も0になるため予測には
        # 配分されないが、見積もり分類で総量を指定された場合の受け皿になる。
        shape[g] = acc / wsum if wsum > 0 else np.full(n_bin, 1.0 / n_bin)
        group_sample_n[g] = n_used

    # --- カーブシート用: 案件ごとの総工数形状(正準軸) ---
    project_shapes = {}
    for pid, c in curves.items():
        tot = c.monthly.sum(axis=1).to_numpy()
        can = monthly_to_canonical(tot, c.edges, c.warp, n_bin, recon=recon)
        # 平均カーブと並べて見せるシートなので、平均側と同じ変換を通しておく。
        project_shapes[pid] = deflate(can / can.sum(), c.warp, interval_elasticity)

    # --- Step 5: マイルストーン位置分布 ---
    rows = []
    all_names = sorted({nm for c in curves.values() for nm in c.ms_t},
                       key=lambda nm: float(np.mean([c.ms_t[nm] for c in curves.values()
                                                     if nm in c.ms_t])))
    for nm in all_names:
        vals = np.array([c.ms_t[nm] for c in curves.values() if nm in c.ms_t])
        rows.append({
            "マイルストーン名": nm,
            "平均位置": float(vals.mean()),
            "標準偏差": float(vals.std(ddof=1)) if len(vals) > 1 else np.nan,
            "件数": int(len(vals)),
            "背骨": "○" if nm in backbone else "",
        })
    ms_stats = pd.DataFrame(rows, columns=["マイルストーン名", "平均位置", "標準偏差", "件数", "背骨"])

    # 契約人月と実績の合計は厳密には一致しない(未消化・超過・記録漏れ)。
    # 学習は各案件のカーブを正規化してから平均するので、この乖離は結果に影響しない。
    # ただし乖離が大きい案件は実績側の欠落を疑うべきなので、比率を並べて見せる。
    rows = []
    for pid, c in curves.items():
        actual_mm = c.total_hours / hours_per_mm
        gap = (actual_mm / c.contract_mm - 1.0) if c.contract_mm and c.contract_mm > 0 else np.nan
        rows.append({
            "案件ID": pid, "名称": c.name, "種別": c.ptype,
            "重み": weights.get(pid, 1.0),
            "契約人月": c.contract_mm,
            "実績人月": round(actual_mm, 1),
            "乖離率(%)": round(gap * 100, 1) if not np.isnan(gap) else None,
            "期間(月)": len(c.months),
            "マイルストーン数": len(c.ms_t),
        })
    contributors = pd.DataFrame(rows).sort_values("案件ID").reset_index(drop=True)

    return Model(groups=groups, n_bin=n_bin, shape=shape, group_ratio=group_ratio,
                 ms_stats=ms_stats, backbone=backbone, canonical_anchor=canonical_anchor,
                 contributors=contributors, align=align, project_shapes=project_shapes,
                 hours_per_mm=hours_per_mm, group_sample_n=group_sample_n,
                 recon=recon, warp_strength=warp_strength, max_stretch=max_stretch,
                 interval_elasticity=interval_elasticity)


def group_names(agg: pd.DataFrame, phase_map: pd.DataFrame, group_col: str) -> list[str]:
    """行程グループの一覧を phase_map の並び順で返す。

    グループの定義は master.xlsx が唯一の正。実績側に現れた値を拾って
    列を増やすことはしない。実績にしか無い行程は aggregate_actuals が
    既に除外しているので、ここに来る時点で agg は phase_map の範囲に収まっている。
    出力Excelの列順もこの並び(= 利用者がシートで並べた順)になる。
    """
    order, seen = [], set()
    for v in phase_map[group_col]:
        if v not in seen:
            seen.add(v)
            order.append(v)
    return order
