"""
読み込み層(設計書 4-2)

「DataFrame を複数返す関数」という形に閉じ込める。
将来データベース化する際は、このモジュールだけ差し替えれば移行できる。

計算部・出力部はここで返す Dataset だけに依存し、
Excel/CSV というファイル形式の存在を知らない。
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import os
import re
from dataclasses import dataclass, field

import pandas as pd

# 実績CSVのパース仕様を変えたらここを上げる。
# キャッシュキーに含めることで、ロジック変更後に古い(バグ入り)パース結果の
# parquetキャッシュを誤って使い続けるのを防ぐ。
CACHE_SCHEMA_VERSION = "v3"

# 全角の数字・小数点・桁区切りを半角に変換してから数値パースする。
_ZEN_DIGITS = str.maketrans("０１２３４５６７８９．，", "0123456789.,")

# projects シートの固定列。これ以外の列はすべてマイルストーン列として読む。
# 「プロジェクト情報の右にマイルストーンを足していく」入力形式の土台になる。
PROJECT_BASE_COLS = ("案件ID", "名称", "種別", "契約人月", "開始", "終了", "タグ", "ステータス")

# 同じマイルストーンが複数回あるときは、1つのセルにカンマ区切りで並べて書く。
#   α版 の列に  2020-04-18, 2020-08-22, 2021-01-12
# 列を増やす方式は、記入する列を間違える・列を足し忘れるといった事故が起きるため使わない。
_DATE_SEPARATORS = re.compile(r"[,、;；\n\r]+")

# 月までの表記。開始・終了と同じ粒度で書けるようにする。
#   2020-08 / 2020/8 / 2020年8月
# 月表記はその月の中央として扱う。月初として扱うと、実際の日付が月内に散っている分
# だけ全部のマイルストーンが半月ぶん前へずれ、系統的な誤差になる(README の実測参照)。
_MONTH_ONLY = re.compile(r"^\s*(\d{4})\s*[-/年.]\s*(\d{1,2})\s*月?\s*$")

# 見出しが重複したときの後始末。
# 利用者が同じ見出しを2つ作ってしまうと pandas が α版.1 のように直すので、
# その分は同じマイルストーンとして読み直す(セルを移し替えさせない)。
_PANDAS_DEDUP_SUFFIX = re.compile(r"\.\d+$")
_REPEAT_SUFFIX = re.compile(
    r"(?:\s*[#＃]\s*\d+|\s*[（(]\s*\d+\s*回目\s*[)）])\s*$")

# 同じマイルストーンが複数回あったとき、初回に付ける接尾辞。
# 「レビューに入った日」と「通過した日」は別の意味を持つので、別のマイルストーンとして扱う。
FIRST_ATTEMPT_SUFFIX = "(初回)"

DEFAULT_SETTINGS = {
    "人月換算係数": 160.0,
    "集約軸": "行程グループ",
    "位置合わせ": "ON",
    "背骨マイルストーン": "自動",
    "背骨最小カバー率": 0.6,
    "マイルストーン精度": "月",
    "位置合わせ強度": 1.0,
    "伸縮率上限": 0.0,
    "マイルストーン最小件数": 3,
    "行程グループ最小件数": 3,
    "カーブ解像度": 100,
    "カーブ復元": "月内均等",
    "立ち上がり上限": "自動",
    "段階予測のマイルストーン": "すべて",
    "k": 3.0,
    "タグ重み係数": 0.5,
}


@dataclass
class Dataset:
    """システム内で扱うデータの唯一の入り口。"""

    projects: pd.DataFrame     # 案件ID / 名称 / 種別 / 契約人月 / 開始 / 終了 / タグ / ステータス
    milestones: pd.DataFrame   # 案件ID / マイルストーン名 / 日付
    phase_map: pd.DataFrame    # 実績行程名 / 標準行程名 / 行程グループ / 大分類 (+任意の集約軸列)
    settings: dict             # パラメータ名 -> 値
    actuals: pd.DataFrame      # 案件ID / 行程 / メンバー / 所属 / チーム / 月 / 時間
    estimates: pd.DataFrame = field(default_factory=pd.DataFrame)
    # 案件ID / 行程グループ / 見積人月。これから着手する案件は実績が1行も無く、
    # 分かっているのは要素別の見積もりだけになる。その入力口(設計書 6 Step1)。
    ms_attempts: pd.DataFrame = field(default_factory=pd.DataFrame)
    # 案件ID / マイルストーン名 / 回数 / 初回 / 最終。
    # 同じマイルストーンを複数回通過した(= 一度で通らずやり直した)案件の記録。
    # 位置合わせのアンカーには最終回だけを使うが、回数そのものは
    # 「その案件が難航した」という情報なので捨てずに残す。
    source: dict = field(default_factory=dict)
    warnings: list = field(default_factory=list)
    # 読み込み〜学習の途中で気づいた注意点をここに溜め、出力Excelの「条件」シートに載せる。
    # 標準出力に流すだけだと、Excelを受け取った人には何も伝わらない。

    # ---- 便利アクセサ -------------------------------------------------
    @property
    def hours_per_mm(self) -> float:
        return float(self.settings["人月換算係数"])

    @property
    def known_ids(self) -> set:
        """projects シートに登録されている案件IDの集合。"""
        return set(self.projects["案件ID"].astype(str))

    @property
    def group_col(self) -> str:
        return str(self.settings["集約軸"])

    def learning_projects(self) -> pd.DataFrame:
        """実績があり学習に使える案件。"""
        have = set(self.actuals["案件ID"].unique())
        return self.projects[self.projects["案件ID"].isin(have)].copy()

    def target_projects(self) -> pd.DataFrame:
        """ステータス=予測対象 の案件。"""
        return self.projects[self.projects["ステータス"] == "予測対象"].copy()

    def project(self, pid: str) -> pd.Series:
        hit = self.projects[self.projects["案件ID"] == pid]
        if hit.empty:
            raise KeyError(f"案件が projects シートに見つかりません: {pid}")
        return hit.iloc[0]

    def milestones_of(self, pid: str) -> pd.DataFrame:
        return self.milestones[self.milestones["案件ID"] == pid].copy()

    def estimates_of(self, pid: str) -> dict[str, float]:
        """案件の 行程グループ -> 見積時間。未入力なら空の辞書。

        見積もりが1行でもあれば、そこに現れない行程グループは
        「この案件では行わない業務」として予測から外す(設計書 6 Step1)。
        """
        if self.estimates.empty:
            return {}
        sub = self.estimates[self.estimates["案件ID"].astype(str) == str(pid)]
        if sub.empty:
            return {}
        hpm = self.hours_per_mm
        out: dict[str, float] = {}
        for _, r in sub.iterrows():
            g = str(r["行程グループ"]).strip()
            try:
                mm = float(r["見積人月"])
            except (TypeError, ValueError):
                continue
            if g and mm > 0:
                out[g] = out.get(g, 0.0) + mm * hpm
        return out

    def tags_of(self, pid: str) -> list[str]:
        raw = self.project(pid).get("タグ", "")
        if not isinstance(raw, str) or not raw.strip():
            return []
        return [t.strip() for t in raw.split(";") if t.strip()]


# ---------------------------------------------------------------------------
def _read_table(path: str, sheet: str) -> pd.DataFrame:
    """シートを読み、最初の空行以降を切り捨てる。

    マスタの各シート末尾には凡例(使い方の注記)を置いてある。
    データ本体と凡例の間には必ず空行が1行入るので、そこで打ち切れば
    凡例がデータ行として紛れ込まない。利用者は自由に注記を書き足せる。
    """
    df = pd.read_excel(path, sheet_name=sheet)
    blank = df.isna().all(axis=1).to_numpy()
    if blank.any():
        # ラベルではなく位置で切る。データが0行(見出しの直後が空行)でも成立させるため。
        df = df.iloc[: int(blank.argmax())]
    return df.dropna(how="all").reset_index(drop=True)


def _read_settings(path: str) -> dict:
    try:
        df = _read_table(path, "settings")
    except ValueError:
        return dict(DEFAULT_SETTINGS)

    out = dict(DEFAULT_SETTINGS)
    for _, r in df.iterrows():
        key = str(r["パラメータ"]).strip()
        val = r["値"]
        if key in ("人月換算係数", "k", "タグ重み係数", "背骨最小カバー率"):
            val = float(val)
        elif key in ("マイルストーン最小件数", "行程グループ最小件数", "カーブ解像度"):
            val = int(val)
        else:
            val = str(val).strip()
        out[key] = val
    return out


def month_center(month) -> pd.Timestamp:
    """月をその月の中央の時点として返す。

    月までしか分からないマイルストーンを時間軸上の1点に置くとき、
    月初に置くと実際の日付が月内に散っているぶんだけ半月ぶん前へ寄る。
    全案件で同じ向きにずれるので、これは平均で消えない系統誤差になる。
    月央なら、月内で一様に散っていると仮定したときのずれが期待値ゼロになる。
    """
    p = pd.Period(str(month)[:7], freq="M")
    return p.start_time + (p.end_time - p.start_time) / 2


def _parse_dates_cell(v) -> list[pd.Timestamp]:
    """1セルの値を日付のリストにする。

    通常は1つ。同じマイルストーンを複数回通過した案件では、
    カンマ区切りで並べて書く(「2020-04-18, 2020-08-22」)。
    区切りはカンマ・読点・セミコロン・改行を受け付ける。
    日付そのものが 2020/4/18 のようにスラッシュを含むため、
    スラッシュは区切りにしない。

    開始・終了と同じ「2020-08」という月までの表記も書ける。その場合は月央として扱う。
    """
    if v is None or (not isinstance(v, str) and pd.isna(v)):
        return []
    if isinstance(v, (pd.Timestamp, dt.datetime, dt.date)):
        return [pd.Timestamp(v)]
    parts = [p.strip() for p in _DATE_SEPARATORS.split(str(v)) if p.strip()]
    out = []
    for p in parts:
        m = _MONTH_ONLY.match(p)
        if m:
            # 13月のような値は月表記として成立しないので、読めない値として扱う。
            try:
                out.append(month_center(f"{int(m.group(1)):04d}-{int(m.group(2)):02d}"))
            except Exception:
                out.append(None)
            continue
        d = pd.to_datetime(p, errors="coerce")
        out.append(d if pd.notna(d) else None)
    return out


def _base_milestone_name(col: str, all_cols: list[str]) -> str:
    """マイルストーン列の見出しから、繰り返し回数の接尾辞を落として本来の名前を返す。

    通常はそのまま。「α版#2」「α版(2回目)」のように書かれていれば「α版」に戻す。
    「α版.1」は pandas が重複見出しを機械的に直した形なので、
    接尾辞を落とした名前が同じシートに実在するときだけ落とす。
    「ver1.5」のような、本当に数字で終わる名前を壊さないための条件。
    """
    name = str(col).strip()
    stripped = _REPEAT_SUFFIX.sub("", name).strip()
    if stripped and stripped != name:
        return stripped
    dedup = _PANDAS_DEDUP_SUFFIX.sub("", name).strip()
    if dedup and dedup != name and dedup in {str(c).strip() for c in all_cols}:
        return dedup
    return name


def _milestones_from_projects(projects: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """projects シートの横持ちマイルストーン列を縦持ちに開く。

    案件情報の右にマイルストーンを列として並べ、見出し行をマイルストーン名にする形式。
    別シートに分けるのと比べて、

      - マイルストーン名が見出し行で1回決まるので、表記の揺れが起きない
      - どの案件に何が記入済みかが、シートを見るだけで分かる
      - 案件1行を書けば入力が終わる(シート間を往復しない)

    固定列(PROJECT_BASE_COLS)以外の列をマイルストーン列とみなす。
    ただしメモ欄などを誤って取り込まないよう、
    日付として読める値が1つも無い列は無視する。

    戻り値: (マイルストーン列を除いた projects, 縦持ちマイルストーン, 警告)
    """
    warnings: list[str] = []
    extra = [c for c in projects.columns if str(c).strip() not in PROJECT_BASE_COLS]
    if not extra:
        return projects, pd.DataFrame(columns=["案件ID", "マイルストーン名", "日付"]), warnings

    rows, ms_cols, ignored = [], [], []
    pids = projects["案件ID"].astype(str).str.strip().tolist()
    for col in extra:
        parsed = [_parse_dates_cell(v) for v in projects[col]]
        if not any(any(d is not None for d in lst) for lst in parsed):
            ignored.append(str(col))
            continue
        ms_cols.append(col)
        name = _base_milestone_name(col, list(projects.columns))
        bad = [(pid, projects[col].iloc[i]) for i, (pid, lst) in enumerate(zip(pids, parsed))
               if any(d is None for d in lst)]
        if bad:
            shown = ", ".join(f"{pid}({v!r})" for pid, v in bad[:5])
            warnings.append(
                f"projects シートの「{col}」列に日付として読めない値があり、その値だけ"
                f"無視しました: {shown}" + (" ほか" if len(bad) > 5 else "")
                + "。複数回あるマイルストーンはカンマ区切りで書くこと(例: 2020-04-18, 2020-08-22)。")
        for pid, lst in zip(pids, parsed):
            for d in lst:
                if d is not None:
                    rows.append({"案件ID": pid, "マイルストーン名": name, "日付": d})

    if ignored:
        warnings.append(
            f"projects シートの次の列は日付が1つも入っていないため、"
            f"マイルストーン列として読みませんでした: {', '.join(ignored)}。"
            "マイルストーン列なら日付を入れること。メモ欄ならこのままで問題ありません。")

    keep = [c for c in projects.columns if c not in ms_cols]
    return projects[keep].copy(), pd.DataFrame(
        rows, columns=["案件ID", "マイルストーン名", "日付"]), warnings


def _normalize_milestone_repeats(ms: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """同じマイルストーンが同じ案件に複数回あるケースを整理する。

    マイルストーンは「このまま進めてよいか」を判断する機会でもあるので、
    通過できずにやり直し、同じ名前のマイルストーンが2回・3回と訪れることがある。

    時間軸の位置合わせに使うアンカーは、案件ごとに1つの名前につき1点でなければならない
    (2点あるとワープが一対一でなくなり、逆変換が壊れる)。そこで意味で切り分ける。

        最終回 -> 「α版」        その工程を実際に抜けた日。工程の境目はここ。
        初回   -> 「α版(初回)」  レビューに入った日。ここから最終回までが手戻り期間。
        中間   -> 使わない        3回目以降の中間はアンカーとして意味が定まらない

    初回を別のマイルストーンとして残すのが要点で、
    「α版(初回)」と「α版」の間隔がそのまま手戻り期間の長さになり、
    そこに工数が積み上がる形を学習カーブが自然に持てる。
    背骨に採用するかどうかは、他のマイルストーンと同じくカバー率が決める。

    利用者が自分で「α版(初回)」列を作っている場合はそちらを正とし、上書きしない。
    """
    cols = ["案件ID", "マイルストーン名", "日付"]
    empty_att = pd.DataFrame(columns=["案件ID", "マイルストーン名", "回数", "初回", "最終"])
    if ms.empty:
        return ms.reindex(columns=cols), empty_att, []

    ms = ms.dropna(subset=["日付"]).sort_values(["案件ID", "マイルストーン名", "日付"])
    existing = set(zip(ms["案件ID"], ms["マイルストーン名"]))
    out, att, warnings = [], [], []

    for (pid, name), sub in ms.groupby(["案件ID", "マイルストーン名"], sort=False):
        dates = sub["日付"].tolist()
        if len(dates) == 1:
            out.append({"案件ID": pid, "マイルストーン名": name, "日付": dates[0]})
            continue
        out.append({"案件ID": pid, "マイルストーン名": name, "日付": dates[-1]})
        first_name = f"{name}{FIRST_ATTEMPT_SUFFIX}"
        if (pid, first_name) not in existing:
            out.append({"案件ID": pid, "マイルストーン名": first_name, "日付": dates[0]})
        att.append({"案件ID": pid, "マイルストーン名": name, "回数": len(dates),
                    "初回": dates[0], "最終": dates[-1]})

    attempts = pd.DataFrame(att, columns=["案件ID", "マイルストーン名", "回数", "初回", "最終"])
    if not attempts.empty:
        n_pj = attempts["案件ID"].nunique()
        warnings.append(
            f"同じマイルストーンが複数回記録されている案件が {n_pj} 件あります。"
            f"位置合わせのアンカーには最終回(= 実際に通過した日)を使い、"
            f"初回は「名前{FIRST_ATTEMPT_SUFFIX}」という別のマイルストーンとして扱いました。"
            "その間隔が手戻り期間になります。")
        for _, r in attempts.sort_values("回数", ascending=False).head(10).iterrows():
            span = (r["最終"] - r["初回"]).days
            warnings.append(
                f"  {r['案件ID']} {r['マイルストーン名']}: {int(r['回数'])} 回 "
                f"({r['初回']:%Y-%m-%d} 〜 {r['最終']:%Y-%m-%d} / {span} 日)")
        if len(attempts) > 10:
            warnings.append(f"  ほか {len(attempts) - 10} 件")

    return (pd.DataFrame(out, columns=cols)
            .sort_values(["案件ID", "日付"]).reset_index(drop=True),
            attempts, warnings)


def load_all(master_path: str, actuals_path: str,
             overrides: dict | None = None,
             use_cache: bool = True) -> Dataset:
    """マスタと実績を読み込み、Dataset を返す。

    overrides で settings の値をコマンドラインから上書きできる。
    """
    if not os.path.exists(master_path):
        raise FileNotFoundError(f"マスタが見つかりません: {master_path}")
    if not os.path.exists(actuals_path):
        raise FileNotFoundError(f"実績データが見つかりません: {actuals_path}")

    projects = _read_table(master_path, "projects")
    projects["案件ID"] = projects["案件ID"].astype(str).str.strip()
    for c in ("開始", "終了"):
        projects[c] = projects[c].apply(_to_month_str)
    if "ステータス" not in projects.columns:
        projects["ステータス"] = "完了"
    # 学習に使うかどうかがこの列で決まるようになったため、表記を揃えておく。
    # 空欄は「完了」とみなす(列そのものが無い場合と同じ扱い)。
    # astype(str) は欠損を欠損のまま残す(pandas 3 の str dtype)ので、
    # 文字列 "nan" を置換するのではなく fillna で埋める。
    projects["ステータス"] = (projects["ステータス"].astype(str).str.strip()
                              .fillna("完了").replace({"": "完了"}))
    if "タグ" not in projects.columns:
        projects["タグ"] = ""
    projects["タグ"] = projects["タグ"].fillna("")

    settings = _read_settings(master_path)
    if overrides:
        settings.update({k: v for k, v in overrides.items() if v is not None})

    # マイルストーンは projects シートの右側に列として書く形が本命。
    # 旧形式(別シート)も読めるようにしてあるので、移行の途中でも動く。
    projects, wide_ms, ms_warnings = _milestones_from_projects(projects)

    try:
        sheet_ms = _read_table(master_path, "milestones")
        sheet_ms["案件ID"] = sheet_ms["案件ID"].astype(str).str.strip()
        sheet_ms["マイルストーン名"] = sheet_ms["マイルストーン名"].astype(str).str.strip()
        sheet_ms["日付"] = pd.to_datetime(sheet_ms["日付"], errors="coerce")
        if not wide_ms.empty and not sheet_ms.empty:
            ms_warnings.append(
                "projects シートの横持ちマイルストーンと milestones シートの両方に"
                "記入があります。両方を読み込みましたが、二重管理は表記の揺れの元なので、"
                "どちらかに寄せること(scripts/migrate_master_wide.py で統合できます)。")
    except ValueError:
        # マイルストーンが1件も無くても動く(設計書 3-2)
        sheet_ms = pd.DataFrame(columns=["案件ID", "マイルストーン名", "日付"])

    milestones = pd.concat([wide_ms, sheet_ms], ignore_index=True)

    # 「2020-08」と書いたつもりでも、Excel が勝手に 2020-08-01 という日付に
    # 変換してしまうことがある。そうなると読み込み側では月表記と区別がつかず、
    # 全マイルストーンが半月ぶん前に寄ったまま気づけない。
    # そこで既定は 月 とし、日付の細部を捨てて月央へ揃える。
    # 実績が月単位である以上、日まで分かっても効果は小さく(完了30件で相対4%、
    # 統計的にははっきりしない)、月初へ寄る事故のほうが害が大きい(同 12% 悪化)。
    # 日付が正確に分かっていてその精度を使いたい場合だけ 自動 にする。
    precision = str(settings.get("マイルストーン精度", "月")).strip() or "月"
    if precision not in ("自動", "月"):
        raise ValueError(
            f"マイルストーン精度 は 自動 / 月 のいずれかで指定してください(指定値: {precision!r})")
    if precision == "月" and not milestones.empty:
        keys = ["案件ID", "マイルストーン名", "日付"]
        before = len(milestones.drop_duplicates(subset=keys))
        snapped = milestones["日付"].apply(month_center)
        # 月央そのものでない日付が入っていた = 日まで書かれていた、ということ。
        # 既定は月精度なのでその細部は使われない。黙って捨てると
        # 「日まで入力したのに反映されない」の原因が追えなくなる。
        n_day = int((milestones["日付"] != snapped).sum())
        milestones["日付"] = snapped
        after = len(milestones.drop_duplicates(subset=keys))
        if n_day:
            ms_warnings.append(
                f"マイルストーン精度=月 のため、日まで書かれた {n_day} 件の日付を月央に丸めました。"
                "日の精度をそのまま使いたい場合は settings の マイルストーン精度 を 自動 にすること"
                "(ただし Excel が「2020-08」を 2020-08-01 に変換していないか要確認)。")
        if after < before:
            ms_warnings.append(
                f"月に丸めた結果、同じ月に入った複数回のマイルストーン {before - after} 件が"
                "1件にまとまりました。手戻り期間が1ヶ月未満だった分は区別できません。")

    milestones = milestones.drop_duplicates(subset=["案件ID", "マイルストーン名", "日付"])
    milestones, ms_attempts, repeat_warnings = _normalize_milestone_repeats(milestones)
    ms_warnings.extend(repeat_warnings)

    try:
        estimates = _read_table(master_path, "estimates")
        need = ["案件ID", "行程グループ", "見積人月"]
        missing = [c for c in need if c not in estimates.columns]
        if missing:
            raise ValueError(f"estimates シートに必須列がありません: {missing}")
        estimates["案件ID"] = estimates["案件ID"].astype(str).str.strip()
        estimates["行程グループ"] = estimates["行程グループ"].astype(str).str.strip()
        estimates["見積人月"] = pd.to_numeric(estimates["見積人月"], errors="coerce")
    except ValueError as e:
        # estimates シートが無くても動く。着手前の案件に見積もりが無い段階では
        # 契約人月 + 学習した工数比率だけで予測する(設計書 6 Step1 の第1階層)。
        if "estimates シートに必須列がありません" in str(e):
            raise
        estimates = pd.DataFrame(columns=["案件ID", "行程グループ", "見積人月"])

    phase_map = _read_table(master_path, "phase_map")
    for c in phase_map.columns:
        phase_map[c] = phase_map[c].astype(str).str.strip()

    # 集約軸は settings と overrides の両方で決まるため、確定した後に検証する。
    phase_map, phase_map_warnings = _check_phase_map(phase_map, str(settings["集約軸"]))

    actuals, actuals_warnings = _read_actuals(actuals_path, use_cache=use_cache)

    ds = Dataset(
        projects=projects,
        milestones=milestones,
        phase_map=phase_map,
        settings=settings,
        actuals=actuals,
        estimates=estimates,
        ms_attempts=ms_attempts,
        source={"master": os.path.abspath(master_path),
                "actuals": os.path.abspath(actuals_path)},
    )
    ds.warnings.extend(ms_warnings)
    ds.warnings.extend(phase_map_warnings)
    ds.warnings.extend(actuals_warnings)
    _check_consistency(ds)
    return ds


def _check_phase_map(phase_map: pd.DataFrame, group_col: str) -> tuple[pd.DataFrame, list[str]]:
    """phase_map の不備を取り除き、警告として返す。

    ある行程を集計するか否か、するならどのグループに入れるかは、
    この表だけが判断材料になっている。ここが静かに壊れると、
    実績CSVが正しくても集計対象そのものがずれる。

    見ているのは2点。
      1. 集約軸の値が空欄の行。groups 一覧に欠損が紛れ、出力に中身のない
         列が生える。さらにその行程は「phase_map に無い行程」として
         除外されるため、行を足したのに集計されないという状態になる。
      2. 実績行程名の重複。マッピングは dict(zip(...)) で作るため後勝ちになり、
         先に書いた割り当てが黙って捨てられる。工数は消えないが
         別グループに付け替わるので、工数比率だけが静かに歪む。
    """
    warnings: list[str] = []
    if group_col not in phase_map.columns or "実績行程名" not in phase_map.columns:
        return phase_map, warnings      # 列自体が無い場合は aggregate_actuals が止める

    blank = phase_map[group_col].isna() | (phase_map[group_col].astype(str).str.strip() == "")
    if blank.any():
        names = phase_map.loc[blank, "実績行程名"].astype(str).tolist()
        shown = ", ".join(names[:8]) + (" ほか" if len(names) > 8 else "")
        warnings.append(
            f"phase_map の「{group_col}」が空欄の行が {int(blank.sum())} 行あり、無視しました: {shown}。"
            "この行程は集計対象になりません。集計したい場合は行程グループを入力すること。")
        phase_map = phase_map.loc[~blank].reset_index(drop=True)

    dup = phase_map["実績行程名"].duplicated(keep=False)
    if dup.any():
        conflict = [nm for nm, sub in phase_map[dup].groupby("実績行程名", observed=True)
                    if sub[group_col].nunique() > 1]
        for nm in conflict[:8]:
            hit = phase_map.loc[phase_map["実績行程名"] == nm, group_col].astype(str).tolist()
            warnings.append(
                f"phase_map で行程「{nm}」が複数の{group_col}に登録されています({' / '.join(hit)})。"
                f"最後の「{hit[-1]}」だけが使われ、他は無視されます。"
                "この行程の工数はすべてそのグループに集計されるため、工数比率が意図とずれる。")
        if len(conflict) > 8:
            warnings.append(f"  ほか {len(conflict) - 8} 行程が重複登録されています。")

    return phase_map, warnings


def _check_consistency(ds: Dataset) -> None:
    """マスタと実績のズレを検出し、警告として記録する。

    実績CSVは機械が吐く前提なので、廃番の案件・検証用の案件・別部門の案件など、
    projects シートに載っていない案件IDが混ざるのが普通。
    これは異常ではなく想定内なので、止めずに除外し、何を除外したかを残す。
    """
    known = ds.known_ids
    found = set(ds.actuals["案件ID"].astype(str).unique())

    # 実績にあって projects に無い案件は build_project_curves が人月つきで報告する。
    # ここで重ねて警告すると同じ話が2回出て、本当に見てほしい警告が埋もれる。
    no_actual = sorted(known - found - set(
        ds.projects.loc[ds.projects["ステータス"] == "予測対象", "案件ID"].astype(str)))
    if no_actual:
        ds.warnings.append(
            f"projects にあって実績が1行も無い案件 {len(no_actual)} 件: "
            + ", ".join(no_actual[:8]) + (" ほか" if len(no_actual) > 8 else ""))

    if ds.milestones.empty:
        ds.warnings.append(
            "マイルストーンが1件も登録されていません。位置合わせは無効となり、"
            "素朴版(全案件のカーブを単純平均)と同じ結果になります。")


def _to_month_str(v) -> str:
    """'2021-04' / Timestamp / datetime を 'YYYY-MM' に正規化する。"""
    if isinstance(v, str):
        return pd.Period(v.strip()[:7], freq="M").strftime("%Y-%m")
    return pd.Period(pd.Timestamp(v), freq="M").strftime("%Y-%m")


_MONTH_SEP = re.compile(r"^(\d{4})\D(\d{1,2})(?:\D.*)?$")   # 2019-4 / 2019/04 / 2019-04-01
_MONTH_FLAT = re.compile(r"^(\d{4})(\d{2})(?:\d{2})?$")      # 201904 / 20190401


def _to_month_key(v: str) -> str | None:
    """実績側の年月表記を 'YYYY-MM' に正規化する。解釈できなければ None。

    受け付ける表記: 2019-4 / 2019-04 / 2019/4 / 2019.4 / 2019-04-01 / 201904 …
    """
    t = v.translate(_ZEN_DIGITS).strip()
    if not t:
        return None
    m = _MONTH_SEP.match(t) or _MONTH_FLAT.match(t)
    if not m:
        return None
    year, month = int(m.group(1)), int(m.group(2))
    if not 1 <= month <= 12:
        return None
    return f"{year:04d}-{month:02d}"


def _normalize_months(raw: pd.Series) -> tuple[pd.Series, dict | None, dict | None]:
    """「月」列を 'YYYY-MM' に揃える。

    ここを素通しにすると被害が大きい。案件の月リストは
    pd.period_range から作られるため常にゼロ埋めの 'YYYY-MM' だが、
    実績側が '2019-4' のようにゼロ埋めされていないと文字列として一致せず、
    build_project_curves の reindex で丸ごと落ちる。
    実績CSV側の表記が首尾一貫していても関係ない。
    ゼロ埋めなしで完全に統一されていても、10〜12月だけが偶然どちらの
    表記でも同じ文字列になるため生き残り、1〜9月の9ヶ月分が消える。
    しかも文字列比較では '2019-4' > '2019-04' となるため、
    期間内のはずの実績が「契約期間外(終了後)」として除外され、
    警告を読んでも真因が年月の桁揃えだとはわからない。
    1〜9月がすべてこれに該当するので、実績の7〜8割が消えることもある。

    月の異なり数はたかだか数百なので、ユニーク値だけ変換して貼り直す。
    80万行でも一瞬で終わる。

    戻り値: (正規化後の月, 表記ゆれの情報, 解釈不能の情報)
    """
    s = raw.fillna("").astype(str).str.strip()
    uniq = s.unique()

    mapping, bad, reformatted = {}, [], []
    for v in uniq:
        key = _to_month_key(v)
        if key is None:
            mapping[v] = v          # 解釈できない値は原文のまま残し、警告で報告する
            if v:
                bad.append(v)
            continue
        mapping[v] = key
        if key != v:
            reformatted.append((v, key))

    out = s.map(mapping)

    fixed_info = None
    if reformatted:
        n = int(s.isin([v for v, _ in reformatted]).sum())
        shown = ", ".join(f"{v!r}→{k!r}" for v, k in reformatted[:3])
        fixed_info = {"count": n, "examples": shown}

    bad_info = None
    if bad:
        n = int(s.isin(bad).sum())
        bad_info = {"count": n, "examples": ", ".join(repr(v) for v in bad[:5])}

    return out, fixed_info, bad_info


def _parse_hours(raw: pd.Series) -> tuple[pd.Series, dict | None]:
    """「時間」列を数値に変換する。

    手入力・他システムからのエクスポートが混ざる実績CSVでは、
    3桁区切りカンマ("1,234.5")や全角数字("３８．９")が
    人の目には正常な数値に見えたまま紛れ込む。
    これらを pd.to_numeric にそのまま渡すと NaN になり、
    かつては警告ゼロで 0 に落ちていた(=実績が黙って消える)。

    ここでは全角→半角変換と桁区切りカンマの除去を行ったうえで数値化し、
    それでもパースできなかった行(空欄ではないのに数値にならない行)だけを
    「本当に壊れているデータ」として件数・実例つきで報告する。
    空欄はもともと「その月の実績なし」を意味するので警告の対象にしない。
    """
    s = raw.fillna("").astype(str).str.strip()
    normalized = s.str.translate(_ZEN_DIGITS).str.replace(",", "", regex=False)
    parsed = pd.to_numeric(normalized, errors="coerce")

    blank = s == ""
    bad = parsed.isna() & ~blank
    parsed = parsed.fillna(0.0)

    if not bad.any():
        return parsed, None
    examples = s[bad].unique()[:5].tolist()
    return parsed, {"count": int(bad.sum()), "examples": examples}


def _read_actuals(path: str, use_cache: bool = True) -> tuple[pd.DataFrame, list[str]]:
    """実績CSVを読む。

    設計書 4-4 のとおり dtype=category を指定し、
    読み込み結果を parquet にキャッシュして再実行を速くする。

    戻り値は (実績DataFrame, 読み込み時に検出した警告) のタプル。
    警告はパース時にしか分からない(一度 category/float に変換すると
    元の生の文字列は失われる)ため、parquet と一緒にサイドカーへ
    キャッシュし、キャッシュヒット時にも再現する。
    """
    cache = _cache_path(path)
    wcache = _warnings_cache_path(cache) if cache else None
    if use_cache and cache and os.path.exists(cache):
        try:
            df = pd.read_parquet(cache)
            warnings: list[str] = []
            if wcache and os.path.exists(wcache):
                with open(wcache, encoding="utf-8") as f:
                    warnings = json.load(f)
            return df, warnings
        except Exception:
            pass  # キャッシュが壊れていたら普通に読む

    # 案件ID・行程・月 は projects/milestones/phase_map 側では既に
    # str.strip() 済み(load_all 参照)。実績側だけ素通しだと、
    # 前後の半角/全角スペースが1文字あるだけで別物として扱われ、
    # 「未登録案件」「phase_map に無い行程」として静かに除外される。
    # 除外自体は警告に出るが、原因が空白だとは表示上わからず、
    # 見た目には正しいIDなのに実績が消えたように見える。
    df = pd.read_csv(path, dtype=str, encoding="utf-8-sig")
    required = ["案件ID", "行程", "月", "時間"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"実績データに必須列がありません: {missing}")
    for c in ("メンバー", "所属", "チーム"):
        if c not in df.columns:
            df[c] = pd.NA  # 属性は欠損してよい(設計書 3-3)

    for c in ("案件ID", "行程", "メンバー", "所属", "チーム", "月"):
        df[c] = df[c].str.strip()

    warnings = []

    df["月"], fixed_months, bad_months = _normalize_months(df["月"])
    if fixed_months:
        warnings.append(
            f"実績データの「月」列に、本システムが内部で使う 'YYYY-MM' 形式"
            f"(月を必ず2桁ゼロ埋め)と異なる表記が {fixed_months['count']:,} 行あり、"
            f"正規化しました(例: {fixed_months['examples']})。"
            "正規化しない場合は案件の月リストと文字列一致せず、期間内の実績が"
            "「契約期間外」として除外される。1〜9月だけが該当するため"
            "(10〜12月は両表記が同一)、実績の約4分の3が消える。"
        )
    if bad_months:
        warnings.append(
            f"実績データの「月」列に年月として解釈できない値が {bad_months['count']:,} 行あり、"
            f"どの月にも割り当てられませんでした。例: {bad_months['examples']} 。"
            "この分は集計から丸ごと漏れる。"
        )

    df["時間"], bad = _parse_hours(df["時間"])

    if bad:
        shown = ", ".join(repr(v) for v in bad["examples"])
        warnings.append(
            f"実績データの「時間」列に数値として読み取れない値が {bad['count']:,} 行あり、"
            f"0として扱いました(桁区切りカンマ・単位付き・全角混在などが疑われる)。例: {shown} 。"
            "この分は集計から漏れるため、実績合計が入力データの見た目の値より小さくなる。"
            "元データの表記を確認すること。"
        )

    for c in ("案件ID", "行程", "メンバー", "所属", "チーム", "月"):
        df[c] = df[c].astype("category")

    if use_cache and cache:
        try:
            df.to_parquet(cache, index=False)
            if wcache:
                with open(wcache, "w", encoding="utf-8") as f:
                    json.dump(warnings, f, ensure_ascii=False)
        except Exception:
            pass  # pyarrow が無い環境でも動作を止めない
    return df, warnings


def _cache_path(path: str) -> str | None:
    try:
        st = os.stat(path)
        key = hashlib.md5(
            f"{CACHE_SCHEMA_VERSION}|{os.path.abspath(path)}|{st.st_mtime_ns}|{st.st_size}"
            .encode()).hexdigest()[:12]
        d = os.path.join(os.path.dirname(os.path.abspath(path)), ".cache")
        os.makedirs(d, exist_ok=True)
        return os.path.join(d, f"actuals_{key}.parquet")
    except Exception:
        return None


def _warnings_cache_path(cache_path: str) -> str:
    return cache_path[: -len(".parquet")] + ".warnings.json"
