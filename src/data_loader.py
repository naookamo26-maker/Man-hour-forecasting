"""
読み込み層(設計書 4-2)

「DataFrame を複数返す関数」という形に閉じ込める。
将来データベース化する際は、このモジュールだけ差し替えれば移行できる。

計算部・出力部はここで返す Dataset だけに依存し、
Excel/CSV というファイル形式の存在を知らない。
"""

from __future__ import annotations

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

DEFAULT_SETTINGS = {
    "人月換算係数": 160.0,
    "集約軸": "行程グループ",
    "位置合わせ": "ON",
    "背骨マイルストーン": "自動",
    "背骨最小カバー率": 0.6,
    "マイルストーン最小件数": 3,
    "カーブ解像度": 100,
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
        elif key in ("マイルストーン最小件数", "カーブ解像度"):
            val = int(val)
        else:
            val = str(val).strip()
        out[key] = val
    return out


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
    if "タグ" not in projects.columns:
        projects["タグ"] = ""
    projects["タグ"] = projects["タグ"].fillna("")

    try:
        milestones = _read_table(master_path, "milestones")
        milestones["案件ID"] = milestones["案件ID"].astype(str).str.strip()
        milestones["マイルストーン名"] = milestones["マイルストーン名"].astype(str).str.strip()
        milestones["日付"] = pd.to_datetime(milestones["日付"])
    except ValueError:
        # マイルストーンシートが無くても動く(設計書 3-2)
        milestones = pd.DataFrame(columns=["案件ID", "マイルストーン名", "日付"])

    phase_map = _read_table(master_path, "phase_map")
    for c in phase_map.columns:
        phase_map[c] = phase_map[c].astype(str).str.strip()

    settings = _read_settings(master_path)
    if overrides:
        settings.update({k: v for k, v in overrides.items() if v is not None})

    actuals, actuals_warnings = _read_actuals(actuals_path, use_cache=use_cache)

    ds = Dataset(
        projects=projects,
        milestones=milestones,
        phase_map=phase_map,
        settings=settings,
        actuals=actuals,
        source={"master": os.path.abspath(master_path),
                "actuals": os.path.abspath(actuals_path)},
    )
    ds.warnings.extend(actuals_warnings)
    _check_consistency(ds)
    return ds


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
            f"実績データの「月」列が 'YYYY-MM' で揃っていない行が {fixed_months['count']:,} 行あり、"
            f"正規化しました(例: {fixed_months['examples']})。"
            "正規化しないと案件の月リストと文字列一致せず、期間内の実績が"
            "「契約期間外」として除外される。出力側の表記を揃えることが望ましい。"
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
