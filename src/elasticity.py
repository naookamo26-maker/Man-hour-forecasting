"""区間弾力性 ─ 狭く潰れたマイルストーン区間への工数配分を減らす

設計書 3-1(案A)は「区間の工数比率は学習値のまま、期間だけを伸縮する」と決めた。
その帰結として、区間が短いほど工数密度が上がる。区間が2ヶ月しかなくても
学習値どおり全体の30%を置くので、月次工数が鋭い山になる。

案Aの根拠は「期間が延びるのは承認待ちや会議間隔が理由であり、作業量が増える
わけではない」だった。これは正しい。ただし **これは区間が伸びる側の話** である。

    伸びる側  密度が薄くなるだけ。実在しない何かを要求してはいない  → 案Aで正しい
    縮む側    2ヶ月に30%を置くには要員を3倍にするしかない          → 案Aは破綻する

短い区間はたいてい「判断のための版を作った」区間で、工数そのものが軽い。
計画の期間と金額は守られるので、予算消化はおおむね時間軸に沿って進む。
一方でカーブの形(山谷の付き方)はマイルストーン区間に強く従う。
そこで **潰れた区間の総量だけを減らし、区間内の形は学習カーブのまま** とする。

── 定式化

比べる相手は「時間比率」ではなく **学習データでのその区間の長さ** である。
α版〜β版が平均より濃いのは学習した実態であって、直すべき歪みではない。
直したいのは「その案件でだけ区間が異常に潰れている」ぶんだけである。

区間 i について

    s  正準区間長   学習データでのその区間の長さ(マイルストーンの平均位置の間隔)
    t  実区間長     この案件でのその区間の長さ
    j  = t / s      伸縮率。ワープの stretch_factors そのもの。1 未満 = 潰れている
    c  正準比率     学習カーブがその区間に置く工数比率

    f = j^λ  ただし j < 1 の区間だけ。j ≧ 1 の区間は f = 1
    w = c·f / Σ c·f

λ=0 で w = c、すなわち案Aと完全に一致する(既定値)。
j = 1(学習どおりの長さ)の区間は λ によらず動かない。
つまり **マイルストーンが学習データの平均どおりに並んでいる案件では何も起きない。**

λ=1 のとき、潰れた区間の工数密度 w/t は c/s ―― 圧縮が無かった場合の密度 ――
に一致する。すなわち λ は「区間が潰れたことによる密度の上昇を、どれだけ
打ち消すか」の割合そのものである。

伸縮率上限(Warp.max_stretch)が同じ目的を時間軸を縮めて果たすのに対し、
こちらはアンカーを1日も動かさない。指定したマイルストーンの日付は守られる。

── 学習と予測で同じ変換を使う

予測でだけ λ を効かせると、学習カーブ側に残っている区間長の影響と衝突する。
学習時は逆変換(deflate)で各案件の実績から潰れの影響を抜き、
予測時は順変換(inflate)で対象案件の潰れ具合に応じて入れ直す。
f は c に依存しないので、逆変換は c = (a/f) を正規化するだけで厳密に解ける。
λ=0 ではどちらも恒等変換なので、既定の挙動は一切変わらない。
"""

from __future__ import annotations

import numpy as np

# λ の上限。1 で「区間が潰れたことによる密度上昇を完全に打ち消す」ところまで届くので、
# それ以上に振る意味はない(振ると潰れた区間が学習より薄くなる)。
ELASTICITY_MAX = 1.0


def _overlap(n_bin: int, bounds: np.ndarray) -> np.ndarray:
    """正準軸のビン × 区間 の重なり割合 O[j, i] を返す。行和は 1。

    区間の境界(マイルストーンの正準位置)はビン境界とは一致しないので、
    またぐビンは按分する。按分しないと境界のビン1個分(既定で全体の1%)が
    まるごとどちらかの区間に寄り、区間の総量がその分だけずれる。
    """
    e = np.linspace(0.0, 1.0, n_bin + 1)
    lo = np.maximum(e[:-1, None], bounds[None, :-1])
    hi = np.minimum(e[1:, None], bounds[None, 1:])
    return np.clip(hi - lo, 0.0, None) * n_bin   # ビン幅 1/n_bin で割る


def interval_shares(shape: np.ndarray, bounds: np.ndarray) -> np.ndarray:
    """正準軸の形状が、各区間に置いている工数比率。"""
    shape = np.asarray(shape, dtype=float)
    return shape @ _overlap(len(shape), bounds)


def squeeze_factors(warp, lam: float) -> np.ndarray:
    """区間ごとの配分倍率 f。潰れた区間だけ 1 未満になる。

    j = 伸縮率(実区間長 ÷ 正準区間長)。j ≧ 1 の区間は 1 のまま返す。
    """
    j = np.asarray(warp.stretch_factors(), dtype=float)
    return np.where(j < 1.0, np.power(np.maximum(j, 1e-12), lam), 1.0)


def _renormalize(shape: np.ndarray, bounds: np.ndarray,
                 factors: np.ndarray) -> np.ndarray:
    """区間ごとに倍率をかけ、合計 1 に戻す。区間の中の形は変えない。"""
    O = _overlap(len(shape), bounds)
    out = np.asarray(shape, dtype=float) * (O @ factors)
    s = out.sum()
    return out / s if s > 0 else np.asarray(shape, dtype=float)


def bounds_of(warp) -> np.ndarray | None:
    """ワープから正準側の区間境界を取り出す。

    区間が1つしか無い(= マイルストーンが無く恒等写像)なら None。
    そのときは配分を変える余地がそもそも無い。
    """
    if warp is None or warp.is_identity:
        return None
    s = np.asarray(warp.anchors_canonical, dtype=float)
    if len(s) < 3 or np.any(np.diff(s) <= 0):
        return None
    if np.any(np.diff(np.asarray(warp.anchors_actual, dtype=float)) <= 0):
        return None
    return s


def inflate(shape: np.ndarray, warp, lam: float) -> np.ndarray:
    """予測用。潰れた区間への配分を減らし、その分を他の区間へ回す。

    区間の中の形は動かさないので、追い込みのような「区間内での後ろ寄り」は残る。
    動くのは「どの区間にいくら置くか」だけである。
    """
    shape = np.asarray(shape, dtype=float)
    b = bounds_of(warp)
    if b is None or lam <= 0.0:
        return shape
    return _renormalize(shape, b, squeeze_factors(warp, lam))


def deflate(shape: np.ndarray, warp, lam: float) -> np.ndarray:
    """学習用。実績カーブから、その案件で区間が潰れていた影響を抜く。

    inflate の逆。これを通してから平均することで、学習カーブは
    「区間の潰れを均した形」になり、予測側の inflate と二重にかからない。
    """
    shape = np.asarray(shape, dtype=float)
    b = bounds_of(warp)
    if b is None or lam <= 0.0:
        return shape
    return _renormalize(shape, b, 1.0 / squeeze_factors(warp, lam))


def describe(shape: np.ndarray, warp, lam: float) -> str:
    """区間ごとに「案Aが置く量 → 弾力性を効かせた量」を1行で表す(注記用)。"""
    b = bounds_of(warp)
    if b is None or lam <= 0.0:
        return ""
    c = interval_shares(shape, b)
    w = interval_shares(inflate(shape, warp, lam), b)
    names = ["開始"] + list(warp.names) + ["終了"]
    parts = [f"{names[i]}→{names[i + 1]}: {c[i]:.1%}→{w[i]:.1%}"
             for i in range(len(c)) if abs(w[i] - c[i]) >= 0.005]
    return " / ".join(parts)
