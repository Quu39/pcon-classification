"""データの読み込み、.npy キャッシュ、パスのパース、元写真の復元。

使い方（プロジェクトルートで）:
    python -m src.data            # train/test のキャッシュを作る（既にあれば検証してスキップ）
    python -m src.data --force    # 作り直す
"""
from __future__ import annotations

import argparse
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TILE = 256
N_COLS = 18
N_ROWS = 14
N_TILES = N_COLS * N_ROWS  # 252
# 一部のファイルは "7107-28_2.jpg" のように _2 が付く（SDNET2018 側の重複名対策と思われる）。タイル番号は "-" の後の数字。
PATH_RE = re.compile(r"^W/(CW|UW)/(\d+)-(\d+)(?:_(\d+))?\.jpg$")


# ---------------------------------------------------------------- config
def load_config(path: str | Path | None = None) -> dict:
    path = Path(path) if path else PROJECT_ROOT / "config.yaml"
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_path(cfg: dict, key: str) -> Path:
    p = Path(cfg["paths"][key])
    return p if p.is_absolute() else PROJECT_ROOT / p


def get_sdnet_root(cfg: dict) -> Path:
    """config の sdnet_root から、W/CW と W/UW を持つフォルダを返す。"""
    raw = str(cfg.get("sdnet_root") or "").strip()
    if not raw or raw.startswith("<"):
        raise RuntimeError("config.yaml の sdnet_root が未設定です（SDNET2018 のパスを書いてください）")
    root = Path(raw)
    for cand in (root, root / "SDNET2018"):
        if (cand / "W" / "CW").is_dir() and (cand / "W" / "UW").is_dir():
            return cand
    raise FileNotFoundError(f"{root} の下に W/CW と W/UW が見つかりません")


def sdnet_is_set(cfg: dict) -> bool:
    try:
        get_sdnet_root(cfg)
        return True
    except (RuntimeError, FileNotFoundError):
        return False


# ---------------------------------------------------------------- パスのパース
def parse_path(p: str) -> tuple[str, int, int, int]:
    """'W/CW/7069-3.jpg' -> ('CW', 7069, 3, 0)、'W/CW/7107-28_2.jpg' -> ('CW', 7107, 28, 2)"""
    m = PATH_RE.match(p)
    if m is None:
        raise ValueError(f"想定外のパス形式: {p!r}")
    return m.group(1), int(m.group(2)), int(m.group(3)), int(m.group(4) or 0)


def tile_rc(tile: np.ndarray | int, n_cols: int = N_COLS):
    """タイル番号(1始まり) -> (row, col)。行優先。"""
    t = np.asarray(tile) - 1
    return t // n_cols, t % n_cols


def add_meta(df: pd.DataFrame, n_cols: int = N_COLS, n_rows: int = N_ROWS) -> pd.DataFrame:
    parsed = [parse_path(p) for p in df["path"]]
    df = df.copy()
    df["crack"] = [x[0] for x in parsed]
    df["photo_id"] = np.array([x[1] for x in parsed], dtype=np.int64)
    df["tile"] = np.array([x[2] for x in parsed], dtype=np.int64)
    df["suffix"] = np.array([x[3] for x in parsed], dtype=np.int64)  # _2 なら 2、無ければ 0
    # 同じ (写真ID, タイル番号) が複数あるとき、suffix の小さい方以外を True にする（test の 7074-105 だけ）
    df["is_dup"] = df.sort_values(["photo_id", "tile", "suffix"]).duplicated(["photo_id", "tile"]).reindex(df.index)
    r, c = tile_rc(df["tile"].to_numpy(), n_cols)
    df["row"] = r
    df["col"] = c
    df["is_edge"] = (r == 0) | (r == n_rows - 1) | (c == 0) | (c == n_cols - 1)
    return df


def load_train_df(cfg: dict) -> pd.DataFrame:
    df = pd.read_csv(resolve_path(cfg, "train_label"), header=None, names=["path", "label"])
    df["label"] = df["label"].astype(np.int64)
    g = cfg.get("grid", {})
    return add_meta(df, g.get("n_cols", N_COLS), g.get("n_rows", N_ROWS))


def load_test_df(cfg: dict) -> pd.DataFrame:
    df = pd.read_csv(resolve_path(cfg, "test_list"), header=None, names=["path"])
    g = cfg.get("grid", {})
    return add_meta(df, g.get("n_cols", N_COLS), g.get("n_rows", N_ROWS))


def load_df(cfg: dict, split: str) -> pd.DataFrame:
    return {"train": load_train_df, "test": load_test_df}[split](cfg)


# ---------------------------------------------------------------- 画像の読み込みとキャッシュ
def read_image(path: Path) -> tuple[np.ndarray, bool]:
    """RGB uint8 (256,256,3) を返す。日本語パス対策で imdecode を使う。2つ目は resize したかどうか。"""
    buf = np.fromfile(str(path), dtype=np.uint8)
    img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    if img is None:
        raise IOError(f"画像を読めません: {path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    resized = False
    if img.shape[:2] != (TILE, TILE):
        img = cv2.resize(img, (TILE, TILE), interpolation=cv2.INTER_AREA)
        resized = True
    return img, resized


def cache_paths(cfg: dict, split: str) -> tuple[Path, Path]:
    d = resolve_path(cfg, "cache_dir")
    return d / f"{split}_images.npy", d / f"{split}_index.csv"


def build_cache(cfg: dict, split: str, force: bool = False) -> Path:
    """split の全画像を (N,256,256,3) uint8 の .npy に書く。行順は train_label.csv / test.txt と同じ。"""
    df = load_df(cfg, split)
    npy_path, idx_path = cache_paths(cfg, split)
    if npy_path.exists() and idx_path.exists() and not force:
        idx = pd.read_csv(idx_path)
        arr = np.load(npy_path, mmap_mode="r")
        if idx["path"].tolist() == df["path"].tolist() and arr.shape == (len(df), TILE, TILE, 3):
            print(f"[cache] {split}: 既存キャッシュを使用 {npy_path} {arr.shape}")
            return npy_path
        print(f"[cache] {split}: 既存キャッシュが一致しないので作り直します")

    root = get_sdnet_root(cfg)
    files = [root / p for p in df["path"]]
    missing = [str(f) for f in files if not f.exists()]
    if missing:
        raise FileNotFoundError(f"{split}: {len(missing)} 件の画像が見つかりません。例: {missing[:3]}")

    npy_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = npy_path.with_suffix(".tmp.npy")
    arr = np.lib.format.open_memmap(tmp, mode="w+", dtype=np.uint8, shape=(len(df), TILE, TILE, 3))
    n = len(files)
    n_resized = 0
    t0 = time.time()
    step = max(1, n // 20)
    with ThreadPoolExecutor(max_workers=int(cfg.get("cache", {}).get("num_threads", 8))) as ex:
        for i, (img, rs) in enumerate(ex.map(read_image, files)):
            arr[i] = img
            n_resized += rs
            if (i + 1) % step == 0 or i + 1 == n:
                el = time.time() - t0
                eta = el / (i + 1) * (n - i - 1)
                print(f"[cache] {split}: {i + 1}/{n}  経過 {el:.0f}s  残り約 {eta:.0f}s", flush=True)
    arr.flush()
    del arr
    tmp.replace(npy_path)
    df[["path"]].to_csv(idx_path, index=False)
    print(f"[cache] {split}: 完了 {npy_path}  ({time.time() - t0:.1f}s, resize={n_resized} 件)")
    return npy_path


def load_images(cfg: dict, split: str, mmap: bool = True) -> np.ndarray:
    """キャッシュを読む。行順が CSV と一致することを確認する。"""
    npy_path, idx_path = cache_paths(cfg, split)
    if not npy_path.exists():
        raise FileNotFoundError(f"キャッシュがありません: {npy_path}（python -m src.data を先に実行）")
    idx = pd.read_csv(idx_path)
    df = load_df(cfg, split)
    assert idx["path"].tolist() == df["path"].tolist(), "キャッシュと CSV の行順が一致しません"
    return np.load(npy_path, mmap_mode="r" if mmap else None)


# ---------------------------------------------------------------- 元写真の復元
def tile_positions(tiles: np.ndarray, n_cols: int = N_COLS, n_rows: int = N_ROWS,
                   order: str = "row", offset: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """タイル番号 -> (row, col)。order='row' は行優先、'col' は列優先。offset は番号のずらし量。"""
    pos = np.asarray(tiles) - 1 + offset
    if order == "row":
        return pos // n_cols, pos % n_cols
    if order == "col":
        return pos % n_rows, pos // n_rows
    raise ValueError(order)


def photo_tiles(images: np.ndarray, df: pd.DataFrame, photo_id: int,
                dedup: bool = True) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """写真 1 枚分の (tiles(n,256,256,3), tile番号(n,), df の行インデックス(n,)) をタイル番号順に返す。
    dedup=True なら同じタイル番号の重複（is_dup）を除く。欠けたタイル番号はそのまま欠ける。"""
    m = df["photo_id"].to_numpy() == photo_id
    if dedup:
        m &= ~df["is_dup"].to_numpy()
    sub = df.index[m].to_numpy()
    sub = sub[np.argsort(df.loc[sub, "tile"].to_numpy())]
    return np.asarray(images[sub]), df.loc[sub, "tile"].to_numpy(), sub


def reconstruct_photo(tiles: np.ndarray, tile_ids: np.ndarray, n_cols: int = N_COLS, n_rows: int = N_ROWS,
                      order: str = "row", offset: int = 0) -> np.ndarray:
    """タイルを並べて元写真を復元する。既定は 18列×14行・行優先 → (3584, 4608, 3)。欠けたタイルは黒。"""
    r, c = tile_positions(tile_ids, n_cols, n_rows, order, offset)
    H = int(max(n_rows, r.max() + 1)) * TILE
    W = int(max(n_cols, c.max() + 1)) * TILE
    canvas = np.zeros((H, W, 3), dtype=np.uint8)
    filled = np.zeros((H // TILE, W // TILE), bool)
    for t, ri, ci in zip(tiles, r, c):
        if filled[ri, ci]:  # 重複タイルは最初のものを使う
            continue
        filled[ri, ci] = True
        canvas[ri * TILE:(ri + 1) * TILE, ci * TILE:(ci + 1) * TILE] = t
    return canvas


# ---------------------------------------------------------------- fold
def _sgkf_folds(df: pd.DataFrame, n_splits: int, seed: int) -> np.ndarray:
    from sklearn.model_selection import StratifiedGroupKFold

    fold = np.full(len(df), -1)
    sp = StratifiedGroupKFold(n_splits, shuffle=True, random_state=seed).split(df, df["label"], df["photo_id"])
    for k, (_, va) in enumerate(sp):
        fold[va] = k
    return fold


def _balanced_group_folds(df: pd.DataFrame, n_splits: int, seed: int, n_trials: int = 200_000) -> np.ndarray:
    """写真単位で 5 等分（8 枚ずつ）し、fold ごとの陽性数の std → CW 数の std が最小の割り当てを
    seed 固定のランダム探索で選ぶ。sklearn の StratifiedGroupKFold はどの seed でも陽性数が 32〜47 に
    偏るため（Step 1 で確認）、こちらを既定にしている。"""
    per = df.groupby("photo_id").agg(pos=("label", "sum"), cw=("crack", lambda s: (s == "CW").sum()))
    ids, pos, cw = per.index.to_numpy(), per["pos"].to_numpy(), per["cw"].to_numpy()
    rng = np.random.default_rng(seed)
    best_key, best_perm = None, None
    for _ in range(n_trials):
        perm = rng.permutation(len(ids))
        key = (round(pos[perm].reshape(n_splits, -1).sum(1).std(), 6), cw[perm].reshape(n_splits, -1).sum(1).std())
        if best_key is None or key < best_key:
            best_key, best_perm = key, perm
    photo_fold = {int(p): k for k, grp in enumerate(ids[best_perm].reshape(n_splits, -1)) for p in grp}
    return df["photo_id"].map(photo_fold).to_numpy()


def make_folds(cfg: dict, n_splits: int = 5) -> pd.DataFrame:
    df = load_train_df(cfg)
    seed = int(cfg.get("folds_seed", cfg.get("seed", 42)))
    assert df["photo_id"].nunique() % n_splits == 0
    path = resolve_path(cfg, "folds")
    path.parent.mkdir(parents=True, exist_ok=True)
    cols = ["path", "photo_id", "tile", "crack", "label"]
    df[cols].assign(fold=_sgkf_folds(df, n_splits, seed)).to_csv(path.with_name("folds_sgkf.csv"), index=False)
    out = df[cols].assign(fold=_balanced_group_folds(df, n_splits, seed))
    out.to_csv(path, index=False)
    pos = out.groupby("fold")["label"].sum()
    print(f"[folds] 写真単位・陽性数均等化 seed={seed}（陽性数 std={pos.std(ddof=0):.2f}）→ {path}")
    summ = out.groupby("fold").agg(photos=("photo_id", "nunique"), tiles=("path", "size"), pos=("label", "sum"))
    summ["pos_rate"] = (summ["pos"] / summ["tiles"]).round(4)
    summ["photo_ids"] = out.groupby("fold")["photo_id"].apply(lambda s: " ".join(map(str, sorted(s.unique()))))
    print(summ.to_string())
    return out


def load_folds(cfg: dict) -> pd.DataFrame:
    f = pd.read_csv(resolve_path(cfg, "folds"))
    assert f["path"].tolist() == load_train_df(cfg)["path"].tolist(), "folds.csv と train_label.csv の行順が違います"
    return f


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--split", choices=["train", "test", "all"], default="all")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--make-folds", action="store_true", help="outputs/folds.csv を作る（既にあれば --force が必要）")
    a = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    cfg = load_config(a.config)
    if a.make_folds:
        if resolve_path(cfg, "folds").exists() and not a.force:
            raise SystemExit("folds.csv は既にあります（固定）。作り直すなら --force")
        make_folds(cfg)
        return
    t0 = time.time()
    for s in (["train", "test"] if a.split == "all" else [a.split]):
        build_cache(cfg, s, force=a.force)
    print(f"[cache] 合計 {time.time() - t0:.1f}s")


if __name__ == "__main__":
    sys.exit(main())
