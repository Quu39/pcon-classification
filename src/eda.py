"""Step 0: 元写真の並び（18列×14行・行優先）の検証と EDA。

    python -m src.eda                 # 画像キャッシュがあれば画像部分も実行
    python -m src.eda --labels-only   # ラベル CSV だけで分かる部分のみ

出力はすべて outputs/eda/ に保存する。テスト画像は数値（境界の差）にだけ使い、画像としては保存しない。
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter

import cv2
import numpy as np
import pandas as pd

from src.data import (N_COLS, N_ROWS, TILE, cache_paths, load_config, load_images, load_test_df,
                      load_train_df, photo_tiles, reconstruct_photo, resolve_path, tile_positions)

RED = (255, 0, 0)


def save_rgb(path, img: np.ndarray, quality: int = 90) -> None:
    ok, buf = cv2.imencode(path.suffix, cv2.cvtColor(img, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, quality])
    assert ok
    buf.tofile(str(path))


# ======================================================================== ラベルだけで分かること
def per_photo_table(tr: pd.DataFrame, te: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for split, df in (("train", tr), ("test", te)):
        for pid, g in df.groupby("photo_id"):
            missing = sorted(set(range(1, N_COLS * N_ROWS + 1)) - set(g["tile"]))
            rows.append(dict(split=split, photo_id=pid, n_tiles=len(g), tile_min=g["tile"].min(),
                             tile_max=g["tile"].max(), n_suffix=int((g["suffix"] > 0).sum()),
                             n_dup=int(g["is_dup"].sum()), missing_tiles=" ".join(map(str, missing)),
                             n_cw=int((g["crack"] == "CW").sum()),
                             n_pos=int(g["label"].sum()) if "label" in g else np.nan))
    return pd.DataFrame(rows)


def components_4(cells: set[tuple[int, int]]) -> list[list[tuple[int, int]]]:
    """4近傍の連結成分。"""
    seen, comps = set(), []
    for s in cells:
        if s in seen:
            continue
        stack, comp = [s], []
        seen.add(s)
        while stack:
            r, c = stack.pop()
            comp.append((r, c))
            for nb in ((r + 1, c), (r - 1, c), (r, c + 1), (r, c - 1)):
                if nb in cells and nb not in seen:
                    seen.add(nb)
                    stack.append(nb)
        comps.append(comp)
    return comps


def label_layout_check(tr: pd.DataFrame) -> dict:
    """陽性タイル同士の番号差と、列数仮説ごとの「4近傍に陽性の仲間がいる割合」。"""
    diffs = Counter()
    wrap_plus1 = 0
    for _, g in tr[tr["label"] == 1].groupby("photo_id"):
        t = sorted(g["tile"].tolist())
        s = set(t)
        for a in t:
            for d in (1, 17, 18, 19):
                if a + d in s:
                    diffs[d] += 1
                    if d == 1 and (a - 1) % N_COLS == N_COLS - 1:
                        wrap_plus1 += 1  # 18列仮説だと行の折り返しをまたぐ +1 ペア
    width_score = {}
    for w in range(14, 23):
        has_nb = tot = 0
        for _, g in tr[tr["label"] == 1].groupby("photo_id"):
            pos = {((t - 1) // w, (t - 1) % w) for t in g["tile"]}
            for r, c in pos:
                tot += 1
                has_nb += any(nb in pos for nb in ((r + 1, c), (r - 1, c), (r, c + 1), (r, c - 1)))
        width_score[w] = has_nb / tot
    return dict(pos_pair_diffs={int(k): int(v) for k, v in sorted(diffs.items())},
                plus1_pairs_crossing_row_wrap_18cols=wrap_plus1,
                frac_pos_with_4nb_pos_by_width={int(k): round(v, 4) for k, v in width_score.items()})


def label_eda(tr: pd.DataFrame) -> dict:
    out = {}
    out["n_train"], out["n_pos"] = len(tr), int(tr["label"].sum())
    out["pos_rate"] = round(tr["label"].mean(), 4)
    cw = tr.groupby("crack")["label"].agg(["count", "sum", "mean"])
    out["by_crack"] = {k: dict(n=int(v["count"]), pos=int(v["sum"]), rate=round(v["mean"], 4)) for k, v in cw.iterrows()}

    # 盤面上の位置
    is_corner = tr["row"].isin([0, N_ROWS - 1]) & tr["col"].isin([0, N_COLS - 1])
    pos_cls = np.where(is_corner, "corner", np.where(tr["is_edge"], "edge(非角)", "interior"))
    t = tr.assign(pos_cls=pos_cls).groupby("pos_cls")["label"].agg(["count", "sum", "mean"])
    out["by_position"] = {k: dict(n=int(v["count"]), pos=int(v["sum"]), rate=round(v["mean"], 4)) for k, v in t.iterrows()}
    # 外周からの距離（0=外周）
    ring = np.minimum.reduce([tr["row"], N_ROWS - 1 - tr["row"], tr["col"], N_COLS - 1 - tr["col"]])
    t = tr.assign(ring=ring).groupby("ring")["label"].agg(["count", "sum", "mean"])
    out["by_ring"] = {int(k): dict(n=int(v["count"]), pos=int(v["sum"]), rate=round(v["mean"], 4)) for k, v in t.iterrows()}

    # 陽性のかたまり（4近傍連結成分 ≒ Pコン跡 1 個分の目安）
    sizes, touching_edge = [], 0
    for _, g in tr[tr["label"] == 1].groupby("photo_id"):
        for comp in components_4(set(zip(g["row"], g["col"]))):
            sizes.append(len(comp))
            touching_edge += any(r in (0, N_ROWS - 1) or c in (0, N_COLS - 1) for r, c in comp)
    out["pos_components"] = dict(n=len(sizes), size_hist={int(k): int(v) for k, v in sorted(Counter(sizes).items())},
                                 touching_photo_edge=int(touching_edge))
    return out


def save_position_heatmap(tr: pd.DataFrame, eda_dir) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cnt = np.zeros((N_ROWS, N_COLS), int)
    pos = np.zeros((N_ROWS, N_COLS), int)
    np.add.at(cnt, (tr["row"], tr["col"]), 1)
    np.add.at(pos, (tr["row"], tr["col"]), tr["label"])
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.6))
    for ax, m, title in ((axes[0], pos, "positive tiles per grid cell (train, 40 photos)"),
                         (axes[1], pos / np.maximum(cnt, 1), "positive rate per grid cell")):
        im = ax.imshow(m, cmap="Reds")
        ax.set_title(title)
        ax.set_xticks(range(N_COLS))
        ax.set_yticks(range(N_ROWS))
        ax.set_xlabel("col")
        ax.set_ylabel("row")
        fig.colorbar(im, ax=ax, fraction=0.035)
        for r in range(N_ROWS):
            for c in range(N_COLS):
                if pos[r, c]:
                    ax.text(c, r, str(pos[r, c]) if m is pos else f"{m[r, c]:.2f}", ha="center", va="center", fontsize=6)
    fig.tight_layout()
    fig.savefig(eda_dir / "position_heatmap.png", dpi=120)
    plt.close(fig)


# ======================================================================== 画像を使う検証
def to_gray(tiles: np.ndarray) -> np.ndarray:
    return (tiles.astype(np.float32) @ np.array([0.299, 0.587, 0.114], np.float32))


def boundary_metric(g: np.ndarray, r: np.ndarray, c: np.ndarray) -> tuple[list[float], list[float]]:
    """並び (r,c) で隣り合うタイル境界の画素差（平均絶対差, 0-255）を横・縦それぞれ返す。"""
    inv = {(int(a), int(b)): i for i, (a, b) in enumerate(zip(r, c))}
    h, v = [], []
    for i, (a, b) in enumerate(zip(r, c)):
        j = inv.get((int(a), int(b) + 1))
        if j is not None:
            h.append(float(np.abs(g[i][:, -1] - g[j][:, 0]).mean()))
        j = inv.get((int(a) + 1, int(b)))
        if j is not None:
            v.append(float(np.abs(g[i][-1, :] - g[j][0, :]).mean()))
    return h, v


def hypotheses(tile_ids: np.ndarray, rng: np.random.Generator):
    yield "A 行優先 18列×14行（想定）", tile_positions(tile_ids, 18, 14, "row")
    yield "B 行優先 17列（列数を1ずらし）", tile_positions(tile_ids, 17, 15, "row")
    yield "C 行優先 19列（列数を1ずらし）", tile_positions(tile_ids, 19, 14, "row")
    yield "D 行優先 18列・番号を1ずらし", tile_positions(tile_ids, 18, 15, "row", offset=1)
    yield "E 列優先 14行×18列", tile_positions(tile_ids, 18, 14, "col")
    perm = rng.permutation(len(tile_ids))
    r, c = tile_positions(tile_ids, 18, 14, "row")
    yield "F ランダムな並び", (r[perm], c[perm])


def image_layout_check(images: np.ndarray, df: pd.DataFrame, rng: np.random.Generator) -> dict:
    """全写真で、並びの仮説ごとの境界差・タイル内部の基準差・最良一致の番号差を集計する。"""
    agg: dict[str, dict[str, list]] = {}
    ref_h, ref_v, wrap = [], [], []
    best_right, best_down = Counter(), Counter()
    right_ok = right_n = down_ok = down_n = 0
    for pid in sorted(df["photo_id"].unique()):
        tiles, tids, _ = photo_tiles(images, df, pid)
        g = to_gray(tiles)
        ref_h.append(float(np.abs(g[:, :, 127] - g[:, :, 128]).mean()))  # タイル内部の隣接列（自然なつながりの基準）
        ref_v.append(float(np.abs(g[:, 127, :] - g[:, 128, :]).mean()))
        for name, (r, c) in hypotheses(tids, rng):
            h, v = boundary_metric(g, r, c)
            d = agg.setdefault(name, {"h": [], "v": []})
            d["h"] += h
            d["v"] += v
        # 18列仮説で「行の折り返し」をまたぐ i → i+1（本当は隣でない）の境界差
        tpos = {int(t): i for i, t in enumerate(tids)}
        for i, t in enumerate(tids):
            if (t - 1) % 18 == 17 and (t + 1) in tpos:
                wrap.append(float(np.abs(g[i][:, -1] - g[tpos[t + 1]][:, 0]).mean()))
        # 仮説を置かない検証: 各タイルの右端/下端に最もよく合うタイルの番号差
        R, L = g[:, :, -1], g[:, :, 0]
        B, T = g[:, -1, :], g[:, 0, :]
        Dh = np.abs(R[:, None, :] - L[None, :, :]).mean(-1)
        Dv = np.abs(B[:, None, :] - T[None, :, :]).mean(-1)
        np.fill_diagonal(Dh, np.inf)
        np.fill_diagonal(Dv, np.inf)
        for i, t in enumerate(tids):
            dr = int(tids[Dh[i].argmin()] - t)
            dd = int(tids[Dv[i].argmin()] - t)
            best_right[dr] += 1
            best_down[dd] += 1
            if (t - 1) % 18 != 17:  # 右隣が存在するはずのタイル
                right_n += 1
                right_ok += dr == 1
            if (t - 1) // 18 != 13:
                down_n += 1
                down_ok += dd == 18
    res = {"reference_inside_tile": dict(h=round(np.mean(ref_h), 2), v=round(np.mean(ref_v), 2))}
    res["hypotheses"] = {k: dict(h_mean=round(np.mean(v["h"]), 2), v_mean=round(np.mean(v["v"]), 2),
                                 all_mean=round(np.mean(v["h"] + v["v"]), 2), n_pairs=len(v["h"]) + len(v["v"]))
                         for k, v in agg.items()}
    res["row_wrap_pairs_18cols"] = dict(mean=round(np.mean(wrap), 2) if wrap else None, n=len(wrap))
    res["best_match_right_offset_top5"] = best_right.most_common(5)
    res["best_match_down_offset_top5"] = best_down.most_common(5)
    res["best_match_right_is_+1"] = f"{right_ok}/{right_n} ({right_ok / max(right_n, 1):.1%})"
    res["best_match_down_is_+18"] = f"{down_ok}/{down_n} ({down_ok / max(down_n, 1):.1%})"
    return res


def suffix_tile_check(images: np.ndarray, df: pd.DataFrame) -> list[dict]:
    """_2 付きタイルと重複タイルが、18列行優先の位置で隣のタイルと自然につながるかを数値で見る。
    比較対象: 同じ写真の普通のタイルの境界差の平均（typical）。"""
    out = []
    for pid in sorted(df.loc[(df["suffix"] > 0) | df["is_dup"], "photo_id"].unique()):
        tiles, tids, rows = photo_tiles(images, df, pid)
        g = to_gray(tiles)
        r, c = tile_positions(tids, 18, 14, "row")
        h, v = boundary_metric(g, r, c)
        typical = float(np.mean(h + v))
        by_tile = {int(t): i for i, t in enumerate(tids)}
        cand = df[(df["photo_id"] == pid) & ((df["suffix"] > 0) | df["is_dup"])]
        # 重複タイルは両方の版を評価する
        cand = df[(df["photo_id"] == pid) & df["tile"].isin(cand["tile"])]
        for idx, row in cand.iterrows():
            gi = to_gray(np.asarray(images[idx])[None])[0]
            t = int(row["tile"])
            d = []
            for dt, a, b in ((-1, (slice(None), 0), (slice(None), -1)), (1, (slice(None), -1), (slice(None), 0)),
                             (-18, (0, slice(None)), (-1, slice(None))), (18, (-1, slice(None)), (0, slice(None)))):
                nb = t + dt
                if nb not in by_tile or (abs(dt) == 1 and (nb - 1) // 18 != (t - 1) // 18):
                    continue
                d.append(float(np.abs(gi[a] - g[by_tile[nb]][b]).mean()))
            out.append(dict(path=row["path"], n_neighbors=len(d), mean_diff=round(np.mean(d), 2) if d else None,
                            photo_typical=round(typical, 2), used_in_reconstruction=not bool(row["is_dup"])))
    return out


def draw_photo(canvas: np.ndarray, pos_rc, scale: int, grid: bool, tile_ids=None, rc=None) -> np.ndarray:
    small = cv2.resize(canvas, (canvas.shape[1] // scale, canvas.shape[0] // scale), interpolation=cv2.INTER_AREA)
    s = TILE // scale
    if grid:
        for x in range(0, small.shape[1] + 1, s):
            cv2.line(small, (x, 0), (x, small.shape[0]), (255, 255, 0), 1)
        for y in range(0, small.shape[0] + 1, s):
            cv2.line(small, (0, y), (small.shape[1], y), (255, 255, 0), 1)
        for t, (r, c) in zip(tile_ids, zip(*rc)):
            cv2.putText(small, str(t), (c * s + 3, r * s + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1, cv2.LINE_AA)
    for r, c in pos_rc:
        cv2.rectangle(small, (c * s + 1, r * s + 1), ((c + 1) * s - 2, (r + 1) * s - 2), RED, 3)
    return small


def save_reconstructions(images, tr, photo_ids, eda_dir) -> list[str]:
    saved = []
    for k, pid in enumerate(photo_ids):
        tiles, tids, rows = photo_tiles(images, tr, pid)
        labels = tr.loc[rows, "label"].to_numpy()
        r, c = tile_positions(tids, 18, 14, "row")
        canvas = reconstruct_photo(tiles, tids, 18, 14)
        assert canvas.shape == (3584, 4608, 3), canvas.shape
        pos_rc = [(int(a), int(b)) for a, b, l in zip(r, c, labels) if l == 1]
        n = f"photo_{pid}"
        save_rgb(eda_dir / f"{n}_rowmajor18.jpg", draw_photo(canvas, pos_rc, 4, False))
        save_rgb(eda_dir / f"{n}_rowmajor18_grid.jpg", draw_photo(canvas, pos_rc, 4, True, tids, (r, c)))
        saved += [f"{n}_rowmajor18.jpg", f"{n}_rowmajor18_grid.jpg"]
        # 陽性のかたまりの中心まわり 4×4 タイルを等倍で切り出す（左: そのまま, 右: 枠つき）
        if pos_rc:
            comp = max(components_4(set(pos_rc)), key=len)
            cr = int(np.clip(min(p[0] for p in comp) - 1, 0, 10))
            cc = int(np.clip(min(p[1] for p in comp) - 1, 0, 14))
            crop = canvas[cr * TILE:(cr + 4) * TILE, cc * TILE:(cc + 4) * TILE].copy()
            boxed = crop.copy()
            for a, b in pos_rc:
                if cr <= a < cr + 4 and cc <= b < cc + 4:
                    y, x = (a - cr) * TILE, (b - cc) * TILE
                    cv2.rectangle(boxed, (x + 2, y + 2), (x + TILE - 3, y + TILE - 3), RED, 4)
            pair = np.concatenate([crop, np.full((crop.shape[0], 16, 3), 255, np.uint8), boxed], 1)
            pair = cv2.resize(pair, (pair.shape[1] // 2, pair.shape[0] // 2), interpolation=cv2.INTER_AREA)
            save_rgb(eda_dir / f"{n}_zoom_r{cr}-{cr + 3}_c{cc}-{cc + 3}.jpg", pair)
            saved.append(f"{n}_zoom_r{cr}-{cr + 3}_c{cc}-{cc + 3}.jpg")
        if k == 0:  # 比較用: 間違った並び（19列）
            r19, c19 = tile_positions(tids, 19, 14, "row")
            wrong = reconstruct_photo(tiles, tids, 19, 14)
            pos19 = [(int(a), int(b)) for a, b, l in zip(r19, c19, labels) if l == 1]
            save_rgb(eda_dir / f"{n}_WRONG_rowmajor19.jpg", draw_photo(wrong, pos19, 4, False))
            saved.append(f"{n}_WRONG_rowmajor19.jpg")
    return saved


def save_thumbnails(images, tr, eda_dir, rng, n=16, size=160) -> None:
    def panel(idx, title):
        cells = []
        for i in idx:
            im = cv2.resize(np.asarray(images[i]), (size, size), interpolation=cv2.INTER_AREA)
            row = tr.loc[i]
            cv2.putText(im, f"{row.photo_id}-{row.tile} {row.crack}", (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                        (255, 255, 0), 1, cv2.LINE_AA)
            cells.append(im)
        grid = np.concatenate([np.concatenate(cells[r * 4:(r + 1) * 4], 1) for r in range(4)], 0)
        head = np.full((28, grid.shape[1], 3), 255, np.uint8)
        cv2.putText(head, title, (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1, cv2.LINE_AA)
        return np.concatenate([head, grid], 0)

    pos = rng.choice(tr.index[tr["label"] == 1].to_numpy(), n, replace=False)
    neg = rng.choice(tr.index[tr["label"] == 0].to_numpy(), n, replace=False)
    a, b = panel(pos, "POSITIVE (label=1)"), panel(neg, "NEGATIVE (label=0)")
    save_rgb(eda_dir / "thumbnails_pos16_neg16.jpg",
             np.concatenate([a, np.full((a.shape[0], 20, 3), 255, np.uint8), b], 1))


# ======================================================================== main
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--labels-only", action="store_true")
    ap.add_argument("--photos", type=int, nargs="*", default=None, help="復元する train 写真ID（既定: 陽性の多い 3 枚）")
    a = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    cfg = load_config(a.config)
    eda_dir = resolve_path(cfg, "eda_dir")
    eda_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(cfg.get("seed", 42))
    timings, summary = {}, {}

    t0 = time.time()
    tr, te = load_train_df(cfg), load_test_df(cfg)
    pp = per_photo_table(tr, te)
    pp.to_csv(eda_dir / "per_photo.csv", index=False)
    summary["photos"] = dict(train=int((pp.split == "train").sum()), test=int((pp.split == "test").sum()),
                             tiles_per_photo=sorted(pp["n_tiles"].unique().tolist()),
                             tile_range=[int(pp.tile_min.min()), int(pp.tile_max.max())],
                             photo_id_overlap=len(set(tr.photo_id) & set(te.photo_id)))
    summary["label_layout_check"] = label_layout_check(tr)
    summary["label_eda"] = label_eda(tr)
    summary["test_crack"] = te["crack"].value_counts().to_dict()
    save_position_heatmap(tr, eda_dir)
    timings["labels"] = round(time.time() - t0, 1)

    have_cache = all(cache_paths(cfg, s)[0].exists() for s in ("train", "test"))
    if not a.labels_only and not have_cache:
        print("[eda] 画像キャッシュが無いので画像部分をスキップ（sdnet_root を設定して python -m src.data を実行）")
    if not a.labels_only and have_cache:
        t1 = time.time()
        tri, tei = load_images(cfg, "train"), load_images(cfg, "test")
        summary["image_layout_check_train"] = image_layout_check(tri, tr, rng)
        summary["image_layout_check_test"] = image_layout_check(tei, te, rng)  # 数値のみ。画像は保存しない
        summary["suffix_tile_check"] = suffix_tile_check(tri, tr) + suffix_tile_check(tei, te)
        timings["layout_check"] = round(time.time() - t1, 1)
        t1 = time.time()
        ids = a.photos or pp[pp.split == "train"].sort_values("n_pos", ascending=False)["photo_id"].head(3).tolist()
        summary["reconstructed_photos"] = save_reconstructions(tri, tr, ids, eda_dir)
        save_thumbnails(tri, tr, eda_dir, rng)
        timings["images"] = round(time.time() - t1, 1)

    timings["total"] = round(time.time() - t0, 1)
    summary["timings_sec"] = timings
    (eda_dir / "eda_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=str),
                                              encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
