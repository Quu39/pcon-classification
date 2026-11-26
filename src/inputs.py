"""1 段目 CNN の入力ソース。

- margin=0 : タイル単体 (256×256)
- margin=m : 復元した元写真から対象タイルを中心に (256+2m)^2 を切り出す。写真の外側は reflect padding、
             欠けたタイルは train 全体の平均色で埋める（重複タイル 7074-105 は 2 つが同一画像なので最初の方）。
画像はすべて GPU に置き、切り出しも GPU 上で行う。
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from src.data import N_COLS, N_ROWS, TILE, photo_tiles, reconstruct_photo


class TileSource:
    def __init__(self, images: np.ndarray, df: pd.DataFrame, margin: int = 0, fill_rgb=None, device="cuda"):
        """images: (N,256,256,3) uint8（df と同じ行順）。df: load_train_df / load_test_df の結果。"""
        self.margin = int(margin)
        self.size = TILE + 2 * self.margin
        self.device = device
        if self.margin == 0:
            self.tiles = torch.from_numpy(np.ascontiguousarray(images)).to(device)
            return
        m = self.margin
        fill = np.asarray(fill_rgb if fill_rgb is not None else images[::20].reshape(-1, 3).mean(0), np.float64)
        fill = np.clip(np.round(fill), 0, 255).astype(np.uint8)
        pids = sorted(df["photo_id"].unique())
        H, W = N_ROWS * TILE + 2 * m, N_COLS * TILE + 2 * m
        self.canvas = torch.empty((len(pids), H, W, 3), dtype=torch.uint8, device=device)
        for i, pid in enumerate(pids):
            tiles, tids, _ = photo_tiles(images, df, pid)
            c = reconstruct_photo(tiles, tids)
            r, cc = (tids - 1) // N_COLS, (tids - 1) % N_COLS
            have = np.zeros((N_ROWS, N_COLS), bool)
            have[r, cc] = True
            for rr, c2 in zip(*np.where(~have)):  # 欠けたタイル → 平均色
                c[rr * TILE:(rr + 1) * TILE, c2 * TILE:(c2 + 1) * TILE] = fill
            c = np.pad(c, ((m, m), (m, m), (0, 0)), mode="reflect")
            self.canvas[i] = torch.from_numpy(c).to(device)
        pmap = {p: i for i, p in enumerate(pids)}
        self.pidx = df["photo_id"].map(pmap).to_numpy()
        self.y0 = (((df["tile"].to_numpy() - 1) // N_COLS) * TILE).astype(np.int64)
        self.x0 = (((df["tile"].to_numpy() - 1) % N_COLS) * TILE).astype(np.int64)

    def __call__(self, idx) -> torch.Tensor:
        """idx: 行インデックス（np.ndarray / list / torch.Tensor）→ (B,S,S,3) uint8 cuda"""
        if self.margin == 0:
            it = idx if torch.is_tensor(idx) else torch.as_tensor(np.asarray(idx), device=self.device)
            return self.tiles[it]
        idx = idx.cpu().numpy() if torch.is_tensor(idx) else np.asarray(idx)
        S = self.size
        return torch.stack([self.canvas[self.pidx[i], self.y0[i]:self.y0[i] + S, self.x0[i]:self.x0[i] + S]
                            for i in idx])
