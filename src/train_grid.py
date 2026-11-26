"""exp012: 3×3 タイル同時予測モデル（exp005 ベース: margin 256 → 768px を 384 に縮小、ResNet18）。

    python -m src.train_grid --exp configs/exp012.yaml                       # 全 fold（学習 + OOF + test）
    python -m src.train_grid --exp configs/exp012.yaml --folds 0 --max-steps 20 --name smoke012   # スモークテスト
    python -m src.train_grid --selftest                                       # 拡張・ラベル整合のテスト

- 出力: 最終特徴マップ (12×12) → AdaptiveAvgPool2d(3) → 1×1 conv で 3×3 個の logit（1 マス = 1 タイル）
- ラベル: crop の 3×3 タイルのラベル。写真の外と欠けたタイルはマスクして loss に入れない（マスク付き BCE、有効マスの平均）
- 拡張: 90°回転・反転は画像と 3×3 ラベル・マスクに同じ変換をかける。色の拡張は GpuAugment（幾何はオフ）
- サンプリング: 3×3 に陽性を含む crop : 含まない crop ≈ 1:3
- 評価は (b) 最終 epoch。各タイルの予測は「中心 crop だけ」と「自身 + 8 近傍を中心にした crop の最大 9 個の確率平均」の 2 通り
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from torch.utils.data import WeightedRandomSampler

from src.data import N_COLS, N_ROWS, PROJECT_ROOT, TILE, load_config, load_folds, load_images, load_test_df, load_train_df
from src.inputs import TileSource
from src.metrics import average_precision_score, best_threshold, evaluate_oof, neighbor_ratio
from src.train import GpuAugment, Logger, channel_stats, load_exp, normalize, resize, seed_everything, u8_to_float


class GridNet(nn.Module):
    def __init__(self, name: str, grid: int = 3, **kwargs):
        super().__init__()
        self.backbone = timm.create_model(name, pretrained=False, num_classes=0, global_pool="", **kwargs)
        self.pool = nn.AdaptiveAvgPool2d(grid)
        self.head = nn.Conv2d(self.backbone.num_features, 1, 1)

    def forward(self, x):  # (B,3,H,W) -> (B,3,3)
        return self.head(self.pool(self.backbone.forward_features(x))).squeeze(1)


def grid_labels(df: pd.DataFrame):
    """各行のタイルを中心とする 3×3 のラベル L とマスク M (N,3,3)。写真外・欠けたタイルは M=0。"""
    pids = sorted(df["photo_id"].unique())
    pmap = {p: i for i, p in enumerate(pids)}
    p = df["photo_id"].map(pmap).to_numpy()
    t = df["tile"].to_numpy()
    r, c = (t - 1) // N_COLS, (t - 1) % N_COLS
    lab = np.zeros((len(pids), N_ROWS + 2, N_COLS + 2), np.float32)
    val = np.zeros_like(lab, dtype=bool)
    val[p, r + 1, c + 1] = True
    if "label" in df:
        np.maximum.at(lab, (p, r + 1, c + 1), df["label"].to_numpy().astype(np.float32))
    ii = np.arange(3)
    P, R, C = p[:, None, None], r[:, None, None] + ii[None, :, None], c[:, None, None] + ii[None, None, :]
    return lab[P, R, C], val[P, R, C], p, r, c


def geo_aug(x, y, m):
    """x (B,3,H,W), y/m (B,3,3): 同じ 90°回転・左右反転・上下反転をかける。"""
    B, dev = x.shape[0], x.device
    k = torch.randint(0, 4, (B,), device=dev)
    xo, yo, mo = x.clone(), y.clone(), m.clone()
    for r in (1, 2, 3):
        s = k == r
        if s.any():
            xo[s], yo[s], mo[s] = torch.rot90(x[s], r, (2, 3)), torch.rot90(y[s], r, (1, 2)), torch.rot90(m[s], r, (1, 2))
    for dx, dy in ((3, 2), (2, 1)):
        s = torch.rand(B, device=dev) < 0.5
        xo = torch.where(s.view(B, 1, 1, 1), xo.flip(dx), xo)
        yo = torch.where(s.view(B, 1, 1), yo.flip(dy), yo)
        mo = torch.where(s.view(B, 1, 1), mo.flip(dy), mo)
    return xo, yo, mo


@torch.no_grad()
def predict_grid(model, source, idx, bs, mean, std, amp_dtype, size) -> np.ndarray:
    model.eval()
    out = []
    for s in range(0, len(idx), bs):
        x = normalize(resize(u8_to_float(source(idx[s:s + bs])), size), mean, std)
        with torch.autocast("cuda", dtype=amp_dtype):
            out.append(model(x).float())
    return torch.cat(out).cpu().numpy()


def aggregate(logits: np.ndarray, p, r, c):
    """logits (n,3,3)（各行のタイルを中心とする crop）→ 中心だけの確率と、最大 9 個の crop の確率平均。"""
    prob = 1 / (1 + np.exp(-logits))
    key = {}
    for i, k in enumerate(zip(p, r, c)):
        key.setdefault(k, i)  # 重複タイル（test 7074-105）は同一画像なので最初の行
    center = prob[:, 1, 1].copy()
    avg = np.empty(len(prob))
    cnt = np.empty(len(prob), np.int64)
    for i in range(len(prob)):
        s, n = 0.0, 0
        for di in (-1, 0, 1):
            for dj in (-1, 0, 1):
                j = key.get((p[i], r[i] + di, c[i] + dj))
                if j is not None:  # 写真外・欠けたタイルを中心とする crop はない
                    s += prob[j, 1 - di, 1 - dj]
                    n += 1
        avg[i], cnt[i] = s / n, n
    return center, avg, cnt


def train_fold(exp, fold, src_tr, src_te, folds, te_df, G, Gte, out_dir, mean_std, log, max_steps=None):
    t_cfg, a_cfg = exp["train"], dict(exp.get("aug", {}))
    size = exp["input"]["resize"]
    seed = int(t_cfg.get("seed", 42)) + fold
    seed_everything(seed)
    dev = torch.device("cuda")
    amp_dtype = torch.bfloat16
    mean = torch.tensor(mean_std[0], device=dev).view(1, 3, 1, 1)
    std = torch.tensor(mean_std[1], device=dev).view(1, 3, 1, 1)
    L, M, p, r, c = G
    fo = folds["fold"].to_numpy()
    tr_idx, va_idx = np.where(fo != fold)[0], np.where(fo == fold)[0]
    crop_pos = ((L * M).max((1, 2)) > 0)[tr_idx]
    q = float(t_cfg.get("pos_fraction", 0.25))
    w = np.where(crop_pos, q / crop_pos.sum(), (1 - q) / (~crop_pos).sum())
    g = torch.Generator().manual_seed(seed)
    sampler = WeightedRandomSampler(torch.as_tensor(w, dtype=torch.double), num_samples=len(tr_idx),
                                    replacement=True, generator=g)
    Lg, Mg = torch.as_tensor(L, device=dev), torch.as_tensor(M, dtype=torch.float32, device=dev)

    model = GridNet(exp["model"]["name"]).to(dev).to(memory_format=torch.channels_last)
    opt = torch.optim.AdamW(model.parameters(), lr=t_cfg["lr"], weight_decay=t_cfg.get("weight_decay", 0.05))
    bs, epochs = int(t_cfg["batch_size"]), int(t_cfg["epochs"])
    spe = len(tr_idx) // bs if not max_steps else max_steps
    total, warm = spe * epochs, int(spe * t_cfg.get("warmup_epochs", 1))
    min_f = float(t_cfg.get("min_lr_factor", 0.01))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / warm if s < warm else
        min_f + (1 - min_f) * 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, total - warm))))
    a_cfg.update(rot90=False, hflip=0, vflip=0)  # 幾何は geo_aug で画像とラベルに同時にかける
    color = GpuAugment(a_cfg)
    eval_bs = int(exp.get("eval", {}).get("batch_size", 128))
    y_va = folds["label"].to_numpy()[va_idx]
    t0 = time.time()
    for ep in range(epochs):
        te = time.time()
        model.train()
        order = tr_idx[np.fromiter(iter(sampler), dtype=np.int64)]
        tot, nl = 0.0, 0
        for st in range(spe):
            bi = order[st * bs:(st + 1) * bs]
            bt = torch.as_tensor(bi, device=dev)
            x, y, m = geo_aug(resize(u8_to_float(src_tr(bi)), size), Lg[bt], Mg[bt])
            x = normalize(color(x), mean, std)
            with torch.autocast("cuda", dtype=amp_dtype):
                logit = model(x).float()
            loss = (F.binary_cross_entropy_with_logits(logit, y, reduction="none") * m).sum() / m.sum()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            if st % 20 == 0:
                tot += loss.item()
                nl += 1
        lg = predict_grid(model, src_tr, va_idx, eval_bs, mean, std, amp_dtype, size)
        pc = 1 / (1 + np.exp(-lg[:, 1, 1]))
        thr, f1 = best_threshold(y_va, pc)
        ap = float(average_precision_score(y_va, pc))
        eta = (time.time() - t0) / (ep + 1) * (epochs - ep - 1)
        log(f"[fold {fold}] ep {ep + 1:2d}/{epochs} lr {opt.param_groups[0]['lr']:.2e} train_loss {tot / max(1, nl):.4f} "
            f"valid_F1(中心) {f1:.4f} (thr {thr:.2f}) AP {ap:.4f}  {time.time() - te:.1f}s  残り約 {eta / 60:.1f}分")
    torch.save(dict(state_dict=model.state_dict(), epoch=epochs, mean=mean_std[0], std=mean_std[1],
                    model=exp["model"], fold=fold, input=exp["input"]), out_dir / f"model_fold{fold}_last.pt")
    center, avg, cnt = aggregate(lg, p[va_idx], r[va_idx], c[va_idx])
    base = folds.iloc[va_idx][["path", "photo_id", "tile", "label", "fold"]]
    base.assign(prob_center=center, prob_avg9=avg, n_crops=cnt).to_csv(out_dir / f"oof_last_fold{fold}.csv", index=False)
    f_c, f_a = best_threshold(y_va, center)[1], best_threshold(y_va, avg)[1]
    lt = predict_grid(model, src_te, np.arange(len(te_df)), eval_bs, mean, std, amp_dtype, size)
    tc, ta, _ = aggregate(lt, *Gte[2:])
    np.save(out_dir / f"test_fold{fold}.npy", np.stack([tc, ta]))
    log(f"[fold {fold}] 完了 valid F1 中心 {f_c:.4f} / 9個平均 {f_a:.4f}  所要 {(time.time() - t0) / 60:.1f}分")


def finalize(out_dir: Path, n_folds: int, log) -> None:
    cfg = load_config()
    tr, te = load_train_df(cfg), load_test_df(cfg)
    files = [out_dir / f"oof_last_fold{k}.csv" for k in range(n_folds)]
    if not all(f.exists() for f in files):
        log("[finalize] 未完了の fold あり")
        return
    oof = pd.concat([pd.read_csv(f) for f in files]).set_index("path").loc[tr["path"]].reset_index()
    T = np.mean([np.load(out_dir / f"test_fold{k}.npy") for k in range(n_folds)], 0)
    res = {}
    for name, col, ti in (("center", "prob_center", 0), ("avg9", "prob_avg9", 1)):
        o = oof.assign(prob=oof[col])
        m = evaluate_oof(o)
        m["neighbor_ratio"] = neighbor_ratio(o, o["prob"].to_numpy() >= m["threshold"])
        m["test_pos"] = int((T[ti] >= m["threshold"]).sum())
        m["test_neighbor_ratio"] = neighbor_ratio(te, T[ti] >= m["threshold"])
        res[name] = m
        log(f"[{name}] OOF F1 {m['oof_f1']:.4f} thr {m['threshold']:.2f} ± {m['f1_around']} AP {m['ap']:.4f} "
            f"fold {m['fold_f1_at_pooled_thr']} 陽性 {m['pred_pos']} ({m['pred_pos_rate']:.2%}) test {m['test_pos']} "
            f"隣接率 {m['neighbor_ratio']['ratio']} / {m['test_neighbor_ratio']['ratio']}")
    ch = max(res, key=lambda k: res[k]["oof_f1"])
    ti = 0 if ch == "center" else 1
    oof.assign(prob=oof[f"prob_{ch}"]).to_csv(out_dir / "oof_last.csv", index=False)
    pd.DataFrame({"path": te["path"], "prob": T[ti], "prob_center": T[0], "prob_avg9": T[1]}).to_csv(
        out_dir / "test_pred_last.csv", index=False)
    (out_dir / "metrics.json").write_text(json.dumps({"last": res[ch], "chosen": ch, **res}, ensure_ascii=False, indent=2),
                                          encoding="utf-8")
    log(f"[finalize] 採用版（OOF F1 が高い方）: {ch}")


def selftest() -> None:
    """(1) geo_aug 後の画像の各マスとラベル・マスクが一致するか (2) 3×3 ラベルと crop 画像が隣のタイルと一致するか。"""
    torch.manual_seed(0)
    B = 64
    y = torch.rand(B, 3, 3, device="cuda")
    m = (torch.rand(B, 3, 3, device="cuda") > 0.3).float()
    x = torch.stack([y, m, y * 0], 1).repeat_interleave(128, 2).repeat_interleave(128, 3)  # 1 マス 128px
    xo, yo, mo = geo_aug(x, y, m)
    pooled = xo[:, :, 64::128, 64::128]  # 各マスの中心画素
    assert torch.equal(pooled[:, 0], yo) and torch.equal(pooled[:, 1], mo), "geo_aug 不一致"
    changed = (yo != y).flatten(1).any(1).float().mean().item()
    print(f"[selftest] geo_aug: 画像とラベル・マスクが一致（{B} 枚中 {changed:.0%} が変換済み）")

    cfg = load_config()
    tr = load_train_df(cfg)
    im = np.asarray(load_images(cfg, "train", mmap=False))
    L, M, p, r, c = grid_labels(tr)
    src = TileSource(im, tr, margin=256, fill_rgb=np.array(channel_stats(im)[0]) * 255)
    lut = {(pid, t): i for i, (pid, t) in enumerate(zip(tr["photo_id"], tr["tile"]))}
    rng = np.random.default_rng(0)
    pos_rows = np.where(tr["label"].to_numpy() == 1)[0]
    rows = np.concatenate([rng.choice(len(tr), 150, replace=False), rng.choice(pos_rows, 50, replace=False)])
    n_img = n_lab = n_out = 0
    for i in rows:
        crop = src(np.array([i]))[0].cpu().numpy()
        for a in range(3):
            for b in range(3):
                rr, cc = r[i] - 1 + a, c[i] - 1 + b
                inside = 0 <= rr < N_ROWS and 0 <= cc < N_COLS
                j = lut.get((tr["photo_id"].iloc[i], rr * N_COLS + cc + 1)) if inside else None
                assert M[i, a, b] == (j is not None), (i, a, b)
                if j is None:
                    n_out += 1
                    continue
                assert L[i, a, b] == tr["label"].iloc[j], (i, a, b)
                n_lab += 1
                if (crop[a * TILE:(a + 1) * TILE, b * TILE:(b + 1) * TILE] == im[j]).all():
                    n_img += 1
    print(f"[selftest] 3×3 ラベル: {len(rows)} crop（陽性 50 含む）の有効マス {n_lab} 個すべて一致、"
          f"写真外・欠損でマスク {n_out} 個 / crop 画像がタイル画像と一致 {n_img}/{n_lab}")
    print(f"[selftest] 3×3 に陽性を含む crop の割合 {((L * M).max((1, 2)) > 0).mean():.2%}")


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp")
    ap.add_argument("--folds", type=int, nargs="*", default=None)
    ap.add_argument("--max-steps", type=int, default=None)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--name", default=None)
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--finalize-only", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        selftest()
        return
    cfg = load_config()
    exp = load_exp(a.exp)
    if a.epochs:
        exp["train"]["epochs"] = a.epochs
    if a.name:
        exp["name"] = a.name
    out_dir = PROJECT_ROOT / "outputs" / exp["name"]
    out_dir.mkdir(parents=True, exist_ok=True)
    log = Logger(out_dir / "log.txt")
    folds = load_folds(cfg)
    n_folds = int(folds["fold"].max()) + 1
    if a.finalize_only:
        finalize(out_dir, n_folds, log)
        return
    with open(out_dir / "config.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(exp, f, allow_unicode=True, sort_keys=False)
    torch.backends.cudnn.benchmark = True
    t0 = time.time()
    tr, te = load_train_df(cfg), load_test_df(cfg)
    assert (folds["path"].to_numpy() == tr["path"].to_numpy()).all()
    im = np.asarray(load_images(cfg, "train", mmap=False))
    mean_std = channel_stats(im)
    fill = np.array(mean_std[0]) * 255
    margin = int(exp["input"]["margin"])
    src_tr = TileSource(im, tr, margin=margin, fill_rgb=fill)
    del im
    src_te = TileSource(np.asarray(load_images(cfg, "test", mmap=False)), te, margin=margin, fill_rgb=fill)
    G, Gte = grid_labels(tr), grid_labels(te.drop(columns=["label"], errors="ignore"))
    log(f"[{exp['name']}] {exp.get('description', '')}  入力 {src_tr.size}px → {exp['input']['resize']}  "
        f"準備 {time.time() - t0:.1f}s")
    for k in (a.folds if a.folds is not None else range(n_folds)):
        train_fold(exp, k, src_tr, src_te, folds, te, G, Gte, out_dir, mean_std, log, a.max_steps)
    finalize(out_dir, n_folds, log)


if __name__ == "__main__":
    main()
