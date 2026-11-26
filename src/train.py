"""1 段目 CNN の学習（fold ごと）。画像キャッシュを GPU に載せ、拡張も GPU で行う（DataLoader 不使用）。

    python -m src.train --exp configs/exp001.yaml               # 全 fold
    python -m src.train --exp configs/exp001.yaml --folds 0     # fold 0 だけ（所要時間の計測）
    python -m src.train --exp configs/exp001.yaml --folds 1 2 3 4

各 fold の出力: outputs/{name}/model_fold{k}.pt（valid F1 最良 epoch）, model_fold{k}_last.pt（最終 epoch）,
oof_fold{k}.csv / oof_last_fold{k}.csv, history_fold{k}.csv, log.txt
5 fold 揃うと oof.csv / oof_last.csv と metrics.json（best / last の 2 通り）を作り、experiments.md 用の行を表示する。
入力は exp の input.margin（周辺文脈のマージン px）と input.resize（縮小後のサイズ）で指定する。
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import WeightedRandomSampler

from src.data import PROJECT_ROOT, load_config, load_folds, load_images, load_train_df
from src.inputs import TileSource
from src.metrics import average_precision_score, best_threshold, evaluate_oof, neighbor_ratio
from src.models import build_from_cfg

GRAY_W = (0.299, 0.587, 0.114)


# ---------------------------------------------------------------- utils
def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_exp(path: str | Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


class Logger:
    def __init__(self, path: Path):
        self.path = path

    def __call__(self, msg: str) -> None:
        print(msg, flush=True)
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(msg + "\n")


# ---------------------------------------------------------------- GPU 拡張
class GpuAugment:
    """x: float (B,3,H,W) in [0,1]。幾何（90°回転・反転）と色（明るさ・コントラスト・彩度・色相・色かぶり・グレー化）。"""

    def __init__(self, a: dict):
        self.a = a or {}

    @staticmethod
    def _u(lo, hi, shape, dev):
        return torch.empty(shape, device=dev).uniform_(lo, hi)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        a, B, dev = self.a, x.shape[0], x.device
        if a.get("rot90"):
            k = torch.randint(0, 4, (B,), device=dev)
            out = x.clone()
            for r in (1, 2, 3):
                m = k == r
                if m.any():
                    out[m] = torch.rot90(x[m], r, (2, 3))
            x = out
        for key, dim in (("hflip", 3), ("vflip", 2)):
            if a.get(key, 0) > 0:
                m = (torch.rand(B, device=dev) < a[key]).view(B, 1, 1, 1)
                x = torch.where(m, x.flip(dim), x)
        w = torch.tensor(GRAY_W, device=dev).view(1, 3, 1, 1)
        if a.get("brightness", 0) > 0:
            b = a["brightness"]
            x = x * self._u(1 - b, 1 + b, (B, 1, 1, 1), dev)
        if a.get("contrast", 0) > 0:
            c = a["contrast"]
            mu = x.mean((1, 2, 3), keepdim=True)
            x = (x - mu) * self._u(1 - c, 1 + c, (B, 1, 1, 1), dev) + mu
        if a.get("saturation", 0) > 0:
            s = a["saturation"]
            g = (x * w).sum(1, keepdim=True)
            x = g + (x - g) * self._u(1 - s, 1 + s, (B, 1, 1, 1), dev)
        if a.get("hue_deg", 0) > 0:  # YIQ 空間での色相回転
            th = self._u(-1, 1, (B,), dev) * math.radians(a["hue_deg"])
            M = torch.tensor([[0.299, 0.587, 0.114], [0.596, -0.274, -0.322], [0.211, -0.523, 0.312]], device=dev)
            Mi = torch.linalg.inv(M)
            R = torch.zeros(B, 3, 3, device=dev)
            R[:, 0, 0] = 1
            R[:, 1, 1], R[:, 1, 2], R[:, 2, 1], R[:, 2, 2] = th.cos(), -th.sin(), th.sin(), th.cos()
            T = Mi @ R @ M
            x = torch.einsum("bij,bjhw->bihw", T, x)
        if a.get("channel_gain", 0) > 0:  # 写真ごとの青み・黄みの色かぶり
            gch = a["channel_gain"]
            x = x * self._u(1 - gch, 1 + gch, (B, 3, 1, 1), dev)
        if a.get("grayscale_p", 0) > 0:
            m = (torch.rand(B, device=dev) < a["grayscale_p"]).view(B, 1, 1, 1)
            x = torch.where(m, (x * w).sum(1, keepdim=True).expand_as(x), x)
        return x.clamp_(0, 1)


# ---------------------------------------------------------------- 入力
def u8_to_float(u8: torch.Tensor) -> torch.Tensor:
    """(B,H,W,3) uint8 -> (B,3,H,W) float [0,1]"""
    return u8.permute(0, 3, 1, 2).float().div_(255)


def resize(x: torch.Tensor, size: int | None) -> torch.Tensor:
    if not size or x.shape[-1] == size:
        return x
    return F.interpolate(x, size=(size, size), mode="bilinear", antialias=True, align_corners=False)


def normalize(x: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    return ((x - mean) / std).contiguous(memory_format=torch.channels_last)


def channel_stats(images: np.ndarray, step: int = 20) -> tuple[list[float], list[float]]:
    s = images[::step].astype(np.float32) / 255
    return s.mean((0, 1, 2), dtype=np.float64).tolist(), s.std((0, 1, 2), dtype=np.float64).tolist()


@torch.no_grad()
def predict_logits(model, get_u8, n: int, bs: int, mean, std, amp_dtype, tta: bool = False,
                   size: int | None = None) -> torch.Tensor:
    """get_u8(start, end) -> (b,H,W,3) uint8 cuda。tta=True なら 8 通り（回転4×反転2）の logit 平均。"""
    model.eval()
    out = []
    for s in range(0, n, bs):
        x = normalize(resize(u8_to_float(get_u8(s, min(n, s + bs))), size), mean, std)
        views = [x] if not tta else [torch.rot90(v, r, (2, 3)) for v in (x, x.flip(3)) for r in range(4)]
        acc = 0
        for v in views:
            with torch.autocast("cuda", dtype=amp_dtype):
                acc = acc + model(v.contiguous(memory_format=torch.channels_last)).float().squeeze(1)
        out.append(acc / len(views))
    return torch.cat(out)


# ---------------------------------------------------------------- 学習
def train_fold(exp: dict, fold: int, source: TileSource, folds: pd.DataFrame, out_dir: Path,
               mean_std: tuple, log: Logger) -> dict:
    t_cfg, a_cfg = exp["train"], exp.get("aug", {})
    size = exp.get("input", {}).get("resize")
    seed = int(t_cfg.get("seed", 42)) + fold
    seed_everything(seed)
    dev = torch.device("cuda")
    amp_dtype = torch.bfloat16 if t_cfg.get("amp_dtype", "bf16") == "bf16" else torch.float16
    mean = torch.tensor(mean_std[0], device=dev).view(1, 3, 1, 1)
    std = torch.tensor(mean_std[1], device=dev).view(1, 3, 1, 1)

    y_all = np.array(folds["label"].to_numpy(), dtype=np.int64)
    tr_idx = np.where(folds["fold"].to_numpy() != fold)[0]
    va_idx = np.where(folds["fold"].to_numpy() == fold)[0]
    y_tr, y_va = y_all[tr_idx], y_all[va_idx]
    r = float(t_cfg.get("pos_fraction", 0.2))  # 陽性:陰性 = 1:4
    w = np.where(y_tr == 1, r / y_tr.sum(), (1 - r) / (len(y_tr) - y_tr.sum()))
    g = torch.Generator().manual_seed(seed)
    sampler = WeightedRandomSampler(torch.as_tensor(w, dtype=torch.double), num_samples=len(tr_idx),
                                    replacement=True, generator=g)

    model = build_from_cfg(exp["model"]).to(dev)
    model = model.to(memory_format=torch.channels_last)
    opt = torch.optim.AdamW(model.parameters(), lr=t_cfg["lr"], weight_decay=t_cfg.get("weight_decay", 0.05))
    bs, epochs = int(t_cfg["batch_size"]), int(t_cfg["epochs"])
    steps_per_epoch = len(tr_idx) // bs
    total, warm = steps_per_epoch * epochs, int(steps_per_epoch * t_cfg.get("warmup_epochs", 1))
    min_f = float(t_cfg.get("min_lr_factor", 0.01))

    def lr_lambda(s):
        if s < warm:
            return (s + 1) / warm
        return min_f + (1 - min_f) * 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, total - warm)))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
    aug = GpuAugment(a_cfg)
    y_gpu = torch.as_tensor(y_all, dtype=torch.float32, device=dev)
    eval_bs = int(exp.get("eval", {}).get("batch_size", 256))
    ck_base = dict(mean=mean_std[0], std=mean_std[1], model=exp["model"], fold=fold, input=exp.get("input", {}))

    best = dict(f1=-1.0, ap=-1.0)
    hist = []
    t0 = time.time()
    for ep in range(epochs):
        te = time.time()
        model.train()
        order = tr_idx[np.fromiter(iter(sampler), dtype=np.int64)]
        tot_loss, n_loss = 0.0, 0
        for st in range(steps_per_epoch):
            bi = order[st * bs:(st + 1) * bs]
            x = normalize(aug(resize(u8_to_float(source(bi)), size)), mean, std)
            with torch.autocast("cuda", dtype=amp_dtype):
                logit = model(x).float().squeeze(1)
            loss = F.binary_cross_entropy_with_logits(logit, y_gpu[torch.as_tensor(bi, device=dev)])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            if st % 20 == 0:
                tot_loss += loss.item()
                n_loss += 1
        tr_loss = tot_loss / max(1, n_loss)

        logits = predict_logits(model, lambda s, e: source(va_idx[s:e]), len(va_idx), eval_bs, mean, std,
                                amp_dtype, size=size)
        va_loss = F.binary_cross_entropy_with_logits(logits, y_gpu[torch.as_tensor(va_idx, device=dev)]).item()
        prob = torch.sigmoid(logits).cpu().numpy()
        thr, f1 = best_threshold(y_va, prob)
        ap = float(average_precision_score(y_va, prob))
        improved = (f1, ap) > (best["f1"], best["ap"])
        if improved:
            best = dict(f1=f1, ap=ap, thr=thr, epoch=ep + 1, prob=prob.copy())
            torch.save(dict(state_dict=model.state_dict(), epoch=ep + 1, **ck_base), out_dir / f"model_fold{fold}.pt")
        el = time.time() - t0
        eta = el / (ep + 1) * (epochs - ep - 1)
        hist.append(dict(epoch=ep + 1, lr=opt.param_groups[0]["lr"], train_loss=tr_loss, valid_loss=va_loss,
                         valid_f1=f1, valid_thr=thr, valid_ap=ap, sec=time.time() - te))
        log(f"[fold {fold}] ep {ep + 1:2d}/{epochs} lr {opt.param_groups[0]['lr']:.2e} "
            f"train_loss {tr_loss:.4f} valid_loss {va_loss:.4f} valid_F1 {f1:.4f} (thr {thr:.2f}) AP {ap:.4f}"
            f"{' *' if improved else ''}  {time.time() - te:.1f}s  残り約 {eta / 60:.1f}分")
    # 最終 epoch の重みと OOF も残す（(b) 最終 epoch 固定の評価用）
    torch.save(dict(state_dict=model.state_dict(), epoch=epochs, **ck_base), out_dir / f"model_fold{fold}_last.pt")
    pd.DataFrame(hist).to_csv(out_dir / f"history_fold{fold}.csv", index=False)
    base = folds.iloc[va_idx][["path", "photo_id", "tile", "label", "fold"]]
    base.assign(prob=best["prob"]).to_csv(out_dir / f"oof_fold{fold}.csv", index=False)
    base.assign(prob=prob).to_csv(out_dir / f"oof_last_fold{fold}.csv", index=False)
    sec = time.time() - t0
    log(f"[fold {fold}] best epoch {best['epoch']}  valid_F1 {best['f1']:.4f} (thr {best['thr']:.2f}) "
        f"AP {best['ap']:.4f} | last valid_F1 {f1:.4f} (thr {thr:.2f})  所要 {sec / 60:.1f}分")
    return dict(fold=fold, best_epoch=best["epoch"], f1=best["f1"], thr=best["thr"], ap=best["ap"],
                last_f1=f1, sec=sec)


def finalize(exp: dict, out_dir: Path, n_folds: int, log: Logger) -> dict | None:
    secs = {}
    for k in range(n_folds):
        p = out_dir / f"fold{k}_summary.json"
        if p.exists():
            secs[k] = json.loads(p.read_text(encoding="utf-8"))
    res = {}
    for kind, pat in (("best", "oof_fold{}.csv"), ("last", "oof_last_fold{}.csv")):
        files = [out_dir / pat.format(k) for k in range(n_folds)]
        if not all(f.exists() for f in files):
            log(f"[finalize] {kind}: 未完了の fold {[k for k, f in enumerate(files) if not f.exists()]}")
            continue
        oof = pd.concat([pd.read_csv(f) for f in files])
        oof = oof.set_index("path").loc[load_train_df(load_config())["path"]].reset_index()  # train_label.csv の行順
        oof.to_csv(out_dir / ("oof.csv" if kind == "best" else "oof_last.csv"), index=False)
        m = evaluate_oof(oof)
        m["neighbor_ratio"] = neighbor_ratio(oof, oof["prob"].to_numpy() >= m["threshold"])
        res[kind] = m
    if not res:
        return None
    res["fold_best_epoch"] = [secs[k]["best_epoch"] for k in sorted(secs)]
    res["train_minutes"] = round(sum(s["sec"] for s in secs.values()) / 60, 1)
    (out_dir / "metrics.json").write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
    log("[finalize] " + json.dumps(res, ensure_ascii=False))
    for kind in ("best", "last"):
        if kind in res:
            m = res[kind]
            log(f"[experiments.md:{kind}] {exp['name']} | {exp.get('description', '')} | {m['oof_f1']:.4f} | "
                f"{m['threshold']:.2f} | {' / '.join(f'{v:.3f}' for v in m['fold_f1_at_pooled_thr'])} "
                f"(std {m['fold_f1_std']:.3f}) | {m['pred_pos_rate'] * 100:.2f}% ({m['pred_pos']}) | "
                f"{res['train_minutes']}分 | 隣接率 {m['neighbor_ratio']['ratio']}")
    return res


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", required=True)
    ap.add_argument("--config", default=None)
    ap.add_argument("--folds", type=int, nargs="*", default=None)
    ap.add_argument("--epochs", type=int, default=None, help="動作確認・時間計測用の上書き")
    ap.add_argument("--name", default=None, help="出力名の上書き（動作確認用）")
    ap.add_argument("--finalize-only", action="store_true")
    a = ap.parse_args()
    cfg = load_config(a.config)
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
        finalize(exp, out_dir, n_folds, log)
        return
    with open(out_dir / "config.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(exp, f, allow_unicode=True, sort_keys=False)

    torch.backends.cudnn.benchmark = True
    t0 = time.time()
    images = np.asarray(load_images(cfg, "train", mmap=False))
    mean_std = channel_stats(images)
    inp = exp.get("input", {}) or {}
    fill = np.array(mean_std[0]) * 255  # 欠けたタイルは train 全体の平均色
    source = TileSource(images, load_train_df(cfg), margin=int(inp.get("margin", 0)), fill_rgb=fill)
    del images
    log(f"[{exp['name']}] {exp.get('description', '')}  入力 {source.size}px"
        f"{' → ' + str(inp['resize']) if inp.get('resize') else ''}  準備 {time.time() - t0:.1f}s"
        f"  mean {np.round(mean_std[0], 3).tolist()} std {np.round(mean_std[1], 3).tolist()}")

    for k in (a.folds if a.folds is not None else range(n_folds)):
        res = train_fold(exp, k, source, folds, out_dir, mean_std, log)
        (out_dir / f"fold{k}_summary.json").write_text(json.dumps(res), encoding="utf-8")
    finalize(exp, out_dir, n_folds, log)


if __name__ == "__main__":
    main()
