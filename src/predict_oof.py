"""保存済みの fold モデルで OOF を予測し直す（TTA 用）。各 fold モデルは自分の valid fold だけを予測する。

    python -m src.predict_oof --exp outputs/exp005 --which last --tta    # → oof_last_tta.csv
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

from src.data import PROJECT_ROOT, load_config, load_folds, load_images, load_train_df
from src.inputs import TileSource
from src.metrics import evaluate_oof
from src.models import build_from_cfg
from src.train import predict_logits


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", required=True)
    ap.add_argument("--which", choices=["best", "last"], default="last")
    ap.add_argument("--tta", action="store_true")
    a = ap.parse_args()
    cfg = load_config()
    exp_dir = Path(a.exp) if Path(a.exp).is_absolute() else PROJECT_ROOT / a.exp
    exp = yaml.safe_load((exp_dir / "config.yaml").read_text(encoding="utf-8"))
    amp_dtype = torch.bfloat16 if exp["train"].get("amp_dtype", "bf16") == "bf16" else torch.float16
    inp = exp.get("input", {}) or {}
    bs = int(exp.get("eval", {}).get("batch_size", 256))
    t0 = time.time()
    tr = load_train_df(cfg)
    folds = load_folds(cfg)
    fold = folds["fold"].to_numpy()
    suffix = "" if a.which == "best" else "_last"
    ck0 = torch.load(exp_dir / f"model_fold0{suffix}.pt", map_location="cpu", weights_only=False)
    source = TileSource(np.asarray(load_images(cfg, "train", mmap=False)), tr, margin=int(inp.get("margin", 0)),
                        fill_rgb=np.array(ck0["mean"]) * 255)
    prob = np.full(len(tr), np.nan)
    for k in sorted(np.unique(fold)):
        ck = torch.load(exp_dir / f"model_fold{k}{suffix}.pt", map_location="cuda", weights_only=False)
        assert ck["fold"] == k
        model = build_from_cfg(ck["model"]).cuda()
        model.load_state_dict(ck["state_dict"])
        model = model.to(memory_format=torch.channels_last)
        mean = torch.tensor(ck["mean"], device="cuda").view(1, 3, 1, 1)
        std = torch.tensor(ck["std"], device="cuda").view(1, 3, 1, 1)
        va = np.where(fold == k)[0]
        logit = predict_logits(model, lambda s, e: source(va[s:e]), len(va), bs, mean, std, amp_dtype,
                               tta=a.tta, size=inp.get("resize")).cpu().numpy()
        prob[va] = 1 / (1 + np.exp(-logit))
        print(f"[oof] fold {k} 完了 {time.time() - t0:.0f}s", flush=True)
    assert not np.isnan(prob).any()
    oof = folds[["path", "photo_id", "tile", "label", "fold"]].assign(prob=prob)
    name = f"oof{suffix}" + ("_tta" if a.tta else "_re") + ".csv"
    oof.to_csv(exp_dir / name, index=False)
    m = evaluate_oof(oof)
    print(f"[oof] {exp_dir.name} {a.which}{' TTA' if a.tta else ''}: OOF F1 {m['oof_f1']:.4f} thr {m['threshold']:.2f} "
          f"AP {m['ap']:.4f}  → {name}  ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
