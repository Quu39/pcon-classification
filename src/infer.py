"""テスト推論: outputs/{exp}/model_fold*.pt で test を予測し、fold 平均を保存する。

    python -m src.infer --exp outputs/exp002                 # valid F1 最良 epoch の重み → test_pred.csv
    python -m src.infer --exp outputs/exp002 --which last    # 最終 epoch の重み → test_pred_last.csv
    python -m src.infer --exp outputs/exp002 --tta           # 8 通り TTA → *_tta.csv

出力列: path, fold{k}（確率）, logit{k}, prob（5 fold の確率平均）, prob_logitmean（logit 平均の sigmoid）
あわせて、しきい値ごとの test 陽性数（確率平均 / logit 平均）と、予測陽性の隣接率を表示・保存する。
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

from src.data import PROJECT_ROOT, load_config, load_images, load_test_df
from src.inputs import TileSource
from src.metrics import neighbor_ratio
from src.models import build_from_cfg
from src.train import predict_logits


def test_report(exp_dir: Path, pred: pd.DataFrame, te: pd.DataFrame, which: str) -> dict:
    """OOF と test の分布比較、しきい値ごとの test 陽性数（確率平均 vs logit 平均）、隣接率。"""
    q = [0.5, 0.9, 0.98, 0.99, 0.995]
    rep = {}
    for col in ("prob", "prob_logitmean"):
        rep[col] = dict(mean=round(float(pred[col].mean()), 4),
                        quantiles={str(k): round(float(v), 4) for k, v in pred[col].quantile(q).items()})
    mpath = exp_dir / "metrics.json"
    if mpath.exists():
        m = json.loads(mpath.read_text(encoding="utf-8"))[which]
        thr = m["threshold"]
        oof = pd.read_csv(exp_dir / ("oof.csv" if which == "best" else "oof_last.csv"))
        rep["oof"] = dict(mean=round(float(oof["prob"].mean()), 4),
                          quantiles={str(k): round(float(v), 4) for k, v in oof["prob"].quantile(q).items()})
        thrs = sorted({round(t, 2) for t in (0.3, 0.5, 0.7, 0.8, 0.9, thr - 0.05, thr, thr + 0.05) if 0 < t < 1})
        rep["pos_by_threshold"] = {f"{t:.2f}": dict(oof=int((oof["prob"] >= t).sum()),
                                                    test_probmean=int((pred["prob"] >= t).sum()),
                                                    test_logitmean=int((pred["prob_logitmean"] >= t).sum()))
                                   for t in thrs}
        rep["oof_threshold"] = thr
        rep["neighbor_ratio"] = dict(
            oof=m.get("neighbor_ratio"),
            test_probmean=neighbor_ratio(te, pred["prob"].to_numpy() >= thr),
            test_logitmean=neighbor_ratio(te, pred["prob_logitmean"].to_numpy() >= thr))
    return rep


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", required=True)
    ap.add_argument("--config", default=None)
    ap.add_argument("--which", choices=["best", "last"], default="best")
    ap.add_argument("--tta", action="store_true", help="回転4×反転2 の 8 通り平均")
    ap.add_argument("--batch-size", type=int, default=None)
    a = ap.parse_args()
    cfg = load_config(a.config)
    exp_dir = Path(a.exp) if Path(a.exp).is_absolute() else PROJECT_ROOT / a.exp
    exp = yaml.safe_load((exp_dir / "config.yaml").read_text(encoding="utf-8"))
    amp_dtype = torch.bfloat16 if exp["train"].get("amp_dtype", "bf16") == "bf16" else torch.float16
    inp = exp.get("input", {}) or {}
    bs = a.batch_size or int(exp.get("eval", {}).get("batch_size", 256))
    te = load_test_df(cfg)
    t0 = time.time()
    pat = "model_fold?.pt" if a.which == "best" else "model_fold?_last.pt"
    ckpts = sorted(exp_dir.glob(pat))
    assert len(ckpts) == 5, f"{exp_dir} の {pat} が 5 個ありません: {ckpts}"
    ck0 = torch.load(ckpts[0], map_location="cpu", weights_only=False)
    fill = np.array(ck0["mean"]) * 255  # 欠けたタイルは train 全体の平均色（学習時と同じ）
    source = TileSource(np.asarray(load_images(cfg, "test", mmap=False)), te, margin=int(inp.get("margin", 0)),
                        fill_rgb=fill)
    out = te[["path"]].copy()
    idx = np.arange(len(te))
    for cp in ckpts:
        ck = torch.load(cp, map_location="cuda", weights_only=False)
        model = build_from_cfg(ck["model"]).cuda()
        model.load_state_dict(ck["state_dict"])
        model = model.to(memory_format=torch.channels_last)
        mean = torch.tensor(ck["mean"], device="cuda").view(1, 3, 1, 1)
        std = torch.tensor(ck["std"], device="cuda").view(1, 3, 1, 1)
        logit = predict_logits(model, lambda s, e: source(idx[s:e]), len(te), bs, mean, std, amp_dtype,
                               tta=a.tta, size=inp.get("resize")).cpu().numpy()
        k = ck["fold"]
        out[f"logit{k}"] = logit
        out[f"fold{k}"] = 1 / (1 + np.exp(-logit))
        print(f"[infer] {cp.name} (epoch {ck['epoch']}) 完了 {time.time() - t0:.1f}s", flush=True)
    fcols = sorted(c for c in out.columns if c.startswith("fold"))
    lcols = sorted(c for c in out.columns if c.startswith("logit"))
    out["prob"] = out[fcols].mean(axis=1)
    out["prob_logitmean"] = 1 / (1 + np.exp(-out[lcols].mean(axis=1)))
    name = "test_pred" + ("_last" if a.which == "last" else "") + ("_tta" if a.tta else "") + ".csv"
    out.to_csv(exp_dir / name, index=False)
    rep = test_report(exp_dir, out, te, a.which)
    (exp_dir / name.replace(".csv", "_report.json")).write_text(json.dumps(rep, ensure_ascii=False, indent=2),
                                                                encoding="utf-8")
    print("[infer] " + json.dumps(rep, ensure_ascii=False))
    print(f"[infer] → {exp_dir / name}  ({time.time() - t0:.1f}s)")


if __name__ == "__main__":
    main()
