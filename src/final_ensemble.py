"""最終アンサンブル: モデルの種類ごとに seed 平均 → 種類を等しい重みで平均（重みの最適化はしない）。

    python -m src.final_ensemble --name exp009 \
        --group exp001b exp001b_s1 exp001b_s2 exp001b_s3 \
        --group exp005 exp005_s1 exp005_s2 exp005_s3

(b) 最終 epoch の予測（oof_last.csv / test_pred_last.csv）を使う。終わっていない seed は飛ばす。
しきい値はこのアンサンブルの OOF 上で決める。
"""
from __future__ import annotations

import argparse
import json
import sys

import numpy as np
import pandas as pd
import yaml

from src.data import PROJECT_ROOT, load_config, load_test_df, load_train_df
from src.metrics import evaluate_oof, neighbor_ratio


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--group", nargs="+", action="append", required=True)
    a = ap.parse_args()
    cfg = load_config()
    tr, te = load_train_df(cfg), load_test_df(cfg)
    out_dir = PROJECT_ROOT / "outputs" / a.name
    out_dir.mkdir(parents=True, exist_ok=True)

    base = None
    g_oof, g_test, used, per = [], [], [], {}
    for grp in a.group:
        po, pt, mem = [], [], []
        for e in grp:
            d = PROJECT_ROOT / "outputs" / e
            if not ((d / "oof_last.csv").exists() and (d / "test_pred_last.csv").exists()):
                print(f"[skip] {e}（未完了）")
                continue
            o = pd.read_csv(d / "oof_last.csv").set_index("path").loc[tr["path"]].reset_index()
            t = pd.read_csv(d / "test_pred_last.csv")
            assert (o["label"].to_numpy() == tr["label"].to_numpy()).all()
            assert t["path"].tolist() == te["path"].tolist()
            base = o[["path", "photo_id", "tile", "label", "fold"]] if base is None else base
            po.append(o["prob"].to_numpy())
            pt.append(t["prob"].to_numpy())
            mem.append(e)
        assert mem, f"group {grp} に完了したモデルがありません"
        go, gt = np.mean(po, 0), np.mean(pt, 0)
        m = evaluate_oof(base.assign(prob=go))
        per[grp[0]] = dict(members=mem, oof_f1=m["oof_f1"], threshold=m["threshold"], ap=m["ap"],
                           single_f1=[evaluate_oof(base.assign(prob=p))["oof_f1"] for p in po])
        print(f"[group {grp[0]}] seed {len(mem)} 本 {mem}: seed 平均 OOF F1 {m['oof_f1']:.4f} (thr {m['threshold']:.2f}, "
              f"AP {m['ap']:.4f}) / 単体 {per[grp[0]]['single_f1']}")
        g_oof.append(go)
        g_test.append(gt)
        used.append(mem)

    po, pt = np.mean(g_oof, 0), np.mean(g_test, 0)
    oof = base.assign(prob=po)
    m = evaluate_oof(oof)
    thr = m["threshold"]
    m["neighbor_ratio"] = neighbor_ratio(oof, po >= thr)
    m["test_pos"] = int((pt >= thr).sum())
    m["test_pos_rate"] = round(float((pt >= thr).mean()), 4)
    m["test_neighbor_ratio"] = neighbor_ratio(te, pt >= thr)
    m["groups"] = per
    oof.to_csv(out_dir / "oof_last.csv", index=False)
    oof.to_csv(out_dir / "oof.csv", index=False)
    pd.DataFrame({"path": te["path"], "prob": pt}).to_csv(out_dir / "test_pred_last.csv", index=False)
    (out_dir / "metrics.json").write_text(json.dumps({"last": m, "best": m}, ensure_ascii=False, indent=2),
                                          encoding="utf-8")
    with open(out_dir / "config.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(dict(name=a.name, description="種類ごとに seed 平均 → 2 種類を等重みで平均（(b) 最終 epoch・TTA なし）",
                            groups=used), f, allow_unicode=True, sort_keys=False)
    print(f"[{a.name}] OOF F1 {m['oof_f1']:.4f} thr {thr:.2f} ±: {m['f1_around']}")
    print(f"  fold F1 {m['fold_f1_at_pooled_thr']} (std {m['fold_f1_std']}) AP {m['ap']}")
    print(f"  予測陽性 OOF {m['pred_pos']} ({m['pred_pos_rate']:.2%}) / test {m['test_pos']} ({m['test_pos_rate']:.2%})"
          f"  隣接率 OOF {m['neighbor_ratio']['ratio']} / test {m['test_neighbor_ratio']['ratio']}")


if __name__ == "__main__":
    main()
