"""1 段目モデルの OOF / test 予測の平均アンサンブル。

    python -m src.ensemble --exps exp001b exp003 --name exp006

(a) best epoch / (b) 最終 epoch × 確率平均 / logit 平均 の 4 通りで OOF F1 などを計算し、
- 方式: (b) の OOF F1 が高い方（確率平均 / logit 平均）
- (a)/(b): 差が 0.01 以内なら (b)、そうでなければ高い方
を採用する。test も同じ方式で平均する（確率平均 = 全モデル・全 fold の確率の平均、logit 平均 = 全 logit の平均の sigmoid）。
outputs/{name}/ に oof.csv, oof_last.csv, test_pred.csv, test_pred_last.csv（prob 列 = 採用方式）, metrics.json を保存。
"""
from __future__ import annotations

import argparse
import json
import sys

import numpy as np
import pandas as pd
import yaml

from src.data import PROJECT_ROOT, load_config, load_test_df
from src.metrics import evaluate_oof, neighbor_ratio

EPS = 1e-6


def logit(p):
    p = np.clip(np.asarray(p, dtype=np.float64), EPS, 1 - EPS)
    return np.log(p / (1 - p))


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--exps", nargs="+", required=True)
    ap.add_argument("--name", required=True)
    a = ap.parse_args()
    cfg = load_config()
    te = load_test_df(cfg)
    out_dir = PROJECT_ROOT / "outputs" / a.name
    out_dir.mkdir(parents=True, exist_ok=True)
    dirs = [PROJECT_ROOT / "outputs" / e for e in a.exps]

    results, oofs, tests = {}, {}, {}
    for which, of, tf in (("best", "oof.csv", "test_pred.csv"), ("last", "oof_last.csv", "test_pred_last.csv")):
        o = [pd.read_csv(d / of) for d in dirs]
        t = [pd.read_csv(d / tf) for d in dirs]
        for x in o[1:]:
            assert (x["path"].to_numpy() == o[0]["path"].to_numpy()).all()
        for x in t:
            assert x["path"].tolist() == te["path"].tolist()
        base = o[0][["path", "photo_id", "tile", "label", "fold"]]
        P = np.stack([x["prob"].to_numpy() for x in o])
        TP = np.concatenate([x[[c for c in x.columns if c.startswith("fold")]].to_numpy().T for x in t])
        TL = np.concatenate([x[[c for c in x.columns if c.startswith("logit")]].to_numpy().T for x in t])
        ens = {"prob": (P.mean(0), TP.mean(0)), "logit": (sigmoid(logit(P).mean(0)), sigmoid(TL.mean(0)))}
        for meth, (po, pt) in ens.items():
            oof = base.assign(prob=po)
            m = evaluate_oof(oof)
            thr = m["threshold"]
            m["neighbor_ratio"] = neighbor_ratio(oof, po >= thr)
            m["test_pos"] = int((pt >= thr).sum())
            m["test_neighbor_ratio"] = neighbor_ratio(te, pt >= thr)
            m["pos_by_threshold"] = {f"{s:.2f}": dict(oof=int((po >= s).sum()), test=int((pt >= s).sum()))
                                     for s in sorted({round(v, 2) for v in (0.5, 0.7, thr - 0.05, thr, thr + 0.05)
                                                      if 0 < v < 1})}
            results[f"{which}_{meth}"] = m
            oofs[(which, meth)] = oof
            tests[(which, meth)] = pt

    meth = max(("prob", "logit"), key=lambda k: results[f"last_{k}"]["oof_f1"])
    fa, fb = results[f"best_{meth}"]["oof_f1"], results[f"last_{meth}"]["oof_f1"]
    which = "last" if abs(fa - fb) <= 0.01 or fb > fa else "best"
    chosen = dict(method=meth, which=which, oof_f1=results[f"{which}_{meth}"]["oof_f1"],
                  threshold=results[f"{which}_{meth}"]["threshold"],
                  rule="方式は (b) の OOF F1 が高い方。(a)/(b) は差 0.01 以内なら (b)")
    for w, of, tf in (("best", "oof.csv", "test_pred.csv"), ("last", "oof_last.csv", "test_pred_last.csv")):
        oofs[(w, meth)].to_csv(out_dir / of, index=False)
        tdf = te[["path"]].assign(prob=tests[(w, meth)], prob_probmean=tests[(w, "prob")],
                                  prob_logitmean=tests[(w, "logit")])
        tdf.to_csv(out_dir / tf, index=False)
    metrics = {"best": results[f"best_{meth}"], "last": results[f"last_{meth}"], "chosen": chosen,
               "all": results, "members": a.exps}
    (out_dir / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    with open(out_dir / "config.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(dict(name=a.name, description=f"アンサンブル {' + '.join(a.exps)}（{meth} 平均）",
                            members=a.exps, method=meth, which=which), f, allow_unicode=True, sort_keys=False)

    print(f"| {a.name} | 評価 | 方式 | OOF F1 | thr | F1 thr±0.05 | fold別F1 (std) | OOF陽性 | test陽性 | 隣接率 OOF / test | AP |")
    for k, m in results.items():
        w, mt = k.split("_")
        ar = m["f1_around"]
        thr = m["threshold"]
        print(f"| {'(a) best' if w == 'best' else '(b) last'} | {mt} | {m['oof_f1']:.4f} | {thr:.2f} | "
              f"{ar.get(f'{thr - 0.05:.2f}', '-')} / {ar.get(f'{thr + 0.05:.2f}', '-')} | "
              f"{' / '.join(f'{v:.3f}' for v in m['fold_f1_at_pooled_thr'])} ({m['fold_f1_std']:.3f}) | "
              f"{m['pred_pos']} | {m['test_pos']} | {m['neighbor_ratio']['ratio']:.3f} / "
              f"{m['test_neighbor_ratio']['ratio']:.3f} | {m['ap']:.3f} |")
    print("[chosen] " + json.dumps(chosen, ensure_ascii=False))
    for k, m in results.items():
        print(f"[{k}] 陽性数 OOF/test: " + ", ".join(f"{t}: {v['oof']}/{v['test']}" for t, v in m["pos_by_threshold"].items()))


if __name__ == "__main__":
    main()
