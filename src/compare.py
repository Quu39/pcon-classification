"""実験の比較表: python -m src.compare exp001b exp003 exp004 exp002

(a) best epoch / (b) 最終 epoch の OOF F1、しきい値、±0.05、fold 別 F1、予測陽性、隣接率（OOF / test）、
しきい値ごとの test 陽性数（確率平均 / logit 平均）を並べる。
"""
from __future__ import annotations

import json
import sys

from src.data import PROJECT_ROOT


def load(exp: str) -> dict:
    d = PROJECT_ROOT / "outputs" / exp
    m = json.loads((d / "metrics.json").read_text(encoding="utf-8"))
    rep = {}
    for which, f in (("best", "test_pred_report.json"), ("last", "test_pred_last_report.json")):
        p = d / f
        if p.exists():
            rep[which] = json.loads(p.read_text(encoding="utf-8"))
    return dict(m=m, rep=rep)


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    exps = sys.argv[1:]
    print("| exp | 評価 | OOF F1 | thr | F1 @thr-0.05 / +0.05 | fold別F1 (std) | OOF陽性 | AP | "
          "隣接率 OOF | test陽性 確率平均 / logit平均 | 隣接率 test(確率平均) | 学習 |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for e in exps:
        x = load(e)
        m = x["m"]
        for which in ("best", "last"):
            if which not in m:
                continue
            k = m[which]
            thr = k["threshold"]
            ar = k["f1_around"]
            lo, hi = ar.get(f"{thr - 0.05:.2f}", "-"), ar.get(f"{thr + 0.05:.2f}", "-")
            r = x["rep"].get(which, {})
            pb = r.get("pos_by_threshold", {}).get(f"{thr:.2f}", {})
            nb = r.get("neighbor_ratio", {})
            print(f"| {e} | {'(a) best' if which == 'best' else '(b) last'} | {k['oof_f1']:.4f} | {thr:.2f} | "
                  f"{lo} / {hi} | {' / '.join(f'{v:.3f}' for v in k['fold_f1_at_pooled_thr'])} "
                  f"({k['fold_f1_std']:.3f}) | {k['pred_pos']} | {k['ap']:.3f} | {k['neighbor_ratio']['ratio']:.3f} | "
                  f"{pb.get('test_probmean', '-')} / {pb.get('test_logitmean', '-')} | "
                  f"{nb.get('test_probmean', {}).get('ratio', '-')} | {m.get('train_minutes', '-')}分 |")
    print()
    for e in exps:
        x = load(e)
        for which, r in x["rep"].items():
            print(f"[{e} {which}] しきい値ごとの陽性数（OOF / test 確率平均 / test logit平均）: " +
                  ", ".join(f"{t}: {v['oof']}/{v['test_probmean']}/{v['test_logitmean']}"
                            for t, v in r.get("pos_by_threshold", {}).items()))


if __name__ == "__main__":
    main()
