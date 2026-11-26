"""experiments.md に 1 行追記: python -m src.log_exp exp005_s1 "変更点" "判断" """
import json
import sys

from src.data import PROJECT_ROOT

name, change, verdict = sys.argv[1], sys.argv[2], sys.argv[3]
m = json.loads((PROJECT_ROOT / "outputs" / name / "metrics.json").read_text(encoding="utf-8"))
b, a = m["last"], m["best"]
line = (f"| {name} | {change} | (b) {b['oof_f1']:.4f} / (a) {a['oof_f1']:.4f} | {b['threshold']:.2f} | "
        f"(b) {' / '.join(f'{v:.3f}' for v in b['fold_f1_at_pooled_thr'])}（std {b['fold_f1_std']:.3f}） | "
        f"{b['pred_pos_rate'] * 100:.2f}%（{b['pred_pos']}） | {m['train_minutes']}分 | {verdict} |\n")
with open(PROJECT_ROOT / "experiments.md", "a", encoding="utf-8") as f:
    f.write(line)
print(line.strip())
