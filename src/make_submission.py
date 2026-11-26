"""提出ファイルの作成と形式チェック（CLAUDE.md 8 章）。submissions/README.md に記録を追記する。

    python -m src.make_submission --exp outputs/exp001 --out submissions/fallback.csv
    （しきい値は既定で outputs/{exp}/metrics.json の OOF 最適しきい値）
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from src.data import PROJECT_ROOT, load_config, resolve_path

README_HEADER = """# 提出ファイルの記録

| ファイル名 | 元の実験 | OOF F1 | しきい値 | 予測陽性数 (率) | 作成時刻 | public スコア | メモ |
|---|---|---|---|---|---|---|---|
"""


def check_submission(path: Path, test_paths: list[str], sample_paths: list[str]) -> int:
    """CLAUDE.md 8 章の要件を assert する。予測陽性数を返す。"""
    raw = path.read_text(encoding="utf-8").splitlines()
    assert raw[0].startswith("W/"), f"ヘッダーがあるようです: {raw[0]!r}"
    sub = pd.read_csv(path, header=None, dtype=str)
    assert sub.shape[1] == 2, f"列数が 2 ではありません: {sub.shape}"
    assert len(sub) == len(test_paths) == 8059, f"行数が違います: {len(sub)} / test.txt {len(test_paths)}"
    assert sub[0].tolist() == test_paths, "パスの順序が test.txt と一致しません"
    assert sub[0].tolist() == sample_paths, "パスの順序が sample_submission.csv と一致しません"
    assert not sub[0].duplicated().any(), "パスが重複しています"
    assert sub.notna().all().all(), "欠損があります"
    assert set(sub[1].unique()) <= {"0", "1"}, f"値が 0/1 以外: {set(sub[1].unique())}"
    size = path.stat().st_size
    assert size <= 10 * 1024 * 1024, f"ファイルサイズが 10MB を超えています: {size}"
    n_pos = int((sub[1] == "1").sum())
    print(f"[check] OK  行数 {len(sub)}  サイズ {size / 1024:.0f}KB  予測陽性数 {n_pos} ({n_pos / len(sub):.2%})"
          f"  目安 150 前後")
    return n_pos


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--pred", default="test_pred.csv")
    ap.add_argument("--which", choices=["best", "last"], default="best", help="metrics.json のどちらのしきい値を使うか")
    ap.add_argument("--col", default="prob", help="prob（確率平均）/ prob_logitmean（logit 平均）")
    ap.add_argument("--threshold", type=float, default=None)
    ap.add_argument("--note", default="")
    a = ap.parse_args()
    cfg = load_config()
    exp_dir = Path(a.exp) if Path(a.exp).is_absolute() else PROJECT_ROOT / a.exp
    m = json.loads((exp_dir / "metrics.json").read_text(encoding="utf-8"))
    m = m.get(a.which, m)  # exp001 は旧形式（best のみ）
    thr = a.threshold if a.threshold is not None else m["threshold"]

    test_paths = pd.read_csv(resolve_path(cfg, "test_list"), header=None)[0].tolist()
    sample_paths = pd.read_csv(resolve_path(cfg, "sample_submission"), header=None)[0].tolist()
    pred = pd.read_csv(exp_dir / a.pred)
    assert pred["path"].tolist() == test_paths, "test_pred の順序が test.txt と一致しません"
    assert pred[a.col].notna().all()
    label = (pred[a.col].to_numpy() >= thr).astype(np.int64)

    out = Path(a.out) if Path(a.out).is_absolute() else PROJECT_ROOT / a.out
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"path": test_paths, "label": label}).to_csv(out, header=False, index=False, lineterminator="\n")
    n_pos = check_submission(out, test_paths, sample_paths)

    readme = out.parent / "README.md"
    if not readme.exists():
        readme.write_text(README_HEADER, encoding="utf-8")
    row = (f"| {out.name} | {exp_dir.name} ({a.pred}:{a.col}, {a.which}) | {m['oof_f1']:.4f} | {thr:.2f} | {n_pos} ({n_pos / len(label):.2%}) | "
           f"{datetime.now():%Y-%m-%d %H:%M} |  | {a.note} |\n")
    with open(readme, "a", encoding="utf-8") as f:
        f.write(row)
    print(f"[submission] → {out}\n[README] {row.strip()}")


if __name__ == "__main__":
    main()
