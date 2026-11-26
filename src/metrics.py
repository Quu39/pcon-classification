"""F1・しきい値探索・OOF 評価（CLAUDE.md 6 章）。予測は prob >= thr で 1。"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

THRS = np.round(np.arange(0.01, 1.00, 0.01), 2)


def f1_curve(y: np.ndarray, p: np.ndarray, thrs: np.ndarray = THRS) -> np.ndarray:
    y = np.asarray(y).astype(bool)
    pred = np.asarray(p)[None, :] >= thrs[:, None]
    tp = (pred & y).sum(1)
    fp = (pred & ~y).sum(1)
    fn = (~pred & y).sum(1)
    den = 2 * tp + fp + fn
    return np.where(den > 0, 2 * tp / np.maximum(den, 1), 0.0)


def best_threshold(y, p) -> tuple[float, float]:
    f = f1_curve(y, p)
    i = int(f.argmax())
    return float(THRS[i]), float(f[i])


def f1_at(y, p, thr: float) -> float:
    return float(f1_curve(y, p, np.array([thr]))[0])


def evaluate_oof(oof: pd.DataFrame, prob_col: str = "prob") -> dict:
    """oof: label, fold, prob を持つ DataFrame。pooled OOF で評価する。"""
    y, p = oof["label"].to_numpy(), oof[prob_col].to_numpy()
    thr, f1 = best_threshold(y, p)
    curve = f1_curve(y, p)
    around = {f"{t:.2f}": round(float(curve[np.argmin(np.abs(THRS - t))]), 4)
              for t in (thr - 0.05, thr - 0.02, thr, thr + 0.02, thr + 0.05) if 0.005 < t < 0.995}
    folds = sorted(oof["fold"].unique())
    fold_f1 = [f1_at(y[oof["fold"] == k], p[oof["fold"] == k], thr) for k in folds]
    fold_best = [best_threshold(y[oof["fold"] == k], p[oof["fold"] == k]) for k in folds]
    n_pred = int((p >= thr).sum())
    return dict(
        oof_f1=round(f1, 4), threshold=thr, f1_around=around,
        fold_f1_at_pooled_thr=[round(v, 4) for v in fold_f1],
        fold_f1_mean=round(float(np.mean(fold_f1)), 4), fold_f1_std=round(float(np.std(fold_f1)), 4),
        fold_best=[dict(thr=t, f1=round(v, 4)) for t, v in fold_best],
        pred_pos=n_pred, pred_pos_rate=round(n_pred / len(p), 4), true_pos_rate=round(float(y.mean()), 4),
        ap=round(float(average_precision_score(y, p)), 4), auc=round(float(roc_auc_score(y, p)), 4),
    )


def neighbor_ratio(df: pd.DataFrame, pred: np.ndarray) -> dict:
    """予測陽性のうち、同じ写真で上下左右に予測陽性のタイルがある割合（18列行優先）。
    重複タイル（同じ写真ID・タイル番号）は 1 つにまとめる（どちらかが陽性なら陽性）。"""
    d = df[["photo_id", "tile"]].assign(pred=np.asarray(pred).astype(bool))
    d = d.groupby(["photo_id", "tile"], as_index=False)["pred"].max()
    d = d[d["pred"]]
    pos = set(zip(d["photo_id"].tolist(), d["tile"].tolist()))
    has = 0
    for p, t in pos:
        c = (t - 1) % 18
        nb = [t - 18, t + 18] + ([t - 1] if c > 0 else []) + ([t + 1] if c < 17 else [])
        has += any((p, n) in pos for n in nb)
    return dict(ratio=round(has / max(1, len(pos)), 4), n_pos_tiles=len(pos), with_neighbor=has)
