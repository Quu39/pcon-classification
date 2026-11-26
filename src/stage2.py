"""2 段目モデル: 1 段目の確率を元写真の盤面（14×18）に並べ、近傍の確率から特徴量を作って再分類する。

    python -m src.stage2 --stage1 exp001b exp005 --name exp007

リーク防止（CLAUDE.md 2.5）:
- 学習特徴量は 1 段目の OOF（(b) 最終 epoch: oof_last.csv）だけから作る。test は test_pred_last.csv（5 fold 平均）。
- 2 段目も同じ folds.csv で CV（fold k を除いて学習 → fold k を予測）。同じ写真のタイルは必ず同じ fold。
- test 予測は 5 つの fold モデルの平均。
欠けたタイル・盤外は NaN（HistGradientBoosting はそのまま扱う。ロジスティック回帰は 0 で埋める）。
重複タイル（test 7074-105）は同じ位置の平均を盤面に置き、test.txt の 2 行とも同じ特徴量になる。
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import warnings

import numpy as np
import pandas as pd
import yaml
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.data import N_COLS, N_ROWS, PROJECT_ROOT, load_config, load_folds, load_test_df, load_train_df
from src.metrics import evaluate_oof, neighbor_ratio

NB8 = [(-1, 0, "u"), (1, 0, "d"), (0, -1, "l"), (0, 1, "r"), (-1, -1, "ul"), (-1, 1, "ur"), (1, -1, "dl"), (1, 1, "dr")]


def to_grid(df: pd.DataFrame, prob: np.ndarray) -> tuple[np.ndarray, dict]:
    pids = sorted(df["photo_id"].unique())
    pmap = {p: i for i, p in enumerate(pids)}
    g = np.full((len(pids), N_ROWS, N_COLS), np.nan)
    s = np.zeros_like(g)
    c = np.zeros_like(g)
    pi, r, cc = df["photo_id"].map(pmap).to_numpy(), df["row"].to_numpy(), df["col"].to_numpy()
    np.add.at(s, (pi, r, cc), prob)
    np.add.at(c, (pi, r, cc), 1)
    m = c > 0
    g[m] = s[m] / c[m]
    return g, pmap


def _nanstat(stack: np.ndarray, fn) -> np.ndarray:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return fn(stack, axis=0)


def features_for(df: pd.DataFrame, prob: np.ndarray, prefix: str) -> pd.DataFrame:
    """1 つの 1 段目モデルの確率から、各タイルの特徴量を作る。"""
    g, pmap = to_grid(df, prob)
    P = 2
    gp = np.pad(g, ((0, 0), (P, P), (P, P)), constant_values=np.nan)
    pi, r, c = df["photo_id"].map(pmap).to_numpy(), df["row"].to_numpy(), df["col"].to_numpy()

    def at(dr, dc):
        return gp[pi, r + P + dr, c + P + dc]

    f = {f"{prefix}self": g[pi, r, c]}
    for dr, dc, n in NB8:
        f[f"{prefix}{n}"] = at(dr, dc)
    nb8 = np.stack([at(dr, dc) for dr, dc, _ in NB8])
    nb24 = np.stack([at(dr, dc) for dr in range(-2, 3) for dc in range(-2, 3) if (dr, dc) != (0, 0)])
    f[f"{prefix}nb8_max"] = _nanstat(nb8, np.nanmax)
    f[f"{prefix}nb8_mean"] = _nanstat(nb8, np.nanmean)
    f[f"{prefix}nb24_max"] = _nanstat(nb24, np.nanmax)
    f[f"{prefix}nb24_mean"] = _nanstat(nb24, np.nanmean)
    # 同じ行（左右 1 マス以内を除く）・同じ列（上下 1 マス以内を除く）の最大と上位 2 個の平均
    rows = g[pi, r, :].copy()  # (N,18)
    cols_ = g[pi, :, c].copy()  # (N,14)
    ci = np.arange(N_COLS)[None, :]
    ri = np.arange(N_ROWS)[None, :]
    rows[np.abs(ci - c[:, None]) <= 1] = np.nan
    cols_[np.abs(ri - r[:, None]) <= 1] = np.nan
    f[f"{prefix}row_far_max"] = _nanstat(rows.T, np.nanmax)
    rs = -np.sort(-np.nan_to_num(rows, nan=-1), axis=1)
    f[f"{prefix}row_far_top2"] = rs[:, :2].mean(1)
    f[f"{prefix}col_far_max"] = _nanstat(cols_.T, np.nanmax)
    # 写真全体（Pコン密度の目安）
    flat = g.reshape(len(g), -1)
    f[f"{prefix}photo_sum"] = np.nansum(flat, 1)[pi]
    srt = -np.sort(-np.nan_to_num(flat, nan=-1), axis=1)
    for k in (5, 10):
        f[f"{prefix}photo_top{k}"] = srt[:, :k].mean(1)[pi]
    return pd.DataFrame(f, index=df.index)


def build_features(df: pd.DataFrame, probs: dict[str, np.ndarray], use_pos: bool) -> pd.DataFrame:
    X = pd.concat([features_for(df, p, f"{k}_") for k, p in probs.items()], axis=1)
    if use_pos:
        X["row"] = df["row"].to_numpy()
        X["col"] = df["col"].to_numpy()
        X["is_edge"] = df["is_edge"].astype(int).to_numpy()
    return X


def make_model(kind: str, seed: int):
    if kind == "lr":
        return make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=2000))
    return HistGradientBoostingClassifier(learning_rate=0.05, max_iter=200, max_leaf_nodes=8, max_depth=3,
                                          min_samples_leaf=40, l2_regularization=1.0, early_stopping=False,
                                          random_state=seed)


def prep(kind: str, X: pd.DataFrame) -> pd.DataFrame:
    return X.fillna(0.0) if kind == "lr" else X


def run_cv(kind: str, X: pd.DataFrame, y: np.ndarray, fold: np.ndarray, Xt: pd.DataFrame, seed: int,
           importance: bool = False):
    oof = np.zeros(len(y))
    test = np.zeros(len(Xt))
    imps = []
    for k in sorted(np.unique(fold)):
        tr, va = fold != k, fold == k
        m = make_model(kind, seed)
        m.fit(prep(kind, X[tr]), y[tr])
        oof[va] = m.predict_proba(prep(kind, X[va]))[:, 1]
        test += m.predict_proba(prep(kind, Xt))[:, 1] / len(np.unique(fold))
        if importance:
            pi = permutation_importance(m, prep(kind, X[va]), y[va], scoring="average_precision", n_repeats=3,
                                        random_state=seed)
            imps.append(pi.importances_mean)
    imp = pd.Series(np.mean(imps, 0), index=X.columns).sort_values(ascending=False) if importance else None
    return oof, test, imp


def dist(x: np.ndarray) -> dict:
    q = np.quantile(x, [0.5, 0.9, 0.98, 0.99, 0.995])
    return dict(mean=round(float(x.mean()), 4), q50=round(q[0], 4), q90=round(q[1], 4), q98=round(q[2], 4),
                q99=round(q[3], 4), q995=round(q[4], 4), n_ge_0_5=int((x >= 0.5).sum()))


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage1", nargs="+", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    t0 = time.time()
    cfg = load_config()
    tr = load_train_df(cfg)
    te = load_test_df(cfg)
    folds = load_folds(cfg)
    y = tr["label"].to_numpy()
    fold = folds["fold"].to_numpy()
    out_dir = PROJECT_ROOT / "outputs" / a.name
    out_dir.mkdir(parents=True, exist_ok=True)

    p_oof, p_test, dists = {}, {}, {}
    for e in a.stage1:
        o = pd.read_csv(PROJECT_ROOT / "outputs" / e / "oof_last.csv")
        t = pd.read_csv(PROJECT_ROOT / "outputs" / e / "test_pred_last.csv")
        # oof_last.csv は fold 順に並んでいるので、train_label.csv の行順に path で揃える
        o = o.set_index("path").loc[tr["path"]].reset_index()
        assert o["path"].tolist() == tr["path"].tolist() and t["path"].tolist() == te["path"].tolist()
        assert (o["label"].to_numpy() == tr["label"].to_numpy()).all()
        p_oof[e], p_test[e] = o["prob"].to_numpy(), t["prob"].to_numpy()
        dists[e] = dict(oof=dist(p_oof[e]), test=dist(p_test[e]))

    results, preds, imps = {}, {}, {}
    for use_pos in (False, True):
        X = build_features(tr, p_oof, use_pos)
        Xt = build_features(te, p_test, use_pos)
        for kind in ("lr", "hgb"):
            key = f"{kind}_{'pos' if use_pos else 'nopos'}"
            oof, test, imp = run_cv(kind, X, y, fold, Xt, a.seed, importance=(kind == "hgb"))
            df_oof = folds[["path", "photo_id", "tile", "label", "fold"]].assign(prob=oof)
            m = evaluate_oof(df_oof)
            thr = m["threshold"]
            m["neighbor_ratio"] = neighbor_ratio(df_oof, oof >= thr)
            m["test_pos"] = int((test >= thr).sum())
            m["test_neighbor_ratio"] = neighbor_ratio(te, test >= thr)
            m["oof_dist"], m["test_dist"] = dist(oof), dist(test)
            results[key] = m
            preds[key] = (df_oof, test)
            if imp is not None:
                imps[key] = imp
            if kind == "lr":
                lr = make_model("lr", a.seed).fit(prep("lr", X), y)
                imps[key] = pd.Series(lr[-1].coef_[0], index=X.columns).sort_values(key=np.abs, ascending=False)
            print(f"[{key}] OOF F1 {m['oof_f1']:.4f} thr {thr:.2f} AP {m['ap']:.4f} "
                  f"fold {[round(v, 3) for v in m['fold_f1_at_pooled_thr']]} std {m['fold_f1_std']:.3f} "
                  f"OOF陽性 {m['pred_pos']} test陽性 {m['test_pos']} 隣接率 {m['neighbor_ratio']['ratio']:.3f}/"
                  f"{m['test_neighbor_ratio']['ratio']:.3f}  ({time.time() - t0:.0f}s)", flush=True)

    best_key = max(results, key=lambda k: results[k]["oof_f1"])
    df_oof, test = preds[best_key]
    df_oof.to_csv(out_dir / "oof_last.csv", index=False)
    df_oof.to_csv(out_dir / "oof.csv", index=False)
    pd.DataFrame({"path": te["path"], "prob": test}).to_csv(out_dir / "test_pred_last.csv", index=False)
    for k, (o, t) in preds.items():
        o.to_csv(out_dir / f"oof_{k}.csv", index=False)
        pd.DataFrame({"path": te["path"], "prob": t}).to_csv(out_dir / f"test_pred_{k}.csv", index=False)
    metrics = {"last": results[best_key], "best": results[best_key], "chosen": best_key, "all": results,
               "stage1": a.stage1, "stage1_dist": dists,
               "importance": {k: {i: round(float(v), 5) for i, v in s.head(15).items()} for k, s in imps.items()}}
    (out_dir / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    with open(out_dir / "config.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(dict(name=a.name, description=f"2 段目（{' + '.join(a.stage1)} の (b) OOF から近傍特徴）",
                            stage1=a.stage1, chosen=best_key), f, allow_unicode=True, sort_keys=False)
    print(f"[chosen] {best_key}")
    print("[stage1 分布] " + json.dumps(dists, ensure_ascii=False))
    for k, s in imps.items():
        print(f"[importance {k}] " + ", ".join(f"{i}={v:.4f}" for i, v in s.head(10).items()))
    print(f"[done] {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
