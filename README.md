# Pコン跡分類（SDNET2018 壁画像）

256×256 の壁タイルにPコン跡があるかどうかを 2 値分類する（評価は陽性クラスの F1）。

最終提出は `submissions/final_candidate.csv`（exp009）。

- 構成: タイル単体 ResNet18 × 4 seed と、周辺文脈・中央プーリング ResNet18 × 4 seed の等重み平均
- OOF F1 0.8889、しきい値 0.72、public 0.851
- 事前学習済み重みは使っていない（すべて scratch）

解法の詳細は [docs/solution.md](docs/solution.md)、実験ログは [experiments.md](experiments.md) にある。

## 1. 環境構築

動作確認した環境: Windows 11、Python 3.12.14、PyTorch 2.10.0+cu128（CUDA 12.8）、NVIDIA GeForce RTX 5070（12GB）。

```bash
conda create -n pcon python=3.12
conda activate pcon
pip install torch==2.10.0 --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
```

RTX 50 系（Blackwell / sm_120）では、CUDA 12.8 以上のビルド（cu128）の PyTorch が必要。古いビルドでは `no kernel image is available` が出る。

## 2. データの配置

1. **配布物**（`train_label.csv`, `test.txt`, `sample_submission.csv`）を `data/` に置く（このリポジトリには含めていない）。

   ```
   data/train_label.csv
   data/test.txt
   data/sample_submission.csv
   ```

2. **SDNET2018** を任意の場所に展開し、`config.yaml` の `sdnet_root` をそのパスに書き換える。
   - 指定するのは、`W/CW` と `W/UW` を含むフォルダ。
   - `SDNET2018/SDNET2018/W` のように一段深い場合は自動で探す。

   ```yaml
   sdnet_root: "D:/data/SDNET2018"
   ```

   画像は配布物のパス（`W/CW/{写真ID}-{タイル番号}.jpg`）で SDNET2018 から読み込む。W/CW と W/UW を合わせた 18,138 枚が train + test になる。D・P は使わない。

## 3. 実行コマンド（最終提出 exp009 の再現）

```bash
# 1. 画像キャッシュ（outputs/cache/、約 3.6GB）と fold
python -m src.data
python -m src.data --make-folds --force   # outputs/folds.csv（同梱のものと同じになる。同梱版をそのまま使ってもよい）
python -m src.eda                         # 任意: 18 列 × 14 行の配置の検証と EDA

# 2. 1 段目モデルの学習（5 fold × 10 epoch）と test 推論（最終 epoch の重み）
for e in exp001b exp001b_s1 exp001b_s2 exp001b_s3 exp005 exp005_s1 exp005_s2 exp005_s3; do
  python -m src.train --exp configs/$e.yaml
  python -m src.infer --exp outputs/$e --which last
done

# 3. seed 平均 → 等重み平均（outputs/exp009/。しきい値は OOF 上で決まる）
python -m src.final_ensemble --name exp009 \
  --group exp001b exp001b_s1 exp001b_s2 exp001b_s3 \
  --group exp005 exp005_s1 exp005_s2 exp005_s3

# 4. 提出ファイルの作成と形式チェック
python -m src.make_submission --exp outputs/exp009 --which last --pred test_pred_last.csv \
  --out submissions/final_candidate.csv --note "exp009"
```

- 所要時間の目安（RTX 5070）: exp001b 系 4.7 分 / 本、exp005 系 11.0 分 / 本。合計約 63 分。
- 同梱の `outputs/exp009/` には、作成時の OOF 予測（`oof_last.csv`）、test 予測（`test_pred_last.csv`）、評価値（`metrics.json`）が入っている。

## 4. 再現性についての注意

- seed は各 config の `train.seed` で固定している（fold k では seed + k）。
- ただし GPU の演算（cuDNN の `benchmark=True`、bf16 autocast、atomic 加算など）は非決定的なので、再実行すると OOF F1・しきい値・予測陽性数が少しずれることがある。
- 陽性は 194 件と少なく、OOF F1 の差 0.01〜0.02 はこの範囲のぶれに入る（docs/solution.md 4 章）。

## 5. 構成

```
config.yaml            # sdnet_root、seed、パス、タイル配置（18 列 × 14 行）
configs/exp*.yaml      # 実験ごとの設定（exp001〜exp012、seed 違いを含む）
src/                   # コード（各ファイルの役割は docs/solution.md 8 章）
docs/solution.md       # 解法の説明
experiments.md         # 実験ログ（exp001〜exp014）
outputs/folds.csv      # 写真単位・陽性数を均等にした 5 fold（全実験で固定）
outputs/exp009/        # 最終アンサンブルの OOF / test 予測と評価値
submissions/final_candidate.csv   # 最終提出ファイル
```

含めていないもの: SDNET2018 の画像、画像キャッシュ（outputs/cache/）、学習済み重み（*.pt）、配布物の CSV。
