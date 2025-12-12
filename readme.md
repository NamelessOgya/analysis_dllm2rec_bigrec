# BIGRec / DLLM2Rec 再現実験

このリポジトリは、BIGRecおよびDLLM2Recの再現実験を行うためのコードとスクリプトを含んでいます。
実行はDockerコンテナ内で行うことを想定しています。

## ディレクトリ構成

- `BIGRec/`: BIGRecのソースコード
- `DLLM2Rec/`: DLLM2Recのソースコード
- `cmd/`: 実行用スクリプトおよびDockerfile
- `test/`: テストコード

## 実行手順

### 1. コンテナの準備

まず、Dockerイメージをビルドし、コンテナを起動します。

```bash
# イメージのビルド
./cmd/build_container.sh

# コンテナの起動（デタッチドモード）
./cmd/start_container.sh
```

コンテナが起動したら、以下のコマンドでコンテナ内のシェルに入ります。

```bash
docker exec -it dllm2rec_bigrec_container /bin/bash
```

**以降の手順は、コンテナ内で実行してください。**

### 2. Hugging Faceへのログイン

LlamaなどのGated Modelを使用する場合は、Hugging Faceへのログインが必要です。

```bash
# トークンを引数で渡す場合
./cmd/login_huggingface.sh <YOUR_HF_TOKEN>

# 対話的にログインする場合
./cmd/login_huggingface.sh
```

### 1. データセットの準備 (game_bigrec)

BIGRecとDLLM2Recの両方で完全に動作する、高品質なゲームデータセット(`game_bigrec`)を構築します。
本プロジェクトでは、論文の実験設定を完全に再現するため、**Amazon V2 (5-core)** データセットを使用します。

```bash
# 1. Amazon V2データ（5-core）のダウンロード
# (cmd/download_data_v2.sh を使用)
./cmd/download_data_v2.sh

# 2. データの前処理
# - タイトルフィルタリング (len > 1) を適用
# - BIGRec用JSONとDLLM2Rec用DataFrameを同時生成
# - 論文の統計 (Items: 17,408, Seqs: 149,796) を再現
docker exec dllm2rec_bigrec_container ./cmd/run_preprocess_game_bigrec.sh

# 3. データ整合性の検証（推奨）
# train_data.df と train.json のID順序が一致しているか検証します
docker exec dllm2rec_bigrec_container python3 verify_data_alignment.py
```

### 3. データ前処理

BIGRec用のデータ前処理を行います。デフォルトは `movie` データセットです。

```bash
# Movieデータセット（MovieLens 10M）を使用する場合
./cmd/run_preprocess_movie.sh
```

# Gameデータセット
## 1. BIGRec準拠 (推奨)
BIGRecの論文/実装本来のロジック (`process.ipynb`) に完全準拠したデータセットです。IDの整合性が取れており、**再現実験にはこちらを使用することを強く推奨します**。
```bash
./cmd/run_preprocess_game_bigrec.sh
```

## 2. DLLM2Rec準拠 (非推奨)
DLLM2Recのリポジトリに含まれていたデータセット (`date/game`) です。
今回の実験で使用した `Video_Games_5.json` と比較して**データ規模が大きく、使用している元データが異なる可能性が高い**ため、今回の実験では参照用として扱います。
```bash
./cmd/run_preprocess_game.sh
```

### データセットの差異について

| 項目 | BIGRec準拠 (`game_bigrec`) | DLLM2Rec準拠 (`game`) |
| :--- | :--- | :--- |
| **Max Item ID** | ~10,673 | ~17,408 |
| **Train Rows** | ~61,144 | ~119,836 |
| **生成ロジック** | BIGRec `process.ipynb` 準拠 | 不明 (より緩いフィルタリング?) |
| **DLLM2Recファイル** | 自動生成 (`game_bigrec/process.py`) | リポジトリ同梱 |

※ `game_bigrec` では、BIGRec用のJSONファイル生成時に、DLLM2Rec学習用の `train_data.df` 等も整合性を保った状態で自動生成されるように拡張しています。

### 4. BIGRecの学習

BIGRecモデル（LLM）のFine-tuningを行います。

```bash
# 引数: <dataset> <gpu_id> <seed> <sample> <batch_size> <micro_batch_size> <base_model> <num_epochs> <prompt_file>
# デフォルト: movie 0 0 1024 128 4 "Qwen/Qwen2-0.5B" 50 ""


# 例: Gameデータセット (BIGRec準拠) で学習
./cmd/run_bigrec_train.sh game_bigrec 0 0 1024 128 128 "Qwen/Qwen2-0.5B" 50

# 例: A100などで高速に学習する場合
./cmd/run_bigrec_train.sh movie 0 0 1024 128 128 "Qwen/Qwen2-0.5B" 50

# 例: 並列実行（GPU 0でQwen, GPU 1でLlama）
# 出力ディレクトリはモデル名を含むため競合しません
./cmd/run_bigrec_train.sh movie 0 0 1024 128 128 "Qwen/Qwen2-0.5B" 50 &
./cmd/run_bigrec_train.sh movie 1 0 1024 128 128 "meta-llama/Llama-2-7b-hf" 50 &

# 例: カスタムプロンプトを使用する場合
# (第9引数にプロンプトファイルのパスを指定)
./cmd/run_bigrec_train.sh movie 0 0 1024 128 128 "Qwen/Qwen2-0.5B" 50 "my_templates/custom_prompt.txt"

```

### 5. BIGRecの推論

学習したモデルを使って推論（データの生成）を行います。生成されたデータは `BIGRec/results/...` に保存されます。

```bash
# 引数: <dataset> <gpu_id> <base_model> <seed> <sample> <skip_inference> <target_split> <batch_size> <debug_limit> <use_embedding_model> <use_popularity> <popularity_gamma> <checkpoint_epoch>
# デフォルト: movie 0 "Qwen/Qwen2-0.5B" 0 1024 false "valid_test" 16 -1 false false 0.0 "best"

# 例: 基本的な実行（デフォルト設定、Best Model使用）
./cmd/run_bigrec_inference.sh movie 0 "Qwen/Qwen2-0.5B"

# 例: 特定のEpoch (例: 10) のチェックポイントを使用して推論
# (第13引数にEpoch数を指定)
./cmd/run_bigrec_inference.sh movie 0 "Qwen/Qwen2-0.5B" 0 1024 false "valid_test" 16 -1 false false 0.0 10

# 例: train.json を対象に実行する場合 (target_split="train.json")
# ./cmd/run_bigrec_inference.sh movie 0 "Qwen/Qwen2-0.5B" 0 1024 false "train.json"
```
※ 推論完了後、自動的に評価スクリプト (`evaluate.py`) が実行され、結果は `BIGRec/results/...` に保存されます。

### 6. BIGRecの推論 (vLLMによる高速化版)

vLLMを使用して推論を高速に実行します。
**引数がフラグ形式 (`--option value`) に刷新され、使いやすくなりました。**

```bash
# 基本使用法
./cmd/run_bigrec_inference_vllm.sh --dataset movie --gpu 0

# 主なオプション:
#   --model <name>           ベースモデル (デフォルト: Qwen/Qwen2-0.5B)
#   --checkpoint <epoch>     チェックポイント ("best" または Epoch数)
#   --test_data <file>       対象データ ("test_5000.json", "all" など)
#   --limit <int>            処理件数制限 (-1: 全件)
#   --correction <type>      補正モード ("none", "popularity", "ci")
#   --resource <path>        補正用リソース (人気度ファイル または CIスコアディレクトリ)
```

### 実行例 (Copy & Paste用)

コンソールにそのまま貼り付けて実行できる、最も一般的な設定の例です。

**Gameデータセット (Popularity調整あり):**
```bash
./cmd/run_bigrec_inference_vllm.sh \
    --dataset game_bigrec \
    --model "Qwen/Qwen2-0.5B" \
    --checkpoint "best" \
    --test_data "test_5000.json" \
    --correction popularity \
    --gpu 0
```

**Gameデータセット (CI注入あり - SASRecスコア指定):**
```bash
./cmd/run_bigrec_inference_vllm.sh \
    --dataset game_bigrec \
    --model "Qwen/Qwen2-0.5B" \
    --correction ci \
    --resource "DLLM2Rec/results/game_bigrec/sasrec_no_distillation/2024/alpha_1.0/" \
    --gpu 0
```

**使用例:**

**1. 人気度バイアス調整 (Popularity Adjustment)**
デフォルトのgamma (0.0) を指定してValidationセットで自動チューニングする場合:
```bash
./cmd/run_bigrec_inference_vllm.sh \
    --dataset movie \
    --correction popularity
```

**2. SASRec協調情報注入 (CI Injection)**
別途学習したSASRecのスコア (val.pt/test.pt) を注入する場合:
```bash
./cmd/run_bigrec_inference_vllm.sh \
    --dataset movie \
    --correction ci \
    --resource "DLLM2Rec/results/game/sasrec_no_distillation/2024/alpha_1.0/"
```
※ `resource` には `val.pt`, `test.pt` が保存されているディレクトリを指定します。Validationセットを用いて最適なgammaが自動探索されます。

---

### 9. SASRecベースラインの実行 (＆ CI用スコア生成)

SASRecモデルを学習させます。CI注入用のスコア生成にも使用します。

```bash
# 引数: <dataset> <gpu_id> <epoch> <seed> <alpha>
# デフォルト: game 0 200 2024 0

# 例: Gameデータセットで学習 (alpha=0, 通常のSASRec)
./cmd/run_sasrec_baseline.sh game 0 200 2024

# 例: DROS用学習 (alpha=1.0) でスコアを生成
# 結果は DLLM2Rec/results/.../alpha_1.0/ に保存されます
./cmd/run_sasrec_baseline.sh game 0 200 2024 1.0
```

### 10. パイプライン実行（YAML設定ベース）


設定ファイル（YAML）に基づいて、BIGRec学習からDLLM2Rec学習までを一括実行します。
※ 前処理は事前に済ませておく必要があります。

1.  **設定ファイルの編集**: `pipeline_config.yaml` を編集してパラメータを指定します。
    ```yaml
    common:
      dataset: "game"
      gpu_id: 0
    ...
    ```

2.  **パイプラインの実行**:
    ```bash
    python cmd/run_pipeline.py
    ```

### テストの実行

前処理ロジックなどの簡易テストを実行します。

```bash
python test/test_preprocess.py
```
