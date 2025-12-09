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
... (existing content) ...
```
※ 推論完了後、自動的に評価スクリプト (`evaluate.py`) が実行され、結果は `BIGRec/results/...` に保存されます。

### 6. BIGRecの推論 (vLLMによる高速化版)

vLLMを使用して推論を高速に実行します。

```bash
# 引数: <dataset> <gpu_id> <base_model> <seed> <sample> <skip_inference> <target_split> <batch_size> <limit> <prompt_file> <use_embedding_model>
# デフォルト: movie 0 "Qwen/Qwen2-0.5B" 0 1024 false "test_5000.json" 16 -1 "" false

# 例: 基本的な実行（デフォルト設定）
./cmd/run_bigrec_inference_vllm.sh movie 0 "Qwen/Qwen2-0.5B"

# 例: train.json の最初の100件のみを実行（テスト用）
# 第9引数で limit を指定します
./cmd/run_bigrec_inference_vllm.sh movie 0 "Qwen/Qwen2-0.5B" 0 1024 false "train.json" 16 100 "" false

# 例: train.json 全件を実行（本番用）
./cmd/run_bigrec_inference_vllm.sh movie 0 "Qwen/Qwen2-0.5B" 0 1024 false "train.json" 16 -1 "" false

# 例: カスタムプロンプトを使用して推論する場合
# (第10引数にプロンプトファイルのパスを指定)
./cmd/run_bigrec_inference_vllm.sh movie 0 "Qwen/Qwen2-0.5B" 0 1024 false "test_5000.json" 16 -1 "my_templates/inference_prompt.txt"

# 例: E5モデル (multilingual-e5-large) をアイテム埋め込みと予測評価に使用する場合
# (第11引数をtrueに指定。プロンプトファイルがない場合は空文字 "" を指定)
./cmd/run_bigrec_inference_vllm.sh movie 0 "Qwen/Qwen2-0.5B" 0 1024 false "test_5000.json" 16 -1 "" true
```

### 6. データ転送

BIGRecの推論結果（埋め込み表現やランキング情報）をDLLM2Recが利用できる形式で配置します。

```bash
# 引数: <dataset> <gpu_id> <base_model> <seed> <sample>
# デフォルト: movie 0 "Qwen/Qwen2-0.5B" 0 1024

# 例: Qwenモデルの結果を転送
./cmd/transfer_data.sh movie 0 "Qwen/Qwen2-0.5B"
```

### 7. DLLM2Recの学習

BIGRecの知識を蒸留してSASRecなどの学生モデルを学習させます。

```bash
# 引数: <dataset> <model_name> <gpu_id> <ed_weight> <lam> <bigrec_base_model> <bigrec_seed> <bigrec_sample>
# デフォルト: game SASRec 0 0.3 0.7 "" 0 1024

# 例: Gameデータセット、SASRec、GPU 0で学習（デフォルトの蒸留重み）
#     (事前に transfer_data.sh で配置したデータを使用)
./cmd/run_dllm2rec_train.sh game SASRec 0

# 例: 蒸留の重みを変更して学習 (ed_weight=0.5, lam=0.5)
./cmd/run_dllm2rec_train.sh game SASRec 0 0.5 0.5

# 例: BIGRecの結果ディレクトリを直接参照する場合 (推奨)
#     (転送ステップ不要。参照するモデル名、Seed、Sample数を指定)
./cmd/run_dllm2rec_train.sh game SASRec 0 0.3 0.7 "Qwen/Qwen2-0.5B" 0 1024
```
※ DLLM2Recのデフォルトデータセットは `game` になっていますが、前処理したデータセットに合わせて変更してください。

### 8. SASRecベースラインの実行

蒸留を行わず、純粋なSASRecモデルを学習させてベースライン性能を測定します。

```bash
# 引数: <dataset> <gpu_id> <epoch>
# デフォルト: game 0 200

# 例: Gameデータセットでベースラインを実行
./cmd/run_sasrec_baseline.sh game 0 200
```

### 9. パイプライン実行（YAML設定ベース）

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
