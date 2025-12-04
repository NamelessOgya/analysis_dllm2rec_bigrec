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

### 3. データ前処理

BIGRec用のデータ前処理を行います。デフォルトは `movie` データセットです。

```bash
# Movieデータセット（MovieLens 10M）を使用する場合
./cmd/run_preprocess_movie.sh

# Gameデータセット（Amazon Video Games）を使用する場合
./cmd/run_preprocess_game.sh
```

### 4. BIGRecの学習

BIGRecモデル（LLM）のFine-tuningを行います。

```bash
# 引数: <dataset> <gpu_id> <seed> <sample> <batch_size> <micro_batch_size> <base_model> <num_epochs>
# デフォルト: movie 0 0 1024 128 4 "Qwen/Qwen2-0.5B" 50

# 例: A100などで高速に学習する場合
./cmd/run_bigrec_train.sh movie 0 0 1024 128 128 "Qwen/Qwen2-0.5B" 50

# 例: 並列実行（GPU 0でQwen, GPU 1でLlama）
# 出力ディレクトリはモデル名を含むため競合しません
./cmd/run_bigrec_train.sh movie 0 0 1024 128 128 "Qwen/Qwen2-0.5B" 50 &
./cmd/run_bigrec_train.sh movie 1 0 1024 128 128 "meta-llama/Llama-2-7b-hf" 50 &
```

### 5. BIGRecの推論

学習したモデルを用いて推論を行い、推薦結果を生成します。

```bash
# 引数: <dataset> <gpu_id> <base_model> <seed> <sample> <skip_inference> <test_data> <batch_size>
# デフォルト: movie 0 "Qwen/Qwen2-0.5B" 0 1024 false "test_5000.json" 16

# 例: Qwenで推論と評価を実行（テストデータ: test_5000.json）
./cmd/run_bigrec_inference.sh movie 0 "Qwen/Qwen2-0.5B"

# 例: 学習データ(train.json)に対して推論を実行し、DLLM2Rec転送用データを生成（バッチサイズ32）
./cmd/run_bigrec_inference.sh movie 0 "Qwen/Qwen2-0.5B" 0 1024 false train.json 32

# 例: 推論をスキップして評価のみ実行（推論結果が既にある場合）
./cmd/run_bigrec_inference.sh movie 0 "Qwen/Qwen2-0.5B" 0 1024 true
```
※ 推論完了後、自動的に評価スクリプト (`evaluate.py`) が実行され、結果は `BIGRec/results/...` に保存されます。

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
# 引数: <dataset> <model_name> <gpu_id>
./cmd/run_dllm2rec_train.sh game SASRec 0
```
※ DLLM2Recのデフォルトデータセットは `game` になっていますが、前処理したデータセットに合わせて変更してください。

### テストの実行

前処理ロジックなどの簡易テストを実行します。

```bash
python test/test_preprocess.py
```
