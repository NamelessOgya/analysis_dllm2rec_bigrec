# Pipeline Specification

## 概要
複数の実験パターン（SASRec -> Active Learning Data -> BIGRec Train -> Inference -> DLLM2Rec）を効率的に実行するためのDVCベースのパイプラインを実装しました。
実験パラメータをCSVファイルで管理し、自動的にDVCパイプライン定義（`dvc.yaml`）を生成します。

## 特徴
1.  **DVCによる再実行制御**: 完了済みのステップはスキップされ、依存関係に基づいて必要なステップのみ実行されます。
2.  **最小限のディレクトリ構造**: `experiments/` ディレクトリ配下に、実験パラメータに基づいた階層構造で結果を保存します。不要な重複を排除しています。
3.  **マルチGPU対応**: `GPUID` カラムに基づいて、GPUごとに独立したDVCファイル (`dvc_gpu0.yaml`, `dvc_gpu1.yaml` 等) を生成します。

## 使用方法

### 1. パラメータファイルの準備
`cmd/pipeline_params.csv` (または任意のCSVファイル) を作成してください。
形式は以下の通りです。

| 変数名 | 詳細 | 例 |
| --- | --- | --- |
| dataset_name | データセット名 | game_bigrec |
| seed | シード値 | 2024 |
| alpha | SASRec学習時のalpha (DROS重み) | 1.0 |
| sampling_strategy | Active Learningの手法 | loss, random, clustering 等 |
| sample_num | サンプリング数 | 10000 |
| al_ratio | Active Learningの割合 | 1.0 |
| base_model_name | ベースモデル名 | Qwen/Qwen2-0.5B |
| templete | プロンプトテンプレートファイル (任意) | |
| ed_weight | DLLM2Rec蒸留パラメータ (Collaborative) | 0.3 |
| lambda | DLLM2Rec蒸留パラメータ (Ranking) | 0.7 |
| GPUID | 使用するGPU ID (0, 1, ...) | 0 |

**サンプル**: `cmd/pipeline_params_sample.txt` を参照。

### 2. パイプライン定義の生成
以下のコマンドを実行して、`dvc_gpu*.yaml` を生成します。

```bash
python cmd/generate_dvc_pipeline.py pipeline_params.csv
```

### 3. パイプラインの実行
DVCを使用してパイプラインを実行します。

**GPU 0用:**
```bash
dvc repro -f dvc_gpu0.yaml
```

**GPU 1用:**
```bash
dvc repro -f dvc_gpu1.yaml
```

### 4. 結果の集計
実験結果 (`metrics.json`) をまとめてCSVに出力するスクリプトも用意しました。

```bash
python cmd/aggregate_results.py
```

実行すると、`experiments/summary.csv` が生成されます。ディレクトリ構造からパラメータ（手法、シード、ハイパーパラメータ等）を自動的に抽出してカラムに追加します。

## ディレクトリ構造
出力ファイルは `experiments/` 配下に以下のように整理されます。

```text
experiments/{dataset_name}/
  ├── sasrec/
  │   └── seed_{seed}/
  │       └── alpha_{alpha}/
  │           ├── train.pt     (SASRecスコア)
  │           └── train_uids.pt
  │
  ├── active_learning/
  │   └── {strategy}_{sample_num}_{al_ratio}_seed_{seed}_alpha_{alpha}.json  (学習データ)
  │
  └── {base_model_safe}/
      ├── bigrec_train/
      │   └── {strategy}_{sample_num}_{al_ratio}_seed_{seed}_alpha_{alpha}/  (BIGRecモデル)
      │
      ├── bigrec_infer_train/
      │   └── {strategy}_{sample_num}_{al_ratio}_seed_{seed}_alpha_{alpha}/
      │       ├── train_epoch_best.json
      │       ├── train_epoch_best_rank.txt  (蒸留用ランク)
      │       └── train_epoch_best_score.txt (蒸留用スコア)
      │
      └── dllm2rec_final/
          └── {strategy}_{sample_num}_{al_ratio}_seed_{seed}_alpha_{alpha}/
              └── ed_{ed_weight}_lam_{lambda}/
                  ├── metrics.json   (最終結果)
                  └── best_model.pth
```

## 実装の詳細
以下のスクリプトを修正・作成しました。

- **`cmd/generate_dvc_pipeline.py`**: [NEW] CSVからDVC YAMLを生成するスクリプト。
- **`DLLM2Rec/main.py`**: `--output_dir` 引数を追加し、出力先を制御可能に変更。従来の自動生成ロジックは削除されました。
- **`cmd/run_sasrec_baseline.sh`**: `OUTPUT_DIR` 環境変数が**必須**になりました。
- **`cmd/create_active_learning_data.sh`**: `OUTPUT_JSON` 環境変数が**必須**になりました。
- **`cmd/run_bigrec_train.sh`**: `OUTPUT_DIR` 環境変数が**必須**になり、`TRAIN_DATA_FILE` の絶対パスをサポート。
- **`cmd/run_bigrec_inference_vllm.sh`**: `RESULT_DIR` 環境変数が**必須**になりました。 `LORA_WEIGHTS` もサポート。
- **`cmd/run_dllm2rec_train.sh`**: `OUTPUT_DIR` 環境変数が**必須**になり、入力パス(`RANKING_PATH`等)のオーバーライドをサポート。
