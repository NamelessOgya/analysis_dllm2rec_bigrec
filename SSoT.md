# Single Source of Truth (SSoT) - DLLM2Rec / BIGRec Project Information

本ドキュメントは、本プロジェクトにおけるDLLM2RecおよびBIGRecの連携、実行仕様、ディレクトリ構造に関する唯一の正解情報（Single Source of Truth）を定義します。
実装や実行手順は本ドキュメントの記述に準拠する必要があります。

## 1. 概要
本プロジェクトは、LLMを用いた推薦システム（BIGRec）と、その知識を従来の推薦モデル（SASRec等）に蒸留するシステム（DLLM2Rec）の再現・検証を目的としています。

## 2. 実行仕様 (DLLM2Rec)

### 2.1 学習スクリプト
*   **スクリプト**: `cmd/run_dllm2rec_train.sh`
*   **実行引数**:
    ```bash
    ./cmd/run_dllm2rec_train.sh <DATASET> <MODEL_NAME> <GPU_ID> <ED_WEIGHT> <LAM> <BIGREC_BASE_MODEL> <BIGREC_SEED> <BIGREC_SAMPLE>
    ```
    *   `DATASET`: データセット名 (例: `game`)
    *   `MODEL_NAME`: 生徒モデル名 (例: `SASRec`)
    *   `GPU_ID`: 使用するGPU ID
    *   `ED_WEIGHT`: Collaborative Embedding Distillationの重み (Default: 0.3)
    *   `LAM`: Ranking Distillationの重み (Default: 0.7)
    *   `BIGREC_BASE_MODEL` (Optional): 教師モデル（BIGRec）のベースモデル名 (例: `Qwen/Qwen2-0.5B`)
    *   `BIGREC_SEED` (Optional): 教師モデルの学習シード (生徒モデルもこのシードを共有します)
    *   `BIGREC_SAMPLE` (Optional): 教師モデルの学習サンプル数

### 2.2 データ読み込み仕様
*   **推奨**: `BIGREC_BASE_MODEL` 以降の引数を指定し、BIGRecの出力ディレクトリ（`BIGRec/results/...`）から直接ランキング・信頼度・埋め込みデータを読み込む方式。
*   **レガシー**: `transfer_data.sh` を使用して `DLLM2Rec/tocf/` 配下にデータをコピーしてから実行する方式（引数を省略した場合に使用）。

### 2.3 ハイパーパラメータ探索スクリプト
*   **スクリプト**: `cmd/run_dllm2rec_hyparam_search.sh`
*   **実行引数**:
    ```bash
    ./cmd/run_dllm2rec_hyparam_search.sh <DATASET> <MODEL_NAME> <GPU_ID> <BIGREC_BASE_MODEL> <BIGREC_SEED> <BIGREC_SAMPLE>
    ```
*   **動作**: `ed_weight` (0.0-1.0) と `lam` (0.0-1.0) を0.2刻みで探索し、HR@20が最大のパラメータを特定します。
*   **出力**: `results/.../[SEED]_[TEACHER_SAMPLE]/best_params.json` に最適値とスコアを保存します。

## 3. 出力仕様

### 3.1 ディレクトリ構造
DLLM2Recの学習・評価結果は以下の命名規則に従って保存されます。

**基本パス**: `DLLM2Rec/results/[DATASET]/`

*   **蒸留ありの場合 (Distillation)**:
    ```
    [STUDENT_MODEL]_distilled_[TEACHER_MODEL]/[SEED]_[TEACHER_SAMPLE]/ed_[ED_WEIGHT]_lam_[LAM]/
    ```
    *   `STUDENT_MODEL`: 生徒モデル名 (lower case)
    *   `TEACHER_MODEL`: 教師モデル名 (スラッシュはアンダースコアに置換)
    *   `SEED`: 学習シード値 (教師モデルと共通)
    *   `TEACHER_SAMPLE`: 教師モデルの学習サンプル数
    *   `ED_WEIGHT`: Collaborative Embedding Distillationの重み
    *   `LAM`: Ranking Distillationの重み
    
    例: `results/game/sasrec_distilled_Qwen_Qwen2-0.5B/0_1024/ed_0.3_lam_0.7/`

*   **蒸留なしの場合 (No Distillation)**:
    ```
    [STUDENT_MODEL]_no_distillation/[SEED]/alpha_[ALPHA]/
    ```
    *   `ALPHA`: DROの重み

    例: `results/game/sasrec_no_distillation/0/alpha_0.5/`

### 3.2 Metricsファイルフォーマット
評価結果（`test_metrics.json`, `valid_metrics.json`）は、BIGRec側の出力形式と統一された以下のJSONフォーマットで出力されます。

```json
{
    "test_metrics": {
        "NDCG": [ 
            0.123, // @1
            0.145, // @3
            0.189, // @5
            0.234, // @10
            0.289, // @20
            0.345  // @50
        ],
        "HR": [ 
            0.345, // @1
            0.390, // @3
            0.420, // @5
            0.456, // @10
            0.512, // @20
            0.589  // @50
        ]
    }
}
```
*   キー名は `test_metrics` または `valid_metrics` となります。
*   配列の各要素は以下の **k値** における評価値に対応します：

| インデックス | k値 | 説明 |
| :--- | :--- | :--- |
| 0 | **@1** | Top-1 |
| 1 | **@3** | Top-3 |
| 2 | **@5** | Top-5 |
| 3 | **@10** | Top-10 |
| 4 | **@20** | Top-20 (Early Stopping / Best Model Selectionの基準) |
| 5 | **@50** | Top-50 |

## 4. ハイパーパラメータ定義 (DLLM2Rec)

| パラメータ名 | 引数名 | Default | 説明 |
| :--- | :--- | :--- | :--- |
| **Embedding Distillation Weight** | `--ed_weight` | 0.2 (Script Default: 0.3) | LLM Embeddingの蒸留重み |
| **Ranking Distillation Lambda** | `--lam` | 0.8 (Script Default: 0.7) | Ranking Distillation損失の重み |
| **Ranking KD - Position Weight** | `--gamma_position` | 0.3 | 順位（Position）に基づく重み |
| **Ranking KD - Confidence Weight** | `--gamma_confidence` | 0.5 | 確信度（Confidence）に基づく重み |
| **Ranking KD - Consistency Weight** | `--gamma_consistency` | 0.1 | 整合性（Consistency）に基づく重み |
| **DRO Alpha** | `--alpha` | 1.0 (Script Default: 0.5) | DRO損失の重み |

※ Script Defaultは `run_dllm2rec_train.sh` で設定されているデフォルト値を指します。

## 5. シード管理
*   再現性を担保するため、教師モデル（BIGRec）の学習・推論で使用したシード値を、生徒モデル（DLLM2Rec）の学習時にも同一の値として使用します。
*   これにより、実験条件の完全な紐付けを行います。
