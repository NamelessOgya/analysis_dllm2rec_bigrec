# Active Learning Implementation Plan for BIGRec

本ドキュメントでは、`research/al_methods.md` で提案した6つのActive Learning手法を実装するための計画を記述します。

## 1. 新規スクリプトの作成
**File**: `BIGRec/data/game_bigrec/sample_data.py`

このスクリプトは、既存の `train.json` と DROS (SASRec) の推論結果を受け取り、指定された手法に基づいてサブセット抽出を行い、新しい `train_{method}_{ratio}.json` を生成します。

### 引数 (Arguments)
*   `--input_json`: 元の学習データ (例: `train.json`)
*   `--input_df`: DROS学習用データ (例: `train_data.df`) - UIDマッピング用
*   `--dros_score`: DROS推論スコア (例: `.../train.pt`) - 配列形状 `[N, ItemNum]`
*   `--dros_uid`: DROS推論スコアに対応するUID (例: `.../train_uids.pt`)
*   `--item_emb`: アイテム埋め込み (例: `.../all_embeddings.pt`) - Clustering手法で使用
*   `--method`: 使用するサンプリング手法 (`random`, `pop_inverse`, `diversity`, `loss`, `entropy`, `error_rank`, `clustering`)
*   `--ratio`: サンプリング率 (例: 0.1, 0.5) 
*   `--output_json`: 出力ファイル名
*   `--seed`: 乱数シード

### 処理フロー
1.  **データ読み込み**:
    *   `train.json` を読み込み、`uid` をキーにした辞書を作成。
    *   `train_data.df` を読み込み、アイテムIDや正解ラベル (`next`) を取得。
2.  **DROS結果の結合 (DROSベース手法の場合)**:
    *   `train.pt` (Tensor/Numpy) と `train_uids.pt` をロード。
    *   DROSの出力行と `train.json` のエントリをUIDを用いて紐付け。
    *   欠損がある場合 (DROS側でBatch dropなど) は、共通するUIDのみを対象とする。
3.  **スコアリング (Scoring)**:
    メモリ圧迫を防ぐため、**バッチ処理 (Batch Processing)** で実行します。
    `train.pt` 全体をGPUに乗せることはせず、CPU上で保持し、バッチごとにGPUへ転送して計算します。
    各サンプルに対して、手法に応じたスコアを計算します。
    *   **Hardness (Loss)**: `CrossEntropy(logits[uid], target[uid])`
    *   **Uncertainty (Entropy)**: `Entropy(Softmax(logits[uid]))`
    *   **Error-Correction**: `Rank(target[uid] in sorted_logits)`
    *   **Clustering**: `KMeans(embeddings).predict(target_item)` -> Stratified Sampling
    *   **Pop-Inverse**: `1 / ItemFreq[target_item]`
    *   **Diversity**: Greedy Selection (Set Coverage)
4.  **サンプリング (Sampling)**:
    *   スコアの上位 `K` 件 (または確率的サンプリング) を抽出。
5.  **出力**:
    *   抽出されたエントリのみを含むJSONリストを作成し、`output_json` に保存。

## 2. 既存コードの修正・確認事項
*   **BIGRec/train.py**: 特段の修正は不要。新しいJSONファイルを `--data_path` 引数等で指定するだけで動作するはずである。
*   **DLLM2Rec**: 既に `train.pt`, `train_uids.pt` をエクスポートする機能が存在するため、修正不要。

## 3. 実行ワークフロー (例)

1.  **DROS (SASRec) の学習 & 推論結果出力**
    ```bash
    bash cmd/run_sasrec_baseline.sh game 0 200 2024 0
    # -> results/game/sasrec_no_distillation/.../train.pt が生成される
    ```

2.  **サンプリング実行**
    ```bash
    python BIGRec/data/game_bigrec/sample_data.py \
      --input_json BIGRec/data/game_bigrec/train.json \
      --dros_score results/game/.../train.pt \
      --dros_uid results/game/.../train_uids.pt \
      --method loss \
      --ratio 0.5 \
      --output_json BIGRec/data/game_bigrec/train_loss_0.5.json
    ```

3.  **BIGRec の学習 (Finetuning)**
    ```bash
    # 既存の学習スクリプトを使用 (データパスを変更)
    bash cmd/run_bigrec_train.sh ...
    ```

## 4. 検証計画 (Verification Plan)
1.  **Unit Test**:
    *   `sample_data.py` の各スコアリング関数が正しい値を返すか確認するテストを作成 (ダミーデータ使用)。
    *   UIDのマッピングが正しく行われているか確認 (LogitsとTargetの整合性)。
2.  **Integration Test**:
    *   実際に `train.pt` をロードし、生成された `train_sampled.json` のフォーマットが `train.json` と同一であり、レコード数が指定通りであることを確認。
    *   生成されたJSONを用いてBIGRecの学習がエラーなく開始できることを確認 (1エポック等で確認)。

## 課題・リスク
*   **メモリ使用量 (Critical)**:
    *   `train.pt` は約3GBと比較的大容量です。
    *   **対策1 (CPU Loading)**: いきなり `to(device)` でGPUに転送せず、CPUメモリ (`map_location='cpu'`) にロードします。System RAMであれば3GBは十分に許容範囲です。
    *   **対策2 (Batch Processing)**: スコア計算（各サンプルのLossやEntropy計算）を行う際は、全データをまとめて処理せず、必ず**バッチ処理 (例えば batch_size=1024)** で行います。
        *   ループ内で `batch_logits = all_logits[start:end].to(device)` のように必要な分だけGPUに転送して計算し、スコアのみをCPUに戻して確保します。
        *   これにより、Peak Memory Usageを低く抑えます。
    *   **対策3 (Half Precision)**: 可能であれば `torch.float16` や `bfloat16` での計算を維持し、メモリ帯域と容量を節約します。
*   **UID整合性**: DROSとBIGRecでデータのバージョン違い等によりUIDがズレていないか、`verify_data_alignment.py` (過去タスクで作成済み想定) 等で事前に保証されている前提とする。
