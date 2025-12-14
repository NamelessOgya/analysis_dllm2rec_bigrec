# Active Learning Methods for BIGRec (Proposal)

本ドキュメントでは、BIGRec (LLM Recommender) の学習データサンプリングにおけるActive Learning手法を6つ提案します。
DROS (Sequential Recommendation Model, SASRec等) の推論結果を利用するもの3つと、利用しないもの3つで構成されています。

## A. DROS (SASRec) 推論結果を利用する手法 (3選)
これらの手法は、`DLLM2Rec` 側で生成される `train.pt` (Training Dataに対するDROSのLogits) を利用して、サンプルの「難易度」や「不確実性」を測定し、学習効果の高いサンプルを選定します。

### 1. Hardness-Based Sampling (Loss Maximization)
*   **概要**: DROSモデルが正解アイテムを予測する際の**Cross-Entropy Loss (損失)** が大きいサンプルを抽出します。
*   **理論的背景**: 損失が大きいサンプルは、DROSモデルにとって「難しい」サンプルです。これらは既存のSequential Patternだけでは予測困難なケース（特異な遷移やノイズなど）を含んでいる可能性が高く、LLMの持つ意味的（Semantic）な推論能力で補完すべきデータであると考えられます。Hard Negative Miningの概念に近いアプローチです。
*   **実装**: `train.pt` (Logits) と 正解ラベル (`train_data.df['next']`) からCross-Entropy Lossを計算し、Top-Kを選定します。

### 2. Uncertainty-Based Sampling (Entropy Maximization)
*   **概要**: DROSモデルの予測確率分布の**エントロピー (Entropy)** が大きいサンプルを抽出します。
*   **理論的背景**: エントロピーが高い状態は、モデルが次に来るアイテムを絞りきれていない（迷っている）「不確実」な状態を示します。このような曖昧なコンテキストを持つデータこそ、LLMの広範な知識や推論能力によって解決すべき重要なデータポイントです。
*   **実装**: `train.pt` (Logits) にSoftmaxを適用して確率分布 $P$ を得て、$-\sum P(x) \log P(x)$ を計算し、Top-Kを選定します。

### 3. Error-Correction Sampling (Low-Rank Selection)
*   **概要**: DROSモデルにおける正解アイテムの**ランク (順位)** が低い（例えば20位以下など）サンプルを抽出します。
*   **理論的背景**: DROSが完全に予測を外している（下位にランク付けしている）サンプルは、DROSのアーキテクチャでは捉えきれないパターンを含んでいます。これらを重点的にBIGRecに学習させることで、Sequential Modelの弱点をLLMで補完・修正（Error Correction）する効果を狙います。
*   **実装**: `train.pt` のLogitsをソートし、正解アイテムの順位を算出。順位が悪い（値が大きい）順に選定します。

---

## B. DROS 推論結果を利用しない手法 (3選)
これらの手法は、DROSの学習済みモデルを必要とせず、データの統計的性質や外部指標に基づいてサンプリングを行います。

### 4. Clustering-Based Sampling (Semantic Diversity)
*   **概要**: 事前学習済みアイテム埋め込み（LLM Embeddingsなど）を用いてアイテムをクラスタリングし、各クラスタから均等に、または重心に近いサンプルを抽出します。
*   **理論的背景**: 単なるIDベースのカバー率（手法6）とは異なり、アイテムの「意味的（Semantic）」な多様性を最大化します。似通ったジャンルのゲームばかりではなく、意味空間上で分散した多様なコンテキストを学習させることで、未知のカテゴリに対する未学習領域を減らし、汎化性能を向上させます。
*   **実装**: `all_embeddings.pt` (または `id2name.txt` から生成した埋め込み) に対して K-Means クラスタリングを行い、各クラスタからサンプルを選定します。

### 5. Popularity-Inverse Sampling (Long-Tail Focus)
*   **概要**: 出現頻度の低い（不人気な）アイテムを正解として持つインタラクションを優先的にサンプリングします。
*   **理論的背景**: 推薦システムは一般に人気アイテムにバイアスがかかりやすく（Popularity Bias）、ロングテールアイテムの精度が低くなりがちです。LLMのSFT（Supervised Fine-Tuning）において、レアなアイテムのパターンを多く見せることで、カタログ全体の網羅性とロングテールに対する精度向上を狙います。
*   **実装**: 全学習データのアイテム出現頻度を計算し、その逆数に比例した確率でサンプリングを行います。

### 6. Interaction Diversity Sampling (Coverage Maximization)
*   **概要**: カタログ内のなるべく多くのユニークな「アイテム」や「ユーザー」をカバーするようにサンプリングします。
*   **理論的背景**: データセット縮小時に特定のアイテムやユーザーが完全に消失することを防ぎます。多様なコンテキストをLLMに学習させることで、特定カテゴリへの過学習を防ぎ、汎化性能を高めます。
*   **実装**: まだ選択されていないアイテムを含むサンプルを貪欲法（Greedy）で優先的に追加し、アイテムカバレッジが最大化するよう選定します。
