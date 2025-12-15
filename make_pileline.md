# pipelineの実装. 

## 目的. 
実験の効率化のために、pipelineの機能を実装したい。  
以下のコマンドを自動実行するpipelineを実装する. 
- ./cmd/run_sasrec_baseline.sh. 
- ./cmd/run_create_learning_active_data.sh 
- ./cmd/train_bigrec.sh. 
- ./cmd/inference_vllm.sh.
- ./cmd/run_dllm2rec_train.sh
  
実験効率化のために、以下の機能を備えるものとする。  
### dvcを用いた適切な再実行判断. 
dvcを用いて、既に実行済み処理はスキップして後続の実行を行う。  

### 管理のための最小限のディレクトリ構造. 
実験に際して必要なパラーメータだけを記録したディレクトリ構造を補助. 
  
### 集計機能. 
実行結果のファイルを集計してまとめる機能. 
  
## 実装内容. 
### ディレクトリ構造の変更. 
現状の保存ファイルのディレクトリ構造には、実験に際して不要なものも含まれているので、
最小限の情報のみキープする構造に変更する。  
  
| 変数名 | 詳細 |   
| --- | --- |   
| dataset_name | game_bigrecなど、使用するデータセット名 |   
| seed | seed値.全処理で統一のシード値を用いて実験する。 |
| alpha | sasrecの際に用いるalpha値 | 
| sampling_strategy | active learningの際に用いる手法 |  
| sample_num | 学習データの総サンプル数 |
| al_ratio | active learningの際に用いる手法 |  
| base_model_name | 学習に用いるベースモデルの名前 |
| templete | 学習に用いるプロンプトテンプレートの名前 |
| ed_weight | DLLM2Rec蒸留のパラメータ | 
| lambda | DLLM2Rec蒸留のパラメータ |

### pipeline機能の実装. 
以下のような形で実験変数がcsvファイルとして与えられたときに、
全パターンを実行できるようなパイプラインを作成する。  
(必要に応じてcsv → dvc pipeline定義ファイルへの変換shを作っても良い。)

| dataset_name | seed | ... | lambda | GPUID | 
| --- | --- | --- | --- | --- | 
| game_bigrec | 0 | ... | 0.7 | 0 | 
| game_bigrec | 0 | ... | 0.8 | 1 |

パイプライン実行の際は既に実行されている場合は再度実行せずスキップする仕組みをdvcを用いて実装する。  
dvcに搭載されている、ファイルの復元機能は使わなくても良い。
  
この環境にはないが、GPUを2台使える。  
煩雑さを避けるために、二つのbashからGPU1用、GPU0用のpipeline処理を実行するようにしたい。  
ので、GPU1用のインパイプラインとGPU0用のパイプラインを分ける必要がある。
