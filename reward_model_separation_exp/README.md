# Dreamer: 内発的・外発的報酬モデル分離実験

## プロジェクト概要

このプロジェクトは、モデルベース強化学習エージェント「Dreamer」の学習において、**内発的報酬（Intrinsic Reward）と外発的報酬（Extrinsic Reward）をそれぞれ専門に予測するモデルを分離して学習する**実験を行うものです。これにより、エージェントが「ゲームのスコア獲得」と「好奇心による環境の探査」という二つの目的をどのように統合し、より賢い行動を学習するかを検証します。

また、既存の「Noisy-TV」機能も組み合わせて、観測ノイズ下でのエージェントの頑健性や探査性能への影響も評価します。

## 主要な変更点と特徴

このプロジェクトは、元の`final_version`に対して以下の主要な変更を加えています。

*   **デュアル報酬モデル**:
    *   `RewardModel`: ゲームのスコア（外発的報酬）を予測します。
    *   `IntrinsicRewardModel`: ワールドモデルの予測誤差から計算される好奇心（内発的報酬）を予測します。
*   **デュアル価値モデル (Critic)**:
    *   `ext_critic`: 外発的報酬に基づく価値関数を学習します。
    *   `intr_critic`: 内発的報酬に基づく価値関数を学習します。
*   **分離された報酬処理**:
    *   `ReplayBuffer`は、外発的報酬と内発的報酬を別々に格納します。
    *   World Modelの学習フェーズでは、それぞれの報酬モデルが対応する報酬を予測するように学習されます。
*   **重み付けされたActorの目的関数**:
    *   `Actor`（行動モデル）は、`intr_value_scale`（Configクラスで定義）というパラメータで重み付けされた外発的価値と内発的価値の合計を最大化するように学習します。これにより、エージェントが好奇心とスコアのバランスをどのように取るかを調整できます。
*   **Noisy-TV機能の利用**:
    *   観測にノイズを加える`NoisyTVWrapper`（`Discrete`版、`Continuous`版、`EnvWrapperCIFAR`版）を利用することで、ノイズ環境下でのエージェントの性能を評価できます。

## セットアップと実行方法

### 1. Google Colabでの実行

この実験は、Google Colab上で`Run_Separate_Reward_Model_Colab.ipynb`ノートブックを利用して実行することを想定しています。

#### ステップ1: Google Driveのマウントと作業ディレクトリへの移動
*   `Run_Separate_Reward_Model_Colab.ipynb`を開きます。
*   ノートブックの指示に従い、Google Driveをマウントし、作業ディレクトリをこのプロジェクトフォルダ（`World_Model_Research/reward_model_separation_exp/`）に設定します。

#### ステップ2: 依存ライブラリのインストール
*   ノートブック内のセルを実行し、必要なPythonライブラリ（`gymnasium`, `moviepy`, `wandb`など）をインストールします。

#### ステップ3: WandBログイン (任意)
*   学習ログをWeights & Biases (WandB) で追跡したい場合は、WandBアカウントにログインします。

#### ステップ4: 学習の実行
*   ノートブックの最終セルで、実行したい実験パターン（Noisy-TVの有無やタイプ）に対応するコマンドのコメントアウト（`#`）を外し、セルを実行します。

### 2. コマンドラインでの直接実行 (Python環境が整っている場合)

Python環境が整っている場合は、`train.py`を直接実行することも可能です。

```bash
# プロジェクトディレクトリに移動
cd World_Model_Research/reward_model_separation_exp/

# 例: Noisy-TVなしで実行
python train.py --env-name "ALE/Breakout-v5" --steps 50000 --wandb --wandb-project "Dreamer-Separate-Rewards" --wandb-run-name "breakout-no-noisy-tv"

# 例: Noisy-TVあり（単純なノイズ混入版）で実行
python train.py --env-name "ALE/Breakout-v5" --steps 50000 --noisy-tv --noisy-wrapper-type simple --wandb --wandb-project "Dreamer-Separate-Rewards" --wandb-run-name "breakout-noisy-simple"

# 例: Noisy-TVあり（行動空間拡張版）で実行
python train.py --env-name "ALE/Breakout-v5" --steps 50000 --noisy-tv --noisy-wrapper-type action_space --wandb --wandb-project "Dreamer-Separate-Rewards" --wandb-run-name "breakout-noisy-action-space"
```

## ハイパーパラメータの調整

`Config`クラス（`train.py`内）で定義されている以下のパラメータは、実験結果に大きな影響を与える可能性があります。

*   `self.intr_reward_scale`: 内発的報酬が学習に与える影響の度合い。
*   `self.intr_value_scale`: Actorが内発的価値をどれだけ重視するか（外発的価値とのバランス）。

これらの値を調整することで、エージェントが好奇心とスコア獲得のバランスをどのように取るかを探ることができます。

## 開発経緯

このプロジェクトは、既存のDreamerエージェントの実装をベースに、ユーザーからの要望を受けて以下の研究テーマを実装したものです。

1.  **内発的報酬（予測誤差）の導入**: 既存コードに実装済みであることを確認。
2.  **内発的報酬と外発的報酬のモデル分離**:
    *   `student_code.py`に`IntrinsicRewardModel`を追加。
    *   `train.py`を大規模に修正し、デュアル報酬モデル、デュアルCritic、分離された損失計算、重み付けされたActorの目的関数を実装。
3.  **Noisy-TV機能の統合**: `train.py`で`NoisyTVWrapper`クラスが利用されるように統合。
4.  **Colabでの実行環境整備**: `Run_Separate_Reward_Model_Colab.ipynb`を作成し、容易な実行を可能に。
5.  **デバッグ**: 実装中に発生した複数の`NameError`, `IndentationError`, `RuntimeError`（特にPyTorchの複雑な自動微分グラフとテンソル形状の不一致によるもの）を、ユーザーとの対話を通じて解決。

## ライセンス

**重要**: このコードはGoogle Gemini CLIによって生成されたものであり、明示的なライセンスは含まれていません。

もしこのプロジェクトを他の人と共有したり、公開したりする場合は、**ご自身で適切なオープンソースライセンス（例: MIT License, Apache License 2.0 など）を選択し、プロジェクトに追加することを強く推奨します。**これにより、コードの利用条件が明確になり、将来的な法的問題を避けることができます。

（この情報は法的なアドバイスではありません。法的な問題については専門家にご相談ください。）
