# Final Version - README

このフォルダには、Dreamerエージェントの学習に必要なすべてのPythonスクリプトが含まれています。Colabノートブックのタイムアウト問題を避けるため、各スクリプトは直接ファイルとして作成されています。

以下の手順に従って、新しいColabノートブックから学習を実行してください。

---

## Colabでの学習実行手順

### ステップ1: 新しいノートブックを開く
Google Colabで、**新しい空のノートブック**を開きます。

### ステップ2: Google Driveをマウント
最初のセルに以下のコードを貼り付けて実行し、Google Driveをマウントします。

```python
from google.colab import drive
drive.mount('/content/drive')
```

### ステップ3: 作業ディレクトリへ移動
次のセルに以下のコードを貼り付けて実行します。`project_path` は、あなたの環境に合わせて適宜修正してください。

```python
import os

# このプロジェクトのファイルがあるGoogle Driveのパス
project_path = '/content/drive/MyDrive/final_version'

try:
    os.chdir(project_path)
    print(f"カレントディレクトリを次に変更しました: {os.getcwd()}")
    # ファイルが存在するか確認
    assert os.path.exists('train.py'), "train.pyが見つかりません。"
    print("必要なスクリプトファイルが見つかりました。")
except (FileNotFoundError, AssertionError) as e:
    print(f"エラー: {e}")
```

### ステップ4: 依存ライブラリのインストール
次のセルに以下のコードを貼り付けて実行し、学習に必要なライブラリをインストールします。

```python
!apt-get update
!apt-get install -y ffmpeg xvfb libsdl2-dev python3-opengl cmake zlib1g-dev
!pip install gymnasium[classic_control,atari,accept-rom-license] moviepy imageio wandb opencv-python matplotlib tqdm pillow torchvision
```

### ステップ5: WandB ログイン
（任意）Weights & Biasesで学習ログを記録する場合、次のセルを実行してログインします。

```python
import wandb
wandb.login()
```

### ステップ6: 学習の実行
最後のセルに以下のコードを貼り付けます。実行したい実験パターンの行のコメント（`#`）を解除し、セルを実行してください。

```python
import os
os.environ['MUJOCO_GL'] = 'egl'

# --- 実行したいコマンドの行頭の「#」を削除してください（一度に一つのみ） --- #

# パターン1: Noisy-TV機能なしで実行
# !python train.py --env-name "ALE/Breakout-v5" --steps 50000 --wandb

# パターン2: Noisy-TV機能あり（当初の、単純なノイズ混入版）
!python train.py --env-name "ALE/Breakout-v5" --steps 50000 --noisy-tv --noisy-wrapper-type simple --wandb

# パターン3: Noisy-TV機能あり（行動空間拡張版）
# !python train.py --env-name "ALE/Breakout-v5" --steps 50000 --noisy-tv --noisy-wrapper-type action_space --wandb
```