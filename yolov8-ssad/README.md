# YOLOv8-SSAD

歯科画像における歯の検出と番号付け（歯の列挙）のためのYOLOv8ベースのSSAD（Segment and Self-Attention Detection）モデル実装です。

## 概要

このリポジトリは、YOLOv8をベースに自己注意機構を組み込んだSSADモデルを使用して、歯科パノラマX線写真から歯を検出し、FDI表記（国際歯科連盟：Fédération Dentaire Internationale）に基づいて歯番号を付与するためのモデルを提供します。

FDI表記の歯番号システム:
- 上顎右側: 11-18
- 上顎左側: 21-28
- 下顎左側: 31-38
- 下顎右側: 41-48

## 環境のセットアップ

### 必要条件

- Python 3.12以上
- CUDA対応GPUを推奨（トレーニング:RTX4090を使用）

### セットアップ手順

1. リポジトリのクローン

```bash
git https://github.com/NewFan-TakashiHatakeyama/scogr.git
cd yolov8-ssad
```

2. 仮想環境の作成（オプション）

```bash
python -m venv venv
# Windowsの場合
venv\Scripts\activate
# Linuxの場合
source venv/bin/activate
```

3. 依存パッケージのインストール

```bash
pip install -r requirements.txt
```

4. Dockerを使用する場合（オプション）

```bash
docker-compose up -d
```

## データセット

このモデルはDentExデータセット用に設計されています。データセットは以下の構造で配置します：

```
dentex_dataset/
├── coco/
│   └── enumeration32/
│       └── annotations/
│           └── instances_train2017.json
├── yolo/
│   └── enumeration32/
│       ├── images/
│       │   ├── train/
│       │   └── val/
│       └── labels/
│           ├── train/
│           └── val/
```

COCOフォーマットからYOLOフォーマットへの変換は `convert_yolo.py` スクリプトを使用します。
（こちらの処理は既に完了しています）：

```bash
python convert_yolo.py --coco_json_path path/to/coco/annotations.json --yolo_save_dir path/to/yolo/labels
```
## 重みファイルの取得
google Dribeの歯式関連フォルダにモデルのweightsを格納しています。best.ptが対象となります。
https://drive.google.com/drive/u/1/folders/1iAdQgfKTGZqer-8DQJZ-0LB-xz5Cdlkd

プロジェクト直下にweightsフォルダを作成し、モデルの重みを追加してください。

## モデルのトレーニング

SSADモデルをトレーニングするには、以下のコマンドを実行します：

```bash
python train_ssad.py --cfg hyp.onlyssad.yaml --data dentex_enum32.yaml --weights yolov8l.pt --batch 4 --epochs 300 --device 0
```

パラメータの説明：
- `--cfg`: ハイパーパラメータ設定ファイル
- `--data`: データセット設定ファイル
- `--weights`: 事前学習済みの重み
- `--batch`: バッチサイズ
- `--epochs`: エポック数
- `--device`: 使用するGPUデバイス（`cpu`も指定可能）

## 予測と推論

トレーニング済みモデルで推論を行うには：

```bash
python predict_ssad.py --weights path/to/weights/best.pt --source path/to/image.jpg --output_dir results
```

パラメータの説明：
- `--weights`: トレーニング済みモデルの重みファイル
- `--source`: 入力画像または画像ディレクトリ
- `--output_dir`: 結果を保存するディレクトリ
- `--conf`: 信頼度閾値（デフォルト: 0.5）
- `--random`: バリデーションデータからランダムサンプルを選択して推論（デフォルト: False）

## モデル構造

YOLOv8-SSADモデルは以下の特徴を持っています：

- YOLOv8をベースとしたアーキテクチャ
- 自己注意機構（Self-Attention）による特徴強化
- マスキング戦略による学習効率の向上
- FDI歯科表記システムに基づく32クラス分類（歯番号11-48）

## ライセンス

このプロジェクトはUltralyticsのライセンスに従います。詳細は[Ultralytics License](https://ultralytics.com/license)を参照してください。

## 引用

このリポジトリを研究で使用する場合は、以下を引用してください：

```
@article{jocher2023ultralytics,
  title={Ultralytics YOLOv8: A State-of-the-Art Deep Learning Model for Object Detection and Image Segmentation},
  author={Jocher, Glenn and others},
  journal={GitHub. https://github.com/ultralytics/ultralytics},
  year={2023}
}
``` 