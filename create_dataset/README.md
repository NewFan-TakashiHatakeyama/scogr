# 歯科用X線画像 YOLOデータセット作成ツール

このプロジェクトは、歯科用X線画像に対してYOLOフォーマットのアノテーションを作成するためのツールです。歯の状態、歯科修復物、および病理所見に焦点を当てています。

## プロジェクト構造

```
create_dataset/
├── main.py              # コマンドラインインターフェースを持つメインスクリプト
├── src/                 # ソースコードモジュール
│   ├── __init__.py
│   ├── condition.py     # 歯の状態アノテーションモジュール
│   ├── condition_split.py # 歯の状態データセットの学習・検証分割
│   ├── condition_viz.py # 歯の状態アノテーションの可視化
│   ├── pathology.py     # 歯科病理所見アノテーションモジュール
│   ├── pathology_split.py # 病理所見データセットの学習・検証分割
│   ├── pathology_viz.py # 病理所見アノテーションの可視化
│   ├── restoration.py   # 歯科修復物アノテーションモジュール
│   ├── restoration_split.py # 修復物データセットの学習・検証分割
│   └── restoration_viz.py # 修復物アノテーションの可視化
├── Dockerfile           # Dockerコンテナのビルド設定
├── docker-compose.yml   # Docker Compose設定ファイル
└── dataset/             # データセットファイル
    └── Teeth Segmentation on dental X-ray images/
        ├── annotation/  # JSONアノテーション
        ├── images/      # X線画像
        ├── labels/      # 歯のバウンディングボックスラベル
        ├── condition_labels/ # 生成された歯の状態アノテーション
        ├── pathology_labels/ # 生成された病理所見アノテーション
        └── restoration_labels/ # 生成された修復物アノテーション
```

## 使用方法

メインスクリプト（`main.py`）は、アノテーションの作成と可視化のためのコマンドラインインターフェースを提供します：

```bash
# ヘルプの表示
python main.py --help

# 歯の状態アノテーションの作成
python main.py condition [--base-dir PATH] [--output-dir PATH]

# 歯の状態アノテーションの可視化
python main.py condition-viz [--base-dir PATH] [--condition-labels-dir PATH] [--images-dir PATH] [--labels-dir PATH] [--output-dir PATH]

# 歯の状態データセットの学習・検証分割
python main.py condition-split [--base-dir PATH] [--condition-labels-dir PATH] [--images-dir PATH] [--val-ratio RATIO]

# 修復物アノテーションの作成
python main.py restoration [--base-dir PATH] [--output-dir PATH]

# 修復物アノテーションの可視化
python main.py restoration-viz [--base-dir PATH] [--restoration-labels-dir PATH] [--images-dir PATH] [--labels-dir PATH] [--output-dir PATH]

# 修復物データセットの学習・検証分割
python main.py restoration-split [--base-dir PATH] [--restoration-labels-dir PATH] [--images-dir PATH] [--val-ratio RATIO]

# 病理所見アノテーションの作成
python main.py pathology [--base-dir PATH] [--output-dir PATH]

# 病理所見アノテーションの可視化
python main.py pathology-viz [--base-dir PATH] [--pathology-labels-dir PATH] [--images-dir PATH] [--labels-dir PATH] [--output-dir PATH]

# 病理所見データセットの学習・検証分割
python main.py pathology-split [--base-dir PATH] [--pathology-labels-dir PATH] [--images-dir PATH] [--val-ratio RATIO]

# 特定のデータセットタイプに対して全ステップを実行
python main.py all condition [--base-dir PATH] [--val-ratio RATIO]
python main.py all restoration [--base-dir PATH] [--val-ratio RATIO]
python main.py all pathology [--base-dir PATH] [--val-ratio RATIO]
```

## Dockerでの使用方法

このプロジェクトはDockerコンテナで実行することもできます。

### 準備

```bash
# Dockerイメージをビルド
docker-compose build
```

### 実行例

```bash
# ヘルプを表示
docker-compose run dataset-creator

# 歯の状態アノテーションを作成
docker-compose run dataset-creator condition

# 病理所見データセットの全処理を実行
docker-compose run dataset-creator all pathology
```

詳しい使用方法は `docker-usage.md` を参照してください。

## データセットの詳細

### 歯の状態クラス

| クラスID | 状態 | 英語表記 |
|----------|------|----------|
| 0 | 未萌歯 | Unerupted tooth |
| 1 | 先欠歯 | Congenitally missing tooth |
| 2 | 埋伏歯 | Impacted tooth |

### 歯科修復物クラス

| クラスID | 修復物の種類 | 説明 |
|----------|------------|------|
| 0 | インレー | 歯の詰め物（金属製など） |
| 1 | クラウン | 歯冠補綴物 |
| 2 | 処置歯 | 歯内療法や保存修復処置が施された歯 |
| 3 | その他 | 上記以外の修復物 |

### 歯科病理所見クラス

| クラスID | 病理所見 | 英語表記 |
|----------|---------|----------|
| 0 | う蝕 | Caries |
| 1 | 根尖病変 | Periapical lesion |
| 2 | 歯根吸収 | Root resorption |
| 3 | 骨吸収 | Bone loss |

## 使用例

```bash
# 歯の状態アノテーションの作成、可視化、およびデータセット分割
python main.py all condition

# 修復物アノテーションの作成、可視化、およびデータセット分割
python main.py all restoration

# 病理所見アノテーションの作成、可視化、およびデータセット分割
python main.py all pathology

# 歯の状態アノテーションのみ作成
python main.py condition

# 修復物アノテーションの可視化のみ実行
python main.py restoration-viz

# 病理所見データセットの学習・検証分割のみ実行
python main.py pathology-split
```

## 出力結果

スクリプトは以下を生成します：

1. YOLOフォーマットのアノテーション（*.txtファイル）
2. データセットの統計情報（README.md）
3. アノテーションの可視化結果
4. 学習・検証データの分割
5. YOLO設定ファイル（*.yaml） 