# SCOGR

## 概要

SCOGRは、Domain-Driven DesignとClean Architectureの原則に基づいて構築された画像認識・物体検出アプリケーションです。YOLOv8をベースとした物体検出と、データセットの作成・管理・可視化機能を提供します。プロジェクトは明確に分離されたレイヤーを持ち、ビジネスロジックとインフラストラクチャの懸念を適切に分離しています。

## 主要コンポーネント

### coco-viewer
COCOフォーマットのデータセットを視覚的に確認・閲覧するためのツールです。アノテーションの確認やデータセットの品質チェックに利用できます。

```powershell
# 使用例
cd tools/coco-viewer
python coco_viewer.py --images /path/to/images --annotations /path/to/annotations.json
```

### create_dataset
物体検出用のデータセットを作成・管理するためのユーティリティです。画像の収集、前処理、アノテーション変換などの機能を提供します。

```powershell
# 使用例
cd tools/create_dataset
python create_dataset.py --source /path/to/source --output /path/to/output --format yolo
```

### draw-YOLO-box
YOLOフォーマットのアノテーションをもとに、画像上にバウンディングボックスを描画するツールです。検出結果の可視化や、アノテーションの正確性の確認に使用できます。

```powershell
# 使用例
cd tools/draw-YOLO-box
python draw_boxes.py --input /path/to/image --labels /path/to/labels --output /path/to/output
```

### yolov8-ssad
YOLOv8をベースにした半教師あり物体検出（Semi-Supervised Anomaly Detection）の実装です。少量のラベル付きデータと大量の未ラベルデータを活用して高精度な物体検出を実現します。

```powershell
# トレーニング例
cd detection/yolov8-ssad
python train.py --data /path/to/data.yaml --weights yolov8s.pt --epochs 100

# 推論例
python predict.py --weights /path/to/best.pt --source /path/to/images
```

## アーキテクチャ

このプロジェクトは以下のレイヤーで構成されています：

- **ドメイン層** - ビジネスルールとエンティティを含みます
- **アプリケーション層** - ユースケースを実装し、ドメインサービスを調整します
- **インフラストラクチャ層** - データベースアクセス、外部APIとの連携などを担当します
- **プレゼンテーション層** - UIとエンドユーザーとのインタラクションを処理します

## 技術スタック

### 物体検出・画像処理
- YOLOv8
- OpenCV
- PyTorch
- COCO API

### バックエンド
- Python (FastAPI)
- SQLAlchemy (ORM)
- Pydantic (バリデーション)
- pytest (テスト)

### フロントエンド
- TypeScript
- Next.js / React
- Jest & React Testing Library

## 開発環境のセットアップ

### 前提条件
- Python 3.9+
- Node.js 16+
- CUDA対応GPUとドライバー（高速な物体検出処理のため推奨）
- PostgreSQL

### インストール

```powershell
# 共通の依存関係
pip install -r requirements.txt

# YOLOv8の追加依存関係
pip install ultralytics

# バックエンドのセットアップ
cd backend
pip install -r requirements.txt
# または Poetry を使用する場合
# poetry install

# 環境変数の設定
copy .env.example .env
# .env ファイルを編集して必要な設定を行う

# フロントエンドのセットアップ
cd ../frontend
npm install
# または Yarn を使用する場合
# yarn install
```

### 開発サーバーの起動

```powershell
# バックエンド
cd backend
uvicorn app.main:app --reload

# フロントエンド
cd frontend
npm run dev
# または
# yarn dev
```

## プロジェクト構造

```
scogr/
├── backend/
│   ├── app/
│   │   ├── domain/       # エンティティ、バリューオブジェクト、ドメインサービス
│   │   ├── application/  # ユースケース、DTOの変換
│   │   ├── infrastructure/ # リポジトリの実装、外部サービス連携
│   │   └── presentation/ # APIエンドポイント、コントローラー
│   ├── tests/          # テストケース
│   └── config/         # 設定ファイル
│
├── detection/
│   ├── yolov8-ssad/    # 半教師あり物体検出の実装
│   └── models/         # 事前学習済みモデルと設定
│
├── tools/
│   ├── coco-viewer/    # COCOデータセット可視化ツール
│   ├── create_dataset/ # データセット作成ユーティリティ
│   └── draw-YOLO-box/  # バウンディングボックス描画ツール
│
├── frontend/
│   ├── src/
│   │   ├── components/  # 再利用可能なUIコンポーネント
│   │   ├── pages/       # ページコンポーネント
│   │   ├── domain/      # ドメインモデル（TypeScript）
│   │   ├── application/ # アプリケーションサービス
│   │   ├── infrastructure/ # API呼び出し、状態管理
│   │   └── utils/       # ユーティリティ関数
│   └── tests/          # フロントエンドテスト
│
└── docs/              # プロジェクトドキュメント、API仕様など
```

## テスト

```powershell
# バックエンドテスト
cd backend
pytest

# 物体検出モデルテスト
cd detection/yolov8-ssad
python test.py --weights best.pt --data test_data.yaml

# フロントエンドテスト
cd frontend
npm test
# または
# yarn test
```

## コード規約

- **Python**: flake8 と black を使用
- **TypeScript**: ESLint と Prettier を使用

## CI/CD

GitHub Actions を使用して以下を自動化:
- コードの静的解析
- テストの実行
- ビルドの検証
- モデルのベンチマーク評価

## 貢献ガイドライン

1. GitHubフローに従ってください（featureブランチからmainへのPR）
2. コミットメッセージは `feat/fix/chore: 概要` の形式で記述
3. PRを出す前に、テストが全て通ることを確認
4. コードレビューは少なくとも1人の承認が必要

## ライセンス

このプロジェクトは [ライセンス名] のもとで公開されています。詳細はLICENSEファイルを参照してください。 
