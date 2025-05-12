# Dockerコンテナでの使用方法

このプロジェクトはDockerコンテナで実行できるように設定されています。これにより、環境構築の手間を省き、どのプラットフォームでも同じ結果を得ることができます。

## 前提条件

- [Docker](https://www.docker.com/products/docker-desktop)がインストールされていること
- [Docker Compose](https://docs.docker.com/compose/install/)がインストールされていること（Windows/Macでは通常Dockerに同梱）

## 基本的な使い方

### 1. イメージのビルド

プロジェクトのルートディレクトリで以下のコマンドを実行してDockerイメージをビルドします。

```bash
docker-compose build
```

### 2. ヘルプの表示

以下のコマンドでツールのヘルプを表示できます。

```bash
docker-compose run dataset-creator
```

### 3. 特定のコマンドの実行

コマンドを指定して実行するには、`command`パラメータを使用します。例えば、歯の状態アノテーションを作成するには：

```bash
docker-compose run dataset-creator condition
```

修復物アノテーションの可視化を実行するには：

```bash
docker-compose run dataset-creator restoration-viz
```

### 4. パラメータの指定

コマンドラインパラメータを指定することもできます：

```bash
docker-compose run dataset-creator condition --base-dir /app/dataset --output-dir /app/dataset/condition_labels
```

### 5. 一括処理の実行

特定のデータセットタイプに対して全ステップを実行するには：

```bash
docker-compose run dataset-creator all condition
```

## データの共有

Dockerコンテナと実行環境（ホスト）の間で、`./dataset`ディレクトリが共有されています。つまり：

1. ホストの`./dataset`フォルダに入れたデータはコンテナ内の`/app/dataset`から参照できます
2. コンテナが生成した出力データは自動的にホストの`./dataset`フォルダに保存されます

## Windows環境での注意点

Windows環境でDocker Desktopを使用する場合、パスの扱いに注意が必要です。WSL2バックエンドを使用すると良いでしょう。

```bash
# Windowsのコマンドプロンプトでの実行例
docker-compose run dataset-creator condition --base-dir /app/dataset
```

## トラブルシューティング

### ボリュームマウントの問題

データが正しく共有されない場合は、Docker設定でボリュームが正しくマウントされているか確認してください。

### パーミッションエラー

Linux/macOSで実行する場合、コンテナ内で生成されたファイルの所有者が`root`になることがあります。その場合は以下のコマンドでパーミッションを修正できます：

```bash
sudo chown -R $(id -u):$(id -g) ./dataset
``` 