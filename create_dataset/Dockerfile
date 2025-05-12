FROM python:3.9-slim

WORKDIR /app

# 必要なパッケージをインストール
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# 依存パッケージをインストール
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# アプリケーションコードをコピー
COPY main.py .
COPY src/ src/

# 作業ディレクトリを作成
RUN mkdir -p /app/dataset/output

# 実行時のコマンド
ENTRYPOINT ["python", "main.py"]

# デフォルトコマンド（ヘルプを表示）
CMD ["--help"] 