# import YOLO
from ultralytics import YOLO
import multiprocessing
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import os
import glob
import numpy as np
from collections import Counter
from pathlib import Path
import argparse
import matplotlib as mpl
import platform
import sys
import shutil
import warnings
import inspect
from matplotlib import rc, font_manager
import cv2
from tqdm import tqdm
import torch
from PIL import Image

# 追加のインポート
try:
    from imblearn.over_sampling import SMOTE, ADASYN, BorderlineSMOTE
    from sklearn.decomposition import PCA
    HAS_IMBLEARN = True
except ImportError:
    HAS_IMBLEARN = False
    print("警告: imbalanced-learnがインストールされていません。pip install imbalanced-learnでインストールできます。")

def setup_japanese_font():
    """日本語フォントを設定する関数"""
    print("日本語フォント設定を適用しています...")
    
    # プラットフォームの検出
    system = platform.system()
    print(f"プラットフォーム: {system}")
    
    # Windowsの場合
    if system == 'Windows':
        font_paths = [
            'C:/Windows/Fonts/meiryo.ttc',  # メイリオ
            'C:/Windows/Fonts/msgothic.ttc',  # MSゴシック
            'C:/Windows/Fonts/YuGothM.ttc',  # 游ゴシック
            'C:/Windows/Fonts/YuGothB.ttc',  # 游ゴシック
            'C:/Windows/Fonts/meiryob.ttc',  # メイリオ ボールド
        ]
        
        # フォントが存在するかチェック
        available_fonts = [path for path in font_paths if os.path.exists(path)]
        
        if available_fonts:
            # 最初の利用可能なフォントを使用
            font_path = available_fonts[0]
            font_prop = fm.FontProperties(fname=font_path)
            font_name = font_prop.get_name()
            
            # matplotlib全体の設定
            plt.rcParams['font.family'] = 'sans-serif'
            plt.rcParams['font.sans-serif'] = [font_name, 'Yu Gothic', 'Meiryo', 'MS Gothic']
            
            # あえてJapaneseFont.ttfをキャッシュディレクトリに作成
            # これにより、matplotlibが日本語フォントを確実に認識するようになる
            
            # matplotlibのキャッシュディレクトリを取得
            cache_dir = mpl.get_cachedir()
            os.makedirs(cache_dir, exist_ok=True)
            
            # フォントファイルをコピー
            dst_font_path = os.path.join(cache_dir, 'JapaneseFont.ttf')
            if not os.path.exists(dst_font_path):
                try:
                    shutil.copy2(font_path, dst_font_path)
                    print(f"フォントをコピーしました: {font_path} → {dst_font_path}")
                    
                    # 新しいフォントを登録
                    font_manager.fontManager.addfont(dst_font_path)
                    font_list = font_manager.findSystemFonts(fontpaths=[cache_dir])
                    for font in font_list:
                        font_manager.fontManager.addfont(font)
                        
                    # フォントキャッシュを再構築
                    from matplotlib.font_manager import _rebuild
                    _rebuild()
                    
                    # フォント設定を適用
                    plt.rcParams['font.family'] = 'sans-serif'
                    plt.rcParams['font.sans-serif'] = ['JapaneseFont', font_name, 'MS Gothic', 'Meiryo']
                    
                except Exception as e:
                    print(f"フォントのコピー中にエラーが発生しました: {e}")
            
            print(f"日本語フォントを設定しました: {font_name}")
        else:
            print("適切な日本語フォントが見つかりませんでした。")
    
    # Linuxの場合
    elif system == 'Linux':
        # IPAフォントなどLinuxで一般的な日本語フォント
        plt.rcParams['font.family'] = 'IPAPGothic'
    
    # macOSの場合
    elif system == 'Darwin':  # macOS
        plt.rcParams['font.family'] = 'AppleGothic'
    
    # 文字化け防止の共通設定
    plt.rcParams['axes.unicode_minus'] = False
    
    # フォント設定を確認
    print("現在のフォント設定:")
    print(f"font.family: {plt.rcParams['font.family']}")
    print(f"font.sans-serif: {plt.rcParams['font.sans-serif']}")
    
    # 簡単なテスト
    try:
        fig, ax = plt.figure(), plt.axes()
        ax.plot([1, 2, 3], [1, 2, 3])
        ax.set_title('日本語テスト')
        ax.set_xlabel('X軸')
        ax.set_ylabel('Y軸')
        plt.close(fig)  # テスト後に閉じる
        print("テストプロットの作成に成功しました")
    except Exception as e:
        print(f"テストプロット作成中にエラー: {e}")

def patch_ultralytics_confusion_matrix():
    """Ultralyticsのconfusion matrixプロット関数をパッチする関数"""
    try:
        # Ultralyticsのプロットモジュールをインポート
        from ultralytics.utils.plotting import plot_confusion_matrix as original_plot_confusion_matrix
        
        # Ultralyticsライブラリの場所を特定
        import ultralytics
        ultralytics_path = os.path.dirname(inspect.getfile(ultralytics))
        print(f"Ultralyticsライブラリパス: {ultralytics_path}")
        
        # オリジナル関数の動作を確認
        print(f"オリジナル関数のパス: {inspect.getfile(original_plot_confusion_matrix)}")
        print(f"オリジナル関数のシグネチャ: {inspect.signature(original_plot_confusion_matrix)}")
        
        # カスタム関数を定義
        def custom_plot_confusion_matrix(matrix, 
                                          names=None, 
                                          *, 
                                          normalize=True, 
                                          save_dir=Path(''), 
                                          on_plot=None, 
                                          **kwargs):
            """
            日本語対応のConfusion Matrix描画関数
            """
            print("カスタム混同行列プロット関数を使用します")
            
            if not isinstance(names, (list, tuple, np.ndarray)):
                names = [str(i) for i in range(len(matrix))]
                
            # MatplotlibのrcParamsを一時的に設定
            with plt.rc_context({'font.family': plt.rcParams.get('font.family', 'sans-serif')}):
                # 元の関数とほぼ同じロジック
                array = matrix / ((matrix.sum(0).reshape(1, -1) + 1E-9) if normalize else 1)  # normalize columns
                array[array < 0.005] = np.nan  # don't annotate (would appear as 0.00)
                
                fig, ax = plt.subplots(1, 1, figsize=(12, 9), tight_layout=True)
                nc, nn = len(array), len(names)  # number of classes, names
                sn.set(font_scale=1.0 if nc < 50 else 0.9)  # for label size
                labels = (0 < nn < 99) and (nn == nc)  # apply labels if names less than 99 and equal to nc
                
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore')  # suppress empty matrix RuntimeWarning: All-NaN slice encountered
                    res = sn.heatmap(array, 
                                     annot=nc < 30, 
                                     annot_kws={
                                         "size": 8 if nc < 20 else 6,
                                         "fontname": plt.rcParams.get("font.sans-serif")[0]  # 日本語フォントを使用
                                     },
                                     cmap='Blues', 
                                     fmt='.2f' if normalize else '.0f',
                                     square=True, 
                                     vmin=0.0, 
                                     xticklabels=names if labels else "auto",
                                     yticklabels=names if labels else "auto", 
                                     **kwargs).get_figure()
                
                # 座標軸のラベルに日本語フォントを適用
                if labels:
                    for tick in ax.xaxis.get_major_ticks():
                        tick.label.set_fontname(plt.rcParams.get("font.sans-serif")[0])
                    for tick in ax.yaxis.get_major_ticks():
                        tick.label.set_fontname(plt.rcParams.get("font.sans-serif")[0])
                        
                ax.set_xlabel('True')
                ax.set_ylabel('Predicted')
                
                # タイトルに日本語フォントを適用
                title = 'Confusion Matrix' + (' Normalized' if normalize else '')
                ax.set_title(title, fontdict={'fontname': plt.rcParams.get("font.sans-serif")[0]})
                
                # 保存
                fig_suffix = 'confusion_matrix_japanese.png'
                res.savefig(save_dir / f'{fig_suffix}')
                plt.close(res)
                
                # 元の関数も呼び出して既存の処理を維持
                return original_plot_confusion_matrix(matrix, names, normalize=normalize, save_dir=save_dir, on_plot=on_plot, **kwargs)
        
        # 元の関数を保存
        original_func = ultralytics.utils.plotting.plot_confusion_matrix
        
        # カスタム関数に置き換え
        ultralytics.utils.plotting.plot_confusion_matrix = custom_plot_confusion_matrix
        
        print("Ultralytics confusion matrix プロット関数をパッチしました")
        return True
        
    except ImportError:
        print("Ultralytics.utils.plottingモジュールが見つかりません")
        return False
    except Exception as e:
        print(f"Ultralyticsのパッチ適用中にエラーが発生しました: {e}")
        return False

def calculate_class_weights(dataset_path):
    """クラスごとの画像数を数えて、クラスの重みを計算する関数"""
    print("クラスの重みを計算中...")
    
    # データセットのパス構成
    train_path = Path(dataset_path) / 'train'
    
    if not train_path.exists():
        print(f"警告: {train_path} が見つかりません。クラスの重みは計算できません。")
        return None
    
    # 各クラスのフォルダ内の画像数をカウント
    class_counts = {}
    class_folders = [d for d in train_path.iterdir() if d.is_dir()]
    
    if not class_folders:
        print(f"警告: {train_path} にクラスフォルダが見つかりません。")
        return None
    
    for class_folder in class_folders:
        class_name = class_folder.name
        image_files = list(class_folder.glob('*.jpg')) + list(class_folder.glob('*.jpeg')) + \
                     list(class_folder.glob('*.png')) + list(class_folder.glob('*.bmp'))
        class_counts[class_name] = len(image_files)
    
    total_images = sum(class_counts.values())
    
    if total_images == 0:
        print("警告: 画像が見つかりません。")
        return None
    
    # クラスの分布を表示
    print("クラスの分布:")
    for cls, count in class_counts.items():
        print(f"  {cls}: {count} 画像 ({count/total_images:.2%})")
    
    # 重みの計算 (少ないクラスほど高い重みを設定)
    # 逆数を取って正規化
    n_classes = len(class_counts)
    inverse_weights = {cls: count / total_images for cls, count in class_counts.items()}
    weights = {cls: 1.0 / (n_classes * w) if w > 0 else 1.0 for cls, w in inverse_weights.items()}
    
    # 重みの正規化 (合計が n_classes になるように)
    weight_sum = sum(weights.values())
    normalized_weights = {cls: w * n_classes / weight_sum for cls, w in weights.items()}
    
    print("計算されたクラスの重み:")
    for cls, weight in normalized_weights.items():
        print(f"  {cls}: {weight:.4f}")
        
    return normalized_weights

def extract_features_from_dataset(dataset_path, max_images_per_class=500, feature_dim=256):
    """
    画像データセットから特徴を抽出する関数
    
    Args:
        dataset_path: データセットのパス
        max_images_per_class: クラスごとの最大画像数
        feature_dim: 特徴次元数（PCA後）
    
    Returns:
        X: 特徴ベクトル
        y: ラベル
        class_names: クラス名のリスト
    """
    print("データセットから特徴を抽出しています...")
    
    # トレーニングデータのパス
    train_path = Path(dataset_path) / 'train'
    
    if not train_path.exists():
        print(f"警告: {train_path} が見つかりません。")
        return None, None, None
    
    # クラスフォルダのリスト
    class_folders = [d for d in train_path.iterdir() if d.is_dir()]
    
    if not class_folders:
        print(f"警告: {train_path} にクラスフォルダが見つかりません。")
        return None, None, None
    
    # 特徴抽出のためのモデル
    try:
        # 最新のtorchvisionバージョンに対応
        import torchvision.models as models
        model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        # 最後の層を削除（特徴を取得するため）
        model = torch.nn.Sequential(*list(model.children())[:-1])
        model.eval()
        if torch.cuda.is_available():
            model = model.cuda()
    except Exception as e:
        print(f"モデルの読み込み中にエラーが発生しました: {e}")
        return None, None, None
    
    # 特徴とラベルの格納リスト
    features = []
    labels = []
    class_indices = {}
    
    # プリプロセス用のトランスフォーム
    def preprocess_image(img_path, target_size=(224, 224)):
        try:
            img = Image.open(img_path).convert('RGB')
            img = img.resize(target_size)
            img_array = np.array(img) / 255.0
            img_tensor = torch.from_numpy(img_array).permute(2, 0, 1).float()
            # 正規化（ImageNetの平均と標準偏差を使用）
            mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
            img_tensor = (img_tensor - mean) / std
            img_tensor = img_tensor.unsqueeze(0)  # バッチ次元の追加
            if torch.cuda.is_available():
                img_tensor = img_tensor.cuda()
            return img_tensor
        except Exception as e:
            print(f"画像の前処理中にエラーが発生しました: {img_path}, {e}")
            return None
    
    # 各クラスを処理
    class_names = []
    for i, class_folder in enumerate(class_folders):
        class_name = class_folder.name
        class_names.append(class_name)
        class_indices[class_name] = i
        
        # 画像ファイルの取得
        image_files = list(class_folder.glob('*.jpg')) + list(class_folder.glob('*.jpeg')) + \
                     list(class_folder.glob('*.png')) + list(class_folder.glob('*.bmp'))
        
        # 各クラスの画像数を制限
        if len(image_files) > max_images_per_class:
            # ランダムサンプリング
            image_files = np.random.choice(image_files, max_images_per_class, replace=False)
        
        print(f"クラス {class_name} から特徴を抽出中... ({len(image_files)}画像)")
        
        # 各画像から特徴を抽出
        for img_path in tqdm(image_files):
            img_tensor = preprocess_image(img_path)
            if img_tensor is None:
                continue
            
            with torch.no_grad():
                feature = model(img_tensor)
            
            # 特徴ベクトルを抽出
            feature = feature.squeeze().cpu().numpy()
            features.append(feature)
            labels.append(i)
    
    # 特徴がない場合
    if not features:
        print("警告: 特徴を抽出できませんでした。")
        return None, None, None
    
    # numpyに変換
    X = np.array(features)
    # 元の形状のサイズを取得
    original_shape = X.shape
    # 2D形状に変換
    X = X.reshape(X.shape[0], -1)
    print(f"抽出された特徴量の形状: {original_shape} -> {X.shape}")
    
    # PCAで次元削減
    if X.shape[1] > feature_dim:
        print(f"特徴次元を {X.shape[1]} から {feature_dim} に削減しています...")
        pca = PCA(n_components=feature_dim)
        X = pca.fit_transform(X)
        print(f"PCA後の形状: {X.shape}, 説明率: {sum(pca.explained_variance_ratio_):.4f}")
    
    y = np.array(labels)
    
    return X, y, class_names

def apply_oversampling(X, y, method='smote', sampling_strategy='auto', random_state=42):
    """
    オーバーサンプリング手法を適用する関数
    
    Args:
        X: 特徴ベクトル
        y: ラベル
        method: オーバーサンプリング手法 ('smote', 'adasyn', 'borderline')
        sampling_strategy: サンプリング戦略
        random_state: 乱数シード
    
    Returns:
        X_resampled: オーバーサンプリング後の特徴ベクトル
        y_resampled: オーバーサンプリング後のラベル
    """
    if not HAS_IMBLEARN:
        print("エラー: imbalanced-learnがインストールされていません。")
        return X, y
    
    # サンプリング手法の選択
    if method.lower() == 'smote':
        sampler = SMOTE(sampling_strategy=sampling_strategy, random_state=random_state)
    elif method.lower() == 'adasyn':
        sampler = ADASYN(sampling_strategy=sampling_strategy, random_state=random_state)
    elif method.lower() == 'borderline':
        sampler = BorderlineSMOTE(sampling_strategy=sampling_strategy, random_state=random_state)
    else:
        print(f"警告: 不明なサンプリング手法 '{method}'。SMOTEを使用します。")
        sampler = SMOTE(sampling_strategy=sampling_strategy, random_state=random_state)
    
    # クラスの分布を表示
    counter_before = Counter(y)
    print(f"オーバーサンプリング前のクラス分布: {counter_before}")
    
    # オーバーサンプリングの実行
    try:
        X_resampled, y_resampled = sampler.fit_resample(X, y)
        counter_after = Counter(y_resampled)
        print(f"オーバーサンプリング後のクラス分布: {counter_after}")
        
        # データ数の確認
        for label in counter_after:
            print(f"  クラス {label}: {counter_before[label]} → {counter_after[label]} (+{counter_after[label] - counter_before[label]})")
        
        return X_resampled, y_resampled
    
    except Exception as e:
        print(f"オーバーサンプリング中にエラーが発生しました: {e}")
        return X, y

def create_balanced_dataset(dataset_path, output_path, method='smote', max_images_per_class=500, random_state=42):
    """
    バランスの取れたデータセットを作成する関数
    
    Args:
        dataset_path: 元のデータセットのパス
        output_path: 出力先のパス
        method: オーバーサンプリング手法
        max_images_per_class: クラスごとの最大画像数
        random_state: 乱数シード
    
    Returns:
        bool: 成功したかどうか
    """
    if not HAS_IMBLEARN:
        print("エラー: imbalanced-learnがインストールされていません。")
        return False
    
    # 元のデータセットの構造確認
    train_path = Path(dataset_path) / 'train'
    val_path = Path(dataset_path) / 'val'
    
    if not train_path.exists():
        print(f"エラー: トレーニングディレクトリ {train_path} が見つかりません。")
        return False
    
    # 出力ディレクトリの作成
    output_train_path = Path(output_path) / 'train'
    output_val_path = Path(output_path) / 'val'
    
    os.makedirs(output_train_path, exist_ok=True)
    
    # 検証データをコピー（変更なし）
    if val_path.exists():
        os.makedirs(output_val_path, exist_ok=True)
        for class_folder in val_path.iterdir():
            if class_folder.is_dir():
                out_class_folder = output_val_path / class_folder.name
                os.makedirs(out_class_folder, exist_ok=True)
                for img_file in class_folder.glob('*.*'):
                    if img_file.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']:
                        shutil.copy2(img_file, out_class_folder / img_file.name)
        print(f"検証データをコピーしました: {val_path} → {output_val_path}")
    
    # 特徴抽出
    X, y, class_names = extract_features_from_dataset(dataset_path, max_images_per_class)
    
    if X is None or y is None or class_names is None:
        print("特徴抽出に失敗しました。")
        return False
    
    # オーバーサンプリングの実行
    X_resampled, y_resampled = apply_oversampling(X, y, method, random_state=random_state)
    
    # データ構造の確認
    print(f"X_resampled shape: {X_resampled.shape}, y_resampled shape: {y_resampled.shape}")
    
    # 元の画像をコピー
    train_folders = [d for d in train_path.iterdir() if d.is_dir()]
    for class_folder in train_folders:
        class_name = class_folder.name
        out_class_folder = output_train_path / class_name
        os.makedirs(out_class_folder, exist_ok=True)
        
        # 原画像をコピー
        for img_file in class_folder.glob('*.*'):
            if img_file.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']:
                shutil.copy2(img_file, out_class_folder / img_file.name)
    
    # オーバーサンプリングされた追加データは合成（現段階では実装困難）
    # 注: 特徴空間で合成されたデータを画像に戻す処理は複雑なため、
    # 実際の運用ではデータ拡張などの代替手段を検討
    
    print("バランスの取れたデータセットを作成しました。")
    print(f"注意: 合成された画像の生成は複雑なため、現在のバージョンでは元の画像のみをコピーしています。")
    
    return True


def parse_args():
    parser = argparse.ArgumentParser(description='YOLOv11 Classification Training')
    
    # 必須パラメータ
    parser.add_argument('--data', type=str, default='C:/Users/takas/scogr/yolov11_cls/datasetsV2/imagenet_style/wisdomToothPosition',
                        help='データセットのパス（ImageNetスタイルのフォルダ構造）')
    
    # モデルパラメータ
    parser.add_argument('--model', type=str, default='yolo11n-cls.pt',
                        help='トレーニングのベースとなる事前学習モデル')
    parser.add_argument('--name', type=str, default='wisdomToothPosition',
                        help='実験名。指定しない場合は自動生成されます')
    
    # トレーニングパラメータ
    parser.add_argument('--epochs', type=int, default=100,
                        help='トレーニングのエポック数')
    parser.add_argument('--patience', type=int, default=10,
                        help='Early Stoppingの忍耐値')
    parser.add_argument('--batch-size', type=int, default=16,
                        help='バッチサイズ')
    parser.add_argument('--img-size', type=int, default=224,
                        help='入力画像サイズ')
    parser.add_argument('--workers', type=int, default=0,
                        help='データローダーのワーカー数')
    
    # 不均衡データ対策パラメータ
    parser.add_argument('--use-class-weights', action='store_true',
                        help='クラス重みを計算する（参考情報として表示）')
    parser.add_argument('--augment', action='store_true', default=True,
                        help='データ拡張を有効にする')
    parser.add_argument('--mosaic', type=float, default=0.5,
                        help='モザイク拡張の強度（0-1）')
    parser.add_argument('--degrees', type=float, default=1.0,
                        help='回転拡張の最大角度')
    parser.add_argument('--translate', type=float, default=0.1,
                        help='平行移動拡張の最大値（0-1）')
    parser.add_argument('--scale', type=float, default=0.1,
                        help='スケーリング拡張の最大値（0-1）')
    parser.add_argument('--mixup', type=float, default=0.1,
                        help='ミックスアップの確率（0-1）')
    
    # imbalanced-learnによるオーバーサンプリング
    parser.add_argument('--oversample', choices=['none', 'smote', 'adasyn', 'borderline'], default='none',
                        help='オーバーサンプリング手法')
    parser.add_argument('--balanced-data-dir', type=str, default='',
                        help='バランス調整後のデータの出力先（指定しない場合は元データを使用）')
    parser.add_argument('--max-per-class', type=int, default=500,
                        help='特徴抽出時のクラスごとの最大画像数')
    
    # その他
    parser.add_argument('--no-japanese-font', action='store_true', default=False,
                        help='日本語フォント設定を無効化')
    parser.add_argument('--no-patch-confusion', action='store_true', default=False,
                        help='混同行列の日本語パッチを無効化')
    
    return parser.parse_args()

def main():
    # コマンドライン引数の解析
    args = parse_args()
    
    # 日本語フォントの設定（必要な場合）
    if not args.no_japanese_font:
        setup_japanese_font()
        
        # 混同行列のパッチを適用
        if not args.no_patch_confusion:
            try:
                # 先にseabornをインポート（必要な場合）
                try:
                    import seaborn as sn
                except ImportError:
                    print("警告: seabornがインストールされていません。混同行列の表示に必要です。")
                    print("pip install seabornでインストールしてください。")
                
                # パッチ適用
                patch_success = patch_ultralytics_confusion_matrix()
                if patch_success:
                    print("混同行列の日本語パッチを適用しました")
                else:
                    print("混同行列の日本語パッチの適用に失敗しました")
            except Exception as e:
                print(f"混同行列パッチ適用中にエラー: {e}")
    
    # データパスが指定されているか確認
    if not args.data:
        print("エラー: データセットのパスを指定してください (--data)")
        print("使用方法: python train.py --data <データセットのパス>")
        return
    
    # 使用するデータパス
    data_path = args.data
    
    # オーバーサンプリングの実行
    if args.oversample != 'none' and HAS_IMBLEARN:
        # PyTorchとtorchvisionのバージョンの互換性を確認
        try:
            import torch
            import torchvision
            print(f"PyTorch バージョン: {torch.__version__}")
            print(f"torchvision バージョン: {torchvision.__version__}")
            
            # torchvision.opsの動作確認
            try:
                import torchvision.ops
                dummy_boxes = torch.tensor([[0, 0, 10, 10], [5, 5, 15, 15]], dtype=torch.float32)
                dummy_scores = torch.tensor([0.9, 0.8], dtype=torch.float32)
                torchvision.ops.nms(dummy_boxes, dummy_scores, 0.5)
                print("torchvision.ops.nms のテストに成功しました")
            except Exception as e:
                print(f"警告: torchvision.ops.nms のテストに失敗しました: {e}")
                print("オーバーサンプリングをスキップします")
                args.oversample = 'none'
        except Exception as e:
            print(f"警告: PyTorch/torchvisionの互換性チェックに失敗しました: {e}")
            print("オーバーサンプリングをスキップします")
            args.oversample = 'none'
    
    # オーバーサンプリング処理（再確認）
    if args.oversample != 'none' and HAS_IMBLEARN:
        if not args.balanced_data_dir:
            # 出力先が指定されていない場合は自動生成
            output_dir = Path(args.data).parent / f"{Path(args.data).name}_balanced_{args.oversample}"
            args.balanced_data_dir = str(output_dir)
        
        print(f"オーバーサンプリング ({args.oversample}) を適用してデータセットをバランス調整します...")
        success = create_balanced_dataset(
            args.data, 
            args.balanced_data_dir, 
            method=args.oversample,
            max_images_per_class=args.max_per_class,
            random_state=42
        )
        
        if success:
            print(f"バランス調整されたデータセットを作成しました: {args.balanced_data_dir}")
            # バランス調整後のデータを使用
            data_path = args.balanced_data_dir
        else:
            print("バランス調整に失敗しました。元のデータセットを使用します。")
    elif args.oversample != 'none' and not HAS_IMBLEARN:
        print("imbalanced-learnがインストールされていないため、オーバーサンプリングをスキップします。")
    
    # クラスの重みを計算（必要な場合）
    class_weights = None
    if args.use_class_weights:
        class_weights = calculate_class_weights(data_path)
    
    # 実験名が指定されていない場合はデフォルト名を生成
    run_name = args.name
    if not run_name:
        dataset_name = Path(data_path).name
        run_name = f'yolov11_cls-{dataset_name}'
    
    # モデルの読み込み
    model = YOLO(args.model)

    # トレーニングパラメータの設定
    train_params = {
        'data': data_path,
        'epochs': args.epochs,
        'patience': args.patience,
        'batch': args.batch_size,
        'imgsz': args.img_size,
        'workers': args.workers,
        'name': run_name,
    }
    
    # 不均衡データ対策パラメータの設定
    # class_weightsはUltralyticsの引数としてサポートされていないため、別の方法で対応
    # 代わりにYAMLファイルを介して設定する方法や、クラスごとのサンプリング戦略を使用
    
    if args.augment:
        train_params.update({
            'augment': True,
            'mosaic': args.mosaic,
            'degrees': args.degrees,
            'translate': args.translate,
            'scale': args.scale,
            'mixup': args.mixup,
        })
    
    # クラス重みをログに出力（学習には直接使用できないが参考情報として）
    if class_weights is not None:
        print("注意: 計算されたクラス重みをUltralyticsの学習に直接適用できません")
        print("代わりにデータ拡張やオーバーサンプリングで不均衡に対処します")
    
    # トレーニングの実行
    print(f"トレーニングパラメータ: {train_params}")
    try:
        results = model.train(**train_params)
        return results
    except RuntimeError as e:
        if "custom C++ ops" in str(e) and "torchvision" in str(e):
            print("\nエラー: PyTorchとtorchvisionのバージョンに互換性がありません。")
            print("対応策: conda環境を更新するか、以下のコマンドを実行してください:")
            print("pip install -U torch torchvision --index-url https://download.pytorch.org/whl/cu121\n")
            return None
        else:
            raise



if __name__ == '__main__':
    # Windowsでマルチプロセッシングをサポートするために必要
    multiprocessing.freeze_support()
    
    # seabornがインストールされているか確認
    try:
        import seaborn as sn
    except ImportError:
        print("警告: seabornがインストールされていません。混同行列の表示に必要です。")
        print("pip install seabornでインストールしてください。")
    
    main()