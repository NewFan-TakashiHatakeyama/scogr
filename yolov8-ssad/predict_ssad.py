import torch
import cv2
import numpy as np
from ultralytics import YOLO
from pathlib import Path
import yaml
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import base64
from PIL import Image
import io
import random
import os
import glob
import plotly.io as pio
import argparse

# SSADモデルの読み込み
def load_model(weights_path):
    model = YOLO(weights_path)
    model.task = "ssad"
    return model

# クラスIDから歯番号へのマッピングを読み込む
def load_class_names(yaml_path):
    # 歯番号のマッピング（クラスID -> 歯番号）
    # 標準的なFDI表記で定義（上顎右側: 11-18, 上顎左側: 21-28, 下顎左側: 31-38, 下顎右側: 41-48）
    tooth_mapping = {
        0: '11', 1: '12', 2: '13', 3: '14', 4: '15', 5: '16', 6: '17', 7: '18',
        8: '21', 9: '22', 10: '23', 11: '24', 12: '25', 13: '26', 14: '27', 15: '28',
        16: '31', 17: '32', 18: '33', 19: '34', 20: '35', 21: '36', 22: '37', 23: '38',
        24: '41', 25: '42', 26: '43', 27: '44', 28: '45', 29: '46', 30: '47', 31: '48'
    }
    
    # 直接内部マッピングを返す（YAMLファイルよりも優先）
    return {str(k): v for k, v in tooth_mapping.items()}

# 画像をBase64エンコードに変換する関数
def numpy_to_base64(img_array):
    """NumPy配列をbase64エンコードした文字列に変換"""
    img = Image.fromarray(img_array)
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    img_str = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{img_str}"

# 画像の前処理
def preprocess_image(image_path, target_size=640):
    # 画像の読み込み
    img = cv2.imread(image_path)
    
    if img is None:
        raise ValueError(f"画像を読み込めませんでした: {image_path}")
    
    # オリジナル画像のサイズを保存（可視化用）
    original_img = img.copy()
    
    # 画像のアスペクト比を維持しながらリサイズ
    h, w = img.shape[:2]
    ratio = target_size / max(h, w)
    new_h, new_w = int(h * ratio), int(w * ratio)
    resized_img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    
    # パディングの追加
    padded_img = np.zeros((target_size, target_size, 3), dtype=np.uint8)
    padded_img[:new_h, :new_w, :] = resized_img
    
    # 正規化 [0-255] → [0-1]
    normalized_img = padded_img.astype(np.float32) / 255.0
    
    return original_img, normalized_img, (ratio, (0, 0))  # 元画像、前処理済み画像、変換パラメータ

# 予測関数
def predict(model, image_path, conf_threshold=0.5):
    # 画像の前処理
    original_img, preprocessed_img, transform_params = preprocess_image(image_path)
    
    # PyTorchテンソルに変換
    img_tensor = torch.from_numpy(preprocessed_img).permute(2, 0, 1).unsqueeze(0)
    
    # モデルに入力して推論
    results = model(img_tensor, conf=conf_threshold, task="ssad")
    
    return results, original_img, transform_params

# Plotlyを使用した結果の可視化
def visualize_results_plotly(original_img, results, class_names, transform_params=None, output_path=None):
    # 変換パラメータの取得
    ratio, pad = transform_params if transform_params else (1.0, (0, 0))
    
    # BGRからRGBに変換
    img_rgb = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
    
    # 画像をBase64エンコードされた文字列に変換
    img_base64 = numpy_to_base64(img_rgb)
    
    # 検出結果の取得
    boxes = results[0].boxes
    
    # 検出結果をDataFrameに変換
    detection_data = []
    for box in boxes:
        # バウンディングボックスの座標（元の画像サイズに変換）
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
        
        # スケーリングを元に戻す
        if transform_params:
            x1, y1 = x1 / ratio - pad[0], y1 / ratio - pad[1]
            x2, y2 = x2 / ratio - pad[0], y2 / ratio - pad[1]
        
        # クラスとconfidence
        cls = int(box.cls[0])
        conf = float(box.conf[0])
        
        # 歯番号を取得
        tooth_number = class_names.get(str(cls), str(cls))
        
        # 検出データを追加
        detection_data.append({
            'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2,
            'cls': cls, 'tooth_number': tooth_number, 'confidence': conf
        })
    
    # Plotlyでの可視化
    fig = go.Figure()
    
    # 背景画像を追加
    fig.add_layout_image(
        dict(
            source=img_base64,  # Base64エンコードされた画像を使用
            xref="x",
            yref="y",
            x=0,
            y=0,
            sizex=img_rgb.shape[1],
            sizey=img_rgb.shape[0],
            sizing="stretch",
            opacity=1,
            layer="below"
        )
    )
    
    # カラーマップの定義（歯の位置ごとに色分け）
    # 上顎右側(1x): 赤色系, 上顎左側(2x): 青色系, 下顎左側(3x): 緑色系, 下顎右側(4x): 紫色系
    color_groups = {
        '1': 'rgba(255, 0, 0,', '2': 'rgba(0, 0, 255,', 
        '3': 'rgba(0, 128, 0,', '4': 'rgba(128, 0, 128,'
    }
    
    # 検出された各歯に対してボックスとラベルを追加
    for det in detection_data:
        # 歯番号の最初の数字に基づいて色を割り当て
        quadrant = det['tooth_number'][0]
        color_base = color_groups.get(quadrant, 'rgba(128, 128, 128,')
        
        # ボックスを追加
        fig.add_shape(
            type="rect",
            x0=det['x1'], y0=det['y1'], x1=det['x2'], y1=det['y2'],
            line=dict(color=f"{color_base} 1.0)", width=2),
            fillcolor=f"{color_base} 0.2)"
        )
        
        # ラベルを追加（ホバー時に表示）
        fig.add_trace(go.Scatter(
            x=[(det['x1'] + det['x2']) / 2],
            y=[det['y1']],
            mode="markers+text",
            marker=dict(size=10, color=f"{color_base} 0.8)"),
            text=f"#{det['tooth_number']}",
            textposition="top center",
            textfont=dict(color="white", size=10, family="Arial Black"),
            hoverinfo="text",
            hovertext=f"歯番号: {det['tooth_number']}<br>信頼度: {det['confidence']:.2f}",
            showlegend=False
        ))
    
    # レイアウトの設定
    fig.update_layout(
        autosize=True,
        width=img_rgb.shape[1],
        height=img_rgb.shape[0],
        margin=dict(l=0, r=0, t=0, b=0),
        xaxis=dict(
            showgrid=False,
            showticklabels=False,
            range=[0, img_rgb.shape[1]]
        ),
        yaxis=dict(
            showgrid=False,
            showticklabels=False,
            range=[img_rgb.shape[0], 0],
            scaleanchor="x",
            scaleratio=1
        ),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        title=dict(
            text="歯科検出結果",
            x=0.5,
            y=0.98,
            font=dict(size=24)
        ),
        updatemenus=[dict(
            type="buttons",
            buttons=[dict(
                label="Teeth Info",
                method="update",
                args=[{"visible": [True] * len(detection_data)}]
            )],
            direction="down",
            showactive=True,
            x=0.1,
            y=1.15,
        )]
    )
    
    # 凡例を追加（歯の位置ごとにグループ化）
    legend_items = {
        '上顎右側(11-18)': 'rgba(255, 0, 0, 0.7)',
        '上顎左側(21-28)': 'rgba(0, 0, 255, 0.7)',
        '下顎左側(31-38)': 'rgba(0, 128, 0, 0.7)',
        '下顎右側(41-48)': 'rgba(128, 0, 128, 0.7)'
    }
    
    legend_x = img_rgb.shape[1] - 200
    legend_y = 50
    
    for idx, (label, color) in enumerate(legend_items.items()):
        fig.add_shape(
            type="rect",
            x0=legend_x, y0=legend_y + idx*30, x1=legend_x + 20, y1=legend_y + idx*30 + 20,
            line=dict(color=color, width=2),
            fillcolor=color
        )
        fig.add_trace(go.Scatter(
            x=[legend_x + 30],
            y=[legend_y + idx*30 + 10],
            mode="text",
            text=label,
            textposition="middle right",
            showlegend=False
        ))
    
    # 検出結果の集計
    result_summary = f"検出歯数: {len(detection_data)}本"
    fig.add_annotation(
        x=10, y=30,
        xref="x", yref="y",
        text=result_summary,
        showarrow=False,
        font=dict(size=14, color="black", family="Arial"),
        bgcolor="rgba(255, 255, 255, 0.7)",
        bordercolor="black",
        borderwidth=1,
        borderpad=4,
        align="left"
    )
    
    # 結果の保存
    if output_path:
        try:
            # HTMLとして保存
            html_path = output_path.replace('.png', '.html')
            fig.write_html(html_path)
            print(f"インタラクティブな結果をHTMLとして保存しました: {html_path}")
            
            # 画像としても保存
            fig.write_image(output_path)
            print(f"静的な画像として保存しました: {output_path}")
        except Exception as e:
            print(f"ファイル保存中にエラーが発生しました: {e}")
            print("HTMLのみ保存を試みます...")
            try:
                fig.write_html(html_path)
                print(f"インタラクティブな結果をHTMLとして保存しました: {html_path}")
            except Exception as e2:
                print(f"HTML保存中にもエラーが発生しました: {e2}")
    
    # 表示
    try:
        # オプションで表示（開発中のみ使用）
        # fig.show()
        # 明示的にfigureをクローズして次の処理に移行
        pio.renderers.default = None  # デフォルトレンダラーを無効化
    except Exception as e:
        print(f"図の後処理中にエラーが発生しました: {e}")

# YOLOラベルファイルを読み込む関数
def load_yolo_labels(label_path, img_width, img_height):
    """
    YOLOラベルファイルから正解ラベル情報を読み込む
    
    Args:
        label_path: ラベルファイルのパス
        img_width: 元画像の幅
        img_height: 元画像の高さ
        
    Returns:
        list: (クラスID, x1, y1, x2, y2)のリスト
    """
    boxes = []
    
    try:
        with open(label_path, 'r') as f:
            for line in f:
                data = line.strip().split()
                if len(data) >= 5:
                    # クラスIDとバウンディングボックス情報の取得
                    cls_id = int(data[0])
                    
                    # YOLOフォーマット（中心x, 中心y, 幅, 高さ）から絶対座標に変換
                    x_center, y_center = float(data[1]), float(data[2])
                    width, height = float(data[3]), float(data[4])
                    
                    # 絶対座標（左上、右下）の計算
                    x1 = int((x_center - width/2) * img_width)
                    y1 = int((y_center - height/2) * img_height)
                    x2 = int((x_center + width/2) * img_width)
                    y2 = int((y_center + height/2) * img_height)
                    
                    boxes.append((cls_id, x1, y1, x2, y2))
    except Exception as e:
        print(f"ラベルファイルの読み込みに失敗しました: {e}")
    
    return boxes

# ランダムな画像とラベルの選択
def select_random_samples(image_dir, label_dir, num_samples=10):
    """
    指定ディレクトリからランダムに画像とラベルのペアを選択
    
    Args:
        image_dir: 画像ディレクトリのパス
        label_dir: ラベルディレクトリのパス
        num_samples: 選択するサンプル数
        
    Returns:
        list: (画像パス, ラベルパス)のリスト
    """
    # 画像ファイルの一覧を取得
    image_files = glob.glob(os.path.join(image_dir, "*.jpg")) + glob.glob(os.path.join(image_dir, "*.png"))
    
    if not image_files:
        print(f"警告: 画像が見つかりません: {image_dir}")
        return []
    
    # ランダムに選択
    if len(image_files) > num_samples:
        selected_images = random.sample(image_files, num_samples)
    else:
        selected_images = image_files
        print(f"警告: 要求されたサンプル数({num_samples})よりも少ない画像({len(image_files)})しか見つかりませんでした")
    
    # 対応するラベルファイルのパスを取得
    samples = []
    for img_path in selected_images:
        img_basename = os.path.basename(img_path)
        img_name, _ = os.path.splitext(img_basename)
        label_path = os.path.join(label_dir, f"{img_name}.txt")
        
        if os.path.exists(label_path):
            samples.append((img_path, label_path))
        else:
            print(f"警告: ラベルファイルが見つかりません: {label_path}")
    
    return samples

# Plotlyを使用した予測結果と正解ラベルの比較可視化
def visualize_comparison(original_img, results, true_boxes, class_names, transform_params=None, output_path=None):
    """
    予測結果と正解ラベルの比較可視化
    
    Args:
        original_img: 元画像
        results: 予測結果
        true_boxes: 正解ラベル（クラスID, x1, y1, x2, y2のリスト）
        class_names: クラス名のマッピング
        transform_params: 変換パラメータ
        output_path: 出力ファイルパス
    """
    # 変換パラメータの取得
    ratio, pad = transform_params if transform_params else (1.0, (0, 0))
    
    # BGRからRGBに変換
    img_rgb = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
    
    # 画像をBase64エンコードされた文字列に変換
    img_base64 = numpy_to_base64(img_rgb)
    
    # 検出結果の取得
    boxes = results[0].boxes
    
    # 検出結果をDataFrameに変換
    detection_data = []
    for box in boxes:
        # バウンディングボックスの座標（元の画像サイズに変換）
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
        
        # スケーリングを元に戻す
        if transform_params:
            x1, y1 = x1 / ratio - pad[0], y1 / ratio - pad[1]
            x2, y2 = x2 / ratio - pad[0], y2 / ratio - pad[1]
        
        # クラスとconfidence
        cls = int(box.cls[0])
        conf = float(box.conf[0])
        
        # 歯番号を取得
        tooth_number = class_names.get(str(cls), str(cls))
        
        # 検出データを追加
        detection_data.append({
            'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2,
            'cls': cls, 'tooth_number': tooth_number, 'confidence': conf,
            'type': '予測'
        })
    
    # 正解ラベルをDataFrameに変換
    for cls_id, x1, y1, x2, y2 in true_boxes:
        tooth_number = class_names.get(str(cls_id), str(cls_id))
        detection_data.append({
            'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2,
            'cls': cls_id, 'tooth_number': tooth_number, 'confidence': 1.0,
            'type': '正解'
        })
    
    # Plotlyでの可視化
    fig = go.Figure()
    
    # 背景画像を追加
    fig.add_layout_image(
        dict(
            source=img_base64,
            xref="x",
            yref="y",
            x=0,
            y=0,
            sizex=img_rgb.shape[1],
            sizey=img_rgb.shape[0],
            sizing="stretch",
            opacity=1,
            layer="below"
        )
    )
    
    # カラーマップの定義（歯の位置ごとに色分け）
    color_groups = {
        '1': 'rgba(255, 0, 0,', '2': 'rgba(0, 0, 255,', 
        '3': 'rgba(0, 128, 0,', '4': 'rgba(128, 0, 128,'
    }
    
    # 予測と正解を異なるスタイルで表示
    style_map = {
        '予測': {'line_dash': 'solid', 'line_width': 2, 'opacity': 0.7},
        '正解': {'line_dash': 'dash', 'line_width': 3, 'opacity': 0.9}
    }
    
    # 検出された各歯に対してボックスとラベルを追加
    for det in detection_data:
        # 歯番号の最初の数字に基づいて色を割り当て
        quadrant = det['tooth_number'][0]
        color_base = color_groups.get(quadrant, 'rgba(128, 128, 128,')
        
        # 予測か正解かによってスタイルを変更
        style = style_map[det['type']]
        
        # ボックスを追加
        fig.add_shape(
            type="rect",
            x0=det['x1'], y0=det['y1'], x1=det['x2'], y1=det['y2'],
            line=dict(
                color=f"{color_base} 1.0)", 
                width=style['line_width'],
                dash=style['line_dash']
            ),
            fillcolor=f"{color_base} {0.1 if det['type'] == '予測' else 0.2})"
        )
        
        # ラベルを追加（ホバー時に表示）
        fig.add_trace(go.Scatter(
            x=[(det['x1'] + det['x2']) / 2],
            y=[det['y1'] - 10 if det['type'] == '予測' else det['y1'] - 5],
            mode="markers+text",
            marker=dict(size=8, color=f"{color_base} 0.8)"),
            text=f"{det['type']}#{det['tooth_number']}",
            textposition="top center",
            textfont=dict(
                color="white" if det['type'] == '予測' else "yellow", 
                size=9, 
                family="Arial Black"
            ),
            hoverinfo="text",
            hovertext=(
                f"歯番号: {det['tooth_number']}<br>"
                f"信頼度: {det['confidence']:.2f}<br>"
                f"タイプ: {det['type']}"
            ),
            showlegend=False
        ))
    
    # レイアウトの設定
    fig.update_layout(
        autosize=True,
        width=img_rgb.shape[1],
        height=img_rgb.shape[0],
        margin=dict(l=0, r=0, t=30, b=0),
        xaxis=dict(
            showgrid=False,
            showticklabels=False,
            range=[0, img_rgb.shape[1]]
        ),
        yaxis=dict(
            showgrid=False,
            showticklabels=False,
            range=[img_rgb.shape[0], 0],
            scaleanchor="x",
            scaleratio=1
        ),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        title=dict(
            text="予測結果と正解ラベルの比較",
            x=0.5,
            y=0.98,
            font=dict(size=18)
        )
    )
    
    # 凡例を追加
    legend_items = [
        {'label': '予測ボックス', 'color': 'rgba(255, 0, 0, 0.7)', 'dash': 'solid'},
        {'label': '正解ボックス', 'color': 'rgba(255, 0, 0, 0.7)', 'dash': 'dash'},
    ]
    
    legend_x = img_rgb.shape[1] - 150
    legend_y = 30
    
    for idx, item in enumerate(legend_items):
        # 凡例のボックス
        fig.add_shape(
            type="line",
            x0=legend_x, y0=legend_y + idx*30 + 10,
            x1=legend_x + 30, y1=legend_y + idx*30 + 10,
            line=dict(
                color=item['color'],
                width=3,
                dash=item['dash']
            )
        )
        
        # 凡例のテキスト
        fig.add_trace(go.Scatter(
            x=[legend_x + 40],
            y=[legend_y + idx*30 + 10],
            mode="text",
            text=item['label'],
            textposition="middle right",
            showlegend=False
        ))
    
    # 結果の保存
    if output_path:
        try:
            # HTMLとして保存
            html_path = output_path.replace('.png', '.html')
            fig.write_html(html_path)
            print(f"インタラクティブな結果をHTMLとして保存しました: {html_path}")
            
            # 画像としても保存
            fig.write_image(output_path)
            print(f"静的な画像として保存しました: {output_path}")
            pio.renderers.default = None
        except Exception as e:
            print(f"ファイル保存中にエラーが発生しました: {e}")
            print("HTMLのみ保存を試みます...")
            try:
                fig.write_html(html_path)
                print(f"インタラクティブな結果をHTMLとして保存しました: {html_path}")
            except Exception as e2:
                print(f"HTML保存中にもエラーが発生しました: {e2}")
    
    

# メイン関数
def main():
    # コマンドライン引数の処理
    parser = argparse.ArgumentParser(description='YOLOv8-SSAD: 歯科画像における歯の検出と番号付け')
    parser.add_argument('--weights', type=str, default='./weights/best.pt', 
                        help='モデルの重みファイルのパス')
    parser.add_argument('--source', type=str, default='sample.png',
                        help='入力画像のパスまたはディレクトリ')
    parser.add_argument('--output_dir', type=str, default='results',
                        help='出力結果を保存するディレクトリ')
    parser.add_argument('--yaml', type=str, default='dentex_enum32.yaml',
                        help='クラス名定義のYAMLファイル')
    parser.add_argument('--conf', type=float, default=0.5,
                        help='検出信頼度の閾値')
    parser.add_argument('--random', action='store_true',
                        help='バリデーションデータからランダムに10件サンプルを選んで推論する')
    parser.add_argument('--val_image_dir', type=str, 
                        default='./dentex_dataset/yolo/enumeration32/images/val',
                        help='バリデーション画像のディレクトリ')
    parser.add_argument('--val_label_dir', type=str,
                        default='./dentex_dataset/yolo/enumeration32/labels/val',
                        help='バリデーションラベルのディレクトリ')
    parser.add_argument('--num_samples', type=int, default=10,
                        help='ランダムサンプリングする数')
    
    args = parser.parse_args()
    
    # 出力ディレクトリの作成
    os.makedirs(args.output_dir, exist_ok=True)
    
    # クラス名の読み込み
    class_names = load_class_names(args.yaml)
    
    # モデルのロード
    model = load_model(args.weights)
    
    if args.random:
        # バリデーションデータからランダムに指定件数選んで推論
        print(f"バリデーションデータからランダムに{args.num_samples}件選んで推論します...")
        
        # ランダムにサンプルを選択
        samples = select_random_samples(args.val_image_dir, args.val_label_dir, args.num_samples)
        
        print(f"選択されたサンプル数: {len(samples)}")
        
        # 各サンプルに対して推論と比較
        for i, (img_path, label_path) in enumerate(samples):
            print(f"サンプル {i+1}/{len(samples)}: {os.path.basename(img_path)}")
            img = cv2.imread(img_path)
            if img is None:
                print(f"画像の読み込みに失敗: {img_path}")
                continue
                
            height, width = img.shape[:2]
                
            # 正解ラベルの読み込み
            true_boxes = load_yolo_labels(label_path, width, height)
            print(f"正解ラベル数: {len(true_boxes)}")
                
            # 推論実行
            results, original_img, transform_params = predict(model, img_path, conf_threshold=args.conf)
                
            # 出力ファイルパス
            base_name = os.path.basename(img_path)
            output_path = os.path.join(args.output_dir, f"result_{i+1}_{base_name}")
                
            # 予測結果と正解ラベルの比較可視化
            visualize_comparison(
                original_img, results, true_boxes, class_names, 
                transform_params, output_path
            )
        
        print("ランダムサンプルの推論が完了しました！")
    
    else:
        # 単一画像または指定ディレクトリ内の画像で推論
        source_path = Path(args.source)
        
        # ディレクトリかファイルかを判断
        if source_path.is_dir():
            # ディレクトリ内の全画像に対して推論
            image_files = [f for f in source_path.glob('*') if f.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']]
            print(f"{len(image_files)}枚の画像で推論します...")
            
            for i, img_path in enumerate(image_files):
                print(f"画像 {i+1}/{len(image_files)}: {img_path.name}")
                
                # 推論実行（前処理を含む）
                results, original_img, transform_params = predict(model, str(img_path), conf_threshold=args.conf)
                
                # 結果の可視化と保存
                output_path = os.path.join(args.output_dir, f"result_{img_path.stem}.png")
                visualize_results_plotly(original_img, results, class_names, transform_params, output_path)
        else:
            # 単一画像での推論
            print(f"画像: {args.source} で推論します...")
            
            # 推論実行（前処理を含む）
            results, original_img, transform_params = predict(model, args.source, conf_threshold=args.conf)
            
            # 結果の可視化と保存
            output_path = os.path.join(args.output_dir, f"result_{source_path.stem}.png")
            visualize_results_plotly(original_img, results, class_names, transform_params, output_path)
        
        print("推論が完了しました！")

if __name__ == "__main__":
    main() 