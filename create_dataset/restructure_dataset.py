import json
import logging
import os
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
from typing import Optional, Dict, List, Tuple, Any, Union

# ロガーの設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

mapping = {0:11,
          1:12,
          2:13,
          3:14,
          4:15,
          5:16,
          6:17,
          7:18,
          8:21,
          9:22,
          10:23,
          11:24,
          12:25,
          13:26,
          14:27,
          15:28,
          16:31,
          17:32,
          18:33,
          19:34,
          20:35,
          21:36,
          22:37,
          23:38,
          24:41,
          25:42,
          26:43,
          27:44,
          28:45,
          29:46,
          30:47,
          31:48
          }

class FileProcessingError(Exception):
    """ファイル処理中のエラーを表すカスタム例外"""
    pass

def safe_rename_file(old_path: Path, new_path: Path) -> bool:
    """安全にファイルをリネームする

    Args:
        old_path (Path): 元のファイルパス
        new_path (Path): 新しいファイルパス

    Returns:
        bool: リネームが成功したかどうか

    Raises:
        FileProcessingError: ファイル操作に失敗した場合
    """
    try:
        if new_path.exists():
            logger.warning(f"Target file already exists: {new_path}")
            return False
        
        old_path.rename(new_path)
        return True
    except Exception as e:
        raise FileProcessingError(f"Failed to rename {old_path} to {new_path}: {str(e)}")

def get_new_name(filename: str) -> Optional[str]:
    """ファイル名から新しい名前を抽出する

    Args:
        filename (str): 元のファイル名

    Returns:
        Optional[str]: 新しいファイル名、抽出できない場合はNone
    """
    try:
        return filename.split("-")[1]
    except IndexError:
        return None

def rename_image_files(dataset_dir: str) -> None:
    """画像ファイルをリネームする

    Args:
        dataset_dir (str): データセットのディレクトリパス

    Raises:
        FileProcessingError: ファイル処理に失敗した場合
    """
    try:
        dataset_dir = Path(dataset_dir)
        images_dir = dataset_dir / "images"

        if not images_dir.exists():
            raise FileProcessingError(f"Images directory not found: {images_dir}")

        success_count = 0
        skip_count = 0
        error_count = 0

        for img_path in images_dir.glob("*"):
            try:
                new_name = get_new_name(img_path.name)
                if new_name is None:
                    logger.warning(f"Could not extract new name from: {img_path.name}")
                    skip_count += 1
                    continue

                if safe_rename_file(img_path, images_dir / new_name):
                    success_count += 1
                else:
                    skip_count += 1

            except Exception as e:
                logger.error(f"Error processing {img_path}: {str(e)}")
                error_count += 1

        logger.info(f"Image renaming completed. Success: {success_count}, Skipped: {skip_count}, Errors: {error_count}")

    except Exception as e:
        raise FileProcessingError(f"Failed to process image files: {str(e)}")

def rename_label_files(dataset_dir: str) -> None:
    """ラベルファイルをリネームする

    Args:
        dataset_dir (str): データセットのディレクトリパス

    Raises:
        FileProcessingError: ファイル処理に失敗した場合
    """
    try:
        dataset_dir = Path(dataset_dir)
        labels_dir = dataset_dir / "labels"

        if not labels_dir.exists():
            raise FileProcessingError(f"Labels directory not found: {labels_dir}")

        success_count = 0
        skip_count = 0
        error_count = 0

        for label_path in labels_dir.glob("*"):
            try:
                new_name = get_new_name(label_path.name)
                if new_name is None:
                    logger.warning(f"Could not extract new name from: {label_path.name}")
                    skip_count += 1
                    continue

                if safe_rename_file(label_path, labels_dir / new_name):
                    success_count += 1
                else:
                    skip_count += 1

            except Exception as e:
                logger.error(f"Error processing {label_path}: {str(e)}")
                error_count += 1

        logger.info(f"Label renaming completed. Success: {success_count}, Skipped: {skip_count}, Errors: {error_count}")

    except Exception as e:
        raise FileProcessingError(f"Failed to process label files: {str(e)}")

def draw_polygon(draw: ImageDraw.Draw, points: List[Tuple[float, float]], color: str, width: int = 3) -> None:
    """ポリゴンを描画する関数

    Args:
        draw: ImageDrawオブジェクト
        points: ポリゴンの頂点座標のリスト [(x1,y1), (x2,y2), ...]
        color: 描画色
        width: 線の太さ
    """
    # ポリゴンの辺を描画
    n_points = len(points)
    for i in range(n_points):
        start = points[i]
        end = points[(i + 1) % n_points]  # 最後の点は最初の点と結ぶ
        draw.line([start, end], fill=color, width=width)

def get_bbox_from_polygon(points: List[Tuple[float, float]]) -> Tuple[float, float, float, float]:
    """ポリゴン頂点から外接矩形（バウンディングボックス）を計算

    Args:
        points: ポリゴンの頂点座標のリスト [(x1,y1), (x2,y2), ...]

    Returns:
        tuple: (x1, y1, x2, y2) 左上と右下の座標
    """
    x_coords = [p[0] for p in points]
    y_coords = [p[1] for p in points]
    return (min(x_coords), min(y_coords), max(x_coords), max(y_coords))

def create_folder_name(base_name: str, data_dict: Dict, key_field: str, 
                      secondary_field: Optional[str] = None, 
                      normal_value: str = "正常") -> Tuple[str, str]:
    """複数の属性値からフォルダ名を生成する

    Args:
        base_name (str): ベースディレクトリ名
        data_dict (dict): データ辞書
        key_field (str): 主属性のキー
        secondary_field (str, optional): 副属性のキー
        normal_value (str, optional): データがない場合のデフォルト値

    Returns:
        tuple: (フォルダパス, フォルダ名)
    """
    # 主キーの値を取得
    main_value = data_dict.get(key_field)
    
    # 主キーの値がなく、副キーもない場合は正常
    if main_value is None and (secondary_field is None or not data_dict.get(secondary_field, False)):
        return base_name, normal_value
    
    # 主キーの値があり、副キーの値がある場合は組み合わせたフォルダ名
    if main_value is not None and secondary_field and data_dict.get(secondary_field, False):
        folder_name = f"{main_value}_{secondary_field}"
    # 主キーの値があり、副キーの値がない場合は主キーのみのフォルダ名
    elif main_value is not None:
        folder_name = main_value
    # 主キーの値がなく、副キーの値がある場合は副キーのみのフォルダ名
    elif secondary_field and data_dict.get(secondary_field, False):
        folder_name = secondary_field
    # どちらもない場合は正常とする
    else:
        folder_name = normal_value
        
    return base_name, folder_name

def initialize_output_directories(output_dir: Path) -> Dict[str, Path]:
    """出力ディレクトリを初期化する

    Args:
        output_dir (Path): 基本出力ディレクトリのパス

    Returns:
        Dict[str, Path]: 各種出力ディレクトリのパスを含む辞書
    """
    # 基本出力ディレクトリの作成
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # バウンディングボックス可視化用ディレクトリ
    bbox_vis_dir = output_dir / "bbox_visualization"
    bbox_vis_dir.mkdir(parents=True, exist_ok=True)
    
    # 各属性用のベースディレクトリを作成
    status_dir = output_dir / "status"
    restoration_dir = output_dir / "restoration"
    pathology_dir = output_dir / "pathology"
    wisdom_tooth_dir = output_dir / "wisdomToothPosition"
    bridge_role_dir = output_dir / "bridgeRole"
    
    for base_dir in [status_dir, restoration_dir, pathology_dir, wisdom_tooth_dir, bridge_role_dir]:
        base_dir.mkdir(parents=True, exist_ok=True)
    
    return {
        "bbox_vis": bbox_vis_dir,
        "status": status_dir,
        "restoration": restoration_dir,
        "pathology": pathology_dir,
        "wisdom_tooth": wisdom_tooth_dir,
        "bridge_role": bridge_role_dir
    }

def find_image_path(images_dir: Path, photo_id: str) -> Optional[Path]:
    """写真IDに対応する画像ファイルを探す

    Args:
        images_dir (Path): 画像ディレクトリのパス
        photo_id (str): 写真ID

    Returns:
        Optional[Path]: 画像ファイルのパス、見つからなかった場合はNone
    """
    for ext in (".jpg", ".jpeg", ".png"):
        img_path = images_dir / f"{photo_id}{ext}"
        if img_path.exists():
            return img_path
    return None

def setup_drawing_font(img_size: Tuple[int, int]) -> Optional[ImageFont.FreeTypeFont]:
    """描画用フォントをセットアップする

    Args:
        img_size (Tuple[int, int]): 画像サイズ (幅, 高さ)

    Returns:
        Optional[ImageFont.FreeTypeFont]: フォントオブジェクト、読み込めない場合はNone
    """
    try:
        # サイズ調整（画像サイズに応じて適切に変更可能）
        W, H = img_size
        font_size = max(12, min(W, H) // 30)
        return ImageFont.truetype("arial.ttf", font_size)
    except IOError:
        logger.warning("フォントが見つかりません。テキストなしでバウンディングボックスを描画します。")
        return None

def get_color_for_condition(condition: Optional[str]) -> str:
    """歯の状態に応じた色を返す

    Args:
        condition (Optional[str]): 歯の状態

    Returns:
        str: 対応する色のコード
    """
    if condition == "残存歯":
        return "green"
    elif condition == "欠損歯":
        return "red"
    else:
        return "yellow"

def parse_yolo_label(line: str, img_size: Tuple[int, int]) -> Tuple[float, List[float], List[Tuple[int, int]]]:
    """YOLO形式のラベルを解析する

    Args:
        line (str): ラベル行
        img_size (Tuple[int, int]): 画像サイズ (幅, 高さ)

    Returns:
        Tuple[float, List[float], List[Tuple[int, int]]]: 
            クラスID, 原始値リスト, 座標リスト (ポリゴンまたはバウンディングボックス)
    """
    W, H = img_size
    values = line.strip().split()
    cls_id = float(values[0])
    
    # 行の要素数に応じて処理を分岐
    if len(values) >= 9:  # ポリゴン形式 (9列)
        # ポリゴンの頂点座標を抽出 (正規化座標からピクセル座標へ変換)
        polygon_points = []
        for i in range(1, 9, 2):
            if i+1 < len(values):
                x = float(values[i]) * W
                y = float(values[i+1]) * H
                polygon_points.append((x, y))
        
        # ポリゴンから外接矩形を計算
        x1, y1, x2, y2 = get_bbox_from_polygon(polygon_points)
        coords = [(int(x1), int(y1), int(x2), int(y2)), polygon_points]
    else:  # 通常のYOLO形式 (5列)
        # YOLO形式のラベルを解析
        x_c_n, y_c_n, w_n, h_n = map(float, values[1:5])
        
        # YOLO正規化座標 → ピクセル座標
        x_c = x_c_n * W
        y_c = y_c_n * H
        w = w_n * W
        h = h_n * H
        x1, y1 = int(x_c - w/2), int(y_c - h/2)
        x2, y2 = int(x_c + w/2), int(y_c + h/2)
        
        coords = [(x1, y1, x2, y2), []]
    
    return cls_id, values, coords

def draw_tooth_annotation(draw: ImageDraw.Draw, coords: List, is_polygon: bool, 
                          tooth_no: str, color: str, font: Optional[ImageFont.FreeTypeFont]) -> None:
    """歯のアノテーションを描画する

    Args:
        draw (ImageDraw.Draw): 描画オブジェクト
        coords (List): 座標情報
        is_polygon (bool): ポリゴン形式かどうか
        tooth_no (str): 歯番号
        color (str): 描画色
        font (Optional[ImageFont.FreeTypeFont]): フォントオブジェクト
    """
    if is_polygon:
        # ポリゴンを描画
        x1, y1, x2, y2 = coords[0]
        polygon_points = coords[1]
        draw_polygon(draw, polygon_points, color, width=3)
    else:
        # バウンディングボックスを描画
        x1, y1, x2, y2 = coords[0]
        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
    
    # 歯番号を表示
    if font:
        text = f"{tooth_no}"
        # テキスト背景を追加して見やすくする
        text_w, text_h = draw.textbbox((0, 0), text, font=font)[2:]
        draw.rectangle([x1, y1-text_h-2, x1+text_w, y1], fill=color)
        draw.text((x1, y1-text_h-2), text, fill="white", font=font)

def process_status_info(tooth_info: Dict, output_dirs: Dict[str, Path], 
                       crop: Image.Image, out_name: str) -> None:
    """歯の状態情報を処理してフォルダに保存する

    Args:
        tooth_info (Dict): 歯の情報
        output_dirs (Dict[str, Path]): 出力ディレクトリ情報
        crop (Image.Image): 切り取った画像
        out_name (str): 出力ファイル名
    """
    status = tooth_info.get("status", {})
    condition = status.get("condition")
    type_val = status.get("type")
    
    if condition:
        if type_val:
            # typeがある場合は"condition_type"のフォルダへ
            status_folder = f"{condition}_{type_val}"
        else:
            # typeがnullの場合は"condition"のフォルダへ
            status_folder = condition
            
        folder_path = output_dirs["status"] / status_folder
        folder_path.mkdir(parents=True, exist_ok=True)
        crop.save(folder_path / out_name)

def process_restoration_info(tooth_info: Dict, output_dirs: Dict[str, Path], 
                            output_dir: Path, crop: Image.Image, out_name: str) -> None:
    """修復情報を処理してフォルダに保存する

    Args:
        tooth_info (Dict): 歯の情報
        output_dirs (Dict[str, Path]): 出力ディレクトリ情報
        output_dir (Path): 基本出力ディレクトリ
        crop (Image.Image): 切り取った画像
        out_name (str): 出力ファイル名
    """
    restoration = tooth_info.get("restoration", {})
    if isinstance(restoration, dict):
        base_dir, folder_name = create_folder_name(
            "restoration", 
            restoration, 
            "type", 
            "secondaryCaries"
        )
        
        folder_path = output_dir / base_dir / folder_name
        folder_path.mkdir(parents=True, exist_ok=True)
        crop.save(folder_path / out_name)
    elif restoration:  # 文字列の場合（以前のコードとの互換性を維持）
        folder_path = output_dirs["restoration"] / restoration
        folder_path.mkdir(parents=True, exist_ok=True)
        crop.save(folder_path / out_name)

def process_pathology_info(tooth_info: Dict, output_dirs: Dict[str, Path], 
                          crop: Image.Image, out_name: str) -> None:
    """病理情報を処理してフォルダに保存する

    Args:
        tooth_info (Dict): 歯の情報
        output_dirs (Dict[str, Path]): 出力ディレクトリ情報
        crop (Image.Image): 切り取った画像
        out_name (str): 出力ファイル名
    """
    pathology = tooth_info.get("pathology", {})
    if isinstance(pathology, dict):
        # 各病理が存在する場合はそれぞれのフォルダに保存
        has_pathology = False
        
        # 齲蝕
        if pathology.get("caries"):
            folder_path = output_dirs["pathology"] / "caries"
            folder_path.mkdir(parents=True, exist_ok=True)
            crop.save(folder_path / out_name)
            has_pathology = True
            
        # 根尖病変
        if pathology.get("periapicalLesion"):
            folder_path = output_dirs["pathology"] / "periapicalLesion"
            folder_path.mkdir(parents=True, exist_ok=True)
            crop.save(folder_path / out_name)
            has_pathology = True
            
        # 歯周病
        if pathology.get("periodontalDisease"):
            folder_path = output_dirs["pathology"] / "periodontalDisease"
            folder_path.mkdir(parents=True, exist_ok=True)
            crop.save(folder_path / out_name)
            has_pathology = True
            
        # 病理がない場合は「正常」フォルダに保存
        if not has_pathology:
            folder_path = output_dirs["pathology"] / "正常"
            folder_path.mkdir(parents=True, exist_ok=True)
            crop.save(folder_path / out_name)
    elif pathology:  # 文字列の場合（以前のコードとの互換性を維持）
        folder_path = output_dirs["pathology"] / pathology
        folder_path.mkdir(parents=True, exist_ok=True)
        crop.save(folder_path / out_name)

def process_wisdom_tooth_info(tooth_info: Dict, output_dirs: Dict[str, Path], 
                             crop: Image.Image, out_name: str) -> None:
    """親知らずの位置情報を処理してフォルダに保存する

    Args:
        tooth_info (Dict): 歯の情報
        output_dirs (Dict[str, Path]): 出力ディレクトリ情報
        crop (Image.Image): 切り取った画像
        out_name (str): 出力ファイル名
    """
    wisdom_pos = tooth_info.get("wisdomToothPosition")
    if wisdom_pos:
        folder_path = output_dirs["wisdom_tooth"] / wisdom_pos
    else:
        # wisdomToothPositionがnullの場合は「その他」に振り分ける
        folder_path = output_dirs["wisdom_tooth"] / "その他"
    
    folder_path.mkdir(parents=True, exist_ok=True)
    crop.save(folder_path / out_name)

def process_bridge_role_info(tooth_info: Dict, output_dirs: Dict[str, Path], 
                            crop: Image.Image, out_name: str) -> None:
    """ブリッジの役割情報を処理してフォルダに保存する

    Args:
        tooth_info (Dict): 歯の情報
        output_dirs (Dict[str, Path]): 出力ディレクトリ情報
        crop (Image.Image): 切り取った画像
        out_name (str): 出力ファイル名
    """
    bridge_role = tooth_info.get("bridgeRole")
    if bridge_role:
        folder_path = output_dirs["bridge_role"] / bridge_role
        folder_path.mkdir(parents=True, exist_ok=True)
        crop.save(folder_path / out_name)

def process_single_entry(entry: Dict, dataset_paths: Dict[str, Path], 
                      output_dirs: Dict[str, Path], output_dir: Path) -> None:
    """単一のエントリ（写真）を処理する

    Args:
        entry (Dict): 写真のエントリ情報
        dataset_paths (Dict[str, Path]): データセットのパス情報
        output_dirs (Dict[str, Path]): 出力ディレクトリの情報
        output_dir (Path): 基本出力ディレクトリ
    """
    try:
        photo_id = entry["photoId"]
        teeth_info = entry.get("teeth", {})

        # 画像ファイル探索
        img_path = find_image_path(dataset_paths["images"], photo_id)
        if img_path is None:
            logger.warning(f"[WARN] 画像が見つかりません: {photo_id}.*")
            return

        # YOLOラベルファイル
        label_path = dataset_paths["labels"] / f"{photo_id}.txt"
        if not label_path.exists():
            logger.warning(f"[WARN] ラベルファイルがありません: {photo_id}.txt")
            return

        # 元画像を読み込む
        img = Image.open(img_path)
        img_size = img.size
        
        # バウンディングボックス表示用の画像を複製
        bbox_img = img.copy()
        draw = ImageDraw.Draw(bbox_img)
        
        # 歯番号を表示するフォントを設定
        font = setup_drawing_font(img_size)

        # ラベルごとに crop して状態フォルダへ保存
        process_labels(label_path, img, img_size, draw, font, teeth_info, output_dirs, output_dir, photo_id)
        
        # バウンディングボックス表示画像を保存
        bbox_out_path = output_dirs["bbox_vis"] / f"{photo_id}_bbox.png"
        bbox_img.save(bbox_out_path)
        logger.info(f"[INFO] バウンディングボックス可視化画像を保存: {bbox_out_path}")
        
        logger.info(f"[INFO] 処理完了: photoId={photo_id}")
    except Exception as e:
        logger.error(f"[ERROR] エントリの処理中にエラー: {str(e)}")

def process_labels(label_path: Path, img: Image.Image, img_size: Tuple[int, int], 
                  draw: ImageDraw.Draw, font: Optional[ImageFont.FreeTypeFont], 
                  teeth_info: Dict, output_dirs: Dict[str, Path], output_dir: Path,
                  photo_id: str) -> None:
    """ラベルファイルを処理して歯の情報を抽出・保存する

    Args:
        label_path (Path): ラベルファイルのパス
        img (Image.Image): 元画像
        img_size (Tuple[int, int]): 画像サイズ
        draw (ImageDraw.Draw): 描画オブジェクト
        font (Optional[ImageFont.FreeTypeFont]): フォントオブジェクト
        teeth_info (Dict): 歯の情報辞書
        output_dirs (Dict[str, Path]): 出力ディレクトリの情報
        output_dir (Path): 基本出力ディレクトリ
        photo_id (str): 写真ID
    """
    with open(label_path, "r", encoding="utf-8") as lf:
        for line in lf:
            try:
                # ラベルを解析
                cls_id, values, coords = parse_yolo_label(line, img_size)
                tooth_no = str(mapping[int(cls_id)])  # class_id を歯番号として扱う

                # 歯の情報を取得
                tooth_info = teeth_info.get(tooth_no, {})
                
                # 座標情報を抽出
                is_polygon = len(values) >= 9
                bbox_coords = coords[0]  # (x1, y1, x2, y2)
                x1, y1, x2, y2 = bbox_coords
                
                # バウンディングボックスまたはポリゴンを描画
                status = tooth_info.get("status", {})
                condition = status.get("condition")
                color = get_color_for_condition(condition)
                
                # アノテーションを描画
                draw_tooth_annotation(draw, coords, is_polygon, tooth_no, color, font)

                # クロップイメージを作成
                crop = img.crop((x1, y1, x2, y2))
                out_name = f"{photo_id}_{tooth_no}.png"
                
                # 各種情報をフォルダに保存
                process_status_info(tooth_info, output_dirs, crop, out_name)
                process_restoration_info(tooth_info, output_dirs, output_dir, crop, out_name)
                process_pathology_info(tooth_info, output_dirs, crop, out_name)
                process_wisdom_tooth_info(tooth_info, output_dirs, crop, out_name)
                process_bridge_role_info(tooth_info, output_dirs, crop, out_name)
                
            except Exception as e:
                logger.error(f"[ERROR] 歯データの処理中にエラー: error={str(e)}")

def process_annotation_file(ann_path: Path, dataset_paths: Dict[str, Path], 
                           output_dirs: Dict[str, Path], output_dir: Path) -> None:
    """アノテーションファイルを処理する

    Args:
        ann_path (Path): アノテーションファイルのパス
        dataset_paths (Dict[str, Path]): データセットのパス情報
        output_dirs (Dict[str, Path]): 出力ディレクトリの情報
        output_dir (Path): 基本出力ディレクトリ
    """
    try:
        raw = json.loads(ann_path.read_text(encoding="utf-8"))
        ann_list = raw if isinstance(raw, list) else [raw]

        for entry in ann_list:
            process_single_entry(entry, dataset_paths, output_dirs, output_dir)
    except json.JSONDecodeError as e:
        logger.error(f"[ERROR] JSONの解析に失敗: {ann_path}, error={str(e)}")
    except Exception as e:
        logger.error(f"[ERROR] アノテーションファイルの処理中にエラー: {ann_path}, error={str(e)}")

def restructure_to_imagenet_style(dataset_dir: str, output_dir: str) -> None:
    """
    ・dataset_dir/
        ├ images/         （例：1.jpg, 2.jpg, …）
        ├ labels/         （例：1.txt, 2.txt, …　YOLO形式）
        └ annotation/     （例：アノテーション1.json, アノテーション2.json, …）
    を読み込み、
    各歯ごとに crop して
    以下のルールに従ってフォルダを作成し保存します：
    
    1. status：status/<condition>_<type> フォルダ（typeがnullの場合は<condition>のみ）
    2. restoration: restoration/<value> フォルダ
    3. pathology: pathology/<value> フォルダ
    4. wisdomToothPosition: wisdomToothPosition/<value> フォルダ
    5. bridgeRole: bridgeRole/<value> フォルダ
    
    各属性でnullの場合はそのフォルダを作成しません。
    """
    try:
        # パスの準備
        dataset_dir = Path(dataset_dir)
        output_dir = Path(output_dir)
        
        dataset_paths = {
            "images": dataset_dir / "images",
            "labels": dataset_dir / "labels",
            "annotation": dataset_dir / "annotation"
        }
        
        # 出力ディレクトリの初期化
        output_dirs = initialize_output_directories(output_dir)
        
        # annotationディレクトリ内のすべてのJSONを処理
        for ann_path in sorted(dataset_paths["annotation"].glob("*.json")):
            process_annotation_file(ann_path, dataset_paths, output_dirs, output_dir)
            
    except Exception as e:
        logger.error(f"[ERROR] データセットの処理中にエラー: {str(e)}")
        raise FileProcessingError(f"データセット処理に失敗しました: {str(e)}")

if __name__ == "__main__":
    try:
        # ★ 環境に合わせて書き換えてください ★
        dataset_directory = "./dataset/Teeth Segmentation on dental X-ray images"
        output_directory  = "./dataset/imagenet_style"
        rename_image_files(dataset_directory)
        rename_label_files(dataset_directory)
        restructure_to_imagenet_style(dataset_directory, output_directory)
        logger.info("すべての処理が正常に完了しました。")
    except FileProcessingError as e:
        logger.error(f"処理に失敗しました: {str(e)}")
        import sys
        sys.exit(1)