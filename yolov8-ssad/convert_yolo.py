import os
import json
import argparse
from pathlib import Path

def convert_coco_to_yolo(coco_json_path, yolo_txt_save_dir):
    # 出力ディレクトリの作成
    os.makedirs(yolo_txt_save_dir, exist_ok=True)
    
    # JSONファイルの読み込み
    with open(coco_json_path, 'r') as json_file:
        json_load = json.load(json_file)

    annotations = json_load['annotations']
    images = json_load['images']

    # 画像IDとファイル名の対応辞書を作成
    image_id_to_file_name = {image['id']: image for image in images}

    # 各アノテーションに対して処理
    for annotation in annotations:
        image_info = image_id_to_file_name[annotation['image_id']]
        file_name = image_info['file_name']
        im_w = image_info['width']
        im_h = image_info['height']
        
        # テキストファイルのパスを作成
        txt_path = os.path.join(yolo_txt_save_dir, os.path.splitext(os.path.basename(file_name))[0] + '.txt')
        
        # バウンディングボックスの座標を変換
        bbox = annotation['bbox']
        x_center = bbox[0] + bbox[2] / 2
        y_center = bbox[1] + bbox[3] / 2
        bbox_converted = [x_center / im_w, y_center / im_h, bbox[2] / im_w, bbox[3] / im_h]
        
        # カテゴリーIDを取得
        cls = int(annotation['category_id'])
        
        # 書き込む行を作成
        line = (cls, *bbox_converted)
        
        # テキストファイルに追記モードで書き込み
        with open(txt_path, 'a') as f:
            f.write(('%g ' * len(line)).rstrip() % line + '\n')
    
    print(f"変換完了！アノテーション数: {len(annotations)}、ファイル数: {len(images)}")
    return len(annotations), len(images)

def main():
    parser = argparse.ArgumentParser(description='COCOフォーマットのアノテーションファイルをYOLOフォーマットに変換します')
    parser.add_argument('--coco_json_path', type=str, 
                        default='dentex_dataset/coco/enumeration32/annotations/instances_train2017.json',
                        help='入力COCOフォーマットJSONファイルのパス')
    parser.add_argument('--yolo_save_dir', type=str, 
                        default='dentex_dataset/yolo/enumeration32/labels/train',
                        help='出力YOLOフォーマットラベルを保存するディレクトリ')
    
    args = parser.parse_args()
    
    # 変換処理を実行
    num_annotations, num_images = convert_coco_to_yolo(args.coco_json_path, args.yolo_save_dir)
    
    print(f"変換が完了しました: {args.coco_json_path} -> {args.yolo_save_dir}")
    print(f"処理されたアノテーション数: {num_annotations}")
    print(f"処理された画像数: {num_images}")

if __name__ == "__main__":
    main()