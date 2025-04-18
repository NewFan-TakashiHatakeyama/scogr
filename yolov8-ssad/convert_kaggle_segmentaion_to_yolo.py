import os
import json
from pathlib import Path
from PIL import Image

def segmentation_to_yolo(
    dataset_dir: str,
    output_label_dir: str = None
):
    """
    dataset_dir/
      images/
        6.jpg, 7.jpg, ...
      labels/
        6.jpg.json, 7.jpg.json, ...
    
    からセグメンテーション JSON を読み込み、YOLO bbox フォーマットの
    .txt を (デフォルトで) dataset_dir/labels_yolo/ に生成します。
    """
    dataset_dir    = Path(dataset_dir)
    images_dir     = dataset_dir / "images"
    jsons_dir      = dataset_dir / "labels"
    # 出力先ディレクトリ（指定なければ labels_yolo）
    output_dir     = Path(output_label_dir) if output_label_dir else dataset_dir / "yolo"
    output_dir.mkdir(parents=True, exist_ok=True)

    for img_path in images_dir.glob("*.jpg"):
        stem      = img_path.stem                  # e.g. "6"
        json_path = jsons_dir / f"{stem}.jpg.json" # e.g. "6.jpg.json"
        if not json_path.exists():
            print(f"[WARN] JSON not found for {img_path.name}, skipping")
            continue

        # 画像サイズ取得
        with Image.open(img_path) as im:
            W, H = im.size

        # JSON 読み込み
        data = json.loads(json_path.read_text(encoding="utf-8"))

        yolo_lines = []
        for obj in data.get("objects", []):
            # ポリゴン外周
            pts = obj["points"]["exterior"]  # [[x,y], ...]
            xs  = [p[0] for p in pts]
            ys  = [p[1] for p in pts]
            x_min, x_max = min(xs), max(xs)
            y_min, y_max = min(ys), max(ys)
            w = x_max - x_min
            h = y_max - y_min
            x_center = x_min + w/2
            y_center = y_min + h/2

            # 正規化
            x_center_norm = x_center / W
            y_center_norm = y_center / H
            w_norm        = w        / W
            h_norm        = h        / H

            # クラス ID の決め方：classTitle を整数にしたものをそのまま使うか、
            # 0 始まりにリマップしたい場合は -1 などの調整を。
            class_id = int(obj["classTitle"]) -1
            mapping = {0:7,
                       1:6,
                       2:5,
                       3:4,
                       4:3,
                       5:2,
                       6:1,
                       7:0,
                       8:8,
                       9:9,
                       10:10,
                       11:11,
                       12:12,
                       13:13,
                       14:14,
                       15:15,
                       16:23,
                       17:22,
                       18:21,
                       19:20,
                       20:19,
                       21:18,
                       22:17,
                       23:16,
                       24:24,
                       25:25,
                       26:26,
                       27:27,
                       28:28,
                       29:29,
                       30:30,
                       31:31
                       }
            class_id = mapping[class_id]

            # YOLO 行フォーマット
            yolo_lines.append(f"{class_id} "
                              f"{x_center_norm:.6f} "
                              f"{y_center_norm:.6f} "
                              f"{w_norm:.6f} "
                              f"{h_norm:.6f}")

        # 出力
        out_txt = output_dir / f"{stem}.txt"
        out_txt.write_text("\n".join(yolo_lines), encoding="utf-8")
        print(f"Saved {out_txt} ({len(yolo_lines)} objects)")

if __name__ == "__main__":
    # --- 設定 ---
    dataset_directory    = "Teeth Segmentation on dental X-ray images"       # images/, labels/ を含むフォルダ
    output_label_folder  = None    # None → dataset/labels_yolo に書き出し

    segmentation_to_yolo(dataset_directory, output_label_folder)