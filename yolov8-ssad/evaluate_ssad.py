#!/usr/bin/env python3
# SSADモデルの歯検出結果を評価するスクリプト
# Usage: python evaluate_ssad.py --weights ./weights/best.pt --image_dir path/to/images --label_dir path/to/labels --output results.json --conf 0.5

import os
import glob
import json
import argparse
import pandas as pd
from pathlib import Path
import torch
from ultralytics import YOLO

# FDI 歯番号 (標準的な FDI 表記)
FDI_NUMS = list(range(11,19)) + list(range(21,29)) + list(range(31,39)) + list(range(41,49))
FDI_STRS = [str(n) for n in FDI_NUMS]
# クラスID (0-31) -> FDI番号
CLASS_TO_FDI = {i: FDI_STRS[i] for i in range(len(FDI_STRS))}

def load_gt_labels(label_path):
    # YOLOラベルファイルからGT歯クラスIDを読み込んでFDI番号セットを返す
    gt_ids = []
    with open(label_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 5:
                try:
                    cls_id = int(parts[0])
                    gt_ids.append(cls_id)
                except:
                    continue
    return set(CLASS_TO_FDI[i] for i in gt_ids if i in CLASS_TO_FDI)

def predict_tooth_nums(model, img_path, conf_thresh):
    # モデル推論を実行し、予測されたFDI番号セットを返す
    results = model(str(img_path), conf=conf_thresh, task='ssad')
    res = results[0]
    # .boxes.cls は Tensor
    cls_ids = res.boxes.cls.cpu().numpy().astype(int).tolist()
    return set(CLASS_TO_FDI[i] for i in cls_ids if i in CLASS_TO_FDI)

def main():
    parser = argparse.ArgumentParser(description='SSADモデル歯検出評価スクリプト')
    parser.add_argument('--weights', type=str, default='./weights/best.pt', 
                        help='モデルの重みファイルのパス')
    parser.add_argument('--image_dir', type=str, default='./Test Dataset/images/',  help='画像ディレクトリ')
    parser.add_argument('--label_dir', type=str, default='./Test Dataset/labels/',  help='ラベルディレクトリ')
    parser.add_argument('--output', type=str, default='./results/evaluation_results.json', help='出力JSONファイル')
    parser.add_argument('--conf', type=float, default=0.5, help='信頼度閾値')
    args = parser.parse_args()

    # モデルロード
    model = YOLO(args.weights)
    model.task = 'ssad'

    # 画像ファイル取得
    image_paths = sorted(glob.glob(os.path.join(args.image_dir, '*.*')))
    results = []
    total_correct = 0
    total_teeth = 0
    for img_path in image_paths:
        label_path = os.path.join(args.label_dir, Path(img_path).stem + '.txt')
        if not os.path.exists(label_path):
            print(f"Warning: ラベルファイルが見つかりません: {label_path}")
            continue
        gt_set = load_gt_labels(label_path)
        pred_set = predict_tooth_nums(model, img_path, args.conf)
        gt_map = {num: (1 if num in gt_set else 0) for num in FDI_STRS}
        pred_map = {num: (1 if num in pred_set else 0) for num in FDI_STRS}
        correct = sum(gt_map[num] == pred_map[num] for num in FDI_STRS)
        total_correct += correct
        total_teeth += len(FDI_STRS)
        acc = correct / len(FDI_STRS)
        results.append({
            'image': Path(img_path).name,
            'accuracy': acc,
            'gt': gt_map,
            'pred': pred_map
        })
        print(f"{Path(img_path).name}: {acc:.4f}")
    overall_acc = total_correct / total_teeth if total_teeth > 0 else 0.0
    print(f"Overall accuracy: {overall_acc:.4f}")
    out_dict = {
        'overall_accuracy': overall_acc,
        'results': results
    }
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(out_dict, f, ensure_ascii=False, indent=2)
    # --- CSV出力処理開始 ---
    # CSV保存先ディレクトリ：JSONファイルと同じ場所
    csv_dir = Path(args.output).parent
    os.makedirs(csv_dir, exist_ok=True)
    all_records = []
    for item in results:
        image_name = item['image']
        acc = item['accuracy']
        gt_map = item['gt']
        pred_map = item['pred']
        # ファイル名に画像名とスコアを含める
        csv_filename = f"{Path(image_name).stem}_{acc:.4f}.csv"
        df = pd.DataFrame({
            'FDI': list(gt_map.keys()),
            'gt': [gt_map[k] for k in gt_map.keys()],
            'pred': [pred_map.get(k, 0) for k in gt_map.keys()]
        })
        df.to_csv(csv_dir / csv_filename, index=False, encoding='utf-8')
        all_records.append({'csv_file': csv_filename, 'score': acc})
    # 全画像のスコアをまとめたCSVを出力
    df_all = pd.DataFrame(all_records)
    df_all.to_csv(csv_dir / 'all_score.csv', index=False, encoding='utf-8')
    print(f"Saved per-image CSVs and all_score.csv to {csv_dir}")

    # --- 歯番号ごとの精度算出 ---
    # JSON結果を読み込み
    with open(args.output, 'r', encoding='utf-8') as jf:
        data = json.load(jf)
    results = data['results']
    # 歯番号ごとの正解数と総数を集計
    tooth_stats = {num: {'correct': 0, 'total': 0} for num in FDI_STRS}
    for item in results:
        gt = item['gt']
        pred = item['pred']
        for num in FDI_STRS:
            if gt[num] == pred[num]:
                tooth_stats[num]['correct'] += 1
            tooth_stats[num]['total'] += 1
    # 精度を計算してCSVに保存
    acc_list = [{'FDI': num, 'accuracy': tooth_stats[num]['correct'] / tooth_stats[num]['total']} for num in FDI_STRS]
    df_tooth = pd.DataFrame(acc_list)
    df_tooth.to_csv(csv_dir / 'per_tooth_accuracy.csv', index=False, encoding='utf-8')
    print(f"Saved per-tooth accuracy to {csv_dir / 'per_tooth_accuracy.csv'}")

if __name__ == '__main__':
    main() 