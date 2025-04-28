#!/usr/bin/env python3
"""
Aggregate per-image tooth correctness from evaluation_results.json
Usage:
  python aggregate_tooth_correctness.py --eval_json evaluation_results.json --output per_image_tooth_correctness.csv
"""
import json
import argparse
import pandas as pd
from pathlib import Path

# 標準的な FDI 表記 11-18,21-28,31-38,41-48
FDI_NUMS = list(range(11,19)) + list(range(21,29)) + list(range(31,39)) + list(range(41,49))
FDI_STRS = [str(n) for n in FDI_NUMS]

def main():
    parser = argparse.ArgumentParser(description='Aggregate per-image tooth correctness')
    parser.add_argument('--eval_json', type=str, required=True,
                        help='Path to evaluation_results.json')
    parser.add_argument('--output', type=str, default='per_image_tooth_correctness.csv',
                        help='Output CSV filename')
    args = parser.parse_args()

    # JSON 読み込み
    with open(args.eval_json, 'r', encoding='utf-8') as f:
        data = json.load(f)
    results = data.get('results', [])

    records = []
    for item in results:
        row = {'image': item.get('image', '')}
        gt_map = item.get('gt', {})
        pred_map = item.get('pred', {})
        for num in FDI_STRS:
            # GT と予測の一致なら 1, そうでなければ 0
            row[num] = 1 if gt_map.get(num, 0) == pred_map.get(num, 0) else 0
        records.append(row)

    df = pd.DataFrame(records)
    # カラム順: image, 11,12,...,48
    df = df[['image'] + FDI_STRS]
    df.to_csv(args.output, index=False, encoding='utf-8')
    print(f"Saved per-image tooth correctness CSV to {args.output}")

if __name__ == '__main__':
    main() 