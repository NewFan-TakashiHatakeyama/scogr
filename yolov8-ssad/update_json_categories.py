import json
import os
from pathlib import Path

# 新しいカテゴリ情報
new_categories = {
    0: '11',
    1: '12',
    2: '13',
    3: '14',
    4: '15',
    5: '16',
    6: '17',
    7: '18',
    8: '21',
    9: '22',
    10: '23',
    11: '24',
    12: '25',
    13: '26',
    14: '27',
    15: '28',
    16: '31',
    17: '32',
    18: '33',
    19: '34',
    20: '35',
    21: '36',
    22: '37',
    23: '38',
    24: '41',
    25: '42',
    26: '43',
    27: '44',
    28: '45',
    29: '46',
    30: '47',
    31: '48'
}

# JSONファイルのパス
json_files = [
    "dentex_dataset/coco/enumeration32/annotations/instances_val2017.json",
    "dentex_dataset/coco/enumeration32/annotations/instances_train2017.json"
]

def update_json_categories(json_path):
    # JSONファイルを読み込む
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # カテゴリ情報を更新
    for category in data['categories']:
        cat_id = category['id']
        if cat_id in new_categories:
            new_name = new_categories[cat_id]
            category['name'] = new_name
            category['supercategory'] = new_name
    
    # ファイルに書き込む前にバックアップを作成
    backup_path = json_path + '.backup'
    if not os.path.exists(backup_path):
        with open(backup_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)
        print(f"バックアップを作成しました: {backup_path}")
    
    # 更新したデータをファイルに書き込む
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2)
    
    print(f"カテゴリを更新しました: {json_path}")
    
    # カテゴリ情報を表示
    print("更新されたカテゴリ:")
    for category in data['categories']:
        print(f"ID: {category['id']}, Name: {category['name']}, Supercategory: {category['supercategory']}")

if __name__ == "__main__":
    for json_file in json_files:
        if os.path.exists(json_file):
            print(f"\n処理中: {json_file}")
            update_json_categories(json_file)
        else:
            print(f"ファイルが見つかりません: {json_file}")
    
    print("\n処理が完了しました。") 