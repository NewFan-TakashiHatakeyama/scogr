import os

# 日本語フォルダ名 → 英語フォルダ名のマッピング
to_english = {
    'インレー': 'Inlay',
    'インレー_secondaryCaries': 'Inlay_secondaryCaries',
    'クラウン': 'Crown',
    'クラウン_secondaryCaries': 'Crown_secondaryCaries',
    'その他': 'Other',
    'その他_secondaryCaries': 'Other_secondaryCaries',
    '処置歯': 'Treated_tooth',
    '処置歯_secondaryCaries': 'Treated_secondaryCaries',
    '角度付': 'Angled',
    '水平': 'Horizontal',
    '正常': 'Normal',
    '半萌出': 'Semi-erupted',
    '未萌歯_永久歯': 'Unerupted_Permanent',
    '残存歯': 'Remaining_Teeth',
    '残存歯_永久歯': 'Remaining_Permanent',
    '残存歯_乳歯': 'Remaining_Deciduous',
    '先欠歯_永久歯': 'Predecessor_Permanent',
    '埋伏歯': 'Buried_Teeth',
    '埋伏歯_永久歯': 'Buried_Permanent',
    '未萌歯': 'Unerupted',
    'なし': 'None',
    '支台歯': 'Propping_Tooth',
}

def rename_japanese_dirs(root_path):
    """
    指定ルート以下のディレクトリで、日本語名をキーにマッピングして英語名にリネームします
    """
    for dirpath, dirnames, filenames in os.walk(root_path, topdown=True):
        for name in list(dirnames):
            if name in to_english:
                src = os.path.join(dirpath, name)
                dst = os.path.join(dirpath, to_english[name])
                try:
                    print(f"Renaming: {src} -> {dst}")
                    os.rename(src, dst)
                except Exception as e:
                    print(f"Failed to rename {src}: {e}")
                # ネストするディレクトリ名も更新済みの名前で探索を続行
                dirnames[dirnames.index(name)] = to_english[name]

if __name__ == '__main__':
    # 作業ディレクトリをプロジェクトルートに合わせます
    project_root = os.path.abspath(os.path.dirname(__file__))
    datasets_root = os.path.join(project_root, 'datasetsV2', 'imagenet_style')
    print(f"Datasets root: {datasets_root}")
    rename_japanese_dirs(datasets_root) 