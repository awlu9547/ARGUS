import os
import random
import argparse
import pandas as pd
from sklearn.model_selection import StratifiedKFold


def extract_common_info(path):
    """自定义从路径中提取slide相关识别码的函数"""
    # 假设文件名为：xxx_name_slide_x_y_sp.jpg -> 提取slide
    fname = os.path.basename(path)
    parts = fname.split('_')
    if 'slide' in parts:
        idx = parts.index('slide')
        return '_'.join(parts[:idx + 2])  # 例如：label=1_slide001
    else:
        return fname  # fallback


def read_dir(folder):
    """读取文件夹内所有文件完整路径"""
    if not os.path.exists(folder):
        return []
    return [os.path.join(folder, f) for f in os.listdir(folder)]


def match_tiles_to_fea_adj(tiles, fea_paths, adj_paths):
    fea_dict = {extract_common_info(fea): fea for fea in fea_paths}
    adj_dict = {extract_common_info(adj): adj for adj in adj_paths}

    matched_tiles, matched_feas, matched_adjs = [], [], []

    for tile_path in tiles:
        core = extract_common_info(tile_path)
        fea_path = fea_dict.get(core)
        adj_path = adj_dict.get(core)

        if fea_path and adj_path:
            matched_tiles.append(tile_path)
            matched_feas.append(fea_path)
            matched_adjs.append(adj_path)
        else:
            print(f"[Warning] Not matched for: {tile_path}")

    print(f"✅ Matched {len(matched_tiles)} / {len(tiles)} tiles.")
    return matched_tiles, matched_feas, matched_adjs


def create_arg_parser():
    parser = argparse.ArgumentParser(description='Generate KFold split')
    parser.add_argument('--dir', type=str, default='<YOUR_DATASET_PATH>')
    parser.add_argument('--csv', type=str, default='<YOUR_CSV_PATH>')
    parser.add_argument('--fea_adj', type=str, default='<YOUR_FEA_ADJ_PATH>')
    parser.add_argument('--k', '-k', type=int, default=5)
    parser.add_argument('--on', type=str, choices=['name', 'slide'], default='slide')
    parser.add_argument('--seed', type=int, default=7)
    return parser


def main():
    parser = create_arg_parser()
    args = parser.parse_args()

    patch_dir = os.path.join(args.dir, [x for x in os.listdir(args.dir) if x.endswith('_s')][0])
    assert os.path.exists(patch_dir), 'Patch directory not found'
    assert len([x for x in os.listdir(args.dir) if x.endswith('_s')]) == 1, 'Ambiguous patch directory'

    df = pd.read_csv(args.csv)[['name', 'slide', 'label']].drop_duplicates()
    assert df['label'].nunique() > 2, 'Only multi-label classification supported'
    assert df['label'].dtype == 'int64', "Label must be integer"

    patient_df = df[['name', 'label']].drop_duplicates().reset_index(drop=True)
    patient_df['slide_count'] = patient_df['name'].map(df['name'].value_counts())

    skf = StratifiedKFold(n_splits=args.k, shuffle=True, random_state=args.seed)
    patient_splits = list(skf.split(patient_df['name'], patient_df['label']))

    save_path = os.path.join(args.dir, 'UNI_kf')
    os.makedirs(save_path, exist_ok=True)

    for fold, (train_idx, val_idx) in enumerate(patient_splits, 1):
        print(f'\nCreating fold {fold}')

        train_patients = patient_df.iloc[train_idx]['name'].tolist()
        val_patients = patient_df.iloc[val_idx]['name'].tolist()

        train_slides = df[df['name'].isin(train_patients)]['slide'].unique().tolist()
        val_slides = df[df['name'].isin(val_patients)]['slide'].unique().tolist()

        assert len(set(train_patients) & set(val_patients)) == 0, "患者重叠"

        print(f"Train patients: {len(train_patients)}, Slides: {len(train_slides)}")
        print(f"Val patients: {len(val_patients)}, Slides: {len(val_slides)}")

        df_train, df_val = pd.DataFrame(), pd.DataFrame()

        for slide in df['slide'].unique():
            slide_path = os.path.join(patch_dir, slide)
            if not os.path.exists(slide_path):
                print(f'Warning: Missing slide {slide}')
                continue

            slide_meta = df[df['slide'] == slide].iloc[0]
            label = slide_meta['label']

            tiles = read_dir(slide_path)
            tiles = random.sample(tiles, len(tiles))

            feature_matrix_path = os.path.join(args.fea_adj, 'feature_matrix', slide)
            cooadj_matrix_path = os.path.join(args.fea_adj, 'cooadj_matrix', slide)

            feature_matrix_paths = read_dir(feature_matrix_path)
            cooadj_matrix_paths = read_dir(cooadj_matrix_path)

            matched_tiles, matched_fea_matrix, matched_cooadj_matrix = match_tiles_to_fea_adj(
                tiles, feature_matrix_paths, cooadj_matrix_paths)

            assert len(matched_tiles) == len(matched_fea_matrix) == len(matched_cooadj_matrix)

            patch_num = len(matched_tiles)

            # 新增 pt 路径替换
            slide_m_pt_path = os.path.join('<YOUR_SLIDE_M_PT_PATH>', f'{slide}.pt')
            slide_l_pt_path = os.path.join('<YOUR_SLIDE_L_PT_PATH>', f'{slide}.pt')

            entry = {
                'slide': slide,
                'tile-s': [''] * patch_num,
                'tile-m': [slide_m_pt_path] * patch_num,
                'tile-l': [slide_l_pt_path] * patch_num,
                'patch-num': [patch_num] * patch_num,
                'feature-matrix': matched_fea_matrix,
                'cooadj-matrix': matched_cooadj_matrix,
                'label': label
            }

            df_entry = pd.DataFrame(entry)

            if slide in train_slides:
                df_train = pd.concat([df_train, df_entry], ignore_index=True)
            elif slide in val_slides:
                df_val = pd.concat([df_val, df_entry], ignore_index=True)

        df_train.to_csv(os.path.join(save_path, f'{fold}_train.csv'), index=False)
        df_val.to_csv(os.path.join(save_path, f'{fold}_val.csv'), index=False)
        print(f"Fold {fold} saved. Train: {len(df_train)} entries, Val: {len(df_val)} entries")


if __name__ == '__main__':
    main()
