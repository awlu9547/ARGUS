# import os
# import random
# import argparse
# import pandas as pd
# from sklearn.model_selection import StratifiedKFold
#
#
# def read_dir(path):
#     file_list = [os.path.join(path, file) for file in os.listdir(path) if os.path.isfile(os.path.join(path, file))]
#     return file_list
#
#
# def create_arg_parser():
#     parser = argparse.ArgumentParser(description='Generate KFold split')
#     parser.add_argument('--dir', type=str, default=r"G:\Projects\hiUNI_dataset", help='Directory to process')
#     parser.add_argument('--csv', type=str, default=r"G:\Projects\hiUNI_dataset\dataset_metadata.csv", help='CSV label file to process')
#     parser.add_argument('--k', '-k', type=int, default=5, help='K-fold cross validation, number of folds')
#     parser.add_argument('--on', type=str, choices=['name', 'slide'], default='slide',
#                         help='Split on name or slide')
#     parser.add_argument('--seed', type=int, default=7, help='Random seed for shuffling')
#     return parser
#
#
# def main():
#     """
#     一个病人存在多张切片，如果以slide_level的形式进行五折划分，存在数据泄露问题
#     在slide划分时需要考虑同一patient不可重复出现在train与val中
#     """
#     parser = create_arg_parser()
#     args = parser.parse_args()
#
#     patch_dir = args.dir + '/' + [x for x in os.listdir(args.dir) if x.endswith('_s')][0]
#     assert patch_dir, 'Patch directory not found'
#     assert len([x for x in os.listdir(args.dir) if x.endswith('_s')]) == 1, \
#         'Ambiguous patch directory, please make sure there is only one directory ends with "_s"'
#     print('Patch directory:', patch_dir)
#
#     total_split = args.k
#     df = pd.read_csv(args.csv)[['name', 'slide', 'label']]
#
#     assert df['label'].nunique() != 2, 'Only multi-label classification is supported'
#     # label must be digits
#     assert df['label'].dtype == 'int64', "Label col 'label' must be integer"
#
#     save_path = r"G:\Projects\hiUNI_dataset"
#     kf_path = os.path.join(save_path, 'kf')
#     os.makedirs(kf_path, exist_ok=True)
#
#     skf = StratifiedKFold(n_splits=total_split, shuffle=True, random_state=args.seed).split(df[args.on], df['label'])
#
#     for fold, (train_index, test_index) in enumerate(skf):
#         fold = fold + 1
#         print('creating fold {}'.format(fold))
#         slide_train, slide_val = df[args.on].iloc[train_index], df[args.on].iloc[test_index]
#         df_train = pd.DataFrame(columns=['slide', 'tile-s', 'tile-m', 'tile-l', 'label'])
#         df_val = pd.DataFrame(columns=['slide', 'tile-s', 'tile-m', 'tile-l', 'label'])
#
#         for slide in df[args.on]:
#             print('processing slide {}'.format(slide))
#             slides_loc = os.path.join(patch_dir, slide)
#             if not os.path.exists(slides_loc):
#                 print('slide {} not found'.format(slide))
#                 continue
#             slides_loc_list = read_dir(slides_loc)
#             slide = df[df[args.on] == slide]['slide'].tolist()[0]
#             label = df[df[args.on] == slide]['label'].tolist()[0]
#
#             slides_loc_list = random.sample(slides_loc_list, len(slides_loc_list))
#
#             slides_loc_list_s = slides_loc_list
#             slides_loc_list_m = [x.replace('_s', '_m') for x in slides_loc_list]
#             slides_loc_list_l = [x.replace('_s', '_l') for x in slides_loc_list]
#             df_new = pd.DataFrame({'slide': slide,
#                                    'tile-s': slides_loc_list_s, 'tile-m': slides_loc_list_m,
#                                    'tile-l': slides_loc_list_l,
#                                    'label': label})
#             if slide in slide_train.tolist():
#                 df_train = pd.concat([df_train, df_new], ignore_index=True)
#             elif slide in slide_val.tolist():
#                 df_val = pd.concat([df_val, df_new], ignore_index=True)
#             else:
#                 print('[warning] slide {} not found'.format(slide))
#
#         train_patient_slides = len(df_train['slide'].unique())
#         val_patient_slides = len(df_val['slide'].unique())
#         print(f'Fold {fold}: train-set patient_slides: {train_patient_slides} val-set patient_slides: {val_patient_slides}')
#
#         df_train.to_csv(f'{kf_path}/{fold}_train.csv', index=False)
#         df_val.to_csv(f'{kf_path}/{fold}_val.csv', index=False)
#
#
# if __name__ == '__main__':
#     main()

"""----------------以上版本暂未考虑同一患者存在多张切片的情况，slide_level划分存在数据泄露---------------"""

# import os
# import argparse
# import random
# import pandas as pd
# from sklearn.model_selection import StratifiedKFold
# import numpy as np
#
#
# def read_dir(path):
#     file_list = [os.path.join(path, f) for f in os.listdir(path)
#                  if os.path.isfile(os.path.join(path, f))]
#     return file_list
#
#
# def create_arg_parser():
#     parser = argparse.ArgumentParser(description='Generate KFold split')
#     parser.add_argument('--dir', type=str, default='<YOUR_DATASET_PATH>',
#                         help='Directory to process')
#     parser.add_argument('--csv', type=str, default='<YOUR_CSV_PATH>',
#                         help='CSV label file to process')
#     parser.add_argument('--k', '-k', type=int, default=5, help='K-fold cross validation, number of folds')
#
#     parser.add_argument('--on', type=str, choices=['name', 'slide'], default='slide',
#                         help='Split on name or slide , here the slide_level must consider the patient, because the singe patient maybe have multiple patients')
#     parser.add_argument('--seed', type=int, default=7, help='Random seed for shuffling')
#     return parser
#
#
# def main():
#     """the total number of patches is 76248"""
#     parser = create_arg_parser()
#     args = parser.parse_args()
#
#     # 获取补丁目录路径
#     patch_dir = os.path.join(args.dir, [x for x in os.listdir(args.dir) if x.endswith('_s')][0])
#     assert os.path.exists(patch_dir), 'Patch directory not found'
#     assert len([x for x in os.listdir(args.dir) if x.endswith('_s')]) == 1, \
#         'Ambiguous patch directory'
#     print(f'Patch directory: {patch_dir}')
#
#     # 加载并预处理数据
#     df = pd.read_csv(args.csv)[['name', 'slide', 'label']].drop_duplicates()
#     assert df['label'].nunique() > 2, 'Only multi-label classification supported'
#     assert df['label'].dtype == 'int64', "Label must be integer"
#
#     # 创建患者级元数据
#     patient_df = df[['name', 'label']].drop_duplicates().reset_index(drop=True)
#     patient_df['slide_count'] = patient_df['name'].map(df['name'].value_counts())
#
#     # 分层患者划分
#     skf = StratifiedKFold(n_splits=args.k, shuffle=True, random_state=args.seed)
#     patient_splits = list(skf.split(patient_df['name'], patient_df['label']))
#
#     # 创建输出目录
#     save_path = os.path.join(args.dir, 'kf')
#     os.makedirs(save_path, exist_ok=True)
#
#     for fold, (train_idx, val_idx) in enumerate(patient_splits, 1):
#         print(f'\nCreating fold {fold}')
#
#         # 获取患者划分
#         train_patients = patient_df.iloc[train_idx]['name'].tolist()
#         val_patients = patient_df.iloc[val_idx]['name'].tolist()
#
#         # 获取对应的slide列表
#         train_slides = df[df['name'].isin(train_patients)]['slide'].unique().tolist()
#         val_slides = df[df['name'].isin(val_patients)]['slide'].unique().tolist()
#
#         # 数据校验
#         overlap = set(train_patients) & set(val_patients)
#         assert len(overlap) == 0, f"患者重叠: {overlap}"
#         print(f"Train patients: {len(train_patients)}, Slides: {len(train_slides)}")
#         print(f"Val patients: {len(val_patients)}, Slides: {len(val_slides)}")
#
#         # 构建数据框
#         df_train, df_val = pd.DataFrame(), pd.DataFrame()
#
#         # 处理所有slide
#         for slide in df['slide'].unique():
#             slide_path = os.path.join(patch_dir, slide)
#             if not os.path.exists(slide_path):
#                 print(f'Warning: Missing slide {slide}')
#                 continue
#
#             # 获取slide元数据
#             slide_meta = df[df['slide'] == slide].iloc[0]
#             label = slide_meta['label']
#
#             # 加载tiles路径
#             tiles = read_dir(slide_path)
#             tiles = random.sample(tiles, len(tiles))  # 随机打乱
#
#             # 构建数据条目
#             entry = {
#                 'slide': slide,
#                 'tile-s': tiles,
#                 'tile-m': [t.replace('_s', '_m') for t in tiles],
#                 'tile-l': [t.replace('_s', '_l') for t in tiles],
#                 'label': label
#             }
#             df_entry = pd.DataFrame(entry)
#
#             # 分配数据集
#             if slide in train_slides:
#                 df_train = pd.concat([df_train, df_entry], ignore_index=True)
#             elif slide in val_slides:
#                 df_val = pd.concat([df_val, df_entry], ignore_index=True)
#
#         # 保存结果
#         df_train.to_csv(os.path.join(save_path, f'{fold}_train.csv'), index=False)
#         df_val.to_csv(os.path.join(save_path, f'{fold}_val.csv'), index=False)
#
#         print(f"Fold {fold} saved. Train: {len(df_train)} entries, Val: {len(df_val)} entries")
"""----------------有效副本，以下版本将在生成五折文件中增加特征矩阵与邻接矩阵---------------"""
import os
import argparse
import random
import pandas as pd
from sklearn.model_selection import StratifiedKFold
import numpy as np
import re


def read_dir(path):
    # 返回所要检索的jpg、pt格式的文件
    valid_exts = ['.pt', '.jpg']

    file_list = [
        os.path.join(path, f)
        for f in os.listdir(path)
        if os.path.isfile(os.path.join(path, f))
           and os.path.splitext(f)[1].lower() in valid_exts
           and not f.lower().endswith('.db')
           and not f.startswith('.')
    ]
    return file_list


def extract_common_info(filename):
    """
    从tile或pt文件名中提取关键字段用于匹配。
    返回形如：slide=100755-1,label=1,x=4517,y=45172,sp=0.5
    """
    base = os.path.basename(filename)
    if base.endswith('.jpg'):
        core = base.replace('.jpg', '')
    elif '_feature.pt' in base:
        core = base.replace('nucleus_', '').replace('_feature.pt', '')
    elif '_adj.pt' in base:
        core = base.replace('nucleus_', '').replace('_adj.pt', '')
    else:
        core = ''
    return core


def match_tiles_to_fea_adj(tiles, fea_paths, adj_paths):
    """
    根据文件名中的slide/label/x/y/sp等信息，将tile、feature、adj路径一一对应匹配。
    输入均为文件路径列表（完整路径）。
    返回匹配成功的tile路径、对应fea路径、adj路径。
    """
    # 创建dict便于快速查找
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
    parser.add_argument('--dir', type=str, default='<YOUR_DATASET_PATH>',
                        help='Directory to process')
    parser.add_argument('--csv', type=str, default='<YOUR_CSV_PATH>',
                        help='CSV label file to process'),
    parser.add_argument('--fea_adj', type=str, default='<YOUR_FEA_ADJ_PATH>',
                        help='pt file of "feature matrix" and "cooadj matrix"  to process')
    parser.add_argument('--k', '-k', type=int, default=5, help='K-fold cross validation, number of folds')

    parser.add_argument('--on', type=str, choices=['name', 'slide'], default='slide',
                        help='Split on name or slide , here the slide_level must consider the patient, because the singe patient maybe have multiple patients')
    parser.add_argument('--seed', type=int, default=7, help='Random seed for shuffling')
    return parser


def main():
    """the total number of patches is 76248"""
    parser = create_arg_parser()
    args = parser.parse_args()

    # 获取补丁目录路径
    patch_dir = os.path.join(args.dir, [x for x in os.listdir(args.dir) if x.endswith('_s')][0])
    assert os.path.exists(patch_dir), 'Patch directory not found'
    assert len([x for x in os.listdir(args.dir) if x.endswith('_s')]) == 1, \
        'Ambiguous patch directory'
    # print(f'Patch directory: {patch_dir}')

    # 加载并预处理数据
    df = pd.read_csv(args.csv)[['name', 'slide', 'label']].drop_duplicates()
    # assert df['label'].nunique() > 2, 'Only multi-label classification supported'
    assert df['label'].dtype == 'int64', "Label must be integer"

    # 创建患者级元数据
    patient_df = df[['name', 'label']].drop_duplicates().reset_index(drop=True)
    patient_df['slide_count'] = patient_df['name'].map(df['name'].value_counts())

    # 分层患者划分
    skf = StratifiedKFold(n_splits=args.k, shuffle=True, random_state=args.seed)
    patient_splits = list(skf.split(patient_df['name'], patient_df['label']))

    # 创建输出目录
    save_path = os.path.join(args.dir, 'UNI_kf')
    os.makedirs(save_path, exist_ok=True)

    for fold, (train_idx, val_idx) in enumerate(patient_splits, 1):
        print(f'\nCreating fold {fold}')

        # 获取患者划分
        train_patients = patient_df.iloc[train_idx]['name'].tolist()
        val_patients = patient_df.iloc[val_idx]['name'].tolist()

        # 获取对应的slide列表
        train_slides = df[df['name'].isin(train_patients)]['slide'].unique().tolist()
        val_slides = df[df['name'].isin(val_patients)]['slide'].unique().tolist()

        # 数据校验
        overlap = set(train_patients) & set(val_patients)
        assert len(overlap) == 0, f"患者重叠: {overlap}"
        print(f"Train patients: {len(train_patients)}, Slides: {len(train_slides)}")
        print(f"Val patients: {len(val_patients)}, Slides: {len(val_slides)}")

        # 构建数据框
        df_train, df_val = pd.DataFrame(), pd.DataFrame()

        # 处理所有slide
        for slide in df['slide'].unique():
            slide_path = os.path.join(patch_dir, slide)
            if not os.path.exists(slide_path):
                print(f'Warning: Missing slide {slide}')
                continue

            # 获取slide元数据
            slide_meta = df[df['slide'] == slide].iloc[0]
            label = slide_meta['label']

            # 加载tiles路径
            tiles = read_dir(slide_path)
            tiles = random.sample(tiles, len(tiles))  # 随机打乱
            # print(tiles)

            # 构建特征矩阵和邻接矩阵路径
            feature_matrix_path = os.path.join(args.fea_adj, 'feature_matrix', slide)
            cooadj_matrix_path = os.path.join(args.fea_adj, 'cooadj_matrix', slide)

            feature_matrix_paths = read_dir(feature_matrix_path)
            cooadj_matrix_paths = read_dir(cooadj_matrix_path)
            # print(feature_matrix_paths)

            matched_tiles, matched_fea_matrix, matched_cooadj_matrix = match_tiles_to_fea_adj(tiles,
                                                                                              feature_matrix_paths,
                                                                                              cooadj_matrix_paths)
            assert len(matched_tiles) == len(matched_fea_matrix) == len(matched_cooadj_matrix), "匹配后的文件数量不一致"

            patch_num = len(tiles)
            # print(patch_num)

            # 构建数据条目
            entry = {
                'slide': slide,
                'tile-s': matched_tiles,
                'tile-m': [t.replace('_s', '_m') for t in matched_tiles],
                'tile-l': [t.replace('_s', '_l') for t in matched_tiles],
                'patch-num': [patch_num] * patch_num,
                'feature-matrix': matched_fea_matrix,
                'cooadj-matrix': matched_cooadj_matrix,
                'label': label
            }
            df_entry = pd.DataFrame(entry)

            # 分配数据集
            if slide in train_slides:
                df_train = pd.concat([df_train, df_entry], ignore_index=True)
            elif slide in val_slides:
                df_val = pd.concat([df_val, df_entry], ignore_index=True)

        # 保存结果
        df_train.to_csv(os.path.join(save_path, f'{fold}_train.csv'), index=False)
        df_val.to_csv(os.path.join(save_path, f'{fold}_val.csv'), index=False)

        print(f"Fold {fold} saved. Train: {len(df_train)} entries, Val: {len(df_val)} entries")


if __name__ == '__main__':
    main()
