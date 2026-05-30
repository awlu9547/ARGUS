import os

import torch

from torch.utils.data import Dataset
from torchvision import transforms
import pandas as pd

from PIL import Image

transform = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
])


class PathologyDataset(Dataset):
    def __init__(self, mode, combination):
        self.mode = mode
        self.combination = combination
        self.data = pd.read_csv(f'{mode}.csv')

        self.slide = self.data['slide']
        self.tile_paths = {
            's': self.data['tile-s'],
            'm': self.data['tile-m'],
            'l': self.data['tile-l']
        }
        self.y_data = self.data['label']
        self.transform = transform

    def __len__(self):
        return len(self.name)

    def __getitem__(self, index):
        # load tile
        slide = self.slide.iloc[index]
        tiles = []

        for combination_key in self.combination:
            tile_path = self.tile_paths[combination_key].iloc[index]
            tile = self.transform(Image.open(tile_path))
            tiles.append(tile)

        tile = torch.stack(tiles)

        # load label
        label = self.y_data.iloc[index]
        label = torch.as_tensor(label, dtype=torch.long)

        if self.mode == 'train':
            return tile, label
        elif self.mode == 'val':
            return slide, tile, label
        return None


# class PathologyDatasetKFold(Dataset):
#     def __init__(self, mode, combination, fold):
#         self.mode = mode
#         self.combination = combination
#         self.kf_path = '<YOUR_KF_PATH>'
#         self.data = pd.read_csv(f'{self.kf_path}/{fold}_{mode}.csv')
#
#         self.slide = self.data['slide']
#         self.tile_paths = {
#             's': self.data['tile-s'] if 'tile-s' in self.data.columns else None,
#             'm': self.data['tile-m'] if 'tile-m' in self.data.columns else None,
#             'l': self.data['tile-l'] if 'tile-l' in self.data.columns else None
#         }
#         self.fea_paths = self.data['feature-matrix'] if 'feature-matrix' in self.data.columns else None
#         self.cooadj_paths = self.data['cooadj-matrix'] if 'cooadj-matrix' in self.data.columns else None
#
#         self.label_data = self.data['label']
#         self.transform = transform
#         self.patch_nums = {
#             scale: self._compute_patch_nums(scale)
#             for scale in self.combination
#         }
#
#     def _compute_patch_nums(self, scale):
#         patch_counts = []
#         for idx in range(len(self)):
#             tile_paths = self.tile_paths[scale].iloc[idx]
#             if isinstance(tile_paths, str):
#                 count = len(tile_paths.split(','))
#             else:
#                 count = 1
#             patch_counts.append(count)
#         return patch_counts
#
#     def __len__(self):
#         return len(self.slide)
#
#     def __getitem__(self, index):
#         # load tile
#         # 这里的index相当于逐行选取slide、label、tile信息
#         slide = self.slide.iloc[index]
#         print(f"===> Slide: {slide}")
#         tiles = []
#
#         for combination_key in self.combination:
#             tile_path = self.tile_paths[combination_key].iloc[index]
#             tile = self.transform(Image.open(tile_path))
#             tiles.append(tile)
#
#         tile = torch.stack(tiles)  # [len(cmb), 3, 224, 224]
#
#         # load label
#         label = self.label_data.iloc[index]
#         label = torch.as_tensor(label, dtype=torch.long)
#
#         # load feature matrix and cooadj matrix
#         fea_path = self.fea_paths.iloc[index] if self.fea_paths is not None else None
#         cooadj_path = self.cooadj_paths.iloc[index] if self.cooadj_paths is not None else None
#
#         if self.mode == 'train':
#             return tile, label, fea_path, cooadj_path
#         elif self.mode == 'val':
#             return slide, tile, label, fea_path, cooadj_path
#         return None


class PathologyDatasetKFold(Dataset):
    def __init__(self, mode, combination, fold):
        self.mode = mode
        self.combination = combination
        self.kf_path = '<YOUR_KF_PATH>'
        self.data = pd.read_csv(f'{self.kf_path}/{fold}_{mode}.csv')

        self.slide = self.data['slide']
        self.tile_paths = {
            's': self.data['tile-s'] if 'tile-s' in self.data.columns else None,
            'm': self.data['tile-m'] if 'tile-m' in self.data.columns else None,
            'l': self.data['tile-l'] if 'tile-l' in self.data.columns else None
        }
        self.patch_nums = self.data['patch-num'] if 'patch-num' in self.data.columns else None

        self.fea_paths = self.data['feature-matrix'] if 'feature-matrix' in self.data.columns else None
        self.cooadj_paths = self.data['cooadj-matrix'] if 'cooadj-matrix' in self.data.columns else None

        self.label_data = self.data['label']
        self.transform = transform

    def __len__(self):
        return len(self.slide)

    def __getitem__(self, index):
        # load tile
        # 这里的index相当于逐行选取slide、label、tile信息
        slide = self.slide.iloc[index]
        # print(f"===> Slide: {slide}")

        patch_nums = self.patch_nums.iloc[index]

        tiles = []

        for combination_key in self.combination:
            tile_path = self.tile_paths[combination_key].iloc[index]
            tile = self.transform(Image.open(tile_path))
            tiles.append(tile)

        tile = torch.stack(tiles)  # [len(cmb), 3, 224, 224]

        # load label
        label = self.label_data.iloc[index]
        label = torch.as_tensor(label, dtype=torch.long)
        patch_nums = torch.as_tensor(patch_nums, dtype=torch.long)
        # print(f'label shape: {label.shape}, patch_nums shape: {patch_nums.shape}')

        # load feature matrix and cooadj matrix
        fea_path = self.fea_paths.iloc[index] if self.fea_paths is not None else None
        cooadj_path = self.cooadj_paths.iloc[index] if self.cooadj_paths is not None else None

        feature_list = torch.load(fea_path, weights_only=True)  # [num_nodes, 304]
        cooadj_list = torch.load(cooadj_path, weights_only=True)  # [num_nodes, num_nodes]

        if self.mode == 'train':
            return tile, patch_nums, label, feature_list, cooadj_list
        elif self.mode == 'val':
            # 这里输出的slide 与 tile 一一对应
            return slide, tile, patch_nums, label, feature_list, cooadj_list
        return None


class Extra_Dataset(Dataset):
    def __init__(self, data_path):
        self.data_path = data_path
        self.data = pd.read_csv(f'{self.data_path}.csv')

        self.slide = self.data['slide']
        self.tile_paths = {
            's': self.data['tile-s'] if 'tile-s' in self.data.columns else None,
            'm': self.data['tile-m'] if 'tile-m' in self.data.columns else None,
            'l': self.data['tile-l'] if 'tile-l' in self.data.columns else None
        }
        self.patch_nums = self.data['patch-num'] if 'patch-num' in self.data.columns else None

        self.fea_paths = self.data['feature-matrix'] if 'feature-matrix' in self.data.columns else None
        self.cooadj_paths = self.data['cooadj-matrix'] if 'cooadj-matrix' in self.data.columns else None

        self.label_data = self.data['label']
        self.transform = transform

    def __len__(self):
        return len(self.slide)

    def __getitem__(self, index):
        slide = self.slide.iloc[index]
        patch_nums = self.patch_nums.iloc[index]
        tiles = []

        for combination_key in self.combination:
            tile_path = self.tile_paths[combination_key].iloc[index]
            tile = self.transform(Image.open(tile_path))
            tiles.append(tile)

        tile = torch.stack(tiles)  # [len(cmb), 3, 224, 224]

        # load label
        label = self.label_data.iloc[index]
        label = torch.as_tensor(label, dtype=torch.long)
        patch_nums = torch.as_tensor(patch_nums, dtype=torch.long)

        # load feature matrix and cooadj matrix
        fea_path = self.fea_paths.iloc[index] if self.fea_paths is not None else None
        cooadj_path = self.cooadj_paths.iloc[index] if self.cooadj_paths is not None else None

        feature_list = torch.load(fea_path, weights_only=True)  # [num_nodes, 304]
        cooadj_list = torch.load(cooadj_path, weights_only=True)  # [num_nodes, num_nodes]

        return slide, tile, patch_nums, label, feature_list, cooadj_list


class GCNDatasetKFold(Dataset):
    def __init__(self, mode, fold):
        self.mode = mode
        self.kf_path = '<YOUR_KF_PATH>'
        self.data = pd.read_csv(f'{self.kf_path}/{fold}_{mode}.csv')

        self.slide = self.data['slide']

        self.fea_paths = self.data['feature-matrix'] if 'feature-matrix' in self.data.columns else None
        self.cooadj_paths = self.data['cooadj-matrix'] if 'cooadj-matrix' in self.data.columns else None

        self.label_data = self.data['label']

    def __len__(self):
        return len(self.slide)

    def __getitem__(self, index):
        # load tile
        # 这里的index相当于逐行选取slide、label、tile信息
        slide = self.slide.iloc[index]

        # load label
        label = self.label_data.iloc[index]
        label = torch.as_tensor(label, dtype=torch.long)
        # print(f'label shape: {label.shape}, patch_nums shape: {patch_nums.shape}')

        # load feature matrix and cooadj matrix
        fea_path = self.fea_paths.iloc[index] if self.fea_paths is not None else None
        cooadj_path = self.cooadj_paths.iloc[index] if self.cooadj_paths is not None else None

        feature_list = torch.load(fea_path, weights_only=True)  # [num_nodes, 304]
        cooadj_list = torch.load(cooadj_path, weights_only=True)  # [num_nodes, num_nodes]

        if self.mode == 'train':
            return label, feature_list, cooadj_list
        elif self.mode == 'val':
            # 这里输出的slide 与 tile 一一对应
            return slide, label, feature_list, cooadj_list
        return None


if __name__ == '__main__':
    dataset = PathologyDatasetKFold(mode='train', combination='sml', fold=1)
    print(dataset[0])

    from torch.utils.data import DataLoader

    loader = DataLoader(dataset, batch_size=1, shuffle=False)

    for i, (tiles, label, fea_path, cooadj_path, patch_counts) in enumerate(loader):
        print(f"Batch {i}: patch counts => {patch_counts}")
        if i > 1:
            break
