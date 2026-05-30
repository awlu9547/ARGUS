# coding=utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import logging
import math

from os.path import join as pjoin

import torch
import torch.nn as nn
import numpy as np

from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, Dropout, Softmax, Linear, Conv2d, LayerNorm
from torch.nn.modules.utils import _pair
from scipy import ndimage

from . import configs as configs
from .attention import Attention
from .embed import Embeddings
from .mlp import Mlp
from .block import Block


class Encoder(nn.Module):
    def __init__(self, config, vis):
        super(Encoder, self).__init__()
        self.vis = vis
        self.layer = nn.ModuleList()
        self.encoder_norm = LayerNorm(config.hidden_size, eps=1e-6)
        for i in range(config.transformer["num_layers"]):
            if i < 2:
                layer = Block(config, vis, mm=True)
            else:
                layer = Block(config, vis)
            self.layer.append(copy.deepcopy(layer))

    def forward(self, patch_embedding, graph_embedding=None):
        attn_weights = []

        for (i, layer_block) in enumerate(self.layer):
            if i == 2:
                # 这里的combined_embedding为两个不同模态在token长度上(即此处的dim 1）的拼接，而非特征维度拼接，[:-1]仍是768维
                combined_embedding = torch.cat((patch_embedding, graph_embedding), 1)
                # print(f" i = {i} before cat : patch embedding : {combined_embedding.shape}")
                combined_embedding, weights = layer_block(combined_embedding)
                # print(f" i = {i} after cat : patch embedding : {combined_embedding.shape}")
            elif i < 2:
                patch_embedding, graph_embedding, weights = layer_block(patch_embedding, graph_embedding)
                # print(f" i = {i} : patch embedding : {patch_embedding.shape}")
                # print(f" i = {i} : graph embedding : {graph_embedding.shape}")
            else:
                combined_embedding, weights = layer_block(combined_embedding)
                # print(f" i = {i} : patch embedding : {combined_embedding.shape}")

            if self.vis:
                attn_weights.append(weights)

        # print(f"before encoder_norm: {combined_embedding}")
        # 对patch_embedding进行标准化处理
        encoded = self.encoder_norm(combined_embedding)
        # print(f"encoded： {encoded.shape}")

        return encoded, attn_weights
