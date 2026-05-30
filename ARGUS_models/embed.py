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
import pdb


class Embeddings(nn.Module):
    """
    Construct the embeddings from Patch_embedding, graph_embedding.
    """

    def __init__(self, config):
        super(Embeddings, self).__init__()

        self.patch_embeddings = Linear(config.patch_dim, config.hidden_size)
        self.graph_embeddings = Linear(config.graph_dim, config.hidden_size)
        self.hidden_size = config.hidden_size

        self.cls_token = nn.Parameter(torch.zeros(1, 1, config.hidden_size))
        self.dropout = Dropout(config.transformer["dropout_rate"])

    def forward(self, x1, x2):
        # # 获取两个embeddings的batch_size
        # B1 = x1.shape[0]
        # B2 = x2.shape[0]

        B1, P, _ = x1.shape  # x1: [B, patch_num, patch_dim]
        B2, N, _ = x2.shape  # x2: [B, nucleus_num, graph_dim]
        device = x1.device

        # 动态生成 position embeddings
        patch_position_embeddings = nn.Parameter(torch.zeros(1, 1 + P, self.hidden_size, device=device))
        graph_position_embeddings = nn.Parameter(torch.zeros(1, 1 + N, self.hidden_size, device=device))

        # 两个不同patch维度上的CLS token
        cls1_tokens = self.cls_token.expand(B1, -1, -1)
        cls2_tokens = self.cls_token.expand(B2, -1, -1)

        # 对每个patch embedding进行线性变换
        x1 = self.patch_embeddings(x1)  # 1536=>768
        x2 = self.graph_embeddings(x2)  # 768=>768

        # 在每个patch embedding上增加CLS token
        x1 = torch.cat((cls1_tokens, x1), dim=1)
        x2 = torch.cat((cls2_tokens, x2), dim=1)

        # 在每个patch embedding上增加位置编码
        new_patch_embeddings = x1 + patch_position_embeddings
        new_graph_embeddings = x2 + graph_position_embeddings

        # 在增加位置编码后的embedding中添加正则化Dropout
        new_patch_embeddings = self.dropout(new_patch_embeddings)
        new_graph_embeddings = self.dropout(new_graph_embeddings)

        return new_patch_embeddings, new_graph_embeddings
