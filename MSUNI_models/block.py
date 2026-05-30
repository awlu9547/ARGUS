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


ATTENTION_Q = "MultiHeadDotProductAttention_1/query"
ATTENTION_K = "MultiHeadDotProductAttention_1/key"
ATTENTION_V = "MultiHeadDotProductAttention_1/value"
ATTENTION_OUT = "MultiHeadDotProductAttention_1/out"
FC_0 = "MlpBlock_3/Dense_0"
FC_1 = "MlpBlock_3/Dense_1"
ATTENTION_NORM = "LayerNorm_0"
MLP_NORM = "LayerNorm_2"


class Block(nn.Module):
    def __init__(self, config, vis, mm=False):
        super(Block, self).__init__()
        self.hidden_size = config.hidden_size

        self.attention_norm_patch = LayerNorm(config.hidden_size, eps=1e-6)
        self.ffn_norm_patch = LayerNorm(config.hidden_size, eps=1e-6)
        self.ffn_patch = Mlp(config)

        if mm:
            self.attention_norm_graph = LayerNorm(config.hidden_size, eps=1e-6)
            self.ffn_norm_graph = LayerNorm(config.hidden_size, eps=1e-6)
            self.ffn_graph = Mlp(config)

        self.attention = Attention(config, vis, mm)

    def forward(self, patch_embedding, graph_embedding=None):
        if graph_embedding is None:
            # h为输入副本，服务于残差链接
            h = patch_embedding
            patch_embedding = self.attention_norm_patch(patch_embedding)
            patch_embedding, weights = self.attention(patch_embedding)
            patch_embedding = patch_embedding + h

            # 将patch embedding进行归一化与MLP处理
            h = patch_embedding
            patch_embedding = self.ffn_norm_patch(patch_embedding)
            patch_embedding = self.ffn_patch(patch_embedding)
            patch_embedding = patch_embedding + h
            return patch_embedding, weights
        else:
            # 分别对两个多模态变量进行归一化操作
            h = patch_embedding
            h_graph_embedding = graph_embedding
            patch_embedding = self.attention_norm_patch(patch_embedding)
            graph_embedding = self.attention_norm_graph(graph_embedding)

            # 将两个多模态embeddings输入注意力block
            patch_embedding, graph_embedding, weights = self.attention(patch_embedding, graph_embedding)
            patch_embedding = patch_embedding + h
            graph_embedding = graph_embedding + h_graph_embedding

            # 将两个多模态embeddings进行归一化与MLP处理
            h = patch_embedding
            h_graph_embedding = graph_embedding
            patch_embedding = self.ffn_norm_patch(patch_embedding)
            graph_embedding = self.ffn_norm_graph(graph_embedding)
            patch_embedding = self.ffn_patch(patch_embedding)
            graph_embedding = self.ffn_graph(graph_embedding)
            patch_embedding = patch_embedding + h
            graph_embedding = graph_embedding + h_graph_embedding

            return patch_embedding, graph_embedding, weights

    # def load_from(self, weights, n_block):
    #     ROOT = f"Transformer/encoderblock_{n_block}"
    #     with torch.no_grad():
    #         query_weight = np2th(weights[pjoin(ROOT, ATTENTION_Q, "kernel")]).view(
    #             self.hidden_size, self.hidden_size).t()
    #         key_weight = np2th(weights[pjoin(ROOT, ATTENTION_K, "kernel")]).view(
    #             self.hidden_size, self.hidden_size).t()
    #         value_weight = np2th(weights[pjoin(ROOT, ATTENTION_V, "kernel")]).view(
    #             self.hidden_size, self.hidden_size).t()
    #         out_weight = np2th(weights[pjoin(ROOT, ATTENTION_OUT, "kernel")]).view(
    #             self.hidden_size, self.hidden_size).t()
    #
    #         query_bias = np2th(
    #             weights[pjoin(ROOT, ATTENTION_Q, "bias")]).view(-1)
    #         key_bias = np2th(
    #             weights[pjoin(ROOT, ATTENTION_K, "bias")]).view(-1)
    #         value_bias = np2th(
    #             weights[pjoin(ROOT, ATTENTION_V, "bias")]).view(-1)
    #         out_bias = np2th(
    #             weights[pjoin(ROOT, ATTENTION_OUT, "bias")]).view(-1)
    #
    #         self.attn.query.weight.copy_(query_weight)
    #         self.attn.key.weight.copy_(key_weight)
    #         self.attn.value.weight.copy_(value_weight)
    #         self.attn.out.weight.copy_(out_weight)
    #         self.attn.query.bias.copy_(query_bias)
    #         self.attn.key.bias.copy_(key_bias)
    #         self.attn.value.bias.copy_(value_bias)
    #         self.attn.out.bias.copy_(out_bias)
    #
    #         mlp_weight_0 = np2th(weights[pjoin(ROOT, FC_0, "kernel")]).t()
    #         mlp_weight_1 = np2th(weights[pjoin(ROOT, FC_1, "kernel")]).t()
    #         mlp_bias_0 = np2th(weights[pjoin(ROOT, FC_0, "bias")]).t()
    #         mlp_bias_1 = np2th(weights[pjoin(ROOT, FC_1, "bias")]).t()
    #
    #         self.ffn.fc1.weight.copy_(mlp_weight_0)
    #         self.ffn.fc2.weight.copy_(mlp_weight_1)
    #         self.ffn.fc1.bias.copy_(mlp_bias_0)
    #         self.ffn.fc2.bias.copy_(mlp_bias_1)
    #
    #         self.attention_norm.weight.copy_(
    #             np2th(weights[pjoin(ROOT, ATTENTION_NORM, "scale")]))
    #         self.attention_norm.bias.copy_(
    #             np2th(weights[pjoin(ROOT, ATTENTION_NORM, "bias")]))
    #         self.ffn_norm.weight.copy_(
    #             np2th(weights[pjoin(ROOT, MLP_NORM, "scale")]))
    #         self.ffn_norm.bias.copy_(
    #             np2th(weights[pjoin(ROOT, MLP_NORM, "bias")]))
