import torch
import torch.nn as nn
from torch.nn import BCEWithLogitsLoss,CrossEntropyLoss, Dropout, Softmax, Linear, Conv2d, LayerNorm
from . import configs as configs
import math


class Attention(nn.Module):
    def __init__(self, config, vis, mm=True):
        # vis: visualize? , mm: multi-modal?
        super(Attention, self).__init__()
        self.vis = vis
        self.num_attention_heads = config.transformer["num_heads"]
        self.attention_head_size = int(config.hidden_size / self.num_attention_heads)  # 768/12=64
        self.all_head_size = self.num_attention_heads * self.attention_head_size  # 12*64=768

        # patch_embedding QKV
        self.query_patch = Linear(config.hidden_size, self.all_head_size)
        self.key_patch = Linear(config.hidden_size, self.all_head_size)
        self.value_patch = Linear(config.hidden_size, self.all_head_size)

        # gate attention
        self.gate_patch = nn.Parameter(torch.tensor(0.5))
        self.gate_graph = nn.Parameter(torch.tensor(0.5))

        if mm:
            # graph_embedding QKV
            self.query_graph = Linear(config.hidden_size, self.all_head_size)
            self.key_graph = Linear(config.hidden_size, self.all_head_size)
            self.value_graph = Linear(config.hidden_size, self.all_head_size)

            self.out_graph = Linear(config.hidden_size, config.hidden_size)
            self.attn_dropout_graph = Dropout(config.transformer["attention_dropout_rate"])
            self.attn_dropout_ig = Dropout(config.transformer["attention_dropout_rate"])
            self.attn_dropout_gi = Dropout(config.transformer["attention_dropout_rate"])
            self.proj_dropout_graph = Dropout(config.transformer["attention_dropout_rate"])

        self.out = Linear(config.hidden_size, config.hidden_size)
        self.attn_dropout_patch = Dropout(config.transformer["attention_dropout_rate"])
        self.proj_dropout_patch = Dropout(config.transformer["attention_dropout_rate"])

        self.softmax = Softmax(dim=-1)

    def transpose_for_scores(self, x):
        # 输入x[batch_size,patch_feature_len,hidden_dim]
        # 重构x[batch_size,patch_feature_len,num_heads, head_size]，使不同注意力头的计算相互独立，同时保留序列信息。
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        x = x.permute(0, 2, 1, 3)  # [batch_size, num_heads, patch_feature_len, head_size]
        return x

    def forward(self, patch_embedding, graph_embedding=None):

        patch_query_layer = self.query_patch(patch_embedding)
        patch_key_layer = self.key_patch(patch_embedding)
        patch_value_layer = self.value_patch(patch_embedding)

        if graph_embedding is not None:
            graph_query_layer = self.query_graph(graph_embedding)
            graph_key_layer = self.key_graph(graph_embedding)
            graph_value_layer = self.value_graph(graph_embedding)

        query_layer1 = self.transpose_for_scores(patch_query_layer)
        key_layer1 = self.transpose_for_scores(patch_key_layer)
        value_layer1 = self.transpose_for_scores(patch_value_layer)

        if graph_embedding is not None:
            query_layer_patch = query_layer1
            key_layer_patch = key_layer1
            value_layer_patch = value_layer1

            query_layer_graph = self.transpose_for_scores(graph_query_layer)
            key_layer_graph = self.transpose_for_scores(graph_key_layer)
            value_layer_graph = self.transpose_for_scores(graph_value_layer)

        # 单向注意力
        if graph_embedding is None:
            patch_attention_scores = torch.matmul(query_layer1, key_layer1.transpose(-1, -2))
            patch_attention_scores = patch_attention_scores / math.sqrt(self.attention_head_size)
            patch_attention_probs = self.softmax(patch_attention_scores)
            weights = patch_attention_probs if self.vis else None
            attention_probs1 = self.attn_dropout_patch(patch_attention_probs)

            context_layer1 = torch.matmul(attention_probs1, value_layer1)
            context_layer1 = context_layer1.permute(0, 2, 1, 3).contiguous()
            new_context_layer_shape1 = context_layer1.size()[:-2] + (self.all_head_size,)
            context_layer1 = context_layer1.view(*new_context_layer_shape1)
            attention_output1 = self.out(context_layer1)
            attention_output1 = self.proj_dropout_patch(attention_output1)
            return attention_output1, weights

        # 双向注意力
        else:
            # patch_embedding的自注意力score
            attention_scores_patch = torch.matmul(query_layer_patch, key_layer_patch.transpose(-1, -2))
            attention_scores_graph = torch.matmul(query_layer_graph, key_layer_graph.transpose(-1, -2))

            # patch_embedding到graph_embedding的跨注意力score
            attention_scores_ig = torch.matmul(query_layer_patch, key_layer_graph.transpose(-1, -2))
            attention_scores_gi = torch.matmul(query_layer_graph, key_layer_patch.transpose(-1, -2))

            # 使用math对自注意力分数进行缩放，并通过softmax将注意力分数输出为概率分布
            attention_scores_patch = attention_scores_patch / math.sqrt(self.attention_head_size)
            attention_probs_patch = self.softmax(attention_scores_patch)

            # 如vis存在(即可视化)，则return注意力
            weights = attention_probs_patch if self.vis else None

            attention_probs_patch = self.attn_dropout_patch(attention_probs_patch)
            attention_scores_graph = attention_scores_graph / math.sqrt(self.attention_head_size)
            attention_probs_graph = self.softmax(attention_scores_graph)
            attention_probs_graph = self.attn_dropout_graph(attention_probs_graph)

            attention_scores_ig = attention_scores_ig / math.sqrt(self.attention_head_size)
            attention_probs_ig = self.softmax(attention_scores_ig)
            attention_probs_ig = self.attn_dropout_ig(attention_probs_ig)

            attention_scores_gi = attention_scores_gi / math.sqrt(self.attention_head_size)
            attention_probs_gi = self.softmax(attention_scores_gi)
            attention_probs_gi = self.attn_dropout_gi(attention_probs_gi)

            # 分别对patch，graph embedding的QK运算结果分别与V矩阵计算
            context_layer_patch = torch.matmul(attention_probs_patch, value_layer_patch)
            context_layer_patch = context_layer_patch.permute(0, 2, 1, 3).contiguous()
            context_layer_graph = torch.matmul(attention_probs_graph, value_layer_graph)
            context_layer_graph = context_layer_graph.permute(0, 2, 1, 3).contiguous()

            # 使用注意力概率对value向量进行加权求和，并调整维度顺序[batch_size, fea_len, num_heads, head_dim]
            context_layer_ig = torch.matmul(attention_probs_ig, value_layer_graph)
            context_layer_ig = context_layer_ig.permute(0, 2, 1, 3).contiguous()
            context_layer_gi = torch.matmul(attention_probs_gi, value_layer_patch)
            context_layer_gi = context_layer_gi.permute(0, 2, 1, 3).contiguous()

            # 将两个上下文张量的最后两维展平(num_head*head_dim=hidden_dim)并映射到统一特征空间
            new_context_layer_shape1 = context_layer_patch.size()[:-2] + (self.all_head_size,)
            context_layer_patch = context_layer_patch.view(*new_context_layer_shape1)
            new_context_layer_shape2 = context_layer_graph.size()[:-2] + (self.all_head_size,)
            context_layer_graph = context_layer_graph.view(*new_context_layer_shape2)

            # 将两个上下文张量的最后两维展平(num_head*head_dim=hidden_dim)并映射到统一特征空间
            new_context_layer_shapeig = context_layer_ig.size()[:-2] + (self.all_head_size,)
            context_layer_ig = context_layer_ig.view(*new_context_layer_shapeig)
            new_context_layer_shapegi = context_layer_gi.size()[:-2] + (self.all_head_size,)
            context_layer_gi = context_layer_gi.view(*new_context_layer_shapegi)

            # 对双向注意力输出进行融合与降维处理
            gate_patch_weight = torch.sigmoid(self.gate_patch)
            gate_graph_weight = torch.sigmoid(self.gate_graph)

            # 门控融合：weighted sum
            fused_patch = gate_patch_weight * context_layer_patch + (1 - gate_patch_weight) * context_layer_ig
            fused_graph = gate_graph_weight * context_layer_graph + (1 - gate_graph_weight) * context_layer_gi

            # 映射 + dropout
            attention_output_patch = self.out(fused_patch)
            attention_output_graph = self.out(fused_graph)

            attention_output_patch = self.proj_dropout_patch(attention_output_patch)
            attention_output_graph = self.proj_dropout_graph(attention_output_graph)

            # print(f'gate patch weight: {gate_patch_weight}')
            # print(f'gate graph weight: {gate_graph_weight}')

            return attention_output_patch, attention_output_graph, weights
