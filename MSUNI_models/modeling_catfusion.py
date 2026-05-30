# coding=utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging

import torch
import torch.nn as nn
import numpy as np

from scipy import ndimage

from . import configs
from .embed import Embeddings
from .encoder import Encoder

logger = logging.getLogger(__name__)


def np2th(weights, conv=False):
    """Possibly convert HWIO to OIHW."""
    if conv:
        weights = weights.transpose([3, 2, 0, 1])
    return torch.from_numpy(weights)


class Transformer(nn.Module):
    def __init__(self, config, vis):
        super(Transformer, self).__init__()
        self.embeddings = Embeddings(config)
        self.encoder = Encoder(config, vis)

    def forward(self, patch, graph=None):
        patch_embedding, graph_embedding = self.embeddings(patch, graph)
        # print(f"patch embedding: {patch_embedding}, \n graph embedding: {graph_embedding}")

        encoded, attn_weights = self.encoder(patch_embedding, graph_embedding)

        return encoded, attn_weights


class CATfusion(nn.Module):
    def __init__(self, config, num_classes=3, zero_head=True, vis=True):
        super(CATfusion, self).__init__()
        self.num_classes = num_classes
        self.zero_head = zero_head
        self.classifier = config.classifier

        self.transformer = Transformer(config, vis)
        self.mlp = nn.Sequential(
            nn.Linear(config.hidden_size * 2, config.hidden_size), nn.ReLU(), nn.Dropout(p=0.25),
            nn.Linear(config.hidden_size, 128), nn.ReLU(), nn.Dropout(p=0.25)
        )

        # 两个变量作用：对模型输出进行线性变换，通常适用于生存分析任务中
        self.output_range = torch.nn.Parameter(
            torch.FloatTensor([8]), requires_grad=False)
        self.output_shift = torch.nn.Parameter(
            torch.FloatTensor([-4]), requires_grad=False)

        self.head = nn.Sequential(
            nn.Linear(128, num_classes))

    def forward(self, patch, graph=None):
        assert len(graph) > 0, "please input at least one graph features"
        # 此时x为Encoder block编码后的特征[batch_size, feature_len, hidden_size]
        encoded, attn_weights = self.transformer(patch, graph)

        # 此时的h融合了全局(CLS token)与局部信息(mean tokens)
        h = self.mlp(torch.cat((encoded[:, 0, :], torch.mean(encoded[:, 1:, :], dim=1)), dim=1))
        # print(f"h shape:{h.shape}")

        logits = self.head(h)
        # logits = self.head(h) * self.output_range + self.output_shift
        # print("logits", logits.shape)  # [batch_size, num_classes]

        return logits

    def load_from(self, weights):
        with torch.no_grad():
            if self.zero_head:
                nn.init.zeros_(self.head.weight)
                nn.init.zeros_(self.head.bias)
            else:
                self.head.weight.copy_(np2th(weights["head/kernel"]).t())
                self.head.bias.copy_(np2th(weights["head/bias"]).t())

            self.transformer.embeddings.patch_embeddings.weight.copy_(
                np2th(weights["embedding/kernel"], conv=True))
            self.transformer.embeddings.patch_embeddings.bias.copy_(
                np2th(weights["embedding/bias"]))
            self.transformer.embeddings.cls_token.copy_(np2th(weights["cls"]))
            self.transformer.encoder.encoder_norm.weight.copy_(
                np2th(weights["Transformer/encoder_norm/scale"]))
            self.transformer.encoder.encoder_norm.bias.copy_(
                np2th(weights["Transformer/encoder_norm/bias"]))

            posemb = np2th(weights["Transformer/posembed_input/pos_embedding"])
            posemb_new = self.transformer.embeddings.position_embeddings
            # print(posemb.size(), posemb_new.size())
            if posemb.size() == posemb_new.size():
                self.transformer.embeddings.position_embeddings.copy_(posemb)
            else:
                logger.info("load_pretrained: resized variant: %s to %s" %
                            (posemb.size(), posemb_new.size()))
                ntok_new = posemb_new.size(1)

                if self.classifier == "token":
                    posemb_tok, posemb_grid = posemb[:, :1], posemb[0, 1:]
                    ntok_new -= 1
                else:
                    posemb_tok, posemb_grid = posemb[:, :0], posemb[0]

                gs_old = int(np.sqrt(len(posemb_grid)))
                gs_new = int(np.sqrt(ntok_new))
                print('load_pretrained: grid-size from %s to %s' %
                      (gs_old, gs_new))
                posemb_grid = posemb_grid.reshape(gs_old, gs_old, -1)

                zoom = (gs_new / gs_old, gs_new / gs_old, 1)
                posemb_grid = ndimage.zoom(posemb_grid, zoom, order=1)
                posemb_grid = posemb_grid.reshape(1, gs_new * gs_new, -1)
                posemb = np.concatenate([posemb_tok, posemb_grid], axis=1)
                self.transformer.embeddings.position_embeddings.copy_(
                    np2th(posemb))

            for bname, block in self.transformer.encoder.named_children():
                for uname, unit in block.named_children():
                    unit.load_from(weights, n_block=uname)

            if self.transformer.embeddings.hybrid:
                self.transformer.embeddings.hybrid_model.root.conv.weight.copy_(
                    np2th(weights["conv_root/kernel"], conv=True))
                gn_weight = np2th(weights["gn_root/scale"]).view(-1)
                gn_bias = np2th(weights["gn_root/bias"]).view(-1)
                self.transformer.embeddings.hybrid_model.root.gn.weight.copy_(
                    gn_weight)
                self.transformer.embeddings.hybrid_model.root.gn.bias.copy_(
                    gn_bias)

                for bname, block in self.transformer.embeddings.hybrid_model.body.named_children():
                    for uname, unit in block.named_children():
                        unit.load_from(weights, n_block=bname, n_unit=uname)


CONFIGS = {
    'CATfusion': configs.get_CATfusion_config(),
}

if __name__ == '__main__':
    """
    经测试，目前model输出的结果logits没有问题
    """

    configs = CONFIGS['CATfusion']
    device = torch.device("cpu")
    print(f"运行环境: {device}")

    batch_size = 1
    num_patches = 3
    num_graph_nodes = 5
    torch.manual_seed(42)

    patch_input = torch.randn(batch_size, num_patches, configs.patch_dim).to(device)
    graph_input = torch.randn(batch_size, num_graph_nodes, configs.graph_dim).to(device)

    model = CATfusion(configs, num_classes=3, zero_head=True).to(device)

    logits = model(patch_input, graph_input)

    # 打印输出
    print("\n测试输出结果:")
    print(f"输入维度:")
    print(f"  - 图像特征: {tuple(patch_input.shape)}")
    print(f"  - 图特征  : {tuple(graph_input.shape)}")
    print(f"输出维度:")
    print(f"  - logits : {tuple(logits.shape)}")
    print(f"预测概率 (未归一化):\n{logits}")
    print(f"类别预测结果: {torch.argmax(logits, dim=1)}")