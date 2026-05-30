import os.path

import torch

import torch.nn as nn
import timm

from copy import deepcopy


def initialize_weights(model_name):
    # Initialize weights
    for m in model_name.modules():
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)


def freeze_params(params):
    for param in params:
        param.requires_grad = False


def freeze_layers(vit_model, freeze_blocks_ratio, freeze_patch_embed=True,
                  freeze_norm=True):
    layers = list(vit_model.blocks)
    num_layers_to_freeze = int(len(layers) * freeze_blocks_ratio)
    print(f'Freezing {num_layers_to_freeze} ({100 * freeze_blocks_ratio} %) ViT blocks')
    for layer in layers[:num_layers_to_freeze]:
        freeze_params(layer.parameters())
    if freeze_patch_embed:
        freeze_params(vit_model.patch_embed.parameters())
    if freeze_norm:
        freeze_params(vit_model.norm.parameters())


def create_model(model_name, freeze_ratio, ckpt_path=None):
    timm_kwargs = {
        'img_size': 224,
        'patch_size': 14,
        'depth': 24,
        'num_heads': 24,
        'init_values': 1e-5,
        'embed_dim': 1536,
        'mlp_ratio': 2.66667 * 2,
        'num_classes': 0,
        'no_embed_class': True,
        'mlp_layer': timm.layers.SwiGLUPacked,
        'act_layer': torch.nn.SiLU,
        'reg_tokens': 8,
        'dynamic_img_size': True
    }
    uni_model = timm.create_model(model_name, pretrained=False, **timm_kwargs)

    if ckpt_path is not None:
        uni_model.load_state_dict(torch.load(ckpt_path, map_location=torch.device("cpu"), weights_only=True),
                                  strict=True)
        print(f'Loaded pretrained weights from {ckpt_path}')
    else:
        initialize_weights(uni_model)
        print('No pretrained weights loaded. Initialized weights.')
    if 1 >= freeze_ratio > 0:
        freeze_layers(uni_model, freeze_ratio)
    return uni_model


def get_trainable_params(_model):
    return sum(p.numel() for p in _model.parameters() if p.requires_grad)


from hfa import MultiScaleModel_1, MultiScaleModel_4, MultiScaleModel_5, MultiScaleModel_6, \
    MultiScaleModel_7, \
    MultiScaleModel_8


class ml_UNI(nn.Module):
    def __init__(self, n_classes, freeze_ratio, cmb, ckpt_path, use_multiscale=True):
        super().__init__()

        backbone = create_model("vit_giant_patch14_224", freeze_ratio, ckpt_path=ckpt_path)

        self.cmb = cmb
        self.cmb_len = len(cmb)

        self.backbones = nn.ModuleList()
        for _ in cmb:
            self.backbones.append(deepcopy(backbone))

        self.use_multiscale = use_multiscale

        self.classifier1 = nn.Sequential(
            nn.Linear(1536, 768),
            nn.GELU(),
            nn.Dropout(0.25),
            nn.Linear(768, n_classes)
        )

        self.classifier2 = nn.Sequential(
            nn.Linear(self.cmb_len * 1536, 1536),
            nn.GELU(),
            nn.Dropout(0.25),
            nn.Linear(1536, n_classes)
        )

        if use_multiscale:
            self.ms_model = MultiScaleModel_8(dim=1536, out_dim=None)

    def forward(self, tile, return_features=False):
        # patch_num为tensor格式，其中包含一个batch中各个样本的patch num
        feats = []
        for i, backbone in enumerate(self.backbones):
            # 假设 backbone(x_i) 返回 [B, 1536] 的特征向量
            feat = backbone(tile[:, i, :, :, :])
            feats.append(feat)

        cat_feats = torch.cat(feats, dim=1)  # [B, 1536*len(cmb)] 融合前拼接

        # 如果只用两个分支做 multi-scale 融合
        if self.use_multiscale:
            x1, x2 = feats  # 分别为两个分辨率的输出
            x3 = self.ms_model(x1, x2)
            out_logits = self.classifier1(x3)

        else:
            # 直接拼接所有分支特征再分类
            x_cat = torch.stack(feats, dim=1)  # [B, len(cmb), 1536]
            x_flat = x_cat.view(x_cat.size(0), -1)  # [B, len(cmb)*1536]
            out_logits = self.classifier2(x_flat)

        if return_features and self.use_multiscale:
            x3 = torch.mean(x3, dim=1)
            return out_logits, cat_feats, x3  # 融合前、融合后
        else:
            return out_logits
