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


class sml_UNI(nn.Module):
    def __init__(self, n_classes, freeze_ratio, cmb, ckpt_path):
        super().__init__()

        backbone = create_model("vit_giant_patch14_224", freeze_ratio, ckpt_path=ckpt_path)

        self.cmb = cmb
        self.cmb_len = len(cmb)

        self.backbones = nn.ModuleList()
        for _ in cmb:
            self.backbones.append(deepcopy(backbone))

        self.classifier = nn.Sequential(
            nn.Linear(self.cmb_len * 1536, 1536),
            nn.GELU(),
            nn.Dropout(0.25),
            nn.Linear(1536, n_classes)
        )

    def forward(self, x):
        outputs = [backbone(x[:, i, :, :, :]) for i, backbone in enumerate(self.backbones)]
        x = torch.stack(outputs, dim=1)  # [batch_size, cmb_len, 1024]
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x
