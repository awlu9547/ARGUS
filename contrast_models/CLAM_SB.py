import os.path

import torch

import torch.nn as nn
import timm

from copy import deepcopy
import torch.nn.functional as F

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


class CLAMSB_Attention(nn.Module):
    """
    Gated Attention Network
    """
    def __init__(self, L=1536, D=512, K=1):
        super().__init__()
        
        self.L = L
        self.D = D
        self.K = K
        
        self.attention_V = nn.Sequential(
            nn.Linear(self.L, self.D),
            nn.Tanh()
        )
        
        self.attention_U = nn.Sequential(
            nn.Linear(self.L, self.D),
            nn.Sigmoid()
        )
        
        self.attention_weights = nn.Linear(self.D, self.K)
        
    def forward(self, x):
        """
        x: [N, L]
        """
        A_V = self.attention_V(x)  # [N, D]
        A_U = self.attention_U(x)  # [N, D]
        A = self.attention_weights(A_V * A_U)  # element wise multiplication # [N, K]
        A = torch.transpose(A, 1, 0)  # [K, N]
        A = F.softmax(A, dim=1)  # [K, N]
        z = torch.mm(A, x)  # [K, L]
        return z, A


class CLAMSB_Classifier(nn.Module):
    """
    CLAM-SB Classifier
    """
    def __init__(self, feature_extractor, attention_net, n_classes):
        super().__init__()
        
        self.feature_extractor = feature_extractor
        self.attention_net = attention_net
        self.classifier = nn.Linear(1536, n_classes)
        
    def forward(self, x, return_attn=False):
        """
        x: [N, 3, 224, 224]
        """
        # 提取特征
        feats = []
        for i in range(x.size(0)):
            feat = self.feature_extractor(x[i].unsqueeze(0))  # [1, 1536]
            feats.append(feat)
        h = torch.cat(feats, dim=0)  # [N, 1536]
        
        # 注意力机制
        z, A = self.attention_net(h)  # [1, 1536], [1, N]
        
        # 分类
        logits = self.classifier(z)  # [1, n_classes]
        
        if return_attn:
            return logits, A
        return logits


class CLAM_SB_m(nn.Module):
    def __init__(self, n_classes, freeze_ratio, cmb, ckpt_path, use_multiscale=False):
        super().__init__()

        backbone = create_model("vit_giant_patch14_224", freeze_ratio, ckpt_path=ckpt_path)

        self.cmb = cmb
        self.cmb_len = len(cmb)

        self.backbones = nn.ModuleList()
        for _ in cmb:
            self.backbones.append(deepcopy(backbone))

        self.use_multiscale = use_multiscale

        # 修改为CLAM-SB结构
        self.attention_nets = nn.ModuleList()
        for _ in cmb:
            attn_net = CLAMSB_Attention()
            self.attention_nets.append(attn_net)
        
        self.classifiers = nn.ModuleList()
        for _ in cmb:
            clf = nn.Linear(1536, n_classes)
            self.classifiers.append(clf)

    def forward(self, tile, return_attn=False):
        feats = []
        attentions = []
        
        for i, (backbone, attn_net, clf) in enumerate(zip(self.backbones, self.attention_nets, self.classifiers)):
            # 处理单个scale
            scale_tile = tile[:, i, :, :, :]  # [B, 3, 224, 224]
            B = scale_tile.size(0)
            
            # 提取特征
            feat = []
            for b in range(B):
                f = backbone(scale_tile[b].unsqueeze(0))  # [1, 1536]
                feat.append(f)
            h = torch.cat(feat, dim=0)  # [B, 1536]
            
            # 注意力机制
            z, A = attn_net(h)  # [1, 1536], [1, B]
            
            # 分类
            logits = clf(z)  # [1, n_classes]
            
            feats.append(logits)
            attentions.append(A)

        # 多尺度融合
        if self.use_multiscale and len(feats) > 1:
            logits = torch.mean(torch.stack(feats), dim=0)
            A = torch.mean(torch.stack(attentions), dim=0)
        else:
            logits = feats[0]
            A = attentions[0]

        if return_attn:
            return logits, A
        return logits