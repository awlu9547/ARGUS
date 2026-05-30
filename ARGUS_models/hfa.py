import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.nn import init


class MultiScaleModel_1(nn.Module):
    def __init__(self, dim=1536, out_dim=768, n_heads=8, num_classes=3, proj_dim=128):
        super().__init__()

        # ——— 1) 交叉注意力 ——#
        self.cross_attn = nn.MultiheadAttention(dim, n_heads, dropout=0.2)

        # ——— 2) 分类头（共享）——#
        self.class_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )

        # ——— 3) 各类信息降维/归一化网络 ———#
        # 原始特征 x1,x2 投影到 proj_dim
        self.x_proj = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, proj_dim),
            nn.ReLU()
        )
        # 分类概率 prob1,prob2 投影到 proj_dim
        self.p_proj = nn.Sequential(
            nn.Linear(num_classes * 2, proj_dim),
            nn.ReLU()
        )
        # 注意力特征 attn_feat 投影到 proj_dim
        self.a_proj = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, proj_dim),
            nn.ReLU()
        )

        # ——— 4) 贡献值生成器 δ_net ——#
        # 在这里使用nn.LayerNorm对不同来源的特征进行归一化可能消除数据差异导致性能变差！
        self.delta_net = nn.Sequential(
            nn.LayerNorm(proj_dim * 4),
            nn.Linear(proj_dim * 4, 256),
            nn.ReLU(),
            nn.Linear(256, 2),
            nn.Softmax(dim=1)
        )

        # ——— 5) 可学习 α ——#
        self.alpha = nn.Parameter(torch.tensor(0.5))

        # ——— 6) 融合后特征调整 ——#
        self.fusion_proj = nn.Linear(dim, out_dim) if out_dim else nn.Identity()

    def forward(self, x1, x2):
        """
        x1,x2: [B, dim]
        返回：
          x3: 融合特征 [B, dim]
          logits_fuse: 融合分类 logits [B, num_classes]
          aux: (logits1, logits2, delta_weights)
        """
        B, _ = x1.shape

        # 1) 各分支独立分类 logits & prob
        logits1 = self.class_head(x1)  # [B, C]
        logits2 = self.class_head(x2)  # [B, C]
        prob1 = F.softmax(logits1, dim=1)  # [B, C]
        prob2 = F.softmax(logits2, dim=1)  # [B, C]

        # 2) 交叉注意力特征
        feat_cat = torch.stack([x1, x2], dim=1)  # [B,2,dim]
        attn_out, _ = self.cross_attn(feat_cat, feat_cat, feat_cat)  # [B,2,dim]
        attn_feat = attn_out.mean(dim=1)  # [B,dim]

        # 3) 各类信息降维+归一化
        x1_p = self.x_proj(x1)  # [B,proj_dim]
        x2_p = self.x_proj(x2)  # [B,proj_dim]
        p_p = self.p_proj(torch.cat([prob1, prob2], dim=1))  # [B,proj_dim]
        a_p = self.a_proj(attn_feat)  # [B,proj_dim]

        # 4) 贡献权重 δ1,δ2
        delta_input = torch.cat([x1_p, x2_p, p_p, a_p], dim=1)  # [B, proj_dim*4]
        # 如果只想用三种信息可以删掉 one of x_proj/p_proj/a_proj
        delta_weights = self.delta_net(delta_input)  # [B,2]
        w1, w2 = delta_weights.split(1, dim=1)  # [B,1], [B,1]
        # print(f"w1: {w1}, w2: {w2}")

        # 5) 最终融合特征 x3
        α = torch.sigmoid(self.alpha)
        x3 = w1 * x1 + w2 * x2 + α * attn_feat  # [B,dim]
        x3 = self.fusion_proj(x3)

        return x3


class MultiScaleModel_2(nn.Module):
    """
    在MultiScaleModel_1的基础上去除分类概率的计算，冗余？
    proj_dim=128，linear（256*3，256）——————best AUC：
    """
    def __init__(self, dim=1536, out_dim=None, n_heads=4, proj_dim=128):
        super().__init__()

        # ——— 交叉注意力 ——#
        self.cross_attn = nn.MultiheadAttention(dim, n_heads, dropout=0.2)

        # ———  各类信息降维/归一化网络 ———#
        # 原始特征 x1,x2 投影到 proj_dim
        self.x_proj = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, proj_dim),
            nn.ReLU()
        )

        # 注意力特征 attn_feat 投影到 proj_dim
        self.a_proj = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, proj_dim),
            nn.ReLU()
        )

        # ——— 4) 贡献值生成器 δ_net ——#
        # 拼接后总长度 = proj_dim * 3
        self.delta_net = nn.Sequential(
            # nn.LayerNorm(proj_dim * 3),
            nn.Linear(proj_dim * 3, 256),
            nn.ReLU(),
            nn.Linear(256, 2),
            nn.Softmax(dim=1)
        )

        # ——— 5) 可学习 α ——#
        self.alpha = nn.Parameter(torch.tensor(0.5))

        # ——— 6) 融合后特征调整 ——#
        self.fusion_proj = nn.Linear(dim, out_dim) if out_dim else nn.Identity()

    def forward(self, x1, x2):
        """
        x1,x2: [B, dim]
        返回：
          x3: 融合特征 [B, dim]
          logits_fuse: 融合分类 logits [B, num_classes]
          aux: (logits1, logits2, delta_weights)
        """
        B, _ = x1.shape

        # 交叉注意力特征
        feat_cat = torch.stack([x1, x2], dim=1)  # [B,2,dim]
        attn_out, _ = self.cross_attn(feat_cat, feat_cat, feat_cat)  # [B,2,dim]
        attn_feat = attn_out.mean(dim=1)  # [B,dim]

        # 各类信息降维+归一化
        x1_p = self.x_proj(x1)  # [B,proj_dim]
        x2_p = self.x_proj(x2)  # [B,proj_dim]
        a_p = self.a_proj(attn_feat)  # [B,proj_dim]

        # 4) 贡献权重 δ1,δ2
        delta_input = torch.cat([x1_p, x2_p, a_p], dim=1)  # [B, proj_dim*4]
        # 如果你只想用三种信息可以删掉 one of x_proj/p_proj/a_proj
        delta_weights = self.delta_net(delta_input)  # [B,2]
        w1, w2 = delta_weights.split(1, dim=1)  # [B,1], [B,1]
        # print(f"w1: {w1}, w2: {w2}")

        # 5) 最终融合特征 x3
        α = torch.sigmoid(self.alpha)
        x3 = w1 * x1 + w2 * x2 + α * attn_feat  # [B,dim]
        x3 = self.fusion_proj(x3)

        return x3


class MultiScaleModel_3(nn.Module):
    """
    在MultiScaleModel_1的基础上去除α学习因子控制attn_feat的占比权重，
    而是采用gate门控机制分别对多尺度特征以及交互注意力特征赋予权重
    """
    def __init__(self, dim=1536, proj_dim=128, out_dim=None, n_heads=4, num_classes=3):
        super().__init__()
        # —— 独立分类头（共享） ——
        self.class_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )

        # ——— 交叉注意力 ——#
        self.cross_attn = nn.MultiheadAttention(dim, n_heads, dropout=0.2)

        # —— 投影特征 ——
        self.x_proj = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, proj_dim),
            nn.ReLU()
        )
        self.a_proj = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, proj_dim),
            nn.ReLU()
        )

        # —— 分类概率投影 ——
        self.p_proj = nn.Sequential(
            nn.Linear(num_classes * 2, proj_dim),
            nn.ReLU()
        )

        # —— 权重生成（单层） ——
        self.delta_net = nn.Sequential(
            nn.LayerNorm(proj_dim * 4),
            nn.Linear(proj_dim * 4, 512),
            nn.ReLU(),
            nn.Linear(512, 2),
            nn.Softmax(dim=1)
        )

        # —— 可学习融合门控 ——
        self.gate = nn.Parameter(torch.tensor(0.5))

        self.fusion_proj = nn.Linear(dim, out_dim) if out_dim else nn.Identity()

    def forward(self, x1, x2):
        # 1) 使用共享分类头计算各分支分类概率
        log1 = self.class_head(x1)
        log2 = self.class_head(x2)
        p1, p2 = F.softmax(log1, 1), F.softmax(log2, 1)

        # 2) 注意力交互
        feat_cat = torch.stack([x1, x2], dim=1)  # [B,2,dim]
        attn_out, _ = self.cross_attn(feat_cat, feat_cat, feat_cat)  # [B,2,dim]
        attn_feat = attn_out.mean(dim=1)  # [B,dim]

        # 3) 各分支线性投影
        x1_p = self.x_proj(x1)
        x2_p = self.x_proj(x2)
        p_p = self.p_proj(torch.cat([p1, p2], dim=1))
        a_p = self.a_proj(attn_feat)

        # 4) 计算各分支权重
        dw = self.delta_net(torch.cat([x1_p, x2_p, p_p, a_p], dim=1))  # [B,2]
        # print(f"dw: {dw}")
        w1, w2 = dw[:, 0:1], dw[:, 1:2]
        # print(f"w1: {w1}, w2: {w2}")

        # 5) 门控融合（动态计算 分支加权和 与 注意力特征的关键性）
        gate = torch.sigmoid(self.gate)
        # print(f"gate: {gate}")

        # 关于为何同时使用多尺度特征以及注意力交互特征？
        # 多尺度特征：有些场景依赖高尺度或多尺度，而非共同特征
        # 注意力交互特征：有些复杂病理场景中，不仅仅依赖单一尺度，而是需要多尺度下共同出现的关键特征
        x3 = gate * (w1 * x1 + w2 * x2) + (1 - gate) * attn_feat
        x3 = self.fusion_proj(x3)

        return x3


class MultiScaleModel_4(nn.Module):
    """
    在MultiScaleModel_3的基础上去除分类概率的计算，冗余？
    proj_dim=256，linear（256*3，512）——————best AUC：
    proj_dim=256，linear（256*3，256）——————best AUC：
    """
    def __init__(self, dim=1536, proj_dim=256, out_dim=None, n_heads=4):
        super().__init__()
        # ——— 交叉注意力 ——#
        self.cross_attn = nn.MultiheadAttention(dim, n_heads, dropout=0.2)

        # —— 投影特征 ——
        self.x_proj = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, proj_dim),
            nn.ReLU()
        )
        self.a_proj = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, proj_dim),
            nn.ReLU()
        )

        # —— 权重生成（单层） ——
        self.delta_net = nn.Sequential(
            nn.Linear(proj_dim * 3, 256),
            nn.ReLU(),
            nn.Linear(256, 2),
            nn.Softmax(dim=1)
        )

        # —— 可学习融合门控 ——
        self.gate = nn.Parameter(torch.tensor(0.5))

        self.fusion_proj = nn.Linear(dim, out_dim) if out_dim else nn.Identity()

    def forward(self, x1, x2):

        # 2) 注意力交互
        feat_cat = torch.stack([x1, x2], dim=1)  # [B,2,dim]
        attn_out, _ = self.cross_attn(feat_cat, feat_cat, feat_cat)  # [B,2,dim]
        attn_feat = attn_out.mean(dim=1)  # [B,dim]

        # 3) 各分支线性投影
        x1_p = self.x_proj(x1)
        x2_p = self.x_proj(x2)
        a_p = self.a_proj(attn_feat)

        # 4) 计算各分支权重
        dw = self.delta_net(torch.cat([x1_p, x2_p, a_p], dim=1))  # [B,2]
        # print(f"dw: {dw}")
        w1, w2 = dw[:, 0:1], dw[:, 1:2]
        # print(f"w1: {w1}, w2: {w2}")

        # 5) 门控融合（动态计算 分支加权和 与 注意力特征的关键性）
        gate = torch.sigmoid(self.gate)
        # print(f"gate: {gate}")

        # 关于为何同时使用多尺度特征以及注意力交互特征？
        # 多尺度特征：有些场景依赖高尺度或多尺度，而非共同特征
        # 注意力交互特征：有些复杂病理场景中，不仅仅依赖单一尺度，而是需要多尺度下共同出现的关键特征
        x3 = gate * (w1 * x1 + w2 * x2) + (1 - gate) * attn_feat
        x3 = self.fusion_proj(x3)

        return x3


class MultiScaleModel_5(nn.Module):
    """
    在delta_net中拼接x1，x2，attn_feat容易加入冗余信息，反而会降低性能
    """
    def __init__(self, dim=1536, out_dim=None, n_heads=4):
        super().__init__()
        # ——— 交叉注意力 ——#
        self.cross_attn = nn.MultiheadAttention(dim, n_heads, dropout=0.2)

        # —— 特征注意力模块（Conv + Pool + Sigmoid）—— #
        self.att_conv = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim // 2),
            nn.ReLU(),
            nn.Linear(dim // 2, 1),
            nn.Sigmoid()
        )

        # —— 可学习融合门控 ——
        self.gate = nn.Parameter(torch.tensor(0.5))

        self.fusion_proj = nn.Linear(dim, out_dim) if out_dim else nn.Identity()

    def forward(self, x1, x2):
        # 注意力交互
        feat_cat = torch.stack([x1, x2], dim=1)  # [B,2,dim]
        attn_out, _ = self.cross_attn(feat_cat, feat_cat, feat_cat)  # [B,2,dim]
        attn_feat = attn_out.mean(dim=1)  # [B,dim]

        # 计算各分支权重
        attention1 = self.att_conv(x1)
        attention2 = self.att_conv(x2)
        sum_attention = attention1 + attention2

        attention1 = attention1 / sum_attention
        attention2 = attention2 / sum_attention
        # print(f"attention1:{attention1}, attention2:{attention2}")

        # 5) 门控融合（动态计算 分支加权和 与 注意力特征的关键性）
        gate = torch.sigmoid(self.gate)
        # print(f"gate: {gate}")

        # 关于为何同时使用多尺度特征以及注意力交互特征？
        # 多尺度特征：有些场景依赖高尺度或多尺度，而非共同特征
        # 注意力交互特征：有些复杂病理场景中，不仅仅依赖单一尺度，而是需要多尺度下共同出现的关键特征
        x3 = gate * (attention1 * x1 + attention2 * x2) + (1 - gate) * attn_feat
        x3 = self.fusion_proj(x3)

        return x3


class MLPMixerBlock(nn.Module):
    def __init__(self, num_tokens=2, dim=1536, token_mlp_dim=256, channel_mlp_dim=768, num_layers=2):
        super().__init__()
        self.layers = nn.ModuleList()

        for _ in range(num_layers):  # 堆叠多层
            self.layers.append(nn.ModuleDict({
                'norm1': nn.LayerNorm(dim),
                'token_mlp': nn.Sequential(
                    nn.Linear(num_tokens, token_mlp_dim),
                    nn.GELU(),
                    nn.Linear(token_mlp_dim, token_mlp_dim),
                    nn.GELU(),
                    nn.Dropout(0.2),
                    nn.Linear(token_mlp_dim, num_tokens)
                ),
                'norm2': nn.LayerNorm(dim),
                'channel_mlp': nn.Sequential(
                    nn.Linear(dim, channel_mlp_dim),
                    nn.GELU(),
                    nn.Linear(channel_mlp_dim, channel_mlp_dim),
                    nn.GELU(),
                    nn.Dropout(0.2),
                    nn.Linear(channel_mlp_dim, dim)
                )
            }))

    def forward(self, x):
        # x: [B, N, C]，两个token（尺度），每个C=dim维度[B, 2, 1536]
        for layer in self.layers:
            # Token mixing
            y = layer['norm1'](x)
            y = y.transpose(1, 2)
            y = layer['token_mlp'](y)
            y = y.transpose(1, 2)
            x = x + y

            # Channel mixing
            y = layer['norm2'](x)
            y = layer['channel_mlp'](y)
            x = x + y
        return x


class MultiScaleModel_6(nn.Module):
    """
    在先前通过cat x1，x2输入nn.MultiheadAttention(),只是对拼接后的特征进行自注意力，本身只有两个token，
    而对于注意力机制的优势是处理多tokens，此时的token只有两个，意义不大。
    这里采用MLP-Mixer机制对于此处只有两个尺度特征向量更有优势，更好的在token与channel上交互
    是采用mixed_feat = self.mlp_mixer(feat_cat)[:, 0, :]与[:, 2, :]获取新的x1，x2，然后再使用注意力加权
    还是对mixed_feat取平均，然后使用门控？
    方案1：
    """
    def __init__(self, dim=1536, out_dim=None):
        super().__init__()
        self.mlp_mixer = MLPMixerBlock(num_tokens=2, dim=dim)

        self.att_conv = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim // 2),
            nn.ReLU(),
            nn.Linear(dim // 2, 1),
            nn.Sigmoid()
        )

        self.gate = nn.Parameter(torch.tensor(0.5))  # 初始化为 0.5

        self.fusion_proj = nn.Linear(dim, out_dim) if out_dim else nn.Identity()

    def forward(self, x1, x2):
        """打印经过mlp-mixer前后特征差异，检验该模块是否正确交互"""
        feat_cat = torch.stack([x1, x2], dim=1)  # [B, 2, 1536]
        mixed1_feat = self.mlp_mixer(feat_cat)
        new_x1 = mixed1_feat[:, 0, :]  # [B, 1536]
        new_x2 = mixed1_feat[:, 1, :]  # [B, 1536]
        cosine_sim_before = F.cosine_similarity(x1, x2, dim=1).mean().item()
        cosine_sim_after = F.cosine_similarity(new_x1, new_x2, dim=1).mean().item()
        # 如果降低，则特征从冗余变成互补
        # print(f"相似度 | 前：{cosine_sim_before:.3f} → 后：{cosine_sim_after:.3f}")

        # print(f"----验证mlp-mixer是否正确交互----")
        # print(f'{x1},{new_x1}\n')
        # print(f"{x2},{new_x2}\n")

        attn_weights = torch.softmax(self.att_conv(mixed1_feat), dim=1)  # [B,2,1]
        mixed_feat = (mixed1_feat * attn_weights).sum(dim=1)
        # mixed_feat = mixed1_feat.mean(dim=1)  # [B, 1536]
        # print(f"mixed feat : {mixed_feat}")

        # 通道注意力加权
        """经过多轮验证打印发现两个attention值差异较大(attention >> attention)，说明该模块没问题"""
        attention1 = self.att_conv(x1)
        attention2 = self.att_conv(x2)
        sum_attention = attention1 + attention2
        attention1 = attention1 / sum_attention
        attention2 = attention2 / sum_attention
        # print(f'***attention1: {attention1}, attention2: {attention2}***')

        weighted_sum = attention1 * x1 + attention2 * x2

        gate = torch.sigmoid(self.gate)
        # print(f'===== gate: {gate} =====')

        x3 = (1 - gate) * weighted_sum + gate * mixed_feat

        x3 = self.fusion_proj(x3)
        return x3


class MultiScaleModel_7(nn.Module):
    """
    这里采用MLP-Mixer机制对于此处只有两个尺度特征向量更有优势，更好的在token与channel上交互
    是采用mixed_feat = self.mlp_mixer(feat_cat)[:, 0, :]与[:, 2, :]获取新的x1，x2，然后再使用注意力加权
    还是对mixed_feat取平均，然后使用门控？
    方案2： max auc=0.8713
    """
    def __init__(self, dim=1536, out_dim=None):
        super().__init__()
        self.mlp_mixer = MLPMixerBlock(num_tokens=2, dim=dim)

        self.weight_learner = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim // 2),
            nn.ReLU(),
            nn.Linear(dim // 2, 1),
            nn.Sigmoid()
        )

        self.fusion_proj = nn.Linear(dim, out_dim) if out_dim else nn.Identity()

    def forward(self, x1, x2):
        feat_cat = torch.stack([x1, x2], dim=1)  # [B, 2, 1536]
        mixed_feat = self.mlp_mixer(feat_cat)  # [B, 2, 1536]

        new_x1 = mixed_feat[:, 0, :]  # [B, 1536]
        new_x2 = mixed_feat[:, 1, :]  # [B, 1536]

        # 通道注意力加权
        attention1 = self.weight_learner(new_x1)
        attention2 = self.weight_learner(new_x2)
        sum_attention = attention1 + attention2
        attention1 = attention1 / sum_attention
        attention2 = attention2 / sum_attention

        # 门控融合
        x3 = attention1 * x1 + attention2 * x2
        x3 = self.fusion_proj(x3)

        return x3


def init_weights(m):
    if isinstance(m, nn.Linear):
        init.kaiming_uniform_(m.weight, nonlinearity='relu')
        if m.bias is not None:
            init.zeros_(m.bias)
    elif isinstance(m, nn.MultiheadAttention):
        init.xavier_uniform_(m.in_proj_weight)
        init.xavier_uniform_(m.out_proj.weight)
        if m.in_proj_bias is not None:
            init.zeros_(m.in_proj_bias)
        if m.out_proj.bias is not None:
            init.zeros_(m.out_proj.bias)


class CrossAttentionBlock(nn.Module):
    def __init__(self, dim=1536, num_heads=8):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            batch_first=True,
            kdim=dim,
            vdim=dim
        )
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        # Feed-Forward Network，使用GELU学习特征非线性关系
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim // 2),
            nn.GELU(),
            nn.Linear(dim // 2, dim)
        )
        self.apply(init_weights)

    def forward(self, x1, x2):
        x1_norm = self.norm1(x1)
        # 缩放注意力分数
        attn_out, _ = self.cross_attn(
            query=x1_norm,
            key=x2,
            value=x2
        )
        x = attn_out + x1
        x = x + self.ffn(self.norm2(x))

        return x.squeeze(1)


class MultiScaleModel_8(nn.Module):
    def __init__(self, dim=1536, out_dim=None):
        super().__init__()
        self.cross_attn_1 = CrossAttentionBlock(dim)
        self.cross_attn_2 = CrossAttentionBlock(dim)

        # 动态权重学习
        self.weight_learner = nn.Sequential(
            nn.Linear(dim * 3, dim * 2),
            nn.LayerNorm(dim * 2),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(dim * 2, dim),
            nn.GELU(),
            nn.Linear(dim, 2),
            nn.Softmax(dim=1)
        )

        self.fusion_proj = nn.Linear(dim, out_dim) if out_dim else nn.Identity()

    def forward(self, x1, x2):
        # 双向交叉注意力
        x1_to_x2 = self.cross_attn_1(x1, x2)  # x1关注x2 [B,1536]
        x2_to_x1 = self.cross_attn_2(x2, x1)  # x2关注x1 [B,1536]

        # ----检测fusion前后余弦相似度变化，降低：冗余——>互补
        cosine_sim_before = F.cosine_similarity(x1, x2, dim=1).mean().item()
        cosine_sim_after = F.cosine_similarity(x1_to_x2, x2_to_x1, dim=1).mean().item()
        # print(f"相似度 | 前：{cosine_sim_before:.3f} → 后：{cosine_sim_after:.3f}")

        # 动态权重融合,目前来说打印出来的weights两个权重有明显区分
        weight_input = torch.cat([x1, x2, x1_to_x2 + x2_to_x1], dim=1)
        weights = self.weight_learner(weight_input)  # [B,2]
        # print(weights[:, 0:1])
        # print(weights[:, 1:2])

        x3 = weights[:, 0:1] * x1_to_x2 + weights[:, 1:2] * x2_to_x1
        x3 = self.fusion_proj(x3)

        return x3


if __name__ == '__main__':
    device = torch.device("cpu")
    print(f"current state device is {device}")
    model = MultiScaleModel_1().to(device)
    x1 = torch.randn(32, 1536)
    x2 = torch.randn(32, 1536)
    output = model(x1, x2)  # X3:[B,1536]
    print(output['x3'],
          output['prob1'],
          output['prob2'],
          output['delta'])
