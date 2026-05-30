import torch
import torch.nn as nn
import torch.nn.functional as F

class AB_MIL(nn.Module):
    def __init__(self, input_dim=512, num_classes=4, dropout=0.25):
        super(AB_MIL, self).__init__()
        self.L = input_dim  # 特征维度
        self.D = 128         # attention 隐层维度
        self.K = 1           # attention heads 数量

        # Attention Network
        self.attention = nn.Sequential(
            nn.Linear(self.L, self.D),
            nn.Tanh(),
            nn.Linear(self.D, self.K)
        )

        # Classifier
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(self.L * self.K, num_classes)
        )

        # 初始化权重
        self.apply(self._initialize_weights)

    def _initialize_weights(self, module):
        for m in module.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.weight, 1.0)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x, return_WSI_attn=False):
        """
        Args:
            x: 输入 patch 特征，shape (N, L)
        Returns:
            logits: slide-level 分类 logit，shape (1, num_classes)
            attn_weights: 返回注意力权重（可选）
        """
        A = self.attention(x)  # (N, K=1)
        A = torch.transpose(A, -1, -2)  # (K=1, N)
        A = F.softmax(A, dim=-1)  # softmax over patches

        M = torch.mm(A, x)  # (1, L)
        logits = self.classifier(M)

        output = {'logits': logits}
        if return_WSI_attn:
            output['WSI_attn'] = A.squeeze(0).cpu().numpy()

        return output