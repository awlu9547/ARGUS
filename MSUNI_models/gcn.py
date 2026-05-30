from torch import nn
import torch.nn.functional as F
import torch
import torch.nn.init as init


class GraphConvolution(nn.Module):
    def __init__(self, input_dim, output_dim, use_bias=True):
        """一层图卷积网络

        完整GCN函数
        f = sigma(D^-1/2 A D^-1/2 * H * W)
        卷积公式 = D^-1/2 A D^-1/2 * H * W

        adjacency = D^-1/2 A D^-1/2 已经经过归一化，标准化的拉普拉斯矩阵

        Args:
        ----------
            input_dim: int
                节点输入特征的维度
            output_dim: int
                输出特征维度
            use_bias : bool, optional
                是否使用偏置
        """
        super(GraphConvolution, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.use_bias = use_bias

        # 定义GCN层的 W 权重形状
        self.weight = nn.Parameter(torch.Tensor(input_dim, output_dim))

        # 定义GCN层的 b 权重矩阵
        if self.use_bias:
            self.bias = nn.Parameter(torch.Tensor(output_dim))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    # 这里才是声明初始化 nn.Module 类里面的W,b参数
    def reset_parameters(self):
        init.kaiming_uniform_(self.weight)
        if self.use_bias:
            init.zeros_(self.bias)

    def forward(self, feature_matrix, adjacency):
        """邻接矩阵是稀疏矩阵，因此在计算时使用稀疏矩阵乘法

        Args:
        -------
            adjacency: torch.sparse.FloatTensor
                邻接矩阵
            input_feature: torch.Tensor
                输入特征
        """
        support = torch.mm(feature_matrix, self.weight)  # 矩阵相乘 m由matrix缩写
        output = torch.sparse.mm(adjacency, support)  # sparse稀疏矩阵运算
        if self.use_bias:
            output += self.bias  # bias 偏置
        return output

    # 一般是为了打印类实例的信息而重写的内置函数
    def __repr__(self):
        return self.__class__.__name__ + ' (' \
            + str(self.input_dim) + ' -> ' \
            + str(self.output_dim) + ')'


class AttentionPool(nn.Module):
    def __init__(self, feat_dim):
        super().__init__()
        self.attn_net = nn.Sequential(
            nn.Linear(feat_dim, 16),
            nn.Tanh(),
            nn.Linear(16, 1)
        )

    def forward(self, x):
        attn_scores = F.softmax(self.attn_net(x), dim=0)  # (N,1)
        return torch.sum(x * attn_scores, dim=0)  # (feat_dim,)


class GCNModel(nn.Module):
    """
    定义一个包含两层GraphConvolution的模型
    """
    def __init__(self, input_dim, out_dim):
        super(GCNModel, self).__init__()
        self.gcn1 = GraphConvolution(input_dim, 128)
        self.bn1 = nn.BatchNorm1d(128)  # 批归一化加速收敛
        self.gcn2 = GraphConvolution(128, 64)
        self.bn2 = nn.BatchNorm1d(64)

        self.pool = AttentionPool(64)
        self.proj = nn.Sequential(
            nn.Linear(64, out_dim),
            nn.ReLU(),
            nn.Dropout(0.25)
        )

    def forward(self, feature_list, adj_list):
        batch_output = []
        for fea, adj in zip(feature_list, adj_list):
            # 添加维度验证
            assert fea.dim() == 2, f"特征矩阵维度错误，应为2维，实际得到：{fea.shape}"
            assert adj.dim() == 2, f"邻接矩阵维度错误，应为2维，实际得到：{adj.shape}"
            assert fea.size(0) == adj.size(0), f"节点数不匹配：特征矩阵{fea.shape} vs 邻接矩阵{adj.shape}"

            # 第一层GCN + 批归一化
            fea = F.relu(self.bn1(self.gcn1(fea, adj)))  # (num_nodes, hidden_dim1)
            fea = F.dropout(fea, p=0.3, training=self.training)

            # 第二层GCN
            fea = F.relu(self.bn2(self.gcn2(fea, adj)))  # (num_nodes, hidden_dim2)

            # 全局平均池化（太暴力容易丢失信息）
            # graph_pred = x.mean(dim=0)
            # batch_output.append(graph_pred)

            # 注意力池化 + 分类
            graph_embed = self.pool(fea)
            graph_feat = self.proj(graph_embed)  # shape: [out_dim]
            batch_output.append(graph_feat)

        # Stack all graph outputs and zero pooling loss
        return torch.stack(batch_output)