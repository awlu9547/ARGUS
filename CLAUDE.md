# CLAUDE.md

本文件为 Claude Code (claude.ai/code) 在此仓库中工作时提供指引。

## 项目概述

hi-UNI (hierarchical UNI) — 弱监督全切片图像 (WSI) 分类，用于子宫内膜癌分子亚型预测。发表于 *Bioinformatics* (2025)。使用预训练 UNI (ViT-Giant, MahmoodLab) 作为骨干网络，结合层次化多尺度 patch 处理和基于 GCN 的图融合模块。源码中的内部名称为 MSUNI。

## 常用命令

```bash
# 安装依赖
pip install -r requirements.txt

# 从原始 1024x1024 patch 创建层次化 patch
python MSUNI_utils/create_hi_patches.py --input <输入目录> --output <输出目录> --how non-blank

# 生成 k-fold 划分（患者级别，防止数据泄露）
python MSUNI_utils/gen_kfold_split.py --csv <CSV路径> --dir <patch目录> --k 5 --on patient

# 训练单个 fold
python train.py --fold 1

# 训练全部 5 个 fold（Windows / Linux）
python scripts/train_kf.py
sh scripts/train_kf.sh

# 在外部测试集上评估
python evaluator.py
```

项目未配置测试套件和代码检查工具。

## 模型架构

### 前向流程

```
patch 图像 → UNI 骨干网络（每个尺度一个）→ MultiScaleModel_8（双向交叉注意力融合）
                                                  ↓
                                            patch 特征 (768-d)
                                                  ↓
                      GCN 模块（细胞核特征 304-d + 邻接矩阵）→ 图特征 (768-d)
                                                  ↓
                                      CATfusion transformer（多模态交叉注意力，4 层）
                                                  ↓
                                          CLS + 均值池化 → MLP 分类器 → logits
```

### 核心组件

- **UNI 骨干网络**：通过 timm 创建 `vit_giant_patch14_224`（embed_dim=1536, depth=24）。权重来自 MahmoodLab/UNI HuggingFace checkpoint。通过 `freeze_ratio` 控制冻结比例。
- **MultiScaleModel_8**（`MSUNI_models/multi_scale_feature_model.py`）：双向交叉注意力 + 动态权重学习，将两个尺度分支融合为 768-d 特征。
- **GCNModule**（`MSUNI_models/main_model.py`）：2 层 GCN（304→128→64）+ 注意力池化，处理预计算的细胞核图。
- **CATfusion**（`MSUNI_models/modeling_catfusion.py`）：Transformer 编码器（hidden=768, heads=4, layers=4, mlp_dim=3072），用于 patch 特征和图特征的跨模态融合。
- **消融变体**（`main_model.py`）：`MSUNI_Ablation_Catfusion`、`MSUNI_Ablation_MSfusion`、`GCN_branch`。

### 尺度组合（`cmb` 配置项）

控制使用的 patch 尺度：`s`（256×256）、`m`（512×512）、`l`（1024×1024）。组合：`sm`、`sl`、`ml`（默认）、`sml`。每个尺度有独立的 UNI 骨干网络副本。

### 目录结构

| 路径 | 用途 |
|---|---|
| `MSUNI_models/` | 核心模型：main_model、multi_scale_feature_model、GCN、CATfusion、UNI 骨干工具 |
| `MSUNI_utils/` | 数据集类、评估指标/验证、k-fold 划分、patch 创建、可视化 |
| `contrast_models/` | 基线模型（CLAM-SB、AB-MIL、TransMIL、sml_UNI、ml_UNI） |
| `contrast_experiment/` | 基线实验的训练脚本和配置 |
| `Ablation_exp/` | 消融实验训练脚本 |
| `scripts/` | 交叉验证自动化（train_kf.py、train_kf.sh） |

### 数据格式

- CSV：`name,slide,label`（见 `example.csv`）。`name` = 患者 ID，`slide` = 切片标识，`label` = 整数类别。
- 特征矩阵：`.pt` 文件，形状 `[num_nodes, 304]`（细胞核级特征）。
- 邻接矩阵：`.pt` 文件，形状 `[num_nodes, num_nodes]`（细胞核共现关系）。
- K-fold 划分输出到 `kf/` 目录，文件名为 `{fold}_train.csv` 和 `{fold}_val.csv`。

### 训练细节

- 优化器：AdamW（lr=2e-5, weight_decay=1e-3），CosineAnnealingLR（eta_min=1e-6）
- 损失函数：CrossEntropyLoss
- 主要指标：切片级 macro AUC（MeanPoolingMIL 聚合）
- 随机种子：42（在 train.py 中固定）
- 输出目录：`runs/{cmb}_{freeze_ratio}/{fold}/` — 最佳模型为 `{fold}_best.pth`，ROC 图，指标 CSV

### 配置文件

- `config.yaml`：主模型超参数（batch_size、lr、epochs、cmb、freeze_ratio、UNI_path）
- `contrast_experiment/configs.yaml`：基线实验独立配置

## 项目约定

- 代码注释为中文（项目来自南京信息工程大学 IMIC 团队）。
- 无打包配置（无 setup.py/pyproject.toml），直接作为脚本运行。
- 部分文件存在硬编码路径，修改时应通过 config.yaml 参数化。
- 预处理子模块位于 `preprocess/`（WSI_Segmenter），详见 `.gitmodules`。
