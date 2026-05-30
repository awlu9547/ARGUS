# CLAUDE.md

本文件为 Claude Code (claude.ai/code) 在此仓库中工作时提供指引。

## 项目概述

ARGUS (A hieRarchical Geometry-gUided tranSformer) — 弱监督全切片图像 (WSI) 组织学亚型分类，用于原发性肝癌（HCC/ICC）亚型预测。核心模块：HFA（层次化视野对齐）、MGF（微观几何特征）、GPGF（几何先验引导融合）。

## 常用命令

```bash
# 安装依赖
pip install -r requirements.txt

# 创建层次化 patch
python ARGUS_utils/create_hi_patches.py --input <输入目录> --output <输出目录> --how non-blank

# 生成 k-fold 划分
python ARGUS_utils/gen_kfold_split.py --csv <CSV路径> --dir <patch目录> --k 5 --on patient

# 训练单个 fold
python train.py --fold 1

# 训练全部 fold（Windows / Linux）
python scripts/train_kf.py
sh scripts/train_kf.sh

# 评估
python evaluator.py
```

## 模型架构

### 前向流程

```
patch 图像 → UNI 骨干网络（每个尺度一个）→ HFA（层次化视野对齐）
                                                  ↓
                                            patch 特征 (768-d)
                                                  ↓
                      GCN 模块（细胞核特征 304-d + 邻接矩阵）→ 图特征 (768-d)
                                                  ↓
                                      GPGF transformer（几何先验引导融合，4 层）
                                                  ↓
                                          CLS + 均值池化 → MLP 分类器 → logits
```

### 核心组件

| 模块 | 全称 | 文件 | 说明 |
|------|------|------|------|
| **HFA** | Hierarchical FoVs Alignment | `ARGUS_models/hfa.py` | 双向交叉注意力融合多尺度 FoV 特征 |
| **MGF** | Micro-level Geometric Feature | `ARGUS_models/gcn.py` | 细胞核级几何特征（Hover-Net + GCN） |
| **GPGF** | Geometry Prior Guided Fusion | `ARGUS_models/gpgf.py` | 跨模态 Transformer 融合形态学和几何特征 |

### 尺度组合（`cmb` 配置项）

控制使用的 patch 尺度：`s`（256×256）、`m`（512×512）、`l`（1024×1024）。组合：`sm`、`sl`、`ml`（默认）、`sml`。

### 目录结构

| 路径 | 用途 |
|---|---|
| `ARGUS_models/` | 核心模型：ARGUS、HFA、GPGF、GCN、UNI 骨干工具 |
| `ARGUS_utils/` | 数据集类、评估指标、k-fold 划分、patch 创建、可视化 |
| `contrast_models/` | 基线模型（CLAM-SB、AB-MIL、TransMIL 等） |
| `contrast_experiment/` | 基线实验训练脚本和配置 |
| `Ablation_exp/` | 消融实验训练脚本 |
| `scripts/` | 交叉验证自动化 |

### 数据格式

- CSV：`name,slide,label`。`name` = 患者 ID，`slide` = 切片标识，`label` = 整数类别。
- 特征矩阵：`.pt` 文件，形状 `[num_nodes, 304]`。
- 邻接矩阵：`.pt` 文件，形状 `[num_nodes, num_nodes]`。

### 训练细节

- 优化器：AdamW（lr=2e-5, weight_decay=1e-3），CosineAnnealingLR
- 损失函数：CrossEntropyLoss
- 主要指标：切片级 macro AUC
- 输出：`runs/{cmb}_{freeze_ratio}/{fold}/`

### 配置文件

- `config.yaml`：主模型超参数
- `contrast_experiment/configs.yaml`：基线实验配置

## 项目约定

- 代码注释为中文（南京信息工程大学 IMIC 团队）。
- 无打包配置，直接作为脚本运行。
- 预处理子模块位于 `preprocess/`（WSI_Segmenter）。
