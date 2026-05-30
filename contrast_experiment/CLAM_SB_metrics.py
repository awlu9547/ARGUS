import time
import os
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (roc_curve, auc, accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix,classification_report)

import seaborn as sns
from matplotlib import pyplot as plt
import logging
import json

# 设置 matplotlib 的 font_manager 日志等级为 ERROR，屏蔽 font not found 的信息
logging.getLogger('matplotlib.font_manager').setLevel(logging.ERROR)
os.environ["OMP_DISPLAY_ENV"] = "FALSE"

plt.rcParams['font.family'] = 'Times New Roman'
plt.figure(figsize=(6, 5))


"""
Note: This code is for multi-class classification, not binary classification.
"""


def softmax(x):
    exp_x = np.exp(x)
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)


def plot_roc_curve(fpr_dict, tpr_dict, roc_auc, num_classes, class_names, title, save_path):
    import logging
    logging.getLogger('matplotlib.font_manager').disabled = True

    plt.figure(figsize=(6, 5.5), dpi=300)
    macro_average = 0
    for i in range(num_classes):
        plt.plot(fpr_dict[i], tpr_dict[i], lw=2, label=f"{class_names[i]} (AUC = {round(roc_auc[i], 3)})")
        macro_average += roc_auc[i]
    all_fpr = np.unique(np.concatenate([fpr_dict[i] for i in range(num_classes)]))
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(num_classes):
        mean_tpr += np.interp(all_fpr, fpr_dict[i], tpr_dict[i])
    mean_tpr /= num_classes
    fpr_dict["macro"] = all_fpr
    tpr_dict["macro"] = mean_tpr
    # 2 ways of calculating macro-average AUC
    # roc_auc["macro"] = auc(fpr_dict["macro"], tpr_dict["macro"])
    roc_auc["macro"] = macro_average / num_classes
    plt.plot(fpr_dict["macro"], tpr_dict["macro"], lw=2, label=f"Macro-average (AUC = {round(roc_auc['macro'], 3)})",
             color='purple', linestyle='--')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('1 - Specificity')  # False Positive Rate
    plt.ylabel('Sensitivity')  # True Positive Rate
    plt.title('ROC')
    plt.title(title)
    plt.legend(loc="lower right")
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f'ROC curve saved at {save_path}')
    return roc_auc['macro']


def save_fpr_tpr_auc(fpr_dict, tpr_dict, auc_dict, save_path):
    """
    将五折交叉验证中的 FPR、TPR、AUC 保存为 JSON 文件
    参数:
        fpr_dict: dict {class_id: [fpr_fold1, fpr_fold2, ..., fpr_fold5]}
        tpr_dict: dict {class_id: [tpr_fold1, tpr_fold2, ..., tpr_fold5]}
        auc_dict: dict {class_id: [auc_fold1, auc_fold2, ..., auc_fold5]}
        save_path: str, 保存路径
    """

    # 转换为可序列化的格式（ndarray -> list）
    def to_serializable(data):
        if isinstance(data, dict):
            return {str(k): to_serializable(v) for k, v in data.items()}
        elif isinstance(data, list):
            return [to_serializable(d) for d in data]
        elif isinstance(data, np.ndarray):
            return data.tolist()
        else:
            return data

    serializable_data = {
        "fpr": to_serializable(fpr_dict),
        "tpr": to_serializable(tpr_dict),
        "auc": to_serializable(auc_dict)
    }

    # 保存为 JSON 文件
    with open(save_path, 'w') as f:
        json.dump(serializable_data, f, indent=4)

    print(f"Saved FPR, TPR and AUC to {save_path}")


# slide-level AUC, make sure one name with only one label

def val_auc(model, loader, criterion, class_names, save_path, epoch, iter_idx, mil_method='mean', save_as_pdf=False):
    print(f'Validating: iter {iter_idx} ...')
    tic = time.time()
    num_classes = len(class_names)
    device = next(model.parameters()).device

    all_slide_ids = []
    all_labels = []
    all_probs = []

    model.eval()
    total_loss = 0.0

    # 新增：按slide存储所有tile
    slide_tiles = {}
    slide_labels = {}

    with torch.no_grad():
        for batch in loader:
            slide_ids, tile, patch_num, label, feature_list, cooadj_list = batch

            tile = tile.to(device)
            label = label.to(device)

            # 假设 tile shape: (B, N, C), B=batch_size=1, N=patch数量, C=feature_dim
            # output = model(tile)  # [N, C]
            # probs = torch.softmax(output, dim=1).cpu().numpy()  # [N, C]

            # 修改为按slide处理
            for i in range(len(slide_ids)):
                sid = slide_ids[i]
                if sid not in slide_tiles:
                    slide_tiles[sid] = tile[i].unsqueeze(0)  # 添加batch维度
                    slide_labels[sid] = label[i].cpu().numpy()
                else:
                    slide_tiles[sid] = torch.cat([
                        slide_tiles[sid],
                        tile[i].unsqueeze(0)
                    ], dim=0)

        # 对每个slide进行推理
        for sid in slide_tiles:
            # 获取tile数据
            x = slide_tiles[sid].to(device)  # [N, 3, 224, 224]

            # 调用CLAM-SB模型
            logits, A = model(x)  # 假设模型返回logits和注意力权重
            probs = torch.softmax(logits, dim=1).cpu().numpy()

            # 记录结果
            all_slide_ids.append(sid)
            all_labels.append(int(slide_labels[sid]))
            all_probs.append(probs)

    all_slide_ids = np.array(all_slide_ids)
    all_labels = np.array(all_labels)
    all_probs = np.concatenate(all_probs, axis=0)

    # 构造 DataFrame
    df = pd.DataFrame({
        'slide': all_slide_ids,
        'label': all_labels
    })
    for i in range(num_classes):
        df[f'prob_{i}'] = all_probs[:, i]

    # Slide-level Aggregation based on MIL method
    if mil_method == 'mean':
        df_slide = df.groupby('slide')[[f'prob_{i}' for i in range(num_classes)]].mean()
        df_slide['label_pred'] = df_slide.idxmax(axis=1).str.split('_').str[-1].astype(int)

    elif mil_method == 'max':
        df_slide = df.groupby('slide')[[f'prob_{i}' for i in range(num_classes)]].max()
        df_slide['label_pred'] = df_slide.idxmax(axis=1).str.split('_').str[-1].astype(int)

    elif mil_method == 'ABMIL':
        from models.abmil import AB_MIL  # 假设 ABMIL 模型在单独文件中定义

        assert isinstance(model, AB_MIL), "Model must be an instance of AB_MIL when using 'ABMIL'"

        # 处理每个 slide 单独运行 attention pooling
        slide_results = []

        for slide_id in np.unique(all_slide_ids):
            idxs = np.where(all_slide_ids == slide_id)[0]
            x = torch.tensor(all_probs[idxs], dtype=torch.float32).to(device)  # shape: (N, C)

            with torch.no_grad():
                result = model.forward(x, return_WSI_attn=True)
                logits = result['logits'].cpu().numpy()
                probs = torch.softmax(torch.tensor(logits), dim=1).cpu().numpy()
                attn_weights = result['WSI_attn'].cpu().numpy()

            pred_class = np.argmax(probs)
            true_label = all_labels[idxs[0]]

            slide_results.append({
                'slide': slide_id,
                'label': true_label,
                'label_pred': pred_class,
                'probs': probs[0]  # 取出该 slide 的概率向量
            })

        df_slide = pd.DataFrame(slide_results)
        df_slide.set_index('slide', inplace=True)

    elif mil_method == 'TransMIL':
        from models.transmil import TransMIL  # 假设 TransMIL 模型已定义

        assert isinstance(model, TransMIL), "Model must be an instance of TransMIL when using 'TransMIL'"

        slide_results = []

        for slide_id in np.unique(all_slide_ids):
            idxs = np.where(all_slide_ids == slide_id)[0]
            x = torch.tensor(all_probs[idxs], dtype=torch.float32).unsqueeze(0).to(device)  # shape: (1, N, C)

            with torch.no_grad():
                logits, _, _ = model(x)
                probs = torch.softmax(logits, dim=1).cpu().numpy()

            pred_class = np.argmax(probs)
            true_label = all_labels[idxs[0]]

            slide_results.append({
                'slide': slide_id,
                'label': true_label,
                'label_pred': pred_class,
                'probs': probs[0]
            })

        df_slide = pd.DataFrame(slide_results)
        df_slide.set_index('slide', inplace=True)

    else:
        raise ValueError(f"Unsupported MIL method: {mil_method}")

    # 提取真实标签和预测结果
    label_true = df_slide['label'].astype(int)
    label_pred = df_slide['label_pred'].astype(int)

    # 如果是 ABMIL / TransMIL，则 probs 来自 df_slide['probs']
    prob_cols = [f'prob_{i}' for i in range(num_classes)]
    if mil_method in ['ABMIL', 'TransMIL']:
        probs_for_auc = np.stack(df_slide['probs'].values)
    else:
        probs_for_auc = df_slide[prob_cols].values

    # 计算 metrics
    acc = accuracy_score(label_true, label_pred)
    f1 = f1_score(label_true, label_pred, average='macro')
    precision = precision_score(label_true, label_pred, average='macro')
    recall = recall_score(label_true, label_pred, average='macro')
    cm = confusion_matrix(label_true, label_pred, labels=np.arange(num_classes))

    # ROC/AUC
    fpr, tpr, roc_auc = {}, {}, {}
    for i in range(num_classes):
        bin_true = (label_true == i).astype(int)
        fpr[i], tpr[i], _ = roc_curve(bin_true, probs_for_auc[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # 保存 FPR/TPR/AUC
    current_epoch = epoch + 1
    fpr_tpr_auc_path = os.path.join(save_path, f'fpr_tpr_auc_{current_epoch}_{iter_idx}.json')
    save_fpr_tpr_auc(fpr, tpr, roc_auc, fpr_tpr_auc_path)

    # 绘制 ROC 曲线
    ext = 'pdf' if save_as_pdf else 'png'
    roc_path = os.path.join(save_path, f'slide_{current_epoch}_{iter_idx}.{ext}')
    slide_auc = plot_roc_curve(
        fpr, tpr, roc_auc,
        num_classes, class_names,
        title='Slide-level ROC',
        save_path=roc_path
    )

    print(f'Validation time: {time.time()-tic:.2f}s, Loss={total_loss:.4f}')
    return {
        'slide_AUC': slide_auc,
        'slide_ACC': acc,
        'slide_F1': f1,
        'slide_Precision': precision,
        'slide_Recall': recall,
        'confusion_matrix': cm
    }