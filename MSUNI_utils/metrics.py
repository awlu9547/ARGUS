import time
import os
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (roc_curve, auc, accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix, classification_report)

import seaborn as sns
import umap
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


def UMAP(umap_concat, umap_fused, all_labels, save_path, epoch, iter_idx, option=None):
    class_names = ['Fine', 'Small', 'Large']
    palette = {0: 'blue', 1: 'orange', 2: 'green'}

    # 确保 all_labels 是 numpy 数组
    all_labels = np.array(all_labels)

    plt.figure(figsize=(14, 6))

    # 融合前
    plt.subplot(1, 2, 1)
    scatter1 = sns.scatterplot(
        x=umap_concat[:, 0],
        y=umap_concat[:, 1],
        hue=all_labels,
        palette=palette,
        s=10,
        edgecolor='none',
        legend='full'
    )
    plt.title("UNI feature distribution\n(cat fusion)", fontsize=13)
    plt.xlabel("UMAP Component 1")
    plt.ylabel("UMAP Component 2")
    handles1, _ = scatter1.get_legend_handles_labels()
    scatter1.legend(handles=handles1, labels=class_names, title="Subtypes", loc='upper right')
    plt.grid(True)

    # 融合后
    plt.subplot(1, 2, 2)
    scatter2 = sns.scatterplot(
        x=umap_fused[:, 0],
        y=umap_fused[:, 1],
        hue=all_labels,
        palette=palette,
        s=10,
        edgecolor='none',
        legend='full'
    )
    plt.title("UNI feature distribution\n(MultiScale fusion)", fontsize=13)
    plt.xlabel("UMAP Component 1")
    plt.ylabel("UMAP Component 2")
    handles2, _ = scatter2.get_legend_handles_labels()
    scatter2.legend(handles=handles2, labels=class_names, title="Subtypes", loc='upper right')
    plt.grid(True)

    current_epoch = epoch + 1
    tSNE_path = os.path.join(save_path, f'UMAP{option}_{current_epoch}_{iter_idx}.png')

    plt.tight_layout()
    plt.savefig(tSNE_path, dpi=300)


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
def val_auc(model, loader, criterion, class_names, save_path, epoch, iter_idx, save_as_pdf=False):
    print(f'Validating: iter {iter_idx} ...')
    tic = time.time()
    num_classes = len(class_names)
    device = next(model.parameters()).device

    slide_list = []
    label_list = []
    prob_list = []

    # all_concat_feats = []
    # all_fused_feats = []
    # all_labels = []

    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for batch in loader:
            slide, tile, patch_num, label, feature_list, cooadj_list = batch

            tile = tile.to(device)
            patch_num = patch_num.to(device)
            label = label.to(device)
            feature_list = [fea.to(device) for fea in feature_list]
            cooadj_list = [adj.to(device) for adj in cooadj_list]

            output, concat_feats, fused_feats = model(tile, patch_num, feature_list, cooadj_list,
                                                      return_features=True)  # [B, C]

            # all_concat_feats.append(concat_feats.cpu())
            # all_fused_feats.append(fused_feats.cpu())
            # all_labels.append(label.cpu())

            probs = torch.softmax(output, dim=1)

            loss = criterion(output, label)
            total_loss += loss.item()

            slide_list.extend(slide)
            label_list.append(label.cpu().numpy())
            prob_list.append(probs.cpu().numpy())

    # all_concat_feats = torch.cat(all_concat_feats, dim=0).numpy()
    # all_fused_feats = torch.cat(all_fused_feats, dim=0).numpy()
    # all_labels = torch.cat(all_labels, dim=0).numpy()

    """UMAP可视化特征分布"""
    # 当前n_neighbors=30,min_dist=0.2,主要强调类间区分性
    # umap_reducer1 = umap.UMAP(
    #     n_components=2,
    #     n_neighbors=30,
    #     min_dist=0.2,
    #     metric='cosine'
    # )
    # umap_concat1 = umap_reducer1.fit_transform(all_concat_feats)
    # umap_fused1 = umap_reducer1.fit_transform(all_fused_feats)
    # UMAP(umap_concat1, umap_fused1, all_labels, save_path, epoch, iter_idx, option=1)

    # flatten
    label_all = np.concatenate(label_list, axis=0)  # (N,)
    prob_all = np.concatenate(prob_list, axis=0)  # (N, C)
    N = label_all.shape[0]
    assert len(slide_list) == N

    # DataFrame
    df = pd.DataFrame({
        'slide': slide_list,
        'label': label_all
    })
    for i in range(num_classes):
        df[f'prob_{i}'] = prob_all[:, i]

    # slide-level aggregation(get mean preds to aggregate)
    """
    此处对同一slide所有patch预测取均值作为slide_level预测结果----MeanPoolingMIL
    也可对同一slide所有patch预测取最大值作为slide_level预测结果----MaxPoolingMIL
    也可对同一slide所有patch预测结果计算注意力做加权平均作为slide_level预测结果----AttentionMIL
    """
    prob_cols = [f'prob_{i}' for i in range(num_classes)]
    df_slide = df.groupby('slide')[prob_cols].mean()  # MeanPoolingMIL
    df_slide['label_pred'] = df_slide[prob_cols].idxmax(axis=1).str.split('_').str[-1].astype(int)

    # merge true label
    true_df = df[['slide', 'label']].drop_duplicates('slide').set_index('slide')
    df_slide = df_slide.join(true_df, how='left').reset_index()

    label_true = df_slide['label'].astype(int)
    label_pred = df_slide['label_pred'].astype(int)

    # metrics
    acc = accuracy_score(label_true, label_pred)
    f1 = f1_score(label_true, label_pred, average='macro')
    precision = precision_score(label_true, label_pred, average='macro')
    recall = recall_score(label_true, label_pred, average='macro')
    # 混淆矩阵中的数量和为val中所有slides和
    cm = confusion_matrix(label_true, label_pred, labels=np.arange(num_classes))

    # ROC/AUC
    fpr, tpr, roc_auc = {}, {}, {}
    for i in range(num_classes):
        bin_true = (label_true == i).astype(int)

        # fpr 与 tpr 分别表示假正例率和真正例率
        fpr[i], tpr[i], _ = roc_curve(bin_true, df_slide[f'prob_{i}'])
        roc_auc[i] = auc(fpr[i], tpr[i])

    ext = 'pdf' if save_as_pdf else 'png'
    # 在roc_path路径基础上增加epoch信息，防止png图片被覆盖错失最佳model状态曲线
    current_epoch = epoch + 1
    fpr_tpr_auc_path = os.path.join(save_path, f'fpr_tpr_auc_{current_epoch}_{iter_idx}.json')
    save_fpr_tpr_auc(fpr, tpr, roc_auc, fpr_tpr_auc_path)

    roc_path = os.path.join(save_path, f'slide_{current_epoch}_{iter_idx}.{ext}')
    slide_auc = plot_roc_curve(
        fpr, tpr, roc_auc,
        num_classes, class_names,
        title='Slide-level ROC',
        save_path=roc_path
    )

    print(f'Validation time: {time.time() - tic:.2f}s, Loss={total_loss:.4f}')
    return {
        'slide_AUC': slide_auc,
        'slide_ACC': acc,
        'slide_F1': f1,
        'slide_Precision': precision,
        'slide_Recall': recall,
        'confusion_matrix': cm
    }


def val_GCN_branch(model, loader, criterion, class_names, save_path, epoch, iter_idx, save_as_pdf=False):
    print(f'Validating: iter {iter_idx} ...')
    tic = time.time()
    num_classes = len(class_names)
    device = next(model.parameters()).device

    slide_list = []
    label_list = []
    prob_list = []

    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for batch in loader:
            slide, label, feature_list, cooadj_list = batch

            label = label.to(device)
            feature_list = [fea.to(device) for fea in feature_list]
            cooadj_list = [adj.to(device) for adj in cooadj_list]

            output = model(feature_list, cooadj_list)  # [B, C]

            probs = torch.softmax(output, dim=1)

            loss = criterion(output, label)
            total_loss += loss.item()

            slide_list.extend(slide)
            label_list.append(label.cpu().numpy())
            prob_list.append(probs.cpu().numpy())

    # flatten
    label_all = np.concatenate(label_list, axis=0)  # (N,)
    prob_all = np.concatenate(prob_list, axis=0)  # (N, C)
    N = label_all.shape[0]
    assert len(slide_list) == N

    # DataFrame
    df = pd.DataFrame({
        'slide': slide_list,
        'label': label_all
    })
    for i in range(num_classes):
        df[f'prob_{i}'] = prob_all[:, i]

    # slide-level aggregation(get mean preds to aggregate)
    """
    此处对同一slide所有patch预测取均值作为slide_level预测结果----MeanPoolingMIL
    也可对同一slide所有patch预测取最大值作为slide_level预测结果----MaxPoolingMIL
    也可对同一slide所有patch预测结果计算注意力做加权平均作为slide_level预测结果----AttentionMIL
    """
    prob_cols = [f'prob_{i}' for i in range(num_classes)]
    df_slide = df.groupby('slide')[prob_cols].mean()  # MeanPoolingMIL
    df_slide['label_pred'] = df_slide[prob_cols].idxmax(axis=1).str.split('_').str[-1].astype(int)

    # merge true label
    true_df = df[['slide', 'label']].drop_duplicates('slide').set_index('slide')
    df_slide = df_slide.join(true_df, how='left').reset_index()

    label_true = df_slide['label'].astype(int)
    label_pred = df_slide['label_pred'].astype(int)

    # metrics
    acc = accuracy_score(label_true, label_pred)
    f1 = f1_score(label_true, label_pred, average='macro')
    precision = precision_score(label_true, label_pred, average='macro')
    recall = recall_score(label_true, label_pred, average='macro')
    # 混淆矩阵中的数量和为val中所有slides和
    cm = confusion_matrix(label_true, label_pred, labels=np.arange(num_classes))

    # ROC/AUC
    fpr, tpr, roc_auc = {}, {}, {}
    for i in range(num_classes):
        bin_true = (label_true == i).astype(int)

        # fpr 与 tpr 分别表示假正例率和真正例率
        fpr[i], tpr[i], _ = roc_curve(bin_true, df_slide[f'prob_{i}'])
        roc_auc[i] = auc(fpr[i], tpr[i])

    ext = 'pdf' if save_as_pdf else 'png'
    # 在roc_path路径基础上增加epoch信息，防止png图片被覆盖错失最佳model状态曲线
    current_epoch = epoch + 1
    fpr_tpr_auc_path = os.path.join(save_path, f'fpr_tpr_auc_{current_epoch}_{iter_idx}.json')
    save_fpr_tpr_auc(fpr, tpr, roc_auc, fpr_tpr_auc_path)

    roc_path = os.path.join(save_path, f'slide_{current_epoch}_{iter_idx}.{ext}')
    slide_auc = plot_roc_curve(
        fpr, tpr, roc_auc,
        num_classes, class_names,
        title='Slide-level ROC',
        save_path=roc_path
    )

    print(f'Validation time: {time.time() - tic:.2f}s, Loss={total_loss:.4f}')
    return {
        'slide_AUC': slide_auc,
        'slide_ACC': acc,
        'slide_F1': f1,
        'slide_Precision': precision,
        'slide_Recall': recall,
        'confusion_matrix': cm
    }


def val_Ablation_Catfusion(model, loader, criterion, class_names, save_path, epoch, iter_idx, save_as_pdf=False):
    print(f'Validating: iter {iter_idx} ...')
    tic = time.time()
    num_classes = len(class_names)
    device = next(model.parameters()).device

    slide_list = []
    label_list = []
    prob_list = []

    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for batch in loader:
            slide, tile, patch_num, label, feature_list, cooadj_list = batch

            tile = tile.to(device)
            patch_num = patch_num.to(device)
            label = label.to(device)
            feature_list = [fea.to(device) for fea in feature_list]
            cooadj_list = [adj.to(device) for adj in cooadj_list]

            output = model(tile, feature_list, cooadj_list)  # [B, C]

            probs = torch.softmax(output, dim=1)

            loss = criterion(output, label)
            total_loss += loss.item()

            slide_list.extend(slide)
            label_list.append(label.cpu().numpy())
            prob_list.append(probs.cpu().numpy())

    # flatten
    label_all = np.concatenate(label_list, axis=0)  # (N,)
    prob_all = np.concatenate(prob_list, axis=0)  # (N, C)
    N = label_all.shape[0]
    assert len(slide_list) == N

    # DataFrame
    df = pd.DataFrame({
        'slide': slide_list,
        'label': label_all
    })
    for i in range(num_classes):
        df[f'prob_{i}'] = prob_all[:, i]

    # slide-level aggregation(get mean preds to aggregate)
    """
    此处对同一slide所有patch预测取均值作为slide_level预测结果----MeanPoolingMIL
    也可对同一slide所有patch预测取最大值作为slide_level预测结果----MaxPoolingMIL
    也可对同一slide所有patch预测结果计算注意力做加权平均作为slide_level预测结果----AttentionMIL
    """
    prob_cols = [f'prob_{i}' for i in range(num_classes)]
    df_slide = df.groupby('slide')[prob_cols].mean()  # MeanPoolingMIL
    df_slide['label_pred'] = df_slide[prob_cols].idxmax(axis=1).str.split('_').str[-1].astype(int)

    # merge true label
    true_df = df[['slide', 'label']].drop_duplicates('slide').set_index('slide')
    df_slide = df_slide.join(true_df, how='left').reset_index()

    label_true = df_slide['label'].astype(int)
    label_pred = df_slide['label_pred'].astype(int)

    # metrics
    acc = accuracy_score(label_true, label_pred)
    f1 = f1_score(label_true, label_pred, average='macro')
    precision = precision_score(label_true, label_pred, average='macro')
    recall = recall_score(label_true, label_pred, average='macro')
    # 混淆矩阵中的数量和为val中所有slides和
    cm = confusion_matrix(label_true, label_pred, labels=np.arange(num_classes))

    # ROC/AUC
    fpr, tpr, roc_auc = {}, {}, {}
    for i in range(num_classes):
        bin_true = (label_true == i).astype(int)

        # fpr 与 tpr 分别表示假正例率和真正例率
        fpr[i], tpr[i], _ = roc_curve(bin_true, df_slide[f'prob_{i}'])
        roc_auc[i] = auc(fpr[i], tpr[i])

    ext = 'pdf' if save_as_pdf else 'png'
    # 在roc_path路径基础上增加epoch信息，防止png图片被覆盖错失最佳model状态曲线
    current_epoch = epoch + 1
    fpr_tpr_auc_path = os.path.join(save_path, f'fpr_tpr_auc_{current_epoch}_{iter_idx}.json')
    save_fpr_tpr_auc(fpr, tpr, roc_auc, fpr_tpr_auc_path)

    roc_path = os.path.join(save_path, f'slide_{current_epoch}_{iter_idx}.{ext}')
    slide_auc = plot_roc_curve(
        fpr, tpr, roc_auc,
        num_classes, class_names,
        title='Slide-level ROC',
        save_path=roc_path
    )

    print(f'Validation time: {time.time() - tic:.2f}s, Loss={total_loss:.4f}')
    return {
        'slide_AUC': slide_auc,
        'slide_ACC': acc,
        'slide_F1': f1,
        'slide_Precision': precision,
        'slide_Recall': recall,
        'confusion_matrix': cm
    }


def val_Ablation_MSfusion(model, loader, criterion, class_names, save_path, epoch, iter_idx, save_as_pdf=False):
    print(f'Validating: iter {iter_idx} ...')
    tic = time.time()
    num_classes = len(class_names)
    device = next(model.parameters()).device

    slide_list = []
    label_list = []
    prob_list = []

    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for batch in loader:
            slide, tile, patch_num, label, feature_list, cooadj_list = batch

            tile = tile.to(device)
            patch_num = patch_num.to(device)
            label = label.to(device)
            feature_list = [fea.to(device) for fea in feature_list]
            cooadj_list = [adj.to(device) for adj in cooadj_list]

            output = model(tile, patch_num, feature_list, cooadj_list)  # [B, C]

            probs = torch.softmax(output, dim=1)

            loss = criterion(output, label)
            total_loss += loss.item()

            slide_list.extend(slide)
            label_list.append(label.cpu().numpy())
            prob_list.append(probs.cpu().numpy())

    # flatten
    label_all = np.concatenate(label_list, axis=0)  # (N,)
    prob_all = np.concatenate(prob_list, axis=0)  # (N, C)
    N = label_all.shape[0]
    assert len(slide_list) == N

    # DataFrame
    df = pd.DataFrame({
        'slide': slide_list,
        'label': label_all
    })
    for i in range(num_classes):
        df[f'prob_{i}'] = prob_all[:, i]

    # slide-level aggregation(get mean preds to aggregate)
    """
    此处对同一slide所有patch预测取均值作为slide_level预测结果----MeanPoolingMIL
    也可对同一slide所有patch预测取最大值作为slide_level预测结果----MaxPoolingMIL
    也可对同一slide所有patch预测结果计算注意力做加权平均作为slide_level预测结果----AttentionMIL
    """
    prob_cols = [f'prob_{i}' for i in range(num_classes)]
    df_slide = df.groupby('slide')[prob_cols].mean()  # MeanPoolingMIL
    df_slide['label_pred'] = df_slide[prob_cols].idxmax(axis=1).str.split('_').str[-1].astype(int)

    # merge true label
    true_df = df[['slide', 'label']].drop_duplicates('slide').set_index('slide')
    df_slide = df_slide.join(true_df, how='left').reset_index()

    label_true = df_slide['label'].astype(int)
    label_pred = df_slide['label_pred'].astype(int)

    # metrics
    acc = accuracy_score(label_true, label_pred)
    f1 = f1_score(label_true, label_pred, average='macro')
    precision = precision_score(label_true, label_pred, average='macro')
    recall = recall_score(label_true, label_pred, average='macro')
    # 混淆矩阵中的数量和为val中所有slides和
    cm = confusion_matrix(label_true, label_pred, labels=np.arange(num_classes))

    # ROC/AUC
    fpr, tpr, roc_auc = {}, {}, {}
    for i in range(num_classes):
        bin_true = (label_true == i).astype(int)

        # fpr 与 tpr 分别表示假正例率和真正例率
        fpr[i], tpr[i], _ = roc_curve(bin_true, df_slide[f'prob_{i}'])
        roc_auc[i] = auc(fpr[i], tpr[i])

    ext = 'pdf' if save_as_pdf else 'png'
    # 在roc_path路径基础上增加epoch信息，防止png图片被覆盖错失最佳model状态曲线
    current_epoch = epoch + 1
    fpr_tpr_auc_path = os.path.join(save_path, f'fpr_tpr_auc_{current_epoch}_{iter_idx}.json')
    save_fpr_tpr_auc(fpr, tpr, roc_auc, fpr_tpr_auc_path)

    roc_path = os.path.join(save_path, f'slide_{current_epoch}_{iter_idx}.{ext}')
    slide_auc = plot_roc_curve(
        fpr, tpr, roc_auc,
        num_classes, class_names,
        title='Slide-level ROC',
        save_path=roc_path
    )

    print(f'Validation time: {time.time() - tic:.2f}s, Loss={total_loss:.4f}')
    return {
        'slide_AUC': slide_auc,
        'slide_ACC': acc,
        'slide_F1': f1,
        'slide_Precision': precision,
        'slide_Recall': recall,
        'confusion_matrix': cm
    }
