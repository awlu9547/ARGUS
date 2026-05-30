import os
import json
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from sklearn.metrics import auc


def load_fpr_tpr_auc(file_path):
    """
    加载 fpr, tpr, auc 数据
    参数:
        file_path: str, JSON 文件路径
    返回:
        fpr_dict: dict {class_id: list of fpr}
        tpr_dict: dict {class_id: list of tpr}
        auc_dict: dict {class_id: float or list of float}
    """
    with open(file_path, 'r') as f:
        data = json.load(f)

    return {
        "fpr": data["fpr"],
        "tpr": data["tpr"],
        "auc": data["auc"]
    }


def plot_roc_with_ci(fpr_list, tpr_list, auc_list, num_classes, class_names, title, save_path):
    """
    绘制多类别 ROC 曲线，并添加 macro AUC 的标准差阴影区域。
    参数:
        fpr_list: list of dicts, 每个 dict 包含每个类别的 fpr 列表
        tpr_list: list of dicts, 每个 dict 包含每个类别的 tpr 列表
        auc_list: list of dicts, 每个 dict 包含每个类别的 auc 值
        num_classes: int, 类别数量
        class_names: list of str, 类别名称
        title: str, 图表标题
        save_path: str, 图像保存路径
    """
    plt.figure(figsize=(8, 6))
    colors = ['darkorange', 'cornflowerblue', 'green']
    mean_fpr = np.linspace(0, 1, 100)

    # 存储每个类别的 tpr 插值结果
    tprs = [[] for _ in range(num_classes)]
    aucs = [[] for _ in range(num_classes)]

    for i in range(len(fpr_list)):
        for cls in range(num_classes):
            fpr = np.array(fpr_list[i][str(cls)])
            tpr = np.array(tpr_list[i][str(cls)])
            interp_tpr = np.interp(mean_fpr, fpr, tpr)
            interp_tpr[0] = 0.0
            tprs[cls].append(interp_tpr)

            auc_val = auc_list[i][str(cls)]  # 👈 来自已加载的 auc_list
            aucs[cls].append(auc_val)

    # 绘制每个类别的平均 ROC 曲线
    for cls in range(num_classes):
        mean_tpr = np.mean(tprs[cls], axis=0)
        mean_tpr = np.array(mean_tpr)
        mean_tpr[-1] = 1.0
        mean_auc = np.mean(aucs[cls])
        std_auc = np.std(aucs[cls])
        plt.plot(mean_fpr, mean_tpr, color=colors[cls],
                 label=f'{class_names[cls]} (AUC = {mean_auc:.3f} ± {std_auc:.3f})',
                 lw=2, alpha=0.8)

    # 计算 macro 平均
    macro_aucs = []
    all_interp_tprs = []

    for i in range(len(fpr_list)):
        # 计算每个折的 macro tpr
        fold_tprs = []
        fold_auc_vals = []

        for cls in range(num_classes):
            fpr = np.array(fpr_list[i][str(cls)])
            tpr = np.array(tpr_list[i][str(cls)])
            interp_tpr = np.interp(mean_fpr, fpr, tpr)
            interp_tpr[0] = 0.0
            fold_tprs.append(interp_tpr)
            fold_auc_vals.append(auc_list[i][str(cls)])

        mean_fold_tpr = np.mean(fold_tprs, axis=0)
        all_interp_tprs.append(mean_fold_tpr)
        macro_aucs.append(np.mean(fold_auc_vals))

    mean_macro_tpr = np.mean(all_interp_tprs, axis=0)
    std_macro_tpr = np.std(all_interp_tprs, axis=0)
    mean_macro_auc = np.mean(macro_aucs)
    std_macro_auc = np.std([auc(mean_fpr, tpr) for tpr in all_interp_tprs])

    # 绘制 macro 平均 ROC 曲线
    plt.plot(mean_fpr, mean_macro_tpr,
             label=f'Macro-average (AUC = {mean_macro_auc:.3f} ± {std_macro_auc:.3f})',
             color='navy', linestyle='--', linewidth=2)

    # 添加标准差阴影区域
    tpr_upper = np.minimum(mean_macro_tpr + std_macro_tpr, 1)
    tpr_lower = np.maximum(mean_macro_tpr - std_macro_tpr, 0)
    plt.fill_between(mean_fpr, tpr_lower, tpr_upper, color='grey', alpha=0.2)

    plt.plot([0, 1], [0, 1], 'k--', lw=1)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title(title, fontsize=14)
    plt.legend(loc="lower right", fontsize=10)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


if __name__ == '__main__':
    # 设置 fold 数量和路径列表
    num_folds = 5
    ml_paths = [
        '<YOUR_FOLD_1_JSON_PATH>',
        '<YOUR_FOLD_2_JSON_PATH>',
        '<YOUR_FOLD_3_JSON_PATH>',
        '<YOUR_FOLD_4_JSON_PATH>',
        '<YOUR_FOLD_5_JSON_PATH>'
    ]

    num_classes = 3
    class_names = ['Fine', 'Small', 'Large']
    save_path = '<YOUR_SAVE_PATH>'

    # 加载所有 fold 的数据
    fpr_list = []
    tpr_list = []
    auc_list = []

    for path in ml_paths:
        data = load_fpr_tpr_auc(path)
        fpr_list.append(data['fpr'])
        tpr_list.append(data['tpr'])
        auc_list.append(data['auc'])

    # 绘制 ROC 曲线
    plot_roc_with_ci(
        fpr_list=fpr_list,
        tpr_list=tpr_list,
        auc_list=auc_list,
        num_classes=num_classes,
        class_names=class_names,
        title='ROC (5-fold cross-validation)',
        save_path=save_path
    )