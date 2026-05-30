import os

import numpy as np
from matplotlib import pyplot as plt

# 设置保存路径
target_folder = '<YOUR_OUTPUT_PATH>'
save_path = os.path.join(target_folder, "Auc_boxplot.png")

auc_data = {
    "Ours": [0.894,  0.894,  0.869,  0.866,  0.893],
    "TransMIL": [0.852, 0.858, 0.825, 0.845, 0.790],
    "CLAM-SB": [0.871, 0.866, 0.858, 0.868, 0.863],
    "CLAM-MB": [0.875, 0.863, 0.852, 0.869, 0.872],
    "DTFD-MIL": [0.875, 0.859, 0.852, 0.871, 0.868],
    "Patch-GCN": [0.733, 0.744, 0.768, 0.761, 0.729],
    "DS-MIL": [0.857, 0.855, 0.852, 0.843, 0.839],
    "ABMIL": [0.860, 0.854, 0.859, 0.840, 0.846]
}

all_values = [value for sublist in auc_data.values() for value in sublist]
y_min = np.floor(min(all_values) * 100) / 100
y_max = np.ceil(max(all_values) * 100) / 100
y_ticks = np.arange(y_min, y_max + 0.02, 0.02)

# 绘制并保存图像
plt.figure(figsize=(12, 6))
plt.boxplot(auc_data.values(), labels=auc_data.keys(), patch_artist=True)
plt.ylabel('AUC', fontsize=12)
plt.xlabel('Method', fontsize=12)
plt.title('Comparison of AUC Across Methods (5-Fold CV)', fontsize=14)
plt.xticks(rotation=30)
plt.yticks(y_ticks)
plt.ylim(y_min, y_max)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig(save_path, dpi=300)
