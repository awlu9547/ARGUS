# import matplotlib.pyplot as plt
# import seaborn as sns
# import pandas as pd
#
# # 设置图像风格
# sns.set(style="whitegrid")
#
# target_path = r'G:\Projects\MSUNI_Work_789\Box_AUC\nuclei_percent_box.png'
#
# # 数据：按列为各类细胞百分比，每列5个样本
# cell_types = ["Tumor", "Inflammatory", "Stroma", "Necrosis", "Epithelial"]
#
# high_attn = [
#     [91.97, 78.96, 77.31, 88.77, 76.62],
#     [1.60, 4.35, 8.29, 1.31, 5.63],
#     [0.70, 7.48, 4.35, 1.41, 5.43],
#     [0.13, 0.97, 0.64, 0.74, 1.28],
#     [5.60, 8.23, 9.42, 7.77, 11.05]
# ]
#
# low_attn = [
#     [50.73, 38.72, 57.88, 62.95, 40.55],
#     [4.63, 7.04, 9.20, 6.63, 9.89],
#     [2.40, 2.92, 17.30, 3.21, 20.08],
#     [0.12, 0.24, 2.81, 0.40, 10.31],
#     [42.13, 51.08, 12.81, 26.80, 19.17]
# ]
#
# # 构造 DataFrame
# data = []
# for i, cell in enumerate(cell_types):
#     for val in high_attn[i]:
#         data.append({"Cell Type": cell, "Attention": "High Attention", "Percentage": val})
#     for val in low_attn[i]:
#         data.append({"Cell Type": cell, "Attention": "Low Attention", "Percentage": val})
#
# df = pd.DataFrame(data)
#
# # 绘图
# plt.figure(figsize=(10, 6))
# ax = sns.boxplot(x="Cell Type", y="Percentage", hue="Attention", data=df, palette=["#F8766D", "#00BFC4"])
# ax.set_title("HoverNet cell classification (Drum Tower Dataset)", fontsize=14, fontweight='bold')
# ax.set_ylabel("Cell Percentage of Total")
# ax.set_xlabel("")
#
# # 设置图例样式
# plt.legend(title="", loc="upper right", frameon=False)
#
# # 控制轴范围（可选）
# plt.ylim(0, 100)
#
# plt.tight_layout()
# plt.savefig(target_path, dpi=300)

"""--------------------------------------------------------------------------"""
import os
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


def parse_patch_filename(patch_filename):
    """
    从文件名中提取 slide_name, label, x, y, sp
    """
    name = os.path.splitext(patch_filename)[0]
    parts = dict(kv.split("=") for kv in name.split(","))
    return parts["slide"], parts["label"], parts["x"], parts["y"], parts["sp"]


def collect_nucleus_distribution_per_patch(image_folder, nucleus_folder, nucleus_labels=(3, 4, 5, 6, 7), label_name=None):
    """
    收集每个 patch 的细胞核类型像素占比，用于箱线图。
    返回 DataFrame，包括列：Cell Type, Percentage, Attention (High/Low)
    """
    label_names = {
        3: "Tumor",
        4: "Inflammatory",
        5: "Stroma",
        6: "Necrosis",
        7: "Epithelial"
    }

    rows = []
    image_files = [f for f in os.listdir(image_folder) if f.endswith(('.jpg', '.png'))]

    for filename in tqdm(image_files, desc=f"Processing {os.path.basename(image_folder)}"):
        try:
            slide, label, x, y, sp = parse_patch_filename(filename)
            nucleus_filename = f"nucleus_slide={slide},label={label},x={x},y={y},sp={sp}.png"
            nucleus_path = os.path.join(nucleus_folder, nucleus_filename)
            if not os.path.exists(nucleus_path):
                continue

            nucleus_mask = cv2.imread(nucleus_path, cv2.IMREAD_UNCHANGED)
            if nucleus_mask is None:
                continue
            if len(nucleus_mask.shape) == 3:
                nucleus_mask = nucleus_mask[:, :, 0]

            total_instance = 0
            label_instance_counts = {}

            for label_id in nucleus_labels:
                mask = (nucleus_mask == label_id).astype(np.uint8)
                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                count = len(contours)
                label_instance_counts[label_id] = count
                total_instance += count

            if total_instance == 0:
                continue

            for label_id in nucleus_labels:
                percent = label_instance_counts[label_id] / total_instance
                rows.append({
                    "Cell Type": label_names[label_id],
                    "Percentage": percent * 100,
                    "Label": label_name
                })

        except Exception:
            continue

    return pd.DataFrame(rows)


if __name__ == "__main__":
    fine_image_folder = '<YOUR_FINE_IMAGE_PATH>'
    small_image_folder = '<YOUR_SMALL_IMAGE_PATH>'
    large_image_folder = '<YOUR_LARGE_IMAGE_PATH>'

    fine_nucleus_folder = '<YOUR_FINE_NUCLEUS_PATH>'
    small_nucleus_folder = '<YOUR_SMALL_NUCLEUS_PATH>'
    large_nucleus_folder = '<YOUR_LARGE_NUCLEUS_PATH>'

    target_path = '<YOUR_TARGET_PATH>'

    # 分别收集高低注意力patch的细胞核分布
    df_fine = collect_nucleus_distribution_per_patch(fine_image_folder, fine_nucleus_folder, label_name='Fine duct')
    df_small = collect_nucleus_distribution_per_patch(small_image_folder, small_nucleus_folder, label_name='Small duct')
    df_large = collect_nucleus_distribution_per_patch(large_image_folder, large_nucleus_folder, label_name='Large duct')

    # 合并为一个DataFrame用于绘图
    df_all = pd.concat([df_fine, df_small, df_large], ignore_index=True)

    # plt.figure(figsize=(10, 6))
    #
    # label_bag = ['Fine duct', 'Small duct', 'Large duct']
    # palette = sns.color_palette(["#F8766D", "#00BFC4", "#A3A500"])[:len(label_bag)]
    #
    # sns.boxplot(
    #     data=df_all,
    #     x="Cell Type",
    #     y="Percentage",
    #     hue="Label",
    #     hue_order=label_bag,
    #     palette=palette,
    #     dodge=True
    # )
    #
    # plt.title("HoverNet Cell Classification (Drum Tower Dataset)", fontsize=12, fontweight='bold')
    # plt.ylabel("Cell Percentage of Total")
    # plt.xlabel("")
    # plt.ylim(0, 100)
    # plt.legend(
    #     title="",
    #     loc="center left",
    #     bbox_to_anchor=(1.02, 0.5),
    #     frameon=False
    # )
    # plt.tight_layout(rect=[0, 0, 0.9, 1])
    # plt.savefig(target_path, dpi=300, bbox_inches='tight')

    plt.figure(figsize=(10, 6))

    label_bag = ['Fine duct', 'Small duct', 'Large duct']
    palette = sns.color_palette(["#F8766D", "#00BFC4", "#A3A500"])[:len(label_bag)]

    sns.boxplot(
        data=df_all,
        x="Cell Type",
        y="Percentage",
        hue="Label",
        hue_order=label_bag,
        palette=palette,
        dodge=True
    )

    # 设置标题、坐标轴等
    plt.title("HoverNet Cell Classification (Drum Tower Dataset)", fontsize=12, fontweight='bold')
    plt.ylabel("Cell Percentage of Total")
    plt.xlabel("")
    plt.ylim(0, 100)

    # 将图例移动到图内部上方中间位置
    plt.legend(
        title="",
        loc="upper center",
        bbox_to_anchor=(0.5, 1.0),  # 控制位置：横向居中，略高于图像
        ncol=len(label_bag),  # 水平排列
        frameon=False
    )

    # 调整布局：上方留出空间放图例
    plt.tight_layout(rect=[0, 0, 1, 0.88])

    # 保存图片
    plt.savefig(target_path, dpi=300, bbox_inches='tight')




