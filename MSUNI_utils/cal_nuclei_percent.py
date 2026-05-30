import os
import cv2
import numpy as np
from tqdm import tqdm


def parse_patch_filename(patch_filename):
    """
    从文件名中提取 slide_name, label, x, y, sp
    """
    name = os.path.splitext(patch_filename)[0]
    parts = dict(kv.split("=") for kv in name.split(","))
    return parts["slide"], parts["label"], parts["x"], parts["y"], parts["sp"]


def calculate_avg_nucleus_distribution_by_count(image_folder, nucleus_folder, nucleus_labels=(3, 4, 5, 6, 7)):
    """
    对一个 attention 图像目录下的所有 patch 计算细胞核类型的平均“数量占比”，
    即每种类型细胞核的个数占所有目标细胞核总数的比例。
    """
    label_instance_total = {label: 0 for label in nucleus_labels}
    total_instance_count = 0
    image_files = [f for f in os.listdir(image_folder) if f.endswith(('.jpg', '.png'))]

    for filename in tqdm(image_files, desc=f"Processing {os.path.basename(image_folder)}"):
        try:
            slide, label, x, y, sp = parse_patch_filename(filename)
            nucleus_filename = f"nucleus_slide={slide},label={label},x={x},y={y},sp={sp}.png"
            nucleus_path = os.path.join(nucleus_folder, nucleus_filename)
            if not os.path.exists(nucleus_path):
                print(f"❌ 找不到 nucleus patch: {nucleus_path}")
                continue

            nucleus_mask = cv2.imread(nucleus_path, cv2.IMREAD_UNCHANGED)
            if nucleus_mask is None:
                print(f"❌ 无法读取图像: {nucleus_path}")
                continue
            if len(nucleus_mask.shape) == 3:
                nucleus_mask = nucleus_mask[:, :, 0]

            # 对每个 label，统计细胞核数量（连通区域）
            for label in nucleus_labels:
                mask = (nucleus_mask == label).astype(np.uint8)
                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                count = len(contours)
                label_instance_total[label] += count
                total_instance_count += count

        except Exception as e:
            print(f"⚠️ 跳过文件 {filename}, 错误: {e}")
            continue

    # 计算平均数量占比
    avg_distribution = {
        label: (label_instance_total[label] / total_instance_count) if total_instance_count > 0 else 0.0
        for label in nucleus_labels
    }

    return avg_distribution


def analyze_nucleus_distribution_by_attention_group(small_image_folder, nucleus_folder):
    nucleus_labels = (3, 4, 5, 6, 7)

    print("\n📊 正在分析图像块 ...")
    high_dist = calculate_avg_nucleus_distribution_by_count(small_image_folder, nucleus_folder, nucleus_labels)

    print("\n🔍 结果对比：每种细胞核类型在所有目标细胞核中的平均占比")
    print("类别\t注意力占比")
    for label in nucleus_labels:
        print(f"{label} \t{high_dist[label]:.2%}")


if __name__ == "__main__":
    small_image_folder = '<YOUR_SMALL_IMAGE_PATH>'
    small_nucleus_folder = '<YOUR_SMALL_NUCLEUS_PATH>'
    """
    类别	注意力占比
    3 	53.43%
    4 	11.88%
    5 	7.21%
    6 	1.19%
    7 	26.28%
    """
    fine_image_folder = '<YOUR_FINE_IMAGE_PATH>'
    fine_nucleus_folder = '<YOUR_FINE_NUCLEUS_PATH>'
    """
    类别	注意力占比
    3 	84.38%
    4 	2.55%
    5 	1.73%
    6 	0.06%
    7 	11.28%
    """
    large_image_folder = '<YOUR_LARGE_IMAGE_PATH>'
    large_nucleus_folder = '<YOUR_LARGE_NUCLEUS_PATH>'
    """
    类别	注意力占比
    3 	37.00%
    4 	10.92%
    5 	3.38%
    6 	0.09%
    7 	48.60%
    """

    analyze_nucleus_distribution_by_attention_group(large_image_folder, large_nucleus_folder)
