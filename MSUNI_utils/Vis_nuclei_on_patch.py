import cv2
import numpy as np
import os


def visualize_nuclei_on_patch(img_patch_path, nucleus_patch_path, output_path,
                              nucleus_labels=(2, 3, 4, 5, 6, 7),
                              colors=((0, 0, 0), (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255)), alpha=1.0):
    """
    将不同类型的细胞核在原始病理图上以不同颜色叠加并保存。

    :param img_patch_path: str, RGB 病理图像 patch 路径
    :param nucleus_patch_path: str, 单通道 nucleus patch 路径（像素值代表细胞核类型）
    :param output_path: str, 可视化结果保存路径
    :param nucleus_labels: tuple, 要保留的细胞核类型值
    :param colors: tuple, 每类细胞核对应的 BGR 颜色
    :param alpha: float, 叠加透明度
    """
    # 读取图像
    rgb_img = cv2.imread(img_patch_path)  # RGB patch
    nucleus_mask = cv2.imread(nucleus_patch_path, cv2.IMREAD_UNCHANGED)  # 单通道标签图

    if rgb_img is None or nucleus_mask is None:
        raise FileNotFoundError("无法读取输入图像，请检查路径")

    # 如果是三通道 nucleus patch，只保留第一个通道
    if len(nucleus_mask.shape) == 3:
        nucleus_mask = nucleus_mask[:, :, 0]

    # 创建 overlay 层
    overlay = rgb_img.copy()

    # 细胞核数量统计
    label_instance_counts = {}
    total_instances = 0

    # 对每个细胞核类别叠加颜色
    for label, color in zip(nucleus_labels, colors):
        mask = (nucleus_mask == label).astype(np.uint8)

        # 找到轮廓，每个连通区域视为一个细胞
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        instance_count = len(contours)
        label_instance_counts[label] = instance_count
        total_instances += instance_count

        # 可视化填色
        cv2.drawContours(overlay, contours, -1, color, thickness=-1)

    # 将 overlay 与原图混合
    blended = cv2.addWeighted(overlay, alpha, rgb_img, 1 - alpha, 0)

    # 保存结果
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cv2.imwrite(output_path, blended)
    print(f"已保存叠加图到: {output_path}")

    # 打印数量占比
    print("细胞核类型 **数量** 占比统计：")
    for label in nucleus_labels:
        count = label_instance_counts.get(label, 0)
        ratio = count / total_instances if total_instances > 0 else 0
        print(f"  - 类型 {label}: {count} 个细胞, 占比 {ratio:.2%}")


if __name__ == "__main__":
    low_image_path = '<YOUR_LOW_IMAGE_PATH>'
    low_nucleus_path = '<YOUR_LOW_NUCLEUS_PATH>'
    low_target_path = '<YOUR_LOW_TARGET_PATH>'

    fine_image_path = '<YOUR_FINE_IMAGE_PATH>'
    fine_nucleus_path = '<YOUR_FINE_NUCLEUS_PATH>'
    fine_target_path = '<YOUR_FINE_TARGET_PATH>'

    small_image_path = '<YOUR_SMALL_IMAGE_PATH>'
    small_nucleus_path = '<YOUR_SMALL_NUCLEUS_PATH>'
    small_target_path = '<YOUR_SMALL_TARGET_PATH>'

    large_image_path = '<YOUR_LARGE_IMAGE_PATH>'
    large_nucleus_path = '<YOUR_LARGE_NUCLEUS_PATH>'
    large_target_path = '<YOUR_LARGE_TARGET_PATH>'

    # - 2："neopla"，肿瘤上皮细胞核       -------red
    # - 3："inflam"，炎性细胞核          -------green
    # - 4："stromal"，结缔组织细胞核      -------blue
    # - 5："necrosis"，坏死细胞核        -------yellow
    # - 6："Epithelial"，结缔组织细胞核   -------purple
    # visualize_nuclei_on_patch(low_image_path, low_nucleus_path, low_target_path)

    visualize_nuclei_on_patch(small_image_path, small_nucleus_path, small_target_path)


