import os
import cv2
import numpy as np
from glob import glob


def prepare_dataset(source_dir, target_dir_A, target_dir_B,
                    target_size=(256, 256)):
    """
    处理BUSI全部三类数据：
      benign    → 乳腺(绿) + 结节(红)
      malignant → 乳腺(绿) + 肿瘤(红，不规则)
      normal    → 乳腺(绿)，无结节
    """
    os.makedirs(target_dir_A, exist_ok=True)
    os.makedirs(target_dir_B, exist_ok=True)

    # =========================================================
    # ✅ 三个类别全部加入
    # =========================================================
    categories = ['benign', 'normal']

    file_count = 0
    category_stats = {}  # 记录每类数量

    print("开始处理全部数据...\n")

    for category in categories:
        cat_dir = os.path.join(source_dir, category)
        if not os.path.exists(cat_dir):
            print(f"⚠️  找不到文件夹: {cat_dir}，跳过")
            continue

        # 找到该类别下所有原图（排除mask文件）
        all_files = glob(os.path.join(cat_dir, '*.png'))
        real_images = [f for f in all_files if 'mask' not in f.lower()]

        cat_count = 0
        print(f"📁 处理 [{category}]：找到 {len(real_images)} 张原图")

        for real_path in real_images:
            base_name = os.path.basename(real_path).replace('.png', '')
            mask_paths = glob(os.path.join(cat_dir, f"{base_name}_mask*.png"))

            # ----------------------------------------------------------
            # 1. 读取真实超声原图 → trainB
            # ----------------------------------------------------------
            real_img = cv2.imread(real_path, cv2.IMREAD_GRAYSCALE)
            if real_img is None:
                continue
            real_resized = cv2.resize(
                real_img, target_size, interpolation=cv2.INTER_AREA
            )

            # ----------------------------------------------------------
            # 2. 构建彩色Mask → trainA
            # ----------------------------------------------------------
            synthesized_mask = np.zeros(
                (target_size[1], target_size[0], 3), dtype=np.uint8
            )

            # ── 提取乳腺区域（绿色）──────────────────────────────────
            _, tissue_area = cv2.threshold(
                real_resized, 15, 255, cv2.THRESH_BINARY
            )
            kernel = np.ones((5, 5), np.uint8)
            tissue_area = cv2.morphologyEx(
                tissue_area, cv2.MORPH_CLOSE, kernel, iterations=3
            )
            synthesized_mask[tissue_area == 255] = [0, 255, 0]  # 绿色

            # ── 提取结节/肿瘤区域（红色）────────────────────────────
            # normal类没有mask文件，跳过结节绘制
            if category != 'normal' and mask_paths:
                combined_mask = np.zeros(target_size, dtype=np.uint8)

                for m_path in mask_paths:
                    m_img = cv2.imread(m_path, cv2.IMREAD_GRAYSCALE)
                    if m_img is not None:
                        m_resized = cv2.resize(
                            m_img, target_size,
                            interpolation=cv2.INTER_NEAREST
                        )
                        combined_mask[m_resized > 127] = 255

                # 红色覆盖在绿色上
                synthesized_mask[combined_mask == 255] = [0, 0, 255]  # 红色(BGR)

            # ----------------------------------------------------------
            # 3. 保存
            # ----------------------------------------------------------
            save_name = f"{file_count:05d}.png"
            cv2.imwrite(os.path.join(target_dir_A, save_name), synthesized_mask)
            cv2.imwrite(os.path.join(target_dir_B, save_name), real_resized)

            file_count += 1
            cat_count += 1

        category_stats[category] = cat_count
        print(f"  ✓ [{category}] 处理完成：{cat_count} 对\n")

    # 打印统计
    print("=" * 50)
    print(f"🎉 数据处理完成！")
    print(f"{'类别':<12} {'数量':>6}")
    print("-" * 20)
    for cat, count in category_stats.items():
        print(f"  {cat:<10} {count:>6} 张")
    print("-" * 20)
    print(f"  {'合计':<10} {file_count:>6} 张")
    print("=" * 50)
    print(f"\n输出路径:")
    print(f"  Mask  → {os.path.abspath(target_dir_A)}")
    print(f"  超声图 → {os.path.abspath(target_dir_B)}")


if __name__ == '__main__':
    # ⚠️ 修改为你的实际路径
    source_directory = r"C:\Users\86150\Desktop\SOFA_Pro\data\Dataset_BUSI_with_GT"

    project_root = os.path.dirname(os.path.abspath(__file__))
    target_A = os.path.join(project_root, "datasets", "breast_ultrasound", "trainA")
    target_B = os.path.join(project_root, "datasets", "breast_ultrasound", "trainB")

    prepare_dataset(source_directory, target_A, target_B)