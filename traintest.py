# =============================================================================
# quick_test.py  ── 完全模拟训练集Mask格式的验证脚本
# =============================================================================

import os
import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib

matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False

from GAN_train import GeneratorUNet

# ─────────────────────────────────────────────────────────────────────────────
# 配置
# ─────────────────────────────────────────────────────────────────────────────
MODEL_PATH = "checkpoints/generator_best.pth"
IMAGE_SIZE = 256
OUTPUT_DIR = "test_results"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# 模型加载
# ─────────────────────────────────────────────────────────────────────────────
def load_generator(model_path):
    print(f"📂 加载模型: {model_path}")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"找不到: {model_path}")

    generator = GeneratorUNet(in_channels=3, out_channels=1).to(DEVICE)
    ckpt = torch.load(model_path, map_location=DEVICE)

    if isinstance(ckpt, dict) and 'generator' in ckpt:
        generator.load_state_dict(ckpt['generator'])
        print(f"  ✓ 完整检查点，Epoch {ckpt.get('epoch', '?')}")
    else:
        generator.load_state_dict(ckpt)
        print(f"  ✓ 纯权重文件")

    generator.eval()
    print(f"  ✓ 设备: {DEVICE}\n")
    return generator


# ─────────────────────────────────────────────────────────────────────────────
# 核心：模拟真实超声图的亮度分布，然后走和训练集完全一样的处理流程
# ─────────────────────────────────────────────────────────────────────────────
def simulate_real_mask(lesion_center=None,
                       lesion_radius=None,
                       lesion_shape='round',
                       scan_type='standard',
                       size=256,
                       threshold=15):
    """
    完全模拟 prepare_dataset.py 的Mask生成流程：

    Step1: 生成模拟超声图（只有亮度分布，无需真实纹理）
           - 超声扫描区域：亮度 >> 15
           - 图像边角（探头外）：亮度 = 0
    Step2: cv2.threshold(img, 15, 255, THRESH_BINARY) → tissue_area
    Step3: morphologyEx CLOSE → 填补空洞
    Step4: 绿色涂乳腺，红色涂结节

    这和 prepare_dataset.py 的处理完全一致。

    Args:
        lesion_center : (cx, cy) 结节中心像素坐标，None=无结节
        lesion_radius : (rx, ry) 结节半径像素，None=无结节
        lesion_shape  : 'round' / 'irregular'
        scan_type     : 'standard'  标准扫查（扇形超声区域）
                        'full'      区域铺满（探头正压）
                        'offset'    偏向一侧
        size          : 图像尺寸
        threshold     : 与训练集一致，默认15

    Returns:
        mask_bgr : (H,W,3) uint8  BGR格式（与cv2一致）
        mask_rgb : PIL.Image       RGB格式（用于GAN输入）
    """
    S = size

    # ── Step1: 生成模拟超声亮度图 ────────────────────────────────────────────
    # 思路：超声图的有效区域亮度均匀偏高（~128），边角为0
    # 我们直接画一个高亮区域，模拟探头接触面的形状

    sim_img = np.zeros((S, S), dtype=np.uint8)

    if scan_type == 'standard':
        # 黑边减少版：有效区域扩大到98%
        pts = np.array([
            [int(S * 0.01), 0],
            [int(S * 0.99), 0],
            [int(S * 0.99), S - 1],
            [int(S * 0.01), S - 1],
        ], dtype=np.int32)
        cv2.fillPoly(sim_img, [pts], color=200)

    elif scan_type == 'full':
        # 完全铺满：几乎无黑边
        sim_img[:, :] = 200  # 整张图都是有效区域

    elif scan_type == 'offset':
        # 偏左扫查：左边无黑边，右边少量黑边
        pts = np.array([
            [0, 0],
            [int(S * 0.92), 0],
            [int(S * 0.92), S - 1],
            [0, S - 1],
        ], dtype=np.int32)
        cv2.fillPoly(sim_img, [pts], color=200)

    elif scan_type == 'wide':
        # 宽景扫查：整图铺满
        sim_img[:, :] = 200

    # ── Step2: 走和训练集完全相同的处理流程 ──────────────────────────────────
    # 完全复制 prepare_dataset.py 的代码

    # 阈值（与训练集一致：threshold=15）
    _, tissue_area = cv2.threshold(sim_img, threshold, 255, cv2.THRESH_BINARY)

    # 形态学闭运算（与训练集一致：kernel=5×5, iterations=3）
    kernel = np.ones((5, 5), np.uint8)
    tissue_area = cv2.morphologyEx(
        tissue_area, cv2.MORPH_CLOSE, kernel, iterations=3
    )

    # 初始化黑色背景
    mask_bgr = np.zeros((S, S, 3), dtype=np.uint8)

    # 乳腺区域 → 绿色（BGR: [0,255,0]）
    mask_bgr[tissue_area == 255] = [0, 255, 0]

    # ── Step3: 结节区域 → 红色 ───────────────────────────────────────────────
    if lesion_center is not None and lesion_radius is not None:
        cx, cy = lesion_center
        rx, ry = lesion_radius

        # 确保结节在有效扫描区域内
        cx = int(np.clip(cx, rx + 5, S - rx - 5))
        cy = int(np.clip(cy, ry + 5, S - ry - 5))

        lesion_mask = np.zeros((S, S), dtype=np.uint8)

        if lesion_shape == 'round':
            cv2.ellipse(lesion_mask,
                        center=(cx, cy),
                        axes=(rx, ry),
                        angle=0,
                        startAngle=0, endAngle=360,
                        color=255, thickness=-1)

        elif lesion_shape == 'irregular':
            # 不规则结节：用随机多边形模拟
            np.random.seed(42)
            n_pts = 16
            angles = np.linspace(0, 2 * np.pi, n_pts, endpoint=False)
            radii = rx * (0.7 + 0.5 * np.random.rand(n_pts))
            pts_x = (cx + radii * np.cos(angles)).astype(int)
            pts_y = (cy + radii * np.sin(angles)).astype(int)
            polygon = np.column_stack([pts_x, pts_y]).reshape(-1, 1, 2)
            cv2.fillPoly(lesion_mask, [polygon], color=255)

        # 只在乳腺区域内的结节才涂红色（与训练集逻辑一致）
        valid_lesion = (lesion_mask == 255) & (tissue_area == 255)
        # 注意：训练集用BGR，红色 = [0, 0, 255]
        mask_bgr[valid_lesion] = [0, 0, 255]

    # ── Step4: 转换为RGB PIL图像（GAN输入需要RGB）────────────────────────────
    mask_rgb = Image.fromarray(cv2.cvtColor(mask_bgr, cv2.COLOR_BGR2RGB))

    return mask_bgr, mask_rgb


# ─────────────────────────────────────────────────────────────────────────────
# GAN推理
# ─────────────────────────────────────────────────────────────────────────────
def mask_to_ultrasound(generator, mask_rgb_pil):
    """PIL RGB Mask → 超声灰度图"""
    transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])
    tensor = transform(mask_rgb_pil).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        output = generator(tensor)
    arr = output[0, 0].cpu().numpy()
    return np.clip((arr + 1.0) / 2.0 * 255, 0, 255).astype(np.uint8)


# ─────────────────────────────────────────────────────────────────────────────
# 测试用例
# ─────────────────────────────────────────────────────────────────────────────
def run_all_tests(generator):
    """
    6个测试用例，完全模拟真实训练集Mask的样式

    结节位置用像素坐标指定（在256×256画布上）
    scan_type 决定超声有效区域的形状
    """
    S = IMAGE_SIZE

    test_cases = [
        {
            'name': '01_无结节_全铺',
            'scan_type': 'full',  # ← 改为full
            'lesion_center': None,
            'lesion_radius': None,
            'lesion_shape': None,
            'description': '全铺扫查，无结节\n（验证纯乳腺纹理）',
        },
        {
            'name': '02_小结节_中央',
            'scan_type': 'full',  # ← 改为full
            'lesion_center': (S // 2, S // 3),
            'lesion_radius': (18, 15),
            'lesion_shape': 'round',
            'description': '全铺扫查 + 中央小结节',
        },
        {
            'name': '03_中等结节_偏左',
            'scan_type': 'full',  # ← 改为full
            'lesion_center': (S // 3, S // 2),
            'lesion_radius': (30, 26),
            'lesion_shape': 'round',
            'description': '全铺扫查 + 左侧中等结节',
        },
        {
            'name': '04_大结节_偏右',
            'scan_type': 'full',  # ← 改为full
            'lesion_center': (int(S * 0.65), S // 2),
            'lesion_radius': (45, 40),
            'lesion_shape': 'round',
            'description': '全铺扫查 + 右侧大结节',
        },
        {
            'name': '05_不规则结节_中央',
            'scan_type': 'full',  # ← 改为full
            'lesion_center': (S // 2, int(S * 0.55)),
            'lesion_radius': (35, 32),
            'lesion_shape': 'irregular',
            'description': '全铺扫查 + 不规则结节',
        },
        {
            'name': '06_无结节_标准',
            'scan_type': 'standard',  # 保留一个有少量黑边的对比
            'lesion_center': None,
            'lesion_radius': None,
            'lesion_shape': None,
            'description': '标准扫查，无结节\n（与全铺对比）',
        },
    ]

    results = []
    print("=" * 60)
    print(f"🧪 开始测试，共 {len(test_cases)} 个用例")
    print("=" * 60)

    for i, case in enumerate(test_cases):
        print(f"\n[{i + 1}/{len(test_cases)}] {case['name']}")

        # 生成Mask（完全模拟训练集流程）
        mask_bgr, mask_rgb = simulate_real_mask(
            lesion_center=case['lesion_center'],
            lesion_radius=case['lesion_radius'],
            lesion_shape=case.get('lesion_shape', 'round'),
            scan_type=case['scan_type'],
            size=IMAGE_SIZE,
        )

        # 统计Mask内容
        green_pct = (mask_bgr[:, :, 1] > 100).mean() * 100
        red_pct = (mask_bgr[:, :, 2] > 100).mean() * 100
        print(f"  Mask: 绿色{green_pct:.0f}% | 红色{red_pct:.1f}%")

        # GAN推理
        us_arr = mask_to_ultrasound(generator, mask_rgb)

        # 保存对比图
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        fig.suptitle(case['description'], fontsize=13, fontweight='bold')

        axes[0].imshow(mask_rgb)
        axes[0].set_title(f"输入Mask\n绿色{green_pct:.0f}% 红色{red_pct:.1f}%",
                          fontsize=10)
        axes[0].axis('off')

        axes[1].imshow(us_arr, cmap='gray', vmin=0, vmax=255)
        axes[1].set_title("GAN生成超声图", fontsize=10)
        axes[1].axis('off')

        plt.tight_layout()
        save_path = os.path.join(OUTPUT_DIR, f"{case['name']}.png")
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  💾 {save_path}")
        plt.close()

        results.append({
            'name': case['name'],
            'mask': mask_rgb,
            'us': us_arr,
            'desc': case['description'],
        })

    return results


# ─────────────────────────────────────────────────────────────────────────────
# 汇总图
# ─────────────────────────────────────────────────────────────────────────────
def make_summary_grid(results):
    n = len(results)
    fig, axes = plt.subplots(n, 2, figsize=(10, 4 * n))
    fig.suptitle("GAN超声生成 - 完整测试汇总\n（模拟真实训练集Mask格式）",
                 fontsize=14, fontweight='bold')

    for i, res in enumerate(results):
        axes[i, 0].imshow(res['mask'])
        axes[i, 0].set_title(f"{res['name']}\nMask输入", fontsize=8)
        axes[i, 0].axis('off')

        axes[i, 1].imshow(res['us'], cmap='gray', vmin=0, vmax=255)
        axes[i, 1].set_title("GAN生成超声", fontsize=8)
        axes[i, 1].axis('off')

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "00_summary_grid.png")
    plt.savefig(path, dpi=150, bbox_inches='tight')
    print(f"\n📊 汇总图: {path}")
    plt.show()


# ─────────────────────────────────────────────────────────────────────────────
# 真实数据对比（保留原来的功能）
# ─────────────────────────────────────────────────────────────────────────────
def test_with_real_training_samples(generator):
    print("\n" + "=" * 60)
    print("🔍 使用真实训练数据验证")
    print("=" * 60)

    mask_dir = "datasets/breast_ultrasound/trainA"
    real_dir = "datasets/breast_ultrasound/trainB"

    if not os.path.exists(mask_dir):
        print(f"⚠️  找不到: {mask_dir}")
        return

    files = sorted(os.listdir(mask_dir))
    n_sample = min(6, len(files))
    indices = np.linspace(0, len(files) - 1, n_sample, dtype=int)
    selected = [files[i] for i in indices]

    transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])

    fig, axes = plt.subplots(n_sample, 3, figsize=(12, 4 * n_sample))
    fig.suptitle("真实训练数据验证：Mask | GAN生成 | 真实超声",
                 fontsize=13, fontweight='bold')

    for i, fname in enumerate(selected):
        mask_pil = Image.open(os.path.join(mask_dir, fname)).convert("RGB")
        real_path = os.path.join(real_dir, fname)

        tensor = transform(mask_pil).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            out = generator(tensor)
        fake = ((out[0, 0].cpu().numpy() + 1) / 2 * 255).clip(0, 255).astype(np.uint8)

        axes[i, 0].imshow(mask_pil)
        axes[i, 0].set_title(f"Mask: {fname}", fontsize=7)
        axes[i, 0].axis('off')

        axes[i, 1].imshow(fake, cmap='gray')
        axes[i, 1].set_title("GAN生成", fontsize=7)
        axes[i, 1].axis('off')

        if os.path.exists(real_path):
            axes[i, 2].imshow(np.array(Image.open(real_path).convert("L")),
                              cmap='gray')
            axes[i, 2].set_title("真实超声(GT)", fontsize=7)
        axes[i, 2].axis('off')

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "00_real_data_test.png")
    plt.savefig(path, dpi=150, bbox_inches='tight')
    print(f"  💾 {path}")
    plt.show()


# ─────────────────────────────────────────────────────────────────────────────
# 入口
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    print("=" * 60)
    print("  🧪 GAN超声生成快速验证工具（真实Mask格式版）")
    print("=" * 60)
    print(f"  模型: {MODEL_PATH}")
    print(f"  设备: {DEVICE}\n")

    generator = load_generator(MODEL_PATH)

    results = run_all_tests(generator)
    make_summary_grid(results)
    test_with_real_training_samples(generator)

    print("\n" + "=" * 60)
    print("✅ 验证完成！")
    print(f"   结果目录: {OUTPUT_DIR}/")
    print("=" * 60)