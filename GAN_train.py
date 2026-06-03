# =============================================================================
# GAN_train.py —— Pix2Pix 乳腺超声图像生成 (完整优化版)
# 针对 GTX 1650 Ti (4GB VRAM) 深度优化
# =============================================================================

import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from PIL import Image
import random


# ─────────────────────────────────────────────────────────────────────────────
# 0. 跨平台多进程安全入口
#    Windows 必须把所有训练代码放在这个判断之内，否则多进程会崩溃
# ─────────────────────────────────────────────────────────────────────────────
# （训练逻辑全部在文件末尾的 if __name__ == '__main__': 中启动）


# ═════════════════════════════════════════════════════════════════════════════
# 1. 超参数配置（集中管理，方便调参）
# ═════════════════════════════════════════════════════════════════════════════
class Config:
    # ── 路径 ──────────────────────────────────────────────────────────────────
    DATASET_DIR = "datasets/breast_ultrasound/"
    SAMPLE_DIR = "training_samples/"
    CHECKPOINT_DIR = "checkpoints/"

    # ── 训练基础参数 ──────────────────────────────────────────────────────────
    BATCH_SIZE = 2  # 4G显存最稳定值；若OOM请改为1
    EPOCHS = 150  # 总训练轮数
    IMAGE_SIZE = 256  # 图像尺寸（需与数据预处理一致）
    VAL_SPLIT = 0.1  # 10%数据用于验证

    # ── 优化器参数 ────────────────────────────────────────────────────────────
    LR = 0.0002  # 初始学习率（Pix2Pix经典值）
    B1 = 0.5  # Adam β1
    B2 = 0.999  # Adam β2

    # ── 损失权重 ──────────────────────────────────────────────────────────────
    LAMBDA_PIXEL = 100  # L1损失权重（越大生成图越接近真实图，但可能模糊）

    # ── 学习率调度 ────────────────────────────────────────────────────────────
    LR_DECAY_START = 100  # 从第100轮开始线性衰减学习率
    # 衰减规则：第100轮保持LR，第150轮衰减至0

    # ── 日志与保存频率 ────────────────────────────────────────────────────────
    PRINT_FREQ = 20  # 每N个batch打印一次日志
    PREVIEW_FREQ = 150  # 每N个batch保存一张预览图
    SAVE_EPOCH = 10  # 每N轮保存一次模型权重

    # ── 断点续训 ──────────────────────────────────────────────────────────────
    RESUME = False  # 是否从断点续训
    RESUME_EPOCH = 0  # 从第几轮开始续训（需配合下面的路径）
    RESUME_G_PATH = ""  # 生成器权重路径
    RESUME_D_PATH = ""  # 判别器权重路径

    # ── 系统 ──────────────────────────────────────────────────────────────────
    # Windows下num_workers必须为0，Linux/Mac可以设2-4加速数据加载
    NUM_WORKERS = 0 if os.name == 'nt' else 2
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


cfg = Config()


# ═════════════════════════════════════════════════════════════════════════════
# 2. 数据集（含数据增强）
# ═════════════════════════════════════════════════════════════════════════════
class UltrasoundDataset(Dataset):
    """
    加载成对的 (Mask彩色图, 真实超声灰度图)

    目录结构：
        datasets/breast_ultrasound/
            trainA/   ← 彩色Mask (RGB, 3通道)
            trainB/   ← 真实超声 (灰度, 1通道)
    """

    def __init__(self, root_dir, augment=True):
        self.dir_A = os.path.join(root_dir, 'trainA')
        self.dir_B = os.path.join(root_dir, 'trainB')
        self.augment = augment

        # 只取两个文件夹都存在的文件名（防止文件不对齐）
        files_A = set(os.listdir(self.dir_A))
        files_B = set(os.listdir(self.dir_B))
        self.filenames = sorted(files_A & files_B)  # 取交集

        if len(self.filenames) == 0:
            raise RuntimeError(
                f"❌ 数据集为空！请检查路径：\n  {self.dir_A}\n  {self.dir_B}"
            )

        print(f"  ✓ 找到 {len(self.filenames)} 对有效图片")

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        name = self.filenames[idx]

        # 读取图片
        img_A = Image.open(os.path.join(self.dir_A, name)).convert("RGB")  # Mask → RGB
        img_B = Image.open(os.path.join(self.dir_B, name)).convert("L")  # 超声 → 灰度

        # ── 数据增强（训练集专用，验证集不做）────────────────────────────────
        if self.augment:
            img_A, img_B = self._paired_augment(img_A, img_B)

        # ── 转为Tensor并归一化到 [-1, 1] ─────────────────────────────────────
        # Mask：RGB 3通道，归一化 [-1,1]
        tensor_A = transforms.Compose([
            transforms.ToTensor(),  # [0,255] → [0,1]
            transforms.Normalize([0.5, 0.5, 0.5],
                                 [0.5, 0.5, 0.5])  # [0,1]   → [-1,1]
        ])(img_A)

        # 超声：灰度 1通道，归一化 [-1,1]
        tensor_B = transforms.Compose([
            transforms.ToTensor(),  # [0,255] → [0,1]
            transforms.Normalize([0.5], [0.5])  # [0,1]   → [-1,1]
        ])(img_B)

        return tensor_A, tensor_B

    def _paired_augment(self, img_A, img_B):
        """
        成对数据增强：mask和超声图必须做完全相同的变换！
        否则mask和超声图会对不上，网络学不到东西。
        """
        # 1. 随机水平翻转（概率50%）
        if random.random() > 0.5:
            img_A = TF.hflip(img_A)
            img_B = TF.hflip(img_B)

        # 2. 随机轻微旋转（±10°，超声图不适合大角度）
        if random.random() > 0.5:
            angle = random.uniform(-10, 10)
            img_A = TF.rotate(img_A, angle)
            img_B = TF.rotate(img_B, angle)

        # 3. 随机亮度/对比度抖动（只对超声图B，mask不需要）
        if random.random() > 0.5:
            brightness_factor = random.uniform(0.85, 1.15)
            contrast_factor = random.uniform(0.85, 1.15)
            img_B = TF.adjust_brightness(img_B, brightness_factor)
            img_B = TF.adjust_contrast(img_B, contrast_factor)

        return img_A, img_B


# ═════════════════════════════════════════════════════════════════════════════
# 3. 网络架构
# ═════════════════════════════════════════════════════════════════════════════

def weights_init_normal(m):
    """Pix2Pix标准权重初始化"""
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0.0)


# ── 3-A: U-Net 生成器 ─────────────────────────────────────────────────────

class DownBlock(nn.Module):
    """编码器下采样块：Conv → [BN] → LeakyReLU"""

    def __init__(self, in_ch, out_ch, use_norm=True):
        super().__init__()
        layers = [
            nn.Conv2d(in_ch, out_ch, kernel_size=4, stride=2,
                      padding=1, bias=not use_norm)
        ]
        if use_norm:
            layers.append(nn.BatchNorm2d(out_ch))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class UpBlock(nn.Module):
    """
    解码器上采样块：ConvTranspose → BN → [Dropout] → ReLU

    注意：forward返回的是 拼接后 的特征图（含skip connection）
    所以out_ch是拼接前单侧的通道数，拼接后变为 out_ch × 2
    """

    def __init__(self, in_ch, out_ch, use_dropout=False):
        super().__init__()
        layers = [
            nn.ConvTranspose2d(in_ch, out_ch, kernel_size=4,
                               stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
        ]
        if use_dropout:
            layers.append(nn.Dropout(0.5))
        layers.append(nn.ReLU(inplace=True))
        self.model = nn.Sequential(*layers)

    def forward(self, x, skip):
        """
        Args:
            x    : 来自上一个解码器层的特征图
            skip : 来自对应编码器层的特征图（skip connection）
        Returns:
            拼接后的特征图，通道数 = out_ch + skip.channels
        """
        return torch.cat([self.model(x), skip], dim=1)


class GeneratorUNet(nn.Module):
    """
    U-Net 生成器
    输入: 彩色Mask (3通道, 256×256)
    输出: 超声灰度图 (1通道, 256×256)

    通道变化追踪（256×256输入）：
    编码器:
      x      → d1: [B, 64,  128, 128]
      d1     → d2: [B, 128,  64,  64]
      d2     → d3: [B, 256,  32,  32]
      d3     → d4: [B, 512,  16,  16]
      d4     → d5: [B, 512,   8,   8]
      d5     → d6: [B, 512,   4,   4]
      d6     → d7: [B, 512,   2,   2]
      d7     → bn: [B, 512,   1,   1]  ← 瓶颈层
    解码器（含skip concat）：
      bn     → u1: [B,1024,  2,   2]   (512上采样后 + d7跳接)
      u1     → u2: [B,1024,  4,   4]
      u2     → u3: [B,1024,  8,   8]
      u3     → u4: [B,1024, 16,  16]
      u4     → u5: [B, 512, 32,  32]
      u5     → u6: [B, 256, 64,  64]
      u6     → u7: [B, 128,128, 128]
      u7     → out:[B,   1,256, 256]
    """

    def __init__(self, in_channels=3, out_channels=1):
        super().__init__()
        # 编码器
        self.down1 = DownBlock(in_channels, 64, use_norm=False)  # 第一层不用BN
        self.down2 = DownBlock(64, 128)
        self.down3 = DownBlock(128, 256)
        self.down4 = DownBlock(256, 512)
        self.down5 = DownBlock(512, 512)
        self.down6 = DownBlock(512, 512)
        self.down7 = DownBlock(512, 512)
        # 瓶颈层
        self.bottleneck = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True)
        )
        # 解码器（in_ch = 上一层输出 + skip通道数）
        self.up1 = UpBlock(512, 512, use_dropout=True)  # in=512,  out拼接后=1024
        self.up2 = UpBlock(1024, 512, use_dropout=True)  # in=1024, out拼接后=1024
        self.up3 = UpBlock(1024, 512, use_dropout=True)  # in=1024, out拼接后=1024
        self.up4 = UpBlock(1024, 512)  # in=1024, out拼接后=1024
        self.up5 = UpBlock(1024, 256)  # in=1024, out拼接后=512
        self.up6 = UpBlock(512, 128)  # in=512,  out拼接后=256
        self.up7 = UpBlock(256, 64)  # in=256,  out拼接后=128
        # 最终输出层
        self.final = nn.Sequential(
            nn.ConvTranspose2d(128, out_channels, kernel_size=4,
                               stride=2, padding=1),
            nn.Tanh()  # 输出范围 [-1, 1]
        )

    def forward(self, x):
        # 编码
        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        d5 = self.down5(d4)
        d6 = self.down6(d5)
        d7 = self.down7(d6)
        bn = self.bottleneck(d7)
        # 解码（每个UpBlock内部自动拼接skip）
        u1 = self.up1(bn, d7)
        u2 = self.up2(u1, d6)
        u3 = self.up3(u2, d5)
        u4 = self.up4(u3, d4)
        u5 = self.up5(u4, d3)
        u6 = self.up6(u5, d2)
        u7 = self.up7(u6, d1)
        return self.final(u7)


# ── 3-B: PatchGAN 判别器 ──────────────────────────────────────────────────

class Discriminator(nn.Module):
    """
    PatchGAN 判别器

    输入: concat(Mask[3ch], 超声图[1ch]) = 4通道, 256×256
    输出: Patch评分图，尺寸为 (B, 1, 30, 30)

    每个输出像素感受野对应输入图像的 70×70 区域
    → 判断局部纹理是否真实，比全图判断更适合超声纹理生成

    尺寸推导（stride=2下采样4次，最后stride=1）：
      256 → 128 → 64 → 32 → 16 → 30 (最后两层padding影响)
    """

    def __init__(self, in_channels=4):
        super().__init__()

        def d_block(in_ch, out_ch, stride=2, norm=True):
            layers = [nn.Conv2d(in_ch, out_ch, kernel_size=4,
                                stride=stride, padding=1, bias=not norm)]
            if norm:
                layers.append(nn.BatchNorm2d(out_ch))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *d_block(in_channels, 64, norm=False),  # → (B, 64,  128, 128)
            *d_block(64, 128),  # → (B, 128,  64,  64)
            *d_block(128, 256),  # → (B, 256,  32,  32)
            *d_block(256, 512, stride=1),  # → (B, 512,  31,  31) stride=1保留更多空间信息
            nn.Conv2d(512, 1, kernel_size=4,
                      stride=1, padding=1)  # → (B,   1,  30,  30)
        )

    def forward(self, img_mask, img_us):
        """
        Args:
            img_mask : Mask图 (B, 3, 256, 256)
            img_us   : 超声图 (B, 1, 256, 256)
        Returns:
            patch_score : (B, 1, 30, 30)
        """
        x = torch.cat([img_mask, img_us], dim=1)  # → (B, 4, 256, 256)
        return self.model(x)


# ═════════════════════════════════════════════════════════════════════════════
# 4. 工具函数
# ═════════════════════════════════════════════════════════════════════════════

def save_preview_image(mask, real_us, fake_us, epoch, batches_done, save_dir):
    """
    保存训练预览图，并排显示 [输入Mask | 生成超声 | 真实超声]

    图像排列（从左到右）：
        左：输入的彩色Mask（转灰度显示）
        中：生成器生成的超声图
        右：真实超声图（Ground Truth）
    """
    from torchvision.utils import save_image

    # 取第一张（batch中的第一张）
    # mask是3通道，转为灰度（取均值）方便并排对比
    mask_gray = mask[0:1].mean(dim=1, keepdim=True)  # (1,3,H,W) → (1,1,H,W)

    # 三张图横向拼接
    comparison = torch.cat([
        mask_gray,  # 左：输入Mask（灰度化显示）
        fake_us[0:1],  # 中：生成图
        real_us[0:1]  # 右：真实图
    ], dim=3)  # dim=3 沿W(宽度)方向拼接 → 三图横排

    # 反归一化 [-1,1] → [0,1] 再保存
    comparison = (comparison + 1.0) / 2.0
    comparison = torch.clamp(comparison, 0, 1)

    save_path = os.path.join(save_dir, f"ep{epoch:03d}_b{batches_done:05d}.png")
    save_image(comparison, save_path)


def get_patch_label(batch_size, is_real, device):
    """
    动态生成PatchGAN标签

    ✅ 修复原代码硬编码(15,15)的问题
    通过前向传播一次空数据来自动获取正确的patch尺寸

    Args:
        batch_size : 批次大小
        is_real    : True=全1(真实标签), False=全0(虚假标签)
        device     : 目标设备

    Returns:
        标签Tensor, shape=(batch_size, 1, 30, 30)
    """
    # PatchGAN输出固定为 (B, 1, 30, 30)（由网络结构决定）
    PATCH_H, PATCH_W = 30, 30
    value = 1.0 if is_real else 0.0
    return torch.full(
        (batch_size, 1, PATCH_H, PATCH_W),
        fill_value=value,
        dtype=torch.float32,
        device=device,
        requires_grad=False
    )


def check_vram():
    """打印当前VRAM使用情况（仅CUDA）"""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024 ** 2
        reserved = torch.cuda.memory_reserved() / 1024 ** 2
        print(f"  📊 VRAM: 已分配 {allocated:.0f}MB / 已预留 {reserved:.0f}MB")


def save_checkpoint(generator, discriminator,
                    optimizer_G, optimizer_D,
                    epoch, save_dir):
    """保存完整训练状态（支持断点续训）"""
    checkpoint = {
        'epoch': epoch,
        'generator': generator.state_dict(),
        'discriminator': discriminator.state_dict(),
        'optimizer_G': optimizer_G.state_dict(),
        'optimizer_D': optimizer_D.state_dict(),
    }
    path = os.path.join(save_dir, f"checkpoint_epoch_{epoch:03d}.pth")
    torch.save(checkpoint, path)
    print(f"  💾 完整检查点已保存: {path}")


def load_checkpoint(path, generator, discriminator,
                    optimizer_G, optimizer_D, device):
    """加载断点，恢复训练状态"""
    checkpoint = torch.load(path, map_location=device)
    generator.load_state_dict(checkpoint['generator'])
    discriminator.load_state_dict(checkpoint['discriminator'])
    optimizer_G.load_state_dict(checkpoint['optimizer_G'])
    optimizer_D.load_state_dict(checkpoint['optimizer_D'])
    start_epoch = checkpoint['epoch'] + 1
    print(f"  ✅ 已从 Epoch {checkpoint['epoch']} 恢复训练")
    return start_epoch


# ═════════════════════════════════════════════════════════════════════════════
# 5. 验证函数
# ═════════════════════════════════════════════════════════════════════════════

def validate(generator, val_loader, criterion_pixel, device):
    """
    在验证集上评估生成器的L1损失

    Returns:
        平均L1损失（越小说明生成图像越接近真实图像）
    """
    generator.eval()  # 切换到评估模式（关闭Dropout、BN使用统计值）
    total_loss = 0.0

    with torch.no_grad():  # 验证不需要计算梯度，节省显存
        for masks, real_US in val_loader:
            masks = masks.to(device)
            real_US = real_US.to(device)
            fake_US = generator(masks)
            total_loss += criterion_pixel(fake_US, real_US).item()

    generator.train()  # 切回训练模式
    return total_loss / len(val_loader)


# ═════════════════════════════════════════════════════════════════════════════
# 6. 主训练函数
# ═════════════════════════════════════════════════════════════════════════════

def train():
    print("=" * 65)
    print("  🏥 乳腺超声 Pix2Pix GAN 训练器 (GTX 1650 Ti 优化版)")
    print("=" * 65)
    print(f"  🖥️  设备: {cfg.DEVICE}")
    print(f"  📦 批次: {cfg.BATCH_SIZE} | 轮数: {cfg.EPOCHS} | 学习率: {cfg.LR}")

    # 创建输出目录
    os.makedirs(cfg.SAMPLE_DIR, exist_ok=True)
    os.makedirs(cfg.CHECKPOINT_DIR, exist_ok=True)

    # ── 5-1. 加载数据集 ────────────────────────────────────────────────────
    print("\n📂 加载数据集...")
    full_dataset = UltrasoundDataset(cfg.DATASET_DIR, augment=True)

    # 划分训练集/验证集
    val_size = max(1, int(len(full_dataset) * cfg.VAL_SPLIT))
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = random_split(
        full_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)  # 固定种子，保证可复现
    )
    # 验证集不做增强
    val_dataset.dataset.augment = False

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.BATCH_SIZE,
        shuffle=True,
        drop_last=True,  # 丢弃最后不足batch的数据，避免BN出错
        num_workers=cfg.NUM_WORKERS,
        pin_memory=True if cfg.DEVICE.type == 'cuda' else False
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.BATCH_SIZE,
        shuffle=False,
        num_workers=cfg.NUM_WORKERS
    )

    print(f"  ✓ 训练集: {train_size} 张 | 验证集: {val_size} 张")
    print(f"  ✓ 训练批次数: {len(train_loader)}/轮")

    # ── 5-2. 初始化网络 ────────────────────────────────────────────────────
    print("\n🔧 初始化网络...")
    generator = GeneratorUNet(in_channels=3, out_channels=1).to(cfg.DEVICE)
    discriminator = Discriminator(in_channels=4).to(cfg.DEVICE)

    generator.apply(weights_init_normal)
    discriminator.apply(weights_init_normal)

    # 打印参数量
    g_params = sum(p.numel() for p in generator.parameters())
    d_params = sum(p.numel() for p in discriminator.parameters())
    print(f"  ✓ 生成器参数量:   {g_params / 1e6:.2f}M")
    print(f"  ✓ 判别器参数量:   {d_params / 1e6:.2f}M")

    # ── 5-3. 损失函数 ──────────────────────────────────────────────────────
    criterion_GAN = nn.BCEWithLogitsLoss()  # 对抗损失（含sigmoid，数值稳定）
    criterion_pixel = nn.L1Loss()  # 像素级L1损失

    # ── 5-4. 优化器 ────────────────────────────────────────────────────────
    optimizer_G = optim.Adam(
        generator.parameters(), lr=cfg.LR, betas=(cfg.B1, cfg.B2)
    )
    optimizer_D = optim.Adam(
        discriminator.parameters(), lr=cfg.LR, betas=(cfg.B1, cfg.B2)
    )

    # ── 5-5. 学习率调度器（线性衰减）──────────────────────────────────────
    # 前 LR_DECAY_START 轮保持LR不变，之后线性衰减到0
    def lr_lambda(epoch):
        if epoch < cfg.LR_DECAY_START:
            return 1.0
        # 线性从1降到0
        decay_epochs = cfg.EPOCHS - cfg.LR_DECAY_START
        return max(0.0, 1.0 - (epoch - cfg.LR_DECAY_START) / decay_epochs)

    scheduler_G = optim.lr_scheduler.LambdaLR(optimizer_G, lr_lambda)
    scheduler_D = optim.lr_scheduler.LambdaLR(optimizer_D, lr_lambda)

    # ── 5-6. 断点续训 ──────────────────────────────────────────────────────
    start_epoch = 0
    if cfg.RESUME and cfg.RESUME_G_PATH:
        start_epoch = load_checkpoint(
            cfg.RESUME_G_PATH,
            generator, discriminator,
            optimizer_G, optimizer_D,
            cfg.DEVICE
        )
        # 快进调度器到正确的epoch
        for _ in range(start_epoch):
            scheduler_G.step()
            scheduler_D.step()

    # ── 5-7. 显存检查 ──────────────────────────────────────────────────────
    print("\n📊 初始显存状态:")
    check_vram()

    # ══════════════════════════════════════════════════════════════════════
    # 🔁 训练主循环
    # ══════════════════════════════════════════════════════════════════════
    print(f"\n🔥 开始训练！从 Epoch {start_epoch} 到 {cfg.EPOCHS - 1}")
    print("-" * 65)

    best_val_loss = float('inf')  # 追踪最优验证损失

    for epoch in range(start_epoch, cfg.EPOCHS):
        epoch_start = time.time()
        epoch_loss_G = 0.0
        epoch_loss_D = 0.0

        generator.train()
        discriminator.train()

        for i, (masks, real_US) in enumerate(train_loader):
            masks = masks.to(cfg.DEVICE)  # (B, 3, 256, 256)
            real_US = real_US.to(cfg.DEVICE)  # (B, 1, 256, 256)
            B = masks.size(0)

            # 动态获取正确的patch标签尺寸（避免硬编码错误）
            valid = get_patch_label(B, is_real=True, device=cfg.DEVICE)
            fake = get_patch_label(B, is_real=False, device=cfg.DEVICE)

            # ──────────────────────────────────────────────────────────
            # 阶段1：训练生成器 G
            # 目标：让G生成的图骗过D，同时与真实图像素差异小
            # ──────────────────────────────────────────────────────────
            optimizer_G.zero_grad()

            fake_US = generator(masks)  # G生成伪造超声图
            pred_fake = discriminator(masks, fake_US)  # D对伪造图打分
            loss_GAN = criterion_GAN(pred_fake, valid)  # G希望D给出"真"评分
            loss_pixel = criterion_pixel(fake_US, real_US)  # 像素级约束
            loss_G = loss_GAN + cfg.LAMBDA_PIXEL * loss_pixel

            loss_G.backward()
            optimizer_G.step()

            # ──────────────────────────────────────────────────────────
            # 阶段2：训练判别器 D
            # 目标：正确区分真实图(→1)和生成图(→0)
            # ──────────────────────────────────────────────────────────
            optimizer_D.zero_grad()

            pred_real = discriminator(masks, real_US)  # 真图得分
            loss_D_real = criterion_GAN(pred_real, valid)  # 期望→1

            pred_fake_detach = discriminator(masks, fake_US.detach())  # 假图得分
            loss_D_fake = criterion_GAN(pred_fake_detach, fake)  # 期望→0

            # D的总损失取均值（防止D训练太快压制G）
            loss_D = 0.5 * (loss_D_real + loss_D_fake)

            loss_D.backward()
            optimizer_D.step()

            # ── 累计损失 ────────────────────────────────────────────
            epoch_loss_G += loss_G.item()
            epoch_loss_D += loss_D.item()
            batches_done = epoch * len(train_loader) + i

            # ── 打印日志 ────────────────────────────────────────────
            if i % cfg.PRINT_FREQ == 0:
                lr_now = optimizer_G.param_groups[0]['lr']
                print(
                    f"  [Ep {epoch:03d}/{cfg.EPOCHS}]"
                    f"  [Batch {i:03d}/{len(train_loader)}]"
                    f"  D: {loss_D.item():.4f}"
                    f"  G: {loss_G.item():.4f}"
                    f"  (GAN:{loss_GAN.item():.3f}"
                    f"  L1:{loss_pixel.item():.3f})"
                    f"  LR: {lr_now:.6f}"
                )

            # ── 保存预览图 ──────────────────────────────────────────
            if batches_done % cfg.PREVIEW_FREQ == 0:
                save_preview_image(
                    masks, real_US, fake_US,
                    epoch, batches_done, cfg.SAMPLE_DIR
                )

        # ── Epoch 结束：学习率调度 ────────────────────────────────────────
        scheduler_G.step()
        scheduler_D.step()

        # ── Epoch 结束：验证 ──────────────────────────────────────────────
        val_loss = validate(generator, val_loader, criterion_pixel, cfg.DEVICE)

        avg_G = epoch_loss_G / len(train_loader)
        avg_D = epoch_loss_D / len(train_loader)
        elapsed = time.time() - epoch_start

        print(
            f"\n  ⏱️  Ep {epoch:03d} 完成 | "
            f"耗时: {elapsed:.1f}s | "
            f"均值 D: {avg_D:.4f}  G: {avg_G:.4f} | "
            f"验证L1: {val_loss:.4f}"
        )

        # 保存最优模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_path = os.path.join(cfg.CHECKPOINT_DIR, "generator_best.pth")
            torch.save(generator.state_dict(), best_path)
            print(f"  🏆 新最优验证损失 {val_loss:.4f}，已保存: {best_path}")

        # ── 定期保存完整检查点 ────────────────────────────────────────────
        if (epoch + 1) % cfg.SAVE_EPOCH == 0:
            save_checkpoint(
                generator, discriminator,
                optimizer_G, optimizer_D,
                epoch, cfg.CHECKPOINT_DIR
            )

        check_vram()
        print("-" * 65)

    print("\n🎉 训练全部完成！")
    print(f"  最优验证L1损失: {best_val_loss:.4f}")
    print(f"  最优模型路径: {os.path.join(cfg.CHECKPOINT_DIR, 'generator_best.pth')}")


# ═════════════════════════════════════════════════════════════════════════════
# 入口（Windows多进程安全）
# ═════════════════════════════════════════════════════════════════════════════
if __name__ == '__main__':
    # 固定随机种子，保证实验可复现
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    train()