# plot_fx.py
"""
独立 F-x 关系图绘制程序
使用方法：python plot_fx.py fx_data_20240101_120000.txt
或直接运行自动找最新文件：python plot_fx.py
"""
import sys
import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.family'] = 'SimHei'   # 中文字体
matplotlib.rcParams['axes.unicode_minus'] = False


def load_data(filepath: str):
    """读取 txt 数据文件"""
    data = np.loadtxt(filepath, delimiter='\t', skiprows=1)
    if data.ndim == 1:
        data = data.reshape(1, -1)
    displacement = data[:, 0] * 1000  # m → mm，更直观
    force        = data[:, 1]
    return displacement, force


def plot_fx(filepath: str):
    """绘制 F-x 关系图"""
    print(f"读取文件: {filepath}")
    displacement, force = load_data(filepath)
    print(f"共 {len(displacement)} 个数据点")
    print(f"位移范围: {displacement.min():.2f} ~ {displacement.max():.2f} mm")
    print(f"力范围:   {force.min():.3f} ~ {force.max():.3f} N")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(
        f"乳腺形变位移 vs 力反馈合力\n数据文件: {os.path.basename(filepath)}",
        fontsize=13
    )

    # ── 左图：F-x 散点图（核心关系）──────────────────────────────
    ax1 = axes[0]
    sc = ax1.scatter(
        displacement, force,
        c=np.arange(len(displacement)),  # 颜色随时间变化，区分加压/卸压
        cmap='RdYlGn_r',                 # 绿→黄→红，越红越靠后
        s=8,
        alpha=0.7
    )
    plt.colorbar(sc, ax=ax1, label='时间顺序（采样点序号）')

    ax1.set_xlabel('乳腺最大形变位移 (mm)', fontsize=11)
    ax1.set_ylabel('力反馈合力 |F| (N)',    fontsize=11)
    ax1.set_title('F-x 关系散点图',         fontsize=12)
    ax1.grid(True, alpha=0.3)

    # 拟合趋势线（线性）
    if len(displacement) > 2:
        coeffs = np.polyfit(displacement, force, 1)
        x_fit  = np.linspace(displacement.min(), displacement.max(), 100)
        y_fit  = np.polyval(coeffs, x_fit)
        ax1.plot(
            x_fit, y_fit,
            'r--', linewidth=1.5,
            label=f'线性拟合: F = {coeffs[0]:.3f}x + {coeffs[1]:.3f}'
        )
        ax1.legend(fontsize=9)

    # ── 右图：随时间变化的位移和力（看加压/卸压过程）────────────
    ax2 = axes[1]
    t   = np.arange(len(displacement))

    ax2_twin = ax2.twinx()  # 双Y轴

    line1, = ax2.plot(
        t, displacement,
        color='steelblue', linewidth=1.2,
        label='位移 (mm)'
    )
    line2, = ax2_twin.plot(
        t, force,
        color='tomato', linewidth=1.2,
        label='合力 (N)'
    )

    ax2.set_xlabel('采样序号',           fontsize=11)
    ax2.set_ylabel('位移 (mm)',          fontsize=11, color='steelblue')
    ax2_twin.set_ylabel('合力 |F| (N)', fontsize=11, color='tomato')
    ax2.set_title('位移 & 力随时间变化', fontsize=12)
    ax2.grid(True, alpha=0.3)

    # 合并图例
    lines  = [line1, line2]
    labels = [l.get_label() for l in lines]
    ax2.legend(lines, labels, fontsize=9, loc='upper left')

    # ── 统计信息文本框 ────────────────────────────────────────────
    stats_text = (
        f"数据点数: {len(displacement)}\n"
        f"最大位移: {displacement.max():.2f} mm\n"
        f"最大合力: {force.max():.3f} N\n"
        f"平均合力: {force.mean():.3f} N"
    )
    ax1.text(
        0.02, 0.97, stats_text,
        transform=ax1.transAxes,
        fontsize=9,
        verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8)
    )

    plt.tight_layout()

    # ── 保存图片 ──────────────────────────────────────────────────
    save_path = filepath.replace('.txt', '_plot.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"✓ 图表已保存 → {save_path}")

    plt.show()


if __name__ == '__main__':
    # 自动找最新文件 或 指定文件
    if len(sys.argv) > 1:
        filepath = sys.argv[1]
    else:
        files = glob.glob("archive/fx_data/fx_data_*.txt")
        if not files:
            print("❌ 未找到 archive/fx_data/fx_data_*.txt 文件，请先运行仿真")
            sys.exit(1)
        filepath = max(files, key=os.path.getmtime)  # 取最新的
        print(f"自动选择最新文件: {filepath}")

    if not os.path.exists(filepath):
        print(f"❌ 文件不存在: {filepath}")
        sys.exit(1)

    plot_fx(filepath)