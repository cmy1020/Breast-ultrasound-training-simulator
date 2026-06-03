# stiffness_analysis.py
"""
实际采集数据刚度分析：
1. 提取 6N 以内的线性段
2. 计算刚度（N/mm）
3. 输出结果供仿真参数标定使用
"""
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from scipy.stats import linregress

matplotlib.rcParams['font.family']        = 'SimHei'
matplotlib.rcParams['axes.unicode_minus'] = False

# ============================================================
# 参数配置
# ============================================================
CSV_FILE    = "outputs/csv_data/scan_data.csv"   # 修改为你的文件名
PRESS_START = 76                        # 加压开始帧
FORCE_MAX   = 6.0                       # 只分析 6N 以内


def load_and_process(csv_file: str):
    """读取并处理实际数据"""
    df = pd.read_csv(csv_file)

    # ── 截取加压段 ────────────────────────────────────────────
    df = df.iloc[PRESS_START:].copy().reset_index(drop=True)

    # ── 位移：tcp_z 变化量（向下为正）────────────────────────
    tcp_z_init      = df['tcp_z_m'].iloc[0]
    df['disp_mm']   = (tcp_z_init - df['tcp_z_m']) * 1000.0

    # ── 力 ────────────────────────────────────────────────────
    df['force_N']   = df['Fn_z_sensor_N'].abs()

    # ── 截取 6N 以内 ──────────────────────────────────────────
    df = df[df['force_N'] <= FORCE_MAX].copy().reset_index(drop=True)

    # ── 平滑 ──────────────────────────────────────────────────
    win = min(11, len(df) // 2 * 2 - 1)   # 窗口必须为奇数
    win = max(win, 3)
    if len(df) > win:
        df['force_smooth'] = savgol_filter(
            df['force_N'], window_length=win, polyorder=2
        )
        df['disp_smooth'] = savgol_filter(
            df['disp_mm'], window_length=win, polyorder=2
        )
    else:
        df['force_smooth'] = df['force_N']
        df['disp_smooth']  = df['disp_mm']

    # 截断负值
    df['disp_smooth']  = df['disp_smooth'].clip(lower=0)
    df['force_smooth'] = df['force_smooth'].clip(lower=0)

    return df


def analyze_stiffness(df: pd.DataFrame):
    """
    线性回归分析刚度
    返回：slope(N/mm), intercept, r_value, 预测值
    """
    x = df['disp_smooth'].values
    y = df['force_smooth'].values

    # 过滤掉位移为0的初始段（避免干扰线性拟合）
    mask        = x > 0.1
    x_fit, y_fit = x[mask], y[mask]

    if len(x_fit) < 3:
        print("⚠️ 有效数据点太少，无法拟合")
        return None

    slope, intercept, r_value, p_value, std_err = linregress(x_fit, y_fit)

    print("\n" + "=" * 50)
    print("📊 实际数据刚度分析结果（6N以内）")
    print("=" * 50)
    print(f"  刚度 k        = {slope:.4f} N/mm")
    print(f"  截距          = {intercept:.4f} N")
    print(f"  R²            = {r_value**2:.6f}")
    print(f"  标准误差      = {std_err:.6f}")
    print(f"  数据点数      = {len(x_fit)}")
    print(f"  位移范围      = {x_fit.min():.2f} ~ {x_fit.max():.2f} mm")
    print(f"  力范围        = {y_fit.min():.3f} ~ {y_fit.max():.3f} N")
    print("=" * 50)

    # 预测值（用于绘图）
    x_pred = np.linspace(x_fit.min(), x_fit.max(), 200)
    y_pred = slope * x_pred + intercept

    return slope, intercept, r_value, x_pred, y_pred, x_fit, y_fit


def plot_stiffness(df: pd.DataFrame, stiffness_result):
    """绘制刚度分析图"""
    slope, intercept, r_value, x_pred, y_pred, x_fit, y_fit = stiffness_result

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(
        "实际乳腺按压数据刚度分析（≤6N）",
        fontsize=14, fontweight='bold'
    )

    # ── 左图：F-x 散点 + 线性拟合 ────────────────────────────
    ax1 = axes[0]

    ax1.scatter(
        df['disp_smooth'], df['force_smooth'],
        s=15, alpha=0.6,
        color='steelblue',
        label='实测数据点'
    )
    ax1.plot(
        x_pred, y_pred,
        color='tomato', linewidth=2.5, linestyle='--',
        label=f'线性拟合\nk={slope:.4f} N/mm\nR²={r_value**2:.4f}'
    )

    # 标注刚度
    mid_x = (x_pred.min() + x_pred.max()) / 2
    mid_y = slope * mid_x + intercept
    ax1.annotate(
        f'k = {slope:.4f} N/mm',
        xy=(mid_x, mid_y),
        xytext=(mid_x * 0.4, mid_y + 1.0),
        fontsize=11, color='tomato', fontweight='bold',
        arrowprops=dict(arrowstyle='->', color='tomato', lw=1.5)
    )

    ax1.set_xlabel('探头位移 (mm)', fontsize=11)
    ax1.set_ylabel('法向力 (N)',    fontsize=11)
    ax1.set_title('F-x 关系与线性刚度拟合', fontsize=12)
    ax1.legend(fontsize=10, loc='upper left')
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(left=0)
    ax1.set_ylim(bottom=0)

    # 残差阴影
    y_fit_pred = slope * x_fit + intercept
    ax1.fill_between(
        x_fit, y_fit, y_fit_pred,
        alpha=0.15, color='purple',
        label='残差'
    )

    # ── 右图：时间序列（位移和力随帧变化）────────────────────
    ax2      = axes[1]
    ax2_twin = ax2.twinx()

    l1, = ax2.plot(
        df.index, df['disp_smooth'],
        color='steelblue', linewidth=1.8,
        label='位移 (mm)'
    )
    l2, = ax2_twin.plot(
        df.index, df['force_smooth'],
        color='tomato', linewidth=1.8,
        label='法向力 (N)'
    )

    ax2.set_xlabel('采样序号',          fontsize=11)
    ax2.set_ylabel('位移 (mm)',          fontsize=11, color='steelblue')
    ax2_twin.set_ylabel('法向力 (N)',    fontsize=11, color='tomato')
    ax2.set_title('位移 & 力随时间变化', fontsize=12)
    ax2.grid(True, alpha=0.3)

    lines  = [l1, l2]
    labels = [l.get_label() for l in lines]
    ax2.legend(lines, labels, fontsize=9, loc='upper left')

    # 统计信息框
    stats = (
        f"刚度 k  = {slope:.4f} N/mm\n"
        f"截距    = {intercept:.4f} N\n"
        f"R²      = {r_value**2:.6f}\n"
        f"数据点  = {len(x_fit)}"
    )
    ax1.text(
        0.97, 0.05, stats,
        transform=ax1.transAxes,
        fontsize=9,
        verticalalignment='bottom',
        horizontalalignment='right',
        bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9)
    )

    plt.tight_layout()
    save_path = "outputs/figures/stiffness_analysis.png"
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"✓ 分析图已保存 → {save_path}")
    plt.show()


if __name__ == '__main__':
    print("读取实际数据...")
    df = load_and_process(CSV_FILE)
    print(f"有效数据点（≤6N）: {len(df)}")

    result = analyze_stiffness(df)
    if result:
        plot_stiffness(df, result)