# analyze_waypoints.py
"""
从waypoint数据中提取F-x关系
用相对变化量分析刚度
"""
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from scipy.stats  import linregress

matplotlib.rcParams['font.family']        = 'SimHei'
matplotlib.rcParams['axes.unicode_minus'] = False

CSV_FILE = "outputs/csv_data/cmy_sync.csv"

# 扩大候选范围，筛选所有位移>3mm的wp
BEST_WPS = [20, 37, 38, 39, 50, 51, 52, 53,
            54, 64, 65, 66, 67, 78, 79, 80,
            81, 82, 83]


def analyze_single_wp(df_wp: pd.DataFrame, wp_id: int):
    """
    分析单个waypoint的F-x关系
    使用相对变化量（去掉初始偏置）
    """
    df = df_wp.copy().reset_index(drop=True)

    # ── 位移：相对变化（向下为正）────────────────────────────
    tcp_z_ref    = df['tcp_z_m'].iloc[0]
    df['disp_mm'] = (tcp_z_ref - df['tcp_z_m']) * 1000.0

    # ── 力：去掉初始偏置 ──────────────────────────────────────
    fz_ref       = df['Fz_N'].iloc[0]
    df['force_N'] = df['Fz_N'] - fz_ref

    # ── 只取加压段（力>0 且 位移>0）──────────────────────────
    df = df[
        (df['disp_mm'] >= 0) &
        (df['force_N']  >= -0.5)   # 允许小幅负值（噪声）
    ].copy().reset_index(drop=True)

    # ── 找力的峰值，只取上升段 ────────────────────────────────
    if len(df) > 0:
        peak_idx = df['force_N'].idxmax()
        df = df.iloc[:peak_idx + 1].copy().reset_index(drop=True)

    # ── 平滑 ──────────────────────────────────────────────────
    n = len(df)
    if n > 7:
        win = min(7, (n // 2) * 2 - 1)
        win = max(win, 3)
        df['force_s'] = savgol_filter(df['force_N'], win, polyorder=2)
        df['disp_s']  = savgol_filter(df['disp_mm'], win, polyorder=2)
    else:
        df['force_s'] = df['force_N']
        df['disp_s']  = df['disp_mm']

    df['force_s'] = df['force_s'].clip(lower=-0.1)
    df['disp_s']  = df['disp_s'].clip(lower=0)

    return df


def main():
    df_all = pd.read_csv(CSV_FILE)

    # ── 分析每个候选waypoint ──────────────────────────────────
    results   = []
    all_disp  = []
    all_force = []

    for wp in BEST_WPS:
        df_wp = df_all[df_all['waypoint_idx'] == wp].copy()
        df_an = analyze_single_wp(df_wp, wp)

        if len(df_an) < 5:
            print(f"wp{wp}: 有效点太少，跳过")
            continue

        x = df_an['disp_s'].values
        y = df_an['force_s'].values

        # 刚度分析
        mask = (x > 0.5) & (y > -0.3)
        if mask.sum() > 3:
            slope, intercept, r, _, _ = linregress(
                x[mask], y[mask]
            )
            results.append({
                'wp':        wp,
                'k':         slope,
                'intercept': intercept,
                'r2':        r**2,
                'n':         mask.sum(),
                'disp_max':  x.max(),
                'force_max': y.max()
            })
            print(f"wp{wp:>3}: k={slope:.4f}N/mm  "
                  f"R^2={r**2:.3f}  "
                  f"位移={x.max():.1f}mm  "
                  f"力变化={y.max():.2f}N")

            # 收集所有数据用于综合拟合
            all_disp.extend(x[mask].tolist())
            all_force.extend(y[mask].tolist())

    # ── 综合刚度拟合 ──────────────────────────────────────────
    if len(all_disp) > 10:
        all_disp  = np.array(all_disp)
        all_force = np.array(all_force)
        slope_all, intercept_all, r_all, _, _ = linregress(
            all_disp, all_force
        )
        print(f"\n{'='*50}")
        print(f"综合刚度（所有候选wp合并）：")
        print(f"  k      = {slope_all:.4f} N/mm")
        print(f"  截距   = {intercept_all:.4f} N")
        print(f"  R^2    = {r_all**2:.4f}")
        print(f"  数据点 = {len(all_disp)}")
        print(f"{'='*50}")
    else:
        slope_all = intercept_all = r_all = 0

    # ── 绘图 ──────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    fig.suptitle(
        "人体乳腺按压 F-x 关系分析（相对变化量）",
        fontsize=14, fontweight='bold'
    )
    axes = axes.flatten()

    # 各waypoint子图
    for i, wp in enumerate(BEST_WPS[:5]):
        ax = axes[i]
        df_wp = df_all[df_all['waypoint_idx'] == wp].copy()
        df_an = analyze_single_wp(df_wp, wp)

        if len(df_an) < 3:
            ax.set_visible(False)
            continue

        x = df_an['disp_s'].values
        y = df_an['force_s'].values

        ax.scatter(x, y, s=15, alpha=0.6,
                   color='steelblue', label='数据点')

        # 拟合线
        mask = (x > 0.5) & (y > -0.3)
        if mask.sum() > 3:
            s, b, r, _, _ = linregress(x[mask], y[mask])
            x_fit = np.linspace(x.min(), x.max(), 100)
            y_fit = s * x_fit + b
            ax.plot(x_fit, y_fit,
                    color='tomato', lw=2, ls='--',
                    label=f'k={s:.4f}N/mm\nR^2={r**2:.3f}')

        ax.axhline(y=0, color='gray', ls=':', lw=1)
        ax.set_xlabel('位移变化量 (mm)', fontsize=9)
        ax.set_ylabel('力变化量 (N)',    fontsize=9)
        ax.set_title(f'Waypoint {wp}',   fontsize=11)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    # 综合图
    ax = axes[5]
    if len(all_disp) > 0:
        ax.scatter(all_disp, all_force,
                   s=10, alpha=0.4, color='steelblue',
                   label='所有候选wp数据')

        if slope_all > 0:
            x_fit = np.linspace(
                all_disp.min(), all_disp.max(), 200
            )
            y_fit = slope_all * x_fit + intercept_all
            ax.plot(x_fit, y_fit,
                    color='tomato', lw=2.5, ls='--',
                    label=f'综合拟合\n'
                          f'k={slope_all:.4f}N/mm\n'
                          f'R^2={r_all**2:.4f}')

        ax.axhline(y=0, color='gray', ls=':', lw=1)
        ax.set_xlabel('位移变化量 (mm)', fontsize=10)
        ax.set_ylabel('力变化量 (N)',    fontsize=10)
        ax.set_title('综合 F-x 关系',   fontsize=11)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("outputs/figures/waypoint_fx_analysis.png",
                dpi=150, bbox_inches='tight')
    print(f"\n✓ 图已保存 → outputs/figures/waypoint_fx_analysis.png")
    plt.show()

    return slope_all, r_all**2


if __name__ == '__main__':
    main()