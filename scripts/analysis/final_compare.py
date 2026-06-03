# final_compare.py
"""
最终对比图：人体实际采集数据(wp65) vs 仿真输出数据
"""
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.signal  import savgol_filter
from scipy.stats   import linregress

matplotlib.rcParams['font.family']        = 'SimHei'
matplotlib.rcParams['axes.unicode_minus'] = False
matplotlib.rcParams['mathtext.fontset']   = 'dejavusans'

# ============================================================
# 配置
# ============================================================
CSV_FILE  = "outputs/csv_data/cmy_sync.csv"
SIM_FILE  = "archive/fx_data/fx_data_20260408_224358.txt"  # ← 改为你最新的txt文件名
TARGET_WP = 65        # 使用wp65作为实际数据代表


# ============================================================
# 1. 加载实际数据（wp65）
# ============================================================
def load_real_wp65(csv_file: str, wp: int = 65):

    df_all = pd.read_csv(csv_file)
    df_wp  = df_all[
        df_all['waypoint_idx'] == wp
    ].copy().reset_index(drop=True)

    # ── 位移：相对变化（向下为正）────────────────────────────
    tcp_z_ref     = df_wp['tcp_z_m'].iloc[0]
    df_wp['disp_mm'] = (tcp_z_ref - df_wp['tcp_z_m']) * 1000.0

    # ── 力：去掉初始偏置 ──────────────────────────────────────
    fz_ref           = df_wp['Fz_N'].iloc[0]
    df_wp['force_N'] = df_wp['Fz_N'] - fz_ref

    # ── 只取加压段（到力峰值为止）────────────────────────────
    peak_idx = df_wp['force_N'].idxmax()
    df_wp    = df_wp.iloc[
        :peak_idx + 1
    ].copy().reset_index(drop=True)

    # ── 去掉位移为负的点 ──────────────────────────────────────
    df_wp = df_wp[
        df_wp['disp_mm'] >= 0
    ].copy().reset_index(drop=True)

    # ── 平滑 ──────────────────────────────────────────────────
    n = len(df_wp)
    if n > 7:
        win = min(7, (n // 2) * 2 - 1)
        win = max(win, 3)
        df_wp['force_s'] = savgol_filter(
            df_wp['force_N'], win, polyorder=2
        )
        df_wp['disp_s'] = savgol_filter(
            df_wp['disp_mm'], win, polyorder=2
        )
    else:
        df_wp['force_s'] = df_wp['force_N']
        df_wp['disp_s']  = df_wp['disp_mm']

    df_wp['force_s'] = df_wp['force_s'].clip(lower=0)
    df_wp['disp_s']  = df_wp['disp_s'].clip(lower=0)

    print(f"[实际数据 wp{wp}] 有效点: {len(df_wp)}")
    print(f"[实际数据 wp{wp}] 位移: "
          f"{df_wp['disp_s'].min():.2f} ~ "
          f"{df_wp['disp_s'].max():.2f} mm")
    print(f"[实际数据 wp{wp}] 力变化: "
          f"{df_wp['force_s'].min():.3f} ~ "
          f"{df_wp['force_s'].max():.3f} N")

    return df_wp


# ============================================================
# 2. 加载仿真数据
# ============================================================
def load_sim_data(sim_file: str):

    data = np.loadtxt(sim_file, delimiter='\t', skiprows=1)
    if data.ndim == 1:
        data = data.reshape(1, -1)

    disp_mm = data[:, 0] * 1000.0
    force_N = data[:, 1]

    # ── 去掉位移为负的点 ──────────────────────────────────────
    mask    = disp_mm >= 0
    disp_mm = disp_mm[mask]
    force_N = force_N[mask]

    # ── 去掉异常跳变（相邻点位移差>3mm）──────────────────────
    disp_diff = np.abs(np.diff(disp_mm, prepend=disp_mm[0]))
    mask      = disp_diff <= 3.0
    disp_mm   = disp_mm[mask]
    force_N   = force_N[mask]

    # ── 按位移分bin取均值（消除来回抖动）────────────────────
    n_bins  = 50
    bins    = np.linspace(disp_mm.min(), disp_mm.max(), n_bins + 1)
    d_bin, f_bin = [], []
    for i in range(n_bins):
        m = (disp_mm >= bins[i]) & (disp_mm < bins[i + 1])
        if m.sum() >= 1:
            d_bin.append(disp_mm[m].mean())
            f_bin.append(force_N[m].mean())

    d_bin = np.array(d_bin)
    f_bin = np.array(f_bin)

    # ── 按位移排序 ────────────────────────────────────────────
    idx   = np.argsort(d_bin)
    d_bin = d_bin[idx]
    f_bin = f_bin[idx]

    # ── 平滑 ──────────────────────────────────────────────────
    n = len(d_bin)
    if n > 7:
        win   = min(7, (n // 2) * 2 - 1)
        win   = max(win, 3)
        f_bin = savgol_filter(f_bin, win, polyorder=2)
        f_bin = np.clip(f_bin, 0, None)

    print(f"\n[仿真数据] 有效点: {len(d_bin)}")
    print(f"[仿真数据] 位移: "
          f"{d_bin.min():.2f} ~ {d_bin.max():.2f} mm")
    print(f"[仿真数据] 力:   "
          f"{f_bin.min():.3f} ~ {f_bin.max():.3f} N")

    return d_bin, f_bin


# ============================================================
# 3. 刚度分析
# ============================================================
def calc_stiffness(disp, force, label):

    mask = (disp > 0.3) & (force > 0.05)
    x, y = disp[mask], force[mask]

    if len(x) < 3:
        print(f"⚠️ [{label}] 有效点太少")
        return None

    slope, intercept, r, _, _ = linregress(x, y)
    x_fit = np.linspace(x.min(), x.max(), 200)
    y_fit = slope * x_fit + intercept

    print(f"\n[{label}] 刚度分析：")
    print(f"  k    = {slope:.4f} N/mm")
    print(f"  截距 = {intercept:.4f} N")
    print(f"  R^2  = {r**2:.4f}")
    print(f"  点数 = {len(x)}")

    return {
        'slope': slope, 'intercept': intercept,
        'r2': r**2,
        'x_fit': x_fit, 'y_fit': y_fit,
        'x_data': x,    'y_data': y
    }


# ============================================================
# 4. 绘图
# ============================================================
def plot_final(df_real, sim_data, res_real, res_sim):

    d_sim  = sim_data[0]
    f_sim  = sim_data[1]
    d_real = df_real['disp_s'].values
    f_real = df_real['force_s'].values

    # ── 归一化 ────────────────────────────────────────────────
    def norm(arr):
        mn, mx = arr.min(), arr.max()
        return (arr - mn) / (mx - mn) if mx > mn else np.zeros_like(arr)

    dr_n = norm(d_real);  fr_n = norm(f_real)
    ds_n = norm(d_sim);   fs_n = norm(f_sim)

    # ── 插值对齐 ──────────────────────────────────────────────
    from scipy.interpolate import interp1d
    d_min  = max(dr_n.min(), ds_n.min())
    d_max  = min(dr_n.max(), ds_n.max())
    mask_r = (dr_n >= d_min) & (dr_n <= d_max)
    mask_s = (ds_n >= d_min) & (ds_n <= d_max)

    if mask_s.sum() > 2:
        fi          = interp1d(
            ds_n[mask_s], fs_n[mask_s],
            kind='linear',
            bounds_error=False, fill_value='extrapolate'
        )
        sim_aligned = fi(dr_n[mask_r])
        error       = fr_n[mask_r] - sim_aligned
        disp_err    = dr_n[mask_r]
        corr        = np.corrcoef(
            fr_n[mask_r], sim_aligned
        )[0, 1]
    else:
        sim_aligned = np.zeros_like(dr_n)
        error = disp_err = np.zeros_like(dr_n)
        corr  = 0.0

    mae  = np.abs(error).mean()
    rmse = np.sqrt((error**2).mean())

    # ── 画布 ──────────────────────────────────────────────────
    fig = plt.figure(figsize=(16, 11))
    fig.suptitle(
        "乳腺按压力-位移(F-x)关系验证：人体实测 vs 仿真输出",
        fontsize=15, fontweight='bold', y=0.99
    )
    gs  = gridspec.GridSpec(
        2, 2, figure=fig, hspace=0.42, wspace=0.38
    )
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[1, 0])
    ax4 = fig.add_subplot(gs[1, 1])

    # ── 左上：归一化叠加在同一轴（直观对比趋势）─────────────
    ax1.scatter(
        dr_n, fr_n,
        s=25, alpha=0.7, color='steelblue',
        zorder=3, label='人体实测数据点'
    )
    ax1.plot(
        ds_n, fs_n,
        color='tomato', lw=2.5, ls='--',
        zorder=2, label='仿真输出曲线'
    )

    # 实测线性拟合
    if res_real:
        xn = norm(res_real['x_data'])
        yn = norm(res_real['y_data'])
        xn_fit = np.linspace(0, 1, 100)
        # 用归一化空间的斜率近似
        s_n, b_n, _, _, _ = linregress(xn, yn)
        ax1.plot(
            xn_fit, s_n * xn_fit + b_n,
            color='steelblue', lw=1.5,
            ls=':', alpha=0.8,
            label='实测线性趋势'
        )

    ax1.set_xlabel('归一化位移',           fontsize=11)
    ax1.set_ylabel('归一化力',             fontsize=11)
    ax1.set_title('F-x 趋势对比（归一化）', fontsize=12)
    ax1.legend(fontsize=9, loc='upper left')
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(-0.02, 1.05)
    ax1.set_ylim(-0.05, 1.15)

    # 相关系数标注
    # 改为右下角往上一点：
    ax1.text(
        0.97, 0.03,
        f'Pearson r = {corr:.4f}\nMAE = {mae:.4f}',
        transform=ax1.transAxes,
        fontsize=10, ha='right',
        fontweight='bold', color='purple',
        bbox=dict(boxstyle='round',
                  facecolor='lavender', alpha=0.85)
    )
    # ✅ 新增：说明非线性差异的来源
    # 替换为右上角，不遮挡数据点：
    ax1.text(
        0.97, 0.55,
        "实测：轻微非线性\n（软组织特性）\n仿真：线性弹性模型",
        transform=ax1.transAxes,
        fontsize=8, ha='right', color='steelblue',
        bbox=dict(boxstyle='round',
                  facecolor='white', alpha=0.8)
    )

    # ── 右上：原始值对比（双Y轴）─────────────────────────────
    ax2r = ax2.twinx()
    l1,  = ax2.plot(
        d_real, f_real,
        color='steelblue', lw=2,
        label='人体实测（左轴）'
    )
    ax2.scatter(
        d_real, f_real,
        s=20, color='steelblue', alpha=0.6, zorder=3
    )
    l2, = ax2r.plot(
        d_sim, f_sim,
        color='tomato', lw=2, ls='--',
        label='仿真输出（右轴）'
    )
    ax2.set_xlabel('位移 (mm)',              fontsize=11)
    ax2.set_ylabel('实测力变化量 (N)',        fontsize=11,
                   color='steelblue')
    ax2r.set_ylabel('仿真力 (N)',             fontsize=11,
                    color='tomato')
    ax2.set_title('原始 F-x 曲线（双Y轴）',  fontsize=12)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(left=0)
    ax2.set_ylim(bottom=0)
    ax2r.set_ylim(bottom=0)
    ax2.set_xlim(0, 6)  # 只看0~6mm
    ax2r.set_xlim(0, 6)
    ax2.legend(
        [l1, l2],
        [l.get_label() for l in [l1, l2]],
        fontsize=9, loc='upper left'
    )

    # 标注两组数据范围
    ax2.text(
        0.97, 0.35,
        f"实测范围:\n"
        f"  位移 0~{d_real.max():.1f}mm\n"
        f"  力变化 0~{f_real.max():.2f}N",
        transform=ax2.transAxes,
        fontsize=8, ha='right', color='steelblue',
        bbox=dict(boxstyle='round',
                  facecolor='white', alpha=0.8)
    )

    # ── 左下：刚度拟合对比（核心图）─────────────────────────
    ax3.scatter(
        res_real['x_data'], res_real['y_data'],
        s=30, alpha=0.6, color='steelblue',
        zorder=3, label='人体实测数据点'
    )
    ax3.scatter(
        res_sim['x_data'], res_sim['y_data'],
        s=15, alpha=0.4, color='tomato',
        zorder=2, label='仿真数据点'
    )
    ax3.plot(
        res_real['x_fit'], res_real['y_fit'],
        color='steelblue', lw=2.5,
        label=(f"实测线性拟合\n"
               f"k = {res_real['slope']:.4f} N/mm\n"
               f"R^2 = {res_real['r2']:.3f}")
    )
    ax3.plot(
        res_sim['x_fit'], res_sim['y_fit'],
        color='tomato', lw=2.5, ls='--',
        label=(f"仿真线性拟合\n"
               f"k = {res_sim['slope']:.4f} N/mm\n"
               f"R^2 = {res_sim['r2']:.3f}")
    )
    ax3.set_xlabel('位移 (mm)',        fontsize=11)
    ax3.set_ylabel('力 (N)',           fontsize=11)
    ax3.set_title('线性刚度拟合对比',  fontsize=12)
    ax3.grid(True, alpha=0.3)
    ax3.legend(fontsize=8, loc='upper left')
    ax3.set_xlim(left=0)
    ax3.set_ylim(bottom=0)

    # 刚度误差醒目标注
    ratio = (res_real['slope'] / res_sim['slope']
             if res_sim['slope'] > 1e-6 else 0)
    err_pct = abs(1 - ratio) * 100
    color_err = 'green' if err_pct < 10 else 'orange'
    ax3.text(
        0.97, 0.05,
        f"实测 k = {res_real['slope']:.4f} N/mm\n"
        f"仿真 k = {res_sim['slope']:.4f} N/mm\n"
        f"刚度误差 = {err_pct:.1f}%",
        transform=ax3.transAxes,
        fontsize=10, ha='right', fontweight='bold',
        color=color_err,
        bbox=dict(boxstyle='round',
                  facecolor='lightyellow', alpha=0.95)
    )

    # ── 右下：残差分析 ────────────────────────────────────────
    # 用实测数据对线性拟合的残差
    if res_real:
        x_r   = res_real['x_data']
        y_r   = res_real['y_data']
        y_fit = (res_real['slope'] * x_r
                 + res_real['intercept'])
        resid = y_r - y_fit

        ax4.scatter(
            x_r, resid,
            s=25, alpha=0.7, color='steelblue',
            label='实测残差'
        )
        ax4.axhline(
            y=0, color='tomato',
            ls='--', lw=1.5, label='零线'
        )
        ax4.axhline(
            y=resid.std(), color='gray',
            ls=':', lw=1, alpha=0.7
        )
        ax4.axhline(
            y=-resid.std(), color='gray',
            ls=':', lw=1, alpha=0.7,
            label=f'±1σ = ±{resid.std():.4f}N'
        )

        # 填充±1σ区间
        ax4.fill_between(
            [x_r.min(), x_r.max()],
            -resid.std(), resid.std(),
            alpha=0.1, color='gray'
        )

        ax4.set_xlabel('位移 (mm)',            fontsize=11)
        ax4.set_ylabel('残差 (N)',             fontsize=11)
        ax4.set_title('实测数据线性拟合残差分析', fontsize=12)
        ax4.legend(fontsize=9)
        ax4.grid(True, alpha=0.3)
        ax4.set_xlim(left=0)

        # 残差统计
        ax4.text(
            0.97, 0.92,
            f"残差均值: {resid.mean():.4f} N\n"
            f"残差标准差: {resid.std():.4f} N\n"
            f"R^2 = {res_real['r2']:.4f}",
            transform=ax4.transAxes,
            fontsize=9, ha='right',
            bbox=dict(boxstyle='round',
                      facecolor='lightyellow', alpha=0.9)
        )

    plt.tight_layout()
    save_path = "outputs/figures/fx_final_comparison.png"
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\n✓ 最终对比图已保存 → {save_path}")
    plt.show()


# ============================================================
# 主程序
# ============================================================
if __name__ == '__main__':

    print("=" * 50)
    print("步骤1：加载人体实测数据（wp65）")
    print("=" * 50)
    df_real = load_real_wp65(CSV_FILE, TARGET_WP)

    print("\n" + "=" * 50)
    print("步骤2：加载仿真数据")
    print("=" * 50)
    sim_data = load_sim_data(SIM_FILE)

    print("\n" + "=" * 50)
    print("步骤3：刚度分析")
    print("=" * 50)
    res_real = calc_stiffness(
        df_real['disp_s'].values,
        df_real['force_s'].values,
        '人体实测'
    )
    res_sim = calc_stiffness(
        sim_data[0], sim_data[1], '仿真'
    )

    if res_real and res_sim:
        print(f"\n{'='*50}")
        print(f"刚度对比结论：")
        print(f"  人体实测 k = {res_real['slope']:.4f} N/mm")
        print(f"  仿真输出 k = {res_sim['slope']:.4f} N/mm")
        ratio = res_real['slope'] / res_sim['slope']
        print(f"  误差       = {abs(1-ratio)*100:.1f}%")
        print(f"{'='*50}")

    print("\n" + "=" * 50)
    print("步骤4：生成最终对比图")
    print("=" * 50)
    if res_real and res_sim:
        plot_final(df_real, sim_data, res_real, res_sim)