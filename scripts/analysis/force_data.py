# explore_waypoints.py
"""
探索每个waypoint的数据特征
找出哪些waypoint对应垂直按压动作
"""
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

matplotlib.rcParams['font.family']        = 'SimHei'
matplotlib.rcParams['axes.unicode_minus'] = False

CSV_FILE = "outputs/csv_data/cmy_sync.csv"

df = pd.read_csv(CSV_FILE)

# ── 按waypoint分组统计 ────────────────────────────────────────
print(f"{'wp':>4} {'帧数':>6} {'tcp_z_min':>10} "
      f"{'tcp_z_max':>10} {'位移mm':>8} "
      f"{'Fz_min':>8} {'Fz_max':>8} {'Fz_mean':>8}")
print("-" * 75)

results = []
for wp, group in df.groupby('waypoint_idx'):
    tcp_z_min  = group['tcp_z_m'].min()
    tcp_z_max  = group['tcp_z_m'].max()
    disp_mm    = (tcp_z_max - tcp_z_min) * 1000
    fz_min     = group['Fz_N'].min()
    fz_max     = group['Fz_N'].max()
    fz_mean    = group['Fz_N'].mean()
    n          = len(group)

    results.append({
        'wp':       wp,
        'n':        n,
        'tcp_z_min': tcp_z_min,
        'tcp_z_max': tcp_z_max,
        'disp_mm':  disp_mm,
        'fz_min':   fz_min,
        'fz_max':   fz_max,
        'fz_mean':  fz_mean
    })

    print(f"{wp:>4} {n:>6} {tcp_z_min:>10.4f} "
          f"{tcp_z_max:>10.4f} {disp_mm:>8.1f} "
          f"{fz_min:>8.3f} {fz_max:>8.3f} {fz_mean:>8.3f}")

df_res = pd.DataFrame(results)

# ── 自动筛选按压段候选 ────────────────────────────────────────
# 条件：位移>5mm 且 最大力>1N
candidates = df_res[
    (df_res['disp_mm'] > 5.0) &
    (df_res['fz_max']  > 1.0)
].copy()

print(f"\n按压段候选 waypoint（位移>5mm 且 最大力>1N）：")
print(candidates[['wp', 'n', 'disp_mm',
                   'fz_min', 'fz_max']].to_string(index=False))

# ── 画出候选waypoint的F-x图 ──────────────────────────────────
if len(candidates) > 0:
    n_plots = min(len(candidates), 9)  # 最多画9个
    cols    = 3
    rows    = (n_plots + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols,
                             figsize=(15, 4 * rows))
    axes = np.array(axes).flatten()

    for i, (_, row) in enumerate(
            candidates.head(n_plots).iterrows()):
        wp    = int(row['wp'])
        group = df[df['waypoint_idx'] == wp].copy()
        group = group.reset_index(drop=True)

        # 计算位移和力
        tcp_z_ref      = group['tcp_z_m'].iloc[0]
        group['disp']  = (tcp_z_ref - group['tcp_z_m']) * 1000
        group['force'] = group['Fz_N']

        ax = axes[i]
        ax2 = ax.twinx()

        ax.plot(group.index,  group['disp'],
                color='steelblue', lw=1.5, label='位移')
        ax2.plot(group.index, group['force'],
                 color='tomato',    lw=1.5, label='力')

        ax.set_title(f"Waypoint {wp} "
                     f"(位移{row['disp_mm']:.1f}mm "
                     f"力{row['fz_max']:.1f}N)",
                     fontsize=9)
        ax.set_xlabel('帧序号', fontsize=8)
        ax.set_ylabel('位移(mm)', fontsize=8, color='steelblue')
        ax2.set_ylabel('力(N)',   fontsize=8, color='tomato')
        ax.grid(True, alpha=0.3)

    # 隐藏多余子图
    for j in range(n_plots, len(axes)):
        axes[j].set_visible(False)

    plt.suptitle("候选按压段 Waypoint 预览",
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig("outputs/figures/waypoint_candidates.png",
                dpi=120, bbox_inches='tight')
    print(f"\n✓ 图已保存 → outputs/figures/waypoint_candidates.png")
    plt.show()