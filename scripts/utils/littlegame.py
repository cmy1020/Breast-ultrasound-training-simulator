import os
import numpy as np
import trimesh
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
from matplotlib.patches import PathPatch
from matplotlib.path import Path
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


class UltrasoundConfig:
    """
    ========== 超声配置参数说明 ==========
    所有距离单位统一为 米(m)
    """
    # 扇形扫查区域参数
    SECTOR_ANGLE = 60  # 扇形角度（度）- 推荐45~90，越大视野越宽
    SECTOR_DEPTH = 0.15  # 扫查深度（米）- 推荐0.08~0.25，越大看得越深
    SECTOR_ORIGIN_Y_OFFSET = 0  # 探头Y位置偏移 - 一般保持0

    # ========== 圆弧顶部参数 ==========
    SECTOR_TOP_RADIUS = 0.008  # 顶部圆弧半径（米）- 推荐0.005~0.015，越大越圆润
    SECTOR_TOP_OFFSET = 0.005  # 顶部起始距离（米）- 推荐0.003~0.01，扇形离探头的距离

    # 纹理分辨率
    
    TEX_W = 500  # 纹理宽度 - 推荐300~800，越大越细腻但越慢
    TEX_H = 500  # 纹理高度 - 同上

    # ========== 背景与组织对比度 ==========
    BACKGROUND_BRIGHTNESS = 0.12  # 背景亮度 - 推荐0.05~0.25，越小背景越黑
    TISSUE_BASE_BRIGHTNESS = 0.55  # 组织亮度 - 推荐0.4~0.75，越大组织越亮

    # 深度衰减
    ATTENUATION = 1.8  # 衰减系数 - 推荐1.2~3.0，越大深处越暗
    BRIGHTNESS_TOP = 0.99  # 顶部最大亮度 - 一般保持0.99
    BRIGHTNESS_FLOOR = 0.01  # 底部最小亮度 - 一般保持0.01

    # ========== 横向扫描线参数 ==========
    SCAN_LINE_BASE_COUNT = 50  # 扫描线数量 - 推荐30~100，越多越密集
    SCAN_LINE_DENSITY_VAR = 0.6  # 密度变化 - 推荐0.3~1.0，越大底部越稀疏

    SCAN_LINE_BRIGHTNESS = 0.32  # 线条亮度 - 推荐0.2~0.5，越大越明显
    SCAN_LINE_BRIGHTNESS_JITTER = 0.3  # 亮度抖动 - 推荐0.1~0.4，越大明暗变化越大

    SCAN_LINE_POSITION_JITTER = 1.0  # 位置抖动 - 推荐0.5~3.0，越大越不规则
    SCAN_LINE_THICKNESS_JITTER = 0.7  # 粗细抖动 - 推荐0.3~1.5，越大粗细变化越大
    SCAN_LINE_BASE_THICKNESS = 1.6  # 基础粗细 - 推荐1.0~3.0，越大线条越粗

    SCAN_LINE_WAVE_AMP = 2.0  # 波动幅度 - 推荐0.5~4.0，越大线条越弯曲
    SCAN_LINE_WAVE_FREQ = 2.8  # 波动频率 - 推荐1.0~5.0，越大波纹越密集

    # ========== 扫描线断续效果 ==========
    SCAN_LINE_BREAK_PROBABILITY = 0.25  # 断裂概率 - 推荐0.0~0.5，越大虚线越多（0=全实线）
    SCAN_LINE_BREAK_MIN_LENGTH = 0.05  # 最小线段长度 - 推荐0.03~0.15，越小线段越短
    SCAN_LINE_BREAK_MAX_LENGTH = 0.35  # 最大线段长度 - 推荐0.2~0.6，越大可能的最长段越长
    SCAN_LINE_BREAK_GAP_MIN = 0.02  # 最小间隙 - 推荐0.01~0.05，越小间隙越不明显
    SCAN_LINE_BREAK_GAP_MAX = 0.15  # 最大间隙 - 推荐0.08~0.3，越大断点越明显

    # ========== 散斑噪声 ==========
    SPECKLE_STRENGTH = 0.5  # 散斑强度 - 推荐0.3~0.8，越大颗粒感越强
    SPECKLE_SMOOTH_RADIUS = 1  # 散斑平滑 - 推荐0~3，越大越细腻（0=粗糙）

    # ========== 组织底纹 ==========
    TISSUE_TEXTURE = 0.15  # 纹理强度 - 推荐0.05~0.3，越大组织明暗对比越强
    TISSUE_SMOOTH_RADIUS = 5  # 纹理尺寸 - 推荐3~8，越大纹理斑块越大

    # ========== 轮廓高回声 ==========
    CONTOUR_GAIN = 0.60  # 边界增益 - 推荐0.3~0.8，越大边界越亮
    CONTOUR_THICKNESS_PX = 2  # 边界厚度 - 推荐1~5，越大边界线越粗

    # ========== 图像后处理 ==========
    GAMMA = 1.25  # Gamma校正 - 推荐0.8~1.5，<1变亮，>1变暗

    # 扇形边界线
    SECTOR_EDGE_COLOR = '#666666'  # 边界颜色 - HTML颜色代码
    SECTOR_EDGE_ALPHA = 0.45  # 边界透明度 - 推荐0.3~0.7，越大越明显

def mean_blur_2d(img: np.ndarray, radius: int) -> np.ndarray:
    """纯 numpy 均值模糊"""
    if radius <= 0:
        return img
    k = 2 * radius + 1
    pad = radius
    a = np.pad(img, ((pad, pad), (pad, pad)), mode='reflect').astype(np.float32)

    S = np.zeros((a.shape[0] + 1, a.shape[1] + 1), dtype=np.float32)
    S[1:, 1:] = a.cumsum(0).cumsum(1)

    H, W = img.shape
    y0, x0 = 0, 0
    y1, x1 = H, W

    out = (S[y0 + k:y1 + k, x0 + k:x1 + k]
           - S[y0:y1, x0 + k:x1 + k]
           - S[y0 + k:y1 + k, x0:x1]
           + S[y0:y1, x0:x1]) / (k * k)
    return out


def dilate_binary(mask: np.ndarray, radius: int) -> np.ndarray:
    """简单二值膨胀"""
    if radius <= 0:
        return mask
    out = mask.copy()
    for _ in range(radius):
        m = out
        out = (
                m |
                np.roll(m, 1, 0) | np.roll(m, -1, 0) |
                np.roll(m, 1, 1) | np.roll(m, -1, 1) |
                np.roll(np.roll(m, 1, 0), 1, 1) |
                np.roll(np.roll(m, 1, 0), -1, 1) |
                np.roll(np.roll(m, -1, 0), 1, 1) |
                np.roll(np.roll(m, -1, 0), -1, 1)
        )
    return out


class DualViewSliceViewer:
    def __init__(self, stl_file):
        if not os.path.exists(stl_file):
            raise FileNotFoundError(f"找不到文件: {stl_file}")

        print(f"正在加载模型: {stl_file}")
        self.mesh = trimesh.load(stl_file)

        self.mesh.vertices[:, 2] = -self.mesh.vertices[:, 2]  # Z坐标取反
        self.mesh.fix_normals()  # 修复法线方向

        print(f"✓ 加载成功！模型包含 {len(self.mesh.vertices)} 个顶点, {len(self.mesh.faces)} 个面")

        bounds = self.mesh.bounds
        self.z_min = bounds[0][2]
        self.z_max = bounds[1][2]
        self.current_z = (self.z_min + self.z_max) / 2

        # 2D显示范围
        x_padding = (bounds[1][0] - bounds[0][0]) * 0.1
        y_padding = (bounds[1][1] - bounds[0][1]) * 0.1
        self.x_range = [bounds[0][0] - x_padding, bounds[1][0] + x_padding]
        self.y_range = [bounds[0][1] - y_padding, bounds[1][1] + y_padding]

        # 探头位置（顶部）
        self.model_center_x = (bounds[0][0] + bounds[1][0]) / 2
        self.probe_position_x = self.model_center_x
        self.probe_position_y = self.y_range[1] - UltrasoundConfig.SECTOR_ORIGIN_Y_OFFSET

        self.bounds_3d = bounds
        self.current_slice_vertices = None

        self._tex_cache = None

    def get_cross_section_ordered(self, z_position):
        """获取切片轮廓"""
        try:
            s = self.mesh.section(plane_origin=[0, 0, z_position], plane_normal=[0, 0, 1])
            if s is None:
                return None
            if hasattr(s, "discrete"):
                paths = s.discrete
                if paths:
                    arrs = []
                    for p in paths:
                        a = np.asarray(p)
                        if a.ndim == 2 and a.shape[1] >= 2:
                            arrs.append(a[:, :2])
                    if arrs:
                        return max(arrs, key=len)
            return None
        except Exception:
            return None

    def _create_sector_mask_path(self):
        """创建圆弧梯形扇形路径"""
        angle_half = UltrasoundConfig.SECTOR_ANGLE / 2
        top_radius = UltrasoundConfig.SECTOR_TOP_RADIUS
        top_offset = UltrasoundConfig.SECTOR_TOP_OFFSET
        depth = UltrasoundConfig.SECTOR_DEPTH

        verts = []

        # 顶部圆心位置
        arc_center_x = self.probe_position_x
        arc_center_y = self.probe_position_y - top_offset

        # 1. 顶部圆弧
        angles_top = np.linspace(-angle_half, angle_half, 30)
        for angle_offset in angles_top:
            angle_rad = np.radians(270 + angle_offset)
            x = arc_center_x + top_radius * np.cos(angle_rad)
            y = arc_center_y + top_radius * np.sin(angle_rad)
            verts.append((x, y))

        # 2. 右侧直线
        angle_right = np.radians(270 + angle_half)
        x_right = self.probe_position_x + depth * np.cos(angle_right)
        y_right = self.probe_position_y + depth * np.sin(angle_right)
        verts.append((x_right, y_right))

        # 3. 底部圆弧
        angles_bottom = np.linspace(angle_half, -angle_half, 60)
        for angle_offset in angles_bottom:
            angle_rad = np.radians(270 + angle_offset)
            x = self.probe_position_x + depth * np.cos(angle_rad)
            y = self.probe_position_y + depth * np.sin(angle_rad)
            verts.append((x, y))

        # 4. 闭合
        verts.append(verts[0])

        codes = [Path.MOVETO] + [Path.LINETO] * (len(verts) - 2) + [Path.CLOSEPOLY]

        return Path(verts, codes)

    def _prepare_texture_grid(self):
        """准备纹理坐标网格"""
        if self._tex_cache is not None:
            return self._tex_cache

        W, H = UltrasoundConfig.TEX_W, UltrasoundConfig.TEX_H
        xmin, xmax = self.x_range
        ymin, ymax = self.y_range

        xs = np.linspace(xmin, xmax, W)
        ys = np.linspace(ymin, ymax, H)
        X, Y = np.meshgrid(xs, ys)

        dx = X - self.probe_position_x
        dy = self.probe_position_y - Y
        r = np.sqrt(dx * dx + dy * dy)
        theta = np.degrees(np.arctan2(dx, dy))

        # ========== 几何蒙版计算（单位：米）==========
        angle_half = UltrasoundConfig.SECTOR_ANGLE / 2
        depth = UltrasoundConfig.SECTOR_DEPTH  # 已经是米
        top_offset = UltrasoundConfig.SECTOR_TOP_OFFSET
        top_radius = UltrasoundConfig.SECTOR_TOP_RADIUS

        in_angle = np.abs(theta) <= angle_half
        in_depth = (r >= (top_offset - top_radius * 0.5)) & (r <= depth)

        arc_center_y = self.probe_position_y - top_offset
        dist_to_arc = np.sqrt(dx ** 2 + (Y - arc_center_y) ** 2)

        above_center = Y > arc_center_y
        inside_arc = dist_to_arc < top_radius
        top_exclusion = above_center & inside_arc

        sector = in_angle & in_depth & (~top_exclusion)

        self._tex_cache = dict(W=W, H=H, X=X, Y=Y, r=r, theta=theta, sector=sector)
        return self._tex_cache

    def _create_tissue_mask(self, vertices_xy):
        """创建组织区域蒙版"""
        if vertices_xy is None or len(vertices_xy) < 3:
            return None

        cache = self._prepare_texture_grid()
        H, W = cache["H"], cache["W"]
        X, Y = cache["X"], cache["Y"]

        pts = vertices_xy
        if len(pts) > 1000:
            pts = pts[:: max(1, len(pts) // 1000)]
        if not np.allclose(pts[0], pts[-1]):
            pts = np.vstack([pts, pts[0]])

        from matplotlib.path import Path as MplPath
        path = MplPath(pts)

        points = np.column_stack([X.ravel(), Y.ravel()])
        mask = path.contains_points(points).reshape(H, W)

        return mask.astype(np.float32)

    def _contour_mask_texture(self, vertices_xy):
        """创建轮廓边界蒙版"""
        if vertices_xy is None or len(vertices_xy) < 3:
            return None

        cache = self._prepare_texture_grid()
        H, W = cache["H"], cache["W"]
        xmin, xmax = self.x_range
        ymin, ymax = self.y_range

        pts = vertices_xy
        if len(pts) > 1200:
            pts = pts[:: max(1, len(pts) // 1200)]
        if not np.allclose(pts[0], pts[-1]):
            pts = np.vstack([pts, pts[0]])

        mask = np.zeros((H, W), dtype=bool)

        def to_ij(x, y):
            j = int((x - xmin) / (xmax - xmin) * (W - 1))
            i = int((y - ymin) / (ymax - ymin) * (H - 1))
            return i, j

        for x, y in pts:
            i, j = to_ij(x, y)
            if 0 <= i < H and 0 <= j < W:
                mask[i, j] = True

        mask = dilate_binary(mask, UltrasoundConfig.CONTOUR_THICKNESS_PX)
        cm = mean_blur_2d(mask.astype(np.float32), radius=2)
        cm = np.clip(cm, 0, 1)
        return cm

    def _generate_line_segments(self, width, rng):
        """生成断续线段"""
        segments = []
        current_pos = 0.0

        if rng.random() > UltrasoundConfig.SCAN_LINE_BREAK_PROBABILITY:
            return [(0.0, 1.0)]

        while current_pos < 1.0:
            segment_length = rng.uniform(
                UltrasoundConfig.SCAN_LINE_BREAK_MIN_LENGTH,
                UltrasoundConfig.SCAN_LINE_BREAK_MAX_LENGTH
            )

            end_pos = min(current_pos + segment_length, 1.0)
            segments.append((current_pos, end_pos))

            gap = rng.uniform(
                UltrasoundConfig.SCAN_LINE_BREAK_GAP_MIN,
                UltrasoundConfig.SCAN_LINE_BREAK_GAP_MAX
            )
            current_pos = end_pos + gap

            if 1.0 - current_pos < UltrasoundConfig.SCAN_LINE_BREAK_MIN_LENGTH:
                break

        return segments

    def _build_ultrasound_texture(self, vertices_xy, rng):
        cache = self._prepare_texture_grid()
        r = cache["r"]
        sector = cache["sector"]
        H, W = cache["H"], cache["W"]

        # 深度衰减
        depth = np.clip(r / UltrasoundConfig.SECTOR_DEPTH, 0, 1)
        tgc = UltrasoundConfig.BRIGHTNESS_TOP * np.exp(-UltrasoundConfig.ATTENUATION * depth)
        tgc = np.clip(tgc, UltrasoundConfig.BRIGHTNESS_FLOOR, UltrasoundConfig.BRIGHTNESS_TOP)

        tissue_mask = self._create_tissue_mask(vertices_xy)

        if tissue_mask is None:
            I = np.full((H, W), UltrasoundConfig.BACKGROUND_BRIGHTNESS, dtype=np.float32)
            I = np.where(sector, I, 0.0)
            return np.clip(I, 0, 1) ** UltrasoundConfig.GAMMA

        # 扫描线
        scan_line_map = np.zeros((H, W), dtype=np.float32)

        y_coords = np.arange(H, dtype=np.float32)
        y_world = self.y_range[1] - (y_coords / (H - 1)) * (self.y_range[1] - self.y_range[0])
        depth_per_row = np.clip((self.probe_position_y - y_world) / UltrasoundConfig.SECTOR_DEPTH, 0, 1)

        total_lines = UltrasoundConfig.SCAN_LINE_BASE_COUNT

        line_positions = []
        for i in range(total_lines):
            base_pos = i / (total_lines - 1) if total_lines > 1 else 0.5
            transformed_pos = base_pos ** (1.0 + UltrasoundConfig.SCAN_LINE_DENSITY_VAR * 1.5)

            row = int(transformed_pos * (H - 1))
            row_jitter = rng.normal(0, UltrasoundConfig.SCAN_LINE_POSITION_JITTER)
            row = int(np.clip(row + row_jitter, 0, H - 1))

            depth_ratio = depth_per_row[row]
            brightness = UltrasoundConfig.SCAN_LINE_BRIGHTNESS * (1.0 - depth_ratio * 0.7)
            brightness += rng.normal(0, UltrasoundConfig.SCAN_LINE_BRIGHTNESS_JITTER)
            brightness = np.clip(brightness, 0, 1)

            thickness = UltrasoundConfig.SCAN_LINE_BASE_THICKNESS + rng.normal(0,
                                                                               UltrasoundConfig.SCAN_LINE_THICKNESS_JITTER)
            thickness = max(0.5, thickness)

            line_positions.append((row, brightness, thickness))

        x_pixels = np.arange(W, dtype=np.float32)
        x_normalized = x_pixels / (W - 1)

        for row, brightness, thickness in line_positions:
            if row < 0 or row >= H:
                continue

            wave = UltrasoundConfig.SCAN_LINE_WAVE_AMP * np.sin(
                2 * np.pi * x_normalized * UltrasoundConfig.SCAN_LINE_WAVE_FREQ +
                rng.uniform(0, 2 * np.pi)
            )

            segments = self._generate_line_segments(W, rng)

            for seg_start, seg_end in segments:
                col_start = int(seg_start * W)
                col_end = int(seg_end * W)

                for col in range(col_start, min(col_end, W)):
                    offset = wave[col]
                    distance = abs((row + offset) - np.arange(max(0, row - 4), min(H, row + 5)))

                    weight = np.exp(-(distance ** 2) / (2 * thickness ** 2))

                    for idx, r_idx in enumerate(range(max(0, row - 4), min(H, row + 5))):
                        scan_line_map[r_idx, col] += brightness * weight[idx]

        scan_line_map = np.clip(scan_line_map, 0, 1)

        # 组织底纹
        tissue_texture = rng.normal(0, 1, size=(H, W)).astype(np.float32)
        tissue_texture = mean_blur_2d(tissue_texture, radius=UltrasoundConfig.TISSUE_SMOOTH_RADIUS)
        tissue_texture = (tissue_texture - tissue_texture.min()) / (tissue_texture.max() - tissue_texture.min() + 1e-8)
        tissue_texture = (1.0 - UltrasoundConfig.TISSUE_TEXTURE) + UltrasoundConfig.TISSUE_TEXTURE * tissue_texture

        # 散斑噪声
        speckle = rng.rayleigh(scale=1.0, size=(H, W)).astype(np.float32)
        speckle = speckle / (speckle.mean() + 1e-8)
        if UltrasoundConfig.SPECKLE_SMOOTH_RADIUS > 0:
            speckle = mean_blur_2d(speckle, radius=UltrasoundConfig.SPECKLE_SMOOTH_RADIUS)
            speckle = speckle / (speckle.mean() + 1e-8)

        speckle = (1.0 - UltrasoundConfig.SPECKLE_STRENGTH) + UltrasoundConfig.SPECKLE_STRENGTH * speckle

        # 合成
        tissue_brightness = UltrasoundConfig.TISSUE_BASE_BRIGHTNESS * tgc * (
                1.0 + scan_line_map * 0.7) * tissue_texture * speckle

        background_brightness = UltrasoundConfig.BACKGROUND_BRIGHTNESS * speckle * 0.8

        I = tissue_mask * tissue_brightness + (1 - tissue_mask) * background_brightness
        I = np.where(sector, I, 0.0)

        # 轮廓高回声
        cm = self._contour_mask_texture(vertices_xy)
        if cm is not None:
            I = np.clip(I + UltrasoundConfig.CONTOUR_GAIN * cm * tgc * tissue_mask, 0, 1)

        # Gamma校正
        I = np.clip(I, 0, 1) ** UltrasoundConfig.GAMMA
        return I

    def _draw_medical_mesh(self, ax_3d):
        vertices = self.mesh.vertices
        faces = self.mesh.faces
        num_faces = len(faces)

        if num_faces > 5000:
            sampled_faces = faces[:: max(1, num_faces // 3000)]
        elif num_faces > 2000:
            sampled_faces = faces[:: max(1, num_faces // 1500)]
        else:
            sampled_faces = faces

        triangles = [[vertices[f[0]], vertices[f[1]], vertices[f[2]]] for f in sampled_faces]
        coll = Poly3DCollection(
            triangles,
            facecolors=['#FFB6A3'] * len(triangles),
            edgecolors=['#CC8B77'] * len(triangles),
            linewidths=0.2,
            alpha=0.95,
            shade=True,
            antialiased=True,
            zsort='average'
        )
        ax_3d.add_collection3d(coll)

    def _draw_slice_plane(self, ax_3d, z):
        x_min, x_max = self.bounds_3d[0][0], self.bounds_3d[1][0]
        y_min, y_max = self.bounds_3d[0][1], self.bounds_3d[1][1]
        px = (x_max - x_min) * 0.15
        py = (y_max - y_min) * 0.15

        xx, yy = np.meshgrid(
            np.linspace(x_min - px, x_max + px, 15),
            np.linspace(y_min - py, y_max + py, 15)
        )
        zz = np.ones_like(xx) * z

        return ax_3d.plot_surface(xx, yy, zz, alpha=0.3, color='#a0a0a0',
                                  edgecolor='#707070', linewidth=0.5, shade=False, zorder=100)

    def _draw_slice_contour_3d(self, ax_3d, z):
        if self.current_slice_vertices is None:
            return None, None
        v = self.current_slice_vertices
        v3d = np.column_stack([v[:, 0], v[:, 1], np.ones(len(v)) * z])

        contour = ax_3d.plot(v3d[:, 0], v3d[:, 1], v3d[:, 2],
                             color='#e0e0e0', linewidth=4, zorder=101)[0]
        skip = max(1, len(v3d) // 30)
        pts = ax_3d.scatter(v3d[::skip, 0], v3d[::skip, 1], v3d[::skip, 2],
                            c='#ffffff', s=45, alpha=0.9, edgecolors='#cccccc', linewidths=1.2, zorder=102)
        return contour, pts

    @staticmethod
    def _calculate_area(vertices):
        x = vertices[:, 0]
        y = vertices[:, 1]
        return 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))

    def visualize_dual_view(self):
        fig = plt.figure(figsize=(18, 9))
        fig.patch.set_facecolor('#1a1a1a')

        ax_2d = plt.subplot(1, 2, 1)
        ax_3d = plt.subplot(1, 2, 2, projection='3d')
        plt.subplots_adjust(left=0.05, right=0.98, bottom=0.15, top=0.94, wspace=0.15)  # ✅ bottom改大为0.15，给新滑条留空间

        # ---- 2D ----
        ax_2d.set_xlim(self.x_range)
        ax_2d.set_ylim(self.y_range)
        ax_2d.set_aspect('equal')
        ax_2d.set_facecolor('#050505')
        ax_2d.set_xlabel('X (mm)', color='#cccccc', fontsize=11)
        ax_2d.set_ylabel('Y (mm)', color='#cccccc', fontsize=11)
        ax_2d.tick_params(colors='#888888', labelsize=8)
        ax_2d.grid(True, color='#2a2a2a', alpha=0.25, linestyle='-', linewidth=0.5)
        ax_2d.set_title('ULTRASOUND IMAGE', color='#b0b0b0', fontsize=12, fontweight='bold', pad=10, family='monospace')

        sector_path = self._create_sector_mask_path()

        sector_patch = PathPatch(
            sector_path,
            facecolor='none',
            edgecolor=UltrasoundConfig.SECTOR_EDGE_COLOR,
            linewidth=1.2,
            linestyle='--',
            alpha=UltrasoundConfig.SECTOR_EDGE_ALPHA,
            zorder=10
        )
        ax_2d.add_patch(sector_patch)

        rng = np.random.default_rng(12345)
        tex0 = self._build_ultrasound_texture(vertices_xy=None, rng=rng)

        img = ax_2d.imshow(
            tex0,
            extent=[self.x_range[0], self.x_range[1], self.y_range[0], self.y_range[1]],
            origin='lower',
            cmap='gray',
            vmin=0, vmax=1,
            interpolation='bilinear',
            zorder=1
        )
        img.set_clip_path(sector_path, transform=ax_2d.transData)

        line_2d, = ax_2d.plot([], [], color='#f0f0f0', linewidth=1.5, alpha=0.6, zorder=3)
        line_2d.set_clip_path(sector_path, transform=ax_2d.transData)

        status_2d = ax_2d.text(
            0.02, 0.98, '',
            transform=ax_2d.transAxes,
            fontsize=8, color='#d0d0d0',
            verticalalignment='top',
            family='monospace',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#1a1a1a', edgecolor='#555555', alpha=0.9, linewidth=1.2)
        )

        # ---- 3D ----
        ax_3d.set_facecolor('#1a1a1a')
        ax_3d.set_xlabel('X (mm)', color='#cccccc', fontsize=10, labelpad=8)
        ax_3d.set_ylabel('Y (mm)', color='#cccccc', fontsize=10, labelpad=8)
        ax_3d.set_zlabel('Z (mm)', color='#cccccc', fontsize=10, labelpad=8)
        ax_3d.tick_params(colors='#888888', labelsize=8)

        ax_3d.set_xlim(self.bounds_3d[0][0], self.bounds_3d[1][0])
        ax_3d.set_ylim(self.bounds_3d[0][1], self.bounds_3d[1][1])
        ax_3d.set_zlim(self.bounds_3d[0][2], self.bounds_3d[1][2])
        ax_3d.set_box_aspect([
            self.bounds_3d[1][0] - self.bounds_3d[0][0],
            self.bounds_3d[1][1] - self.bounds_3d[0][1],
            self.bounds_3d[1][2] - self.bounds_3d[0][2]
        ])
        ax_3d.grid(True, color='#3a3a3a', alpha=0.3, linestyle='--', linewidth=0.5)
        ax_3d.set_title('3D ANATOMICAL MODEL', color='#b0b0b0', fontsize=12, fontweight='bold', pad=15,
                        family='monospace')

        #  保存原始网格的副本（用于旋转）
        original_vertices = self.mesh.vertices.copy()

        self._draw_medical_mesh(ax_3d)
        slice_plane = None
        slice_contour_3d = None
        slice_points_3d = None

        # ---- Depth Slider ----
        ax_slider = plt.axes([0.15, 0.08, 0.7, 0.025], facecolor='#2a2a2a')  # ✅ Y位置从0.04改为0.08
        slider = Slider(ax_slider, 'DEPTH (mm)', self.z_min, self.z_max, valinit=self.current_z,
                        color='#888888', track_color='#1a1a1a')
        for s in ax_slider.spines.values():
            s.set_color('#555555')
        slider.label.set_color('#cccccc')
        slider.valtext.set_color('#cccccc')

        #  ---- 新增：Z轴旋转滑条 ----
        ax_rotation = plt.axes([0.15, 0.04, 0.7, 0.025], facecolor='#2a2a2a')
        rotation_slider = Slider(ax_rotation, 'ROTATION (°)', 0, 360, valinit=0,
                                 color='#888888', track_color='#1a1a1a')
        for s in ax_rotation.spines.values():
            s.set_color('#555555')
        rotation_slider.label.set_color('#cccccc')
        rotation_slider.valtext.set_color('#cccccc')

        def update(_val):
            nonlocal slice_plane, slice_contour_3d, slice_points_3d

            z = slider.val
            vertices = self.get_cross_section_ordered(z)

            if vertices is not None and len(vertices) > 2:
                if not np.allclose(vertices[0], vertices[-1]):
                    vertices = np.vstack([vertices, vertices[0]])
                self.current_slice_vertices = vertices

                line_2d.set_data(vertices[:, 0], vertices[:, 1])

                area = self._calculate_area(vertices)
                dp = (z - self.z_min) / (self.z_max - self.z_min + 1e-9) * 100
                status_2d.set_text(f'ACTIVE\nPTS: {len(vertices) - 1}\nDEP: {dp:.1f}%\nAREA: {area:.1f}mm²')
                status_2d.set_color('#d0d0d0')
            else:
                self.current_slice_vertices = None
                line_2d.set_data([], [])
                status_2d.set_text('NO SIGNAL')
                status_2d.set_color('#888888')

            rng = np.random.default_rng(12345 + int((z - self.z_min) * 1000))
            tex = self._build_ultrasound_texture(self.current_slice_vertices, rng)
            img.set_data(tex)

            if slice_plane is not None:
                slice_plane.remove()
            if slice_contour_3d is not None:
                slice_contour_3d.remove()
            if slice_points_3d is not None:
                slice_points_3d.remove()

            slice_plane = self._draw_slice_plane(ax_3d, z)
            if self.current_slice_vertices is not None:
                slice_contour_3d, slice_points_3d = self._draw_slice_contour_3d(ax_3d, z)

            fig.canvas.draw_idle()

        #  新增：旋转更新函数
        def update_rotation(_val):
            nonlocal slice_plane, slice_contour_3d, slice_points_3d  #  添加nonlocal声明

            angle_deg = rotation_slider.val
            angle_rad = np.radians(angle_deg)

            # 计算旋转中心（模型中心）
            center = original_vertices.mean(axis=0)

            # 构建Z轴旋转矩阵
            cos_a = np.cos(angle_rad)
            sin_a = np.sin(angle_rad)
            rotation_matrix = np.array([
                [cos_a, -sin_a, 0],
                [sin_a, cos_a, 0],
                [0, 0, 1]
            ])

            # 应用旋转：先平移到原点，旋转，再平移回去
            rotated_vertices = (original_vertices - center) @ rotation_matrix.T + center
            self.mesh.vertices[:] = rotated_vertices

            # 重绘3D网格
            ax_3d.clear()
            ax_3d.set_facecolor('#1a1a1a')
            ax_3d.set_xlabel('X (mm)', color='#cccccc', fontsize=10, labelpad=8)
            ax_3d.set_ylabel('Y (mm)', color='#cccccc', fontsize=10, labelpad=8)
            ax_3d.set_zlabel('Z (mm)', color='#cccccc', fontsize=10, labelpad=8)
            ax_3d.tick_params(colors='#888888', labelsize=8)
            ax_3d.set_xlim(self.bounds_3d[0][0], self.bounds_3d[1][0])
            ax_3d.set_ylim(self.bounds_3d[0][1], self.bounds_3d[1][1])
            ax_3d.set_zlim(self.bounds_3d[0][2], self.bounds_3d[1][2])
            ax_3d.set_box_aspect([
                self.bounds_3d[1][0] - self.bounds_3d[0][0],
                self.bounds_3d[1][1] - self.bounds_3d[0][1],
                self.bounds_3d[1][2] - self.bounds_3d[0][2]
            ])
            ax_3d.grid(True, color='#3a3a3a', alpha=0.3, linestyle='--', linewidth=0.5)
            ax_3d.set_title('3D ANATOMICAL MODEL', color='#b0b0b0', fontsize=12, fontweight='bold', pad=15,
                            family='monospace')

            self._draw_medical_mesh(ax_3d)

            #  重新绘制切片平面和轮廓
            z = slider.val
            slice_plane = self._draw_slice_plane(ax_3d, z)
            if self.current_slice_vertices is not None:
                slice_contour_3d, slice_points_3d = self._draw_slice_contour_3d(ax_3d, z)
            else:
                slice_contour_3d = None
                slice_points_3d = None

            #  更新2D视图（因为旋转影响了切面）
            vertices = self.get_cross_section_ordered(z)
            if vertices is not None and len(vertices) > 2:
                if not np.allclose(vertices[0], vertices[-1]):
                    vertices = np.vstack([vertices, vertices[0]])
                self.current_slice_vertices = vertices
                line_2d.set_data(vertices[:, 0], vertices[:, 1])

                area = self._calculate_area(vertices)
                dp = (z - self.z_min) / (self.z_max - self.z_min + 1e-9) * 100
                status_2d.set_text(f'ACTIVE\nPTS: {len(vertices) - 1}\nDEP: {dp:.1f}%\nAREA: {area:.1f}mm²')
                status_2d.set_color('#d0d0d0')

                rng = np.random.default_rng(12345 + int((z - self.z_min) * 1000))
                tex = self._build_ultrasound_texture(self.current_slice_vertices, rng)
                img.set_data(tex)
            else:
                self.current_slice_vertices = None
                line_2d.set_data([], [])
                status_2d.set_text('NO SIGNAL')
                status_2d.set_color('#888888')

                rng = np.random.default_rng(12345)
                tex = self._build_ultrasound_texture(None, rng)
                img.set_data(tex)

            ax_3d.view_init(elev=20, azim=30)
            fig.canvas.draw_idle()

        slider.on_changed(update)
        rotation_slider.on_changed(update_rotation)  #  绑定旋转滑条

        # ---- Buttons ----
        ax_prev = plt.axes([0.12, 0.96, 0.06, 0.03])
        ax_next = plt.axes([0.85, 0.96, 0.06, 0.03])
        ax_reset = plt.axes([0.46, 0.96, 0.08, 0.03])
        ax_rotate = plt.axes([0.19, 0.96, 0.06, 0.03])
        ax_reset_view = plt.axes([0.26, 0.96, 0.08, 0.03])

        btn_prev = Button(ax_prev, '◄ PREV', color='#2a2a2a', hovercolor='#4a4a4a')
        btn_next = Button(ax_next, 'NEXT ►', color='#2a2a2a', hovercolor='#4a4a4a')
        btn_reset = Button(ax_reset, 'RESET', color='#2a2a2a', hovercolor='#4a4a4a')
        btn_rotate = Button(ax_rotate, '↻ ROT', color='#2a2a2a', hovercolor='#4a4a4a')
        btn_reset_view = Button(ax_reset_view, 'RESET 3D', color='#2a2a2a', hovercolor='#4a4a4a')

        for b in [btn_prev, btn_next, btn_reset, btn_rotate, btn_reset_view]:
            b.label.set_color('#cccccc')
            b.label.set_fontsize(8)
            b.label.set_fontweight('bold')
            b.label.set_family('monospace')

        rot = [30]

        def prev_slice(_e):
            step = (self.z_max - self.z_min) / 50
            slider.set_val(max(self.z_min, slider.val - step))

        def next_slice(_e):
            step = (self.z_max - self.z_min) / 50
            slider.set_val(min(self.z_max, slider.val + step))

        def reset_slice(_e):
            slider.set_val(self.current_z)
            rotation_slider.set_val(0)  #  同时重置旋转

        def rotate_view(_e):
            rot[0] = (rot[0] + 30) % 360
            ax_3d.view_init(elev=20, azim=rot[0])
            fig.canvas.draw_idle()

        def reset_3d_view(_e):
            rot[0] = 30
            ax_3d.view_init(elev=20, azim=30)
            fig.canvas.draw_idle()

        btn_prev.on_clicked(prev_slice)
        btn_next.on_clicked(next_slice)
        btn_reset.on_clicked(reset_slice)
        btn_rotate.on_clicked(rotate_view)
        btn_reset_view.on_clicked(reset_3d_view)

        ax_3d.view_init(elev=20, azim=30)

        update(self.current_z)
        plt.show()

        def prev_slice(_e):
            step = (self.z_max - self.z_min) / 50
            slider.set_val(max(self.z_min, slider.val - step))

        def next_slice(_e):
            step = (self.z_max - self.z_min) / 50
            slider.set_val(min(self.z_max, slider.val + step))

        def reset_slice(_e):
            slider.set_val(self.current_z)

        def rotate_view(_e):
            rot[0] = (rot[0] + 30) % 360
            ax_3d.view_init(elev=20, azim=rot[0])
            fig.canvas.draw_idle()

        def reset_3d_view(_e):
            rot[0] = 30
            ax_3d.view_init(elev=20, azim=30)
            fig.canvas.draw_idle()

        btn_prev.on_clicked(prev_slice)
        btn_next.on_clicked(next_slice)
        btn_reset.on_clicked(reset_slice)
        btn_rotate.on_clicked(rotate_view)
        btn_reset_view.on_clicked(reset_3d_view)

        ax_3d.view_init(elev=20, azim=30)

        update(self.current_z)
        plt.show()


def main():
    stl_file_path = "C:/Users/86150/Desktop/breast_500.stl"
    if not os.path.exists(stl_file_path):
        print("错误：找不到文件")
        return

    viewer = DualViewSliceViewer(stl_file_path)

    print("\n" + "=" * 80)
    print(" MEDICAL ULTRASOUND SIMULATOR v2.1 - UNIT FIXED")
    print("=" * 80)
    print("\n【单位统一】所有距离参数已统一为 米(m)")
    print("\n【参数说明】")
    print("   SECTOR_DEPTH = 0.15 m        # 150mm扫描深度")
    print("   SECTOR_TOP_RADIUS = 0.008 m  # 8mm圆弧半径")
    print("   SECTOR_TOP_OFFSET = 0.005 m  # 5mm顶部偏移")
    print("=" * 80 + "\n")

    viewer.visualize_dual_view()


if __name__ == "__main__":
    main()


    """
    == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == ==
    超声成像配置参数
    == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == ==
    所有距离单位统一为
    米(m)
    == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == ==
    """

    # ========================================================================
    # 📐 扇形扫查区域参数
    # ========================================================================

    SECTOR_ANGLE = 60
    # 扇形角度（度）
    # 推荐范围: 45~90
    # - 45度：窄视野，适合深部器官
    # - 60度：标准视野（推荐）
    # - 90度：宽视野，适合浅表结构
    # 效果：数值越大，扫描范围越宽

    SECTOR_DEPTH = 0.15
    # 扫查深度（米）- 相当于150mm
    # 推荐范围: 0.08~0.25 (80mm~250mm)
    # - 0.08m (80mm): 浅表扫描（皮肤、血管）
    # - 0.15m (150mm): 标准深度（推荐）
    # - 0.25m (250mm): 深部扫描（腹部器官）
    # 效果：数值越大，能看到越深的组织

    SECTOR_ORIGIN_Y_OFFSET = 0
    # 探头Y位置偏移（米）
    # 推荐范围: 0~0.02
    # 效果：正值=探头下移，负值=探头上移
    # 一般保持0即可

    # ========================================================================
    # 🎨 圆弧梯形顶部参数（控制扇形顶部形状）
    # ========================================================================

    SECTOR_TOP_RADIUS = 0.008
    # 顶部圆弧半径（米）- 相当于8mm
    # 推荐范围: 0.005~0.015 (5mm~15mm)
    # - 0.005m: 轻微圆角
    # - 0.008m: 标准圆角（推荐）
    # - 0.015m: 很圆润的顶部
    # 效果：数值越大，顶部越圆润

    SECTOR_TOP_OFFSET = 0.005
    # 顶部起始距离（米）- 探头到扇形起点的距离，相当于5mm
    # 推荐范围: 0.003~0.01 (3mm~10mm)
    # - 0.003m: 扇形靠近探头
    # - 0.005m: 标准距离（推荐）
    # - 0.01m: 扇形远离探头（顶部有较大空白区）
    # 效果：数值越大，扇形离探头越远

    # ========================================================================
    # 🖼️ 纹理分辨率（影响图像质量和性能）
    # ========================================================================

    TEX_W = 500
    TEX_H = 500
    # 纹理宽度/高度（像素）
    # 推荐范围: 300~800
    # - 300: 快速预览（性能好，质量一般）
    # - 500: 标准质量（推荐）
    # - 800: 高质量（慢但细腻）
    # 效果：数值越大，图像越细腻但越慢
    # ⚠️ 注意：过大会导致卡顿

    # ========================================================================
    # 🌓 背景与组织对比度（最重要的视觉参数）
    # ========================================================================

    BACKGROUND_BRIGHTNESS = 0.12
    # 背景亮度（0~1）- 扇形内无组织区域的灰度
    # 推荐范围: 0.05~0.25
    # - 0.05: 很暗的背景（高对比度）
    # - 0.12: 标准背景（推荐）
    # - 0.25: 较亮背景（低对比度）
    # 效果：数值越小，背景越黑，组织越突出

    TISSUE_BASE_BRIGHTNESS = 0.55
    # 组织基础亮度（0~1）- 切片内的基础灰度
    # 推荐范围: 0.4~0.75
    # - 0.4: 偏暗的组织（低回声）
    # - 0.55: 标准组织（推荐）
    # - 0.75: 偏亮的组织（高回声）
    # 效果：数值越大，组织越亮
    # 💡 建议：TISSUE - BACKGROUND 差值 > 0.3 效果最好

    # ========================================================================
    # 📉 深度衰减（模拟声波能量损耗）
    # ========================================================================

    ATTENUATION = 1.8
    # 深度衰减系数
    # 推荐范围: 1.2~3.0
    # - 1.2: 轻微衰减（底部仍然明亮）
    # - 1.8: 标准衰减（推荐）
    # - 3.0: 强烈衰减（底部很暗）
    # 效果：数值越大，深处越暗（符合真实超声）

    BRIGHTNESS_TOP = 0.99
    # 顶部最大亮度（0~1）
    # 推荐范围: 0.9~1.0
    # 一般保持0.99即可

    BRIGHTNESS_FLOOR = 0.01
    # 底部最小亮度（0~1）
    # 推荐范围: 0.0~0.05
    # 一般保持0.01即可

    # ========================================================================
    # 📏 横向扫描线参数（灰色横线效果）
    # ========================================================================

    SCAN_LINE_BASE_COUNT = 50
    # 扫描线总数量
    # 推荐范围: 30~100
    # - 30: 稀疏的线条（间隔大）
    # - 50: 标准密度（推荐）
    # - 100: 密集的线条（几乎连成片）
    # 效果：数值越大，线条越密集

    SCAN_LINE_DENSITY_VAR = 0.6
    # 密度变化系数（0~1）- 控制线条从上到下的分布
    # 推荐范围: 0.3~1.0
    # - 0.3: 均匀分布（上下密度相同）
    # - 0.6: 标准变化（推荐）- 底部稍稀疏
    # - 1.0: 强烈变化（顶部很密，底部很稀）
    # 效果：数值越大，底部线条越稀疏

    SCAN_LINE_BRIGHTNESS = 0.32
    # 扫描线基础亮度（0~1）
    # 推荐范围: 0.2~0.5
    # - 0.2: 淡淡的线条（不明显）
    # - 0.32: 标准亮度（推荐）
    # - 0.5: 很明显的线条（有点刺眼）
    # 效果：数值越大，线条越明显

    SCAN_LINE_BRIGHTNESS_JITTER = 0.3
    # 亮度抖动（0~0.5）- 每条线的随机亮度变化
    # 推荐范围: 0.1~0.4
    # - 0.1: 亮度很均匀
    # - 0.3: 标准抖动（推荐）
    # - 0.4: 亮度差异很大
    # 效果：数值越大，线条明暗变化越大（更真实）

    SCAN_LINE_POSITION_JITTER = 1.0
    # 位置抖动（像素）- 线条位置的上下随机偏移
    # 推荐范围: 0.5~3.0
    # - 0.5: 整齐的线条
    # - 1.0: 标准抖动（推荐）
    # - 3.0: 很不规则的线条
    # 效果：数值越大，线条越不规则（模拟扫描抖动）

    SCAN_LINE_THICKNESS_JITTER = 0.7
    # 粗细抖动（0~2）- 线条粗细的随机变化
    # 推荐范围: 0.3~1.5
    # - 0.3: 粗细很均匀
    # - 0.7: 标准抖动（推荐）
    # - 1.5: 粗细差异很大
    # 效果：数值越大，线条粗细变化越大

    SCAN_LINE_BASE_THICKNESS = 1.6
    # 基础厚度（像素）- 线条的平均粗细
    # 推荐范围: 1.0~3.0
    # - 1.0: 细线
    # - 1.6: 标准粗细（推荐）
    # - 3.0: 粗线
    # 效果：数值越大，线条越粗

    SCAN_LINE_WAVE_AMP = 2.0
    # 横向波动幅度（像素）- 线条的波纹起伏
    # 推荐范围: 0.5~4.0
    # - 0.5: 几乎是直线
    # - 2.0: 标准波纹（推荐）
    # - 4.0: 很明显的波纹
    # 效果：数值越大，线条越弯曲

    SCAN_LINE_WAVE_FREQ = 2.8
    # 波动频率 - 波纹的密集程度
    # 推荐范围: 1.0~5.0
    # - 1.0: 宽松的波纹（长波）
    # - 2.8: 标准频率（推荐）
    # - 5.0: 密集的波纹（短波）
    # 效果：数值越大，波纹越密集

    # ========================================================================
    # ✂️ 扫描线断续效果（虚线效果）
    # ========================================================================

    SCAN_LINE_BREAK_PROBABILITY = 0.25
    # 断裂概率（0~1）- 有多少比例的线条会断开
    # 推荐范围: 0.0~0.5
    # - 0.0: 全部实线（无断裂）
    # - 0.25: 25%的线有断裂（推荐）
    # - 0.5: 50%的线有断裂（很多虚线）
    # 效果：数值越大，虚线越多
    # 💡 0表示全部实线，1表示全部虚线

    SCAN_LINE_BREAK_MIN_LENGTH = 0.05
    # 最小连续段长度（0~1）- 占总宽度比例
    # 推荐范围: 0.03~0.15
    # - 0.03: 很短的线段
    # - 0.05: 标准短段（推荐）
    # - 0.15: 较长的线段
    # 效果：数值越小，线段越短（破碎感越强）

    SCAN_LINE_BREAK_MAX_LENGTH = 0.35
    # 最大连续段长度（0~1）
    # 推荐范围: 0.2~0.6
    # - 0.2: 都是短段
    # - 0.35: 标准长度（推荐）
    # - 0.6: 可能有很长的段
    # 效果：数值越大，可能出现的最长线段越长

    SCAN_LINE_BREAK_GAP_MIN = 0.02
    # 最小间隙（0~1）
    # 推荐范围: 0.01~0.05
    # - 0.01: 很小的间隙（几乎连续）
    # - 0.02: 标准小间隙（推荐）
    # - 0.05: 明显的间隙
    # 效果：数值越小，间隙越不明显

    SCAN_LINE_BREAK_GAP_MAX = 0.15
    # 最大间隙（0~1）
    # 推荐范围: 0.08~0.3
    # - 0.08: 间隙都很小
    # - 0.15: 标准间隙（推荐）
    # - 0.3: 可能有很大的空白
    # 效果：数值越大，可能的最大断点越明显

    # ========================================================================
    # ✨ 散斑噪声（speckle - 超声特有的颗粒感）
    # ========================================================================

    SPECKLE_STRENGTH = 0.5
    # 散斑强度（0~1）
    # 推荐范围: 0.3~0.8
    # - 0.3: 轻微颗粒感（图像平滑）
    # - 0.5: 标准散斑（推荐）
    # - 0.8: 强烈颗粒感（很粗糙）
    # 效果：数值越大，图像越"沙沙"的感觉
    # 💡 这是超声图像的标志性特征

    SPECKLE_SMOOTH_RADIUS = 1
    # 散斑平滑半径（0~5）
    # 推荐范围: 0~3
    # - 0: 粗糙的散斑（像素级）
    # - 1: 标准平滑（推荐）
    # - 3: 很细腻的散斑（几乎看不出颗粒）
    # 效果：数值越大，散斑越细腻

    # ========================================================================
    # 🎨 组织底纹（低频纹理，模拟组织结构差异）
    # ========================================================================

    TISSUE_TEXTURE = 0.15
    # 组织纹理强度（0~1）
    # 推荐范围: 0.05~0.3
    # - 0.05: 很微弱的纹理（组织很均匀）
    # - 0.15: 标准纹理（推荐）
    # - 0.3: 强烈纹理（组织差异明显）
    # 效果：数值越大，组织内部明暗对比越明显

    TISSUE_SMOOTH_RADIUS = 5
    # 组织纹理平滑半径（1~10）
    # 推荐范围: 3~8
    # - 3: 细小的纹理细节
    # - 5: 标准纹理尺寸（推荐）
    # - 8: 大块的纹理区域
    # 效果：数值越大，纹理的"斑块"越大

    # ========================================================================
    # 🔆 轮廓高回声（边界增强效果）
    # ========================================================================

    CONTOUR_GAIN = 0.60
    # 轮廓增益（0~1）
    # 推荐范围: 0.3~0.8
    # - 0.3: 轻微的边界亮化
    # - 0.60: 标准增益（推荐）
    # - 0.8: 很明显的亮边界
    # 效果：数值越大，组织边缘越亮（白色轮廓越明显）
    # 💡 真实超声中，组织界面会产生强回声

    CONTOUR_THICKNESS_PX = 2
    # 轮廓厚度（像素）
    # 推荐范围: 1~5
    # - 1: 细细的边界线
    # - 2: 标准厚度（推荐）
    # - 5: 很粗的边界线
    # 效果：数值越大，边界线越粗

    # ========================================================================
    # 🎚️ 图像后处理
    # ========================================================================

    GAMMA = 1.25
    # Gamma校正（0.5~2.0）
    # 推荐范围: 0.8~1.5
    # - 0.8: 整体提亮（看起来更明亮）
    # - 1.0: 无校正（线性）
    # - 1.25: 标准校正（推荐）- 轻微压暗增加对比度
    # - 1.5: 明显压暗（高对比度）
    # 效果：<1变亮，>1变暗
    # 💡 调整整体明暗和对比度的最后一步

    # ========================================================================
    # 🖌️ 扇形边界线样式
    # ========================================================================

    SECTOR_EDGE_COLOR = '#666666'
    # 边界线颜色（HTML颜色代码）
    # 推荐选项:
    # - '#666666': 中灰色（推荐）
    # - '#888888': 亮灰色
    # - '#444444': 暗灰色
    # - '#00ff00': 绿色（经典超声风格）

    SECTOR_EDGE_ALPHA = 0.45
    # 边界线透明度（0~1）
    # 推荐范围: 0.3~0.7
    # - 0.3: 很淡（几乎看不见）
    # - 0.45: 标准透明度（推荐）
    # - 0.7: 很明显
    # 效果：数值越大，边界线越明显


# ============================================================================
# 💡 快速调参场景示例
# ============================================================================

# 【场景1】高对比度清晰图像（适合教学演示）
# BACKGROUND_BRIGHTNESS = 0.08        # 背景很暗
# TISSUE_BASE_BRIGHTNESS = 0.70       # 组织很亮
# CONTOUR_GAIN = 0.75                 # 边界很明显
# SCAN_LINE_BRIGHTNESS = 0.40         # 扫描线明显
# GAMMA = 1.35                        # 增强对比度

# 【场景2】真实临床风格（噪点多、对比度中等）
# SPECKLE_STRENGTH = 0.65             # 更多散斑
# TISSUE_TEXTURE = 0.25               # 更多纹理
# SCAN_LINE_BREAK_PROBABILITY = 0.35  # 更多虚线
# ATTENUATION = 2.2                   # 更强衰减

# 【场景3】平滑低噪版本（适合图像分析）
# SPECKLE_STRENGTH = 0.35             # 少散斑
# SPECKLE_SMOOTH_RADIUS = 2           # 更平滑
# SCAN_LINE_BASE_COUNT = 30           # 少扫描线
# TISSUE_TEXTURE = 0.08               # 少纹理

# 【场景4】深部扫描模式（看得深但底部暗）
# SECTOR_DEPTH = 0.20                 # 200mm深度
# ATTENUATION = 2.5                   # 强衰减
# BRIGHTNESS_FLOOR = 0.0              # 底部可以全黑
# SCAN_LINE_DENSITY_VAR = 0.8         # 底部线条很稀疏

# 【场景5】浅表高分辨模式（看得浅但很清晰）
# SECTOR_DEPTH = 0.08                 # 80mm浅表
# TEX_W = 700                         # 高分辨率
# TEX_H = 700
# ATTENUATION = 1.2                   # 轻微衰减
# CONTOUR_THICKNESS_PX = 3            # 粗边界

# ============================================================================