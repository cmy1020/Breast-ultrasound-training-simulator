import Sofa.Core
import numpy as np
import trimesh
from matplotlib.path import Path


# ============================================================
# 1. 超声配置参数
# ============================================================
class UltrasoundConfig:
    """所有距离单位统一为 米(m)"""
    SECTOR_ANGLE = 60
    SECTOR_DEPTH = 0.15
    SECTOR_TOP_OFFSET = 0.005
    SECTOR_TOP_RADIUS = 0.008

    # 纹理分辨率
    TEX_W = 500
    TEX_H = 500

    # 亮度与对比度
    BACKGROUND_BRIGHTNESS = 0.12
    TISSUE_BASE_BRIGHTNESS = 0.55
    GAMMA = 1.25

    # 物理效果强度
    ATTENUATION = 1.8
    BRIGHTNESS_TOP = 0.99
    SPECKLE_STRENGTH = 0.5

    # 扫描线效果
    SCAN_LINE_BRIGHTNESS = 0.32


# ============================================================
# 2. 超声物理模拟引擎 (纯计算类)
# ============================================================
class UltrasoundEngine:
    def __init__(self, config):
        self.cfg = config
        self.W, self.H = config.TEX_W, config.TEX_H
        # 扫描窗口范围 (单位：米)
        self.x_range = [-0.1, 0.1]
        self.y_range = [-0.18, 0.02]
        self._prepare_grid()

    def _prepare_grid(self):
        xs = np.linspace(self.x_range[0], self.x_range[1], self.W)
        ys = np.linspace(self.y_range[0], self.y_range[1], self.H)
        self.X, self.Y = np.meshgrid(xs, ys)

        dx = self.X - 0
        dy = 0 - self.Y
        self.r = np.sqrt(dx ** 2 + dy ** 2)
        self.theta = np.degrees(np.arctan2(dx, dy))

        angle_half = self.cfg.SECTOR_ANGLE / 2
        in_angle = np.abs(self.theta) <= angle_half
        in_depth = (self.r >= self.cfg.SECTOR_TOP_OFFSET) & (self.r <= self.cfg.SECTOR_DEPTH)
        self.sector_mask = in_angle & in_depth

    def generate_frame(self, vertices_2d):
        rng = np.random.default_rng()
        depth = np.clip(self.r / self.cfg.SECTOR_DEPTH, 0, 1)
        tgc = self.cfg.BRIGHTNESS_TOP * np.exp(-self.cfg.ATTENUATION * depth)

        # 组织蒙版生成
        tissue_mask = np.zeros((self.H, self.W), dtype=np.float32)
        if vertices_2d is not None and len(vertices_2d) > 3:
            path = Path(vertices_2d)
            points = np.column_stack([self.X.ravel(), self.Y.ravel()])
            tissue_mask = path.contains_points(points).reshape(self.H, self.W).astype(np.float32)

        speckle = rng.rayleigh(scale=1.0, size=(self.H, self.W)).astype(np.float32)
        speckle = (1.0 - self.cfg.SPECKLE_STRENGTH) + self.cfg.SPECKLE_STRENGTH * (speckle / speckle.mean())

        # 简单的合成逻辑
        img = tissue_mask * self.cfg.TISSUE_BASE_BRIGHTNESS * tgc * speckle
        img += (1 - tissue_mask) * self.cfg.BACKGROUND_BRIGHTNESS * speckle * 0.5
        img = np.where(self.sector_mask, img, 0.0)

        img = (np.clip(img, 0, 1) ** self.cfg.GAMMA * 255).astype(np.uint8)
        return img


# ============================================================
# 3. SOFA 控制器 (负责连接场景节点)
# ============================================================
class USImagerController(Sofa.Core.Controller):
    def __init__(self, root_node, breast_node, probe_node, stl_path, config):
        Sofa.Core.Controller.__init__(self)

        # 数据源引用
        self.breast_mo = breast_node.breast_state
        self.probe_mo = probe_node.probe_state

        # --- 获取显示模型引用 ---
        # 1. 探头前端的随动绿片
        try:
            self.slice_ogl = probe_node.SliceIndicator.ScreenOgl
        except AttributeError:
            print("Error: Could not find SliceIndicator.ScreenOgl in probe_node")
            self.slice_ogl = None

        # 2. 画面上的固定窗口
        try:
            self.monitor_ogl = root_node.FixedMonitor.ScreenOgl
        except AttributeError:
            print("Error: Could not find FixedMonitor.ScreenOgl in root_node")
            self.monitor_ogl = None

        # 加载初始网格
        self.mesh = trimesh.load(stl_path)
        # 初始化计算引擎
        self.engine = UltrasoundEngine(config)
        self.count = 0

    def onAnimateBeginEvent(self, _):
        self.count += 1
        if self.count % 2 != 0:
            return  # 隔帧更新，优化性能

        # 1. 同步乳腺变形后的顶点
        self.mesh.vertices = self.breast_mo.position.array()

        # 2. 获取探头在世界坐标系的位置
        # probe_mo.position.value[0] 返回的是 [x, y, z, qx, qy, qz, qw]
        probe_pose = self.probe_mo.position.value[0]
        probe_pos = probe_pose[:3]

        try:
            # 3. 执行三维切片 (以探头位置为原点，Z轴为法线)
            section = self.mesh.section(plane_origin=probe_pos, plane_normal=[0, 0, 1])

            if section is not None:
                planar_mesh, _ = section.to_planar()
                # 取得最大轮廓顶点
                vertices_2d = planar_mesh.discrete[0]

                # 4. 引擎生成图像数据
                img = self.engine.generate_frame(vertices_2d)
                img_data = img.tobytes()

                # 5. 更新两个显示器的纹理
                if self.slice_ogl:
                    self.slice_ogl.setTexture(img_data, self.engine.W, self.engine.H, "L")
                if self.monitor_ogl:
                    self.monitor_ogl.setTexture(img_data, self.engine.W, self.engine.H, "L")
        except Exception as e:
            # 当探头没碰到模型或切片失败时，不闪退
            pass