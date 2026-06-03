import Sofa.Core
import numpy as np
import math

import matplotlib
# 使用 Qt5Agg 后端（利用 SOFA 自带 Qt，不用 TkAgg）
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt


def quaternion_to_rotation_matrix(q):
    """四元数 [qx,qy,qz,qw] -> 3x3 旋转矩阵"""
    qx, qy, qz, qw = q
    xx, yy, zz = qx*qx, qy*qy, qz*qz
    xy, xz, yz = qx*qy, qx*qz, qy*qz
    wx, wy, wz = qw*qx, qw*qy, qw*qz
    R = np.array([
        [1 - 2*(yy+zz),     2*(xy-wz),     2*(xz+wy)],
        [    2*(xy+wz), 1 - 2*(xx+zz),     2*(yz-wx)],
        [    2*(xz-wy),     2*(yz+wx), 1 - 2*(xx+yy)]
    ])
    return R


class UltrasoundSimulator(Sofa.Core.Controller):
    """
    几何超声模拟器（简化版）：
    - 探头 → 扇形扫描线（朝向结节）
    - 与乳腺碰撞表面 + 结节表面相交
    - 乳腺边界较暗，结节边界较亮
    - 使用 matplotlib + Qt5Agg 在独立窗口显示 B 模边界图
    """

    def __init__(self, root, probe, breast, lesion,
                 n_lines=64, n_samples=256,
                 fov_deg=60.0, max_depth=0.08):
        Sofa.Core.Controller.__init__(self)
        self.root = root
        self.probe = probe
        self.breast = breast
        self.lesion = lesion

        self.n_lines = n_lines
        self.n_samples = n_samples
        self.fov = math.radians(fov_deg)
        self.max_depth = max_depth
        self.depths = np.linspace(0.0, max_depth, n_samples)

        # 乳腺 & 结节表面网格
        self.vertices_breast = np.zeros((0, 3))
        self.triangles_breast = np.zeros((0, 3), dtype=int)
        self.vertices_lesion = np.zeros((0, 3))
        self.triangles_lesion = np.zeros((0, 3), dtype=int)

        # matplotlib 图像窗口
        plt.ion()
        self.fig, self.ax = plt.subplots()
        self.im = self.ax.imshow(
            np.zeros((self.n_samples, self.n_lines)),
            cmap='gray', vmin=0.0, vmax=1.0,
            origin='lower', aspect='auto'
        )
        self.ax.set_title("US (breast + lesion)")
        self.ax.set_xlabel("scanline")
        self.ax.set_ylabel("depth")
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

        print("[US] UltrasoundSimulator (breast + lesion, Qt5Agg backend) initialized")
        self._frame = 0

    # ------------------ 网格更新 ------------------
    def update_meshes(self):
        """从 Breast 和 Lesion 中更新表面网格"""

        # 乳腺碰撞表面：CollisionBreast/breast_collision_state
        self.vertices_breast = np.zeros((0, 3))
        self.triangles_breast = np.zeros((0, 3), dtype=int)
        try:
            if hasattr(self.breast, "node"):
                col_node = self.breast.node.getChild("CollisionBreast")
                if col_node:
                    col_state = col_node.getObject("breast_collision_state")
                    if col_state is not None:
                        self.vertices_breast = np.array(col_state.position.value)

                    # 找含有 triangles 的拓扑对象
                    for obj in col_node.objects:
                        if hasattr(obj, "triangles"):
                            self.triangles_breast = np.array(obj.triangles.value, dtype=int)
                            break
        except Exception as e:
            print("[US] update_meshes: breast mesh error:", e)

        # 结节表面
        self.vertices_lesion = np.zeros((0, 3))
        self.triangles_lesion = np.zeros((0, 3), dtype=int)
        try:
            if hasattr(self.lesion, "state") and self.lesion.state is not None:
                self.vertices_lesion = np.array(self.lesion.state.position.value)
            if hasattr(self.lesion, "topology") and hasattr(self.lesion.topology, "triangles"):
                self.triangles_lesion = np.array(self.lesion.topology.triangles.value, dtype=int)
        except Exception as e:
            print("[US] update_meshes: lesion mesh error:", e)

        if not hasattr(self, "_mesh_dbg_printed"):
            print(f"[US] breast mesh: V={self.vertices_breast.shape}, F={self.triangles_breast.shape}")
            print(f"[US] lesion mesh: V={self.vertices_lesion.shape}, F={self.triangles_lesion.shape}")
            if self.vertices_lesion.shape[0] > 0:
                print(f"[US] lesion first vertex: {self.vertices_lesion[0]}")
            self._mesh_dbg_printed = True

    # ------------------ 相交计算 ------------------
    @staticmethod
    def ray_triangle_intersect(ray_origin, ray_dir, v0, v1, v2):
        eps = 1e-8
        edge1 = v1 - v0
        edge2 = v2 - v0
        h = np.cross(ray_dir, edge2)
        a = np.dot(edge1, h)
        if abs(a) < eps:
            return None
        f = 1.0 / a
        s = ray_origin - v0
        u = f * np.dot(s, h)
        if u < 0.0 or u > 1.0:
            return None
        q = np.cross(s, edge1)
        v = f * np.dot(ray_dir, q)
        if v < 0.0 or u + v > 1.0:
            return None
        t = f * np.dot(edge2, q)
        if t > eps:
            return t
        return None

    def compute_scanline_echo(self, origin, direction):
        echo = np.zeros(self.n_samples)

        t_min = None
        amp_at_t_min = 0.0

        # 1) 乳腺边界（亮度 0.5）
        if self.triangles_breast.size > 0 and self.vertices_breast.size > 0:
            for tri in self.triangles_breast:
                v0 = self.vertices_breast[tri[0]]
                v1 = self.vertices_breast[tri[1]]
                v2 = self.vertices_breast[tri[2]]
                t = self.ray_triangle_intersect(origin, direction, v0, v1, v2)
                if t is not None and 0.0 < t < self.max_depth:
                    if t_min is None or t < t_min:
                        t_min = t
                        amp_at_t_min = 0.5

        # 2) 结节边界（亮度 1.0）
        if self.triangles_lesion.size > 0 and self.vertices_lesion.size > 0:
            for tri in self.triangles_lesion:
                v0 = self.vertices_lesion[tri[0]]
                v1 = self.vertices_lesion[tri[1]]
                v2 = self.vertices_lesion[tri[2]]
                t = self.ray_triangle_intersect(origin, direction, v0, v1, v2)
                if t is not None and 0.0 < t < self.max_depth:
                    if t_min is None or t < t_min:
                        t_min = t
                        amp_at_t_min = 1.0

        if t_min is None:
            return echo

        hit_depth = t_min
        idx = int(hit_depth / self.max_depth * (self.n_samples - 1))

        # 让线条厚一点：以 idx 为中心，扩展若干点
        for k in range(-2, 3):
            j = idx + k
            if 0 <= j < self.n_samples:
                echo[j] = max(echo[j], amp_at_t_min * (1.0 - 0.2*abs(k)))

        return echo

    # ------------------ 每帧更新 ------------------
    def onAnimateBeginEvent(self, _):
        self.update_meshes()
        if (self.vertices_lesion.size == 0 and
                self.vertices_breast.size == 0):
            return

        self._frame += 1

        pose = np.array(self.probe.state.position.value[0])
        pos = pose[:3]

        # 以“探头→结节”的方向为扇形中心方向
        if self.vertices_lesion.shape[0] > 0:
            delta = self.vertices_lesion[0] - pos
            delta_dir = delta / (np.linalg.norm(delta) + 1e-9)
        elif self.vertices_breast.shape[0] > 0:
            delta = self.vertices_breast[0] - pos
            delta_dir = delta / (np.linalg.norm(delta) + 1e-9)
        else:
            delta_dir = np.array([0.0, 0.0, 1.0])

        # 与 delta_dir 正交的侧向向量
        world_x = np.array([1.0, 0.0, 0.0])
        side = np.cross(delta_dir, world_x)
        if np.linalg.norm(side) < 1e-6:
            world_y = np.array([0.0, 1.0, 0.0])
            side = np.cross(delta_dir, world_y)
        side /= (np.linalg.norm(side) + 1e-9)

        angles = np.linspace(-self.fov/2, self.fov/2, self.n_lines)
        us_image = np.zeros((self.n_samples, self.n_lines))

        for i, a in enumerate(angles):
            dir_world = (math.cos(a) * delta_dir + math.sin(a) * side)
            dir_world /= np.linalg.norm(dir_world)
            us_image[:, i] = self.compute_scanline_echo(pos, dir_world)

        us_image = np.clip(us_image, 0.0, 1.0)

        # 更新 matplotlib 图
        self.im.set_data(us_image)
        self.im.set_clim(0, 1)
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        plt.pause(0.001)