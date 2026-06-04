# =============================================================================
# 1. 在最顶端增加这些代码（必须在 import numpy 和 Sofa 之前！）
# =============================================================================
import sys
import os

# Fix Windows GBK encoding issue for unicode characters
os.environ['PYTHONIOENCODING'] = 'utf-8'
try:
    import io
    if hasattr(sys.stdout, 'buffer'):
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
except Exception:
    pass

# ✅ 禁用 Intel Fortran 拦截 Ctrl+C（解决 forrtl: error 200 报错）
os.environ['FOR_DISABLE_CONSOLE_CTRL_HANDLER'] = '1'
# ✅ 禁用 PyTorch 退出时的某些段错误报错
os.environ['OMP_NUM_THREADS'] = '1'

import signal
import threading
import time

# ✅ 暴力退出回调，任何人按Ctrl+C直接核弹级强杀
def force_kill_app(*args):
    print("\n🚨 接收到强制退出信号，瞬间终结进程...")
    os._exit(0)

signal.signal(signal.SIGINT, force_kill_app)
signal.signal(signal.SIGTERM, force_kill_app)
try:
    signal.signal(signal.SIGBREAK, force_kill_app)
except AttributeError:
    pass
import numpy as np
import signal
import Sofa
import Sofa.Core
import Sofa.Simulation
import SofaRuntime

# ✅ 覆盖SOFA的信号处理器（SOFA在import时注册，这里立刻覆盖）
signal.signal(signal.SIGINT,  signal.SIG_DFL)  # 恢复SIGINT默认行为
signal.signal(signal.SIGTERM, signal.SIG_DFL)  # 恢复SIGTERM默认行为
try:
    signal.signal(signal.SIGBREAK, signal.SIG_DFL)  # Windows专属Ctrl+Break
except AttributeError:
    pass  # Linux/Mac没有SIGBREAK，忽略

from PyQt5.QtWidgets import (QApplication, QMainWindow, QOpenGLWidget,
                             QVBoxLayout, QHBoxLayout, QWidget, QLabel,
                             QPushButton, QMessageBox, QStatusBar, QComboBox,
                             QSlider)
from PyQt5.QtCore import QTimer, Qt
from PyQt5.QtGui import QImage, QPixmap, QSurfaceFormat
from OpenGL.GL import *
from OpenGL.GLU import gluPerspective
import ctypes
from PIL import Image, ImageDraw

from simulation import createScene
from components.utils import Parameters
from main_window import Ui_MainWindow

from performance_monitor import monitor

# ============================================================
# ✅ 新增：从切面代码复制过来的工具
# ============================================================

class UltrasoundConfig:
    """超声配置参数（GAN版，仅保留扇形和后处理参数）"""

    # 扇形扫查区域
    SECTOR_ANGLE      = 60
    SECTOR_DEPTH      = 0.30
    SECTOR_TOP_RADIUS = 0.015
    SECTOR_TOP_OFFSET = 0.005

    # 图像分辨率
    TEX_W = 512
    TEX_H = 512

    # 后处理
    GAMMA = 1.25

def mean_blur_2d(img: np.ndarray, radius: int) -> np.ndarray:
    """纯 numpy 均值模糊（从你的第二份代码复制）"""
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
    """简单二值膨胀（从你的第二份代码复制）"""
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

# ============================================================
# 插件加载
# ============================================================
plugins_loaded = False


def ensure_plugins():
    global plugins_loaded
    if not plugins_loaded:
        plugins = [
            "SofaComponentAll",
            "Sofa.Component.Collision.Detection.Algorithm",
            "Sofa.Component.Collision.Detection.Intersection",
            "Sofa.Component.Collision.Response.Contact",
        ]
        for p in plugins:
            try:
                SofaRuntime.importPlugin(p)
            except:
                pass
        plugins_loaded = True


# ============================================================
# SOFA 3D 显示窗口 - 正确优化版
# ============================================================
class SofaGLWidget(QOpenGLWidget):
    """SOFA 3D 仿真显示窗口（正确优化版）"""

    def __init__(self, root_node, parent=None, model_name="Breast", controller=None):
        super().__init__(parent)
        self.root_node = root_node
        self.model_name = model_name  # "Breast" or "Liver"

        # Parameter panel overlay (top-left, below pose overlay)
        self.param_panel = ParameterPanel(self, controller=controller)
        self.param_panel.move(10, 170)
        self.param_panel.show()

        # Camera defaults: adjust per model
        if model_name == "Liver":
            self.rotation_x = 90
            self.rotation_y = 0
            self.zoom = -0.15       # close-up for 0.025x scaled liver
            self.translate_x = 0.036 # center (~-1.44*0.025)
            self.translate_y = -0.068 # center (~2.7*0.025)
            self.scan_plane_size = 0.008  # tiny scan plane for liver
        else:
            self.rotation_x = 90
            self.rotation_y = 0
            self.zoom = -0.5
            self.translate_x = +0.1
            self.translate_y = -0.1

        self.last_pos = None
        self.initialized = False

        # 切面控制参数
        self.show_scan_plane = True
        self.scan_plane_offset = -0.00
        self.scan_plane_size = 0.08

        # ============================================================
        # ✅ 优化1：降低切面计算频率
        # ============================================================.
        self.cross_section_update_counter = 0
        self.cross_section_update_interval = 5  # 每 5 帧更新一次切面
        # ============================================================

        self.current_cross_section = None
        self.current_probe_transform = None

        # ============================================================
        # ✅ 优化2：缓存探头静态数据（只加载一次）
        # ============================================================
        self.probe_mesh_cached = False
        self.probe_positions = None
        self.probe_triangles = None
        self.probe_normals = None
        # ============================================================

        # ============================================================
        # ✅ 优化3：缓存乳腺法线
        # ============================================================
        self.breast_normals_cached = None
        self.breast_normals_update_counter = 0
        self.breast_normals_update_interval = 3
        # ============================================================

        # ============================================================
        # ✅ 新增：VBO缓存变量
        # ============================================================
        # 乳腺VBO
        self._breast_vbo_id = None  # 顶点缓冲对象ID
        self._breast_ibo_id = None  # 索引缓冲对象ID
        self._breast_index_count = 0  # 索引总数
        self._breast_vbo_initialized = False  # 是否已初始化

        # 结节VBO
        self._lesion_vbo_id = None
        self._lesion_ibo_id = None
        self._lesion_index_count = 0
        self._lesion_vbo_initialized = False

        # 探头VBO
        self._probe_vbo_id = None
        self._probe_ibo_id = None
        self._probe_index_count = 0
        self._probe_vbo_initialized = False
        # ============================================================

        # 帧计数
        self.frame_counter = 0

        # ============================================================
        # ✅ 新增：左上角姿态小视口配置
        # ============================================================
        self.overlay_size = 150  # 小视口边长（像素）
        self.overlay_margin = 10  # 距左上角边距（像素）
        self._handle_dl = None  # Display List ID，None=未编译
        # ============================================================

    # ============================================================
    # ✅ 新增：VBO工具函数（插入在initializeGL前面）
    # ============================================================

    def _create_vbo(self, positions, triangles, normals, dynamic=True):
        """
        创建VBO并上传数据到GPU

        Args:
            positions: 顶点位置数组
            triangles: 三角形索引数组
            normals:   顶点法线数组
            dynamic:   True=动态(每帧更新位置), False=静态(只上传一次)

        Returns:
            (vbo_id, ibo_id, index_count)
        """
        pos_arr = np.array(positions, dtype=np.float32)
        tri_arr = np.array(triangles, dtype=np.uint32).flatten()
        nor_arr = np.array(normals, dtype=np.float32)

        # 构建交错顶点数组: [x, y, z, nx, ny, nz] × N顶点
        n_verts = len(pos_arr)
        interleaved = np.zeros((n_verts, 6), dtype=np.float32)
        interleaved[:, 0:3] = pos_arr
        interleaved[:, 3:6] = nor_arr
        interleaved_flat = interleaved.flatten()

        usage = GL_DYNAMIC_DRAW if dynamic else GL_STATIC_DRAW

        # 生成并绑定VBO（顶点数据）
        vbo_id = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, vbo_id)
        glBufferData(GL_ARRAY_BUFFER,
                     interleaved_flat.nbytes,
                     interleaved_flat,
                     usage)

        # 生成并绑定IBO（索引数据，始终静态）
        ibo_id = glGenBuffers(1)
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ibo_id)
        glBufferData(GL_ELEMENT_ARRAY_BUFFER,
                     tri_arr.nbytes,
                     tri_arr,
                     GL_STATIC_DRAW)

        # 解绑
        glBindBuffer(GL_ARRAY_BUFFER, 0)
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0)

        index_count = len(tri_arr)
        return vbo_id, ibo_id, index_count

    def _update_vbo_data(self, vbo_id, positions, normals):
        """
        只更新VBO中的顶点位置和法线（不重新分配内存，比重建快10倍）

        Args:
            vbo_id:    已存在的VBO的ID
            positions: 新的顶点位置
            normals:   新的顶点法线
        """
        pos_arr = np.array(positions, dtype=np.float32)
        nor_arr = np.array(normals, dtype=np.float32)

        n_verts = len(pos_arr)
        interleaved = np.zeros((n_verts, 6), dtype=np.float32)
        interleaved[:, 0:3] = pos_arr
        interleaved[:, 3:6] = nor_arr
        interleaved_flat = interleaved.flatten()

        glBindBuffer(GL_ARRAY_BUFFER, vbo_id)
        # glBufferSubData: 只更新数据内容，不重新分配GPU内存
        glBufferSubData(GL_ARRAY_BUFFER, 0,
                        interleaved_flat.nbytes,
                        interleaved_flat)
        glBindBuffer(GL_ARRAY_BUFFER, 0)

    def _draw_vbo(self, vbo_id, ibo_id, index_count):
        """
        用VBO执行绘制（替代原来的glBegin/glEnd循环）
        整个网格只需1次GPU调用，无论多少三角形

        Args:
            vbo_id:      顶点缓冲ID
            ibo_id:      索引缓冲ID
            index_count: 索引总数（三角形数×3）
        """
        stride = 6 * 4  # 6个float，每个4字节 = 24字节

        glBindBuffer(GL_ARRAY_BUFFER, vbo_id)
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ibo_id)

        # 启用顶点数组功能
        glEnableClientState(GL_VERTEX_ARRAY)
        glEnableClientState(GL_NORMAL_ARRAY)

        # 告诉OpenGL数据在哪里
        # 位置：从偏移0开始，每隔stride字节取一个
        glVertexPointer(3, GL_FLOAT, stride, None)
        # 法线：从偏移12字节开始（跳过xyz），每隔stride字节取一个
        glNormalPointer(GL_FLOAT, stride, ctypes.c_void_p(12))

        # ✅ 核心：一次调用绘制全部三角形
        glDrawElements(GL_TRIANGLES, index_count, GL_UNSIGNED_INT, None)

        # 清理状态
        glDisableClientState(GL_VERTEX_ARRAY)
        glDisableClientState(GL_NORMAL_ARRAY)
        glBindBuffer(GL_ARRAY_BUFFER, 0)
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0)

    # ============================================================
    # ✅ VBO工具函数结束
    # ============================================================

    def initializeGL(self):
        """OpenGL 初始化"""
        try:
            glEnable(GL_DEPTH_TEST)
            glDepthFunc(GL_LEQUAL)
            glClearColor(0.15, 0.15, 0.20, 1.0)

            glEnable(GL_LIGHTING)
            glEnable(GL_LIGHT0)
            glEnable(GL_COLOR_MATERIAL)

            glLightfv(GL_LIGHT0, GL_POSITION, [1.0, 1.0, 1.0, 0.0])
            glLightfv(GL_LIGHT0, GL_AMBIENT, [0.3, 0.3, 0.3, 1.0])
            glLightfv(GL_LIGHT0, GL_DIFFUSE, [0.9, 0.9, 0.9, 1.0])
            glLightfv(GL_LIGHT0, GL_SPECULAR, [0.5, 0.5, 0.5, 1.0])

            self.initialized = True
            print("✓ OpenGL 初始化成功")

        except Exception as e:
            print(f"✗ OpenGL 初始化失败: {e}")

    def resizeGL(self, w, h):
        """窗口大小改变"""
        glViewport(0, 0, w, h)
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()

        aspect = w / (h if h > 0 else 1)
        gluPerspective(45, aspect, 0.01, 100.0)

        glMatrixMode(GL_MODELVIEW)

    def paintGL(self):
        if not self.initialized or not self.root_node:
            return

        try:
            if not hasattr(self, '_scene_structure_logged'):
                self._log_scene_structure()
                self._scene_structure_logged = True

            self.frame_counter += 1

            monitor.render_begin()

            # 基础设置
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
            glLoadIdentity()

            glTranslatef(self.translate_x, self.translate_y, self.zoom)
            glRotatef(self.rotation_x, 1, 0, 0)
            glRotatef(self.rotation_y, 0, 0, 1)

            if self.show_scan_plane:
                self._compute_current_cross_section()

            # 渲染主场景
            self.render_sofa_scene()

            # ✅ 新增：绘制左上角姿态小视口
            self._draw_pose_overlay()

            monitor.render_end()

        except Exception as e:
            if not hasattr(self, '_paint_error_count'):
                self._paint_error_count = 0
            self._paint_error_count += 1
            if self._paint_error_count <= 3:
                print(f"[paintGL error #{self._paint_error_count}] {e}")
                import traceback
                traceback.print_exc()

    def _log_scene_structure(self):
        """Print scene structure once on startup"""
        print("\n" + "=" * 60)
        print(f"Scene structure: {self.model_name}")
        print("=" * 60)

        model_node = self.root_node.getChild(self.model_name)
        if model_node:
            print(f"[OK] {self.model_name} node found")

            collision_node = model_node.getChild(f"Collision{self.model_name}")
            if collision_node:
                collision_mech = collision_node.getObject("breast_collision_state")
                topo = collision_node.getObject("topology")
                if collision_mech and hasattr(collision_mech, 'position'):
                    print(f"  collision mesh: {len(collision_mech.position.value)} vertices")
                if topo and hasattr(topo, 'triangles'):
                    print(f"  topology: {len(topo.triangles.value)} triangles")

            # Check for lesion node (breast model only)
            lesion_node = model_node.getChild("Lesion1")
            if not lesion_node:
                for child in model_node.children:
                    if "Lesion" in child.getName():
                        lesion_node = child
                        break
            if lesion_node:
                print(f"  lesion node: {lesion_node.getName()} (found)")
            else:
                print(f"  no lesion (clean organ scan mode)")
        else:
            print(f"[ERR] {self.model_name} node not found!")

        print("=" * 60 + "\n")

    def _get_scan_plane_transform(self):
        """获取扫描平面的位置和法向量"""
        try:
            probe_node = self.root_node.getChild("Probe")
            if not probe_node:
                return None, None, None, None

            mech = probe_node.getObject("probe_state")
            if not mech or not hasattr(mech, 'position'):
                return None, None, None, None

            pose = mech.position.value[0]
            probe_pos = np.array(pose[:3], dtype=np.float64)
            probe_quat = np.array(pose[3:], dtype=np.float64)

            local_normal = np.array([0, 0, -1], dtype=np.float64)
            local_offset = np.array([0, 0, self.scan_plane_offset], dtype=np.float64)

            world_normal = self._rotate_vector_by_quaternion(local_normal, probe_quat)
            world_offset = self._rotate_vector_by_quaternion(local_offset, probe_quat)

            plane_origin = probe_pos + world_offset
            plane_normal = world_normal / np.linalg.norm(world_normal)

            return plane_origin, plane_normal, probe_pos, probe_quat

        except Exception as e:
            return None, None, None, None

    def _rotate_vector_by_quaternion(self, vec, quat):
        """用四元数旋转向量"""
        qx, qy, qz, qw = quat
        vx, vy, vz = vec

        t = 2.0 * np.array([
            qy * vz - qz * vy,
            qz * vx - qx * vz,
            qx * vy - qy * vx
        ], dtype=np.float64)

        result = vec + qw * t + np.array([
            qy * t[2] - qz * t[1],
            qz * t[0] - qx * t[2],
            qx * t[1] - qy * t[0]
        ], dtype=np.float64)

        return result

    def _compute_mesh_plane_intersection(self, positions, triangles, plane_origin, plane_normal):
        """计算网格与平面的交线（向量化优化）"""
        intersection_segments = []

        # 向量化计算
        positions_array = np.array(positions, dtype=np.float64)
        triangles_array = np.array(triangles, dtype=np.int32)

        # 计算所有顶点到平面的距离
        distances = np.dot(positions_array - plane_origin, plane_normal)

        # 获取三角形三个顶点的距离
        d0 = distances[triangles_array[:, 0]]
        d1 = distances[triangles_array[:, 1]]
        d2 = distances[triangles_array[:, 2]]

        # 预筛选：只处理穿过平面的三角形
        signs = np.sign([d0, d1, d2])
        crosses_plane = ~(
                (signs[0] == signs[1]) & (signs[1] == signs[2])
        )

        valid_indices = np.where(crosses_plane)[0]

        # 获取穿过平面的三角形
        for idx in valid_indices:
            tri = triangles_array[idx]
            tri_d = [d0[idx], d1[idx], d2[idx]]
            tri_p = [positions_array[tri[0]], positions_array[tri[1]], positions_array[tri[2]]]

            edges = [
                (tri_p[0], tri_p[1], tri_d[0], tri_d[1]),
                (tri_p[1], tri_p[2], tri_d[1], tri_d[2]),
                (tri_p[2], tri_p[0], tri_d[2], tri_d[0])
            ]

            intersection_points = []

            for pa, pb, da, db in edges:
                if da * db < 0:
                    t = da / (da - db)
                    intersection = pa + t * (pb - pa)
                    intersection_points.append(intersection)

            if len(intersection_points) == 2:
                intersection_segments.append((intersection_points[0], intersection_points[1]))

        return intersection_segments

    def _compute_current_cross_section(self):
        """计算当前探头切面与乳腺+结节的交线（分离版本）"""
        try:
            plane_origin, plane_normal, probe_pos, probe_quat = self._get_scan_plane_transform()

            if plane_origin is None:
                self.current_cross_section = None
                self.current_probe_transform = None
                return

            self.current_probe_transform = {
                'position': probe_pos,
                'quaternion': probe_quat,
                'plane_origin': plane_origin,
                'plane_normal': plane_normal
            }

            # ============================================================
            # ✅ 新增拦截器：获取完实时平面的位置后，开始拦截耗时的交点计算！
            # ============================================================
            self.cross_section_update_counter += 1
            if self.cross_section_update_counter < self.cross_section_update_interval:
                if self.current_cross_section is not None:  # 如果是刚开机(None)，强制算一次
                    return  # 返回！不再执行下方的复杂求交线，避免卡顿
            self.cross_section_update_counter = 0
            # ============================================================
            monitor.cross_begin()# ── ✅ 新增：切面计算计时开始（只在真正计算时才计时）──
            # ============================================================
            # ✅ 关键修改：分别存储乳腺和结节的线段
            # ============================================================
            breast_segments = []
            lesion_segments = []
            # ============================================================

            # ============================================================
            # Step 1: 计算乳腺的交线
            # ============================================================
            breast_node = self.root_node.getChild(self.model_name)
            if breast_node:
                collision_node = breast_node.getChild(f"Collision{self.model_name}")
                if collision_node:
                    collision_mech = collision_node.getObject("breast_collision_state")
                    topo = collision_node.getObject("topology")

                    if (collision_mech and hasattr(collision_mech, 'position') and
                            topo and hasattr(topo, 'triangles')):

                        positions = collision_mech.position.value
                        triangles = topo.triangles.value

                        breast_segments = self._compute_mesh_plane_intersection(
                            positions, triangles, plane_origin, plane_normal
                        )

                        if not hasattr(self, '_breast_cross_logged'):
                            print(f"✓ 乳腺切面: {len(breast_segments)} 条线段")
                            self._breast_cross_logged = True

            # ============================================================
            # ✅ Step 2: 计算结节的交线（分开存储）
            # ============================================================
            if breast_node:
                lesion_node = breast_node.getChild("Lesion1")

                if not lesion_node:
                    for child in breast_node.children:
                        if "Lesion" in child.getName():
                            lesion_node = child
                            break

                if lesion_node:
                    mech = lesion_node.getObject("LesionMecha")
                    if not mech:
                        mech = lesion_node.getObject("FiducialMecha")

                    if mech and hasattr(mech, 'position'):
                        topo = None
                        for topo_name in ["topology", "Topology"]:
                            topo = lesion_node.getObject(topo_name)
                            if topo and hasattr(topo, 'triangles'):
                                break

                        if topo and hasattr(topo, 'triangles'):
                            lesion_positions = mech.position.value
                            lesion_triangles = topo.triangles.value

                            if len(lesion_triangles) > 0 and len(lesion_positions) > 0:
                                lesion_segments = self._compute_mesh_plane_intersection(
                                    lesion_positions, lesion_triangles,
                                    plane_origin, plane_normal
                                )

                                if not hasattr(self, '_lesion_cross_logged'):
                                    print(f"✓ 结节切面: {len(lesion_segments)} 条线段")
                                    self._lesion_cross_logged = True
            # ============================================================

            # ============================================================
            # ✅ Step 3: 分别存储（而不是合并）
            # ============================================================
            self.current_cross_section = {
                'breast': breast_segments,
                'lesion': lesion_segments
            }
            # ── ✅ 新增：切面计算计时结束 ─────────────────────────
            monitor.cross_end()

            if not hasattr(self, '_total_cross_logged'):
                total = len(breast_segments) + len(lesion_segments)
                print(f"✓ 总交线段数: {total} (乳腺: {len(breast_segments)}, 结节: {len(lesion_segments)})")
                self._total_cross_logged = True
            # ============================================================

        except Exception as e:
            monitor.cross_end() # ← 异常时也要结束计时
            self.current_cross_section = None

    def render_sofa_scene(self):
        """渲染 SOFA 场景"""
        self.render_breast()  # 使用实时位置 + 缓存法线
        self.render_probe()  # 使用缓存模型 + 实时位姿
        self.render_lesion()

        if self.show_scan_plane and self.current_probe_transform:
            self.draw_scan_plane_3d()

    # ============================================================
    # ✅ 正确的乳腺渲染：实时位置 + 缓存法线
    # ============================================================
    def render_breast(self):
        """乳腺渲染（VBO版本：实时位置 + 缓存法线）"""
        try:
            breast_node = self.root_node.getChild(self.model_name)
            if not breast_node:
                return
            collision_node = breast_node.getChild(f"Collision{self.model_name}")
            if not collision_node:
                return
            collision_mech = collision_node.getObject("breast_collision_state")
            topo = collision_node.getObject("topology")
            if not (collision_mech and hasattr(collision_mech, 'position') and
                    topo and hasattr(topo, 'triangles')):
                return

            positions = collision_mech.position.value
            triangles = topo.triangles.value
            if len(triangles) == 0 or len(positions) == 0:
                return

            # ── 低频更新法线 ──────────────────────────────────────────────────
            self.breast_normals_update_counter += 1
            need_normal_update = (
                    self.breast_normals_cached is None or
                    self.breast_normals_update_counter >= self.breast_normals_update_interval
            )
            if need_normal_update:
                self.breast_normals_cached = self._compute_vertex_normals_fast(
                    positions, triangles)
                self.breast_normals_update_counter = 0

            # ── VBO初始化（只执行一次）────────────────────────────────────────
            if not self._breast_vbo_initialized:
                self._breast_vbo_id, self._breast_ibo_id, self._breast_index_count = \
                    self._create_vbo(positions, triangles,
                                     self.breast_normals_cached, dynamic=True)
                self._breast_vbo_initialized = True
                print(f"✓ 乳腺VBO已创建: {len(positions)}顶点 "
                      f"{len(triangles)}三角形")
            else:
                # ── 每帧更新顶点位置（乳腺会形变）────────────────────────────
                self._update_vbo_data(self._breast_vbo_id,
                                      positions,
                                      self.breast_normals_cached)

            # ── 材质设置 ──────────────────────────────────────────────────────
            glDisable(GL_COLOR_MATERIAL)
            glEnable(GL_LIGHTING)
            glEnable(GL_BLEND)
            glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

            if self.model_name == "Liver":
                color = [0.45, 0.03, 0.03, 0.99]  # dark blood red
            else:
                color = [0.957, 0.730, 0.582, 0.99]  # skin tone
            glMaterialfv(GL_FRONT_AND_BACK, GL_AMBIENT, color)
            glMaterialfv(GL_FRONT_AND_BACK, GL_DIFFUSE, color)
            glMaterialfv(GL_FRONT_AND_BACK, GL_SPECULAR, [0.3, 0.3, 0.3, 1.0])
            glMaterialf(GL_FRONT_AND_BACK, GL_SHININESS, 20.0)

            # ── VBO绘制（替代原来数万次的glVertex调用）───────────────────────
            self._draw_vbo(self._breast_vbo_id,
                           self._breast_ibo_id,
                           self._breast_index_count)

            glDisable(GL_BLEND)
            glDisable(GL_LIGHTING)

        except Exception as e:
            print(f"render_breast error: {e}")
            import traceback
            traceback.print_exc()

    def _compute_vertex_normals_fast(self, positions, triangles):
        """快速计算顶点法向量（向量化版本）"""
        num_vertices = len(positions)
        vertex_normals = np.zeros((num_vertices, 3), dtype=np.float32)

        positions_array = np.array(positions, dtype=np.float32)
        triangles_array = np.array(triangles, dtype=np.int32)

        # 获取三角形的三个顶点
        v0 = positions_array[triangles_array[:, 0]]
        v1 = positions_array[triangles_array[:, 1]]
        v2 = positions_array[triangles_array[:, 2]]

        # 计算边向量
        edge1 = v1 - v0
        edge2 = v2 - v0

        # 计算面法线
        face_normals = np.cross(edge1, edge2)

        # 归一化
        norms = np.linalg.norm(face_normals, axis=1, keepdims=True)
        norms[norms < 1e-10] = 1.0
        face_normals = face_normals / norms

        # 累加到顶点
        np.add.at(vertex_normals, triangles_array[:, 0], face_normals)
        np.add.at(vertex_normals, triangles_array[:, 1], face_normals)
        np.add.at(vertex_normals, triangles_array[:, 2], face_normals)

        # 再次归一化
        norms = np.linalg.norm(vertex_normals, axis=1, keepdims=True)
        norms[norms < 1e-10] = 1.0
        vertex_normals = vertex_normals / norms

        return vertex_normals

    # ============================================================
    # ✅ 正确的探头渲染：缓存静态模型 + 实时位姿
    # ============================================================
    def render_probe(self):
        """探头渲染（缓存静态模型 + 实时位姿）"""
        try:
            probe_node = self.root_node.getChild("Probe")
            if not probe_node:
                return

            mech = probe_node.getObject("probe_state")
            if not mech or not hasattr(mech, 'position'):
                return

            # ============================================================
            # ✅ 读取实时位姿
            # ============================================================
            pose = mech.position.value[0]
            pos = pose[:3]
            quat = pose[3:]
            # ============================================================

            glPushMatrix()
            glTranslatef(pos[0], pos[1], pos[2])

            rot_matrix = self.quaternion_to_matrix(quat)
            glMultMatrixf(rot_matrix)

            # ============================================================
            # ✅ 只在第一次加载探头模型（然后缓存）
            # ============================================================
            if not self.probe_mesh_cached:
                visual_node = None
                for name in ["VisualProbe", "ProbeVisual", "Visual"]:
                    visual_node = probe_node.getChild(name)
                    if visual_node:
                        break

                if visual_node:
                    loader = visual_node.getObject("probe_visual_loader")
                    if loader and hasattr(loader, 'position') and hasattr(loader, 'triangles'):
                        self.probe_positions = loader.position.value
                        self.probe_triangles = loader.triangles.value

                        # 预计算法线（探头不变形，法线固定）
                        self.probe_normals = self._compute_probe_normals(
                            self.probe_positions,
                            self.probe_triangles
                        )

                        self.probe_mesh_cached = True
                        print(f"✓ 探头模型已缓存: {len(self.probe_positions)} 顶点, {len(self.probe_triangles)} 三角形")
            # ============================================================

            # ============================================================
            # ✅ 渲染探头模型（VBO版本）
            # ============================================================
            if self.probe_mesh_cached and self.probe_positions is not None:

                # 探头模型静态，只在首次创建VBO
                if not self._probe_vbo_initialized:
                    self._probe_vbo_id, self._probe_ibo_id, self._probe_index_count = \
                        self._create_vbo(self.probe_positions,
                                         self.probe_triangles,
                                         self.probe_normals,
                                         dynamic=False)  # ← 静态，不需要每帧更新
                    self._probe_vbo_initialized = True
                    print(f"✓ 探头VBO已创建: {len(self.probe_positions)}顶点 "
                          f"{len(self.probe_triangles)}三角形")

                glEnable(GL_LIGHTING)
                glMaterialfv(GL_FRONT_AND_BACK, GL_AMBIENT, [0.7, 0.7, 0.7, 1.0])
                glMaterialfv(GL_FRONT_AND_BACK, GL_DIFFUSE, [0.7, 0.7, 0.7, 1.0])
                glMaterialfv(GL_FRONT_AND_BACK, GL_SPECULAR, [0.6, 0.6, 0.6, 1.0])
                glMaterialf(GL_FRONT_AND_BACK, GL_SHININESS, 64.0)

                self._draw_vbo(self._probe_vbo_id,
                               self._probe_ibo_id,
                               self._probe_index_count)

                glDisable(GL_LIGHTING)

            else:
                # 备用方案：绘制简单立方体（不变）
                glEnable(GL_LIGHTING)
                size = 0.02
                glMaterialfv(GL_FRONT_AND_BACK, GL_AMBIENT, [0.7, 0.7, 0.7, 1.0])
                glMaterialfv(GL_FRONT_AND_BACK, GL_DIFFUSE, [0.7, 0.7, 0.7, 1.0])
                self.draw_cube_with_normals(size)
                glDisable(GL_LIGHTING)
            # ===========================================================

            glPopMatrix()

        except Exception as e:
            pass

    def _compute_probe_normals(self, positions, triangles):
        """计算探头法线（只计算一次）"""
        num_vertices = len(positions)
        vertex_normals = np.zeros((num_vertices, 3), dtype=np.float32)

        for tri in triangles:
            if tri[0] < num_vertices and tri[1] < num_vertices and tri[2] < num_vertices:
                p0 = np.array(positions[tri[0]])
                p1 = np.array(positions[tri[1]])
                p2 = np.array(positions[tri[2]])

                v1 = p1 - p0
                v2 = p2 - p0
                face_normal = np.cross(v1, v2)

                norm = np.linalg.norm(face_normal)
                if norm > 1e-10:
                    face_normal = face_normal / norm
                    vertex_normals[tri[0]] += face_normal
                    vertex_normals[tri[1]] += face_normal
                    vertex_normals[tri[2]] += face_normal

        for i in range(num_vertices):
            norm = np.linalg.norm(vertex_normals[i])
            if norm > 1e-10:
                vertex_normals[i] = vertex_normals[i] / norm

        return vertex_normals

    def draw_scan_plane_3d(self):
        """在3D视图中绘制半透明切面"""
        try:
            transform = self.current_probe_transform
            if not transform:
                return

            probe_pos = transform['position']
            probe_quat = transform['quaternion']

            glPushMatrix()

            glTranslatef(probe_pos[0], probe_pos[1], probe_pos[2])

            rot_matrix = self.quaternion_to_matrix(probe_quat)
            glMultMatrixf(rot_matrix)

            glEnable(GL_BLEND)
            glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
            glDisable(GL_LIGHTING)

            size = self.scan_plane_size
            z_offset = self.scan_plane_offset

            glColor4f(0.3, 0.7, 1.0, 0.2)

            glBegin(GL_QUADS)
            glVertex3f(-size, -size, z_offset)
            glVertex3f(+size, -size, z_offset)
            glVertex3f(+size, +size, z_offset)
            glVertex3f(-size, +size, z_offset)
            glEnd()

            glColor4f(0.3, 0.7, 1.0, 0.6)
            glLineWidth(2.0)

            glBegin(GL_LINE_LOOP)
            glVertex3f(-size, -size, z_offset)
            glVertex3f(+size, -size, z_offset)
            glVertex3f(+size, +size, z_offset)
            glVertex3f(-size, +size, z_offset)
            glEnd()

            glDisable(GL_BLEND)
            glPopMatrix()

        except Exception as e:
            pass

    def render_lesion(self):
        """Render lesion (breast model only)"""
        try:
            model_node = self.root_node.getChild(self.model_name)
            if not model_node:
                return

            # Debug: only print once
            if not hasattr(self, '_lesion_structure_logged'):
                print(f"\n{'='*60}")
                print(f"Lesion check for {self.model_name}:")
                print("=" * 60)

                lesion_node = model_node.getChild("Lesion1")
                if lesion_node:
                    print(f"  [OK] Found: {lesion_node.getName()}")
                else:
                    for child in model_node.children:
                        if "Lesion" in child.getName():
                            lesion_node = child
                            print(f"  [OK] Found: {child.getName()}")
                            break

                if lesion_node:
                    for mech_name in ["LesionMecha", "FiducialMecha"]:
                        mech = lesion_node.getObject(mech_name)
                        if mech and hasattr(mech, 'position'):
                            print(f"  mech obj: {mech_name}, {len(mech.position.value)} verts")
                    for topo_name in ["topology", "Topology"]:
                        topo = lesion_node.getObject(topo_name)
                        if topo and hasattr(topo, 'triangles'):
                            print(f"  topology: {topo_name}, {len(topo.triangles.value)} tris")
                else:
                    print(f"  No lesion for {self.model_name} model")
                print("=" * 60 + "\n")
                self._lesion_structure_logged = True

            # Find lesion node
            lesion_node = model_node.getChild("Lesion1")
            if not lesion_node:
                for child in model_node.children:
                    if "Lesion" in child.getName():
                        lesion_node = child
                        break
            if not lesion_node:
                return  # No lesion for this model

            # Get mechanical object
            # ============================================================
            mech = lesion_node.getObject("LesionMecha")
            if not mech:
                mech = lesion_node.getObject("FiducialMecha")

            if not mech or not hasattr(mech, 'position'):
                return

            positions = mech.position.value

            # ============================================================
            # Step 2: 获取拓扑
            # ============================================================
            topo = None
            for topo_name in ["topology", "Topology", "triangles"]:
                topo = lesion_node.getObject(topo_name)
                if topo and hasattr(topo, 'triangles'):
                    break

            # ============================================================
            # 情况A：有 STL 模型
            # ============================================================
            if topo and hasattr(topo, 'triangles'):
                triangles = topo.triangles.value

                if len(triangles) == 0 or len(positions) == 0:
                    return

                LESION_3D_SCALE = 2.0  # 范围 0.5-3.0

                lesion_center = np.mean(positions, axis=0)
                positions_scaled = lesion_center + (positions - lesion_center) * LESION_3D_SCALE

                positions = positions_scaled

                if not hasattr(self, '_lesion_scale_logged'):
                    lesion_size = (positions.max(axis=0) - positions.min(axis=0)) * 1000  # 转为 mm
                    print(
                        f"✓ 结节 3D 尺寸: {lesion_size[0]:.1f} x {lesion_size[1]:.1f} x {lesion_size[2]:.1f} mm (缩放 {LESION_3D_SCALE}x)")
                    self._lesion_scale_logged = True
                # ============================================================

                # 法线缓存
                if not hasattr(self, '_lesion_normals_cache'):
                    self._lesion_normals_cache = None
                    self._lesion_normals_update_counter = 0

                self._lesion_normals_update_counter += 1

                if (self._lesion_normals_cache is None or
                        self._lesion_normals_update_counter >= 3):
                    self._lesion_normals_cache = self._compute_vertex_normals_fast(
                        positions, triangles
                    )
                    self._lesion_normals_update_counter = 0

                    if not hasattr(self, '_lesion_render_logged'):
                        print(f"✓ 结节 STL 渲染: {len(positions)} 顶点, {len(triangles)} 三角形")
                        self._lesion_render_logged = True

                vertex_normals = self._lesion_normals_cache

                # 材质设置
                glEnable(GL_LIGHTING)
                glEnable(GL_BLEND)
                glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

                color = [0.8, 0.6, 0.0, 0.99]
                glMaterialfv(GL_FRONT_AND_BACK, GL_AMBIENT, color)
                glMaterialfv(GL_FRONT_AND_BACK, GL_DIFFUSE, color)
                glMaterialfv(GL_FRONT_AND_BACK, GL_SPECULAR, [0.5, 0.5, 0.5, 1.0])
                glMaterialf(GL_FRONT_AND_BACK, GL_SHININESS, 32.0)

                # ✅ VBO渲染结节
                if not self._lesion_vbo_initialized:
                    self._lesion_vbo_id, self._lesion_ibo_id, self._lesion_index_count = \
                        self._create_vbo(positions, triangles,
                                         vertex_normals, dynamic=True)
                    self._lesion_vbo_initialized = True
                    print(f"✓ 结节VBO已创建: {len(positions)}顶点 "
                          f"{len(triangles)}三角形")
                else:
                    # 结节会随乳腺形变，每帧更新位置
                    self._update_vbo_data(self._lesion_vbo_id,
                                          positions,
                                          vertex_normals)

                self._draw_vbo(self._lesion_vbo_id,
                               self._lesion_ibo_id,
                               self._lesion_index_count)

                glDisable(GL_BLEND)
                glDisable(GL_LIGHTING)

            # ============================================================
            # 情况B：只有位置点
            # ============================================================
            elif len(positions) > 0:
                if not hasattr(self, '_lesion_point_logged'):
                    print(f"✓ 结节点渲染: {len(positions)} 个点")
                    self._lesion_point_logged = True

                center = np.mean(positions, axis=0)

                glDisable(GL_LIGHTING)
                glPushMatrix()
                glTranslatef(center[0], center[1], center[2])

                glColor3f(1.0, 0.0, 0.0)
                glPointSize(15.0)
                glBegin(GL_POINTS)
                glVertex3f(0, 0, 0)
                glEnd()

                glEnable(GL_BLEND)
                glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
                glColor4f(1.0, 0.5, 0.5, 0.3)
                glPointSize(25.0)
                glBegin(GL_POINTS)
                glVertex3f(0, 0, 0)
                glEnd()
                glDisable(GL_BLEND)

                glPopMatrix()

        except Exception as e:
            pass

    def quaternion_to_matrix(self, quat):
        """四元数转OpenGL旋转矩阵"""
        qx, qy, qz, qw = quat

        matrix = [
            1 - 2 * qy * qy - 2 * qz * qz, 2 * qx * qy + 2 * qz * qw, 2 * qx * qz - 2 * qy * qw, 0,
            2 * qx * qy - 2 * qz * qw, 1 - 2 * qx * qx - 2 * qz * qz, 2 * qy * qz + 2 * qx * qw, 0,
            2 * qx * qz + 2 * qy * qw, 2 * qy * qz - 2 * qx * qw, 1 - 2 * qx * qx - 2 * qy * qy, 0,
            0, 0, 0, 1
        ]

        return matrix

    def draw_cube_with_normals(self, size):
        """绘制带法向量的立方体"""
        glBegin(GL_QUADS)

        glNormal3f(0, 0, 1)
        glVertex3f(-size, -size, size)
        glVertex3f(size, -size, size)
        glVertex3f(size, size, size)
        glVertex3f(-size, size, size)

        glNormal3f(0, 0, -1)
        glVertex3f(-size, -size, -size)
        glVertex3f(-size, size, -size)
        glVertex3f(size, size, -size)
        glVertex3f(size, -size, -size)

        glNormal3f(0, 1, 0)
        glVertex3f(-size, size, -size)
        glVertex3f(-size, size, size)
        glVertex3f(size, size, size)
        glVertex3f(size, size, -size)

        glNormal3f(0, -1, 0)
        glVertex3f(-size, -size, -size)
        glVertex3f(size, -size, -size)
        glVertex3f(size, -size, size)
        glVertex3f(-size, -size, size)

        glNormal3f(1, 0, 0)
        glVertex3f(size, -size, -size)
        glVertex3f(size, size, -size)
        glVertex3f(size, size, size)
        glVertex3f(size, -size, size)

        glNormal3f(-1, 0, 0)
        glVertex3f(-size, -size, -size)
        glVertex3f(-size, -size, size)
        glVertex3f(-size, size, size)
        glVertex3f(-size, size, -size)

        glEnd()

    # ============================================================
    # ✅ 新增：左上角姿态小视口
    # ============================================================

    def _get_probe_quaternion(self):
        """从SOFA节点读取探头四元数，失败返回单位四元数"""
        try:
            probe_node = self.root_node.getChild("Probe")
            if not probe_node:
                return np.array([0.0, 0.0, 0.0, 1.0])
            mech = probe_node.getObject("probe_state")
            if not mech or not hasattr(mech, 'position'):
                return np.array([0.0, 0.0, 0.0, 1.0])
            pose = mech.position.value[0]
            quat = np.array(pose[3:], dtype=np.float64)  # [qx, qy, qz, qw]
            # 归一化防止数值漂移
            norm = np.linalg.norm(quat)
            if norm > 1e-8:
                quat = quat / norm
            return quat
        except Exception:
            return np.array([0.0, 0.0, 0.0, 1.0])

    def _quat_to_rotation_matrix(self, quat):
        """四元数转3×3旋转矩阵（numpy）"""
        qx, qy, qz, qw = quat
        R = np.array([
            [1 - 2 * (qy ** 2 + qz ** 2), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
            [2 * (qx * qy + qz * qw), 1 - 2 * (qx ** 2 + qz ** 2), 2 * (qy * qz - qx * qw)],
            [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx ** 2 + qy ** 2)]
        ], dtype=np.float64)
        return R

    def _draw_pose_overlay(self):
        """
        在3D画面左上角绘制姿态小视口：
        - 半透明深色背景
        - 彩色坐标轴（X红 Y绿 Z蓝）
        - 灰色长方体（模拟手柄）
        所有内容跟随探头实时旋转
        """
        W = self.width()
        H = self.height()
        S = self.overlay_size  # 小视口尺寸
        M = self.overlay_margin  # 边距

        # 小视口左下角（OpenGL坐标系原点在左下）
        vp_x = M
        vp_y = H - M - S

        # ── 读取探头四元数 ────────────────────────────────────────────────
        quat = self._get_probe_quaternion()

        # ── 保存完整GL状态 ────────────────────────────────────────────────
        glPushAttrib(GL_ALL_ATTRIB_BITS)

        # ── 切换到小视口 ──────────────────────────────────────────────────
        glViewport(vp_x, vp_y, S, S)

        # ── 设置小视口的投影矩阵 ──────────────────────────────────────────
        glMatrixMode(GL_PROJECTION)
        glPushMatrix()
        glLoadIdentity()
        gluPerspective(45.0, 1.0, 0.01, 100.0)

        glMatrixMode(GL_MODELVIEW)
        glPushMatrix()
        glLoadIdentity()

        # ── 清除小视口深度缓冲（不清颜色，背景用半透明quad覆盖）─────────
        glClear(GL_DEPTH_BUFFER_BIT)

        # ── 绘制半透明背景 ────────────────────────────────────────────────
        glMatrixMode(GL_PROJECTION)
        glPushMatrix()
        glLoadIdentity()
        glOrtho(0, 1, 0, 1, -1, 1)
        glMatrixMode(GL_MODELVIEW)
        glPushMatrix()
        glLoadIdentity()

        glDisable(GL_DEPTH_TEST)
        glDisable(GL_LIGHTING)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

        # 圆角感深色背景
        glColor4f(0.08, 0.08, 0.12, 0.75)
        glBegin(GL_QUADS)
        glVertex2f(0, 0)
        glVertex2f(1, 0)
        glVertex2f(1, 1)
        glVertex2f(0, 1)
        glEnd()

        # 细边框
        glColor4f(0.4, 0.6, 1.0, 0.6)
        glLineWidth(1.5)
        glBegin(GL_LINE_LOOP)
        glVertex2f(0.01, 0.01)
        glVertex2f(0.99, 0.01)
        glVertex2f(0.99, 0.99)
        glVertex2f(0.01, 0.99)
        glEnd()

        glMatrixMode(GL_PROJECTION)
        glPopMatrix()
        glMatrixMode(GL_MODELVIEW)
        glPopMatrix()

        # ── 恢复透视投影，开始绘制3D内容 ─────────────────────────────────
        glMatrixMode(GL_PROJECTION)
        glPopMatrix()
        glMatrixMode(GL_MODELVIEW)
        glPopMatrix()

        # 重新设置视图矩阵
        glMatrixMode(GL_PROJECTION)
        glPushMatrix()
        glLoadIdentity()
        gluPerspective(45.0, 1.0, 0.01, 100.0)

        glMatrixMode(GL_MODELVIEW)
        glPushMatrix()
        glLoadIdentity()

        glEnable(GL_DEPTH_TEST)
        glClear(GL_DEPTH_BUFFER_BIT)

        # 相机退后
        glTranslatef(0.0, 0.0, -0.35)

        # ✅ 在任何旋转之前设置光源（固定在世界坐标系）
        glEnable(GL_LIGHTING)
        glEnable(GL_LIGHT0)
        glDisable(GL_COLOR_MATERIAL)
        glLightfv(GL_LIGHT0, GL_AMBIENT, [0.4, 0.4, 0.4, 1.0])
        glLightfv(GL_LIGHT0, GL_DIFFUSE, [1.0, 1.0, 1.0, 1.0])
        glLightfv(GL_LIGHT0, GL_SPECULAR, [0.6, 0.6, 0.6, 1.0])
        # ✅ 多个光源方向，确保正面始终有光
        # 主光：从左上前方打来
        glLightfv(GL_LIGHT0, GL_POSITION, [0.0, 0.0, 1.0, 0.0])

        # ✅ 补光：增加 LIGHT1 从右侧补光，避免死黑
        glEnable(GL_LIGHT1)
        glLightfv(GL_LIGHT1, GL_AMBIENT, [0.2, 0.2, 0.2, 1.0])
        glLightfv(GL_LIGHT1, GL_DIFFUSE, [0.5, 0.5, 0.5, 1.0])
        glLightfv(GL_LIGHT1, GL_SPECULAR, [0.0, 0.0, 0.0, 1.0])
        glLightfv(GL_LIGHT1, GL_POSITION, [1.0, 0.0, 0.5, 0.0])

        # 固定视角旋转（光源已设置完毕，旋转不影响光源）
        glRotatef(90.0, 1.0, 0.0, 0.0)

        # 应用探头旋转
        R = self._quat_to_rotation_matrix(quat)
        gl_mat = [
            R[0, 0], R[1, 0], R[2, 0], 0,
            R[0, 1], R[1, 1], R[2, 1], 0,
            R[0, 2], R[1, 2], R[2, 2], 0,
            0, 0, 0, 1
        ]
        glMultMatrixf(gl_mat)

        # ── Step2：在旋转后的坐标系中绘制坐标轴 ─────────────────────────
        # 坐标轴跟随探头旋转，XYZ直接对应Omega6的物理轴
        glDisable(GL_BLEND)
        glDisable(GL_LIGHTING)
        glLineWidth(2.5)

        axis_len = 0.13
        origin = np.array([0.0, 0.0, 0.0])

        # X轴 → 红色
        glColor3f(1.0, 0.2, 0.2)
        glBegin(GL_LINES)
        glVertex3f(*origin)
        glVertex3f(axis_len, 0, 0)
        glEnd()
        self._draw_axis_arrow(np.array([axis_len, 0, 0]),
                              np.array([1, 0, 0]), 0.025,
                              (1.0, 0.2, 0.2))

        # Y轴 → 绿色
        glColor3f(0.2, 1.0, 0.2)
        glBegin(GL_LINES)
        glVertex3f(*origin)
        glVertex3f(0, axis_len, 0)
        glEnd()
        self._draw_axis_arrow(np.array([0, axis_len, 0]),
                              np.array([0, 1, 0]), 0.025,
                              (0.2, 1.0, 0.2))

        # Z轴 → 蓝色
        glColor3f(0.3, 0.5, 1.0)
        glBegin(GL_LINES)
        glVertex3f(*origin)
        glVertex3f(0, 0, axis_len)
        glEnd()
        self._draw_axis_arrow(np.array([0, 0, axis_len]),
                              np.array([0, 0, 1]), 0.025,
                              (0.3, 0.5, 1.0))

        # ── Step3：叠加视觉补偿旋转，只影响手柄模型外观 ─────────────────
        # 这些旋转只是为了让模型看起来方向正确
        # 不影响坐标轴，不影响实际运动映射
        #glRotatef(180.0, 1.0, 0.0, 0.0)
        #glRotatef(-90.0, 0.0, 0.0, 1.0) #绿色
        #glRotatef(180.0, 0.0, 1.0, 0.0)

        # ── Step4：绘制手柄模型 ───────────────────────────────────────────
        glEnable(GL_LIGHTING)

        if self._handle_dl is None:
            self._handle_dl = glGenLists(1)
            glNewList(self._handle_dl, GL_COMPILE)
            self._draw_handle_box()
            glEndList()
            print("✓ 手柄Display List已编译")

        glCallList(self._handle_dl)
        glDisable(GL_LIGHT1)
        glDisable(GL_LIGHTING)

        # ── 恢复主视口状态 ────────────────────────────────────────────────
        glMatrixMode(GL_PROJECTION)
        glPopMatrix()
        glMatrixMode(GL_MODELVIEW)
        glPopMatrix()

        glPopAttrib()

        # 恢复主视口尺寸
        glViewport(0, 0, W, H)

        # 恢复主视口投影矩阵
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        aspect = W / (H if H > 0 else 1)
        gluPerspective(45, aspect, 0.01, 100.0)
        glMatrixMode(GL_MODELVIEW)

    def _draw_handle_box(self):
        """
        绘制模拟Omega6手柄：
        长轴沿Y轴（绿色），底座在Y轴负方向，顶端接触头在Y轴正方向
        """
        import math

        def draw_cylinder_y(radius, height, y_start, n_sides=20,
                            color_diffuse=None, color_ambient=None,
                            flat_front=False, flat_radius=None):
            """
            绘制沿Y轴方向的圆柱体
            y_start: 起始Y坐标
            height:  沿Y轴的高度
            flat_front: 是否在Z轴正方向切平面
            flat_radius: 切平面到Y轴的距离
            """
            if color_diffuse is None:
                color_diffuse = [0.55, 0.55, 0.60, 1.0]
            if color_ambient is None:
                color_ambient = [0.25, 0.25, 0.28, 1.0]

            glMaterialfv(GL_FRONT_AND_BACK, GL_AMBIENT, color_ambient)
            glMaterialfv(GL_FRONT_AND_BACK, GL_DIFFUSE, color_diffuse)
            glMaterialfv(GL_FRONT_AND_BACK, GL_SPECULAR, [0.7, 0.7, 0.7, 1.0])
            glMaterialf(GL_FRONT_AND_BACK, GL_SHININESS, 60.0)

            y_end = y_start + height

            def get_verts(y):
                verts = []
                for i in range(n_sides):
                    angle = 2 * math.pi * i / n_sides
                    x = radius * math.cos(angle)
                    z = radius * math.sin(angle)
                    # 切平面：Z轴正方向截断
                    if flat_front and flat_radius is not None:
                        if z > flat_radius:
                            z = flat_radius
                    verts.append((x, y, z))
                return verts

            verts_bot = get_verts(y_start)
            verts_top = get_verts(y_end)

            # ── 侧面 ──────────────────────────────────────────────────
            glBegin(GL_QUAD_STRIP)
            for i in range(n_sides + 1):
                idx = i % n_sides
                vb = verts_bot[idx]
                vt = verts_top[idx]
                nx = (vb[0] + vt[0]) / 2.0
                nz = (vb[2] + vt[2]) / 2.0
                nlen = math.sqrt(nx * nx + nz * nz)
                if nlen > 1e-8:
                    nx /= nlen
                    nz /= nlen
                glNormal3f(nx, 0, nz)
                glVertex3f(*verts_bot[idx])
                glVertex3f(*verts_top[idx])
            glEnd()

            # ── 底面（Y轴负方向）──────────────────────────────────────
            glNormal3f(0, -1, 0)
            glBegin(GL_TRIANGLE_FAN)
            glVertex3f(0, y_start, 0)
            for v in reversed(verts_bot):
                glVertex3f(*v)
            glVertex3f(*verts_bot[-1])
            glEnd()

            # ── 顶面（Y轴正方向）──────────────────────────────────────
            glNormal3f(0, 1, 0)
            glBegin(GL_TRIANGLE_FAN)
            glVertex3f(0, y_end, 0)
            for v in verts_top:
                glVertex3f(*v)
            glVertex3f(*verts_top[0])
            glEnd()

            # ── 切平面（Z轴正方向平面）────────────────────────────────
            if flat_front and flat_radius is not None:
                glNormal3f(0, 0, 1)
                glBegin(GL_QUADS)
                glVertex3f(-radius, y_start, flat_radius)
                glVertex3f(radius, y_start, flat_radius)
                glVertex3f(radius, y_end, flat_radius)
                glVertex3f(-radius, y_end, flat_radius)
                glEnd()

        # ── 整体居中 ──────────────────────────────────────────────────
        glPushMatrix()
        glTranslatef(0.0, -0.08, 0.0)  # 沿Y轴居中偏移

        # ── 底座（Y轴最下方）─────────────────────────────────────────
        draw_cylinder_y(
            radius=0.055, height=0.045, y_start=0.1,
            n_sides=24,
            color_diffuse=[0.45, 0.45, 0.50, 1.0],
            color_ambient=[0.20, 0.20, 0.22, 1.0],
            flat_front=False
        )

        # ── 握柄（中间细长）──────────────────────────────────────────
        draw_cylinder_y(
            radius=0.028, height=0.130, y_start=-0.018,
            n_sides=20,
            color_diffuse=[0.50, 0.50, 0.55, 1.0],
            color_ambient=[0.22, 0.22, 0.25, 1.0],
            flat_front=True, flat_radius=0.015
        )

        # ── 顶端接触头（Y轴最上方）───────────────────────────────────
        draw_cylinder_y(
            radius=0.022, height=0.018, y_start=0.112,
            n_sides=16,
            color_diffuse=[0.20, 0.20, 0.25, 1.0],
            color_ambient=[0.10, 0.10, 0.12, 1.0],
            flat_front=True, flat_radius=0.012
        )

        glPopMatrix()

    def _draw_axis_arrow(self, tip_pos, direction, size, color):
        """
        在坐标轴末端绘制箭头锥体

        Args:
            tip_pos:   锥尖位置 (np.array)
            direction: 轴方向单位向量 (np.array)
            size:      锥体大小
            color:     (r, g, b)
        """
        glDisable(GL_LIGHTING)
        glColor3f(*color)

        # 找两个与direction垂直的基向量
        if abs(direction[0]) < 0.9:
            perp1 = np.cross(direction, [1, 0, 0])
        else:
            perp1 = np.cross(direction, [0, 1, 0])
        perp1 = perp1 / (np.linalg.norm(perp1) + 1e-8)
        perp2 = np.cross(direction, perp1)

        base_center = tip_pos - direction * size
        n_sides = 8
        angles = np.linspace(0, 2 * np.pi, n_sides, endpoint=False)

        base_pts = [
            base_center + (np.cos(a) * perp1 + np.sin(a) * perp2) * size * 0.4
            for a in angles
        ]

        # 锥侧面
        glBegin(GL_TRIANGLE_FAN)
        glVertex3f(*tip_pos)
        for pt in base_pts:
            glVertex3f(*pt)
        glVertex3f(*base_pts[0])  # 闭合
        glEnd()

        # 锥底面
        glBegin(GL_TRIANGLE_FAN)
        glVertex3f(*base_center)
        for pt in reversed(base_pts):
            glVertex3f(*pt)
        glVertex3f(*base_pts[-1])
        glEnd()

    # ============================================================
    # ✅ 姿态小视口结束
    # ============================================================

    def mousePressEvent(self, event):
        self.last_pos = event.pos()

    def mouseMoveEvent(self, event):
        if self.last_pos:
            dx = event.x() - self.last_pos.x()
            dy = event.y() - self.last_pos.y()

            if event.buttons() & Qt.LeftButton:
                self.rotation_x += dy * 0.5
                self.rotation_y -= dx * 0.5
                self.update()

            self.last_pos = event.pos()

    def wheelEvent(self, event):
        delta = event.angleDelta().y()
        self.zoom += delta * 0.001
        self.zoom = max(-5.0, min(-0.1, self.zoom))
        self.update()


# ============================================================
# 超声图像显示窗口（保持不变）
# ============================================================
class UltrasoundWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)

        # ============================================================
        # ✅ 使用网格布局黑科技实现“层叠悬浮”效果
        # ============================================================
        from PyQt5.QtWidgets import QGridLayout
        self.layout = QGridLayout(self)
        self.layout.setContentsMargins(0, 0, 0, 0)  # 取消所有边距，让画面顶格

        # 1. 底层：超声图像 (设为纯黑背景)
        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setMinimumSize(400, 400)  # 稍微减小最小高度限制
        self.image_label.setStyleSheet("background-color: black;")

        # 2. 顶层上方：标题字 (添加了半透明黑色背景框，防止看不清)
        self.label_title = QLabel("Real-time Ultrasound Stream")
        self.label_title.setAlignment(Qt.AlignCenter)
        self.label_title.setStyleSheet("""
            color: white; 
            font-weight: bold; 
            font-size: 14px; 
            background-color: rgba(0, 0, 0, 150); /* 半透明黑底 */
            border-radius: 5px;
            padding: 4px 10px;
            margin-top: 10px;
        """)

        # 3. 顶层下方：探头信息字
        self.info_label = QLabel("Probe Position: N/A")
        self.info_label.setAlignment(Qt.AlignCenter)
        self.info_label.setStyleSheet("""
            color: #00ffcc; /* 改成了更科技感的青色 */
            font-size: 12px; 
            font-weight: bold;
            background-color: rgba(0, 0, 0, 180);
            border-radius: 5px;
            padding: 4px 10px;
            margin-bottom: 10px;
        """)

        # === 核心布局逻辑：把它们放在同一个列中 ===
        # 图像占满 3行1列
        self.layout.addWidget(self.image_label, 0, 0, 3, 1)
        # 文字分别停靠在顶部和底部，居中显示
        self.layout.addWidget(self.label_title, 0, 0, Qt.AlignTop | Qt.AlignHCenter)
        self.layout.addWidget(self.info_label, 2, 0, Qt.AlignBottom | Qt.AlignHCenter)

        # ============================================================
        # 初始化超声渲染器 (保持不变)
        # ============================================================
        self.config = UltrasoundConfig()
        self.rng = np.random.default_rng(12345)

        self._tex_cache = None
        self._speckle_cache = None
        self._speckle_cache_time = 0

        # ── GAN后台推理线程 ──
        print("[GAN] Loading model (CUDA init, may take 10-30s)...")
        from gan_worker import GANWorker
        self.gan_worker = GANWorker(
            model_path="checkpoints/generator_best.pth",
            image_size=256
        )
        self.gan_worker.start()
        print("[GAN] Model loaded, worker thread started")

        self._last_gan_frame = None
        self._last_valid_lesion_segments = []  # 缓存最后一次有效结节截面
        self._last_valid_probe_transform = None  # 缓存对应的探头变换

        print("✓ 超声渲染器 + GAN后台线程 初始化完成")

    def update_image(self, probe_pos=None, cross_section_segments=None,
                     probe_transform=None):
        """更新超声图像（只执行一遍逻辑）"""

        breast_vertices_2d = None
        raw_lesion_segments = []
        raw_breast_segments = []

        # ── Step1: 提取截面数据 ───────────────────────────────────────────────
        if cross_section_segments and probe_transform:
            try:
                if isinstance(cross_section_segments, dict):
                    breast_segments = cross_section_segments.get('breast', [])
                    lesion_segments = cross_section_segments.get('lesion', [])
                else:
                    breast_segments = cross_section_segments
                    lesion_segments = []

                raw_breast_segments = breast_segments
                raw_lesion_segments = lesion_segments

                # 乳腺2D轮廓（仅用于接触检测）
                if breast_segments and len(breast_segments) > 0:
                    breast_contour_3d = self._order_contour_segments(breast_segments)
                    if breast_contour_3d is not None and len(breast_contour_3d) >= 3:
                        breast_vertices_2d = self._project_contour_to_probe_2d(
                            breast_contour_3d, probe_transform, is_lesion=False
                        )

            except Exception as e:
                print(f"⚠️ 轮廓处理失败: {e}")

        # ── Step2: 接触检测 ───────────────────────────────────────────────────
        probe_is_scanning = self._check_probe_contact(
            breast_vertices_2d, cross_section_segments
        )

        # ── Step3: 提交GAN推理 ────────────────────────────────────────────────
        if probe_is_scanning and len(raw_breast_segments) > 0:

            # 保留缓存更新（供外部查询最后一次有效结节位置，不用于生成mask）
            if len(raw_lesion_segments) > 0:
                self._last_valid_lesion_segments = raw_lesion_segments
                self._last_valid_probe_transform = probe_transform

            # ✅ 核心修复：只提交当前帧真实截到的结节线段
            # raw_lesion_segments=[] 时，GAN生成无结节的正常乳腺组织超声图
            # 不再使用缓存兜底，彻底杜绝"未相交却显示结节mask"的问题
            self.gan_worker.submit_raw(raw_lesion_segments, probe_transform)

        elif not probe_is_scanning:
            # 完全离开乳腺时清除
            self._last_gan_frame = None
            self._last_valid_lesion_segments = []
            self._last_valid_probe_transform = None
            self.gan_worker.clear_output()

        # ── Step3.5: 结节消失时主动清除旧GAN帧（避免异步延迟导致残影）─────────
        # 背景：GAN推理是异步的，结节消失后新的无结节推理结果需要1~2帧才能到达
        #       在这段延迟内，_last_gan_frame 仍是含结节的旧帧
        #       通过检测"有结节→无结节"的状态切换，主动清除旧帧，强制等待新结果
        current_has_lesion = len(raw_lesion_segments) > 0
        if not hasattr(self, '_prev_has_lesion'):
            self._prev_has_lesion = False
        if self._prev_has_lesion and not current_has_lesion:
            self._last_gan_frame = None  # 强制清除，等待新的无结节推理结果
        self._prev_has_lesion = current_has_lesion
        # ─────────────────────────────────────────────────────────────────────

        # ── Step4: 取结果 ─────────────────────────────────────────────────────
        new_frame = self.gan_worker.get_latest_result()
        if new_frame is not None:
            self._last_gan_frame = new_frame

        # ── Step5: 显示 ───────────────────────────────────────────────────────
        try:
            if probe_is_scanning and self._last_gan_frame is not None:
                display_arr = self._postprocess_gan_output(self._last_gan_frame)
            else:
                cache = self._prepare_texture_grid()
                H, W = cache["H"], cache["W"]
                display_arr = np.zeros((H, W), dtype=np.uint8)

            h, w = display_arr.shape
            q_img = QImage(display_arr.tobytes(), w, h, w,
                           QImage.Format_Grayscale8)
            pixmap = QPixmap.fromImage(q_img).scaled(
                self.image_label.width(),
                self.image_label.height(),
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation
            )
            self.image_label.setPixmap(pixmap)

            if probe_pos is not None:
                stats = self.gan_worker.get_stats()
                breast_count = len(breast_vertices_2d) if breast_vertices_2d is not None else 0
                lesion_count = len(raw_lesion_segments)
                gan_status = f"{stats['last_ms']:.0f}ms" if stats['loaded'] else "未加载"
                scan_status = "🟢扫查中" if probe_is_scanning else "⚫未接触"

                # ✅ 状态文字：只反映当前帧真实情况，不显示"缓存"
                lesion_src = "实时" if lesion_count > 0 else "无"

                self.info_label.setText(
                    f"Probe: ({probe_pos[0]:.3f}, {probe_pos[1]:.3f}, "
                    f"{probe_pos[2]:.3f})\n"
                    f"{scan_status} | Breast:{breast_count}pts | "
                    f"Lesion:{lesion_src}({lesion_count}段) | GAN:{gan_status}"
                )
            else:
                self.info_label.setText("Probe Position: N/A")

        except Exception as e:
            print(f"⚠️ 显示失败: {e}")

    def _check_probe_contact(self, breast_vertices_2d, cross_section_segments):
        """
        判断探头是否真正接触并穿过乳腺

        根据日志分析：
        - 面积比计算因坐标系不一致导致数值异常（可能>1）
        - 改为只用线段数量判断，更可靠
        """
        if cross_section_segments is None:
            return False

        breast_segs = (cross_section_segments.get('breast', [])
                       if isinstance(cross_section_segments, dict)
                       else cross_section_segments)

        seg_count = len(breast_segs)

        # ✅ 只用线段数量判断
        # 日志显示：正常扫查时 71~88 条线段
        # 未接触时：0 条线段
        # 阈值设为 20，给足够的裕量
        MIN_SEGMENTS = 20

        if seg_count < MIN_SEGMENTS:
            return False

        # ✅ 额外检查：2D轮廓点数量（而不是面积）
        if breast_vertices_2d is None or len(breast_vertices_2d) < 10:
            return False

        return True

    def _postprocess_gan_output(self, gan_arr_256):
        """
        对GAN输出的256×256灰度图做后处理：
        1. 缩放到显示分辨率（512×512）
        2. 应用扇形蒙版（扇形外变黑）
        3. Gamma校正

        Args:
            gan_arr_256: (256,256) uint8

        Returns:
            (512,512) uint8
        """
        from PIL import Image as PILImage

        cache   = self._prepare_texture_grid()
        sector  = cache["sector"]       # (512,512) bool
        H, W    = cache["H"], cache["W"]

        # 256 → 512 双线性缩放
        pil_256  = PILImage.fromarray(gan_arr_256, mode='L')
        pil_512  = pil_256.resize((W, H), PILImage.BILINEAR)
        arr_512  = np.array(pil_512, dtype=np.float32) / 255.0

        # 扇形蒙版：扇形外置黑
        arr_512  = np.where(sector, arr_512, 0.0)

        # Gamma校正（与原代码一致）
        arr_512  = np.clip(arr_512, 0, 1) ** self.config.GAMMA

        return (arr_512 * 255).astype(np.uint8)

    def _render_simple_fallback(self, probe_pos):
        """简单降级渲染（备用）"""
        width, height = 512, 512
        img = Image.new('L', (width, height), color=0)

        img_array = np.array(img)
        q_img = QImage(img_array.data, width, height, width, QImage.Format_Grayscale8)
        pixmap = QPixmap.fromImage(q_img)
        self.image_label.setPixmap(pixmap)

        # ✅ 修复：这里也要改
        if probe_pos is not None:
            self.info_label.setText(
                f"Probe: X={probe_pos[0]:.3f}, Y={probe_pos[1]:.3f}, Z={probe_pos[2]:.3f}\n"
                "Render Failed"
            )
        else:
            self.info_label.setText("Probe Position: N/A\nRender Failed")

    def _project_segments_to_probe_plane(self, segments, transform, img_width, img_height):
        """将3D线段投影到2D图像空间"""
        try:
            plane_origin = transform['plane_origin']
            probe_quat = transform['quaternion']

            local_x = self._rotate_vector_by_quat(np.array([1, 0, 0]), probe_quat)
            local_y = self._rotate_vector_by_quat(np.array([0, 1, 0]), probe_quat)

            lines_2d = []
            scan_range = 0.08

            for seg in segments:
                p0_world, p1_world = seg

                p0_local = p0_world - plane_origin
                p1_local = p1_world - plane_origin

                p0_x = np.dot(p0_local, local_x)
                p0_y = np.dot(p0_local, local_y)

                p1_x = np.dot(p1_local, local_x)
                p1_y = np.dot(p1_local, local_y)

                u0 = (p0_x / scan_range + 0.5) * img_width
                v0 = (0.5 + p0_y / scan_range) * img_height

                u1 = (p1_x / scan_range + 0.5) * img_width
                v1 = (0.5 + p1_y / scan_range) * img_height

                if (0 <= u0 < img_width and 0 <= v0 < img_height and
                        0 <= u1 < img_width and 0 <= v1 < img_height):
                    lines_2d.append(((int(u0), int(v0)), (int(u1), int(v1))))

            return lines_2d

        except Exception as e:
            return []

    def _rotate_vector_by_quat(self, vec, quat):
        """用四元数旋转向量"""
        qx, qy, qz, qw = quat
        vx, vy, vz = vec

        t = 2.0 * np.array([
            qy * vz - qz * vy,
            qz * vx - qx * vz,
            qx * vy - qy * vx
        ])

        result = vec + qw * t + np.array([
            qy * t[2] - qz * t[1],
            qz * t[0] - qx * t[2],
            qx * t[1] - qy * t[0]
        ])

        return result

    def _prepare_texture_grid(self):
        """准备纹理坐标网格（只计算一次）"""
        if self._tex_cache is not None:
            return self._tex_cache

        W, H = self.config.TEX_W, self.config.TEX_H

        # 坐标系定义（保持不变）
        x_range = 0.1  # 扫描宽度 100mm
        y_range = 0.15  # 扫描深度 150mm

        xs = np.linspace(-x_range / 2, x_range / 2, W)
        ys = np.linspace(0, y_range, H)
        X, Y = np.meshgrid(xs, ys)

        # 极坐标（保留，供外部可能使用）
        r = np.sqrt(X ** 2 + Y ** 2)
        theta = np.degrees(np.arctan2(np.abs(X), Y))

        # ====================================================================
        # ✅ 修改：矩形蒙版（替换原来的扇形蒙版）
        # ====================================================================

        # 矩形边界参数（单位：米）
        rect_x_half = x_range / 2  # 左右边界：±50mm
        rect_y_top = 0.005  # 上边界：5mm（保留顶部小间距）
        rect_y_bottom = y_range  # 下边界：150mm

        # 矩形蒙版：满足范围内的像素为True
        sector = (
                (np.abs(X) <= rect_x_half) &  # 左右范围
                (Y >= rect_y_top) &  # 上边界
                (Y <= rect_y_bottom)  # 下边界
        )

        # ====================================================================

        # 调试输出
        if not hasattr(self, '_sector_logged'):
            print(f"\n{'=' * 60}")
            print(f"📐 矩形显示区域参数:")
            print(f"  宽度: {rect_x_half * 2 * 1000:.0f}mm "
                  f"(-{rect_x_half * 1000:.0f}mm ~ +{rect_x_half * 1000:.0f}mm)")
            print(f"  深度: {rect_y_top * 1000:.0f}mm ~ {rect_y_bottom * 1000:.0f}mm")
            print(f"  有效像素: {sector.sum()} "
                  f"({sector.sum() / sector.size * 100:.1f}%)")
            print(f"{'=' * 60}\n")
            self._sector_logged = True

        self._tex_cache = dict(W=W, H=H, X=X, Y=Y, r=r, theta=theta, sector=sector)
        return self._tex_cache


    def _order_contour_segments(self, segments):
        """
        将无序线段列表排序为连续轮廓

        Args:
            segments: [(p0, p1), (p2, p3), ...] 3D 线段列表

        Returns:
            np.ndarray: (N, 3) 有序轮廓点
        """
        if not segments or len(segments) == 0:
            return None

        try:
            # 构建邻接表
            from collections import defaultdict

            # 将点转换为元组（可哈希）
            def point_to_tuple(p):
                return (round(p[0], 6), round(p[1], 6), round(p[2], 6))

            adjacency = defaultdict(list)

            for p0, p1 in segments:
                key0 = point_to_tuple(p0)
                key1 = point_to_tuple(p1)
                adjacency[key0].append(p1)
                adjacency[key1].append(p0)

            # 查找起点（度为1的点，如果没有则任选一点）
            start_point = None
            for point, neighbors in adjacency.items():
                if len(neighbors) == 1:
                    start_point = np.array(point)
                    break

            if start_point is None:
                # 闭合轮廓，任选起点
                start_point = np.array(list(adjacency.keys())[0])

            # 遍历构建有序轮廓
            ordered_contour = [start_point]
            visited = {point_to_tuple(start_point)}
            current = start_point

            max_iterations = len(segments) * 2 + 10
            iterations = 0

            while len(visited) < len(adjacency) and iterations < max_iterations:
                iterations += 1
                current_key = point_to_tuple(current)
                neighbors = adjacency[current_key]

                # 找到未访问的邻居
                next_point = None
                for neighbor in neighbors:
                    neighbor_key = point_to_tuple(neighbor)
                    if neighbor_key not in visited:
                        next_point = neighbor
                        break

                if next_point is None:
                    break

                ordered_contour.append(next_point)
                visited.add(point_to_tuple(next_point))
                current = next_point

            if len(ordered_contour) < 3:
                return None

            return np.array(ordered_contour)

        except Exception as e:
            print(f"⚠️ 轮廓排序失败: {e}")
            return None

    def _project_contour_to_probe_2d(self, contour_3d, probe_transform,is_lesion=False):
        """
        将 3D 轮廓投影到探头局部 2D 坐标系（带居中和缩放）

        Args:
            contour_3d: (N, 3) 世界坐标系的轮廓点
            probe_transform: dict

        Returns:
            (N, 2) 探头局部坐标系的 2D 点
        """
        if contour_3d is None or len(contour_3d) == 0:
            return None

        try:
            probe_pos = np.array(probe_transform['position'])
            probe_quat = np.array(probe_transform['quaternion'])

            # 构建旋转矩阵
            qx, qy, qz, qw = probe_quat
            R = np.array([
                [1 - 2 * (qy ** 2 + qz ** 2), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
                [2 * (qx * qy + qz * qw), 1 - 2 * (qx ** 2 + qz ** 2), 2 * (qy * qz - qx * qw)],
                [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx ** 2 + qy ** 2)]
            ])

            # 变换到探头局部坐标系
            points_centered = contour_3d - probe_pos
            points_local = points_centered @ R.T

            # 投影到 XZ 平面
            points_2d = points_local[:, [0, 2]].copy()  # X=左右, Z=深度

            # Z 轴翻转（SOFA的Z向上，超声深度向下）
            points_2d[:, 1] = -points_2d[:, 1]

            # ====================================================================
            # ✅ 步骤1：左右居中（修复偏左问题）
            # ====================================================================
            x_center = (points_2d[:, 0].min() + points_2d[:, 0].max()) / 2
            x_offset = 0 - x_center  # 自动居中
            # x_offset += 0.005  # ← 手动向右偏移5mm（负数向左）
            points_2d[:, 0] = points_2d[:, 0] + x_offset

            # ====================================================================
            # ✅ 步骤2：缩放（分别处理乳腺和结节）
            # ====================================================================
            x_range_current = points_2d[:, 0].max() - points_2d[:, 0].min()
            y_range_current = points_2d[:, 1].max() - points_2d[:, 1].min()

            # ============================================================
            # 超声画面缩放参数 分开调节
            # ============================================================
            if is_lesion:
                # 结节的实际显示尺寸
                x_range_target = 0.03  # 30mm（设置结节大小）
                y_range_target = 0.03  # 30mm
                scale_margin = 1.0  # 1.0 = 真实大小，1.5 = 放大 1.5 倍
            else:
                # 乳腺的目标尺寸（覆盖扇形）
                x_range_target = 0.4  # 100mm 按比例缩放xy
                y_range_target = 0.6  # 150mm
                scale_margin = 0.9  # 留 10% 边距
            # ============================================================

            scale_x = x_range_target / (x_range_current + 1e-8)
            scale_y = y_range_target / (y_range_current + 1e-8)
            scale = min(scale_x, scale_y) * scale_margin

            points_2d = points_2d * scale

            # ====================================================================
            # ✅ 步骤3：深度居中（让轮廓在扇形中垂直居中）
            # ====================================================================
            y_center = (points_2d[:, 1].min() + points_2d[:, 1].max()) / 2
            target_y_center = 0.10  # 目标深度中心100mm（扇形深度的一半）
            y_offset = target_y_center - y_center
            points_2d[:, 1] = points_2d[:, 1] + y_offset
            # ====================================================================

            # 调试输出
            # 替换为：只在首次输出
            if not hasattr(self, '_proj_debug_logged'):
                print(f"[首次投影] 原始尺寸: X={x_range_current * 1000:.1f}mm "
                      f"Y={y_range_current * 1000:.1f}mm | 缩放: {scale:.2f}x")
                self._proj_debug_logged = True

            return points_2d

        except Exception as e:
            print(f"⚠️ 坐标投影失败: {e}")
            import traceback
            traceback.print_exc()
            return None

# ============================================================
# 主窗口 - 修复预定义路径结束检测
# ============================================================
# ============================================================
# ============================================================
# 参数面板 — 3D 画面左上角，姿态小窗口下方
# ============================================================
class ParameterPanel(QWidget):
    """半透明参数面板，实时调节仿真参数"""

    def __init__(self, parent, controller=None):
        super().__init__(parent)
        self.controller = controller
        self.setFixedWidth(185)
        self.setStyleSheet("""
            QWidget {
                background: rgba(18, 18, 28, 0.88);
                color: #ddd;
                font-size: 11px;
                border-radius: 6px;
            }
            QLabel#section_label {
                color: #4ec9ff;
                font-weight: bold;
                font-size: 12px;
                padding: 3px 4px;
            }
            QSlider::groove:horizontal {
                height: 4px; background: #444;
                border-radius: 2px;
            }
            QSlider::handle:horizontal {
                width: 10px; height: 14px;
                background: #4ec9ff; border-radius: 3px;
                margin: -5px 0;
            }
            QPushButton {
                background: #3a3a4a; color: #ddd;
                border: 1px solid #555; border-radius: 3px;
                padding: 3px 6px; font-size: 12px;
                min-width: 22px; min-height: 22px;
            }
            QPushButton:hover { background: #4a4a5a; border-color: #4ec9ff; }
            QPushButton:pressed { background: #2a2a3a; }
            QLabel#val_label {
                color: #aaa; font-size: 10px;
                min-width: 38px;
            }
        """)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 4, 6, 4)
        layout.setSpacing(4)

        # ── Section 1: Material ──
        lbl_mat = QLabel("Materials")
        lbl_mat.setObjectName("section_label")
        layout.addWidget(lbl_mat)

        self._add_slider(layout, "Young", 500, 10000, 3000, self._on_young)  # E: 500-10000 Pa
        self._add_slider(layout, "Density", 500, 2000, 1075, self._on_density)
        self._add_slider(layout, "Poisson", 10, 49, 45, self._on_poisson)  # 0.10-0.49

        # ── Section 2: Haptics ──
        lbl_hap = QLabel("Haptics")
        lbl_hap.setObjectName("section_label")
        layout.addWidget(lbl_hap)

        self._add_slider(layout, "Speed", 10, 100, 60, self._on_speed)    # 0.1-1.0
        self._add_slider(layout, "ForceGain", 10, 500, 300, self._on_force_gain)
        self._add_slider(layout, "MaxForce", 5, 100, 20, self._on_max_force)  # 0.5-10.0N
        self._add_slider(layout, "ContactK", 50, 100000, 30000, self._on_contact_k)  # contact stiffness

        # ── Section 3: Model Shift ──
        lbl_pos = QLabel("Position")
        lbl_pos.setObjectName("section_label")
        layout.addWidget(lbl_pos)

        self.shift_step = 0.005  # 每次位移步长 (m)
        self._add_shift_buttons(layout)

    # ── Slider helper ────────────────────────────────────────────────
    def _add_slider(self, layout, name, vmin, vmax, vdefault, callback):
        row = QHBoxLayout()
        lbl = QLabel(name)
        lbl.setFixedWidth(52)
        row.addWidget(lbl)

        slider = QSlider(Qt.Horizontal)
        slider.setRange(vmin, vmax)
        slider.setValue(vdefault)
        slider.valueChanged.connect(callback)
        row.addWidget(slider)

        val_lbl = QLabel(str(vdefault))
        val_lbl.setObjectName("val_label")
        row.addWidget(val_lbl)

        # Store reference for updating label
        setattr(self, f"_s_{name}", slider)
        setattr(self, f"_v_{name}", val_lbl)

        layout.addLayout(row)

    # ── Camera pan buttons ────────────────────────────────────────────
    def _add_shift_buttons(self, layout):
        gl = self.parent()  # SofaGLWidget
        step = 0.02  # pan step per click

        # Row 1: Up (stretched wide)
        r1 = QHBoxLayout()
        btn_u = QPushButton("Up")
        btn_u.setMinimumWidth(76)
        btn_u.clicked.connect(lambda: self._pan(gl, 0, step))
        r1.addWidget(btn_u)
        layout.addLayout(r1)

        # Row 2: Left, Down, Right
        r2 = QHBoxLayout()
        btn_l = QPushButton("L")
        btn_l.clicked.connect(lambda: self._pan(gl, -step, 0))
        r2.addWidget(btn_l)
        btn_d = QPushButton("Dn")
        btn_d.clicked.connect(lambda: self._pan(gl, 0, -step))
        r2.addWidget(btn_d)
        btn_r = QPushButton("R")
        btn_r.clicked.connect(lambda: self._pan(gl, step, 0))
        r2.addWidget(btn_r)
        layout.addLayout(r2)

    # ── Callbacks ────────────────────────────────────────────────────
    def _on_young(self, val):
        self._v_Young.setText(str(val))
        if self.controller and self.controller.breast:
            self.controller.breast.set_young_modulus(float(val))

    def _on_density(self, val):
        self._v_Density.setText(str(val))
        if self.controller and self.controller.breast:
            self.controller.breast.set_density(float(val))

    def _on_poisson(self, val):
        nu = val / 100.0
        self._v_Poisson.setText(f"{nu:.2f}")
        if self.controller and self.controller.breast:
            self.controller.breast.set_poisson_ratio(nu)

    def _on_speed(self, val):
        s = val / 100.0
        self._v_Speed.setText(f"{s:.2f}")
        if self.controller and self.controller.probe:
            self.controller.probe.position_amplification = s

    def _on_force_gain(self, val):
        self._v_ForceGain.setText(str(val))
        if self.controller and self.controller.probe:
            self.controller.probe.force_gain_deform = float(val)

    def _on_max_force(self, val):
        n = val / 10.0
        self._v_MaxForce.setText(f"{n:.1f}")
        if self.controller and self.controller.probe:
            self.controller.probe.max_force = n

    def _on_contact_k(self, val):
        self._v_ContactK.setText(str(val))
        try:
            cm = self.controller.root.getObject("ContactManager")
            if cm and hasattr(cm, 'responseParams'):
                cm.responseParams.value = f'stiffness={val}'
        except Exception:
            pass

    def _pan(self, gl, dx, dy):
        """Pan the 3D camera view (dx,dy = screen direction)"""
        if gl:
            gl.translate_x -= dx
            gl.translate_y -= dy
            gl.update()


# 主窗口 - 联动画图界面 (Qt Designer) 版本
# ============================================================
class MainApp(QMainWindow, Ui_MainWindow):
    def __init__(self):
        super().__init__()
        # 1. 初始化你在 Qt Designer 里画的界面！
        self.setupUi(self)

        self.setWindowTitle("Breast Biopsy Simulation & Ultrasound")

        self.root = None
        self.controller = None
        self.timer = QTimer()
        self.timer.timeout.connect(self.simulation_step)
        self.is_paused = False
        self.use_omega6 = False
        self.simulation_ended = False

        # 2. 设置状态栏
        self.statusBar_widget = QStatusBar()
        self.setStatusBar(self.statusBar_widget)
        self.statusBar_widget.showMessage("Ready - click Start to begin")

        # 3. 将自定义的 3D视图 和 超声视图 塞进咱们画的占位框里
        self.setup_custom_widgets()

        # 4. 绑定按钮点击事件
        self.connect_signals()

        # 初始化按钮状态
        self.btn_pause.setEnabled(False)
        self.btn_stop.setEnabled(False)

    def setup_custom_widgets(self):
        """将自定义组件注入到 UI 的容器中"""
        # ================= 右侧：塞入超声视图 =================
        self.us_view = UltrasoundWidget()
        # 给右侧容器加个布局，把超声画面填进去
        us_layout = QVBoxLayout(self.container_us)
        us_layout.setContentsMargins(0, 0, 0, 0)
        us_layout.addWidget(self.us_view)

        # ================= 左侧：塞入 3D 占位提示 =================
        self.sofa_view_placeholder = QLabel(
            "Click Start to launch simulation\n\n"
            "Omega6 Haptic: force-feedback control\n"
            "Predefined Path: automatic scanning"
        )
        self.sofa_view_placeholder.setAlignment(Qt.AlignCenter)
        self.sofa_view_placeholder.setStyleSheet(
            "background-color: #2a2a2a; color: #888; font-size: 16px; border: 2px solid #555;"
        )
        # 给左侧容器加个布局，把提示字填进去
        self.sofa_layout = QVBoxLayout(self.container_sofa)
        self.sofa_layout.setContentsMargins(0, 0, 0, 0)
        self.sofa_layout.addWidget(self.sofa_view_placeholder)

        # ================= 工具栏：动态添加模型选择下拉框 =================
        self.label_model = QLabel(" Model:")
        self.label_model.setStyleSheet("color: #ccc; font-size: 13px; font-weight: bold;")
        self.combo_model = QComboBox()
        self.combo_model.setMinimumSize(140, 36)
        self.combo_model.setStyleSheet(
            "QComboBox { background: #333; color: white; border: 1px solid #555; "
            "padding: 4px 8px; border-radius: 4px; }"
            "QComboBox:hover { border-color: #4ec9ff; }"
            "QComboBox::drop-down { border: none; }"
            "QComboBox QAbstractItemView { background: #333; color: white; "
            "selection-background-color: #4ec9ff; }"
        )
        self.combo_model.addItem("Breast", "./input_parameters.yml")
        self.combo_model.addItem("Liver", "./inData/input_parameters_liver.yml")
        # 插入到 btn_stop 之后、spacer 之前
        self.horizontalLayout.insertWidget(4, self.label_model)
        self.horizontalLayout.insertWidget(5, self.combo_model)

    def connect_signals(self):
        """绑定按钮功能"""
        # 这里的 btn_start_omega 等名字，就是你在 Qt Designer 里改的 objectName
        self.btn_start_omega.clicked.connect(lambda: self.start_simulation(use_omega6=True))
        self.btn_start_path.clicked.connect(lambda: self.start_simulation(use_omega6=False))
        self.btn_pause.clicked.connect(self.toggle_pause)
        self.btn_stop.clicked.connect(self.stop_simulation)

    def start_simulation(self, use_omega6=False):
        try:
            mode_text = "Omega6 Haptic" if use_omega6 else "Predefined Path"
            self.statusBar_widget.showMessage(f"Initializing {mode_text}...")
            QApplication.processEvents()

            ensure_plugins()

            self.root = Sofa.Core.Node("root")

            # 从下拉框读取模型配置路径
            yml_path = self.combo_model.currentData()
            if not os.path.exists(yml_path):
                QMessageBox.critical(self, "Error", f"Config file not found: {yml_path}")
                return

            params = Parameters(yml_path)
            params.use_gui = False

            createScene(self.root, params, use_omega6)
            Sofa.Simulation.init(self.root)

            if hasattr(self.root, 'BreastProbe'):
                self.controller = self.root.BreastProbe
                model_name = self.controller.model_name
            else:
                raise Exception("BreastProbe controller not found")

            # ============================================================
            # ✅ 重点修改：用真实的 3D 渲染画面 替换掉 占位提示字
            # ============================================================
            if self.sofa_view_placeholder:
                self.sofa_view_placeholder.deleteLater()
                self.sofa_view_placeholder = None

            self.sofa_view = SofaGLWidget(self.root, model_name=model_name, controller=self.controller)
            self.sofa_layout.addWidget(self.sofa_view) # 将真实 3D 画面装进左侧容器
            # ============================================================

            QApplication.processEvents()

            self.use_omega6 = use_omega6
            self.simulation_ended = False

            self.timer.start(20)  # 50 Hz
            self.is_paused = False

            self.btn_start_omega.setEnabled(False)
            self.btn_start_path.setEnabled(False)
            self.btn_pause.setEnabled(True)
            self.btn_stop.setEnabled(True)
            self.combo_model.setEnabled(False)

            self.statusBar_widget.showMessage(f"Simulation running ({mode_text})")
            self.label_sys_info.setText(f"Simulation started ({mode_text}), scanning...")

        except Exception as e:
            QMessageBox.critical(self, "Startup Failed", f"Failed to start simulation:\n{str(e)}")
            self.statusBar_widget.showMessage("Startup failed")
            import traceback
            traceback.print_exc()

    def simulation_step(self):
        """仿真步骤"""
        if not self.root or self.simulation_ended:
            return

        # ── ✅ 新增：步骤计时开始（在try块之前！）────────────────
        monitor.step_begin()

        try:
            # 1. 检查预定义路径是否结束
            if not self.use_omega6:
                if self._check_predefined_path_ended():
                    self.simulation_ended = True
                    self._on_simulation_ended()
                    return

            # ── ✅ 新增：FEM计时开始 ───────────────────────────────
            monitor.fem_begin()

            # 2. 推进物理仿真
            Sofa.Simulation.animate(self.root, 0.02)

            # ── ✅ 新增：FEM计时结束（紧跟在animate后面！）────────
            monitor.fem_end()

            if hasattr(self, 'sofa_view'):
                self.sofa_view.update()

            # 3. 每 2 帧更新一次图像和【信息提示板】
            if hasattr(self, 'sofa_view') and self.sofa_view.frame_counter % 2 == 0:
                probe_pos = None
                cross_section = self.sofa_view.current_cross_section
                probe_transform = self.sofa_view.current_probe_transform

                if self.controller:
                    probe_pos = self.controller.get_probe_position()

                # 更新右侧超声图像
                self.us_view.update_image(probe_pos, cross_section, probe_transform)

                # ── ✅ 新增：从GAN worker读取推理时间 ─────────────
                if hasattr(self.us_view, 'gan_worker'):
                    gan_stats = self.us_view.gan_worker.get_stats()
                    monitor.record_gan(gan_stats['last_ms'])
                # ──────────────────────────────────────────────────

                # ==========================================================
                # 这里是我们新加的【更新中间提示条】的逻辑
                # ==========================================================

                # A. 获取扫描进度
                progress_text = "Free Control"
                if not self.use_omega6 and self.controller and hasattr(self.controller, 'probe'):
                    probe = self.controller.probe
                    if hasattr(probe, 'current_def') and hasattr(probe, 'num_deformations'):
                        # 获取当前走到第几个点 / 总共几个点
                        progress_text = f"Path step: {probe.current_def} / {probe.num_deformations}"

                # B. 判断是否扫查到了结节
                lesion_text = "No lesion detected"
                if cross_section and isinstance(cross_section, dict):
                    lesion_segments = cross_section.get('lesion', [])
                    if len(lesion_segments) > 0:  # 如果结节的切面线段大于0，说明切到了结节！
                        lesion_text = f"[Lesion Found] contour points: {len(lesion_segments)}"

                # C. 把两段文字合并，显示到界面上！
                self.label_sys_info.setText(f"{progress_text}    |    {lesion_text}")
                # ==========================================================
                # ── ✅ 新增：步骤计时结束（在try块末尾）──────────────
                monitor.step_end()
                # ──────────────────────────────────────────────────────

        except Exception as e:
            pass

    def _check_predefined_path_ended(self):
        try:
            if hasattr(self.controller, 'probe'):
                probe = self.controller.probe
                if hasattr(probe, 'current_def') and hasattr(probe, 'num_deformations'):
                    if probe.current_def > probe.num_deformations:
                        return True
                if hasattr(probe, 'is_path_completed') and probe.is_path_completed:
                    return True
            if hasattr(self.controller, 'breast'):
                breast = self.controller.breast
                if hasattr(breast, 'is_stable') and breast.is_stable:
                    if hasattr(self.controller, 'probe'):
                        probe = self.controller.probe
                        if hasattr(probe, 'current_def') and hasattr(probe, 'num_deformations'):
                            if probe.current_def > probe.num_deformations:
                                return True
            if hasattr(self.root, 'animate') and not self.root.animate:
                return True
            return False
        except Exception as e:
            return False

    def _on_simulation_ended(self):
        self.timer.stop()
        self.statusBar_widget.showMessage("Simulation completed")
        QMessageBox.information(self, "Done", "Predefined path simulation completed.\n\nData saved automatically.")
        self.btn_start_omega.setEnabled(True)
        self.btn_start_path.setEnabled(True)
        self.btn_pause.setEnabled(False)
        self.btn_stop.setEnabled(False)
        self.combo_model.setEnabled(True)
        self.label_sys_info.setText("Automatic scanning path completed.")

    def toggle_pause(self):
        if self.is_paused:
            self.timer.start(20)
            self.is_paused = False
            self.statusBar_widget.showMessage("Simulation resumed")
            self.btn_pause.setText("⏸ Pause")
        else:
            self.timer.stop()
            self.is_paused = True
            self.statusBar_widget.showMessage("Simulation paused")
            self.btn_pause.setText("▶ 继续")

    def stop_simulation(self):
        self.timer.stop()
        self.simulation_ended = True

        if self.controller and hasattr(self.controller, 'probe'):
            if hasattr(self.controller.probe, 'close_omega6'):
                self.controller.probe.close_omega6()

        self.btn_start_omega.setEnabled(True)
        self.btn_start_path.setEnabled(True)
        self.btn_pause.setEnabled(False)
        self.btn_stop.setEnabled(False)
        self.combo_model.setEnabled(True)

        self.statusBar_widget.showMessage("Simulation stopped")
        self.label_sys_info.setText("Simulation stopped. Select a mode and start again.")

    def closeEvent(self, event):
        print("\n🛑 正在关闭系统，准备释放资源...")

        # ── ✅ 新增：打印并保存最终性能报告 ──────────────────────
        monitor.print_final_report()
        # ──────────────────────────────────────────────────────────

        # 1. 停止UI定时器
        self.timer.stop()
        self.simulation_ended = True

        # 2. ✅ 使用子线程去释放硬件，防止卡死主UI线程
        def release_hardware():
            if self.controller and hasattr(self.controller, 'probe'):
                try:
                    if hasattr(self.controller.probe, 'close_omega6'):
                        self.controller.probe.close_omega6()
                        print("  ✓ Omega6硬件已释放")
                except Exception as e:
                    print(f"  ⚠️ Omega6释放异常: {e}")

        hw_thread = threading.Thread(target=release_hardware)
        hw_thread.daemon = True
        hw_thread.start()

        # ✅ 最多只等硬件释放1.0秒，如果1秒还没释放完，直接抛弃不管
        hw_thread.join(timeout=1.0)
        if hw_thread.is_alive():
            print("  ⚠️ Omega6释放超时，放弃等待！")

        # 3. 停止GAN线程
        if hasattr(self, 'us_view') and hasattr(self.us_view, 'gan_worker'):
            try:
                self.us_view.gan_worker.stop()
            except Exception:
                pass

        event.accept()
        print("💥 进程已强制结束！\n")
        # 4. 强杀进程，不给 C++ 库任何执行内存回收的机会（避免段错误卡死）
        os._exit(0)


if __name__ == '__main__':
    fmt = QSurfaceFormat()
    fmt.setVersion(2, 1)
    fmt.setProfile(QSurfaceFormat.CompatibilityProfile)
    fmt.setDepthBufferSize(24)
    fmt.setStencilBufferSize(8)
    fmt.setSamples(4)
    fmt.setSwapBehavior(QSurfaceFormat.DoubleBuffer)
    QSurfaceFormat.setDefaultFormat(fmt)

    app = QApplication(sys.argv)
    app.setStyle('Fusion')

    app.setStyleSheet("""
        QMainWindow { background-color: #2b2b2b; }
        QPushButton {
            background-color: #3c3c3c;
            color: white;
            border: 1px solid #555;
            border-radius: 4px;
            padding: 10px;
        }
        QPushButton:hover { background-color: #4c4c4c; }
        QPushButton:pressed { background-color: #2c2c2c; }
        QPushButton:disabled { background-color: #2b2b2b; color: #666; }
        QStatusBar { background-color: #3c3c3c; color: white; }
    """)

    print("[Startup] Creating window...")
    window = MainApp()
    print("[Startup] Window created, showing...")
    window.show()
    print("[Startup] Entering event loop...")

    print("\n" + "=" * 60)
    print("Breast Biopsy Simulation & Ultrasound")
    print("=" * 60)
    print("  Breast: real-time positions + cached normals")
    print("  Probe:  cached model + real-time pose")
    print("  Optimized: VBO rendering, numpy vectorization")
    print("=" * 60 + "\n")

    sys.exit(app.exec_())