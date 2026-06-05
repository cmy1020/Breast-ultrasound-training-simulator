import Sofa.Core
import Sofa.SofaDeformable

import numpy as np
from typing import Optional, List
import ctypes
import math

from components.tetrahedral import Topology, add_collision_models, add_loader, add_topology, add_visual_models


# ============================================================
# Omega6设备接口
# ============================================================
class Omega6Device:
    """Omega6力反馈设备接口"""

    def __init__(self, dll_path=r"E:\sdk-3.17.6\bin\dhd64.dll"):
        self.dhd = ctypes.CDLL(dll_path)
        self._setup_functions()

        self.device_id = self.dhd.dhdOpen()
        if self.device_id < 0:
            raise RuntimeError("无法打开Omega6设备")

        # 打开力输出
        self.dhd.dhdEnableForce(ctypes.c_int(1), self.device_id)
        print(f"✓ Omega6设备已连接 (ID: {self.device_id})")

        # 设备返回单位是 m（大部分 Force Dimension 设备），如是 mm 可再调整
        self.position_scale = 1.0  # 若实际为 mm，这里改成 1000.0

    def _setup_functions(self):
        self.dhd.dhdOpen.restype = ctypes.c_int

        self.dhd.dhdGetPosition.argtypes = [
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_double),
            ctypes.c_int
        ]
        self.dhd.dhdGetPosition.restype = ctypes.c_int

        self.dhd.dhdGetOrientationRad.argtypes = [
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_double),
            ctypes.c_int
        ]
        self.dhd.dhdGetOrientationRad.restype = ctypes.c_int

        self.dhd.dhdSetForce.argtypes = [
            ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_int
        ]
        self.dhd.dhdSetForce.restype = ctypes.c_int

        self.dhd.dhdEnableForce.argtypes = [ctypes.c_int, ctypes.c_int]
        self.dhd.dhdClose.argtypes = [ctypes.c_int]

    def get_position(self):
        """返回设备末端位置，单位 m"""
        x, y, z = ctypes.c_double(), ctypes.c_double(), ctypes.c_double()
        ret = self.dhd.dhdGetPosition(
            ctypes.byref(x), ctypes.byref(y), ctypes.byref(z), self.device_id
        )
        if ret < 0:
            return None
        return np.array([x.value, y.value, z.value]) / self.position_scale

    def get_orientation(self):
        """返回欧拉角 (roll, pitch, yaw)，单位 rad"""
        roll, pitch, yaw = ctypes.c_double(), ctypes.c_double(), ctypes.c_double()
        ret = self.dhd.dhdGetOrientationRad(
            ctypes.byref(roll), ctypes.byref(pitch), ctypes.byref(yaw), self.device_id
        )
        if ret < 0:
            return None
        return np.array([roll.value, pitch.value, yaw.value])

    def set_force(self, force):
        """输入力向量 [Fx, Fy, Fz]，单位 N"""
        self.dhd.dhdSetForce(
            ctypes.c_double(float(force[0])),
            ctypes.c_double(float(force[1])),
            ctypes.c_double(float(force[2])),
            self.device_id
        )

    def close(self):
        """关闭前清零力"""
        self.set_force(np.zeros(3))
        self.dhd.dhdClose(self.device_id)
        print("✓ Omega6设备已关闭")


def euler_to_quaternion(roll, pitch, yaw): # 欧拉角转四元数函数
    """欧拉角转四元数，返回 [qx, qy, qz, qw]"""
    cy, sy = np.cos(yaw * 0.5), np.sin(yaw * 0.5)
    cp, sp = np.cos(pitch * 0.5), np.sin(pitch * 0.5)
    cr, sr = np.cos(roll * 0.5), np.sin(roll * 0.5)

    qw = cr * cp * cy + sr * sp * sy
    qx = sr * cp * cy - cr * sp * sy
    qy = cr * sp * cy + sr * cp * sy
    qz = cr * cp * sy - sr * sp * cy

    return np.array([qx, qy, qz, qw])


# ============================================================
# Probe 类（支持 Omega6 & 预定义路径）
# ============================================================
class Probe(Sofa.Core.Controller):
    """ Sofa Controller representing an ultrasound probe """

    def __init__(
            self,
            root_node: Sofa.Core.Node,
            visual_filename: Optional[str] = None,
            collision_filename: Optional[str] = None,
            node_name: str = "Probe",
            init_pose: list = [0, 0, 0, 0, 0, 0, 1],
            target_position: np.ndarray = np.zeros((1, 3)),
            velocity: float = 0.01,
            use_omega6: bool = False
    ):
        """
        Args:
            root_node (Sofa.Core.Node): root node of the simulation.
            visual_filename (str, Optional): name of the triangular surface file used for visualization.
            collision_filename (str): name of the triangular surface file used for collision computation.
            node_name (str): name of the created node.
            init_pose (list): initial position and orientation of the probe, [x, y, z, qx, qy, qz, qw]
            target_position (np.ndarray): list of target positions for predefined motion
            velocity (float): probe velocity in predefined mode
            use_omega6 (bool): 是否使用Omega6设备控制
        """

        Sofa.Core.Controller.__init__(self)

        self.init_pose = init_pose
        self.current_pose = None
        self.motion_path: List[np.ndarray] = []
        self.velocity = velocity
        self.root_node = root_node
        self.use_omega6 = use_omega6
        self.current_def_ended = False

        # ============= Omega6 相关状态 =============
        self.omega6: Optional[Omega6Device] = None
        self.omega6_init_pos: Optional[np.ndarray] = None
        self.position_offset = np.zeros(3)

        # ============================================================
        # 力反馈参数（基于诊断数据精确标定）
        # 单位：m（SOFA场景和Omega6均为m）
        # max_disp 实测 ~ 0.006~0.02m
        # prox_raw 实测 ~ 1.3~2.1N（二次方刚度已经是N）
        # ============================================================

        # 接近力增益：prox_raw 已经是 ~2N，
        self.force_gain_contact = 1.0

        # 变形力增益：max_disp ~ 0.01m，期望产生 ~3N
        # 0.01 * 300 = 3N
        self.force_gain_deform = 300.0

        # 力输出上限：Omega6额定12N，设8N安全且手感明显
        self.max_force = 2.0

        # 死区：0.05N以下过滤（原来0.05会把0.08N的合力也过滤掉）
        self.force_deadzone = 0.02

        # 接近力弹簧参数
        # THRESHOLD：0.05m=5cm感应范围，实测最近距离~0.01m合理
        self._proximity_force = np.zeros(3)
        self.PROXIMITY_THRESHOLD = 0.001  # m，距离多近开始感应
        self.SPRING_STIFFNESS = 5.0  # N/m，提升刚度增加Q弹感
        # ============================================================

        # ✅ 新增：力输出平滑器
        # _smoothed_force 是上一帧实际输出给设备的力
        # 每帧向目标力靠近 (1 - FORCE_SMOOTH_ALPHA) 的比例
        self._smoothed_force   = np.zeros(3)   # 平滑后的力（上一帧输出）
        self.FORCE_SMOOTH_ALPHA = 0.80          # 平滑系数：越大越平滑，越小响应越快
                                                # 建议范围：0.7 ~ 0.92
        if self.use_omega6:
            try:
                self.omega6 = Omega6Device()
            except Exception as e:
                print(f"✗ Omega6初始化失败: {e}")
                print("  将使用预定义路径模式")
                self.use_omega6 = False

        # ✅ 新增：初始化数据记录文件（仅Omega6模式）
        self._data_file = None
        if self.use_omega6:
            import os
            from datetime import datetime
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename  = f"fx_data_{timestamp}.txt"
            self._data_file = open(filename, 'w', encoding='utf-8')
            # 写表头
            self._data_file.write("displacement_m\tforce_N\n")
            print(f"✓ 数据记录已启动 → {filename}")

        # ----------------- Probe 节点 -----------------
        probe_node = root_node.addChild(node_name)
        self.state = probe_node.addObject(
            'MechanicalObject',
            name='probe_state',
            position=init_pose,
            template="Rigid3d",
            showObject=1,
            showObjectScale=0.01,
            listening=1
        )
        # ========== 关键修复：添加质量 ==========
        probe_node.addObject(
            'UniformMass',
            name='probe_mass',
            totalMass=0.2,  # 探头质量 200g（典型超声探头重量）
        )

        # ========== 添加刚体固定约束（Omega6模式）==========
        if self.use_omega6:
            # Omega6 控制时，探头位置完全由设备控制
            # 不受物理引擎影响
            probe_node.addObject(
                'RestShapeSpringsForceField',
                name='probe_spring',
                stiffness=1e6,  # 极大刚度 = 近似固定
                angularStiffness=1e6,
                points=[0],  # 只有一个刚体点
                external_rest_shape='@probe_state'
            )
        # 碰撞模型
        probe_collision = probe_node.addChild(f"{node_name}Collision")
        collision_loader = add_loader(
            parent_node=probe_collision,
            filename=collision_filename,
            name='probe_collision_loader',
        )

        add_topology(
            parent_node=probe_collision,
            mesh_loader=collision_loader,
            topology=Topology.TRIANGLE
        )

        self.collision_state = probe_collision.addObject(
            'MechanicalObject',
            name='probe_collision_state',   # 改了名字，避免和上面的 probe_state 混淆
            template='Vec3d',
            showObject=1,
            showObjectScale=3,
            listening=1
        )

        add_collision_models(parent_node=probe_collision)
        probe_collision.addObject('RigidMapping', name='CollisionMapping')

        # 可视化模型
        visual_node = probe_node.addChild(f"Visual{node_name}")
        visual_loader = add_loader(
            parent_node=visual_node,
            filename=visual_filename,
            name='probe_visual_loader',
        )

        add_topology(
            parent_node=visual_node,
            mesh_loader=visual_loader,
            topology=Topology.TRIANGLE
        )

        add_visual_models(
            parent_node=visual_node,
            color=[0.957, 0.730, 0.582, 0.99]
        )

        visual_node.addObject('RigidMapping', name='VisualMapping')

        # ----------------- 运动路径设置 -----------------
        if not self.use_omega6:
            # 预定义路径模式
            self.target_position = target_position
            self.num_deformations = target_position.shape[0]
            print(f"Number of deformations to apply: {self.num_deformations}")
            self.current_def = 1

            if self.num_deformations > 1:
                tp = target_position[self.current_def - 1, :]
            else:
                tp = target_position[0, :]

            self.motion_path = self.create_linear_motion(
                target_position=tp,
                dt=root_node.dt.value,
                velocity=self.velocity
            )
        else:
            # Omega6模式：不创建运动路径
            self.motion_path = []
            print("Omega6模式：探头由设备直接控制")

    #########################################################
    # SOFA 回调
    #########################################################

    def onAnimateBeginEvent(self, __):
        """每个仿真步开始时更新探头位姿"""
        if self.use_omega6 and self.omega6:
            omega_pos = self.omega6.get_position()
            omega_ori = self.omega6.get_orientation()

            if omega_pos is None or omega_ori is None:
                return

            # ── 首次标定 ──────────────────────────────────────────────
            if self.omega6_init_pos is None:
                self.omega6_init_pos = omega_pos.copy()
                probe_init_pos = np.array(self.init_pose[:3])
                self.position_offset = probe_init_pos - self.omega6_init_pos

                # also calibrate orientation (raw device quat, no q_fix)
                init_device_quat = euler_to_quaternion(
                    omega_ori[1] + math.pi,
                    omega_ori[2] + math.pi,
                    omega_ori[0]
                )
                sofa_init_quat = np.array(self.init_pose[3:])
                # offset = SOFA_init * inv(raw_device_init)
                self.orientation_offset = self._multiply_quaternions(
                    sofa_init_quat, self._inverse_quaternion(init_device_quat)
                )

                self.position_amplification = 0.6
                self.smoothing_factor = 0.8
                self.max_speed_far = 0.01
                self.max_speed_near = 0.002
                self.near_threshold = 0.03

                print(f"Omega6 calibrated: pos_offset={self.position_offset}")

            # ── 计算目标位置 ──────────────────────────────────────────
            raw_delta = (omega_pos - self.omega6_init_pos) * self.position_amplification

            # ✅ 终极修正：对号入座
            # 画面[0]管上下，需要接收 手柄[1](上下)
            # 画面[1]管左右，需要接收 手柄[0](左右)
            # 画面[2]管前后，需要接收 手柄[2](前后)
            omega_delta = np.array([
                raw_delta[1],
                raw_delta[0],
               -raw_delta[2],
            ])

            # 💡 [可选] 调试打印：如果你发现方向是对的但跑反了，看看控制台
            # print(f"手柄输入: 左右={raw_delta[0]:.3f}, 上下={raw_delta[1]:.3f}, 前后={raw_delta[2]:.3f}")

            omega_pos_amplified = omega_delta + self.omega6_init_pos
            target_probe_pos = omega_pos_amplified + self.position_offset

            # ── 初始化当前位置 ────────────────────────────────────────
            if not hasattr(self, '_last_probe_pos'):
                self._last_probe_pos = np.array(self.init_pose[:3])
            current_pos = self._last_probe_pos

            # ── 平滑跟随 ──────────────────────────────────────────────
            new_probe_pos = (current_pos
                             + (target_probe_pos - current_pos)
                             * self.smoothing_factor)

            # ── ✅ 修复：稳健的距离检测 ───────────────────────────────
            dist_to_breast = self._get_dist_to_breast(current_pos)

            # ── 动态限速（平滑过渡，无突变）──────────────────────────
            if dist_to_breast > self.near_threshold:
                max_move = self.max_speed_far
            else:
                # 平滑插值：distance 从 near_threshold→0，速度从 far→near
                ratio = dist_to_breast / self.near_threshold
                max_move = (self.max_speed_near
                            + (self.max_speed_far - self.max_speed_near)
                            * ratio)

            delta = new_probe_pos - current_pos
            delta_norm = np.linalg.norm(delta)
            if delta_norm > max_move:
                new_probe_pos = current_pos + (delta / delta_norm) * max_move

            # ── ✅ 修复：碰撞约束，防止穿透 ──────────────────────────
            new_probe_pos = self._apply_collision_constraint(new_probe_pos)

            # ── 姿态更新 ──────────────────────────────────────────────
            quat = euler_to_quaternion(
                omega_ori[1] + math.pi,
                omega_ori[2] + math.pi,
                omega_ori[0]
            )
            # axis compensation (device axes → probe axes)
            half = math.radians(-90.0) * 0.5
            c, s = math.cos(half), math.sin(half)
            q_fix = np.array([s, 0.0, 0.0, c])
            quat = self._multiply_quaternions(q_fix, quat)
            # calibration offset (align initial poses)
            quat = self._multiply_quaternions(self.orientation_offset, quat)

            new_pose = np.concatenate([new_probe_pos, quat])
            self.state.position.value = [new_pose]
            self._last_probe_pos = new_probe_pos.copy()
            self._apply_proximity_force()

            # ── 调试输出（每50帧）────────────────────────────────────
            if not hasattr(self, '_frame_count'):
                self._frame_count = 0
            self._frame_count += 1
            if self._frame_count % 50 == 0:
                print(f"[位置]  dist_to_breast={dist_to_breast * 1000:.1f}mm  "
                      f"max_move={max_move * 1000:.2f}mm/frame")

        else:
            # 预定义路径模式（不变）
            if len(self.motion_path):
                self.current_def_ended = False
                new_pose = np.append(self.motion_path.pop(0), self.init_pose[3:])
                target_val = [new_pose.tolist()]
                self.state.position.value = target_val
                if hasattr(self.state, 'rest_position'):
                    self.state.rest_position.value = target_val
            elif hasattr(self, 'current_def') and self.current_def <= self.num_deformations:
                self.current_def_ended = True
                print(f"End of deformation {self.current_def}")
                self.current_def += 1
                if self.current_def <= self.num_deformations:
                    target_position = self.target_position[self.current_def - 1, :]
                    self.motion_path = self.create_linear_motion(
                        target_position=target_position,
                        dt=self.root_node.dt.value,
                        velocity=self.velocity
                    )

    def onAnimateEndEvent(self, __):
        """
        力反馈输出：变形力 + 接近力，含指数平滑滤波。
        核心逻辑：
          1. 探头不接触乳腺时，力平滑衰减至零（不突变）
          2. 接触时，对输出力做指数平滑，消除高频抖动
        """
        if not (self.use_omega6 and self.omega6):
            return

        # ── 初始化平滑器（首帧）──────────────────────────────────
        if not hasattr(self, '_smoothed_force'):
            self._smoothed_force = np.zeros(3)
            self.FORCE_SMOOTH_ALPHA = 0.80  # 平滑系数：0.7~0.92，越大越平滑

        # ══════════════════════════════════════════════════════════
        # 关键判断：探头是否正在接触乳腺？
        # ══════════════════════════════════════════════════════════
        if hasattr(self, '_last_probe_pos'):
            dist_now = self._get_dist_to_breast(self._last_probe_pos)
        else:
            dist_now = 999.0

        CONTACT_THRESHOLD = 0.01  # 与 _apply_proximity_force 中保持一致

        if dist_now > CONTACT_THRESHOLD:
            # ── 探头已离开乳腺：平滑衰减至零，不突变 ──────────────
            self._smoothed_force = self._smoothed_force * 0.5
            self._proximity_force = np.zeros(3)

            self.current_disp_val = 0.0
            self.current_force_val = 0.0

            # 足够小时彻底清零，避免无限趋近
            if np.linalg.norm(self._smoothed_force) < 0.01:
                self._smoothed_force = np.zeros(3)

            self.omega6.set_force(self._smoothed_force)

            # 调试输出（每50帧）
            if not hasattr(self, '_cnt'):
                self._cnt = 0
            self._cnt += 1
            if self._cnt % 50 == 0:
                print(f"[力反馈]  探头离开乳腺  "
                      f"dist={dist_now * 1000:.1f}mm  "
                      f"衰减输出: {np.linalg.norm(self._smoothed_force):.4f}N")
            return

        # ══════════════════════════════════════════════════════════
        # 以下只在 dist_now <= CONTACT_THRESHOLD 时执行
        # ══════════════════════════════════════════════════════════

        # ── 1) 变形阻尼力 ──────────────────────────────────────────
        deform_force = np.zeros(3)
        max_disp = 0.0
        try:
            breast = self.root_node.BreastProbe.breast
            max_disp = breast.get_max_surface_displacement()
            scalar = max_disp * self.force_gain_deform
            deform_force = self._compute_deform_force(scalar)
        except Exception:
            pass

        # ── 2) 接近力 ──────────────────────────────────────────────
        proximity_force = np.zeros(3)
        if hasattr(self, '_proximity_force'):
            proximity_force = self._proximity_force * self.force_gain_contact

        # ── 3) 合成 ────────────────────────────────────────────────
        total_force = deform_force + proximity_force
        norm_f = np.linalg.norm(total_force)

        # ── 4) 死区过滤 + 限幅 ─────────────────────────────────────
        if norm_f < self.force_deadzone:
            total_force = np.zeros(3)
            norm_f = 0.0
        elif norm_f > self.max_force:
            total_force = (total_force / norm_f) * self.max_force
            norm_f = self.max_force

        # ── 5) 暴露属性供外部图表使用 ──────────────────────────────
        self.current_disp_val = max_disp
        self.current_force_val = norm_f

        # 写入数据文件
        if self._data_file and norm_f > self.force_deadzone:
            self._data_file.write(f"{max_disp:.6f}\t{norm_f:.6f}\n")

        # ── 6) 坐标系逆映射 ────────────────────────────────────────
        device_force_raw = np.array([
            total_force[1],
            total_force[0],
            -total_force[2],
        ])

        # ── 7) 指数平滑滤波 ────────────────────────────────────────
        # 公式：output = alpha × 上帧输出 + (1-alpha) × 本帧目标
        # 效果：力的变化变得连续，消除高频抖动
        alpha = self.FORCE_SMOOTH_ALPHA

        if norm_f == 0.0:
            # 目标力为零（死区内）：快速衰减，不拖尾
            self._smoothed_force = self._smoothed_force * 0.5
        else:
            # 正常接触：平滑跟随目标力
            self._smoothed_force = (alpha * self._smoothed_force
                                    + (1.0 - alpha) * device_force_raw)

        # 平滑后再次限幅保护
        smoothed_norm = np.linalg.norm(self._smoothed_force)
        if smoothed_norm > self.max_force:
            self._smoothed_force = (self._smoothed_force / smoothed_norm
                                    * self.max_force)

        # ── 8) 输出平滑后的力给设备 ────────────────────────────────
        self.omega6.set_force(self._smoothed_force)

        # ── 9) 调试输出（每50帧）───────────────────────────────────
        if not hasattr(self, '_cnt'):
            self._cnt = 0
        self._cnt += 1
        if self._cnt % 50 == 0:
            print(f"[力反馈]  "
                  f"dist={dist_now * 1000:.1f}mm  "
                  f"变形力: {np.linalg.norm(deform_force):.4f}N  "
                  f"接近力: {np.linalg.norm(proximity_force):.4f}N  "
                  f"原始合力: {norm_f:.4f}N  "
                  f"平滑输出: [{self._smoothed_force[0]:.3f}, "
                  f"{self._smoothed_force[1]:.3f}, "
                  f"{self._smoothed_force[2]:.3f}]N")
    #########################################################
    # 自定义方法
    #########################################################
    def _get_dist_to_breast(self, probe_pos: np.ndarray) -> float:
        """
        稳健地获取探头到乳腺表面的最近距离。
        失败时返回一个合理的默认值而不是 999000。
        """
        try:
            breast         = self.root_node.BreastProbe.breast
            collision_node = breast.node.getChild(f"Collision{self.root_node.BreastProbe.model_name}")
            breast_mech    = collision_node.getObject("breast_collision_state")
            breast_surface = np.array(breast_mech.position.value)

            if breast_surface.size == 0:
                return self.near_threshold  # 默认在感应边界

            dists = np.linalg.norm(breast_surface - probe_pos, axis=1)
            return float(dists.min())

        except Exception:
            return self.near_threshold      # ✅ 失败时返回阈值，不返回999

    def _apply_collision_constraint(self, new_probe_pos: np.ndarray) -> np.ndarray:
        try:
            breast = self.root_node.BreastProbe.breast
            collision_node = breast.node.getChild(f"Collision{self.root_node.BreastProbe.model_name}")
            breast_mech = collision_node.getObject("breast_collision_state")
            breast_surface = np.array(breast_mech.position.value)

            if breast_surface.size == 0:
                return new_probe_pos

            # ✅ 同样改用碰撞面中心
            probe_surface = np.array(self.collision_state.position.value)
            if probe_surface.size > 0:
                probe_center = probe_surface.mean(axis=0)
            else:
                probe_center = new_probe_pos

            breast_sampled = breast_surface[::5]
            dists = np.linalg.norm(breast_sampled - probe_center, axis=1)
            min_idx = np.argmin(dists)
            min_dist = dists[min_idx]

            SAFE_DIST = 0.002
            MAX_CORRECT = 0.001

            if min_dist < SAFE_DIST:
                nearest_point = breast_sampled[min_idx]
                push_direction = probe_center - nearest_point
                push_norm = np.linalg.norm(push_direction)

                if push_norm < 1e-6:
                    return new_probe_pos

                push_unit = push_direction / push_norm

                # 方向校验
                breast_center = breast_sampled.mean(axis=0)
                outward = probe_center - breast_center
                if np.dot(push_unit, outward) < 0:
                    return new_probe_pos

                clamped = min(SAFE_DIST - min_dist, MAX_CORRECT)
                new_probe_pos = new_probe_pos + push_unit * clamped

                # ✅ 调试
                if not hasattr(self, '_constraint_cnt'):
                    self._constraint_cnt = 0
                self._constraint_cnt += 1
                if self._constraint_cnt % 10 == 0:
                    print(f"[碰撞约束] min_dist={min_dist * 1000:.2f}mm  "
                          f"correction={clamped * 1000:.2f}mm")

            return new_probe_pos

        except Exception as e:
            print(f"[碰撞约束异常] {e}")
            return new_probe_pos

    def _multiply_quaternions(self, q1, q2):
        """
        四元数乘法 q1 * q2
        格式均为 [qx, qy, qz, qw]
        """
        x1, y1, z1, w1 = q1
        x2, y2, z2, w2 = q2
        return np.array([
            w1*x2 + x1*w2 + y1*z2 - z1*y2,
            w1*y2 - x1*z2 + y1*w2 + z1*x2,
            w1*z2 + x1*y2 - y1*x2 + z1*w2,
            w1*w2 - x1*x2 - y1*y2 - z1*z2
        ])

    def _inverse_quaternion(self, q):
        """四元数的逆（共轭） [qx, qy, qz, qw]"""
        return np.array([-q[0], -q[1], -q[2], q[3]])

    def create_linear_motion(
            self,
            target_position: np.ndarray,
            dt: float,
            velocity: float,
            single_step: bool = False,
            start_position: Optional[np.ndarray] = None
    ) -> List[np.ndarray]:
        """
        Creates movement path to displace probe from its current position
        to the final position provided, at the given velocity.
        """

        if start_position is None:
            current_position = np.array(self.state.position.value[0][:3])
        else:
            assert len(start_position) == 3
            current_position = np.array(start_position)

        displacement = target_position - current_position
        print(f"Now moving to {target_position}, from {current_position}")

        if single_step:
            motion_steps = 1
            motion_path = [target_position]
        else:
            displacement_per_step = velocity * dt
            if displacement_per_step <= 0:
                return [target_position]

            total_dist = np.linalg.norm(displacement)
            motion_steps = int(np.ceil(total_dist / displacement_per_step)) if total_dist > 0 else 1

            progress = np.linspace(0.0, 1.0, motion_steps + 1)[1:]
            motion_path = current_position + displacement * progress[:, np.newaxis]
            motion_path[-1] = target_position

        return np.split(motion_path, motion_steps, axis=0)

    def _apply_proximity_force(self):
        """
        接近力：直接用探头碰撞面中心到乳腺表面的距离
        不再依赖刚体位置，消除几何偏差导致的误判
        """
        try:
            breast = self.root_node.BreastProbe.breast
            collision_node = breast.node.getChild(f"Collision{self.root_node.BreastProbe.model_name}")
            breast_mech = collision_node.getObject("breast_collision_state")
            breast_surface = np.array(breast_mech.position.value)

            # ✅ 关键修改：用碰撞面中心，不用刚体位置
            probe_surface = np.array(self.collision_state.position.value)
            if probe_surface.size == 0 or breast_surface.size == 0:
                self._proximity_force = np.zeros(3)
                return

            probe_center = probe_surface.mean(axis=0)  # 探头碰撞面中心

            # ✅ 降采样
            breast_sampled = breast_surface[::5]
            dists = np.linalg.norm(breast_sampled - probe_center, axis=1)
            min_idx = np.argmin(dists)
            min_dist = dists[min_idx]

            # ✅ 扩大阈值：从 5mm 扩大到 20mm，确保接触前就有阻力
            PROXIMITY_THRESHOLD = 0.01

            if min_dist > PROXIMITY_THRESHOLD:
                self._proximity_force = np.zeros(3)
                return

            nearest_point = breast_sampled[min_idx]
            push_direction = probe_center - nearest_point
            push_norm = np.linalg.norm(push_direction)

            if push_norm < 1e-6:
                self._proximity_force = np.zeros(3)
                return

            push_unit = push_direction / push_norm
            penetration = PROXIMITY_THRESHOLD - min_dist

            # ✅ smoothstep：边界处导数为0，完全消除阶跃
            t = penetration / PROXIMITY_THRESHOLD  # [0, 1]
            smooth_t = t * t * (3.0 - 2.0 * t)  # smoothstep
            force_magnitude = self.SPRING_STIFFNESS * smooth_t * PROXIMITY_THRESHOLD

            self._proximity_force = push_unit * force_magnitude

            # ── 调试输出（每50帧）────────────────────────────────
            if not hasattr(self, '_prox_cnt'):
                self._prox_cnt = 0
            self._prox_cnt += 1
            if self._prox_cnt % 50 == 0:
                print(f"[接近力]  "
                      f"probe_center={probe_center}  "
                      f"最近距离: {min_dist * 1000:.1f}mm  "
                      f"力大小: {force_magnitude:.4f}N")

        except Exception as e:
            print(f"[接近力异常] {e}")
            self._proximity_force = np.zeros(3)


    def _compute_deform_force(self, scalar_force: float) -> np.ndarray:
        """
        根据探头当前位置和乳腺质心，计算阻力方向。
        力方向 = 探头指向乳腺质心的反方向（即阻止探头压入的力）。
        """
        try:
            breast = self.root_node.BreastProbe.breast
            # 计算乳腺质心
            breast_center = np.mean(
                np.array(breast.topology.position.value), axis=0
            )
            probe_pos = self._last_probe_pos

            # 探头→乳腺方向
            direction = breast_center - probe_pos
            norm = np.linalg.norm(direction)

            if norm < 1e-6:
                return np.zeros(3)

            # 阻力方向 = 反向（推开探头）
            resist_direction = -(direction / norm)
            return resist_direction * scalar_force

        except Exception:
            # 降级：退回原来的 -X 方向
            return np.array([-scalar_force, 0.0, 0.0])

    def close_omega6(self):
        """关闭Omega6设备"""
        if self.omega6:
            self.omega6.close()

        # ✅ 新增：关闭数据文件
        if self._data_file:
            self._data_file.close()
            self._data_file = None
            print("✓ F-x 数据文件已保存")