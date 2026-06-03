import numpy as np
import socket
from scipy.signal import butter, lfilter
from typing import Tuple, Optional


class SoftTissueForceFeedback:
    """
    将软组织形变转换为非线性力反馈的模块。
    功能：
        1. 从 SOFA 仿真中读取顶点位移和接触力。
        2. 根据形变深度和速度计算非线性力。
        3. 平滑处理力信号并通过 TCP/UDP 发送到力反馈设备。
    """

    def __init__(
            self,
            sofa_probe_node: object,  # SOFA 探头节点的引用
            sofa_breast_node: object,  # SOFA 乳腺节点的引用
            output_ip: str = "127.0.0.1",
            output_port: int = 12345,
            max_force: float = 8.0,  # 最大力阈值（单位：牛顿）
            cutoff_freq: float = 10.0  # 低通滤波截止频率（Hz）
    ):
        self.probe_node = sofa_probe_node
        self.breast_node = sofa_breast_node
        self.max_force = max_force
        self.cutoff_freq = cutoff_freq
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)  # UDP 通信
        self.output_addr = (output_ip, output_port)

        # 低通滤波器参数
        self.b, self.a = self._butter_lowpass()
        self.last_forces = np.zeros(3)  # 历史力数据（用于滤波）

    def _butter_lowpass(self) -> Tuple[np.ndarray, np.ndarray]:
        """设计低通滤波器（防止力突变）"""
        nyquist_freq = 0.5 * 1000  # 假设采样率 1kHz
        normal_cutoff = self.cutoff_freq / nyquist_freq
        b, a = butter(2, normal_cutoff, btype="low", analog=False)
        return b, a

    def _filter_force(self, force: np.ndarray) -> np.ndarray:
        """应用低通滤波"""
        self.last_forces = lfilter(self.b, self.a, np.vstack([self.last_forces, force])[-1])
        return self.last_forces

    def _compute_deformation_depth(self) -> float:
        """
        计算探头与乳腺的最大形变深度：
        1. 获取探头接触区域的顶点位移。
        2. 取位移向量的平均长度作为形变深度。
        """
        probe_pos = self.probe_node.position.value  # 探头位置（SOFA 格式）
        breast_vertices = self.breast_node.mesh.position.value  # 乳腺顶点坐标
        displacement = breast_vertices - probe_pos  # 形变位移向量
        return np.linalg.norm(displacement, axis=1).mean()  # 平均形变深度

    def _nonlinear_force_mapping(
            self,
            deformation_depth: float,
            deformation_speed: float,
            base_stiffness: float = 100.0  # 基础刚度系数（N/m）
    ) -> np.ndarray:
        """
        非线性力映射模型：
        力 = 刚度 * 形变深度 * exp(速度增强系数)
        参数：
            - deformation_depth: 形变深度（米）
            - deformation_speed: 形变速度（米/秒）
            - base_stiffness: 软组织基础刚度
        """
        speed_factor = np.exp(0.5 * deformation_speed)  # 速度增强项
        force_magnitude = base_stiffness * deformation_depth * speed_factor
        force_direction = -self.probe_node.velocity.value  # 力方向与探头速度相反
        force_direction /= np.linalg.norm(force_direction) + 1e-6  # 归一化
        return force_magnitude * force_direction

    def update_and_get_force(self) -> Optional[np.ndarray]:
        """
        从 SOFA 获取最新数据并计算力反馈：
        返回：3D 力向量（x,y,z），单位牛顿
        """
        try:
            # 1. 获取形变数据
            depth = self._compute_deformation_depth()
            speed = np.linalg.norm(self.probe_node.velocity.value)

            # 2. 计算非线性力
            raw_force = self._nonlinear_force_mapping(depth, speed)

            # 3. 滤波和限幅
            filtered_force = self._filter_force(raw_force)
            clamped_force = np.clip(filtered_force, -self.max_force, self.max_force)

            return clamped_force
        except Exception as e:
            print(f"[Error] Force computation failed: {e}")
            return None

    def send_force_to_device(self, force: np.ndarray) -> bool:
        """通过 UDP 发送力到 CHAI3D 或力反馈设备"""
        try:
            self.socket.sendto(force.tobytes(), self.output_addr)
            return True
        except Exception as e:
            print(f"[Error] Force send failed: {e}")
            return False


# 使用示例 -----------------------------------------------------------------
if __name__ == "__main__":
    # 假设已从 SOFA 中获取探头和乳腺节点
    probe_node = ...  # SOFA 探头 MechanicalObject
    breast_node = ...  # SOFA 乳腺 MechanicalObject

    # 初始化模块
    force_feedback = SoftTissueForceFeedback(
        probe_node, breast_node,
        output_ip="127.0.0.1", output_port=12345
    )


    # 在 SOFA 的动画循环中调用
    def onAnimateBeginEvent(_):
        force = force_feedback.update_and_get_force()
        if force is not None:
            force_feedback.send_force_to_device(force)