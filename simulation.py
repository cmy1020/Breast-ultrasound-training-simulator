#  (必须放在最顶部，在 import SofaRuntime 之前)
import os

SOFA_ROOT = os.environ.get("SOFA_ROOT", r"E:\sofav24.06.00")
SOFAPYTHON3_ROOT = os.environ.get("SOFAPYTHON3_ROOT", os.path.join(SOFA_ROOT, "plugins", "SofaPython3"))

# Windows/Python3.8+：显式添加 DLL 搜索目录（比改 PATH 更可靠）
if os.name == "nt":
    os.add_dll_directory(os.path.join(SOFA_ROOT, "bin"))
    os.add_dll_directory(os.path.join(SOFAPYTHON3_ROOT, "bin"))
# ============================================================
# 可选导入 Sofa.Gui（关键修改）
# ============================================================
import Sofa.Simulation, Sofa.Core, SofaRuntime

# 尝试导入 Gui，如果失败则跳过（用于 Qt 集成）
try:
    import Sofa.Gui

    GUI_AVAILABLE = True
except ImportError:
    GUI_AVAILABLE = False
    print("[INFO] Sofa.Gui 关闭，将运行QT交互模式")

import os, time
import numpy as np

from components.utils import Parameters, Analysis, matrix2xyzquat
from components.header import add_scene_header
from objects.breast import Breast, Lesion
from objects.probe import Probe


class BreastProbe(Sofa.Core.Controller):
    """ Sofa Controller representing an US probe that comes in contact with a breast with an embedded lesion"""

    def __init__(
            self,
            root: Sofa.Core.Node,
            params: Parameters,
            use_omega6: bool = False
    ):
        """
        Args:
            root (Sofa.Core.Node): root node of the simulation.
            params (Parameters): class containing all the parameters provided in the yml configuration file.
        """
        Sofa.Core.Controller.__init__(self)
        self.root = root

        # 模型名称（默认"Breast"，liver 模式时设为"Liver"）
        self.model_name = getattr(params, "model_name", "Breast")

        # ========= 关键修改1：把自己挂到 root，方便 Probe 查找 =========
        root.BreastProbe = self
        # ==========================================================

        # Output files
        outdir = os.path.join(params.outdata_dir, f"tumor{params.tumorID}")
        os.makedirs(outdir, exist_ok=True)
        self.lesion_output_filename = os.path.join(outdir, f"Fiducial")
        self.time_output_filename = os.path.join(outdir, f"Time")

        # Create header
        add_scene_header(root,
                         gravity=params.gravity,
                         dt=params.dt,
                         alarm_distance=params.alarm_distance,
                         contact_distance=params.contact_distance
                         )

        # Initial transform for the model
        breast_transform = np.identity(4)
        # Apply model scale if specified (e.g. liver at 0.025x)
        model_scale = getattr(params, "model_scale", 1.0)
        breast_transform[0, 0] = model_scale
        breast_transform[1, 1] = model_scale
        breast_transform[2, 2] = model_scale
        # Apply model offset if specified (shift model toward probe)
        model_offset = getattr(params, "model_offset", [0.0, 0.0, 0.0])
        breast_transform[0, 3] = float(model_offset[0])
        breast_transform[1, 3] = float(model_offset[1])
        breast_transform[2, 3] = float(model_offset[2])

        # Read fixed indices from file
        fixed_indices = np.loadtxt(params.breast_fixed_file, dtype=int).tolist()

        # Create breast
        self.breast = root.addObject(
            Breast(root,
                   volume_filename=params.breast_volume_file,
                   collision_filename=params.breast_collision_file,
                   init_tf=breast_transform,
                   density=params.rho,
                   material=params.material,
                   E=params.E,
                   nu=params.nu,
                   fixed_indices=fixed_indices,
                   visual_filename=params.breast_visual_file,
                   node_name=self.model_name
                   )
        )

        # Create lesion (skip if tumorID == 0, e.g. liver model)
        if params.tumorID > 0:
            lesion_filename = f"{params.lesion_basedir}{params.tumorID}/Fiducial0.txt"
            lesion_position = np.loadtxt(lesion_filename)

            self.lesion = root.addObject(
                Lesion(self.breast.node,
                       lesion_position=lesion_position,
                       surface_filename=getattr(params, "lesion_surface_file", None),
                       init_tf=np.identity(4),
                       node_name=f"Lesion{params.tumorID}")
            )
        else:
            self.lesion = None
            print(f"  (无结节模式 - {self.model_name} 模型)")

        # ================== 预定义路径读取 ==================
        target_positions = []
        if params.tumorID > 0 and params.num_deformations > 0:
            for i in range(1, params.num_deformations + 1):
                probe_tf_filename = f"{params.lesion_basedir}{params.tumorID}/Transform{i}.txt"
                probe_matrix = np.loadtxt(probe_tf_filename)
                target_positions.append([probe_matrix[0, 3],
                                         probe_matrix[1, 3],
                                         probe_matrix[2, 3]])
            target_positions = np.asarray(target_positions).reshape((-1, 3))
        # ==========================================================

        # Read probe initial transform (or use default for liver)
        if params.tumorID > 0:
            probe_tf_filename = f"{params.lesion_basedir}{params.tumorID}/Transform0.txt"
            probe_matrix = np.loadtxt(probe_tf_filename)
            probe_pose = matrix2xyzquat(probe_matrix,
                                        offset=np.asarray(params.probe_initial_offset))
        else:
            # Liver / no predefined path: use a default starting pose
            probe_pose = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0])

        probe_pose[0] += 0.0  # X轴偏移
        probe_pose[1] -= 0.0  # Y轴偏移
        probe_pose[2] += 0.0  # Z轴偏移

        # Omega6 模式下调整初始位置
        if use_omega6:
            probe_pose[0] += 0.04
            probe_pose[1] += 0.00
            probe_pose[2] += 0.00

            print(f"[OK] Omega6 active: probe initial position adjusted")
            print(f"  位置: X={probe_pose[0]:.3f}, Y={probe_pose[1]:.3f}, Z={probe_pose[2]:.3f}")

        # 根据模式决定 target_positions
        if use_omega6:
            target_positions = np.array([probe_pose[:3]])
            print("Omega6模式：跳过预定义路径加载")

        # Create probe
        self.probe = root.addObject(
            Probe(root,
                  collision_filename=params.probe_collision_file,
                  visual_filename=params.probe_visual_file,
                  init_pose=probe_pose,
                  target_position=target_positions,
                  velocity=params.probe_velocity,
                  use_omega6=use_omega6
                  )
        )

        # Outputs
        self.lesion_positions = []
        self.time_per_def = []
        self.start_time = time.time()

        self.use_omega6 = use_omega6

    def get_probe_position(self):
        """
        获取探头位置 - 供 Qt 窗口调用
        返回: [x, y, z] 或 None
        """
        try:
            if hasattr(self, 'probe') and self.probe:
                probe_node = self.root.getChild("Probe")
                if probe_node:
                    mech = probe_node.getObject("probe_state")
                    if mech and hasattr(mech, 'position'):
                        pose = mech.position.value[0]
                        return pose[:3]
        except Exception as e:
            pass

        return None

    def onKeypressedEvent(self, event):
        """键盘事件处理"""
        key = event['key']
        if key in ['Q', 'q']:
            print("\n用户退出...")
            self.root.animate = False
            if hasattr(self.probe, 'close_omega6'):
                self.probe.close_omega6()

    def onAnimateBeginEvent(self, __):
        if self.use_omega6:
            return

        if hasattr(self.probe, 'current_def_ended') and self.probe.current_def_ended:
            lesion_position = np.mean(self.lesion.state.position.value, axis=0)
            self.lesion_positions.append(lesion_position.tolist())
            self.time_per_def.append(time.time() - self.start_time)
            self.start_time = time.time()
            self.probe.current_def_ended = False

        is_all_path_done = (hasattr(self.probe, 'current_def') and
                            self.probe.current_def > self.probe.num_deformations)

        if is_all_path_done and self.breast.is_stable:
            print("所有任务完成，乳腺已稳定。停止仿真。")
            self.root.animate = False

            self.lesion_positions = np.asarray(self.lesion_positions)
            np.savez_compressed(self.lesion_output_filename, self.lesion_positions)
            print(f"Lesion positions saved in {self.lesion_output_filename}")
            np.savez_compressed(self.time_output_filename, self.time_per_def)
            print(f"Time per deformations saved in {self.time_output_filename}")


def createScene(root, params, use_omega6=False):
    """
    创建 SOFA 场景

    Args:
        root: SOFA 根节点
        params: 参数对象
        use_omega6: 是否使用 Omega6 设备

    Returns:
        root: 配置好的根节点
    """
    root.addObject(BreastProbe(root, params, use_omega6=use_omega6))
    return root


# ============================================================
# 独立运行模式（当直接执行 simulation.py 时）
# ============================================================
if __name__ == '__main__':

    # Make sure to load all SOFA libraries
    plugins = [
        "SofaComponentAll",
        "Sofa.Component.Collision.Detection.Algorithm",
        "Sofa.Component.Collision.Detection.Intersection",
        "Sofa.Component.Collision.Response.Contact",
    ]
    for p in plugins:
        SofaRuntime.importPlugin(p)

    # Create the root node
    root = Sofa.Core.Node("root")

    # Load parameters
    params = Parameters("./input_parameters.yml")

    # 询问模式
    print("\n选择控制模式:")
    print("1. Omega6力反馈模式")
    print("2. 预定义路径模式（原模式）")
    choice = input("请选择 (1/2): ").strip()
    use_omega6 = (choice == '1')

    # Check if the needed analysis type is implemented
    assert Analysis(params.type) in Analysis, f"Invalid choice for simulation approach {params.type}"
    print(f"\n\n\nCreating simulation with {params.type} approach")

    # 创建场景
    createScene(root, params, use_omega6=use_omega6)
    Sofa.Simulation.init(root)

    # Omega6模式自动启动动画
    if use_omega6:
        root.animate = True
        print("\n" + "=" * 60)
        print("[OK] Omega6: simulation auto-started")
        print("  - 直接移动手柄控制探头")
        print("  - 不要点击GUI的播放按钮")
        print("  - 按键盘 Q 键退出")
        print("=" * 60 + "\n")

    # ============================================================
    # 只有在 GUI 可用时才启动 GUI
    # ============================================================
    if not params.use_gui:
        # Execute simulation in background, without GUI
        for iteration in range(100000):
            Sofa.Simulation.animate(root, root.dt.value)
    else:
        if not GUI_AVAILABLE:
            print("[WARNING] Sofa.Gui 不可用，强制使用无GUI模式")
            for iteration in range(100000):
                Sofa.Simulation.animate(root, root.dt.value)
        else:
            # Launch the GUI
            print("Supported GUIs are: " + Sofa.Gui.GUIManager.ListSupportedGUI(","))
            Sofa.Gui.GUIManager.Init("myscene", "qglviewer")
            Sofa.Gui.GUIManager.createGUI(root, __file__)
            Sofa.Gui.GUIManager.SetDimension(1080, 1080)
            Sofa.Gui.GUIManager.MainLoop(root)
            Sofa.Gui.GUIManager.closeGUI()
            print("GUI was closed")

    print("Simulation is done.")