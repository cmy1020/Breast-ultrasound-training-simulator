import Sofa.Core
import Sofa.SofaDeformable

from typing import Optional
import numpy as np

from components.solver import SolverType, ConstraintCorrectionType, add_solver
from components.tetrahedral import (
    Topology,
    add_collision_models,
    add_loader,
    add_tetrahedral_forcefield,
    add_topology,
    add_visual_models,
)


class Breast(Sofa.Core.Controller):
    """Sofa Controller representing the breast"""

    def __init__(
        self,
        root_node: Sofa.Core.Node,
        volume_filename: str,
        collision_filename: str,
        init_tf: np.ndarray = np.identity(4),
        density: float = 1000,
        material: str = "Corotated",
        E: float = 3000,
        nu: float = 0.45,
        fixed_indices: Optional[list] = None,
        node_name: str = "Breast",
        visual_filename: Optional[str] = None,
    ):
        """
        Args:
            root_node (Sofa.Core.Node): root node of the simulation.
            volume_filename (str): name of the tetrahedral volume file.
            collision_filename (str): name of the triangular surface file used for collision computation.
            init_tf (np.ndarray): initial 4x4 transformation to be applied to the model.
            density (float): object density.
            material (str): type of material.
            E (float): elastic modulus of the object.
            nu (float): poisson ratio.
            fixed_indices (list): indices of the volume mesh which are constrained in all directions.
            node_name (str): name of the created node.
            visual_filename (str, Optional): name of the triangular surface file used for visualization.
        """
        Sofa.Core.Controller.__init__(self)

        breast_node = root_node.addChild(node_name)
        self.node = breast_node

        # -------------------------------------------------
        # 体网格与拓扑
        # -------------------------------------------------
        volume_loader = add_loader(
            parent_node=breast_node,
            filename=volume_filename,
            name="breast_volume_loader",
            transformation=init_tf,
        )

        # self.topology 一般为 TetrahedronSetTopologyContainer 或类似，持有 position 数据
        self.topology = add_topology(
            parent_node=breast_node,
            mesh_loader=volume_loader,
            topology=Topology.TETRAHEDRON,
        )

        # 机械对象：必须是 Vec3d，保存每个点的坐标
        self.state = breast_node.addObject(
            "MechanicalObject",
            name="breast_state",
            template="Vec3d",     # <<< 关键修改：原来是 CompressedRowSparseMatrixMat3x3d（错误）
            showObject=0,
            listening=1,
        )

        # 质量
        breast_node.addObject(
            "MeshMatrixMass",
            massDensity=density,
            name="breast_mass",
        )

        # 力场
        add_tetrahedral_forcefield(
            parent_node=breast_node,
            material=material,
            E=E,
            nu=nu,
        )

        # --- 1. 显式添加时间步求解器 (Implicit Solver) ---0104修改
        # 增加 rayleighStiffness 和 rayleighMass 是防止爆炸的核心！
        breast_node.addObject('EulerImplicitSolver',
                              name='odesolver',
                              rayleighStiffness=0.2,  # 耗散由于网格变形产生的多余能量 增大数值会变稳定但是仿真回弹变慢 越小 位移恢复越快
                              rayleighMass=0.2)  # 耗散由于物体整体移动产生的多余能量 越小 表面恢复越快

        # --- 2. 显式添加线性系统求解器 (Linear Solver) ---
        # 这里对应你之前的 SolverType.SOFASPARSE
        breast_node.addObject('SparseLDLSolver', name='preconditioner')

        # --- 3. 显式添加约束修正 (Constraint Correction) ---
        # 这保证了碰撞发生时，力能正确计算且不让系统崩溃
        breast_node.addObject('LinearSolverConstraintCorrection', name='correction')
        # -------------------------------------------------
        # 固定约束（若没提供 fixed_indices，则用 BoxROI 自动选一批点）
        # -------------------------------------------------
        if fixed_indices is None:
            xmin, xmax, ymin, ymax, zmin, zmax = get_bbox(self.topology.position.value)
            # 这里的 box 定义要依据你的几何坐标方向，可再调
            box_fixed = [xmin, ymin, zmax, xmax, ymax, zmin + 0.006]
            breast_node.addObject(
                "BoxROI",
                name="fixed_points_box",
                box=box_fixed,
                drawPoints=True,
                drawSize=0.01,
            )
            fixed_indices = "@fixed_points_box.indices"

        breast_node.addObject(
            "FixedConstraint",
            name="fixed_points",
            indices=fixed_indices,
        )

        # -------------------------------------------------
        # 可视化模型
        # -------------------------------------------------
        if visual_filename is not None:
            visual_node = breast_node.addChild(f"Visual{node_name}")
            visual_loader = add_loader(
                parent_node=visual_node,
                filename=visual_filename,
                name="breast_visual_loader",
                transformation=init_tf,
            )

            add_topology(
                parent_node=visual_node,
                mesh_loader=visual_loader,
                topology=Topology.TRIANGLE,
            )

            add_visual_models(
                parent_node=visual_node,
                color=[0.957, 0.730, 0.582, 0.99],# 0.957, 0.730, 0.582, 0.99
            )
            visual_node.addObject("BarycentricMapping", name="VisualMapping")

        # -------------------------------------------------
        # 碰撞模型
        # -------------------------------------------------
        collision_node = breast_node.addChild(f"Collision{node_name}")
        collision_loader = add_loader(
            parent_node=collision_node,
            filename=collision_filename,
            name="breast_collision_loader",
            transformation=init_tf,
        )

        add_topology(
            parent_node=collision_node,
            mesh_loader=collision_loader,
            topology=Topology.TRIANGLE,
        )

        collision_node.addObject(
            "MechanicalObject",
            name="breast_collision_state",
            template="Vec3d",
            showObject=0,
            listening=1,
        )

        add_collision_models(parent_node=collision_node)

        collision_node.addObject(
            "BarycentricMapping",
            name="CollisionMapping",
        )

        # -------------------------------------------------
        # 状态监控变量
        # -------------------------------------------------
        # 初始位置（体网格位置）
        self.previous_position = np.array(self.topology.position.value)
        self.is_stable = True

        # 保存初始“表面形状”，用于计算最大表面位移（力反馈）
        # 这里直接使用拓扑上的 position（体网格的节点），
        # 如果你希望只看“外表面”，可以改为用 collision_node 的 MechanicalObject.position
        self.surface_rest_position = np.array(self.topology.position.value)
        # ✅ 新增：碰撞力初始化（防止首帧读取报错）
        self._contact_force = np.zeros(3)
        self._use_collision_surface = False

    # -------------------------------------------------
    # 供 Probe 调用：获取最大表面位移 (单位 m)
    # -------------------------------------------------
    def get_max_surface_displacement(self) -> float:
        """
        返回乳腺当前碰撞表面相对于初始形状的最大位移（单位与网格一致）。
        """
        try:
            if getattr(self, '_use_collision_surface', False):
                collision_node = self.node.getChild("CollisionBreast")
                mech = collision_node.getObject("breast_collision_state")
                current = np.array(mech.position.value)
            else:
                current = np.array(self.topology.position.value)

            disp = np.linalg.norm(current - self.surface_rest_position, axis=1)
            return float(np.max(disp))
        except Exception:
            return 0.0

    def onSimulationInitDoneEvent(self, _):
        """仿真初始化完成后，记录碰撞表面的初始位置（用于计算位移）"""
        try:
            collision_node = self.node.getChild("CollisionBreast")
            mech = collision_node.getObject("breast_collision_state")
            self.surface_rest_position = np.array(mech.position.value)
            self._use_collision_surface = True
            print("[Breast] 已使用碰撞表面网格作为位移参考")
        except Exception:
            # 降级：继续用体网格
            self.surface_rest_position = np.array(self.topology.position.value)
            self._use_collision_surface = False
            print("[Breast] 降级：使用体网格作为位移参考")

    def onAnimateEndEvent(self, __):
        """
        每一帧结束时检查数值稳定性。
        注意：这里仅用来检测 NaN，不会让仿真停止。
        """
        current_position = np.array(self.topology.position.value)
        breast_displacement = current_position - self.previous_position
        self.is_stable = is_stable(breast_displacement)

        if not self.is_stable:
            # 只打印一次警告，不重复打印
            if not hasattr(self, "_unstable_warned"):
                print("警告: 检测到仿真不稳定（已忽略）")
                self._unstable_warned = True
            # 将 is_stable 设回 True，保证外面逻辑不会直接停掉仿真
            self.is_stable = True

        # 关键：更新 previous_position，下一帧才能正确计算位移
        self.previous_position = current_position
        # ✅ 新增：每帧更新碰撞力
        self._contact_force = self._read_contact_force()

    def _read_contact_force(self) -> np.ndarray:
        """
        从碰撞节点读取乳腺所受的接触合力。
        """
        try:
            collision_node = self.node.getChild("CollisionBreast")
            if collision_node is None:
                return np.zeros(3)
            mech = collision_node.getObject("breast_collision_state")
            if mech and hasattr(mech, 'force'):
                forces = np.array(mech.force.value)
                if forces.size > 0:
                    return forces.sum(axis=0)
        except Exception:
            pass
        return np.zeros(3)

    def get_contact_force(self) -> np.ndarray:
        """供 Probe 调用，获取当前帧碰撞合力。"""
        if hasattr(self, '_contact_force'):
            return self._contact_force.copy()
        return np.zeros(3)

    def set_density(self, value: float):
        """实时修改材料密度 (kg/m³)"""
        try:
            mass = self.node.getObject("breast_mass")
            if mass and hasattr(mass, 'massDensity'):
                mass.massDensity.value = float(value)
        except Exception:
            pass

    def set_poisson_ratio(self, value: float):
        """实时修改泊松比"""
        try:
            fem = self.node.getObject("FEM")
            if fem and hasattr(fem, 'poissonRatio'):
                fem.poissonRatio.value = float(value)
        except Exception:
            pass

    def shift_model(self, dx: float, dy: float, dz: float):
        """平移整个模型（含体网格 + 碰撞表面 + 参考位置）"""
        offset = np.array([dx, dy, dz], dtype=np.float64)
        try:
            self.topology.position.value += offset
        except Exception:
            pass
        try:
            self.surface_rest_position += offset
        except Exception:
            pass


class Lesion(Sofa.Core.Controller):
    """Sofa Controller representing a point or a small nodule mapped to a parent deformable object"""
    def __init__(
        self,
        parent_node: Sofa.Core.Node,
        lesion_position: np.ndarray = np.zeros((1, 3)),
        surface_filename: Optional[str] = None,
        init_tf: np.ndarray = np.identity(4),
        node_name: str = "Lesion",
    ):
        """
        Args:
            parent_node (Sofa.Core.Node): node of the object the lesion must be attached to.
            lesion_position (np.ndarray): initial position of the lesion (if no surface is given).
            surface_filename (str, Optional): STL/OBJ surface file describing the lesion shape.
            init_tf (np.ndarray): initial 4x4 transformation to be applied to the model.
            node_name (str): name of the created node.
        """

        Sofa.Core.Controller.__init__(self)

        lesion_node = parent_node.addChild(node_name)

        if surface_filename is not None:
            # ============ 有表面模型：用 STL 作为结节形状 ============
            # 1) 拓扑与机械对象（用于映射和跟随乳腺变形）
            surface_loader = add_loader(
                parent_node=lesion_node,
                filename=surface_filename,
                name="lesion_surface_loader",
                transformation=init_tf,   # 这里一般用单位矩阵即可
            )

            self.topology = add_topology(
                parent_node=lesion_node,
                mesh_loader=surface_loader,
                topology=Topology.TRIANGLE,
            )

            self.state = lesion_node.addObject(
                "MechanicalObject",
                name="LesionMecha",
                template="Vec3d",
                showObject=0,        # 不显示默认小点
                listening=1,
            )

            # 2) 可视化模型：结节 STL + 颜色
            visual_node = lesion_node.addChild(f"Visual{node_name}")
            visual_loader = add_loader(
                parent_node=visual_node,
                filename=surface_filename,
                name="lesion_visual_loader",
                transformation=init_tf,
            )

            add_topology(
                parent_node=visual_node,
                mesh_loader=visual_loader,
                topology=Topology.TRIANGLE,
            )

            add_visual_models(
                parent_node=visual_node,
                color=[0.8, 0.6, 0.0,0.99]   # 红色结节，可改
            )

            visual_node.addObject("BarycentricMapping", name="VisualMapping")

        else:
            # ============ 无表面模型：只用一个点表示结节 ============
            self.state = lesion_node.addObject(
                "MechanicalObject",
                name="FiducialMecha",
                position=lesion_position,
                template="Vec3d",
                showObjectScale=10,
                showObject=1,
                listening=1,
                showColor=[1.0, 0.0, 0.0, 1.0],  # 红点
            )

        # 关键：让结节（surface 或点）跟随乳腺变形
        lesion_node.addObject("BarycentricMapping")

        self.node = lesion_node


    def onAnimateEndEvent(self, __):
        pass



# -------------------------------------------------
# 工具函数
# -------------------------------------------------
def get_bbox(position):
    """
    Gets the bounding box of the object defined by the given vertices.

    Arguments
    -----------
    position : list
        List with the coordinates of N points.

    Returns
    ----------
    xmin, xmax, ymin, ymax, zmin, zmax : floats
        min and max coordinates of the object bounding box.
    """
    points_array = np.asarray(position)
    m = np.min(points_array, axis=0)
    xmin, ymin, zmin = m[0], m[1], m[2]

    m = np.max(points_array, axis=0)
    xmax, ymax, zmax = m[0], m[1], m[2]

    return xmin, xmax, ymin, ymax, zmin, zmax


def is_stable(displacement):
    """
    Analyzes the provided displacement and tells if the displacement is associated with
    a stable deformation (i.e., lower than high_thresh).

    Arguments
    -----------
    displacement : array_like
        Nx3 array with x,y,z displacements of N points.

    Returns
    -----------
    bool
        False if there is at least one displacement with NaN value
    """

    displ_norm = np.linalg.norm(displacement, axis=1)
    max_displ_norm = np.amax(displ_norm)

    if np.isnan(max_displ_norm):
        return False
    else:
        return True