# 创建新文件：check_mesh.py
import numpy as np
import meshio


def check_mesh_quality(mesh_file):
    """检查网格质量"""
    mesh = meshio.read(mesh_file)

    points = mesh.points
    if 'tetra' in mesh.cells_dict:
        cells = mesh.cells_dict['tetra']
    else:
        print("错误：未找到四面体网格")
        return

    print(f"网格信息:")
    print(f"  节点数: {len(points)}")
    print(f"  四面体数: {len(cells)}")

    # 计算四面体质量指标
    volumes = []
    aspect_ratios = []

    for i, tet in enumerate(cells[:1000]):  # 抽样1000个
        v0, v1, v2, v3 = points[tet]

        # 计算体积
        vol = np.abs(np.dot(v1 - v0, np.cross(v2 - v0, v3 - v0)) / 6.0)
        volumes.append(vol)

        # 计算最长边和最短边
        edges = [
            np.linalg.norm(v1 - v0),
            np.linalg.norm(v2 - v0),
            np.linalg.norm(v3 - v0),
            np.linalg.norm(v2 - v1),
            np.linalg.norm(v3 - v1),
            np.linalg.norm(v3 - v2),
        ]
        aspect_ratio = max(edges) / min(edges)
        aspect_ratios.append(aspect_ratio)

    volumes = np.array(volumes)
    aspect_ratios = np.array(aspect_ratios)

    print(f"\n质量统计:")
    print(f"  体积范围: {volumes.min():.2e} - {volumes.max():.2e}")
    print(f"  平均体积: {volumes.mean():.2e}")
    print(f"  纵横比范围: {aspect_ratios.min():.2f} - {aspect_ratios.max():.2f}")
    print(f"  平均纵横比: {aspect_ratios.mean():.2f}")

    # 警告检测
    if volumes.min() < 1e-9:
        print("⚠️  警告：存在退化四面体（体积极小）")

    if aspect_ratios.max() > 10:
        print(f"⚠️  警告：存在狭长四面体（纵横比>{aspect_ratios.max():.1f}）")

    if aspect_ratios.mean() > 5:
        print("⚠️  警告：网格整体质量较差")


if __name__ == '__main__':
    check_mesh_quality('./inData/breast_13k.msh')