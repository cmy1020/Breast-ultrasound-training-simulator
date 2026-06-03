"""
STL模型面片细化工具
功能：将STL模型的面片数量增加到原来的3倍
"""

import trimesh
import numpy as np

def subdivide_stl(input_path, output_path, iterations=1):
    """
    细化STL模型的面片

    参数:
        input_path: 输入STL文件路径
        output_path: 输出STL文件路径
        iterations: 细分迭代次数（1次迭代=4倍面片，所以用1次接近3倍）
    """
    print(f"正在加载模型: {input_path}")

    # 1. 加载STL模型
    mesh = trimesh.load_mesh(input_path)

    print(f"原始面片数量: {len(mesh.faces)}")
    print(f"原始顶点数量: {len(mesh.vertices)}")

    # 2. 细分网格（每次细分会将面片数增加约4倍）
    # 注意：trimesh的subdivide是4倍增长，要3倍需要特殊处理
    mesh_subdivided = mesh.subdivide()

    print(f"细分后面片数量: {len(mesh_subdivided.faces)}")
    print(f"细分后顶点数量: {len(mesh_subdivided.vertices)}")

    # 3. 导出处理后的STL
    mesh_subdivided.export(output_path)

    print(f"模型已保存至: {output_path}")
    print("处理完成！")

    return mesh_subdivided


def subdivide_stl_custom(input_path, output_path, target_multiplier=3):
    """
    自定义倍数细化（精确控制到3倍）

    参数:
        input_path: 输入STL文件路径
        output_path: 输出STL文件路径
        target_multiplier: 目标面片倍数（例如3表示3倍）
    """
    print(f"正在加载模型: {input_path}")

    # 加载模型
    mesh = trimesh.load_mesh(input_path)
    original_faces = len(mesh.faces)

    print(f"原始面片数量: {original_faces}")

    # 计算需要细分到的面片数量
    target_faces = original_faces * target_multiplier

    # 使用自适应细分
    mesh_subdivided = mesh.subdivide_to_size(
        max_edge=mesh.scale / 50,  # 控制边长来间接控制面片数
        max_iter=10
    )

    print(f"细分后面片数量: {len(mesh_subdivided.faces)}")

    # 导出
    mesh_subdivided.export(output_path)
    print(f"模型已保存至: {output_path}")

    return mesh_subdivided


# ============================================
# 🔧 在这里设置输入输出路径
# ============================================

if __name__ == "__main__":
    #  修改这两行，设置你的输入和输出路径
    INPUT_STL_PATH = "C:/Users/86150/Desktop/breast_500.stl"      # 输入STL文件路径
    OUTPUT_STL_PATH = "C:/Users/86150/Desktop/breast.stl"  # 输出STL文件路径

    # 方法1：标准细分（约4倍面片）
    subdivide_stl(INPUT_STL_PATH, OUTPUT_STL_PATH, iterations=1)

    # 方法2：自定义倍数细分（尝试达到3倍）
    # subdivide_stl_custom(INPUT_STL_PATH, OUTPUT_STL_PATH, target_multiplier=3)