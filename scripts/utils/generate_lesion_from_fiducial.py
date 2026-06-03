import numpy as np
from stl import mesh
import os

def generate_sphere_at_center(center,
                              radius=0.008,  # 半径 3 mm
                              n_theta=8,#经纬线
                              n_phi=8,
                              filename="lesion_nodule.stl"):
    """
    在给定 center 处生成球面三角网格 STL 文件
    center: [cx, cy, cz]，单位 m
    """
    cx, cy, cz = center

    theta = np.linspace(0, 2 * np.pi, n_theta, endpoint=False)
    phi = np.linspace(0, np.pi, n_phi)

    vertices = []
    for i in range(n_phi):
        for j in range(n_theta):
            th = theta[j]
            ph = phi[i]
            x = cx + radius * np.sin(ph) * np.cos(th)
            y = cy + radius * np.sin(ph) * np.sin(th)
            z = cz + radius * np.cos(ph)
            vertices.append([x, y, z])
    vertices = np.array(vertices)
    triangles = []
    for i in range(n_phi - 1):
        for j in range(n_theta):
            p0 = i * n_theta + j
            p1 = (i + 1) * n_theta + j
            p2 = i * n_theta + ((j + 1) % n_theta)
            p3 = (i + 1) * n_theta + ((j + 1) % n_theta)
            triangles.append([p0, p1, p2])
            triangles.append([p2, p1, p3])
    triangles = np.array(triangles)

    sphere_mesh = mesh.Mesh(np.zeros(triangles.shape[0], dtype=mesh.Mesh.dtype))
    for i, f in enumerate(triangles):
        for k in range(3):
            sphere_mesh.vectors[i][k] = vertices[f[k], :]

    sphere_mesh.save(filename)
    print(f"已生成结节 STL 文件: {os.path.abspath(filename)}")
    print(f"  - 球心: {center}")
    print(f"  - 半径: {radius*1000:.1f} mm")
    print(f"  - 三角面片数: {len(triangles)}")


def main():
    # 1) 读取 Fiducial0.txt 里的病灶中心
    lesion_basedir = "./inData/ground_truth/tumor"
    tumorID = 1  # 你可以根据实际情况改，或通过参数传入
    fid_path = f"{lesion_basedir}{tumorID}/Fiducial0.txt"

    center = np.loadtxt(fid_path)
    center = np.asarray(center).reshape(-1)
    assert center.size == 3, "Fiducial0.txt 中应包含 3 个坐标值"

    # 2) 生成以该点为球心的 STL
    output_stl = "./inData/lesion_nodule.stl"
    generate_sphere_at_center(
        center=center,
        filename=output_stl
    )

if __name__ == "__main__":
    main()