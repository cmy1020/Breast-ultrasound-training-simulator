import math
import os

# 物理模型参数（与CHAI3D代码中的参数完全一致）
hemisphere_radius = 0.4  # 对应C++中的hemisphereRadius
rings = 15  # 对应hemisphereRings
sectors = 48  # 对应hemisphereSectors
save_path = r"C:\Users\86150\Desktop"  # 保存路径
output_file = os.path.join(save_path, "hemisphere_matched.obj")  # 完整文件路径

# 确保保存目录存在
if not os.path.exists(save_path):
    os.makedirs(save_path)
    print(f"创建目录: {save_path}")

# 生成顶点、纹理坐标和法向量
vertices = []
tex_coords = []
normals = []

for i in range(rings + 1):
    # 计算当前环的角度（从顶部0到底部π/2）
    phi = math.pi / 2 * i / rings  # 与物理模型计算完全一致
    sin_phi = math.sin(phi)
    cos_phi = math.cos(phi)

    for j in range(sectors):
        # 计算当前扇区的角度
        theta = 2 * math.pi * j / sectors  # 与物理模型计算完全一致
        sin_theta = math.sin(theta)
        cos_theta = math.cos(theta)

        # 顶点坐标计算（与物理质点位置公式完全相同）
        x = hemisphere_radius * sin_phi * cos_theta
        y = hemisphere_radius * sin_phi * sin_theta
        z = hemisphere_radius * cos_phi  # Z轴向上，与物理模型一致
        vertices.append((x, y, z))

        # 纹理坐标
        u = 1 - (j / sectors)
        v = 1 - (i / rings)
        tex_coords.append((u, v))

        # 法向量（指向外侧）
        nx = sin_phi * cos_theta
        ny = sin_phi * sin_theta
        nz = cos_phi
        normals.append((nx, ny, nz))

# 生成三角面（与物理骨架拓扑匹配）
faces = []
for i in range(rings):
    for j in range(sectors):
        v0 = i * sectors + j
        v1 = (i + 1) * sectors + j
        v2 = (i + 1) * sectors + (j + 1) % sectors
        v3 = i * sectors + (j + 1) % sectors

        # 生成两个三角面
        faces.append((v0 + 1, v1 + 1, v2 + 1))  # OBJ索引从1开始
        faces.append((v0 + 1, v2 + 1, v3 + 1))

# 写入OBJ文件
try:
    with open(output_file, 'w') as f:
        # 写入顶点
        for v in vertices:
            f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")

        # 写入纹理坐标
        for vt in tex_coords:
            f.write(f"vt {vt[0]:.6f} {vt[1]:.6f}\n")

        # 写入法向量
        for vn in normals:
            f.write(f"vn {vn[0]:.6f} {vn[1]:.6f} {vn[2]:.6f}\n")

        # 写入面
        for face in faces:
            f.write(f"f {face[0]}/{face[0]}/{face[0]} {face[1]}/{face[1]}/{face[1]} {face[2]}/{face[2]}/{face[2]}\n")

    print(f"OBJ文件已成功保存到：{output_file}")
    print(f"模型参数：半径={hemisphere_radius}，环数={rings}，扇区数={sectors}")
except Exception as e:
    print(f"保存文件失败：{str(e)}")
    print("请检查路径是否正确或是否有写入权限")
