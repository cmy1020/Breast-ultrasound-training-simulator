import numpy as np

with open("./inData/liver.msh", "r") as f:
    lines = f.readlines()

coords = []
indices = []
in_nodes = False
for line in lines:
    if "$NOD" in line:
        in_nodes = True
        continue
    if "$ENDNOD" in line:
        break
    if in_nodes:
        parts = line.strip().split()
        if len(parts) == 4:
            indices.append(int(parts[0]) - 1)  # SOFA索引从0开始
            coords.append([float(parts[1]), float(parts[2]), float(parts[3])])

coords = np.array(coords)

# 选Y轴最大的前20%节点作为固定点（肝脏顶部/背面，模拟韧带附着）
y_threshold = np.percentile(coords[:, 1], 80)
fixed = [indices[i] for i, c in enumerate(coords) if c[1] >= y_threshold]

np.savetxt("./inData/fixednodes_liver.txt", fixed, fmt="%d")
print(f"固定节点数: {len(fixed)} / {len(coords)}")
print(f"Y阈值: {y_threshold:.3f} cm")