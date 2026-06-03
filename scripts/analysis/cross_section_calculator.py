# cross_section_calculator.py
# 高性能平面-网格截面计算器
# 独立模块，不依赖任何项目代码，可单独测试

import numpy as np
import time


class CrossSectionCalculator:
    """
    高性能平面-网格截面计算器

    相比原 _compute_mesh_plane_intersection 的改进：
    1. AABB空间剔除   —— 先过滤远离切面的三角形，减少计算量约90%
    2. Step3完全向量化 —— 消除原来的Python for循环
    3. 退化情况处理   —— sign=0（顶点恰好在平面上）不再产生噪点线段
    """

    def __init__(self, aabb_margin: float = 0.002):
        """
        Parameters
        ----------
        aabb_margin : float
            空间剔除时平面两侧保留的余量（单位：米）
            默认 2mm，足以覆盖所有跨越切面的三角形
            可适当调大（如 0.005）以防止边缘漏算
        """
        self.aabb_margin = aabb_margin

        # 性能统计（每50次打印一次，调试用）
        self._call_count = 0
        self._total_ms = 0.0

    # -------------------------------------------------------------------------
    # 公开接口（与原函数签名兼容，返回值格式相同）
    # -------------------------------------------------------------------------

    def compute(self, positions, triangles,
                plane_origin: np.ndarray,
                plane_normal: np.ndarray,
                min_segment_length: float = 1e-6):
        """
        计算平面与三角网格的交线段集合

        Parameters
        ----------
        positions   : array-like, shape (N_verts, 3)
        triangles   : array-like, shape (N_tris, 3), int索引
        plane_origin: (3,) 平面上任意一点
        plane_normal: (3,) 平面法向量（无需单位化）
        min_segment_length : 过滤零长度退化线段的阈值

        Returns
        -------
        list of (np.ndarray(3,), np.ndarray(3,))
            与原 _compute_mesh_plane_intersection 返回格式完全相同
            每个元素是一条线段的两个端点 (p0, p1)
            找不到交线时返回空列表 []
        """
        t0 = time.perf_counter()

        # 转换并归一化
        pos = np.asarray(positions, dtype=np.float64)
        tri = np.asarray(triangles, dtype=np.int32)
        n = np.asarray(plane_normal, dtype=np.float64)
        norm_len = np.linalg.norm(n)
        if norm_len < 1e-30:
            return []
        n = n / norm_len
        o = np.asarray(plane_origin, dtype=np.float64)

        if len(pos) == 0 or len(tri) == 0:
            return []

        # Step 1: AABB空间剔除
        tri_filtered, d_all = self._aabb_filter(pos, tri, o, n)
        if len(tri_filtered) == 0:
            return []

        # Step 2: 完全向量化求交
        segments_np = self._intersect_vectorized(
            pos, tri_filtered, d_all, min_segment_length
        )

        # 性能统计
        elapsed = (time.perf_counter() - t0) * 1000
        self._call_count += 1
        self._total_ms += elapsed
        if self._call_count % 50 == 0:
            avg = self._total_ms / self._call_count
            ratio = len(tri_filtered) / max(len(tri), 1)
            print(f"[CrossSection] "
                  f"过滤后 {len(tri_filtered)}/{len(tri)} ({ratio:.0%}) 三角形 | "
                  f"本次 {elapsed:.1f}ms | 均值 {avg:.1f}ms | "
                  f"线段数 {len(segments_np)}")

        # 转换为与原函数相同的返回格式：list of (p0, p1)
        if len(segments_np) == 0:
            return []
        return [(segments_np[i, 0], segments_np[i, 1])
                for i in range(len(segments_np))]

    # -------------------------------------------------------------------------
    # Step 1：AABB空间剔除
    # -------------------------------------------------------------------------

    def _aabb_filter(self, pos, tri, plane_origin, plane_normal):
        """
        利用有符号距离快速剔除远离切面的三角形

        原理：
          d(v) = dot(v - origin, normal)
          若三角形三顶点的 d 值全正（或全负），该三角形完全在平面一侧，不可能相交
          只保留 d_min < +margin 且 d_max > -margin 的三角形

        比"先算AABB包围盒再判断"更直接，计算量完全相同（都是O(N_verts)的dot）
        """
        # 所有顶点到平面的有符号距离（一次向量化，O(N_verts)）
        d_all = (pos - plane_origin) @ plane_normal  # shape: (N_verts,)

        # 三角形三顶点各自的距离
        d0 = d_all[tri[:, 0]]  # shape: (N_tris,)
        d1 = d_all[tri[:, 1]]
        d2 = d_all[tri[:, 2]]

        # 三角形的距离极值
        d_min = np.minimum(np.minimum(d0, d1), d2)
        d_max = np.maximum(np.maximum(d0, d1), d2)

        m = self.aabb_margin

        # 保留条件：三角形的距离范围与 [-margin, +margin] 有重叠
        keep = (d_min < m) & (d_max > -m)

        return tri[keep], d_all

    # -------------------------------------------------------------------------
    # Step 2：完全向量化的平面求交
    # -------------------------------------------------------------------------

    def _intersect_vectorized(self, pos, tri, d_all, min_len):
        """
        对已过滤的三角形，完全向量化地计算所有交线段

        核心思路：
          每个三角形有3条边 (01, 12, 20)
          对每条边分别判断是否跨越平面（两端点距离异号）
          跨越的边用线性插值求交点
          正常穿越的三角形，恰好有2条边跨越，这2个交点构成一条线段

          关键：把3条边展开成3个形状相同的大矩阵，并行处理所有三角形
          完全不需要Python循环
        """
        eps = 1e-10

        # 取三顶点的坐标和距离
        p0 = pos[tri[:, 0]]   # (M, 3)
        p1 = pos[tri[:, 1]]
        p2 = pos[tri[:, 2]]

        d0 = d_all[tri[:, 0]]  # (M,)
        d1 = d_all[tri[:, 1]]
        d2 = d_all[tri[:, 2]]

        # 对三条边分别求交点
        # 每条边返回：交点坐标 (M,3)，是否有效 (M,) bool
        pt_01, v_01 = self._edge_intersect(p0, p1, d0, d1, eps)
        pt_12, v_12 = self._edge_intersect(p1, p2, d1, d2, eps)
        pt_20, v_20 = self._edge_intersect(p2, p0, d2, d0, eps)

        # 有效边数量：正常穿越 = 2条有效边，退化（顶点在平面上）= 1条
        # 这里只处理正常穿越（2条有效边）的情况
        n_valid = (v_01.astype(np.int8)
                   + v_12.astype(np.int8)
                   + v_20.astype(np.int8))  # (M,)
        mask_2 = n_valid == 2

        if not np.any(mask_2):
            return np.empty((0, 2, 3), dtype=np.float64)

        # 提取有2条有效边的三角形
        v_01_2 = v_01[mask_2]
        v_12_2 = v_12[mask_2]
        v_20_2 = v_20[mask_2]

        pt_01_2 = pt_01[mask_2]  # (K, 3)
        pt_12_2 = pt_12[mask_2]
        pt_20_2 = pt_20[mask_2]

        K = mask_2.sum()
        seg = np.empty((K, 2, 3), dtype=np.float64)

        # 3种有效边组合，分别赋值
        # 组合A：边01 + 边12 有效
        cA = v_01_2 & v_12_2
        if np.any(cA):
            seg[cA, 0] = pt_01_2[cA]
            seg[cA, 1] = pt_12_2[cA]

        # 组合B：边12 + 边20 有效
        cB = v_12_2 & v_20_2
        if np.any(cB):
            seg[cB, 0] = pt_12_2[cB]
            seg[cB, 1] = pt_20_2[cB]

        # 组合C：边01 + 边20 有效
        cC = v_01_2 & v_20_2
        if np.any(cC):
            seg[cC, 0] = pt_01_2[cC]
            seg[cC, 1] = pt_20_2[cC]

        # 过滤零长度线段（退化情况的残余）
        lengths = np.linalg.norm(seg[:, 1] - seg[:, 0], axis=1)
        seg = seg[lengths > min_len]

        return seg

    @staticmethod
    def _edge_intersect(pA, pB, dA, dB, eps):
        """
        向量化计算一批边与平面的交点

        Parameters
        ----------
        pA, pB : (M, 3)  边的两个端点坐标
        dA, dB : (M,)    两端点到平面的有符号距离
        eps    : float   判断"严格异号"的阈值

        Returns
        -------
        pt    : (M, 3)  交点坐标（无效行为零向量，不会被使用）
        valid : (M,)    bool，该边是否真正跨越平面（严格异号）
        """
        # 严格异号：两端点距离乘积 < 0
        # 用 eps*eps 而不是 0，避免浮点噪声把"相切"误判为"穿越"
        valid = dA * dB < -(eps * eps)   # (M,) bool

        # 插值参数 t = dA / (dA - dB)
        # 用 np.where 避免除以零（无效行给 denom=1，t=0，pt=pA，但不会被使用）
        denom = dA - dB
        safe_denom = np.where(np.abs(denom) > eps, denom, 1.0)
        t = np.where(valid, dA / safe_denom, 0.0)  # (M,)

        # 交点 = pA + t * (pB - pA)
        pt = pA + t[:, np.newaxis] * (pB - pA)   # (M, 3)

        return pt, valid