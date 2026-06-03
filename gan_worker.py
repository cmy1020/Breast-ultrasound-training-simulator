# =============================================================================
# gan_worker.py  ── 完整版（修复重复定义问题）
# =============================================================================

import threading
import queue
import time
import os
import numpy as np
from PIL import Image
import torch
import torchvision.transforms as transforms

from GAN_train import GeneratorUNet


class GANWorker:

    def __init__(self,
                 model_path="checkpoints/generator_best.pth",
                 image_size=256):

        self.model_path = model_path
        self.image_size = image_size
        self.device     = torch.device("cuda")

        self._input_queue  = queue.Queue(maxsize=1)
        self._output_queue = queue.Queue(maxsize=1)

        self.is_running  = False
        self.is_loaded   = False
        self._thread     = None
        self._generator  = None

        self._infer_count   = 0
        self._total_time_ms = 0.0
        self._last_infer_ms = 0.0

        # 变化检测哈希
        self._last_breast_hash = None
        self._last_lesion_hash = None
        self._last_raw_hash    = None

        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])

    # ─────────────────────────────────────────────────────────────────────────
    # 启动 / 停止
    # ─────────────────────────────────────────────────────────────────────────

    def start(self):
        if self.is_running:
            return
        self._load_model()
        self._warmup()
        self.is_running = True
        self._thread = threading.Thread(
            target=self._worker_loop,
            name="GANWorkerThread",
            daemon=True
        )
        self._thread.start()
        status = "GPU推理" if self.is_loaded else "模型未加载"
        print(f"✓ GAN后台线程已启动 [{self.device} | {status}]")

    def stop(self):
        self.is_running = False
        try:
            self._input_queue.put_nowait(None)
        except queue.Full:
            pass
        if self._thread:
            self._thread.join(timeout=0.5)
        print("✓ GAN后台线程已停止")

    # ─────────────────────────────────────────────────────────────────────────
    # 公开提交接口
    # ─────────────────────────────────────────────────────────────────────────

    def submit_raw(self, lesion_segments_3d, probe_transform):
        """
        提交原始3D线段数据
        结节有无状态 + 探头位置 共同参与哈希
        确保"有结节→无结节"的状态切换必然触发新推理
        """
        if not self.is_loaded:
            return

        probe_pos = np.array(probe_transform.get('position', np.zeros(3)))

        has_lesion = len(lesion_segments_3d) > 0 if lesion_segments_3d else False

        if has_lesion:
            # 有结节：用线段端点坐标 + 探头位置 + True标记
            sample_pts = []
            for seg in lesion_segments_3d[:5]:
                p0, p1 = seg
                sample_pts.extend([float(p0[0]), float(p0[1]), float(p0[2])])
            new_hash = (
                round(sum(sample_pts), 4),
                round(float(probe_pos.sum()), 4),
                True  # ← 结节存在标记
            )
        else:
            # 无结节：哈希中包含 False 标记
            # 当从"有结节(True)"→"无结节(False)"时，哈希必然不同，强制触发推理
            new_hash = (
                round(float(probe_pos.sum()), 4),
                False  # ← 无结节标记，与有结节时的 True 永不相等
            )

        if new_hash == self._last_raw_hash:
            return
        self._last_raw_hash = new_hash

        payload = {
            'mode': 'raw',
            'lesion_segments': lesion_segments_3d,
            'probe_transform': probe_transform,
        }
        self._put_payload(payload)

    def submit(self, breast_vertices_2d, lesion_vertices_2d=None):
        """
        旧接口：保留兼容性（目前已不使用）
        """
        if not self.is_loaded:
            return

        new_lesion_hash = self._hash_contour(lesion_vertices_2d)
        new_breast_hash = 'no_lesion' if lesion_vertices_2d is None else 'has_lesion'

        if (new_breast_hash == self._last_breast_hash and
                new_lesion_hash == self._last_lesion_hash):
            return

        self._last_breast_hash = new_breast_hash
        self._last_lesion_hash = new_lesion_hash

        payload = {
            'mode'  : 'legacy',
            'breast': breast_vertices_2d,
            'lesion': lesion_vertices_2d,
        }
        self._put_payload(payload)

    def _put_payload(self, payload):
        """放入队列（满了先清空）"""
        try:
            self._input_queue.get_nowait()
        except queue.Empty:
            pass
        try:
            self._input_queue.put_nowait(payload)
        except queue.Full:
            pass

    def get_latest_result(self):
        result = None
        while True:
            try:
                result = self._output_queue.get_nowait()
            except queue.Empty:
                break
        return result

    def get_stats(self):
        avg = (self._total_time_ms / self._infer_count
               if self._infer_count > 0 else 0.0)
        return {
            'count'  : self._infer_count,
            'avg_ms' : avg,
            'last_ms': self._last_infer_ms,
            'loaded' : self.is_loaded,
        }

    def clear_output(self):
        """清空队列（探头离开乳腺时调用）"""
        self._last_breast_hash = None
        self._last_lesion_hash = None
        self._last_raw_hash    = None
        while True:
            try:
                self._output_queue.get_nowait()
            except queue.Empty:
                break

    # ─────────────────────────────────────────────────────────────────────────
    # 后台线程主循环（只有一个！）
    # ─────────────────────────────────────────────────────────────────────────

    def _worker_loop(self):
        print("  GAN推理线程：开始运行")

        while self.is_running:
            try:
                payload = self._input_queue.get(timeout=1.0)

                if payload is None:
                    break

                t0   = time.perf_counter()
                mode = payload.get('mode', 'legacy')

                # ── 根据模式选择推理方法 ──────────────────────────────────
                if mode == 'raw':
                    result = self._infer_raw(
                        payload['lesion_segments'],
                        payload['probe_transform']
                    )
                else:
                    result = self._infer_legacy(
                        payload['breast'],
                        payload['lesion']
                    )

                elapsed_ms = (time.perf_counter() - t0) * 1000
                self._infer_count   += 1
                self._total_time_ms += elapsed_ms
                self._last_infer_ms  = elapsed_ms

                if self._infer_count % 30 == 0:
                    avg = self._total_time_ms / self._infer_count
                    print(f"  📊 GAN: 第{self._infer_count}次 | "
                          f"本次{elapsed_ms:.0f}ms | 均值{avg:.0f}ms")

                # 放入输出队列
                try:
                    self._output_queue.put_nowait(result)
                except queue.Full:
                    try:
                        self._output_queue.get_nowait()
                    except queue.Empty:
                        pass
                    try:
                        self._output_queue.put_nowait(result)
                    except queue.Full:
                        pass

            except queue.Empty:
                continue
            except Exception as e:
                print(f"  ⚠️ GAN推理线程异常: {e}")
                import traceback
                traceback.print_exc()

        print("  GAN推理线程：已退出")

    # ─────────────────────────────────────────────────────────────────────────
    # 推理方法
    # ─────────────────────────────────────────────────────────────────────────

    def _infer_raw(self, lesion_segments_3d, probe_transform):
        """新方法：用原始3D线段生成Mask并推理"""
        mask_pil = self._draw_mask_from_segments(lesion_segments_3d, probe_transform)
        return self._run_generator(mask_pil)

    def _infer_legacy(self, breast_pts, lesion_pts):
        """旧方法：用2D轮廓点生成Mask并推理（兼容旧接口）"""
        mask_pil = self._draw_mask_legacy(breast_pts, lesion_pts)
        return self._run_generator(mask_pil)

    def _run_generator(self, mask_pil):
        """公共：PIL图像 → GAN推理 → numpy数组"""
        tensor = self.transform(mask_pil).unsqueeze(0).to(self.device)
        with torch.no_grad():
            output = self._generator(tensor)
        arr = output[0, 0].cpu().numpy()
        return np.clip((arr + 1.0) / 2.0 * 255, 0, 255).astype(np.uint8)

    # ─────────────────────────────────────────────────────────────────────────
    # Mask生成方法
    # ─────────────────────────────────────────────────────────────────────────

    def _draw_mask_from_segments(self, lesion_segments_3d, probe_transform):
        """
        新方法：把3D截面线段直接投影到探头局部XY平面
        """
        import cv2

        S = self.image_size

        # ── 全图绿色背景 ──────────────────────────────────────────────────────
        sim_img = np.full((S, S), 200, dtype=np.uint8)
        _, tissue = cv2.threshold(sim_img, 15, 255, cv2.THRESH_BINARY)
        kernel = np.ones((5, 5), np.uint8)
        tissue = cv2.morphologyEx(tissue, cv2.MORPH_CLOSE, kernel, iterations=3)
        mask_bgr = np.zeros((S, S, 3), dtype=np.uint8)
        mask_bgr[tissue == 255] = [0, 255, 0]

        # ── ✅ 新增：底部随机黑色区域（模拟超声图的边界变化）────────────────
        # 原理：真实超声图中，探头位置不同时
        #       底部边界形状会略有不同（因为乳腺厚度/形状变化）
        #       用探头位置作为随机种子，保证同一位置输出一致
        #       探头移动时自动变化
        probe_pos = np.array(probe_transform.get('position', np.zeros(3)))
        probe_quat = np.array(probe_transform.get('quaternion', np.array([0, 0, 0, 1])))

        # 用探头位置生成确定性随机种子
        # 如果想让变化更敏感（探头稍微移动就变化）：
        # 增大种子的精度
        seed = int(abs(probe_pos.sum() * 1e6)) % (2 ** 31) # 1e6改1e7更敏感 1e6改1e5不敏感
        rng = np.random.default_rng(seed)

        # 随机黑色区域参数
        # n_patches: 生成1~3个随机黑色小斑块
        # 位置：偏向底部（60%~95%高度范围）
        # 大小：较小，不影响主体 改为 (1, 3) 可以减少数量
        n_patches = rng.integers(1, 4)  # 1~3个

        for _ in range(n_patches):
            # 黑色斑块的中心位置（底部区域）
            cx = int(rng.integers(int(S * 0.1), int(S * 0.9)))
            cy = int(rng.integers(int(S * 0.6), int(S * 0.95)))

            # 斑块大小（宽度5%~15%，高度3%~10%）# 改小上限可以减少黑色区域
            rw = int(rng.integers(int(S * 0.05), int(S * 0.15)))
            rh = int(rng.integers(int(S * 0.03), int(S * 0.10)))

            # 随机旋转角度
            angle = int(rng.integers(0, 180))

            # 画黑色椭圆斑块（只在绿色区域内）
            black_patch = np.zeros((S, S), dtype=np.uint8)
            cv2.ellipse(black_patch,
                        center=(cx, cy),
                        axes=(rw, rh),
                        angle=angle,
                        startAngle=0, endAngle=360,
                        color=255, thickness=-1)

            # 只在已有绿色的区域内涂黑
            valid_black = (black_patch == 255) & (tissue == 255)
            mask_bgr[valid_black] = [0, 0, 0]  # BGR黑色

        # ── 结节区域（红色）────────────────────────────────────────────────────
        if lesion_segments_3d and len(lesion_segments_3d) > 0:

            local_x = self._rotate_vec(np.array([1., 0., 0.]), probe_quat)
            local_y = self._rotate_vec(np.array([0., 1., 0.]), probe_quat)

            all_2d = []
            for p0_3d, p1_3d in lesion_segments_3d:
                for p3d in [p0_3d, p1_3d]:
                    v = np.array(p3d) - probe_pos
                    x2d = float(np.dot(v, local_x))
                    y2d = float(np.dot(v, local_y))
                    all_2d.append([x2d, y2d])

            if len(all_2d) >= 3:
                pts_2d = np.array(all_2d, dtype=np.float32)
                xmin, ymin = pts_2d.min(axis=0)
                xmax, ymax = pts_2d.max(axis=0)
                x_range = max(xmax - xmin, 1e-6)
                y_range = max(ymax - ymin, 1e-6)

                target_px = S * 0.20
                scale = target_px / max(x_range, y_range)
                cx_c, cy_c = S // 2, S // 2

                px_list = []
                for x2, y2 in pts_2d:
                    px = int((x2 - (xmin + xmax) / 2) * scale + cx_c)
                    py = int((y2 - (ymin + ymax) / 2) * scale + cy_c)
                    px_list.append([
                        int(np.clip(px, 0, S - 1)),
                        int(np.clip(py, 0, S - 1))
                    ])

                px_arr = np.array(px_list, dtype=np.int32)
                lesion_mask = np.zeros((S, S), dtype=np.uint8)
                hull = cv2.convexHull(px_arr)
                area = cv2.contourArea(hull)

                if area < 50:
                    min_r = max(int(S * 0.08), 10)
                    hull_pts = hull.reshape(-1, 2)
                    cx_h = int(np.mean(hull_pts[:, 0]))
                    cy_h = int(np.mean(hull_pts[:, 1]))
                    rx = max(int((hull_pts[:, 0].max() - hull_pts[:, 0].min()) / 2), min_r)
                    ry = min_r
                    cv2.ellipse(lesion_mask, (cx_h, cy_h), (rx, ry),
                                0, 0, 360, 255, -1)
                else:
                    cv2.fillConvexPoly(lesion_mask, hull, 255)

                # 结节盖在黑色斑块上方（结节优先级更高）
                valid = (lesion_mask == 255) & (tissue == 255)
                mask_bgr[valid] = [0, 0, 255]

        return Image.fromarray(cv2.cvtColor(mask_bgr, cv2.COLOR_BGR2RGB))

    def _draw_mask_legacy(self, breast_pts, lesion_pts):
        """旧方法：全图绿色 + 2D轮廓点画结节（已弃用，保留兼容）"""
        import cv2

        S        = self.image_size
        sim_img  = np.full((S, S), 200, dtype=np.uint8)
        _, tissue = cv2.threshold(sim_img, 15, 255, cv2.THRESH_BINARY)
        kernel   = np.ones((5, 5), np.uint8)
        tissue   = cv2.morphologyEx(tissue, cv2.MORPH_CLOSE, kernel, iterations=3)
        mask_bgr = np.zeros((S, S, 3), dtype=np.uint8)
        mask_bgr[tissue == 255] = [0, 255, 0]

        return Image.fromarray(cv2.cvtColor(mask_bgr, cv2.COLOR_BGR2RGB))

    # ─────────────────────────────────────────────────────────────────────────
    # 工具方法
    # ─────────────────────────────────────────────────────────────────────────

    def _rotate_vec(self, vec, quat):
        qx, qy, qz, qw = quat
        t = 2.0 * np.array([
            qy*vec[2] - qz*vec[1],
            qz*vec[0] - qx*vec[2],
            qx*vec[1] - qy*vec[0]
        ])
        return vec + qw*t + np.array([
            qy*t[2] - qz*t[1],
            qz*t[0] - qx*t[2],
            qx*t[1] - qy*t[0]
        ])

    def _load_model(self):
        if not os.path.exists(self.model_path):
            print(f"⚠️  GAN模型未找到: {self.model_path}")
            return
        try:
            self._generator = GeneratorUNet(
                in_channels=3, out_channels=1
            ).to(self.device)
            ckpt = torch.load(self.model_path, map_location=self.device)
            if isinstance(ckpt, dict) and 'generator' in ckpt:
                self._generator.load_state_dict(ckpt['generator'])
                print(f"✓ GAN模型加载成功 [Epoch {ckpt.get('epoch','?')}]")
            else:
                self._generator.load_state_dict(ckpt)
                print(f"✓ GAN模型加载成功")
            self._generator.eval()
            self.is_loaded = True
            mem = torch.cuda.memory_allocated() / 1024**2
            print(f"  VRAM占用: {mem:.0f}MB")
        except Exception as e:
            print(f"❌ GAN模型加载失败: {e}")
            self.is_loaded = False

    def _warmup(self):
        """
        GPU预热：模型加载后立即执行空推理
        消除第一次真实推理时的CUDA初始化延迟（通常需要2~5秒）
        """
        if not self.is_loaded:
            return
        print("  GAN预热中...")
        dummy = torch.zeros(1, 3, self.image_size, self.image_size,
                            device=self.device)
        try:
            with torch.no_grad():
                for i in range(3):
                    t0 = time.perf_counter()
                    _ = self._generator(dummy)
                    torch.cuda.synchronize()
                    elapsed = (time.perf_counter() - t0) * 1000
                    print(f"  预热第{i+1}次: {elapsed:.0f}ms")
            print("  ✓ GAN预热完成")
        except Exception as e:
            print(f"  ⚠️ 预热失败（不影响正常运行）: {e}")

    @staticmethod
    def _hash_contour(pts):
        if pts is None or len(pts) == 0:
            return None
        step = max(1, len(pts) // 8)
        return round(float(pts[::step].sum()), 4)