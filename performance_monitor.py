# =============================================================================
# performance_monitor.py  ── 完整修复版
# 系统性能指标采集模块
# 修复内容：
#   1. 正确区分 FEM步长 和 端到端延迟
#   2. 过滤初始化阶段异常数据（前50帧）
#   3. 过滤渲染时的异常峰值（>200ms）
#   4. 删除无效的 ui_interval 指标
# =============================================================================

import time
import json
import numpy as np
from collections import deque
from datetime import datetime


class PerformanceMonitor:

    def __init__(self, window_size=1000, warmup_frames=50):
        self.window_size   = window_size
        self.warmup_frames = warmup_frames  # 跳过前N帧的初始化数据

        # ── 各模块独立计时缓冲 ──────────────────────────────────────
        self.fem_times     = deque(maxlen=window_size)  # 仅FEM求解耗时 ms
        self.cross_times   = deque(maxlen=window_size)  # 仅切面提取耗时 ms
        self.render_times  = deque(maxlen=window_size)  # 仅3D渲染耗时 ms
        self.gan_times     = deque(maxlen=window_size)  # GAN推理耗时 ms（异步）
        self.e2e_times     = deque(maxlen=window_size)  # 端到端单步总耗时 ms

        # ── 计时起点（每个模块独立）────────────────────────────────
        self._t_fem        = None
        self._t_cross      = None
        self._t_render     = None
        self._t_step       = None

        # ── 帧计数 ──────────────────────────────────────────────────
        self.total_frames  = 0
        self._fps_buf      = deque(maxlen=200)  # 用于计算FPS

        # ── 日志间隔 ────────────────────────────────────────────────
        self._log_every    = 50    # 每N帧打印一次摘要
        self._save_every   = 500   # 每N帧自动保存一次

        self.is_active     = True
        print("✓ PerformanceMonitor 已启动 "
              f"(预热帧数={warmup_frames})")

    # ─────────────────────────────────────────────────────────────────
    # 公开计时接口
    # ─────────────────────────────────────────────────────────────────

    def step_begin(self):
        """
        在 simulation_step() 最顶部调用。
        记录本步骤的开始时间，用于计算端到端延迟。
        """
        if not self.is_active:
            return
        self._t_step = time.perf_counter()

    def step_end(self):
        """
        在 simulation_step() try块的最末尾调用。
        计算本步骤从开始到结束的总耗时（端到端延迟）。
        """
        if not self.is_active or self._t_step is None:
            return

        elapsed = (time.perf_counter() - self._t_step) * 1000
        self.total_frames += 1
        self._fps_buf.append(time.perf_counter())

        # 跳过预热帧
        if self.total_frames > self.warmup_frames:
            self.e2e_times.append(elapsed)

        # 定期输出
        if self.total_frames % self._log_every == 0:
            self._print_summary()
        if self.total_frames % self._save_every == 0:
            self.save_report()

    def fem_begin(self):
        """
        在 Sofa.Simulation.animate() 之前调用。
        只计量 FEM 求解本身的耗时。
        """
        if not self.is_active:
            return
        self._t_fem = time.perf_counter()

    def fem_end(self):
        """
        在 Sofa.Simulation.animate() 之后立即调用。
        """
        if not self.is_active or self._t_fem is None:
            return
        elapsed = (time.perf_counter() - self._t_fem) * 1000
        if self.total_frames > self.warmup_frames:
            self.fem_times.append(elapsed)
        self._t_fem = None

    def cross_begin(self):
        """
        在 _compute_current_cross_section() 函数入口调用。
        注意：由于每5帧才真正计算一次，
        实际记录的是"真正执行计算"时的耗时，
        跳过的帧不记录（因为几乎是0ms）。
        """
        if not self.is_active:
            return
        self._t_cross = time.perf_counter()

    def cross_end(self):
        """
        在 _compute_current_cross_section() 结束时调用。
        """
        if not self.is_active or self._t_cross is None:
            return
        elapsed = (time.perf_counter() - self._t_cross) * 1000
        if self.total_frames > self.warmup_frames:
            # 只记录真正执行了交叉计算的帧（>0.1ms）
            if elapsed > 0.1:
                self.cross_times.append(elapsed)
        self._t_cross = None

    def render_begin(self):
        """
        在 paintGL() 函数入口调用。
        """
        if not self.is_active:
            return
        self._t_render = time.perf_counter()

    def render_end(self):
        """
        在 paintGL() 函数结束时调用。
        自动过滤掉超过200ms的异常峰值（VBO初始化帧）。
        """
        if not self.is_active or self._t_render is None:
            return
        elapsed = (time.perf_counter() - self._t_render) * 1000
        if self.total_frames > self.warmup_frames:
            # 过滤异常峰值（VBO初始化、GC等）
            if elapsed < 200.0:
                self.render_times.append(elapsed)
        self._t_render = None

    def record_gan(self, ms: float):
        """
        记录GAN推理时间。
        在 simulation_step() 中从 gan_worker.get_stats() 读取后调用。
        """
        if not self.is_active:
            return
        if ms > 10:  # 过滤掉<10ms的无效值（队列为空时的默认值）
            self.gan_times.append(float(ms))

    # ─────────────────────────────────────────────────────────────────
    # 统计计算
    # ─────────────────────────────────────────────────────────────────

    @staticmethod
    def _calc(data):
        """计算一组数据的统计特征"""
        if len(data) < 2:
            return {
                'mean': 0.0, 'std': 0.0,
                'min':  0.0, 'max': 0.0,
                'p95':  0.0, 'n':   len(data)
            }
        arr = np.array(data)
        return {
            'mean': float(np.mean(arr)),
            'std':  float(np.std(arr)),
            'min':  float(np.min(arr)),
            'max':  float(np.max(arr)),
            'p95':  float(np.percentile(arr, 95)),
            'n':    len(arr)
        }

    def get_fps(self):
        """计算近200帧的平均帧率"""
        if len(self._fps_buf) < 2:
            return 0.0
        elapsed = self._fps_buf[-1] - self._fps_buf[0]
        if elapsed < 1e-6:
            return 0.0
        return (len(self._fps_buf) - 1) / elapsed

    def get_stats(self):
        """返回完整统计字典"""
        return {
            'fem_step_ms':      self._calc(self.fem_times),
            'cross_section_ms': self._calc(self.cross_times),
            'render_3d_ms':     self._calc(self.render_times),
            'gan_inference_ms': self._calc(self.gan_times),
            'e2e_latency_ms':   self._calc(self.e2e_times),
            'fps':              self.get_fps(),
            'total_frames':     self.total_frames,
        }

    # ─────────────────────────────────────────────────────────────────
    # 输出与保存
    # ─────────────────────────────────────────────────────────────────

    def _print_summary(self):
        s = self.get_stats()
        print(f"\n{'─'*58}")
        print(f"  📊 性能摘要 [第{self.total_frames}帧 | "
              f"FPS={s['fps']:.1f}]")
        print(f"{'─'*58}")
        rows = [
            ("FEM 求解",     s['fem_step_ms']),
            ("切面提取",     s['cross_section_ms']),
            ("3D 渲染",      s['render_3d_ms']),
            ("GAN 推理",     s['gan_inference_ms']),
            ("端到端延迟",   s['e2e_latency_ms']),
        ]
        for name, d in rows:
            if d['n'] > 0:
                print(f"  {name:<10} "
                      f"均值={d['mean']:7.2f}ms  "
                      f"标准差={d['std']:6.2f}ms  "
                      f"P95={d['p95']:7.2f}ms  "
                      f"n={d['n']}")
        print(f"{'─'*58}\n")

    def save_report(self, filepath=None):
        """保存JSON报告"""
        if filepath is None:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            filepath = f"perf_report_{ts}.json"
        data = self.get_stats()
        data['timestamp'] = datetime.now().isoformat()
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        print(f"✓ 性能报告已保存: {filepath}")
        return filepath

    def print_final_report(self):
        """程序退出前调用，打印并保存最终报告"""
        s = self.get_stats()
        print("\n" + "=" * 70)
        print("  📋 系统性能最终报告（论文 Table 6 数据来源）")
        print("=" * 70)
        print(f"  采集帧数: {self.total_frames}  "
              f"(预热跳过前 {self.warmup_frames} 帧)")
        print(f"  平均帧率: {s['fps']:.2f} FPS\n")

        header = (f"  {'模块':<16} "
                  f"{'均值(ms)':>10} "
                  f"{'标准差':>9} "
                  f"{'最小':>9} "
                  f"{'P95':>9} "
                  f"{'样本数':>7}")
        print(header)
        print(f"  {'─'*64}")

        rows = [
            ("FEM Physics Step",   s['fem_step_ms']),
            ("Cross-sect. Extr.",  s['cross_section_ms']),
            ("3D Rendering",       s['render_3d_ms']),
            ("GAN Inference",      s['gan_inference_ms']),
            ("End-to-End Latency", s['e2e_latency_ms']),
        ]
        for name, d in rows:
            print(f"  {name:<16} "
                  f"{d['mean']:>10.2f} "
                  f"{d['std']:>9.2f} "
                  f"{d['min']:>9.2f} "
                  f"{d['p95']:>9.2f} "
                  f"{d['n']:>7}")
        print("=" * 70)
        return self.save_report("perf_report_final.json")


# =============================================================================
# 全局单例，在 app.py 中 import 使用
# =============================================================================
monitor = PerformanceMonitor(window_size=1000, warmup_frames=50)