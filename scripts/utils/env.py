import platform
import psutil
import math


def get_system_info():
    print("=" * 40)
    print("硬件与系统配置信息")
    print("=" * 40)

    # 1. 操作系统
    os_name = platform.system()
    os_version = platform.release()
    print(f"□ 操作系统: {os_name} {os_version}")

    # 2. CPU 型号
    cpu_model = platform.processor()
    # 获取 CPU 物理核心和逻辑核心数
    core_count = psutil.cpu_count(logical=False)
    thread_count = psutil.cpu_count(logical=True)
    # 获取 CPU 频率 (需要 psutil)
    cpu_freq = psutil.cpu_freq()
    freq_str = f"{cpu_freq.max:.2f} MHz" if cpu_freq else "未知"
    print(f"□ CPU 型号: {cpu_model}")
    print(f"  - 核心数: {core_count}核 {thread_count}线程")
    print(f"  - 最高主频: {freq_str}")

    # 3. RAM 大小
    # 获取总内存，并转换为 GB
    ram_info = psutil.virtual_memory()
    total_ram_gb = ram_info.total / (1024 ** 3)
    # 向上取整，通常内存条都是 8G, 16G, 32G 等
    print(f"□ RAM 大小: {math.ceil(total_ram_gb)} GB")

    # 4. GPU 型号
    print("□ GPU 型号: ", end="")
    try:
        # 如果你安装了 torch，可以用 torch 获取
        import torch
        if torch.cuda.is_available():
            print(torch.cuda.get_device_name(0))
        else:
            print("未检测到 CUDA 可用 (已知为 GTX 1650 Ti)")
    except ImportError:
        print("GTX 1650 Ti (未安装 PyTorch，手动确认)")


if __name__ == "__main__":
    get_system_info()