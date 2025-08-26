from enum import Enum


class Nodetype(Enum):
    NORMAL = 1  # 普通型 (Normal): 模拟物联网（IoT）设备，如环境传感器，其任务特点是数据量小、计算密度低。
    MODERATE = 2  # 中等型 (Moderate): 模拟通用边缘应用，如移动端的图像处理，其任务的数据量和计算需求均处于中等水平。
    HPC = 3  # 计算密集型 (HPC): 模拟需要进行复杂本地计算的设备，如边缘AI推理模块，其任务特点是计算密度极高。


# UE本地缓存
CACHE_SIZE = {
    Nodetype.NORMAL: 5e7,  # 大小适中，可以缓存几十个任务，但如果UAV长时间不来服务，依然会溢出。
    Nodetype.MODERATE: 2e8,  #  缓存较大，可以容纳更多的中等任务。
    Nodetype.HPC: 5e8,  # 拥有最大的缓存，符合其高性能节点的定位
}

# 每个任务的大小 (bit)
DATA_SIZE_RANGES = {
    Nodetype.NORMAL: (1_000_000, 2_000_001),  # 增大了约2倍
    Nodetype.MODERATE: (5_000_000, 10_000_001),  # 增大了约2.5倍
    Nodetype.HPC: (3_000_000, 6_000_001),  # 增大了约2倍
}

# UE本地计算能力
CPU_RANGES = {
    Nodetype.NORMAL: (5e8, 1e9),  # 单位: cycle/s
    Nodetype.MODERATE: (1e9, 2e9),
    Nodetype.HPC: (1e9, 3e9),
}

# 建议新增：计算密度 (单位: cycle/bit)
WORKLOAD_INTENSITY = {
    Nodetype.NORMAL: 400,  # 普通任务，计算需求低
    Nodetype.MODERATE: 800,  # 中等任务，例如你之前的设定
    Nodetype.HPC: 1500,  # 高密度计算，每个bit都需要大量计算周期
}

TASK_TYPE_DISTRIBUTION = {
    Nodetype.NORMAL: 0.6,
    Nodetype.MODERATE: 0.3,
    Nodetype.HPC: 0.1,
}

GROUND_WIDTH = 400  # 场地宽度
GROUND_HEIGHT = 400  # 场地高度
UE_NUM = 20  # ue数量
