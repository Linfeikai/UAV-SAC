from enum import Enum


class Nodetype(Enum):
    NORMAL = 1  # 物联网传感器服务；数据量小，计算需求低（如温度传感器，简单监控设备）
    MODERATE = 2  # 中等任务量：如图像处理
    HPC = 3  # 高计算密度


CACHE_SIZE = {
    Nodetype.NORMAL: 5000_000_000,  # 物联网传感器服务；数据量小，计算需求低（如温度传感器，简单监控设备）
    Nodetype.MODERATE: 5000_000_000,  # 中等任务量：如图像处理
    Nodetype.HPC: 5000_000_000,  # 高计算密度
}

# 数据范围和计算需求的范围
DATA_SIZE_RANGES = {
    Nodetype.NORMAL: (2_000_000, 4_000_001),  # 单位: bit
    Nodetype.MODERATE: (5_000_000, 20_000_001),
    Nodetype.HPC: (5_000_000, 10_000_001),
}

CPU_RANGES = {
    Nodetype.NORMAL: (1e8, 3e8),  # 单位: cycle/s
    Nodetype.MODERATE: (1e8, 8e8),
    Nodetype.HPC: (1e9, 3e9),
}

GROUND_WIDTH = 400  # 场地宽度
GROUND_HEIGHT = 400  # 场地高度
