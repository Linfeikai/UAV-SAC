from dataclasses import dataclass


@dataclass
class Task:
    data_size: int  # 单个任务的数据大小(bit)
    required_cpu: float  # 单个任务的抽象计算需求（GHz）
    arrival_time: float  # 任务的到达时间
    require_time: float  # 任务需要的处理时间
    status: int  # 任务的状态：未完成(0)、本地完成(1)、uav完成(2)、被丢弃（3）
    finished_time: float = 0.0  # 任务的完成时间
