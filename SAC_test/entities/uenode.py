from dataclasses import dataclass, field
from collections import deque
from typing import List, Tuple
import random
import numpy as np
from numpy.random import RandomState
from .task import Task
import logging
from .all_config import (
    GROUND_WIDTH,
    GROUND_HEIGHT,
    CACHE_SIZE,
    DATA_SIZE_RANGES,
    CPU_RANGES,
    Nodetype,
)


@dataclass
class UENode:
    nodetype: Nodetype  # 节点类型
    cache_capacity: int = 0  # 缓存容量（单位为bit）
    current_cache_size: int = 0  # 当前缓存使用大小（单位为bit）
    loc: list = field(default_factory=lambda: [0, 0])  # 初始坐标 [0, 0]
    maxDelay: int = 0  #  最大延迟（单位为秒） (这个还没有用到，但其实最大延迟是应该针对每个任务而不是结点的)
    local_capacity: float = 6e8  # 该ue的本地计算能力 0.6Ghz 这里换算成cycle/s
    task_queue: deque = field(
        default_factory=deque
    )  # 任务队列（缓存），队首视为最早到达的任务
    rng: RandomState = None  # 新增：随机数生成器

    def __post_init__(self):
        self.cache_capacity = CACHE_SIZE[self.nodetype]
        # 每一个episode开始时才初始化结点，所以current_time=0
        # 使用 rng 替代 np.random
        if self.rng is None:
            self.rng = np.random.RandomState()  # 默认无 seed
        self.loc = list(self.rng.uniform(20, GROUND_WIDTH - 20, 2))  # ue不分布在边缘上

        self.generateTask(current_time=0)

    def generateTask(self, current_time):
        # 根据泊松过程生成本时刻的任务数（假设到达率为lambda）
        lambda_task_arrival_rate = self.rng.randint(
            1, 4
        )  # 随机设置到达率为1到4之间的整数
        num_new_tasks = self.rng.poisson(lambda_task_arrival_rate)
        # 生成的任务数为0时，直接返回
        if num_new_tasks == 0:
            return
        for _ in range(num_new_tasks):
            # 根据nodetype确定任务数据大小和required_cpu范围
            data_size = self.rng.randint(
                int(DATA_SIZE_RANGES[self.nodetype][0]),
                int(DATA_SIZE_RANGES[self.nodetype][1]),
            )
            required_cpu = self.rng.uniform(*CPU_RANGES[self.nodetype])
            # data_size: bit
            # local_capacity: cycle/s
            # 500: cycles per bit
            required_time = data_size * 500 / self.local_capacity  # 计算处理时间
            # 新建一个Task对象
            new_task = Task(
                data_size=data_size,
                required_cpu=required_cpu,
                require_time=required_time,
                arrival_time=current_time,
                status=0,
            )
            # print(new_task)
            # 检查当前缓存是否溢出
            if self.current_cache_size + new_task.data_size <= self.cache_capacity:
                self.task_queue.append(new_task)  # 如果没有溢出，则添加新任务
                self.current_cache_size += new_task.data_size
            else:
                # 缓存溢出：移除最早的任务，直到腾出需要的空间
                while (
                    self.current_cache_size + new_task.data_size > self.cache_capacity
                ):
                    if not self.task_queue:  # 队列已空但依然不满足条件
                        logging.error("Cache overflow but no task to remove!")
                        raise RuntimeError(
                            "Cache overflow with no tasks to remove. Consider increasing cache capacity or adjusting task generation."
                        )
                    removed_task = self.task_queue.popleft()
                    removed_task.status = 3  #  标记为丢弃
                    logging.warning(
                        "Task removed due to cache overflow: %s", removed_task
                    )
                    # 这里应该记录或许应该可视化。
                    self.current_cache_size -= removed_task.data_size

                # 添加新任务
                self.task_queue.append(new_task)
                self.current_cache_size += new_task.data_size

    def local_offloading(self, current_time) -> float:
        # 完全本地卸载
        # 假设每一次本地卸载都把任务队列里的所有任务计算完
        total_delay = 0.0
        if not self.task_queue:  # 如果没有任务，直接返回0
            return total_delay
        start_time = current_time
        if self.task_queue:  # 只有当任务队列中有任务时才进行卸载
            for task in self.task_queue:
                task.status = 1
                # 这个是当前任务的完成时间
                # 500指的是一个bit要用500个周期来process
                task.finished_time = start_time + task.require_time
                total_delay += task.finished_time - start_time
                start_time = task.finished_time
        # 清空队列并重置缓存大小
        self.task_queue.clear()
        self.current_cache_size = 0

        return total_delay

    def partial_offloading(self, local_task_list: List[Task], current_time) -> float:
        if not local_task_list:  # 如果没有任务，直接返回0
            return 0.0, current_time
        start_time = current_time  # 记录开始本地处理时当前时间
        total_delay = 0.0
        for task in local_task_list:
            task.status = 1  # 本地完成
            # 这个是当前任务的完成时间
            task.finished_time = start_time + task.require_time
            total_delay += task.finished_time - start_time
            start_time = task.finished_time
        return (
            total_delay,
            local_task_list[-1].finished_time,
        )  # 返回本地处理的总延迟和最后一个任务的完成时间
