from dataclasses import dataclass, field
from collections import deque
from typing import List, Tuple
import random
import numpy as np
from numpy.random import Generator
from .task import Task
import logging
from .all_config import (
    GROUND_WIDTH,
    GROUND_HEIGHT,
    CACHE_SIZE,
    DATA_SIZE_RANGES,
    CPU_RANGES,
    Nodetype,
    WORKLOAD_INTENSITY,
)


@dataclass
class UENode:
    node_id: int  # 新增：节点的唯一标识符
    nodetype: Nodetype  # 节点类型
    cache_capacity: int = 0  # 缓存容量（单位为bit）
    current_cache_size: int = 0  # 当前缓存使用大小（单位为bit）
    loc: np.ndarray = field(default_factory=lambda: np.array([0, 0], dtype=np.float32))
    maxDelay: int = 0  #  最大延迟（单位为秒） (这个还没有用到，但其实最大延迟是应该针对每个任务而不是结点的)
    local_capacity: float = 6e8  # 该ue的本地计算能力 0.6Ghz 这里换算成cycle/s
    task_queue: deque = field(
        default_factory=deque
    )  # 任务队列（缓存），队首视为最早到达的任务
    # --- 关键修改：使用 NumPy 新一代的 Generator API ---
    # 这与 Gymnasium 的 self.np_random 类型一致，是当前推荐的最佳实践。
    rng: Generator = None  # 新增：随机数生成器

    def __post_init__(self):
        self.cache_capacity = CACHE_SIZE[self.nodetype]
        # 每一个episode开始时才初始化结点，所以current_time=0
        # 使用 rng 替代 np.random
        if self.rng is None:
            self.rng = np.random.default_rng()  # 使用新的 API 创建默认生成器
        self.loc = self.rng.uniform(20, GROUND_WIDTH - 20, 2).astype(np.float32)

        self.generateTask(current_time=0)

    def generateTask(self, current_time):
        # 根据泊松过程生成本时刻的任务数（假设到达率为lambda）
        lambda_task_arrival_rate = self.rng.integers(
            1, 4
        )  # 随机设置到达率为1到4之间的整数
        num_new_tasks = self.rng.poisson(lambda_task_arrival_rate)
        # 生成的任务数为0时，直接返回
        if num_new_tasks == 0:
            return
        for _ in range(num_new_tasks):
            # 根据nodetype确定任务数据大小和required_cpu范围
            data_size = self.rng.integers(
                int(DATA_SIZE_RANGES[self.nodetype][0]),
                int(DATA_SIZE_RANGES[self.nodetype][1]),
            )
            required_cpu = self.rng.uniform(*CPU_RANGES[self.nodetype])
            workload_intensity = WORKLOAD_INTENSITY[self.nodetype]
            # 新建一个Task对象
            new_task = Task(
                data_size=data_size,
                required_cpu=required_cpu,
                workload_intensity=workload_intensity,
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

    def local_offloading(
        self, current_time: float, available_time: float
    ) -> Tuple[float, float]:
        """
        完全本地卸载，但会遵守 available_time 的限制。
        它会处理队列中的任务，直到累积处理时间超过 available_time。
        处理完成的任务会从队列中移除。未处理的任务会保留在队列中。

        Args:
            current_time: 开始计算的当前时间。
            available_time: 可用于计算的最大时长。

        Returns:
            一个元组 (处理任务的总延迟, 最后一个完成任务的时间点)。
        """
        total_delay = 0.0
        if not self.task_queue:  # 如果没有任务，直接返回0
            return total_delay, current_time

        start_time = current_time
        cumulative_processing_time = 0.0
        last_completion_time = current_time
        processed_tasks_count = 0

        for task in self.task_queue:
            processing_time = (
                task.data_size * task.workload_intensity / self.local_capacity
            )

            if cumulative_processing_time + processing_time > available_time:
                logging.warning(
                    f"UE local computation timeout. Node  [{self.loc[0]:.2f}, {self.loc[1]:.2f}] with {self.nodetype} processed {processed_tasks_count} tasks. "
                    f"{len(self.task_queue) - processed_tasks_count} tasks remaining."
                )
                break  # 超时，停止处理

            task.status = 1
            task.finished_time = start_time + processing_time
            total_delay += task.finished_time - start_time
            cumulative_processing_time += processing_time
            last_completion_time = task.finished_time
            start_time = task.finished_time
            processed_tasks_count += 1

        # 从队列前端移除已处理的任务
        # 这样当返回的时候，队列只剩下未处理的任务
        for _ in range(processed_tasks_count):
            processed_task = self.task_queue.popleft()
            self.current_cache_size -= processed_task.data_size

        return total_delay, last_completion_time

    def partial_offloading(
        self, local_task_list: List[Task], current_time: float, available_time: float
    ) -> Tuple[float, float]:
        if not local_task_list:  # 如果没有任务，直接返回0
            return 0.0, current_time
        start_time = current_time  # 记录开始本地处理时当前时间
        total_delay = 0.0
        cumulative_processing_time = 0.0
        last_completion_time = current_time

        for task in local_task_list:
            # 动态计算处理时间
            processing_time = (
                task.data_size * task.workload_intensity / self.local_capacity
            )

            if cumulative_processing_time + processing_time > available_time:
                logging.warning(
                    f"UE partial local computation timeout. Stopping processing."
                )
                break  # 超时，停止处理后续任务

            task.status = 1  # 本地完成
            task.finished_time = start_time + processing_time
            total_delay += task.finished_time - start_time
            cumulative_processing_time += processing_time
            last_completion_time = task.finished_time
            start_time = task.finished_time
        return total_delay, last_completion_time
