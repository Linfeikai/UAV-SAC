import matplotlib.pyplot as plt
import numpy as np
import random
from dataclasses import dataclass
from typing import List


@dataclass
class Task:
    task_id: int
    created_time: float
    completed_time: float = None
    process_time: float = None  # 该任务需要的处理时长


class TaskSystem:
    def __init__(self):
        self.clock = 0.0
        self.task_queue: List[Task] = []
        self.completed_tasks = []

    def generate_task(self):
        """生成新任务（随机创建间隔）"""
        # 随机间隔（0.5-3秒）
        self.clock += random.uniform(0.5, 3.0)
        new_task = Task(
            task_id=len(self.task_queue) + 1,
            created_time=self.clock,
            process_time=random.uniform(1.0, 4.0),  # 随机处理时间
        )
        self.task_queue.append(new_task)
        print(
            f"🕒 任务{new_task.task_id:02d} 生成于 {new_task.created_time:.2f}s，需要处理 {new_task.process_time:.2f}s"
        )

    def process_tasks(self):
        """处理所有队列中的任务"""
        while self.task_queue:
            current_task = self.task_queue.pop(0)
            # 记录处理开始时间（可能晚于创建时间）
            start_time = max(self.clock, current_task.created_time)
            # 推进时钟到完成时间
            self.clock = start_time + current_task.process_time
            current_task.completed_time = self.clock
            self.completed_tasks.append(current_task)
            print(
                f"✅ 任务{current_task.task_id:02d} 完成于 {current_task.completed_time:.2f}s | "
                f"总延迟 = {current_task.completed_time - current_task.created_time:.2f}s"
            )


def visualize_tasks(tasks: List[Task]):
    """可视化任务时间线"""
    fig, ax = plt.subplots(figsize=(10, 6))

    # 为每个任务生成随机颜色
    colors = plt.cm.tab20(np.linspace(0, 1, len(tasks)))

    for i, task in enumerate(tasks):
        # 绘制任务持续时间段
        ax.hlines(
            y=i,
            xmin=task.created_time,
            xmax=task.completed_time,
            colors=colors[i],
            lw=15,
            label=f"Task {task.task_id}",
        )

        # 标记关键时间点
        ax.scatter(
            task.created_time, i, color="black", zorder=3, marker="o", label="Created"
        )
        ax.scatter(
            task.completed_time, i, color="red", zorder=3, marker="x", label="Completed"
        )

    # 添加图例和标签
    ax.set_yticks(range(len(tasks)))
    ax.set_yticklabels([f"Task {t.task_id}" for t in tasks])
    ax.set_xlabel("Time (seconds)")
    ax.set_title("Task Processing Timeline")

    # 去重图例
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), loc="upper left")

    plt.grid(linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.show()


# ===================== 运行示例 =====================
if __name__ == "__main__":
    # 初始化系统
    system = TaskSystem()

    # 生成5个随机任务
    for _ in range(5):
        system.generate_task()

    # 处理任务
    print("\n=== 开始处理任务 ===")
    system.process_tasks()

    # 可视化结果
    print("\n=== 生成可视化图表 ===")
    visualize_tasks(system.completed_tasks)

    # 统计报告
    total_delay = sum(t.completed_time - t.created_time for t in system.completed_tasks)
    avg_delay = total_delay / len(system.completed_tasks)
    print(f"\n📊 统计报告:")
    print(f"总系统延迟: {total_delay:.2f}s")
    print(f"平均任务延迟: {avg_delay:.2f}s")
    print(
        f"最长单个延迟: {max(t.completed_time - t.created_time for t in system.completed_tasks):.2f}s"
    )
