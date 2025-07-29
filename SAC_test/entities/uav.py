from dataclasses import dataclass, field
from typing import List
import math
import numpy as np
from .task import Task


@dataclass
class UAVNode:
    # --- 关键修改: 将位置类型从 list 更改为 NumPy 数组，以实现一致和高效的向量运算 ---
    loc: np.ndarray = field(
        default_factory=lambda: np.array([20, 20], dtype=np.float32)
    )
    battery_capacity = 50000  # uav电池容量
    e_battery = 50000  # uav当前电量: 1000kJ.
    # ref: Mobile Edge Computing via a UAV-Mounted Cloudlet: Optimization of Bit Allocation and Path Planning
    flying_height = 60  # 无人机飞行高度 60m
    f_uav = 1.2e9  # UAV的计算频率1.2GHz
    m_uav = 9.65  # uav质量/kg 用来算能量消耗的
    t_fly = 1  # 飞行时间暂定1s
    r = 10 ** (-27)  # 芯片结构对cpu处理的影响因子
    max_speed = 20  # 无人机最大飞行速度 30m/s
    max_acceleration = 5 # 无人机最大加速度

    # --- 移除 __init__，因为 dataclass 会自动处理 flying_speed 的初始化 ---
    flying_speed: float = 20.0

    def moveto(self, dis_fly, theta, loc_ue):
        reward = 0.0  # 初始化奖励
        e_fly = 0.0  # 初始化飞行耗能
        flying_velocity = dis_fly / self.t_fly  # 飞行速度
        flying_power = calculate_uav_power(flying_velocity)  # 计算飞行功率
        e_fly = self.t_fly * flying_power  # 飞行过程的耗能

        self.e_battery -= e_fly  # 减去飞行耗能 获得现在能量

        # --- 优化: 使用 NumPy 的向量化操作计算距离 ---
        # 计算当前离UE的距离
        dis_before_flying = np.linalg.norm(self.loc - loc_ue)

        self.loc += np.array([dis_fly * math.cos(theta), dis_fly * math.sin(theta)])
        # 现在离UE的距离
        dis_after_flying = np.linalg.norm(self.loc - loc_ue)
        # 如果算出来是正，那么奖励就是正的，说明before比after大，说明飞行是有意义的
        # 如果算出来是负的，那么说明飞行是没有意义的，奖励就是负的
        reward = dis_before_flying - dis_after_flying  # 距离变化量

        return e_fly, reward  # 返回飞行耗能和奖励

    def offload(
        self, task_list: List[Task], current_time: float, hover_time: float
    ) -> tuple[float, float, float]:
        """
        处理卸载到UAV的任务，但计算时间不能超过可用的悬停时间。

        Args:
            task_list: 要处理的任务列表。
            current_time: 计算开始的当前时间点。
            hover_time: 在此时间步内可用于计算的最大时长。

        Returns:
            一个元组 (总延迟, 消耗的能量, 最后一个完成任务的时间点)。
        """
        start_time = current_time  # 记录开始卸载时当前时间
        total_delay = 0.0
        cumulative_processing_time = 0.0
        last_completion_time = current_time

        for task in task_list:
            # 动态计算当前任务所需的处理时间
            processing_time = task.data_size * task.workload_intensity / self.f_uav

            # 检查加上这个任务后是否会超时
            if cumulative_processing_time + processing_time > hover_time:
                # print(f"UAV计算超时，在处理任务 {task} 时停止。")
                break  # 停止处理后续任务

            task.status = 2  # 标记为uav完成
            task.finished_time = start_time + processing_time
            total_delay += task.finished_time - start_time
            cumulative_processing_time += processing_time
            last_completion_time = task.finished_time
            start_time = task.finished_time  # 更新开始时间为当前任务的完成时间

        # 只计算实际处理时间的能耗
        hovering_velocity = 0  # 悬停速度为0
        consumed_energy = (
            calculate_uav_power(hovering_velocity) * hover_time
        )  # 悬停时的能量消耗
        self.e_battery -= consumed_energy

        return total_delay, consumed_energy, last_completion_time

    def receive_energy(self, energy):
        """接收能量并更新电池状态"""
        actual_energy = min(energy, self.battery_capacity - self.e_battery)
        self.e_battery += actual_energy
        return actual_energy

    def reset(self):
        self.loc = np.array([20, 20], dtype=np.float32)
        self.e_battery = self.battery_capacity  # 重置电池电量
        self.flying_speed = self.max_speed  # 重置飞行速度


import math

# -----------------------------------------------------------
# 1. 定义常量 (根据 Table 1. KEY SIMULATION SETTINGS)
# -----------------------------------------------------------
P0 = 79.86  # 悬停时的叶型功率 (W)
P1 = 88.63  # 悬停时的诱导功率 (W)
U_tip = 120  # 桨叶尖端速度 (m/s)
v_0 = 4.03  # 悬停时的平均诱导速度 (m/s)
d_0 = 0.6  # 机身阻力系数 (dimensionless)
rho = 1.225  # 空气密度 (kg/m^3)
s = 0.05  # 旋翼实体比 (dimensionless)
A = 0.503  # 旋翼桨盘面积 (m^2)


def calculate_uav_power(v_h):
    """
    计算无人机在给定水平速度下的总功率消耗。

    Args:
        v_h (float): 无人机的水平飞行速度 (m/s).

    Returns:
        float: 该速度下无人机所需的总功率 (W).
    """

    # -----------------------------------------------------------
    # 2. 将公式翻译成代码，分为三个部分
    # -----------------------------------------------------------

    # 第一部分：桨叶外形功率 (Profile Power)
    power_profile = P0 * (1 + (3 * (v_h**2)) / (U_tip**2))

    # 第二部分：诱导功率 (Induced Power)
    # 先计算根号下的复杂项
    term_under_sqrt = math.sqrt(
        1 + (v_h**4) / (4 * (v_0**4)) - (v_h**2) / (2 * (v_0**2))
    )
    power_induced = P1 * term_under_sqrt

    # 第三部分：寄生功率 (Parasite Power)
    power_parasite = 0.5 * d_0 * rho * s * A * (v_h**3)

    # 计算总功率
    total_power = power_profile + power_induced + power_parasite

    return total_power


# -----------------------------------------------------------
# 3. 主程序入口：演示如何使用这个函数
# -----------------------------------------------------------
if __name__ == "__main__":
    # 假设我们想计算无人机以 10 m/s 速度飞行时的功率
    example_velocity = 10.0  # m/s

    # 调用函数进行计算
    power_needed = calculate_uav_power(example_velocity)

    # 打印结果，并格式化输出，保留两位小数
    print(
        f"无人机以 {example_velocity} m/s 的速度飞行时, 所需的总功率为: {power_needed:.2f} W"
    )

    # 再测试一个速度
    example_velocity_2 = 20.0  # m/s
    power_needed_2 = calculate_uav_power(example_velocity_2)
    print(
        f"无人机以 {example_velocity_2} m/s 的速度飞行时, 所需的总功率为: {power_needed_2:.2f} W"
    )

    example_velocity_3 = 0  # m/s
    power_needed_3 = calculate_uav_power(example_velocity_3)
    print(
        f"无人机以 {example_velocity_3} m/s 的速度飞行时, 所需的总功率为: {power_needed_3:.2f} W"
    )
