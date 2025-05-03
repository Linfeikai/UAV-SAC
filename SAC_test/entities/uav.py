from dataclasses import dataclass, field
from typing import List
import math
import numpy as np
from .task import Task


@dataclass
class UAVNode:
    loc: list = field(default_factory=lambda: [200, 200])  # 初始坐标 [200, 200]
    battery_capacity = 500000  # uav电池容量
    e_battery = 500000  # uav当前电量: 500kJ.
    # ref: Mobile Edge Computing via a UAV-Mounted Cloudlet: Optimization of Bit Allocation and Path Planning
    flying_height = 60  # 无人机飞行高度 60m
    f_uav = 1.2e9  # UAV的计算频率1.2GHz
    m_uav = 9.65  # uav质量/kg 用来算能量消耗的
    t_fly = 1  # 飞行时间暂定1s
    r = 10 ** (-27)  # 芯片结构对cpu处理的影响因子
    max_speed = 30  # 无人机最大飞行速度 30m/s

    def __init__(self):
        self.flying_speed = 30  # 无人机最大飞行速度 30m/s

    def moveto(self, dis_fly, theta, loc_ue):
        reward = 0.0  # 初始化奖励
        e_fly = 0.0  # 初始化飞行耗能
        e_fly = (
            (dis_fly / self.t_fly) ** 2 * self.m_uav * self.t_fly * 0.5
        )  # 飞行过程的耗能
        self.e_battery -= e_fly  # 减去飞行耗能 获得现在能量

        # 计算当前离UE的距离
        dis_before_flying = np.sqrt(
            (self.loc[0] - loc_ue[0]) ** 2 + (self.loc[1] - loc_ue[1]) ** 2
        )

        self.loc[0] += dis_fly * math.cos(theta)  # 无人机现在x坐标
        self.loc[1] += dis_fly * math.sin(theta)  # 无人机现在y坐标
        # 现在离UE的距离
        dis_after_flying = np.sqrt(
            (self.loc[0] - loc_ue[0]) ** 2 + (self.loc[1] - loc_ue[1]) ** 2
        )

        if dis_after_flying < dis_before_flying:
            reward = (dis_before_flying - dis_after_flying) * 10  # 离UE近了
        elif dis_after_flying > dis_before_flying:
            reward = (dis_after_flying - dis_before_flying) * (-1)  # 离UE远了
        elif dis_after_flying == dis_before_flying:  # 竟然敢静止不动？？
            print("静止不动")
            reward = -20  # 静止惩罚
        return e_fly, reward  # 返回飞行耗能和奖励

    def offload(
        self, task_list: List[Task], current_time
    ) -> tuple[float, float, float]:
        start_time = current_time  # 记录开始卸载时当前时间
        total_delay = 0.0
        for task in task_list:
            task.status = 2  # 标记为uav完成
            task.finished_time = (
                start_time + task.require_time
            )  # 这个是当前任务的完成时间
            total_delay += task.finished_time - start_time
            start_time = task.finished_time  # 更新开始时间为当前任务的完成时间
        if task_list:
            consumed_energy = (
                self.r * self.f_uav**3 * (task_list[-1].finished_time - current_time)
            )  # uav计算的耗能
            self.e_battery -= consumed_energy
            return (
                total_delay,
                consumed_energy,
                task_list[-1].finished_time,
            )  # 返回计算延迟和能量消耗,和最后一个任务完成的时间
        else:
            return total_delay, 0.0, current_time  # 如果没有任务，返回默认值

    def receive_energy(self, energy):
        """接收能量并更新电池状态"""
        actual_energy = min(energy, self.battery_capacity - self.e_battery)
        self.e_battery += actual_energy
        return actual_energy

    def reset(self):
        self.loc = [20, 20]
        self.e_battery = 500000
