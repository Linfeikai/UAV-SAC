import math
from enum import Enum
from dataclasses import dataclass, field
from collections import deque
from typing import List, Tuple
import numpy as np
import matplotlib.pyplot as plt
import logging
import random
import os


class Nodetype(Enum):
    NORMAL = 1  # 物联网传感器服务；数据量小，计算需求低（如温度传感器，简单监控设备）
    MODERATE = 2  # 中等任务量：如图像处理
    HPC = 3  # 高计算密度


CACHE_SIZE = {
    Nodetype.NORMAL: 500_000_000,  # 物联网传感器服务；数据量小，计算需求低（如温度传感器，简单监控设备）
    Nodetype.MODERATE: 500_000_000,  # 中等任务量：如图像处理
    Nodetype.HPC: 500_000_000,  # 高计算密度
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


@dataclass
class Task:
    data_size: int  # 单个任务的数据大小(bit)
    required_cpu: float  # 单个任务的抽象计算需求（GHz）
    arrival_time: float  # 任务的到达时间
    require_time: float  # 任务需要的处理时间
    status: int  # 任务的状态：未完成(0)、本地完成(1)、uav完成(2)、被丢弃（3）
    finished_time: float = 0.0  # 任务的完成时间


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

    def __post_init__(self):
        self.cache_capacity = CACHE_SIZE[self.nodetype]
        # 每一个episode开始时才初始化结点，所以current_time=0
        self.generateTask(current_time=0)
        self.loc = list(np.random.uniform(0, GROUND_WIDTH, 2))

    def generateTask(self, current_time):
        # self.data_size, self.required_cpu = (
        #     np.random.randint(*DATA_SIZE_RANGES[self.nodetype]),
        #     np.random.uniform(*CPU_RANGES[self.nodetype])
        # )
        # 根据泊松过程生成本时刻的任务数（假设到达率为lambda）
        lambda_task_arrival_rate = random.randint(
            1, 4
        )  # 随机设置到达率为1到4之间的整数
        num_new_tasks = np.random.poisson(lambda_task_arrival_rate)
        # 生成的任务数为0时，直接返回
        if num_new_tasks == 0:
            return
        for _ in range(num_new_tasks):
            # 根据nodetype确定任务数据大小和required_cpu范围
            data_size = np.random.randint(
                int(DATA_SIZE_RANGES[self.nodetype][0]),
                int(DATA_SIZE_RANGES[self.nodetype][1]),
            )
            required_cpu = np.random.uniform(*CPU_RANGES[self.nodetype])
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


@dataclass
class UAVNode:
    loc: list = field(default_factory=lambda: [200, 200])  # 初始坐标 [200, 200]
    battery_capacity = 500000  # uav电池容量
    e_battery = 500000  # uav当前电量: 500kJ.
    # ref: Mobile Edge Computing via a UAV-Mounted Cloudlet: Optimization of Bit Allocation and Path Planning
    flying_height = 60  # 无人机飞行高度 60m
    flying_speed = 30  # 无人机最大飞行速度 30m/s
    f_uav = 1.2e9  # UAV的计算频率1.2GHz
    m_uav = 9.65  # uav质量/kg 用来算能量消耗的
    t_fly = 1  # 飞行时间暂定1s
    r = 10 ** (-27)  # 芯片结构对cpu处理的影响因子

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
        else:
            reward = (dis_after_flying - dis_before_flying) * (-1)  # 离UE远了
        if self.loc[0] > GROUND_WIDTH:
            print("x out of range")
        if self.loc[1] > GROUND_HEIGHT:
            print("y out of range")
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
        self.loc = [200, 200]
        self.e_battery = 500000


@dataclass
class LaserCharger:
    loc: list = field(default_factory=lambda: [250, 250])  # 初始坐标 [0, 0]
    # 从表1初始化参数
    A = 1e-2  # 激光接收器望远镜或收集透镜的面积 (m^2)
    D = 0.05  # 初始激光束大小 (m)
    theta = 0.2  # 综合发射接收光效率
    alpha = 1e-6  # 信道介质衰减系数 (/m)
    beta = 3.4e-5  # 角扩展 The angular spread
    eta_u = 0.8  # 能量转换效率
    P_0 = 1  # 60dBm转换为瓦特为1w
    effective_range = 10  # 有效充电半径

    def charge(self, uav: UAVNode, charge_time):
        # 执行充电操作
        if self._is_uav_in_range(uav):
            # 计算信息增益
            # 1.计算距离
            dx = self.loc[0] - uav.loc[0]
            dy = self.loc[1] - uav.loc[1]
            dh = uav.flying_height
            distance_uav_ap = np.sqrt(dx * dx + dy * dy + dh * dh)
            numerator = self.A * self.theta * np.exp(-self.alpha * distance_uav_ap)
            denominator = (self.D + self.beta * distance_uav_ap) ** 2  # 分母
            g_ap = numerator / denominator
            E_harvest = self.eta_u * g_ap * self.P_0 * charge_time * charge_time
            uav.receive_energy(E_harvest)  # 更新无人机电量
            return E_harvest
        else:
            return 0  # 超出有效范围，无法充电

    def _is_uav_in_range(self, uav: UAVNode):
        dx = self.loc[0] - uav.loc[0]
        dy = self.loc[1] - uav.loc[1]
        dh = uav.flying_height
        distance_uav_ap = np.sqrt(dx * dx + dy * dy + dh * dh)
        return distance_uav_ap <= self.effective_range


class Env(object):
    # 定义环境参数
    # 写在这里下面的是类属性 每个实例都共享
    ground_width = 400  # 场地宽度
    ground_height = 400  # 场地高度
    task_type_distribution = {  # 任务类型分配
        "normal": 0.6,
        "moderate": 0.3,
        "hpc": 0.1,
    }
    ue_num = 20  # UE设备的数量：20
    s = 1000  # 单位bit处理所需cpu圈数1000

    state_dim = 88  # 状态空间维度 # 1 + 2 + 4 * ue_num + 2 = 87
    action_dim = 4
    max_action = (-1, 1)

    # communication model parameters
    bandwidth_nums = 1
    B = bandwidth_nums * 10**6  # 带宽1MHz
    p_noisy_los = 10 ** (-13)  # 噪声功率-100dBm
    p_noisy_nlos = 10 ** (-11)  # 噪声功率-80dBm
    alpha0 = 1e-5  # 距离为1m时的参考信道增益-30dB = 0.001， -50dB = 1e-5
    p_uplink = 0.1  # 上行链路传输功率0.1W

    # episode duration parameters1
    T = 320  # 周期320s
    t_fly = 1  # 思考；self.t_fly 可以用于模拟不同飞行时间下的能耗，如果无人机需要在不同的速度下飞行，那么 self.t_fly 就可以调整以反映不同的飞行时间。
    t_com = 7
    delta_t = t_fly + t_com  # 1s飞行, 后7s用于悬停计算
    slot_num = int(T / delta_t)  # 40个间隔

    def __init__(self):
        self.nodeList: List[UENode] = []
        self.uav = UAVNode()
        self.charger = LaserCharger()
        # 需要频繁地向数组中添加元素，我们先将 self.state 作为列表处理，最后再转换为 NumPy 数组。
        # 这种方法在添加元素时效率更高，因为 NumPy 数组的大小是固定的，而列表的大小是动态的。
        self.service_history = deque(maxlen=5)  # 记录最近5次服务对象
        self.current_step = 0  # 当前步数
        self.current_time = 0.0  # 当前时间

    def reset(self):
        # 一般是在每个episode开始时调用reset函数，重置环境状态
        self.uav.reset()

        # 重置 UE 节点
        light_num = int(self.ue_num * self.task_type_distribution["normal"])
        medium_num = int(self.ue_num * self.task_type_distribution["moderate"])
        heavy_num = int(self.ue_num * self.task_type_distribution["hpc"])

        self.nodeList = (
            [UENode(Nodetype.NORMAL) for _ in range(light_num)]
            + [UENode(Nodetype.MODERATE) for _ in range(medium_num)]
            + [UENode(Nodetype.HPC) for _ in range(heavy_num)]
        )

        # 可视化节点（调试用）
        # self.visualize_nodes()

        self.current_step = 0  # 重置当前步数
        self.current_time = 0.0  # 重置当前时间

        return self._get_obs()

    def visualize_nodes(
        self, flyingTrajectory: List[Tuple[float, float]] = [], current_episode=0
    ):
        # 定义节点类型对应的颜色和标签
        node_colors = {
            Nodetype.NORMAL: "green",
            Nodetype.MODERATE: "blue",
            Nodetype.HPC: "red",
        }
        node_labels = {
            Nodetype.NORMAL: "Normal",
            Nodetype.MODERATE: "Moderate",
            Nodetype.HPC: "HPC",
        }

        # 创建图形
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.set_xlim(0, GROUND_WIDTH)
        ax.set_ylim(0, GROUND_HEIGHT)
        ax.set_title("UE Nodes and Flying Trajectory Visualization")
        ax.set_xlabel("X Coordinate")
        ax.set_ylabel("Y Coordinate")

        # 用于存储 scatters 和 annotations
        scatters = []

        # 绘制节点
        for node in self.nodeList:
            color = node_colors[node.nodetype]
            label = node_labels[node.nodetype]
            x, y = node.loc
            scatter = ax.scatter(x, y, color=color, label=label, s=100, alpha=0.7)
            scatters.append(scatter)

        ax.scatter(
            self.charger.loc[0],
            self.charger.loc[1],
            color="black",
            label="Laser Charger",
        )
        # 绘制飞行轨迹（新增代码）
        if flyingTrajectory:  # 确保轨迹数据非空
            # 提取轨迹的x和y坐标（假设轨迹是 [[x1,y1], [x2,y2], ...] 格式）
            trajectory_x = [point[0] for point in flyingTrajectory]
            trajectory_y = [point[1] for point in flyingTrajectory]

            # 绘制轨迹线（灰色虚线）
            ax.plot(
                trajectory_x,
                trajectory_y,
                color="gray",
                linestyle="--",
                linewidth=2,
                label="Flying Trajectory",
            )

        # 显示图例（需排除重复的轨迹标签）
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))  # 去重
        if "Flying Trajectory" in by_label:
            ax.legend(
                by_label.values(), by_label.keys(), title="Node Types & Trajectory"
            )

        save_dir = "SAC-test/episodeImage"
        os.makedirs(save_dir, exist_ok=True)  # 确保目录存在
        plt.savefig(os.path.join(save_dir, f"episode_{current_episode}.png"))
        plt.close()  # 关闭图形以释放内存

    def step(
        self, action
    ):  # 0: 选择服务的ue编号 ; 1: 方向theta; 2: 距离d; 3: offloading ratio
        self.current_step += 1

        # 初始化奖励和状态
        terminated = False
        out_of_border_penalty = 0  # 越界惩罚
        find_notask_node_penalty = 0  # 找到了没有任务的结点
        action_invalid_penalty = 0  # 动作不合法惩罚
        offloading_imbalance_penalty = (
            0  # 卸载不均衡惩罚（即连续服务同一个结点） 这里还可以用函数或者其他理论优化
        )

        local_time = 0  # 未被uav服务的其他ue 本地计算时间

        # 取出action里面的值
        ue_id, uav_angle, uav_distance, offloading_ratio = (
            action["ue"],
            action["angle"],
            action["speed"],
            action["ratio"],
        )
        # 理论上action的值应该都是合法的才对 这里实际上是二次校验
        ue_id = np.clip(int(ue_id), 0, self.ue_num - 1)  # 确保ue_id在合法范围内
        offloading_ratio = np.clip(offloading_ratio, 0.0, 1.0)  # 限制在[0,1]区间
        if abs(offloading_ratio) < 0.01:
            offloading_ratio = 0.0
        # print(f"offloading_ratio: {offloading_ratio}")
        uav_distance = np.clip(uav_distance, 0.0, 1.0)  # 限制在[0,1]区间

        dis_fly = (
            self.uav.flying_speed * uav_distance * self.t_fly
        )  # 最大速度乘当前比率，得到飞行直线距离

        new_x = self.uav.loc[0] + dis_fly * math.cos(uav_angle)
        new_y = self.uav.loc[1] + dis_fly * math.sin(uav_angle)
        self.service_history.append(ue_id)
        # 如果agent生成的飞行指令不合法（超出边界）
        if (
            new_x > self.ground_width
            or new_x < 0
            or new_y > self.ground_height
            or new_y < 0
        ):
            #     1. 结束episode
            # is_terminal = True
            # reward = -10000
            #     2. 停留在边界处、给予惩罚
            new_x = np.clip(new_x, 0, self.ground_width)
            new_y = np.clip(new_y, 0, self.ground_height)
            dis_fly = np.sqrt(
                (new_x - self.uav.loc[0]) ** 2 + (new_y - self.uav.loc[1]) ** 2
            )
            uav_angle = math.atan2(new_y - self.uav.loc[1], new_x - self.uav.loc[0])
            out_of_border_penalty = -10
            action_invalid_penalty = -10
        # 如果选择卸载的节点没有任务

        if self.nodeList[ue_id].current_cache_size == 0:
            #
            self.find_other_nodes(self.nodeList[ue_id])
            find_notask_node_penalty = -10
            action_invalid_penalty = -10

        # 第一步：先飞行
        flying_energy, flying_reward = self.uav.moveto(
            dis_fly, uav_angle, self.nodeList[ue_id].loc
        )  # 传入新的距离和原来的角度
        self.current_time += self.t_fly  # 更新当前时间
        # 第二步：进行计算
        delay, consumed_energy = self.com_delay(self.nodeList[ue_id], offloading_ratio)
        # 除了被卸载的节点部分卸载，其他结点完全本地计算
        for node in self.nodeList:
            if node is not self.nodeList[ue_id]:
                time = node.local_offloading(self.current_time)  # 完全本地卸载
                # node.generateTask(current_time=self.current_time)
                local_time += time

        charge_time = 1  # 先占位充电时间占1s
        # 第三步：能量收集
        harvest_energy = self.charger.charge(self.uav, charge_time)
        # 更新当前时间
        if harvest_energy > 0:
            self.current_time += charge_time

        # 第四步：所有节点重新生成任务
        for node in self.nodeList:
            node.generateTask(self.current_time)  # 重新生成任务

        # 惩罚1：重复服务ue惩罚
        recent_count = list(self.service_history).count(ue_id)
        offloading_imbalance_penalty = -10 * recent_count  # 每重复一次惩罚增加10

        # reward = (
        #     -delay / 10
        #     - consumed_energy / 10
        #     + out_of_border_penalty
        #     + find_notask_node_penalty
        #     + harvest_energy * 0.1  # 收集能量的奖励
        #     + action_invalid_penalty
        #     + offloading_imbalance_penalty
        # )
        # reward = (
        #     -consumed_energy * 0.01
        #     - flying_energy * 0.01
        #     - delay
        #     + out_of_border_penalty
        #     + flying_reward
        # )
        static_penalty = 0  # 静态惩罚
        if flying_reward == -0.0:
            static_penalty = -30  # 静态惩罚

        # reward = (
        #     -delay
        #     + flying_reward
        #     + out_of_border_penalty
        #     + static_penalty
        #     + offloading_imbalance_penalty
        # )
        reward = -delay
        # print(f"reward: {reward}, flying_reward: {flying_reward}, delay: {delay},")
        # 判断当前飞行是否结束？
        # if all(node.task_queue for node in self.nodeList):
        #     is_terminal = True
        if self.uav.e_battery < 0:
            is_terminal = True  # 电量耗尽

        return self._get_obs(), reward, is_terminal
        # 引入奖励：运动了且不出界要有奖励；卸载任务成功（且不为1）要有奖励；收集能量了要有奖励；离ue近了要有奖励;速度平滑奖励

    def com_delay(self, served_ue: UENode, offloading_ratio):
        # 这个函数的目的是为了计算获得服务的UE的延迟

        # 同时我们会在这里更新当前时间

        # 注意：在执行该函数的时候我们默认ue是有任务的
        # 因为在step函数里我们会对没有任务的UE进行邻居任务转移

        # 这里有一些问题 0.1的卸载率，如果只有一个任务，local_task_num会是0;但是不符合实际情况
        # 是否应该引入惩罚？
        # 我们如何判断应该调度什么任务？

        # 计算有多少任务需要本地计算
        local_task_num = int(len(served_ue.task_queue) * (1 - offloading_ratio))

        # Retrieve tasks for local and UAV processing as separate arrays
        local_tasks = []
        uav_tasks = []

        for i, task in enumerate(served_ue.task_queue):
            if i < local_task_num:
                local_tasks.append(task)
            else:
                uav_tasks.append(task)

        # 1、本地计算时延
        # 返回每个本地卸载任务的delay和最后一个任务的完成时间
        t_local_com, after_local_time = served_ue.partial_offloading(
            local_tasks, self.current_time
        )

        # 2、无人机传输+计算下时延
        total_data_size = sum(task.data_size for task in served_ue.task_queue)
        offload_data_size = total_data_size * offloading_ratio

        # 计算传输速率
        dx = self.uav.loc[0] - served_ue.loc[0]
        dy = self.uav.loc[1] - served_ue.loc[1]
        dh = self.uav.flying_height
        dist_uav_ue = np.sqrt(dx * dx + dy * dy + dh * dh)  # 计算uav和ue之间距离
        g_uav_ue = abs(self.alpha0 / dist_uav_ue**2)  # 信道增益
        p_noisy_los = 10 ** (-13)  # 噪声功率-100dBm
        p_noisy_nlos = 10 ** (-11)  # 噪声功率-80dBm
        # 现在暂时先用的是p_noisy_los
        trans_rate = self.B * math.log2(
            1 + self.p_uplink * g_uav_ue / p_noisy_los
        )  # 上行链路传输速率bps

        # 传输时延
        t_tr = offload_data_size / trans_rate  # 上传时延,1B=8bit
        # uav会调用自己的offload函数，返回delay和耗能,还有处理完后的时间
        t_edge_com, uav_consumed_energy, after_uav_time = self.uav.offload(
            uav_tasks, self.current_time + t_tr
        )

        self.current_time = max(after_local_time, after_uav_time)  # 更新现在时间

        delay = max(t_local_com, t_tr + t_edge_com)  # 总时延
        # 本地计算+传输+uav计算的总时延(其实这里的计算不太对,感觉怪怪的)

        # ue的所有任务都通过uav+本地卸载完成
        # Ensure all tasks are processed or marked before clearing the queue
        for task in served_ue.task_queue:
            if task.status == 0:  # If a task is unprocessed
                logging.warning(f"Unprocessed task found: {task}")
                task.status = 3  # Mark it as discarded
        served_ue.task_queue.clear()  # Clear the task queue after verification
        return delay, uav_consumed_energy
        # 获得状态

    # 这个函数本来是为了降维 现在还没有实现 只是单纯的取uenode数据出来
    def ue_state_process(self):
        self.nodeList
        raw_data = []
        for node in self.nodeList:
            # 这里是把每个节点的状态信息放到一个数组里面
            # raw_data.extend(
            #     [
            #         node.loc[0],
            #         node.loc[1],
            #         node.current_cache_size / node.cache_capacity,
            #         node.local_capacity / self.uav.f_uav,
            #     ]
            # )
            # raw_data.extend([sum(task.data_size for task in node.task_queue)])
            dx = (node.loc[0] - self.uav.loc[0]) / self.ground_width
            dy = (node.loc[1] - self.uav.loc[1]) / self.ground_height
            raw_data.extend(
                [
                    dx,
                    dy,
                    node.current_cache_size / node.cache_capacity,
                    node.local_capacity / self.uav.f_uav,
                ]
            )
        # print("raw_data:", raw_data.__len__())
        return raw_data  # 返回一个数组，包含了所有节点的信息

    def _get_obs(self):
        state = []  # 初始化状态数组
        # 1. 无人机的状态信息
        state.append(
            self.uav.e_battery / self.uav.battery_capacity
        )  # 无人机现在的电量比
        state.append(self.uav.loc[0] / self.ground_width)  # 无人机现在的x坐标
        state.append(self.uav.loc[1] / self.ground_height)
        # 计算uav到边界的距离
        distance_to_border = (
            min(
                self.uav.loc[0],
                self.ground_width - self.uav.loc[0],
                self.uav.loc[1],
                self.ground_height - self.uav.loc[1],
            )
            / self.ground_width
        )  # 归一化到[0,1]
        distance_to_charger = (
            np.sqrt(
                (self.uav.loc[0] - self.charger.loc[0]) ** 2
                + (self.uav.loc[1] - self.charger.loc[1]) ** 2
            )
            / self.ground_width
        )  # 归一化到[0,1]
        state.append(distance_to_charger)
        state.append(distance_to_border)
        state.append(self.uav.flying_speed)

        # 2. 充电器的状态信息
        # state.extend(self.charger.loc)  # 充电器位置 [x, y]
        state.append(self.charger.loc[0] / self.ground_width)  # 充电器x坐标
        state.append(self.charger.loc[1] / self.ground_height)  # 充电器y坐标
        # 这个函数应该返回一个embed数组，加入到self.state中
        state.extend(self.ue_state_process())
        # print("state:", state.__len__())
        # 这里返回的是一个python数组，在main loop里会转换成一个np array数组
        return state

    def find_other_nodes(self, ue_node):
        # 把离当前节点最近节点的任务卸载过来
        min_distance_sq = float("inf")
        nearest_node = None

        for node in self.nodeList:
            if node is ue_node or not node.task_queue:
                continue  # 跳过自身和空任务队列节点
            # 计算欧氏距离平方（避免开根号提升性能）
            dx = ue_node.loc[0] - node.loc[0]
            dy = ue_node.loc[1] - node.loc[1]
            distance_sq = dx**2 + dy**2
            # 更新最近节点
            if distance_sq < min_distance_sq:
                min_distance_sq = distance_sq
                nearest_node = node

        # 检查最近节点是否存在且有任务
        if nearest_node is None:
            logging.warning("No suitable neighbor node found with tasks to offload.")
            return

        remaining_cache = ue_node.cache_capacity - ue_node.current_cache_size
        if remaining_cache <= 0:
            logging.warning("Target node's cache is full. No tasks transferred.")
            return

        # 任务转移控制参数
        transferred_count = 0  # 传输任务的数量
        max_transfer_count = 5  # 最大传输任务数量
        current_attempt = 0  # 当前尝试次数

        while (
            current_attempt < max_transfer_count
            and nearest_node.task_queue
            and remaining_cache > 0
        ):
            # 传输任务
            current_task = nearest_node.task_queue[0]  # 从最近节点的队列中取出任务

            if current_task.data_size > remaining_cache:
                logging.debug(
                    f"Task size {current_task.data_size} exceeds remaining cache {remaining_cache}. Stopping."
                )
                break

            # 执行任务转移
            transferred_task = nearest_node.task_queue.popleft()
            # 在原来的结点减去cache_size current_cache_size指的是现在缓存已经占用了多少
            nearest_node.current_cache_size -= transferred_task.data_size

            ue_node.task_queue.append(transferred_task)  # 将任务添加到当前节点的队列中
            ue_node.current_cache_size += transferred_task.data_size
            remaining_cache -= transferred_task.data_size  # 更新剩余缓存
            transferred_count += 1
            current_attempt += 1  # 增加尝试次数

        if transferred_count > 0:
            logging.info(
                f"Successfully transferred {transferred_count} tasks from {nearest_node}."
            )
        else:
            logging.warning("No tasks transferred despite valid neighbor.")

    # 当目前的节点没有任务时，把附近节点的任务传送过来
    #    这里可以作为一个创新点
    #        1.构建邻近节点集。（问题有，如何挑选节点？传过来多少任务？）
    # 挑选节点：看和节点距离，传过来需要时间t1，再传给uav需要时间t2,uav计算需要时间t3.(忽略返回时间)
    # 接下来的问题：没算完怎么办？
    # 引申问题，在uav没有跟ue建立连接的时候，ue自己也是会进行计算的。所以stage1飞行阶段和stage2能量收集阶段 ue本身在进行计算
    #         先选择最简单的 nearest


def all_local():
    env = Env()
    env.reset()

    MAX_EPISODES = 50

    for i in range(MAX_EPISODES):
        env.reset()
        episode_delay = 0
        for j in range(env.slot_num):
            timeslot_delay = 0
            for node in env.nodeList:
                local_delay = node.local_offloading(current_time=j + 1)
                node.generateTask(current_time={j + 1})
                timeslot_delay += local_delay
            episode_delay += timeslot_delay
        print(f"第{i}个episode的总时延是：{episode_delay}")


def greedy_test():
    np.random.seed(48)  # 设置随机种子

    env = Env()
    env.reset()
    # a = 5
    MAX_EPISODES = 50
    ep_reward_list = []  # 用于存储每个episode的奖励
    for i in range(MAX_EPISODES):
        env.reset()
        episode_reward = 0
        flying_trajectory = []  # 用于存储飞行轨迹
        for j in range(env.slot_num):
            flying_trajectory.append(env.uav.loc.copy())  # 记录飞行轨迹
            served_ue = np.random.randint(0, env.ue_num - 1)  # 随机选择一个UE进行服务
            dis_x = env.nodeList[served_ue].loc[0] - env.uav.loc[0]
            dis_y = env.nodeList[served_ue].loc[1] - env.uav.loc[1]
            flying_angle = math.atan2(dis_y, dis_x)  # 计算飞行角度
            flying_distance = np.sqrt(dis_x**2 + dis_y**2)
            if flying_distance > env.uav.flying_speed:
                flying_distance = 1.0
            else:
                flying_distance = flying_distance / env.uav.flying_speed
            offloading_ratio = 1.0  # 随机选择一个卸载比例
            action = {
                "ue": served_ue,
                "angle": flying_angle,
                "speed": flying_distance,
                "ratio": offloading_ratio,
            }
            s_, r, is_terminal = env.step(action)  # 执行动作
            episode_reward += r  # 累加奖励

            if j == env.slot_num - 1 or is_terminal:
                ep_reward_list.append(episode_reward)
                print(f"Episode {i + 1} finished with reward: {episode_reward:.2f}")
                if (i + 1) % 2 == 0:
                    env.visualize_nodes(flying_trajectory, i)
                with open("output_static.txt", "a") as f:
                    f.write(f"Episode #{i + 1} Finished  ===========\n")
                    f.write(
                        f"\nThis episode finished within {j} steps , accumulated reward: {episode_reward:.2f}"
                    )
                break
    print(f"Average reward per episode: {np.mean(ep_reward_list):.2f}")
    plt.plot(ep_reward_list)
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title("Episode vs Reward")
    plt.savefig("reward_plot.png")
    plt.show()


def main():
    # all_local()

    env = Env()
    _ = env.reset()
    action = {
        "ue": 0,  # 选择服务的UE编号
        "angle": 0.5,  # 飞行角度（弧度）
        "speed": 0.5,  # 飞行距离比例（0到1之间）
        "ratio": 0.5,  # 卸载比例（0到1之间）
    }
    env.step(action)


def test_environment():
    env = Env()
    max_episodes = 3
    max_steps = 20

    for episode in range(max_episodes):
        print(f"\n=== Starting Episode {episode + 1} ===")
        _ = env.reset()
        done = False
        total_reward = 0
        step_count = 0

        while not done and step_count < max_steps:
            # 生成随机动作 [ue_id, angle, distance, ratio]
            action = [
                random.randint(0, env.ue_num - 1),  # ue_id
                random.uniform(0, 2 * math.pi),  # 角度（弧度）
                random.uniform(0, 1),  # 距离比例
                random.uniform(0, 1),  # 卸载比例
            ]

            try:
                next_state, reward, done = env.step(action)
                total_reward += reward
                step_count += 1

                # 打印步骤信息
                print(f"Step {step_count}:")
                print(
                    f"Action: UE={action[0]}, Angle={action[1]:.2f}, Dist={action[2]:.2f}, Ratio={action[3]:.2f}"
                )
                print(
                    f"Reward: {reward:.2f}, UAV Energy: {env.uav.e_battery:.2f}/{env.uav.battery_capacity}"
                )
                print(f"Terminal: {done}\n")

                # 检查状态是否合法
                if any(math.isnan(s) for s in next_state):
                    print("Error: State contains NaN values!")
                    return

            except Exception as e:
                print(
                    f"\n!!! Error occurred at Episode {episode + 1}, Step {step_count}:"
                )
                print(f"Action: {action}")
                print(f"Exception: {str(e)}")
                import traceback

                traceback.print_exc()
                return

        print(
            f"Episode {episode + 1} finished after {step_count} steps. Total Reward: {total_reward:.2f}"
        )


if __name__ == "__main__":
    # test_environment()
    # all_local()
    greedy_test()
