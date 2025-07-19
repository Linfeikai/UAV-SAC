# 强化学习核心库
import gymnasium as gym
from gymnasium import spaces
from gymnasium.envs.registration import register

# 数学与数据处理
import numpy as np

# 可视化
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Arrow

# 自定义的类
from .uav import UAVNode
from .uenode import UENode
from .lasercharger import LaserCharger
from collections import deque
from .all_config import (
    GROUND_WIDTH,
    GROUND_HEIGHT,
    CACHE_SIZE,
    DATA_SIZE_RANGES,
    CPU_RANGES,
    Nodetype,
)

import logging
from typing import List, Tuple
import math


class CustomEnv(gym.Env):
    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 30,
    }

    # --- 1. 配置与代码分离：将所有可调参数和常量集中管理 ---
    class Config:
        # 权重
        W_DELAY = 1.5  # 延迟惩罚的相对重要性
        W_ENERGY_CONSUMED = 1.0  # 能耗惩罚的相对重要性
        W_ENERGY_GAINED = 1.0  # 充电奖励的相对重要性
        W_FAIRNESS = 0.2  # 公平性奖励的相对重要性 (次要目标)
        W_PBRS = 0.1  # PBRS引导奖励的相对重要性 (引导项)

        # 绝对惩罚值 (这些值不参与归一化)
        PENALTY_TASK_FAILURE = -2.0  # 任务失败的惩罚
        PENALTY_OUT_OF_BOUNDS = -2.0
        PENALTY_STATIC = -1.0

        # 归一化分母 (通过运行基准得到)
        NORM_DELAY = 80.0  # 经验最大总延迟
        NORM_ENERGY_CONSUMED = 3000.0  # 经验最大总消耗
        NORM_ENERGY_GAINED = 2500.0  # 经验最大总充电
        NORM_FLYING_DISTANCE = 20  # 经验最大飞行距离

        PENALTY_NO_TASK_NODE = -1.0  # 一个没用上的惩罚

    # --- 2. 环境核心参数 ---
    ground_width = GROUND_WIDTH  # 场地宽度
    ground_height = GROUND_HEIGHT  # 场地高度
    task_type_distribution = {  # 任务类型分配
        "normal": 0.6,
        "moderate": 0.3,
        "hpc": 0.1,
    }
    ue_num = 20  # UE设备的数量：20
    s = 1000  # 单位bit处理所需cpu圈数1000

    state_dim = 86  # 状态空间维度 # 距离border，距离充电器，dx,dy,speed,电量+80=86
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
    t_com = 7  # 悬停时间是7s
    delta_t = t_fly + t_com  # 1s飞行, 后7s用于悬停计算
    slot_num = int(T / delta_t)  # 40个间隔

    def __init__(
        self, render_mode=None
    ):  # 离散值：选择ue；连续值：距离，方向，卸载比率
        super(CustomEnv, self).__init__()

        # 1. 离散部分：选择要服务的UE，范围是 [0, ue_num-1]
        self.discrete_action_space = spaces.Discrete(self.ue_num)

        # 2. 连续部分：角度、速度比例、卸载率
        self.continuous_action_space = spaces.Box(
            low=np.array(
                [-np.pi, 0, 0], dtype=np.float32
            ),  # angle, velocity_ratio, offloading_ratio
            high=np.array([np.pi, 1, 1], dtype=np.float32),
            dtype=np.float32,
        )

        # 组合成一个混合动作空间
        self.action_space = spaces.Tuple(
            (self.discrete_action_space, self.continuous_action_space)
        )

        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(86,), dtype=np.float32
        )

        self.nodeList: List[UENode] = []
        # --- 关键优化：将节点属性向量化，存储在NumPy数组中 ---
        # 这使得我们可以进行高效的批量计算，避免在step中进行Python循环。
        self.ue_locs = np.zeros((self.ue_num, 2), dtype=np.float32)
        self.ue_cache_capacities = np.zeros(self.ue_num, dtype=np.float32)
        self.ue_local_capacities = np.zeros(self.ue_num, dtype=np.float32)
        # --- 动态属性，需要在每一步更新 ---
        self.ue_cache_sizes = np.zeros(self.ue_num, dtype=np.float32)

        # 有条件地初始化历史记录列表，以节省训练时的内存
        # Only create these lists if we are in 'human' render mode to save memory during training.
        self.render_mode = render_mode
        if self.render_mode == "human":
            # --- 用于渲染的数据容器 ---
            self.flying_trajectory = []
            self.uav_battery_progress = []
            self.served_ue_history = []
            self.flying_energy_progress = []
            self.com_energy_progress = []
            self.charging_energy_progress = []
            self.reward_components_history = []
            self.battery_at_decision = []
            self.offloading_ratio_at_decision = []
            self.uav_flying_speed_progress = []

        else:
            self.uav_flying_speed_progress = None
            self.flying_trajectory = None
            self.uav_battery_progress = None
            self.served_ue_history = None
            self.flying_energy_progress = None
            self.com_energy_progress = None
            self.charging_energy_progress = None
            self.reward_components_history = None
            self.battery_at_decision = None
            self.offloading_ratio_at_decision = None

        self.uav = UAVNode()
        self.charger = LaserCharger()
        # 需要频繁地向数组中添加元素，我们先将 self.state 作为列表处理，最后再转换为 NumPy 数组。
        # 这种方法在添加元素时效率更高，因为 NumPy 数组的大小是固定的，而列表的大小是动态的。
        self.service_history = deque(maxlen=5)  # 记录最近5次服务对象
        self.service_counts = np.zeros(self.ue_num, dtype=np.int32)  # 用于计算公平性
        self.current_step = 0  # 当前步数
        self.current_time = 0.0  # 当前时间

        self.previous_fairness_index = 1.0  # 初始化公平性指数

        # # =====================================================================
        # # [临时调试代码] 用于跟踪整个运行期间遇到的最大值
        # # =====================================================================
        # self.max_delay_seen_for_debug = 0.0
        # self.max_energy_consumption_debug = 0.0
        # self.max_energy_charged_debug = 0.0
        # # =====================================================================

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.fig, self.ax = None, None  # Matplotlib 图形对象

    def com_delay(self, served_ue: UENode, offloading_ratio, hover_time):
        """计算服务一个UE所需的总延迟和UAV能耗。

        这个函数现在是一个纯计算函数，它不修改 self.current_time。
        它返回理论上的延迟、能耗和任务完成后的绝对时间点。
        """

        """
        因为我现在已经在被服务的ue和uav的节点中都对时间进行了判断
        所以他们的计算时间是不会超出可用时间的
        那么到底是否超时，只能通过遍历当前服务结点的tasklist看状态
        """
        local_task_num = int(len(served_ue.task_queue) * (1 - offloading_ratio))
        local_tasks = []
        uav_tasks = []
        for i, task in enumerate(served_ue.task_queue):
            if i < local_task_num:
                local_tasks.append(task)
            else:
                uav_tasks.append(task)

        is_task_failed = False
        served_ue_task_penalty = 0.0
        # 1. 本地计算部分
        # 注意：这里的 self.current_time 只是作为计算的起点
        t_local_com, after_local_time = served_ue.partial_offloading(
            local_tasks, self.current_time, hover_time
        )

        # 2. 无人机计算部分
        total_data_size = sum(task.data_size for task in served_ue.task_queue)
        offload_data_size = total_data_size * offloading_ratio

        # 传输速率计算
        # --- 优化点：直接从向量化数组中获取位置信息 ---
        dist_vec = self.uav.loc - self.ue_locs[served_ue.node_id]
        dh = self.uav.flying_height
        dist_uav_ue = np.sqrt(np.sum(dist_vec**2) + dh**2)
        g_uav_ue = abs(self.alpha0 / dist_uav_ue**2)  # 信道增益
        # 现在暂时先用的是p_noisy_los
        trans_rate = self.B * math.log2(
            1 + self.p_uplink * g_uav_ue / self.p_noisy_los
        )  # 上行链路传输速率bps

        t_tr = offload_data_size / trans_rate if trans_rate > 0 else float("inf")

        # UAV 计算
        t_edge_com, uav_consumed_energy, after_uav_time = self.uav.offload(
            uav_tasks, self.current_time + t_tr, hover_time - t_tr
        )

        # 3. 计算总延迟（时间差）和完成时间点
        delay = max(t_local_com, t_tr + t_edge_com)
        after_calculation_time = t_tr + t_edge_com
        completion_time = max(after_local_time, after_uav_time)

        for task in served_ue.task_queue:
            if task.status == 0:  # If a task is unprocessed
                is_task_failed = True
                # logging.warning(f"Unprocessed task found: {task}")
                # 任务未处理，给予惩罚
                if self.render_mode == "human":
                    self.failed_tasks += 1
                task.status = 3  # Mark it as discarded
        if is_task_failed:
            served_ue_task_penalty = self.Config.PENALTY_TASK_FAILURE
        served_ue.task_queue.clear()  # Clear the task queue after verification
        served_ue.current_cache_size = 0  # 清空缓存
        return (
            delay,
            uav_consumed_energy,
            completion_time,
            after_calculation_time,
            served_ue_task_penalty,
        )

    def calculate_boundary_penalty(self, new_loc):
        """
        计算无人机越界惩罚
        :param new_loc: 无人机的新位置
        :return: 越界惩罚值
        (可以扩展为多无人机系统的越界惩罚，现在只有一个)
        """
        margin = 10  # 增加缓冲带
        x, y = new_loc
        penalty = 0

        if x < margin:
            penalty += (margin - x) / margin
        elif x > self.ground_width - margin:
            penalty += (x - (self.ground_width - margin)) / margin

        if y < margin:
            penalty += (margin - y) / margin
        elif y > self.ground_height - margin:
            penalty += (y - (self.ground_height - margin)) / margin

        return -penalty

    def _update_uav_flight(
        self, ue_id: int, angle: float, velocity_ratio: float
    ) -> dict:
        """处理无人机飞行，返回飞行结果和相关惩罚。
        飞行能耗（我们希望最小化）；飞行奖励（靠近ue的奖励）；越界惩罚；静止惩罚；悬停时间。
        """
        dis_fly = self.uav.max_speed * velocity_ratio * self.t_fly
        new_x = self.uav.loc[0] + dis_fly * math.cos(angle)
        new_y = self.uav.loc[1] + dis_fly * math.sin(angle)

        # 检查是否越界
        out_of_border_penalty = 0
        if not (0 <= new_x <= self.ground_width and 0 <= new_y <= self.ground_height):
            out_of_border_penalty = self.Config.PENALTY_OUT_OF_BOUNDS
            # 停留在边界处
            new_x = np.clip(new_x, 0, self.ground_width)
            new_y = np.clip(new_y, 0, self.ground_height)
            dis_fly = np.sqrt(
                (new_x - self.uav.loc[0]) ** 2 + (new_y - self.uav.loc[1]) ** 2
            )
            angle = math.atan2(new_y - self.uav.loc[1], new_x - self.uav.loc[0])

        # 检查是否静止
        # --- 改进：使用平滑函数计算静止惩罚 ---
        # 当 dis_fly 接近0时，惩罚接近 PENALTY_STATIC；当 dis_fly 增大时，惩罚迅速减小
        # k值越小，惩罚衰减越快。k=2.5 意味着在5米处惩罚约为最大值的13%
        static_penalty = self.Config.PENALTY_STATIC * math.exp(-dis_fly / 2.5)

        # 执行飞行
        flying_energy, flying_reward = self.uav.moveto(
            dis_fly, angle, self.nodeList[ue_id].loc
        )
        self.uav.flying_speed = dis_fly

        if self.flying_trajectory is not None:
            self.flying_trajectory.append(self.uav.loc.copy())

        return {
            "flying_energy": flying_energy,
            "flying_distance": flying_reward,
            "out_of_border_penalty": out_of_border_penalty,
            "static_penalty": static_penalty,
            "hover_time": self.delta_t - self.t_fly,
        }

    def _calculate_system_delay(
        self, ue_id: int, offloading_ratio: float, hover_time: float
    ) -> dict:
        """计算整个系统的延迟，并检查任务是否失败。"""
        total_system_delay = 0.0

        # 处理被服务的UE
        (
            served_ue_delay,
            consumed_energy,
            served_ue_completion_time,
            after_calculation_time,
            served_ue_task_penalty,
        ) = self.com_delay(self.nodeList[ue_id], offloading_ratio, hover_time)
        total_system_delay += served_ue_delay

        # --- 核心逻辑修正 ---
        # 任务成功，但我们还不能立刻更新全局时间。
        # 我们需要先计算出所有节点的完成时间，然后取最晚的那个。
        local_failure_penalty = 0.0
        latest_completion_time = served_ue_completion_time

        # 在任务成功后，用剩余的悬停时间进行充电 ---
        charge_time = hover_time - after_calculation_time
        if charge_time > 0:
            # 调用充电函数，获取充入的电量
            harvested_energy = self.charger.charge(self.uav, charge_time)
        else:
            harvested_energy = 0.0

        # 处理其他未被服务的UE（它们在本地计算），并找出最晚的完成时间
        for node in self.nodeList:
            if node is not self.nodeList[ue_id]:
                # 本地节点用整个 delta_t 作为可用计算时间
                local_delay, local_completion_time = node.local_offloading(
                    self.current_time, self.delta_t
                )

                # 检查本地计算是否成功处理完所有任务
                # 因为在node的local_offloading函数里会把已经完成的任务给popleft
                if len(node.task_queue) > 0:
                    # 如果有任务剩余，视为任务失败
                    # print(
                    #     f"警告：在步骤 {self.current_step}，本地节点 {node.loc} 未能处理完所有任务，剩余 {len(node.task_queue)} 个。"
                    # )
                    # 训练智能体具备宏观风险意识的关键机制。
                    # 当这个叠加惩罚出现时，它会给智能体的学习过程带来一次强烈的“震撼教育”。
                    local_failure_penalty += self.Config.PENALTY_TASK_FAILURE
                    # 将剩余任务标记为丢弃并清空队列，以避免累积积压
                    for task in node.task_queue:
                        task.status = 3  # Discarded
                    if self.render_mode == "human":
                        self.failed_tasks += len(node.task_queue)
                    node.task_queue.clear()
                    node.current_cache_size = 0
                else:
                    # 任务成功，计入总延迟并更新最晚完成时间
                    total_system_delay += local_delay
                    if local_completion_time > latest_completion_time:
                        latest_completion_time = local_completion_time

        return {
            "delay": total_system_delay,
            "consumed_energy": consumed_energy,
            "harvested_energy": harvested_energy,  # 将充电量返回
            "local_failure_penalty": local_failure_penalty,
            "served_ue_task_penalty": served_ue_task_penalty,
        }

    def _update_system_state(self):
        """更新充电和任务生成等系统状态。"""
        # 记录当前时间点的电池电量（可能刚刚被充过电）
        if self.uav_battery_progress is not None:
            self.uav_battery_progress.append(self.uav.e_battery)

        # 所有节点生成新任务
        for node in self.nodeList:
            node.generateTask(self.current_time)

    def _compute_reward(
        self,
        flight_info: dict,
        delay_info: dict,
        fairness_improvement: float,
    ) -> float:
        """根据所有组件计算最终的奖励值。"""
        # 创建一个字典来存放所有奖励组件
        reward_components = {}

        # 1.获取原始数值：
        raw_delay = delay_info.get("delay", 0.0)
        raw_flying_energy = flight_info.get("flying_energy", 0.0)
        raw_flying_distance = flight_info.get("flying_distance", 0.0)
        raw_computing_energy = delay_info.get("consumed_energy", 0.0)
        raw_total_consumption = raw_flying_energy + raw_computing_energy
        raw_harvested_energy = delay_info.get("harvested_energy", 0.0)

        # --- 2. 计算所有归一化的奖励/惩罚子项 (值都在0-1之间) ---
        norm_delay = raw_delay / self.Config.NORM_DELAY  # 时延
        reward_components["delay_penalty"] = -norm_delay

        norm_consumption = (
            raw_total_consumption / self.Config.NORM_ENERGY_CONSUMED
        )  # 能耗

        reward_components["energy_penalty"] = -norm_consumption

        norm_gain = raw_harvested_energy / self.Config.NORM_ENERGY_GAINED  # 充电
        reward_components["charge_reward"] = norm_gain

        norm_pbrs = (
            raw_flying_distance / self.Config.NORM_FLYING_DISTANCE
        )  # PBRS引导奖励
        reward_components["pbrs_reward"] = norm_pbrs

        reward_components["fairness_improvement"] = fairness_improvement

        # # --- 新增：低电量惩罚 ---
        # low_battery_penalty = 0.0
        # # --- 改进：使用平滑函数计算低电量惩罚 ---
        # # 使用一个S型函数，当电量低于某个阈值（如0.4）时，惩罚平滑地增加
        # battery_ratio = self.uav.e_battery / self.uav.battery_capacity
        # # scale=15 控制曲线的陡峭程度，threshold=0.4 是惩罚开始显著增加的电量水平
        # penalty_factor = 1 / (1 + math.exp((battery_ratio - 0.4) * 15))
        # low_battery_penalty = (
        #     self.Config.REWARD_WEIGHT_LOW_BATTERY_PENALTY * penalty_factor
        # )

        # --- 3. 应用权重，计算核心奖励 ---
        reward = 0.0
        reward -= self.Config.W_DELAY * norm_delay
        reward -= self.Config.W_ENERGY_CONSUMED * norm_consumption
        reward += self.Config.W_ENERGY_GAINED * norm_gain
        reward += self.Config.W_PBRS * norm_pbrs
        reward += self.Config.W_FAIRNESS * fairness_improvement

        # --- 4. 叠加上下文相关的绝对惩罚 ---
        # 任务失败的惩罚现在从delay_info里获取

        served_ue_penalty = delay_info.get("served_ue_task_penalty", 0.0)
        local_failure_penalty = delay_info.get("local_failure_penalty", 0.0)
        out_of_border_penalty = flight_info.get("out_of_border_penalty", 0.0)
        static_penalty = flight_info.get("static_penalty", 0.0)

        reward = reward + (
            served_ue_penalty
            + local_failure_penalty
            + out_of_border_penalty
            + static_penalty
        )
        # 将它们也存入字典以供观察
        reward_components["served_ue_penalty"] = served_ue_penalty
        reward_components["local_failure_penalty"] = local_failure_penalty
        reward_components["out_of_border_penalty"] = out_of_border_penalty
        reward_components["static_penalty"] = static_penalty

        # # no_Task_penalty 没有加入，是因为当前env设置下所有node会重新生成任务。
        # reward = (
        #     self.Config.REWARD_WEIGHT_DELAY * r_norm_delay  # 时延奖励
        #     + low_battery_penalty  # 低电量要给惩罚吗？
        #     + charging_reward  # 充电了要给奖励吗？ -> 如果给了会不会reward hacking？
        #     + self.Config.REWARD_WEIGHT_FLYING_ENERGY * r_norm_flying_energy  # 能耗奖励
        #     + delay_info[
        #         "local_failure_penalty"
        #     ]  # 本地任务失败的惩罚 ->这个惩罚应该多大呢？
        #     + flight_info["out_of_border_penalty"]  # 出界惩罚
        #     + self.Config.REWARD_WEIGHT_FLYING_CLOSER
        #     * flight_info["flying_reward"]  # 飞行奖励
        #     + flight_info["static_penalty"]  # 静止不动惩罚
        #     + fairness_penalty  # 公平性惩罚
        # )
        return float(reward), reward_components

    def step(
        self, action: Tuple[int, np.ndarray]
    ) -> Tuple[np.ndarray, float, bool, bool, dict]:
        self.current_step += 1  # 增加当前步数

        # 在执行任何动作之前，记录当前的电池状态
        battery_before_step = self.uav.e_battery

        # 1. 解码动作
        ue_id, continuous_parts = action
        angle = float(continuous_parts[0])
        velocity_ratio = float(continuous_parts[1])
        offloading_ratio = float(continuous_parts[2])

        # 如果是渲染模式，则收集决策数据;并收集当前任务数量
        if self.render_mode == "human":
            self.battery_at_decision.append(
                battery_before_step / self.uav.battery_capacity
            )
            self.offloading_ratio_at_decision.append(offloading_ratio)
            for node in self.nodeList:
                self.total_tasks += len(node.task_queue)
            # 更新服务历史和计数
            self.served_ue_history.append(ue_id)

        self.service_counts[ue_id] += 1

        # 3. 检查选择的UE是否有任务，并给予惩罚 目前不会有这个情况
        no_task_penalty = 0
        if self.nodeList[ue_id].current_cache_size == 0:
            self.find_other_nodes(self.nodeList[ue_id])
            no_task_penalty = self.Config.PENALTY_NO_TASK_NODE

        # 4. 更新无人机飞行状态
        flight_info = self._update_uav_flight(ue_id, angle, velocity_ratio)

        # 5. 计算系统延迟
        delay_info = self._calculate_system_delay(
            ue_id, offloading_ratio, flight_info["hover_time"]
        )

        # 712修改 更新全局时间，确保一个step固定推进 delta_t
        self.current_time += self.delta_t  # 简单、清晰、统一

        # # =====================================================================
        # # [临时调试代码] 跟踪和打印最大延迟、能耗、充电值
        # # =====================================================================
        # # B1. 跟踪最大延迟
        # current_delay = delay_info.get("delay", 0.0)
        # if current_delay > self.max_delay_seen_for_debug:
        #     self.max_delay_seen_for_debug = current_delay
        #     print("\n" + "=" * 60)
        #     print(
        #         f"  [调试] 发现新的最大单步延迟: {self.max_delay_seen_for_debug:.4f}s "
        #         f"(在 step {self.current_step})"
        #     )
        #     print("=" * 60 + "\n")

        # # B2. 跟踪最大单步能耗 (飞行 + 计算)
        # current_consumption = flight_info.get("flying_energy", 0.0) + delay_info.get(
        #     "consumed_energy", 0.0
        # )
        # if current_consumption > self.max_energy_consumption_debug:
        #     self.max_energy_consumption_debug = current_consumption
        #     print("\n" + "#" * 60)
        #     print(
        #         f"  [调试] 发现新的最大单步能耗: {self.max_energy_consumption_debug:.4f}J "
        #         f"(在 step {self.current_step})"
        #     )
        #     print("#" * 60 + "\n")

        # # B3. 跟踪最大单步充电量
        # current_charge = delay_info.get("harvested_energy", 0.0)
        # if current_charge > self.max_energy_charged_debug:
        #     self.max_energy_charged_debug = current_charge
        #     print("\n" + "*" * 60)
        #     print(
        #         f"  [调试] 发现新的最大单步充电: {self.max_energy_charged_debug:.4f}J "
        #         f"(在 step {self.current_step})"
        #     )
        #     print("*" * 60 + "\n")
        # =====================================================================

        # 7. 公平性计算
        current_fairness_index = self.calculate_fairness()
        # 计算公平性的提升量（如果为负，则代表公平性恶化了）
        fairness_improvement = current_fairness_index - self.previous_fairness_index

        # 关键：更新“上一步”的公平性指数，为下一个step做准备
        self.previous_fairness_index = current_fairness_index

        # 8. 计算最终奖励 (在检查终止条件之前)
        reward, reward_components = self._compute_reward(
            flight_info, delay_info, fairness_improvement
        )

        # 9. 检查回合是否终止 (在生成新任务之前)
        # 这是正确的时机，因为它反映了当前动作完成后的状态。
        terminated = self.checkfailure()
        truncated = False

        # 10. 为下一个状态生成新任务
        # 这个操作应该在所有当前步的计算和检查完成后进行。
        self._update_system_state()

        # 11. 组装并返回最终结果
        info = {
            **reward_components,  # 包含所有奖励组件
            **flight_info,
            **delay_info,
            "fairness_improvement": fairness_improvement,
        }
        # 记录用于渲染的能耗分量
        if self.render_mode == "human":
            self.flying_energy_progress.append(flight_info.get("flying_energy", 0))
            self.com_energy_progress.append(delay_info.get("consumed_energy", 0))
            self.charging_energy_progress.append(delay_info.get("harvested_energy", 0))
            # 记录无人机的飞行速度
            self.uav_flying_speed_progress.append(self.uav.flying_speed)

            self.reward_components_history.append(reward_components)
            self.total_episode_delay += delay_info.get("delay", 0.0)

        return self._get_obs(), reward, terminated, truncated, info

    def _get_obs(self):
        # --- 关键修复：在获取观察时，同步动态的UE状态 ---
        # ue_cache_sizes 是动态变化的，必须在每次生成观察时从节点对象列表中更新，
        # 以确保观察值反映的是当前环境的真实状态。
        self.ue_cache_sizes = np.array(
            [n.current_cache_size for n in self.nodeList], dtype=np.float32
        )

        state = []
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
        state.append(self.uav.e_battery / self.uav.battery_capacity)  # 电量比例
        state.append(self.uav.loc[0] / self.ground_width)  # 归一化x坐标
        state.append(self.uav.loc[1] / self.ground_height)  # 归一化y坐标
        state.append(self.uav.flying_speed / self.uav.max_speed)  # 飞行速度比例

        state.append(distance_to_charger)
        state.append(distance_to_border)

        # --- 关键优化：使用NumPy向量化操作构建UE状态，取代原有的ue_state_process循环 ---
        # 1. 批量计算所有UE相对于UAV的归一化位置
        uav_loc_tile = np.tile(self.uav.loc, (self.ue_num, 1))
        relative_locs = (self.ue_locs - uav_loc_tile) / self.ground_width

        # 2. 批量计算缓存占用率
        #    为避免除以零，给分母增加一个极小值
        cache_ratios = self.ue_cache_sizes / (self.ue_cache_capacities + 1e-9)

        # 3. 批量计算本地计算能力与UAV计算能力的比率
        capacity_ratios = self.ue_local_capacities / self.uav.f_uav

        # 4. 将所有UE状态展平并合并到主状态向量中
        ue_states = np.column_stack(
            (relative_locs, cache_ratios, capacity_ratios)
        ).flatten()
        state.extend(ue_states)

        assert len(state) == 86, f"State length mismatch: {len(state)}"
        return np.array(state, dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.uav.reset()  # 重置无人机状态
        self.service_counts.fill(0)  # 重置服务计数
        self.previous_fairness_index = 1.0  # 确保每次新回合开始时，都重置公平性历史

        # --- MODIFICATION: Conditionally reset and record history ---
        if self.render_mode == "human":
            self.flying_trajectory.clear()
            self.uav_battery_progress.clear()
            self.served_ue_history.clear()
            self.flying_energy_progress.clear()
            self.com_energy_progress.clear()
            self.charging_energy_progress.clear()
            self.reward_components_history.clear()
            self.battery_at_decision.clear()
            self.offloading_ratio_at_decision.clear()

            self.total_episode_delay = 0.0
            self.total_tasks = 0
            self.failed_tasks = 0

            self.flying_trajectory.append(self.uav.loc.copy())
            self.uav_battery_progress.append(self.uav.e_battery)

        # --- 5. 结构清晰化：简化UE节点的初始化 ---
        self.nodeList = []
        node_counts = {
            Nodetype.NORMAL: int(self.ue_num * self.task_type_distribution["normal"]),
            Nodetype.MODERATE: int(
                self.ue_num * self.task_type_distribution["moderate"]
            ),
            Nodetype.HPC: int(self.ue_num * self.task_type_distribution["hpc"]),
        }

        # 确保总数正确
        total_nodes = sum(node_counts.values())
        if total_nodes < self.ue_num:
            node_counts[Nodetype.NORMAL] += self.ue_num - total_nodes

        # --- 关键改进：使用 spawn 方法为每个 UE 生成独立的 RNG ---
        # 这是 Gymnasium 和 NumPy 推荐的最佳实践，可以确保完全可复现和独立的随机流。
        # self.np_random 是由 super().reset(seed=seed) 创建的主 RNG。
        ue_rngs = self.np_random.spawn(self.ue_num)
        rng_iterator = iter(ue_rngs)

        for node_type, count in node_counts.items():
            for i in range(count):
                # 从迭代器中为每个节点获取一个独立的 RNG
                node_rng = next(rng_iterator)
                node_id = len(self.nodeList)
                new_node = UENode(nodetype=node_type, rng=node_rng, node_id=node_id)
                self.nodeList.append(new_node)

                # --- 关键优化：在创建节点时，填充向量化数组 ---
                self.ue_locs[node_id] = new_node.loc
                self.ue_cache_capacities[node_id] = new_node.cache_capacity
                self.ue_local_capacities[node_id] = new_node.local_capacity
        self.current_step = 0  # 重置当前步数
        self.current_time = 0.0  # 重置当前时间
        info = {}  # 一定要返回额外信息，我还没添加

        return self._get_obs(), info

    def render(self):
        if self.render_mode != "human":
            return
        else:
            plt.ioff()  # 关闭交互模式
        plt.style.use("seaborn-v0_8-whitegrid")  # 专业图表风格

        fig = plt.figure(figsize=(20, 16), constrained_layout=True)

        # --- 设置信息丰富的总标题 ---
        success_rate = (
            (1 - self.failed_tasks / self.total_tasks) if self.total_tasks > 0 else 1.0
        )
        fig.suptitle(
            f"Episode Summary (Steps: {self.current_step}) | "
            f"Task Success: {success_rate:.2%} | "
            f"Avg Delay: {self.total_episode_delay / self.total_tasks if self.total_tasks > 0 else 0:.2f}s | "
            f"Final Fairness: {self.previous_fairness_index:.3f}",
            fontsize=16,
            weight="bold",
        )

        # --- 创建 3x3 网格布局以容纳更多图表 ---
        gs = fig.add_gridspec(3, 3)
        ax_trajectory = fig.add_subplot(gs[0, 0])  # 轨迹图，占用2x2
        ax_battery = fig.add_subplot(gs[0, 1])  # 电池图
        ax_uav_speed = fig.add_subplot(gs[0, 2])  # 添加无人机速度图
        ax_reward_comp = fig.add_subplot(gs[1, 0])  # 奖励分解图
        ax_service_hist = fig.add_subplot(gs[1, 1])  # 服务直方图
        ax_energy_pie = fig.add_subplot(gs[1, 2])  # 能耗饼图
        ax_decision = fig.add_subplot(gs[2, 0])  # 新增：决策图

        # --- 1. 轨迹图 ---
        ax_trajectory.set_title("UAV Trajectory & Node Distribution")
        ax_trajectory.set_xlim(0, self.ground_width)
        ax_trajectory.set_ylim(0, self.ground_height)
        ax_trajectory.set_aspect("equal")
        ax_trajectory.set_xlabel("X Coordinate")
        ax_trajectory.set_ylabel("Y Coordinate")

        node_colors = {
            Nodetype.NORMAL: "green",
            Nodetype.MODERATE: "blue",
            Nodetype.HPC: "red",
        }
        for node in self.nodeList:
            ax_trajectory.scatter(
                node.loc[0],
                node.loc[1],
                color=node_colors[node.nodetype],
                label=node.nodetype.name,
                s=100,
                alpha=0.7,
                edgecolors="w",
            )
        ax_trajectory.scatter(
            self.charger.loc[0],
            self.charger.loc[1],
            color="gold",
            marker="*",
            s=250,
            label="Charger",
            edgecolors="black",
        )

        if self.flying_trajectory and len(self.flying_trajectory) > 1:
            traj = np.array(self.flying_trajectory)
            ax_trajectory.plot(
                traj[:, 0],
                traj[:, 1],
                color="black",
                linestyle="--",
                linewidth=1.5,
                label="Trajectory",
            )
            ax_trajectory.scatter(
                traj[0, 0],
                traj[0, 1],
                color="cyan",
                s=150,
                marker="o",
                label="Start",
                zorder=5,
            )
            ax_trajectory.scatter(
                traj[-1, 0],
                traj[-1, 1],
                color="magenta",
                s=150,
                marker="X",
                label="End",
                zorder=5,
            )

        handles, labels = ax_trajectory.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax_trajectory.legend(by_label.values(), by_label.keys(), loc="upper right")

        # --- 2. 电池电量图 ---
        ax_battery.set_title("UAV Battery Level Over Time")
        if self.uav_battery_progress:
            steps = range(len(self.uav_battery_progress))
            ax_battery.plot(
                steps, self.uav_battery_progress, color="purple", label="Battery (J)"
            )
            ax_battery.set_xlabel("Time Step")
            ax_battery.set_ylabel("Energy (J)")
            ax_battery.set_ylim(0, self.uav.battery_capacity * 1.05)
            ax_battery.grid(True, linestyle="--")

            total_harvested = np.sum(self.charging_energy_progress)
            ax_battery.text(
                0.95,
                0.95,
                f"Harvested: {total_harvested:.2f} J",
                transform=ax_battery.transAxes,
                verticalalignment="top",
                horizontalalignment="right",
                bbox=dict(boxstyle="round,pad=0.5", fc="yellow", alpha=0.5),
            )
            ax_battery.legend()

        # --- 3. 服务分布直方图 ---
        ax_service_hist.set_title("Service Distribution (Fairness)")
        if self.served_ue_history:
            ue_counts = np.bincount(self.served_ue_history, minlength=self.ue_num)
            ue_indices = np.arange(self.ue_num)
            ax_service_hist.bar(
                ue_indices, ue_counts, color="dodgerblue", label="Service Count"
            )
            ax_service_hist.set_xlabel("UE ID")
            ax_service_hist.set_ylabel("Times Served")
            ax_service_hist.set_xticks(ue_indices)
            ax_service_hist.grid(True, axis="y", linestyle="--")
            ax_service_hist.legend()

        # --- 4. 能耗饼图 ---
        ax_energy_pie.set_title("Energy Consumption Breakdown")
        total_flying_energy = np.sum(self.flying_energy_progress)
        total_com_energy = np.sum(self.com_energy_progress)
        total_consumption = total_flying_energy + total_com_energy

        if total_consumption > 0:
            sizes = [total_flying_energy, total_com_energy]
            labels = [
                f"Flying\n({total_flying_energy:.1f} J)",
                f"Computing\n({total_com_energy:.1f} J)",
            ]
            colors = ["skyblue", "lightcoral"]
            ax_energy_pie.pie(
                sizes,
                labels=labels,
                autopct="%1.1f%%",
                startangle=90,
                colors=colors,
                wedgeprops={"edgecolor": "white"},
            )
            ax_energy_pie.axis("equal")
        else:
            ax_energy_pie.text(
                0.5,
                0.5,
                "No Energy Consumed",
                ha="center",
                va="center",
                transform=ax_energy_pie.transAxes,
            )
            ax_energy_pie.set_xticks([])
            ax_energy_pie.set_yticks([])

        # --- 5. 奖励分解图 ---
        ax_reward_comp.set_title("Reward Components Over Time")
        if self.reward_components_history:
            steps = range(len(self.reward_components_history))
            rewards = {
                "charge": [
                    d.get("charge_reward", 0) for d in self.reward_components_history
                ],
                "pbrs": [
                    d.get("pbrs_reward", 0) for d in self.reward_components_history
                ],
            }
            penalties = {
                "delay": [
                    d.get("delay_penalty", 0) for d in self.reward_components_history
                ],
                "energy": [
                    d.get("energy_penalty", 0) for d in self.reward_components_history
                ],
                "boundary": [
                    d.get("out_of_border_penalty", 0)
                    for d in self.reward_components_history
                ],
                "static": [
                    d.get("static_penalty", 0) for d in self.reward_components_history
                ],
                "task_fail": [
                    d.get("served_ue_task_penalty", 0)
                    + d.get("local_failure_penalty", 0)
                    for d in self.reward_components_history
                ],
                "fairness": [
                    d.get("fairness_improvement", 0)
                    for d in self.reward_components_history
                ],
            }

            # 绘制正奖励
            bottom_pos = np.zeros(len(steps))
            for label, values in rewards.items():
                ax_reward_comp.bar(
                    steps, values, bottom=bottom_pos, label=f"R: {label}"
                )
                bottom_pos += np.array(values)

            # 绘制负奖励（惩罚）
            bottom_neg = np.zeros(len(steps))
            for label, values in penalties.items():
                ax_reward_comp.bar(
                    steps, values, bottom=bottom_neg, label=f"P: {label}"
                )
                bottom_neg += np.array(values)

            ax_reward_comp.set_xlabel("Time Step")
            ax_reward_comp.set_ylabel("Reward Value")
            ax_reward_comp.axhline(0, color="black", linewidth=0.8)
            ax_reward_comp.legend(fontsize="small", ncol=2)
            ax_reward_comp.grid(True, linestyle="--")

        # --- 6. 无人机速度图 ---
        ax_uav_speed.set_title("UAV Flying Speed Over Time")
        if self.uav_flying_speed_progress:
            steps = range(len(self.uav_flying_speed_progress))
            ax_uav_speed.plot(
                steps,
                self.uav_flying_speed_progress,
                color="darkgreen",
                label="Flying Speed (m/s)",
            )
            ax_uav_speed.set_xlabel("Time Step")
            ax_uav_speed.set_ylabel("Speed (m/s)")
            ax_uav_speed.set_ylim(0, self.uav.max_speed * 1.05)
            ax_uav_speed.grid(True, linestyle="--")
            ax_uav_speed.legend()
        else:
            ax_uav_speed.text(
                0.5,
                0.5,
                "No Speed Data",
                ha="center",
                va="center",
                transform=ax_uav_speed.transAxes,
            )

        # --- 6. 新增：决策图 (卸载率 vs. 电池) ---
        ax_decision.set_title("Decision: Offloading vs. Battery")
        if self.battery_at_decision and self.offloading_ratio_at_decision:
            # 使用颜色映射来表示时间/步骤的推移
            colors = np.arange(len(self.battery_at_decision))
            scatter = ax_decision.scatter(
                self.battery_at_decision,
                self.offloading_ratio_at_decision,
                c=colors,
                cmap="viridis",
                alpha=0.7,
                edgecolors="w",
                linewidth=0.5,
            )
            ax_decision.set_xlabel("Battery Ratio at Decision")
            ax_decision.set_ylabel("Offloading Ratio")
            ax_decision.set_xlim(0, 1.05)
            ax_decision.set_ylim(-0.05, 1.05)
            ax_decision.grid(True, linestyle="--")
            # 添加颜色条
            cbar = fig.colorbar(scatter, ax=ax_decision, orientation="vertical")
            cbar.set_label("Time Step")
        else:
            ax_decision.text(
                0.5,
                0.5,
                "No Decision Data",
                ha="center",
                va="center",
                transform=ax_decision.transAxes,
            )

        # --- 最终统计信息打印 ---
        print("\n" + "=" * 50)
        print("EPISODE FINISHED - FINAL STATISTICS")
        print(f"  - Total Steps: {self.current_step}")
        print(f"  - Task Success Rate: {success_rate:.2%}")
        if self.total_tasks > 0:
            print(
                f"  - Average Delay per Task: {self.total_episode_delay / self.total_tasks:.3f} s"
            )
            print(
                f"  - Average Energy per Task: {total_consumption / self.total_tasks:.3f} J"
            )
        print(f"  - Final Fairness Index: {self.previous_fairness_index:.4f}")
        print(f"  - Final Battery: {self.uav.e_battery:.2f} J")
        print("=" * 50 + "\n")

        plt.show(block=True)
        plt.close(fig)

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

    def checkfailure(self):
        """检查回合是否应终止。"""
        # 失败条件：无人机电量耗尽
        if self.uav.e_battery <= 0:
            # logging.warning("UAV battery depleted. Episode terminated.")
            return True

        # # --- 优化点: 使用 any() 替代显式循环，更高效、更Pythonic ---
        # # 检查是否还有任何一个节点任务队列不为空
        # if any(node.task_queue for node in self.nodeList):
        #     return False  # 只要有一个不为空，就继续

        # logging.info("All tasks completed. Episode finished successfully.")
        return False  # 所有队列都为空，回合结束

    def calculate_fairness(self):
        """计算当前时刻的Jain公平性指数"""
        service_counts = np.array(self.service_counts) + 1e-6  # 避免除零
        numerator = np.sum(service_counts) ** 2
        denominator = self.ue_num * np.sum(service_counts**2)
        return numerator / denominator


# 注册环境
# register(
#     id="myCustomEnv-v0",
#     entry_point="entities.custom_env:CustomEnv",  # 注意路径修正
#     kwargs={"env_name": "CartPole-v1"},  # 改为 Gymnasium 内置环境
# )
# register(
#     id="myCustomEnv-v0",
#     entry_point="entities.custom_env:CustomEnv",  # 关键修改点
#     kwargs={"max_step": 40},  # 与你的 __init__ 参数一致
# )
# 测试
if __name__ == "__main__":
    # env = gym.make("myCustomEnv-v0")
    # env = gym.make("myCustomEnv-v0", max_step=40)
    env = CustomEnv()
    env.reset()
    # check_env(env)
