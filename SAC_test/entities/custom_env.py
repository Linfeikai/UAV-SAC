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

# 文件处理
import csv  # <-- 新增：导入CSV库用于文件写入
import time
import os


class CustomEnv(gym.Env):
    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 30,
    }

    # --- 1. 配置与代码分离：将所有可调参数和常量集中管理 ---
    class Config:
        # 权重
        W_DELAY = 3.0  # 延迟惩罚的相对重要性
        W_ENERGY_CONSUMED = 0.5  # 能耗惩罚的相对重要性
        W_ENERGY_GAINED = 1.0  # 充电奖励的相对重要性
        W_FAIRNESS = 10  # 公平性奖励的相对重要性 (次要目标)
        W_PBRS = 0.05  # PBRS引导奖励的相对重要性 (引导项)
        REWARD_WEIGHT_LOW_BATTERY_PENALTY = 3.0  # 低电量惩罚的相对重要性
        W_RESCUE = 3.0  # 拯救惩罚

        # 绝对惩罚值 (这些值不参与归一化)
        # PENALTY_TASK_FAILURE = -2.0  # 任务失败的惩罚
        PENALTY_OUT_OF_BOUNDS = -1.5  # 出界惩罚
        PENALTY_STATIC = -1.5  # 静止惩罚
        PENALTY_ACTION_SMOOTH = -1.0  # 动作剧烈变化的总惩罚系数
        PENALTY_PER_DROPPED_TASK = -0.1  # 每丢弃一个任务惩罚0.1

        PENALTY_CRASH = -200  # 无人机因没电提前坠毁了

        # --- 动作平滑的内部权重 (推荐两者相加为1) ---
        W_SMOOTH_VELOCITY = 0.5  # 速度变化在平滑惩罚中的权重
        W_SMOOTH_ANGLE = 0.5  # 角度变化在平滑惩罚中的权重

        # 归一化分母 (通过运行基准得到)
        NORM_DELAY = 189  # 经验最大总延迟
        NORM_ENERGY_CONSUMED = 2400  # 经验最大总消耗
        NORM_ENERGY_GAINED = 2300.0  # 经验最大总充电
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

    state_dim = 108  # 状态空间维度 # 距离border，距离充电器，dx,dy,speed,电量+80=86 +20个历史service_count = 106+上一step的速度、方向
    action_dim = 4
    max_action = (-1, 1)

    # communication model parameters
    bandwidth_nums = 0.5
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
                [-np.pi, -1, 0], dtype=np.float32
            ),  # angle, velocity_ratio, offloading_ratio
            high=np.array([np.pi, 1, 1], dtype=np.float32),
            dtype=np.float32,
        )

        # 组合成一个混合动作空间
        self.action_space = spaces.Tuple(
            (self.discrete_action_space, self.continuous_action_space)
        )

        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.state_dim,), dtype=np.float32
        )

        self.nodeList: List[UENode] = []
        # --- 关键优化：将节点属性向量化，存储在NumPy数组中 ---
        # 这使得我们可以进行高效的批量计算，避免在s   tep中进行Python循环。
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
            self.weighted_reward_components_history = []
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
            self.weighted_reward_components_history = None

            self.battery_at_decision = None
            self.offloading_ratio_at_decision = None

        self.uav = UAVNode()
        self.charger = LaserCharger()
        # 需要频繁地向数组中添加元素，我们先将 self.state 作为列表处理，最后再转换为 NumPy 数组。
        # 这种方法在添加元素时效率更高，因为 NumPy 数组的大小是固定的，而列表的大小是动态的。
        self.service_history = deque(maxlen=5)  # 记录最近5次服务对象
        self.service_counts = np.zeros(self.ue_num, dtype=np.int32)  # 用于计算公平性
        # 追踪每个UE从上一次被服务到现在的步数
        self.steps_since_last_service = np.zeros(self.ue_num, dtype=np.int32)

        self.current_step = 0  # 当前步数
        self.current_time = 0.0  # 当前时间

        self.previous_fairness_index = 1.0  # 初始化公平性指数
        self.last_velocity = 0  # 记录上一步的action 这是为了能够速度平滑
        self.last_continuous_flight_action = np.zeros(2, dtype=np.float32)

        self.ep_arrived = 0
        self.ep_completed = 0
        self.ep_dropped = 0

        self.newly_dropped_in_last_step = 0

        # # =====================================================================
        # # [临时调试代码] 用于跟踪整个运行期间遇到的最大值
        # # =====================================================================
        # self.max_delay_seen_for_debug = 0.0
        # self.max_energy_consumption_debug = 0.0
        # self.max_energy_charged_debug = 0.0
        # # =====================================================================

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.fig, self.ax = None, None  # Matplotlib 图形对象

    def _decode_action(self, action):
        # 其实这里可能也许大概应该还要判断范围一下
        ue_id, continuous_parts = action
        angle = float(continuous_parts[0])
        velocity_ratio = float(continuous_parts[1])
        offloading_ratio = float(continuous_parts[2])
        return ue_id, velocity_ratio, angle, offloading_ratio

    def _record_decision_data(self, battery_before_step, offloading_ratio, ue_id):
        if self.render_mode == "human":
            self.battery_at_decision.append(
                battery_before_step / self.uav.battery_capacity  # 本action时的电量水平
            )
            self.offloading_ratio_at_decision.append(
                offloading_ratio
            )  # 本action做出的卸载比率
            for node in self.nodeList:
                self.total_tasks += len(node.task_queue)  # 本step的总任务数量
            self.served_ue_history.append(ue_id)  # 本step被服务的幸运ue
        self.service_counts[ue_id] += 1  # 不管是否human 我们都需要对被服务的ue+1

    def _update_uav_flight(
        self, ue_id: int, angle: float, velocity_ratio: float
    ) -> dict:
        """处理无人机飞行，返回飞行结果和相关惩罚。
        飞行能耗（我们希望最小化）；飞行奖励（靠近ue的奖励）；越界惩罚；静止惩罚；悬停时间。
        """
        initial_velocity = self.uav.flying_speed
        # 729修改：把速度比率重定义为减速，定速，油门，
        delta_v = self.uav.max_acceleration * velocity_ratio * self.t_fly
        final_velocity = self.uav.flying_speed + delta_v
        new_velocity = np.clip(final_velocity, 0, self.uav.max_speed)
        # 【新增】计算整个过程的平均速度
        average_velocity = (initial_velocity + new_velocity) / 2.0

        # 距离现在基于平均速度计算
        target_fly_distance = average_velocity * self.t_fly

        new_x = self.uav.loc[0] + target_fly_distance * math.cos(angle)
        new_y = self.uav.loc[1] + target_fly_distance * math.sin(angle)

        # 检查是否越界
        out_of_border_penalty = 0
        if not (0 <= new_x <= self.ground_width and 0 <= new_y <= self.ground_height):
            out_of_border_penalty = self.Config.PENALTY_OUT_OF_BOUNDS
            # 停留在边界处
            new_x = np.clip(new_x, 0, self.ground_width)
            new_y = np.clip(new_y, 0, self.ground_height)
            target_fly_distance = np.sqrt(
                (new_x - self.uav.loc[0]) ** 2 + (new_y - self.uav.loc[1]) ** 2
            )
            angle = math.atan2(new_y - self.uav.loc[1], new_x - self.uav.loc[0])
            # 因为撞墙，速度应该衰减为0
            new_velocity = 0.0

        # 3. 调用 uav.moveto()，它会处理电量问题
        #    moveto 现在返回 (actual_distance, energy, pbrs_reward)
        (actual_fly_distance, flying_energy, pbrs_reward) = self.uav.moveto(
            target_fly_distance, angle, self.nodeList[ue_id].loc, average_velocity
        )
        # 4. 根据实际飞行结果更新无人机速度
        #    如果实际飞行距离小于目标距离，说明电量耗尽或撞墙，速度降为0
        if (
            actual_fly_distance < target_fly_distance - 1e-6
        ):  # 减去一个很小的数以应对浮点误差
            self.uav.flying_speed = 0.0
        else:
            self.uav.flying_speed = new_velocity

        # 检查是否静止
        # --- 改进：使用平滑函数计算静止惩罚 ---
        # 当 dis_fly 接近0时，惩罚接近 PENALTY_STATIC；当 dis_fly 增大时，惩罚迅速减小
        # k值越小，惩罚衰减越快。k=2.5 意味着在5米处惩罚约为最大值的13%
        static_penalty = self.Config.PENALTY_STATIC * math.exp(
            -actual_fly_distance / 2.5
        )

        if self.flying_trajectory is not None:
            self.flying_trajectory.append(self.uav.loc.copy())

        return {
            "flying_energy": flying_energy,
            "flying_distance": pbrs_reward,
            "out_of_border_penalty": out_of_border_penalty,
            "static_penalty": static_penalty,
            "hover_time": self.delta_t - self.t_fly,
        }

    def _perform_flight(self, ue_id, velocity_ratio, angle):
        flight_info = self._update_uav_flight(ue_id, angle, velocity_ratio)
        terminated = self.uav.e_battery <= 1e-6
        return flight_info, terminated

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
        # 0. 如果节点没有任务，直接返回
        if not served_ue.task_queue:
            return (
                0,
                0,
                self.current_time,
                0,
                0,
                0,
            )  # (delay, energy, ..., num_processed)
            # 1. 任务分配 (不变)
        original_task_list = list(served_ue.task_queue)  # 创建一个副本用于操作
        local_task_num = int(len(original_task_list) * (1 - offloading_ratio))
        local_tasks_to_process = original_task_list[:local_task_num]
        uav_tasks_to_process = original_task_list[local_task_num:]
        hover_time = max(0.0, hover_time)

        # 计算当前结点任务的紧急程度，即如果他是完全本地卸载，能否在当前step内完成
        T_local_only = (
            sum(task.data_size * task.workload_intensity for task in original_task_list)
            / served_ue.local_capacity
        )
        urgency_ratio = max(T_local_only - hover_time, 0.0) / hover_time

        # 2. 执行计算，并获取【成功处理的任务列表】
        #    a. 本地计算
        total_local_delay, after_local_time, processed_by_local = (
            served_ue.partial_offloading(
                local_tasks_to_process, self.current_time, hover_time
            )
        )

        #    b. 无人机计算部分
        # offload_data_size = sum(task.data_size for task in uav_tasks_to_process)

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

        # 这里增加uav要处理的任务逻辑 让“要传的 + 要算的”永远不超过 7 秒。
        tx_bits, cmp_cycles = 0, 0
        selected = []
        for task in uav_tasks_to_process:
            bi = task.data_size
            wi = task.workload_intensity
            next_tx = (tx_bits + bi) / trans_rate
            next_cmp = (cmp_cycles + bi * wi) / self.uav.f_uav
            if next_tx + next_cmp <= hover_time:
                selected.append(task)
                tx_bits += bi
                cmp_cycles += bi * wi
            else:
                break

        uav_tasks_to_process = selected
        t_tr = tx_bits / trans_rate
        t_tr = min(t_tr, hover_time)
        compute_time = max(hover_time - t_tr, 0)

        # t_tr = offload_data_size / trans_rate if trans_rate > 0 else float("inf")

        # UAV 计算
        total_uav_delay, uav_consumed_energy, after_uav_time, processed_by_uav = (
            self.uav.offload(
                uav_tasks_to_process, self.current_time + t_tr, compute_time, t_tr
            )
        )
        # 3. 合并所有成功处理的任务
        all_processed_tasks = processed_by_local + processed_by_uav
        num_tasks_processed = len(all_processed_tasks)
        self.ep_completed += num_tasks_processed

        # 4. 【核心】从原始任务队列中精确移除已处理的任务
        #    这是一个比较低效的操作，但逻辑清晰。更好的方法是用集合操作。
        new_task_queue = deque(
            [task for task in served_ue.task_queue if task not in all_processed_tasks]
        )
        served_ue.task_queue = new_task_queue
        # 这是一个bool 看这个被服务的ue是否把所有任务都计算完了？
        served_queue_cleared = len(served_ue.task_queue) == 0

        # 5. 【核心】更新缓存大小
        #    我们直接根据新的队列重新计算缓存大小，这是最稳健的方法
        served_ue.current_cache_size = sum(
            task.data_size for task in served_ue.task_queue
        )
        # # 6. 检查是否仍然有未处理的任务（即任务失败）
        # is_task_failed = len(served_ue.task_queue) > 0
        if self.render_mode == "human":
            self.failed_tasks += len(served_ue.task_queue)
        # served_ue_task_penalty = 0.0
        # if is_task_failed:
        #     served_ue_task_penalty = self.Config.PENALTY_TASK_FAILURE

        # 7. 计算总延迟（时间差）和完成时间点
        #    总延迟 = 本地处理的任务的总延迟 + 无人机处理的任务的总延迟
        total_system_delay_for_ue = total_local_delay + total_uav_delay
        uav_com_duration = after_uav_time - self.current_time

        completion_time = max(after_local_time, after_uav_time)

        return (
            total_system_delay_for_ue,  # 被服务ue的总时延
            uav_consumed_energy,  # uav在hover阶段的耗能
            completion_time,  # 这个ue处理完之后的时间点
            uav_com_duration,  # uav计算的过程用了多长时间
            num_tasks_processed,  # 计算完成了多少个任务
            t_tr,  # 传输时延
            urgency_ratio,  # 新增
            1.0 if served_queue_cleared else 0.0,  # 新增
        )

    def _process_served_ue(self, ue_id, offloading_ratio, hover_time):
        served_ue = self.nodeList[ue_id]
        # 这里的completion time指的是当这个被服务的ue完成任务时（本地+uav）到了什么时候
        # after_calc_time指的是uav服务完后的时间，用来计算还有多久可以用来充电
        (
            delay,
            consumed_energy,
            completion_time,
            uav_com_duration,
            num_tasks_processed,
            t_tr,
            urgency_ratio,
            cleared_flag,
        ) = self.com_delay(served_ue, offloading_ratio, hover_time)

        # 充电
        charge_time = hover_time - uav_com_duration
        harvested_energy = (
            self.charger.charge(self.uav, charge_time) if charge_time > 0 else 0.0
        )

        return {
            "delay": delay,
            "consumed_energy": consumed_energy,
            "harvested_energy": harvested_energy,
            "num_tasks_processed": num_tasks_processed,  # <--【关键】把这个信息也返回上去！
            "num_local_failures": 0,  # 新增，避免 KeyError
            "t_tr": t_tr,
            "urgency_ratio": urgency_ratio,
            "cleared_flag": cleared_flag,
        }

    def _process_unserved_ues(self, served_ue_id, delay_info):
        for node in self.nodeList:
            if node is self.nodeList[served_ue_id]:
                continue
            local_delay, local_completion_time, cnt = node.local_offloading(
                self.current_time, self.t_com
            )
            self.ep_completed += cnt
            if len(node.task_queue) > 0:
                # for task in node.task_queue:
                #     task.status = 3
                if self.render_mode == "human":
                    self.failed_tasks += len(node.task_queue)
                # node.task_queue.clear()
                # node.current_cache_size = 0

            delay_info["delay"] += local_delay

    def _calculate_system_delay(
        self, ue_id: int, offloading_ratio: float, hover_time: float
    ) -> dict:
        """计算整个系统的延迟，并检查任务是否失败。"""
        delay_info = self._process_served_ue(ue_id, offloading_ratio, hover_time)
        # 这里把delay_info传入 这个函数内部直接会对delay_info进行修改
        self._process_unserved_ues(ue_id, delay_info)
        return delay_info

    def _calculate_or_skip_delay(self, terminated, ue_id, offloading_ratio, hover_time):
        if terminated:
            self.current_time += self.t_fly
            return {
                "delay": 0.0,
                "consumed_energy": 0.0,
                "harvested_energy": 0.0,
                "num_tasks_processed": 0,
            }
        delay_info = self._calculate_system_delay(ue_id, offloading_ratio, hover_time)
        self.current_time += self.delta_t
        return delay_info

    def calculate_fairness(self):
        """计算当前时刻的Jain公平性指数"""
        if np.sum(self.service_counts) == 0:
            return 0.0  # 或者 1.0/self.ue_num

        service_counts = np.array(self.service_counts) + 1e-6  # 避免除零
        numerator = np.sum(service_counts) ** 2
        denominator = self.ue_num * np.sum(service_counts**2)
        return numerator / denominator

    def _update_fairness(self):
        current_fairness_index = self.calculate_fairness()
        improvement = current_fairness_index - self.previous_fairness_index
        self.previous_fairness_index = current_fairness_index
        return improvement

    def _update_system_state(self):
        """更新充电和任务生成等系统状态。"""
        # 记录当前时间点的电池电量（可能刚刚被充过电）
        if self.uav_battery_progress is not None:
            self.uav_battery_progress.append(self.uav.e_battery)

        # 所有节点生成新任务
        dropped_before = self.ep_dropped
        for node in self.nodeList:
            added, dropped = node.generateTask(self.current_time)
            self.ep_arrived += added
            self.ep_dropped += dropped
            # 记录下这一轮新丢弃了多少任务

        self.newly_dropped_in_last_step = self.ep_dropped - dropped_before

    def _handle_no_task_penalty(self, ue_id):
        if self.nodeList[ue_id].current_cache_size == 0:
            self.find_other_nodes(self.nodeList[ue_id])
            return self.Config.PENALTY_NO_TASK_NODE
        return 0

    def _record_training_metrics(
        self, flight_info, delay_info, reward_components, weighted_reward_components
    ):
        if self.render_mode == "human":
            self.flying_energy_progress.append(flight_info.get("flying_energy", 0))
            self.com_energy_progress.append(delay_info.get("consumed_energy", 0))
            self.charging_energy_progress.append(delay_info.get("harvested_energy", 0))
            self.uav_flying_speed_progress.append(self.uav.flying_speed)
            self.reward_components_history.append(reward_components)
            self.weighted_reward_components_history.append(weighted_reward_components)
            self.total_episode_delay += delay_info.get("delay", 0.0)

    def step(
        self, action: Tuple[int, np.ndarray]
    ) -> Tuple[np.ndarray, float, bool, bool, dict]:
        self.current_step += 1  # 增加当前步数
        # 记录当前电池状态 为了与本step飞行后做对照
        battery_before_step = self.uav.e_battery

        # 1. 解码动作
        ue_id, velocity_ratio, angle, offloading_ratio = self._decode_action(action)

        # 2.如果是渲染模式，则收集决策数据 ;并收集当前任务数量
        self._record_decision_data(battery_before_step, offloading_ratio, ue_id)

        # 3. 检查选择的UE是否有任务，并给予惩罚 目前不会有这个情况
        no_task_penalty = self._handle_no_task_penalty(ue_id)

        # 4. 更新无人机飞行状态
        flight_info, terminated = self._perform_flight(ue_id, velocity_ratio, angle)

        # 5 进行计算任务！
        delay_info = self._calculate_or_skip_delay(
            terminated, ue_id, offloading_ratio, flight_info["hover_time"]
        )

        # 6. 公平性计算
        fairness_improvement = self._update_fairness()

        # 7. 计算最终奖励 (在检查终止条件之前)
        reward, reward_components, weighted_reward_components = self._compute_reward(
            flight_info,
            delay_info,
            fairness_improvement,
            np.array([velocity_ratio, angle], dtype=np.float32),
        )
        # 8.看是否终止？
        if not terminated:
            terminated = self.checkfailure()  # 检查是否还有其他原因导致终止
        truncated = False  # 假设我们没有因为步数限制而截断
        if terminated and self.uav.e_battery <= 1e-6:
            crash_penalty = self.Config.PENALTY_CRASH * (
                (self.slot_num - self.current_step + 1) / self.slot_num
            )
            reward += crash_penalty
            # <-- 新增：将坠机惩罚也记录到最终的奖励项中 -->
            if self.render_mode == "human":
                reward_components["crash_penalty"] = crash_penalty
                weighted_reward_components["crash_penalty"] = crash_penalty

        # 9. 为下一个状态生成新任务 (只有在回合未终止时才进行)
        if not terminated:
            self._update_system_state()

        # 更新上一步的动作记录
        self.last_continuous_flight_action = np.array(
            [velocity_ratio, angle], dtype=np.float32
        )

        # 11. 组装并返回最终结果
        info = {
            **reward_components,  # 包含所有奖励组件
            **flight_info,
            **delay_info,
            "fairness_improvement": fairness_improvement,
        }
        self._record_training_metrics(
            flight_info, delay_info, reward_components, weighted_reward_components
        )

        # 现在的问题是这是一个episode级别的metrics.
        # 要解决的问题是：1.如何在训练过程中监控？2.render那里还没加上，应该加上的。
        # 现在只在最后evaluate policy的时候计算 也就是render那里才有
        # backlog = sum(len(n.task_queue) for n in self.nodeList)
        # A = max(self.ep_arrived, 1)
        # info.update(
        #     {
        #         "completion_rate": self.ep_completed / A,
        #         "drop_rate": self.ep_dropped / A,
        #         "backlog_ratio": backlog / A,
        #     }
        # )

        return self._get_obs(), reward, terminated, truncated, info

    def _calculate_reward_components(
        self, flight_info, delay_info, fairness_improvement, current_action
    ):
        c = {}

        # 原始值
        raw_delay = delay_info.get("delay", 0.0)
        raw_flying_energy = flight_info.get("flying_energy", 0.0)
        raw_flying_distance = flight_info.get("flying_distance", 0.0)
        raw_computing_energy = delay_info.get("consumed_energy", 0.0)
        raw_total_consumption = raw_flying_energy + raw_computing_energy
        raw_harvested_energy = delay_info.get("harvested_energy", 0.0)
        battery_ratio = self.uav.e_battery / self.uav.battery_capacity

        # 归一化
        c["delay_penalty"] = -raw_delay / self.Config.NORM_DELAY
        c["energy_penalty"] = -raw_total_consumption / self.Config.NORM_ENERGY_CONSUMED
        # 充电奖励除了归一化，还要衰减，电量越高，奖励越少
        c["charge_reward"] = (raw_harvested_energy / self.Config.NORM_ENERGY_GAINED) * (
            1.0 - battery_ratio
        )
        # 低电量惩罚
        c["low_battery_penalty"] = -1 / (1 + math.exp((battery_ratio - 0.4) * 15))
        # 引导飞行奖励
        c["pbrs_reward"] = raw_flying_distance / self.Config.NORM_FLYING_DISTANCE
        c["fairness_improvement"] = fairness_improvement

        # 出界惩罚
        c["out_of_border_penalty"] = flight_info.get("out_of_border_penalty", 0.0)
        # 静止惩罚
        c["static_penalty"] = flight_info.get("static_penalty", 0.0)

        # 动作平滑惩罚
        last_vr, last_angle = self.last_continuous_flight_action
        vr_change = (current_action[0] - last_vr) / 2.0
        angle_change = (
            math.atan2(
                math.sin(current_action[1] - last_angle),
                math.cos(current_action[1] - last_angle),
            )
            / np.pi
        )
        # 这里的二次方有两个作用：1.保证为正数；2.放大剧烈惩罚，激励Agent做出微小、平滑的调整
        smooth_cost = self.Config.W_SMOOTH_VELOCITY * (
            vr_change**2
        ) + self.Config.W_SMOOTH_ANGLE * (angle_change**2)
        c["action_smooth_penalty"] = self.Config.PENALTY_ACTION_SMOOTH * smooth_cost

        # 救急奖励
        rescue = 0.0
        if (
            delay_info.get("cleared_flag", 0.0) >= 0.5
            and delay_info.get("urgency_ratio", 0.0) > 0.0
        ):
            rescue = delay_info["urgency_ratio"]
        c["rescue_reward"] = rescue

        # 丢弃任务的惩罚
        c["drop_penalty"] = (
            self.newly_dropped_in_last_step * self.Config.PENALTY_PER_DROPPED_TASK
        )

        return c

    def _apply_reward_weights(self, c):
        weighted_c = {}
        reward = 0.0

        weighted_c["delay_penalty"] = self.Config.W_DELAY * c["delay_penalty"]
        weighted_c["energy_penalty"] = (
            self.Config.W_ENERGY_CONSUMED * c["energy_penalty"]
        )
        weighted_c["low_battery_penalty"] = (
            self.Config.REWARD_WEIGHT_LOW_BATTERY_PENALTY * c["low_battery_penalty"]
        )
        weighted_c["charge_reward"] = self.Config.W_ENERGY_GAINED * c["charge_reward"]
        weighted_c["pbrs_reward"] = self.Config.W_PBRS * c["pbrs_reward"]
        weighted_c["fairness_improvement"] = (
            self.Config.W_FAIRNESS * c["fairness_improvement"]
        )
        weighted_c["rescue_reward"] = self.Config.W_RESCUE * c["rescue_reward"]

        # 将所有加权项相加
        for key in weighted_c:
            reward += weighted_c[key]

        # 绝对惩罚项直接相加，并记录在加权字典中（权重为1）
        weighted_c["out_of_border_penalty"] = c["out_of_border_penalty"]
        weighted_c["static_penalty"] = c["static_penalty"]
        weighted_c["action_smooth_penalty"] = c["action_smooth_penalty"]
        weighted_c["drop_penalty"] = c["drop_penalty"]

        reward += (
            c["out_of_border_penalty"]
            + c["static_penalty"]
            + c["action_smooth_penalty"]
            + c["drop_penalty"]
        )
        return reward, weighted_c

    def _compute_reward(
        self,
        flight_info: dict,
        delay_info: dict,
        fairness_improvement: float,
        current_action: np.ndarray,
    ) -> Tuple[float, dict, dict]:
        """根据所有组件计算最终的奖励值。"""
        components = self._calculate_reward_components(
            flight_info, delay_info, fairness_improvement, current_action
        )
        # 接收加权后的字典
        reward, weighted_components = self._apply_reward_weights(components)
        return float(reward), components, weighted_components

        # # no_Task_penalty 没有加入，是因为当前env设置下所有node会重新生成任务。

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
        # 5.把服务信息进行归一化，也列入到状态中service_counts
        state.extend(self.service_counts / self.slot_num)
        last_action_normalized = [
            self.last_continuous_flight_action[0],  # 速度比率已经是 [-1, 1]
            self.last_continuous_flight_action[1]
            / np.pi,  # 将角度从 [-pi, pi] 映射到 [-1, 1]
        ]
        state.extend(last_action_normalized)

        assert len(state) == self.state_dim, f"State length mismatch: {len(state)}"
        return np.array(state, dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.uav.reset()  # 重置无人机状态
        self.service_counts.fill(0)  # 重置服务计数
        self.previous_fairness_index = 1.0  # 确保每次新回合开始时，都重置公平性历史
        self.last_velocity = 0
        self.last_continuous_flight_action.fill(0)  # 重置时清零
        self.ep_arrived = self.ep_completed = self.ep_dropped = 0
        self.newly_dropped_in_last_step = 0

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
            self.uav_flying_speed_progress.clear()
            self.weighted_reward_components_history.clear()

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
        # success_rate = (
        #     (1 - self.failed_tasks / self.total_tasks) if self.total_tasks > 0 else 1.0
        # )
        backlog = sum(len(n.task_queue) for n in self.nodeList)
        A = max(self.ep_arrived, 1)
        completion_rate = self.ep_completed / A
        drop_rate = self.ep_dropped / A
        backlog_rate = backlog / A

        fig.suptitle(
            f"Episode Summary (Steps: {self.current_step}) | "
            f"completion_rate: {completion_rate:.2%} | "
            f"drop_rate: {drop_rate:.2%} | "
            f"backlog_rate: {backlog_rate:.2%} | "
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
                "rescue_reward": [
                    d.get("rescue_reward", 0) for d in self.reward_components_history
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
                "task_drop": [
                    d.get("drop_penalty", 0) for d in self.reward_components_history
                ],
                "low_battery": [
                    d.get("low_battery_penalty", 0)
                    for d in self.reward_components_history
                ],
                "action_smooth": [
                    d.get("action_smooth_penalty", 0)
                    for d in self.reward_components_history
                ],
            }
            # --- 第2步：动态分离 pbrs_reward 和 fairness_improvement ---
            # 为每个正负不定的参数准备正、负两个列表
            pbrs_pos = []
            pbrs_neg = []
            fairness_pos = []
            fairness_neg = []
            # 遍历一次历史记录，同时处理所有正负不定的参数
            for d in self.reward_components_history:
                # 处理 pbrs_reward
                pbrs_value = d.get("pbrs_reward", 0)
                if pbrs_value > 0:
                    pbrs_pos.append(pbrs_value)
                    pbrs_neg.append(0)
                else:
                    pbrs_pos.append(0)
                    pbrs_neg.append(pbrs_value)

                # 处理 fairness_improvement
                fairness_value = d.get("fairness_improvement", 0)
                if fairness_value > 0:
                    fairness_pos.append(fairness_value)
                    fairness_neg.append(0)
                else:
                    fairness_pos.append(0)
                    fairness_neg.append(fairness_value)
                # --- 第3步：将分离后的数据添加回主字典 ---

            # 将正值部分添加到 rewards 字典
            rewards["pbrs (pos)"] = pbrs_pos
            rewards["fairness (pos)"] = fairness_pos

            # 将负值部分添加到 penalties 字典
            penalties["pbrs (neg)"] = pbrs_neg
            penalties["fairness (neg)"] = fairness_neg

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
        print(f"  - Task completion Rate: {completion_rate:.2%}")
        print(f"  - Task drop Rate: {drop_rate:.2%}")
        print(f"  - Task backlog Rate: {backlog_rate:.2%}")

        if self.total_tasks > 0:
            print(
                f"  - Average Delay per Task: {self.total_episode_delay / self.ep_completed:.3f} s"
            )
            print(
                f"  - Average Energy per Task: {total_consumption / self.ep_completed:.3f} J"
            )
        print(f"  - Final Fairness Index: {self.previous_fairness_index:.4f}")
        print(f"  - Final Battery: {self.uav.e_battery:.2f} J")
        print("=" * 50 + "\n")

        # =====================================================================
        # <-- 新增: 将奖励数据写入CSV文件的核心逻辑 -->
        # =====================================================================
        def write_reward_table_to_csv(file_prefix, data, timestamp):
            """一个辅助函数，用于将奖励历史记录写入CSV文件。"""
            if not data:
                print(f"\nINFO: {file_prefix} - No reward data to save.")
                return

            # 从数据中动态获取所有可能的列名
            all_keys = set()
            for item in data:
                all_keys.update(item.keys())

            headers = sorted(list(all_keys))  # 排序以保证列的顺序
            headers.insert(0, "Step")  # 将Step作为第一列
            filename = f"{file_prefix}_{timestamp}.csv"

            try:
                with open(filename, "w", newline="", encoding="utf-8") as csvfile:
                    writer = csv.DictWriter(csvfile, fieldnames=headers)
                    writer.writeheader()
                    for i, row_data in enumerate(data):
                        # 创建一个包含 'Step' 和所有奖励数据的完整行
                        row_with_step = {"Step": i}
                        row_with_step.update(row_data)
                        writer.writerow(row_with_step)
                print(f"INFO: Reward data successfully saved to '{filename}'")
            except IOError as e:
                print(f"ERROR: Could not save reward data to '{filename}': {e}")

        current_time = int(time.time())

        # 当 episode 结束时，调用函数写入两个CSV文件
        if self.render_mode == "human":
            write_reward_table_to_csv(
                "unweighted_rewards", self.reward_components_history, current_time
            )
            write_reward_table_to_csv(
                "weighted_rewards",
                self.weighted_reward_components_history,
                current_time,
            )

        current_time = int(time.time())
        plt.show(block=True)
        plt.savefig(f"{current_time}")
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
