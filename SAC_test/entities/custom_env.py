# 强化学习核心库
import gymnasium as gym
from gymnasium import spaces
from gymnasium.envs.registration import register
from stable_baselines3 import A2C

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


# 没有对两个东西进行惩罚：静止不动，还有不公平服务
class CustomEnv(gym.Env):
    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 30,
    }

    # 环境参数
    ground_width = 400  # 场地宽度
    ground_height = 400  # 场地高度
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
    t_com = 7
    delta_t = t_fly + t_com  # 1s飞行, 后7s用于悬停计算
    slot_num = int(T / delta_t)  # 40个间隔

    def __init__(
        self, render_mode=None
    ):  # 离散值：选择ue；连续值：距离，方向，卸载比率
        super(CustomEnv, self).__init__()
        # self.action_space = gym.spaces.Dict(
        #     {
        #         "ue": gym.spaces.Discrete(20),
        #         "angle": gym.spaces.Box(-np.pi, np.pi, dtype=np.float32),
        #         "distance": gym.spaces.Box(0, 1, dtype=np.float32),
        #         "offloading_rate": gym.spaces.Box(0, 1, dtype=np.float32),
        #     }
        # )
        # self.action_space = spaces.Box(
        #     low=np.array([0, -np.pi, 0, 0], dtype=np.float32),  # 每个维度的最小值
        #     high=np.array([19, np.pi, 1, 1], dtype=np.float32),  # 每个维度的最大值
        #     dtype=np.float32,  # 数据类型
        # )
        # 先不在参数中传入（为了简化）
        # self.continues_dim = continues_dim  # 连续动作维度
        self.continues_dim = 3  # 角度、距离、卸载率
        # self.continues_low = (
        #     continues_low
        #     if continues_low is not None
        #     else np.array([-1.0, -np.pi, -1.0], dtype=np.float32)
        # )
        self.continues_low = np.array([-1.0, -np.pi, 0.0], dtype=np.float32)  # 最小值

        # self.continues_high = (
        #     continues_high
        #     if continues_high is not None
        #     else np.array([1.0, np.pi, 1.0], dtype=np.float32)
        # )

        self.continues_high = np.array([1.0, np.pi, 1.0], dtype=np.float32)

        self.action_space = spaces.Tuple(
            (
                spaces.Discrete(self.ue_num),  # UE选择
                spaces.Box(
                    low=self.continues_low,
                    high=self.continues_high,
                    shape=(self.continues_dim,),  # 角度、距离、卸载率
                    dtype=np.float32,
                ),
            )
        )

        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(86,), dtype=np.float32
        )

        self.nodeList: List[UENode] = []
        self.flying_trajectory = []  # 飞行轨迹

        self.uav = UAVNode()
        self.charger = LaserCharger()
        # 需要频繁地向数组中添加元素，我们先将 self.state 作为列表处理，最后再转换为 NumPy 数组。
        # 这种方法在添加元素时效率更高，因为 NumPy 数组的大小是固定的，而列表的大小是动态的。
        self.service_history = deque(maxlen=5)  # 记录最近5次服务对象
        self.current_step = 0  # 当前步数
        self.current_time = 0.0  # 当前时间

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode
        self.fig, self.ax = None, None  # Matplotlib 图形对象

    def com_delay(self, served_ue: UENode, offloading_ratio):
        """
        Calculate the delay and energy consumption for a given UE node based on the offloading ratio.

        Args:
            served_ue (UENode): The UE node being served, which contains tasks to be processed.
            offloading_ratio (float): The ratio of tasks to offload to the UAV (0.0 to 1.0).

        Returns:
            tuple: A tuple containing:
                - delay (float): The total delay experienced by the UE node.
                - uav_consumed_energy (float): The energy consumed by the UAV during task processing.
        """
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
        # 现在暂时先用的是p_noisy_los
        trans_rate = self.B * math.log2(
            1 + self.p_uplink * g_uav_ue / self.p_noisy_los
        )  # 上行链路传输速率bps

        # 传输时延
        t_tr = offload_data_size / trans_rate  # 上传时延,1B=8bit
        # uav会调用自己的offload函数，返回delay和耗能,还有处理完后的时间
        t_edge_com, uav_consumed_energy, after_uav_time = self.uav.offload(
            uav_tasks, self.current_time + t_tr
        )

        # Ensure after_local_time and after_uav_time are valid before updating current_time
        if after_local_time is not None and after_uav_time is not None:
            self.current_time = max(after_local_time, after_uav_time)  # 更新现在时间
        else:
            logging.error(
                "Invalid time values encountered in com_delay. Unable to update current_time."
            )

        delay = max(t_local_com, t_tr + t_edge_com)  # 总时延
        # 本地计算+传输+uav计算的总时延(其实这里的计算不太对,感觉怪怪的)

        # ue的所有任务都通过uav+本地卸载完成
        # Ensure all tasks are processed or marked before clearing the queue
        for task in served_ue.task_queue:
            if task.status == 0:  # If a task is unprocessed
                logging.warning(f"Unprocessed task found: {task}")
                task.status = 3  # Mark it as discarded
        served_ue.task_queue.clear()  # Clear the task queue after verification
        served_ue.current_cache_size = 0  # 清空缓存
        return delay, uav_consumed_energy
        # 获得状态

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

    def step(self, action):
        self.current_step += 1  # 增加当前步数

        ue_id = int(np.round(action[0]))  # 离散动作取整
        ue_id = np.clip(ue_id, 0, 19)  # 确保范围有效

        con_action = action[1]  # 连续动作部分
        angle = float(con_action[0])  # 直接使用连续动作
        distance = float(con_action[1])
        offloading_ratio = float(con_action[2])

        terminated = False  # 是否终止
        truncated = False  # 是否被截断（如超时）

        out_of_border_penalty = 0
        action_invalid_penalty = 0
        static_penalty = 0
        find_notask_node_penalty = 0
        offloading_imbalance_penalty = 0

        local_time = 0.0  # 本地计算时间

        ue_id = np.clip(int(ue_id), 0, self.ue_num - 1)  # 确保ue_id在合法范围内
        offloading_ratio = np.clip(float(con_action[2]), 0.0, 1.0)
        if abs(offloading_ratio) < 0.01:
            offloading_ratio = 0.0
        # print(f"offloading_ratio: {offloading_ratio}")
        distance = np.clip(distance, 0.0, 1.0)  # 限制在[0,1]区间

        dis_fly = (
            self.uav.max_speed * distance * self.t_fly
        )  # 最大速度乘当前比率，得到飞行直线距离
        new_x = self.uav.loc[0] + dis_fly * math.cos(angle)
        new_y = self.uav.loc[1] + dis_fly * math.sin(angle)
        self.service_history.append(ue_id)

        #  如果agent生成的飞行指令不合法（超出边界）
        if (
            new_x > self.ground_width
            or new_x < 0
            or new_y > self.ground_height
            or new_y < 0
        ):
            out_of_border_penalty = self.calculate_boundary_penalty([new_x, new_y])
            #     1. 结束episode
            # is_terminal = True
            # reward = -10000
            #     2. 停留在边界处、给予惩罚
            new_x = np.clip(new_x, 0, self.ground_width)
            new_y = np.clip(new_y, 0, self.ground_height)
            dis_fly = np.sqrt(
                (new_x - self.uav.loc[0]) ** 2 + (new_y - self.uav.loc[1]) ** 2
            )

            angle = math.atan2(new_y - self.uav.loc[1], new_x - self.uav.loc[0])
            action_invalid_penalty += -2

        dis_fly = np.clip(dis_fly, 0, 30)
        self.uav.flying_speed = dis_fly

        if dis_fly < 5:
            static_penalty = -1

        # 如果agent找到了没有任务的结点（现在暂时不可能）
        if self.nodeList[ue_id].current_cache_size == 0:
            self.find_other_nodes(self.nodeList[ue_id])
            find_notask_node_penalty = -10
            action_invalid_penalty = -10

        # 第一步：无人机飞行
        flying_energy, flying_reward = self.uav.moveto(
            dis_fly, angle, self.nodeList[ue_id].loc
        )  # 传入新的距离和角度
        self.flying_trajectory.append(self.uav.loc.copy())  # 记录新的飞行轨迹
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

        if self.checkfailure():
            print(f"无人机电量耗尽 在目前{self.current_step}，任务失败")
            terminated = True
            truncated = False

        info = {
            "delay": delay,
            "flying_energy": flying_energy,
            "out_of_border_penalty": out_of_border_penalty,
            "action_invalid_penalty": action_invalid_penalty,
            "flying_reward": flying_reward,
            "terminated": terminated,
        }  # Define an empty dictionary for additional information
        recent_count = list(self.service_history).count(ue_id)

        offloading_imbalance_penalty = -2 * (recent_count - 1)

        # flying_reward已经在uav的moveto里解决了正负，所以这里直接相加。
        reward = (
            -delay
            + out_of_border_penalty
            + action_invalid_penalty
            + flying_reward * 0.25
            + static_penalty
        )
        reward = float(reward)  # 关键修复！确保在返回前转换

        return self._get_obs(), reward, terminated, truncated, info

    def ue_state_process(self):
        # 在这里本来应该用一些网络对state状态进行编码or？?
        raw_data = []
        for node in self.nodeList:
            dx = (node.loc[0] - self.uav.loc[0]) / self.ground_width  # 距离uav的坐标
            dy = (node.loc[1] - self.uav.loc[1]) / self.ground_height  # 距离uav的坐标
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

        state.extend(self.ue_state_process())  # UE节点状态
        assert len(state) == 86, f"State length mismatch: {len(state)}"
        # print("now the state is")
        # print(state)

        # print(st)
        # 5. 转换为NumPy数组并指定dtype
        state_np = np.array([float(x) for x in state], dtype=np.float32)
        return state_np

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.flying_trajectory.clear()  # 清空飞行轨迹
        self.uav.reset()
        self.flying_trajectory.append(self.uav.loc.copy())  # 记录飞行轨迹

        # 重置 UE 节点
        light_num = int(self.ue_num * self.task_type_distribution["normal"])
        medium_num = int(self.ue_num * self.task_type_distribution["moderate"])
        heavy_num = int(self.ue_num * self.task_type_distribution["hpc"])

        # 每个uenode的rng，在reset的时候都重新生成，这样能保证不同 episode 之间的随机性不同
        self.nodeList = (
            [
                UENode(
                    nodetype=Nodetype.NORMAL,
                    rng=np.random.RandomState(self.np_random.integers(1e6)),
                )
                for _ in range(light_num)
            ]
            + [
                UENode(
                    nodetype=Nodetype.MODERATE,
                    rng=np.random.RandomState(self.np_random.integers(1e6)),
                )
                for _ in range(medium_num)
            ]
            + [
                UENode(
                    nodetype=Nodetype.HPC,
                    rng=np.random.RandomState(self.np_random.integers(1e6)),
                )
                for _ in range(heavy_num)
            ]
        )

        # 可视化节点（调试用）
        # self.visualize_nodes()

        self.current_step = 0  # 重置当前步数
        self.current_time = 0.0  # 重置当前时间
        info = {}  # 一定要返回额外信息，我还没添加

        return self._get_obs(), info

    def render(self):
        if self.render_mode == "human":
            plt.ioff()  # 关闭交互模式

            # 创建全新图形
            fig, ax = plt.subplots()
            ax.set_xlim(0, self.ground_width)
            ax.set_ylim(0, self.ground_height)
            ax.set_aspect("equal")
            ax.grid(True)
            ax.set_title(f"Episode {getattr(self, '_current_episode', '')} Trajectory")
            ax.set_xlabel("X Coordinate")
            ax.set_ylabel("Y Coordinate")

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

            # 绘制节点
            for node in self.nodeList:
                color = node_colors[node.nodetype]
                label = node_labels[node.nodetype]
                x, y = node.loc
                ax.scatter(x, y, color=color, label=label, s=100, alpha=0.7)

            # 绘制充电站
            ax.scatter(
                self.charger.loc[0],
                self.charger.loc[1],
                color="black",
                label="Laser Charger",
                s=150,  # 增大标记尺寸
            )

            # 绘制轨迹（确保使用正确的变量名）
            if hasattr(self, "flying_trajectory") and len(self.flying_trajectory) > 1:
                trajectory_x = [point[0] for point in self.flying_trajectory]
                trajectory_y = [point[1] for point in self.flying_trajectory]
                line = ax.plot(
                    trajectory_x,
                    trajectory_y,
                    color="red",  # 改为更醒目的颜色
                    linestyle="-",
                    linewidth=3,
                    label="Flying Trajectory",
                )[0]
                # 添加起点和终点标记
                ax.scatter(
                    trajectory_x[0],
                    trajectory_y[0],
                    color="yellow",
                    s=200,
                    marker="o",
                    label="Start",
                )
                ax.scatter(
                    trajectory_x[-1],
                    trajectory_y[-1],
                    color="blue",
                    s=200,
                    marker="x",
                    label="End",
                )

            # 显示图例（去重）
            handles, labels = ax.get_legend_handles_labels()
            by_label = dict(zip(labels, handles))
            ax.legend(by_label.values(), by_label.keys(), loc="upper right")

            # 阻塞式显示
            plt.show(block=True)

            # 关闭图形释放内存
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
        # 检查无人机电量是否耗尽
        if self.uav.e_battery <= 0:
            logging.warning("UAV battery depleted. Episode terminated.")
            return True
        # 检查所有节点的任务队列是否为空
        for node in self.nodeList:
            if node.task_queue:
                return False


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
