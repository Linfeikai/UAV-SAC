import sys
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
import math
import random

# 将项目根目录添加到 Python 路径
sys.path.append(str(Path(__file__).parent.parent))  # 上两级目录（到 my_project/）

from entities.custom_env import CustomEnv

env = CustomEnv()
env.reset(seed=42)  # 重置环境，设置随机种子


def weighted_geometric_median(points, weights):
    """
    计算加权几何中位数
    :param points: 二维坐标列表，形状 (n, 2)
    :param weights: 权重列表，形状 (n,)
    :return: 最优坐标 (x, y)
    """
    points = np.array(points)
    weights = np.array(weights)
    initial_guess = np.average(points, axis=0, weights=weights)  # 加权质心作为初始值

    def objective(z):
        return np.sum(weights * cdist([z], points)[0])

    result = minimize(objective, x0=initial_guess, method="L-BFGS-B")
    return result.x


def sort_nodes_by_distance(node_list, reference_point):
    """
    Sorts a list of 2D nodes based on their distance from a reference point,
    with the furthest nodes appearing first.

    Args:
        node_list (list of lists): A list where each inner list is [x, y]
                                    representing the coordinates of a node.
        reference_point (tuple): A tuple (ref_x, ref_y) representing the
                                 coordinates of the reference point.

    Returns:
        list: A list of indices, where the index corresponds to the original
              position of the node in the `node_list`, sorted in descending
              order of their distance from the `reference_point`.
    """
    distances = []
    ref_x, ref_y = reference_point

    for x, y in node_list:
        distance = math.sqrt((x - ref_x) ** 2 + (y - ref_y) ** 2)
        distances.append(distance)

    # Create a list of (distance, original_index) tuples
    indexed_distances = list(enumerate(distances))

    # Sort the list based on distance in descending order
    indexed_distances.sort(key=lambda item: item[1], reverse=True)

    # Extract the original indices from the sorted list
    sorted_indices = [index for index, distance in indexed_distances]

    return sorted_indices


def greedy_alo(env, num_episodes=10):
    """
    Greedily allocate resources to the agent based on the current state of the environment.
    """
    flying_angle = 0
    flying_distance = 0
    offloading_ratio = 0.8  # 卸载比率
    rewards = []  # 用来存储每个episode的奖励
    current_rewards = []  # 用来存储当前episode的奖励
    episode_metrics = {
        "delay": [],
    }
    current_episode_metrics = {
        "delay": [],
    }

    for episode in range(num_episodes):
        env.reset(seed=random.randint(1, 40))  # 重置环境，设置随机种子
        ue_loc_list = []
        for node in env.nodeList:
            ue_loc_list.append(node.loc)  # 获取所有UE节点的坐标
        uav_loc = weighted_geometric_median(ue_loc_list, np.ones(env.ue_num))
        env.uav.loc = uav_loc  # UAV的坐标
        served_nodes_list = sort_nodes_by_distance(ue_loc_list, uav_loc)
        for step in range(env.slot_num):
            served_nodes_index = env.current_step % env.ue_num  # 当前服务的节点
            served_node = served_nodes_list[served_nodes_index]  # 当前服务的节点
            action = np.array(
                [served_node, flying_angle, flying_distance, offloading_ratio]
            )
            # 这里的动作是一个包含4个元素的数组，分别表示服务节点、飞行角度、飞行距离和卸载比率
            obs, reward, terminated, truncated, info = env.step(
                action
            )  # 执行动作并获取反馈

            current_rewards.append(reward)
            current_episode_metrics["delay"].append(info["delay"])
            # 当前episode结束后的处理
            if terminated or truncated or step == env.slot_num - 1:
                rewards.append(
                    np.sum(current_rewards)  # 计算当前episode的平均奖励
                )
                episode_metrics["delay"].append(
                    np.sum(current_episode_metrics["delay"])
                )
                print(
                    f"Episode {episode + 1} finished.  Reward: {np.sum(current_rewards)}, Delay: {np.sum(current_episode_metrics['delay'])}"
                )
                current_rewards.clear()  # 清空当前episode的奖励
                current_episode_metrics["delay"].clear()  # 清空当前episode的延迟

                break
    print(
        f"All episodes finished. Average Reward: {np.mean(rewards)}, Delay: {np.mean(episode_metrics['delay'])}"
    )


if __name__ == "__main__":
    greedy_alo(env, num_episodes=10)
    env.close()
