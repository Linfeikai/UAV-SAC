from SAC_test.entities.custom_env import CustomEnv
from custom_components.hybrid_sac_agent import HybridSAC
import gymnasium as gym
from gymnasium.envs.registration import register
from stable_baselines3.common.env_checker import check_env
from stable_baselines3 import SAC, TD3
from stable_baselines3.common.callbacks import BaseCallback, CallbackList
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv

import random
import wandb
from wandb.integration.sb3 import WandbCallback
# from swanlab.integration.sb3 import SwanLabCallback


import pandas as pd  # Optional, but helpful
import os

import numpy as np
import torch
from typing import Dict, List, Optional

from gymnasium.wrappers import TimeLimit
import warnings
import swanlab

swanlab.sync_wandb()
# swanlab.sync_tensorboard_torch()

warnings.filterwarnings("ignore")


# 固定随机种子
SEED = 42


print("Setting random seed for reproducibility...")
register(
    id="UAVEnv-v1",
    entry_point="SAC_test.entities.custom_env:CustomEnv",
    max_episode_steps=40,
)
register(
    id="UAVEnv-v0",
    entry_point="SAC_test.entities.custom_env_sac:CustomEnv",
    max_episode_steps=40,
)

# 1. 定义两个不同的初始学习率
# Actor可以快一点，因为它需要探索。Critic必须稳，所以让它慢得多。
lr_actor_initial = 3e-4  # 保持原来的值
lr_critic_initial = 3e-5  # 降低一个数量级


# 2. 为它们分别创建衰减函数
def linear_schedule(initial_value: float):
    def func(progress_remaining: float) -> float:
        return progress_remaining * initial_value

    return func


lr_actor_schedule = linear_schedule(lr_actor_initial)
lr_critic_schedule = linear_schedule(lr_critic_initial)

# 3.打包进policy_kwargs
policy_kwargs = {
    "lr_actor_schedule": lr_actor_schedule,
    "lr_critic_schedule": lr_critic_schedule,
}

import numpy as np
import wandb
from stable_baselines3.common.callbacks import BaseCallback
from typing import Optional, List, Dict, Any


class ParallelEpisodeMetricCallback(BaseCallback):
    """
    A callback to correctly track and log metrics from multiple parallel environments.
    It logs the mean of metrics for all episodes that finish in a given logging interval.
    """

    def __init__(self, metrics_to_track: Optional[List[str]] = None, verbose: int = 0):
        """
        Args:
            metrics_to_track: List of metric names from `info` to track (e.g., ["delay", "flying_energy"]).
                              If None, all keys in `info` (excluding those starting with '_') will be tracked.
        """
        super().__init__(verbose)
        self.metrics_to_track: Optional[List[str]] = metrics_to_track

        # (# NEW) 记录所有已完成 episode 的总数
        self.total_episodes_finished = 0

        # (# MODIFIED) 为每个并行环境维护一个独立的指标缓冲区
        # 我们将在第一次调用 _on_step 时，根据环境数量来初始化它
        self.env_metric_buffers: Optional[List[Dict[str, List[float]]]] = None

    def _init_callback(self) -> None:
        """
        (# NEW) 在训练开始时初始化所有需要依赖环境信息的变量
        """
        # 从训练环境中获取并行环境的数量
        num_envs = self.training_env.num_envs
        # 为每个环境创建一个独立的字典，用于暂存当前 episode 的指标
        self.env_metric_buffers = [{} for _ in range(num_envs)]

        # 如果用户没有指定要追踪的指标，我们从环境中自动推断
        if self.metrics_to_track is None:
            # 运行一步来获取 info 字典的结构
            # 注意：这假设所有环境的 info 结构都一样
            _ = self.training_env.reset()
            _, _, _, infos = self.training_env.step(
                [self.training_env.action_space.sample() for _ in range(num_envs)]
            )
            self.metrics_to_track = [
                k for k in infos[0].keys() if not k.startswith("_")
            ]

        # 初始化每个环境的缓冲区
        for i in range(num_envs):
            for metric in self.metrics_to_track:
                self.env_metric_buffers[i][metric] = []

    def _on_step(self) -> bool:
        # (# MODIFIED) 在第一次 on_step 时执行初始化
        if self.env_metric_buffers is None:
            self._init_callback()

        # (# NEW) 用于收集在当前这一个 on_step 中所有已完成 episode 的指标均值
        finished_episodes_metrics: Dict[str, List[float]] = {
            metric: [] for metric in self.metrics_to_track
        }

        # (# MODIFIED) 遍历每一个并行环境
        for i in range(self.training_env.num_envs):
            info = self.locals["infos"][i]
            done = self.locals["dones"][i]

            # 持续为每个环境的缓冲区收集数据
            for metric in self.metrics_to_track:
                # 确保 info 字典中有我们想追踪的 metric
                if metric in info:
                    self.env_metric_buffers[i][metric].append(info[metric])

            # (# MODIFIED) 如果当前环境的 episode 结束了
            if done:
                self.total_episodes_finished += 1

                # 计算这个刚结束的 episode 的各项指标的均值
                for metric, values in self.env_metric_buffers[i].items():
                    if values:
                        mean_value = np.mean(values)
                        finished_episodes_metrics[metric].append(mean_value)
                        values.clear()  # 清空这个环境的缓冲区，为下一个 episode 做准备

        # (# NEW) 在所有环境都检查完毕后，进行一次集中的日志记录
        log_dict = {}
        has_new_data = False

        # 计算所有“刚刚完成的” episodes 的最终平均值
        for metric, mean_list in finished_episodes_metrics.items():
            if mean_list:
                final_mean = np.mean(mean_list)
                log_dict[f"episode/{metric}"] = final_mean
                has_new_data = True

        # 只有当这一步中至少有一个 episode 完成时，才记录日志
        if has_new_data:
            # 记录当前已完成的总 episode 数量
            log_dict["rollout/total_episodes"] = self.total_episodes_finished

            # (# MODIFIED) 使用 total_timesteps 作为 wandb 的横坐标 (x-axis)
            # 这是 RL 中最规范、最标准的做法，可以正确反映样本效率
            wandb.log(log_dict, step=self.num_timesteps)

        return True


def SACtest():
    env1 = make_vec_env(
        "UAVEnv-v0",
        n_envs=8,
        vec_env_cls=SubprocVecEnv,  # 使用SubprocVecEnv来真正利用多核CPU
        seed=SEED,  # 设置随机种子以确保可重复性
    )
    log_dir = os.path.join("SAC_v0_model", "logs")
    os.makedirs(log_dir, exist_ok=True)  # 确保日志目录存在

    wandb.init(
        project="UAV-SAC_1",  # 项目名称（wandb 仪表盘中显示）
        name="experiment-SAC",  # 实验名称（可选）
        config={  # 记录超参数（可选）
            "policy": "MlpPolicy",
            "total_timesteps": 100000,
        },
        sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
    )

    metric_callback = ParallelEpisodeMetricCallback(verbose=1)

    # env1._get_obs()
    # check_env(env1.unwrapped, skip_render_check=True)
    # 初始化 SAC 模型
    model = SAC(
        "MlpPolicy",  # 使用多层感知机策略
        env1,
        verbose=1,  # 打印训练日志
        tensorboard_log=log_dir,  # 保存日志用于TensorBoard可视化
        gamma=0.99,  # 折扣因子
        batch_size=256,  # 经验回放的批量大小
        learning_rate=3e-4,  # 学习率
        buffer_size=1_000_000,  # 经验回放的缓冲区大小
        tau=0.005,  # 软更新参数
    )

    # 训练模型（带进度条）
    model.learn(
        total_timesteps=100000,
        callback=metric_callback,  # 显示进度条
        log_interval=10,  # 每10步打印一次日志
    )
    wandb.finish()

    # 保存模型

    import time

    # 确保目标文件夹存在
    timestamp = int(time.time())
    save_dir = os.path.join("SAC_v0_model", "models")
    os.makedirs(save_dir, exist_ok=True)
    model.save(os.path.join(save_dir, f"sac_model_{timestamp}"))


def SAC_hybrid_test():
    env1 = make_vec_env(
        "UAVEnv-v1",
        n_envs=8,
        vec_env_cls=SubprocVecEnv,  # 使用SubprocVecEnv来真正利用多核CPU
        seed=SEED,  # 设置随机种子以确保可重复性
    )
    # env1.reset(seed=SEED)  # 设置随机种子以确保可重复性
    log_dir = os.path.join("hybridSAC_v3_model", "logs")
    os.makedirs(log_dir, exist_ok=True)  # 确保日志目录存在
    # 初始化 WandB
    wandb.init(
        project="UAV-SAC_1",  # 项目名称（wandb 仪表盘中显示）
        name="SAC-multiCritic-autodl",  # 实验名称（可选）
        config={  # 记录超参数（可选）
            "policy": "MlpPolicy",
            "total_timesteps": 100000,
        },
        sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
    )

    metric_callback = ParallelEpisodeMetricCallback(verbose=1)
    # swanlab_callback = SwanLabCallback(
    #     project="UAV-SAC_1",  # 项目名称（wandb 仪表盘中显示）
    #     experiment_name="multi-environment",  # 实验名称（可选）
    #     verbose=1,  # 打印训练日志
    #     tensorboard_log=log_dir,  # 保存日志用于TensorBoard可视化
    # )
    # callbacklist = CallbackList([metric_callback, swanlab_callback])
    model = HybridSAC(
        "HybridSACPolicy",  # 使用自己定义的策略
        env1,
        verbose=1,  # 打印训练日志
        tensorboard_log=log_dir,  # 保存日志用于TensorBoard可视化
        gamma=0.99,  # 折扣因子 # 其实也是默认值
        batch_size=512,  # 经验回放的批量大小 #默认值
        learning_rate=3e-4,  # 学习率 #默认值
        buffer_size=1_000_000,  # 经验回放的缓冲区大小  #默认值
        tau=0.005,  # 软更新参数 #默认值
        ent_coef=0.1,  # 自动调整熵系数
        device="cuda",
        policy_kwargs=policy_kwargs,  # 使用自定义的学习率调度器
    )
    model.learn(
        total_timesteps=100000,
        callback=metric_callback,  # 显示进度条
        log_interval=10,  # 每10步打印一次日志
    )
    swanlab.finish()
    import time

    # 确保目标文件夹存在
    timestamp = int(time.time())
    save_dir = os.path.join("hybridSAC_v3_model", "models")
    os.makedirs(save_dir, exist_ok=True)
    model.save(os.path.join(save_dir, f"hybrid_sac_model_{timestamp}"))


def find_best_hyperparameters_sweep():
    sweep_configuration = {
        "name": "SB3_TD3_Offloading",
        "method": "bayes",
        "metric": {
            "name": "rollout/ep_rew_mean",  # SB3自动记录的平均回合奖励
            "goal": "maximize",
        },
        "parameters": {
            # 核心参数 (与SB3实现严格对应)
            "learning_rate": {
                "distribution": "log_uniform_values",
                "min": 1e-5,  # 1e-5
                "max": 1e-3,  # 1e-3
            },
            "buffer_size": {
                "values": [100000, 300000, 1000000]  # 1e5 to 1e6
            },
            "batch_size": {"values": [64, 128, 256, 512]},
            "tau": {"min": 0.001, "max": 0.01},
            "gamma": {"min": 0.9, "max": 0.999},
            # TD3特有参数
            "policy_delay": {
                "values": [2, 3, 4]  # 必须为整数
            },
            "target_policy_noise": {"min": 0.1, "max": 0.3},
            "target_noise_clip": {"min": 0.3, "max": 0.7},
            # 网络结构参数
            "net_arch": {
                "values": [
                    [64, 64],  # 简单双隐藏层
                    [128, 128],
                    [256, 256],
                    {"pi": [64], "qf": [128]},  # 异构结构
                    {"pi": [128, 128], "qf": [256, 256]},
                ]
            },
        },
        "early_terminate": {"type": "hyperband", "min_iter": 10, "eta": 3},
    }
    sweep_id = wandb.sweep(
        sweep=sweep_configuration,
        project="UAV-TD3-Optimization",
        entity="SACtest",  # 你的 W&B 用户名
    )
    print(f"Sweep ID: {sweep_id}")  # 确认 Sweep 已创建
    wandb.agent(sweep_id, function=TD3_test, entity="SACtest", count=10)  # 运行30次实验


if __name__ == "__main__":
    # vv()
    # test_model()
    # TD3_test()
    # TD3_useThebest()
    # SACtest()
    SAC_hybrid_test()
    # find_best_hyperparameters()
    # find_best_hyperparameters_sweep()
