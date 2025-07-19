from SAC_test.entities.custom_env import CustomEnv

# from custom_components.hybrid_sac_agent import HybridSAC
from diffusion_sac.diffusion_sac_agent import DiffusionSACAgent
import gymnasium as gym
from gymnasium.wrappers import TimeLimit
from gymnasium.envs.registration import register
from stable_baselines3.common.env_checker import check_env
from stable_baselines3 import SAC, TD3, PPO
from stable_baselines3.common.callbacks import (
    BaseCallback,
    CallbackList,
    CheckpointCallback,
)
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv

import yaml
import argparse

import random
from SAC_test.rule_based_agent import RuleBasedAgent
import wandb
from wandb.integration.sb3 import WandbCallback
# from swanlab.integration.sb3 import SwanLabCallback


import pandas as pd  # Optional, but helpful
import os
import torch as th

import numpy as np
from typing import Dict, List, Optional, Tuple

from gymnasium.wrappers import TimeLimit
import warnings
import swanlab

# swanlab.sync_wandb()
# # swanlab.sync_tensorboard_torch()

# warnings.filterwarnings("ignore")


# 固定随机种子
SEED = 42


register(
    id="UAVEnv-v0",
    entry_point="SAC_test.entities.custom_env_sac:CustomEnv",
    max_episode_steps=40,
)
register(
    id="UAVEnv-v1",
    entry_point="SAC_test.entities.custom_env:CustomEnv",
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


class UAVEnvWrapper(gym.Wrapper):
    """
    将需要混合动作(Tuple)的无人机环境，
    伪装成一个接受连续动作(Box)的环境。
    """

    def __init__(self, env: gym.Env, ue_embedding_dim: int = 8):
        super().__init__(env)

        # 从原始环境中获取离散和连续动作空间的信息
        self.num_ues = self.env.action_space.spaces[0].n
        self.continuous_dim = self.env.action_space.spaces[1].shape[0]
        self.ue_embedding_dim = ue_embedding_dim

        # ======================= 关键修正 =======================
        # 创建一个独立的、有固定种子的随机数生成器
        # 种子 '42' 是一个任意选择的数字，只要它固定不变即可
        embedding_rng = np.random.RandomState(42)

        # 使用这个固定的生成器来创建嵌入向量
        ue_embeddings = embedding_rng.randn(self.num_ues, self.ue_embedding_dim).astype(
            np.float32
        )
        # =======================================================

        self.ue_embeddings_th = th.from_numpy(ue_embeddings).to(
            "cuda" if th.cuda.is_available() else "cpu"
        )

        # 定义新的、统一的、对Agent可见的动作空间 (Box)
        new_action_dim = self.ue_embedding_dim + self.continuous_dim
        # 重要：Agent输出的动作范围通常是[-1, 1]，我们需要在这里定义好
        self.action_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(new_action_dim,), dtype=np.float32
        )
        print("UAVEnvWrapper 已创建，动作空间已转换。")

    def _decode_action(self, action: np.ndarray) -> Tuple[int, np.ndarray]:
        """
        解码Agent输出的连续动作向量。
        """
        action = np.clip(action, -1.0, 1.0)

        action_th = th.from_numpy(action).to(self.ue_embeddings_th.device)

        # 解码离散部分 (选择UE)
        ue_vector_from_action = action_th[: self.ue_embedding_dim]
        # self.ue_embeddings_th 的 shape: [num_ues, ue_embedding_dim]
        distances = th.cdist(ue_vector_from_action.unsqueeze(0), self.ue_embeddings_th)
        # distances 的 shape: [1, num_ues], 找到最小距离的索引
        discrete_action = th.argmin(distances, dim=1).item()

        # 解码连续部分 (飞行参数)
        # 注意：Agent输出的连续动作在[-1, 1]范围，需要映射回原始范围
        original_continuous_space = self.env.action_space.spaces[1]
        low = original_continuous_space.low
        high = original_continuous_space.high

        # 从[-1, 1]映射回[low, high]
        continuous_action_normalized = action[self.ue_embedding_dim :]
        continuous_action_rescaled = low + (
            0.5 * (continuous_action_normalized + 1.0) * (high - low)
        )

        return (discrete_action, continuous_action_rescaled)

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, dict]:
        # 将Agent的连续动作解码为环境需要的混合动作
        hybrid_action = self._decode_action(action)
        return self.env.step(hybrid_action)


def run_experiment(config: dict):
    """
    根据给定的配置字典，运行单个实验。

    Args:
        config (dict): 包含单个实验所有参数的字典。
    """
    print(f"--- Running Experiment: {config['experiment_name']} ---")

    agent_type = config.get("agent_type")  # 这里的config就是yaml文件里的一个元素

    if agent_type == "rule_based":
        num_episodes = config.get("num_episodes", 5)
        print("Running Rule-Based Agent...")
        # 这里你需要调用一个函数来运行你的规则智能体并评估它
        run_rule_based_agent(num_episodes)  # 这是一个示例调用
        print("Rule-Based Agent finished.")
        return  # 基准测试运行完后直接返回

    elif agent_type == "sac":
        run_vanilla_sac(config)  # 直接运行SAC的测试函数
    elif agent_type == "ppo":
        run_ppo(config)
    elif agent_type == "diffusion_sac":
        run_diffusion_sac(config)
    # 在这里可以添加你自己改进的SAC算法的逻辑
    # elif agent_type == "my_sac_v1":
    #     model = MySACv1(...) # 使用你的自定义参数

    else:
        raise ValueError(f"Unknown agent_type: {agent_type}")


def run_vanilla_sac(config: dict):
    # 从配置中获取环境和超参数
    experiment_name = config.get("experiment_name", "Vanilla-SAC")
    learning_rate = config.get("learning_rate", 3e-4)
    total_timesteps = config.get("total_timesteps", 100000)
    gamma = config.get("gamma", 0.99)
    batch_size = config.get("batch_size", 256)
    tau = config.get("tau", 0.005)
    buffer_size = config.get("buffer_size", 1_000_000)

    # 使用的是纯连续的环境v0
    env1 = make_vec_env(
        "UAVEnv-v0",
        n_envs=8,
        vec_env_cls=SubprocVecEnv,  # 使用SubprocVecEnv来真正利用多核CPU
        seed=SEED,  # 设置随机种子以确保可重复性
    )
    log_path = os.path.join("logs", experiment_name)
    save_path = os.path.join("models", experiment_name)
    os.makedirs(log_path, exist_ok=True)  # 确保日志目录存在
    os.makedirs(save_path, exist_ok=True)  # 确保模型保存目录存在

    wandb.init(
        project="SAC-new-env",  # 项目名称（wandb 仪表盘中显示）
        name=experiment_name,  # 实验名称（可选）
        # config={  # 记录超参数（可选）
        #     # "policy": "MlpPolicy",
        #     "total_timesteps": 100000,
        #     # "fairness_penalty_weight": 10.0,  # 公平性惩罚权重
        # },
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
        tensorboard_log=log_path,  # 保存日志用于TensorBoard可视化
        gamma=gamma,  # 折扣因子
        batch_size=batch_size,  # 经验回放的批量大小
        learning_rate=learning_rate,  # 学习率
        buffer_size=buffer_size,  # 经验回放的缓冲区大小
        tau=tau,  # 软更新参数
    )

    # 训练模型（带进度条）
    model.learn(
        total_timesteps=total_timesteps,
        callback=metric_callback,  # 显示进度条
        log_interval=10,  # 每10步打印一次日志
    )
    wandb.finish()

    # 保存模型

    import time

    # 确保目标文件夹存在
    timestamp = int(time.time())
    model_name = f"{experiment_name}_{timestamp}"
    model.save(os.path.join(save_path, model_name))


def run_ppo(config: dict):
    print()


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
        batch_size=256,  # 经验回放的批量大小 #默认值
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


def run_diffusion_sac(config: dict):
    # 要创建的并行环境数量
    num_envs = 4

    # Wrapper 和 Policy 的参数
    # 注意：wrapper_kwargs 是用来传递给 Wrapper 的构造函数的
    experiment_name = config.get("experiment_name", "Diffusion-SAC-UAV")
    wrapper_kwargs = config.get("wrapper_kwargs", {})
    policy_kwargs = config.get("policy_kwargs", {})
    qne_k_samples = config.get("qne_k_samples", 32)
    total_timesteps = config.get("total_timesteps", 1_000_000)
    learning_rate = config.get("learning_rate", 3e-4)
    # 保存log和model的路径
    log_path = os.path.join("logs", experiment_name)
    save_path = os.path.join("models", experiment_name)
    os.makedirs(log_path, exist_ok=True)  # 确保日志目录存在
    os.makedirs(save_path, exist_ok=True)  # 确保模型保存目录存在

    # 1. 创建您的原始无人机环境
    env = make_vec_env(
        "UAVEnv-v1",
        n_envs=num_envs,  # 创建多个并行环境
        wrapper_class=UAVEnvWrapper,  # 使用自定义的包装器
        wrapper_kwargs=wrapper_kwargs,  # 传递包装器参数
        vec_env_cls=SubprocVecEnv,  # 使用SubprocVecEnv来真正利用多核CPU
        seed=SEED,  # 设置随机种子以确保可重复性
    )
    # 2.初始化wandb
    wandb.init(
        project="SAC-new-env",  # 项目名称（wandb 仪表盘中显示）
        name=experiment_name,  # 实验名称（可选）
        config={  # 记录超参数（可选）
            # "policy": "MlpPolicy",
            "total_timesteps": total_timesteps,
            "qne_k_samples": qne_k_samples,
            "T_steps": policy_kwargs["T"],
            "ue_embedding_dim": wrapper_kwargs["ue_embedding_dim"],
            # "fairness_penalty_weight": 10.0,  # 公平性惩罚权重
        },
        sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
    )
    metric_callback = ParallelEpisodeMetricCallback(verbose=1)

    #
    # 3. 像之前一样创建并训练您的Agent
    #    Agent本身不需要做任何改动，因为它看到的是一个简单的Box动作空间
    model = DiffusionSACAgent(
        policy="DiffusionSACPolicy",  # 使用自定义的DiffusionSAC策略
        env=env,  # <-- 传入被包裹后的环境
        tensorboard_log=log_path,  # 保存日志用于TensorBoard可视化
        verbose=1,
        learning_starts=10000,  # 经验回放开始训练的步数
        qne_k_samples=qne_k_samples,  # QNE中的K值
        policy_kwargs=policy_kwargs,  # 传入扩散模型和QNE所需的特定超参数
        learning_rate=learning_rate,  # 学习率
        # ... 其他所有超参数 ...
    )

    print("--- 开始在您的自定义无人机环境上训练 DiffusionSACAgent ---")
    model.learn(
        total_timesteps=total_timesteps,
        log_interval=10,
        callback=metric_callback,
    )

    import time

    # 确保目标文件夹存在
    timestamp = int(time.time())
    model_name = f"{experiment_name}_{timestamp}"
    model.save(os.path.join(save_path, model_name))


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
    wandb.agent(
        sweep_id, function=SAC_hybrid_test, entity="SACtest", count=10
    )  # 运行30次实验


def test_model():
    # --- 自动查找并加载最新的模型 ---
    import os
    import re

    model_dir = os.path.join("SAC_v0_model", "models")
    if not os.path.isdir(model_dir):
        print(f"错误：模型目录 '{model_dir}' 不存在。")
        return

    # 1. 找到所有符合 'sac_model_*.zip' 格式的文件
    model_files = [
        f
        for f in os.listdir(model_dir)
        if f.startswith("sac_model_") and f.endswith(".zip")
    ]

    if not model_files:
        print(f"错误：在目录 '{model_dir}' 中没有找到任何模型文件。")
        return

    # 2. 从文件名中解析时间戳，并找到最新的一个
    latest_timestamp = -1
    latest_model_file = ""
    for filename in model_files:
        # 使用正则表达式从 'sac_model_1751615810.zip' 中提取数字
        match = re.search(r"sac_model_(\d+)\.zip", filename)
        if match:
            timestamp = int(match.group(1))
            if timestamp > latest_timestamp:
                latest_timestamp = timestamp
                latest_model_file = filename

    latest_model_path = os.path.join(model_dir, latest_model_file)

    # 加载模型
    model = SAC.load(latest_model_path)
    # 创建环境
    env = gym.make("UAVEnv-v0", render_mode="human")

    # 确保TimeLimit生效的两种方式：
    # 方式1：直接包装（推荐）
    env = TimeLimit(env, max_episode_steps=40)

    # 方式2：检查是否已被TimeLimit包装
    if not isinstance(env, TimeLimit):
        env = TimeLimit(env, max_episode_steps=40)

    # 运行测试
    env = TimeLimit(env, max_episode_steps=40)

    # 运行N个episodes进行测试
    num_episodes = 5

    for episode in range(num_episodes):
        # 设置当前episode编号
        env._current_episode = episode + 1

        obs, _ = env.reset()
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, _, done, truncated, _ = env.step(action)

            # 强制检查步数限制
            if env._elapsed_steps >= 40:
                truncated = True
                break

        print(f"Episode {episode + 1} finished")
        env.render()  # 这会阻塞直到窗口关闭

    env.close()


def run_rule_based_agent(num_episodes: int = 5):
    print("--- 正在测试基于规则的智能体 ---")
    # gym.make 会根据注册信息自动应用 TimeLimit 包装器
    env = gym.make("UAVEnv-v0", render_mode="human")
    agent = RuleBasedAgent()

    num_episodes = 5
    for episode in range(num_episodes):
        # 为渲染设置当前回合编号
        env.unwrapped._current_episode = episode + 1

        obs, _ = env.reset()
        done = False
        total_reward = 0
        step_count = 0

        while not done:
            # 智能体需要访问环境的内部状态来做决策
            action, _ = agent.predict(obs, env.unwrapped)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            step_count += 1
            done = terminated or truncated

        print(
            f"Episode {episode + 1} finished in {step_count} steps. Total reward: {total_reward:.2f}"
        )
        # 在回合结束后渲染结果
        env.render()

    env.close()


if __name__ == "__main__":
    # test_rule_based_agent()
    # test_model()
    # TD3_test()
    # TD3_useThebest()
    # SACtest()
    # test_model()
    # SAC_hybrid_test()
    # find_best_hyperparameters()
    # find_best_hyperparameters_sweep()
    # --- 这是脚本的主入口 ---

    # 1. 创建一个命令行参数解析器
    parser = argparse.ArgumentParser(
        description="Run RL experiments based on a YAML config file."
    )

    # 2. 添加我们需要的命令行参数
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to the experiment configuration YAML file (e.g., experiments.yaml)",
    )
    parser.add_argument(
        "--name",
        type=str,
        required=True,
        help="The 'experiment_name' from the config file to run.",
    )

    # 3. 解析传入的参数
    args = parser.parse_args()

    # 4. 加载YAML配置文件
    try:
        with open(args.config, "r", encoding="utf-8") as f:
            all_experiments = yaml.safe_load(f)
    except FileNotFoundError:
        print(f"Error: Config file not found at {args.config}")
        exit(1)

    # 5. 查找与传入的 --name 匹配的实验配置
    experiment_config = None
    for exp in all_experiments:
        if (
            exp["experiment_name"] == args.name
        ):  # 匹配命令行输入的--name 也就是args.name是否和yaml文件里哪个exp的experiment_name一致
            experiment_config = exp
            break

    # 6. 如果找到了配置，就运行实验
    if experiment_config:
        run_experiment(experiment_config)
    else:
        print(f"Error: Experiment with name '{args.name}' not found in {args.config}")
        exit(1)
