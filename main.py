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
from stable_baselines3.common.vec_env import VecEnv
from stable_baselines3.common.vec_env import DummyVecEnv  # <-- 换成导入DummyVecEnv

from typing import Callable, Union, Any, Tuple

import optuna
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
lr_critic_initial = 1e-5  # 降低一个数量级


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

def evaluate_hybrid_policy(
    model,
    env: VecEnv,
    n_eval_episodes: int = 10,
    deterministic: bool = True,
) -> Tuple[float, float]:
    """
    最终版、防御性、兼容性最强的评估函数。
    """
    unwrapped_env = env.envs[0]
    
    episode_rewards = []
    episode_lengths = []
    
    for _ in range(n_eval_episodes):
        obs, _ = unwrapped_env.reset()
        done = False
        episode_reward = 0.0
        while not done:
            # model.predict接收单个观测，但返回的是一个“批次”的动作
            # action_tuple_batch 的格式是 (np.array([d]), np.array([[c1, c2, ...]]))
            action_tuple_batch, _ = model.predict(obs[None, :], deterministic=deterministic)
            
            # --- 核心修正：在这里手动“拆箱”，取出那个唯一的动作 ---
            # 从 (1,) 的数组中取出整数
            discrete_action = int(action_tuple_batch[0][0])
            # 从 (1, C) 的二维数组中取出那一行，变成 (C,) 的一维数组
            continuous_action = action_tuple_batch[1][0]
            
            # 组合成环境step方法期望的、不带批处理维度的元组
            final_action_for_env = (discrete_action, continuous_action)
            # ----------------------------------------------------

            # 将格式完美的单个动作传递给环境
            obs, reward, done, truncated, info = unwrapped_env.step(final_action_for_env)
            done = done or truncated
            episode_reward += reward
        
        episode_rewards.append(episode_reward)
        
    mean_reward = np.mean(episode_rewards)
    std_reward = np.std(episode_rewards)

    return mean_reward, std_reward
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
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    env1 = make_vec_env(
        "UAVEnv-v1",
        n_envs=8,
        vec_env_cls=SubprocVecEnv,  # 使用SubprocVecEnv来真正利用多核CPU
        seed=SEED,  # 设置随机种子以确保可重复性
    )
    # env1.reset(seed=SEED)  # 设置随机种子以确保可重复性
    log_dir = os.path.join("hybridSAC_v4_model", "logs")
    os.makedirs(log_dir, exist_ok=True)  # 确保日志目录存在
    # 初始化 WandB
    wandb.init(
        project="UAV-SAC_1",  # 项目名称（wandb 仪表盘中显示）
        name="SAC-multiCritic-autodl",  # 实验名称（可选）
        config={  # 记录超参数（可选）
            "policy": "embedding-sac",
            "total_timesteps": 500000,
            "ent_coef": "auto",  # 自动调整熵系数
            "target_entropy": -6,  # 目标熵值，通常设置为动作空间维度的负数
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
        batch_size=1024,  # 经验回放的批量大小 #默认值
        learning_rate=3e-4,  # 学习率 #默认值
        buffer_size=1_000_000,  # 经验回放的缓冲区大小  #默认值
        tau=0.005,  # 软更新参数 #默认值
        ent_coef='auto',  # 自动调整熵系数
        target_entropy = 'auto',  # 目标熵值，通常设置为动作空间维度的负数
        device="cuda",
        policy_kwargs=policy_kwargs,  # 使用自定义的学习率调度器
    )
    model.learn(
        total_timesteps=500000,
        callback=metric_callback,  # 显示进度条
        log_interval=10,  # 每10步打印一次日志
    )
    swanlab.finish()
    import time

    # 确保目标文件夹存在
    timestamp = int(time.time())
    save_dir = os.path.join("hybridSAC_v4_model", "models")
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

# --- 核心修改：将你的测试函数改造成 objective 函数 ---
def objective(trial: optuna.Trial) -> float:
    """
    这是Optuna会调用的目标函数。
    每一次调用，都是一次完整的、使用不同超参数的训练和评估。
    """
    # --- 2. 在这里定义我们要优化的超参数搜索空间 ---
    # Optuna会从这些范围中为本次"trial"（尝试）选择一组值
    
    # a. 学习率 (建议使用对数均匀分布)
    lr_actor_initial = trial.suggest_float("lr_actor", 1e-5, 1e-3, log=True)
    lr_critic_initial = trial.suggest_float("lr_critic", 1e-5, 1e-3, log=True)
    
    # b. 网络和嵌入维度
    embedding_dim = trial.suggest_categorical("embedding_dim", [16, 32, 64])
    # net_arch_size = trial.suggest_categorical("net_arch_size", [128, 256, 512])
    # net_arch = [net_arch_size, net_arch_size]

    # c. SAC核心参数
    batch_size = trial.suggest_categorical("batch_size", [256, 512, 1024])
    tau = trial.suggest_float("tau", 0.001, 0.02, log=True)
    gamma = trial.suggest_float("gamma", 0.98, 0.999, log=True)
    
    # d. 熵相关参数
    # 我们可以让Optuna在“固定熵”和“自动熵”之间选择
    ent_coef_type = trial.suggest_categorical("ent_coef_type", ["fixed", "auto"])
    if ent_coef_type == "fixed":
        ent_coef = trial.suggest_float("ent_coef_fixed", 0.005, 0.2, log=True)
        target_entropy = "auto" # 固定熵时，这个值无所谓
    else: # "auto"
        ent_coef = "auto"
        # 假设连续维度是3，默认是-3。我们让它在-6到-1之间搜索
        target_entropy = trial.suggest_float("target_entropy", -6.0, -1.0)

    # --- 3. 使用建议的超参数来配置和运行你的模型 ---

    # 创建学习率调度函数
    lr_actor_schedule = linear_schedule(lr_actor_initial)
    lr_critic_schedule = linear_schedule(lr_critic_initial)

    # 打包进policy_kwargs
    policy_kwargs = {
        "lr_actor_schedule": lr_actor_schedule,
        "lr_critic_schedule": lr_critic_schedule,
        "embedding_dim": embedding_dim,
        # "net_arch": net_arch,
    }
    
    # 为WandB创建一个唯一的实验名称
    run_name = f"trial_{trial.number}_lrA_{lr_actor_initial:.1e}_lrC_{lr_critic_initial:.1e}_bs_{batch_size}"
    
    # 初始化WandB
    run = wandb.init(
        project="UAV-SAC-Optuna", # 为优化创建一个新项目
        name=run_name,
        config=trial.params, # 将Optuna选择的参数记录到WandB
        reinit=True, # 允许多次init
        sync_tensorboard=True,
    )

    # 创建环境
    env = make_vec_env("UAVEnv-v1", n_envs=8, vec_env_cls=DummyVecEnv, seed=SEED)
    log_dir = f"optuna_logs/trial_{trial.number}"
    os.makedirs(log_dir, exist_ok=True)
    
    # 创建模型
    model = HybridSAC(
        "HybridSACPolicy",
        env,
        verbose=0, # 在优化时通常关闭详细日志，避免刷屏
        tensorboard_log=log_dir,
        gamma=gamma,
        batch_size=batch_size,
        learning_rate=lr_actor_initial, # 顶层LR，主要起占位作用
        tau=tau,
        ent_coef=ent_coef,
        target_entropy=target_entropy,
        device="cuda",
        policy_kwargs=policy_kwargs,
    )

    # 训练模型
    # 注意：为了让优化过程更快，可以在这里使用一个较短的训练步数
    try:
        model.learn(
            total_timesteps=30000, # 先用一个较短的步数来快速筛选
            log_interval=100, # 减少日志频率
            callback=None # 优化时可以先不用自定义callback
        )
    except Exception as e:
        print(f"Trial {trial.number} failed with error: {e}")
        run.finish() # 确保即使出错也关闭wandb run
        # 对于不稳定的组合，我们可以返回一个很差的值，让Optuna知道这是一个坏的尝试
        return -float("inf")


    # --- 4. 评估训练好的模型并返回最终分数 ---
    
    # 在独立的评估环境中评估模型，结果更可靠
    eval_env = model.get_env() # 获取模型正在使用的VecEnv
    
    # 调用我们自己的评估函数
    mean_reward, _ = evaluate_hybrid_policy(
        model, eval_env, n_eval_episodes=20, deterministic=True
    )
    # ----------------------------------------
    
    wandb.log({"eval_mean_reward": mean_reward})
    run.finish()
    
    return mean_reward



def SAC_hybrid_test_sweep():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

        # 初始化 WandB（注意：这里不需要手动指定 config，Sweep 会自动注入）
    print("当前模式:", "Sweep" if wandb.config else "普通训练")
    print("传入的 config:", wandb.config)  # 检查是否收到参数

    run = wandb.init(sync_tensorboard=True)
    # 从 wandb.config 获取超参数
    config = wandb.config

    #使用config的参数或者默认值
    params={
        "gamma": config.gamma if config.gamma else 0.99,
        "batch_size": config.batch_size if config.batch_size else 256,
        "buffer_size": config.buffer_size if config.buffer_size else 1_000_000,
        "tau": config.tau if config.tau else 0.001,
        "ent_coef": config.ent_coef if config.ent_coef else 0.1,
        "device": "cuda",  # 使用GPU
        
    }
    policy_kwargs = {
    "net_arch": config.get("net_arch") or [64, 64],
    "lr_actor_schedule": linear_schedule(config.get("lr_actor") or 3e-4),
    "lr_critic_schedule": linear_schedule(config.get("lr_critic") or 3e-5)
}    
    metric_callback = ParallelEpisodeMetricCallback(verbose=1)
    env1 = make_vec_env(
        "UAVEnv-v1",
        n_envs=8,
        vec_env_cls=SubprocVecEnv,  # 使用SubprocVecEnv来真正利用多核CPU
        seed=SEED,  # 设置随机种子以确保可重复性
    )
    model = HybridSAC(
        "HybridSACPolicy",
        env1,
        verbose=1,
        tensorboard_log=wandb.run.dir,  # 直接使用 WandB 的日志目录
        **params,  # 使用从 wandb.config 获取的参数
        policy_kwargs=policy_kwargs if policy_kwargs else None,
    )
    model.learn(
        total_timesteps=100000,
        callback=metric_callback,  # 显示进度条
        log_interval=10,  # 每10步打印一次日志
    )

def SAC_hybrid_sweep():
    sweep_configuration = {
        "name": "SAC_Hybrid_Offloading",
        "method": "bayes",
        "metric": {
            "name": "rollout/ep_rew_mean",  # SB3自动记录的平均回合奖励
            "goal": "maximize",
        },
        "parameters": {
            # 核心参数 (与SB3实现严格对应)
            "lr_actor": {
                "distribution": "log_uniform_values",
                "min": 1e-5,
                "max": 1e-3
            },
            "lr_critic": {
                "distribution": "log_uniform_values",
                "min": 1e-6,
                "max": 1e-4
            },

            "buffer_size": {
                "values": [100000, 300000, 1000000]  # 1e5 to 1e6
            },
            "batch_size": {"values": [64, 128, 256, 512]},
            "tau": {"min": 0.001, "max": 0.01},
            "gamma": {"min": 0.9, "max": 0.999},
            # SAC特有参数
            "ent_coef": {"values": [ 0.1, 0.2, 0.5]},
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
        project="UAV-SAC-Hybrid-Optimization",
        entity="SACtest",  # 你的 W&B 用户名
    )
    print(f"Sweep ID: {sweep_id}")  # 确认 Sweep 已创建
    wandb.agent(sweep_id, function=SAC_hybrid_test_sweep, entity="SACtest", count=10)

if __name__ == "__main__":
    # vv()
    # test_model()
    # TD3_test()
    # TD3_useThebest()
    # SACtest()
    SAC_hybrid_test()
    # SAC_hybrid_sweep()
    # find_best_hyperparameters()
    # find_best_hyperparameters_sweep()
    # N_TRIALS = 100
    
    # # 创建一个研究，告诉Optuna我们的目标是“最大化”objective函数的返回值
    # study = optuna.create_study(direction="maximize")
    
    # # 启动优化！Optuna会自动调用objective函数N_TRIALS次
    # study.optimize(objective, n_trials=N_TRIALS)

    # # 打印出最佳结果
    # print("Optimization finished.")
    # print("Best trial:")
    # trial = study.best_trial
    # print(f"  Value: {trial.value}")
    # print("  Params: ")
    # for key, value in trial.params.items():
    #     print(f"    {key}: {value}")