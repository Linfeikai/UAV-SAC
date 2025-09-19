# from SAC_test.entities.custom_env import CustomEnv


# HYDRA: 导入 hydra 和 DictConfig
import hydra
from omegaconf import DictConfig, OmegaConf

# from custom_components.hybrid_sac_agent import HybridSAC
from diffusion_sac.diffusion_sac_agent import DiffusionSACAgent
import gymnasium as gym
from gymnasium.wrappers import TimeLimit
from gymnasium.envs.registration import register
from stable_baselines3.common.env_checker import check_env
from stable_baselines3 import SAC, TD3, PPO, DDPG
from stable_baselines3.common.callbacks import (
    BaseCallback,
    CallbackList,
    CheckpointCallback,
)
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.vec_env import VecNormalize

from datetime import datetime

import yaml
import argparse

import random
from SAC_test.rule_based_agent import RuleBasedAgent
import wandb
from wandb.integration.sb3 import WandbCallback
from torch.optim import AdamW
from torch.optim import Adam
# from swanlab.integration.sb3 import SwanLabCallback


import pandas as pd  # Optional, but helpful
import os
os.environ["WANDB_MODE"]="offline"
import torch as th
import sys
import numpy as np
from typing import Dict, List, Optional, Tuple

from gymnasium.wrappers import TimeLimit
import warnings
import statistics
import time
# import swanlab

# swanlab.sync_wandb()
# # swanlab.sync_tensorboard_torch()

# warnings.filterwarnings("ignore")


# 固定随机种子
SEED = 408


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


# 2. 为它们分别创建衰减函数
def linear_schedule(initial_value: float, final_value: float ):
    print(type(initial_value))
    print(type(final_value))
    # 确保初始值不小于最终值
    assert initial_value >= final_value, "初始学习率必须大于或等于最终学习率"

    def func(progress_remaining: float) -> float:
        return (initial_value - final_value) * progress_remaining + final_value

    return func





import numpy as np
import wandb
from stable_baselines3.common.callbacks import BaseCallback
from typing import Optional, List, Dict, Any


from stable_baselines3.common.monitor import Monitor
import gymnasium as gym


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
            inferred_metrics = []
            if infos:  # 确保 infos 列表不为空
                sample_info = infos[0]  # 取第一个环境的 info 字典作为样本
                for k, v in sample_info.items():
                    # 排除以下划线开头的私有键，并且只追踪数值型标量
                    if not k.startswith("_") and isinstance(v, (int, float, np.number)):
                        inferred_metrics.append(k)
                    # 可选：如果你想看到哪些非标量指标被忽略了，可以取消下面这行的注释
                    # elif self.verbose > 0:
                    #     print(f"Warning: Metric '{k}' (type: {type(v)}) will be ignored for averaging.")
            self.metrics_to_track = inferred_metrics

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

            for metric in self.metrics_to_track:
                # 确保 info 字典中有我们想追踪的 metric
                if metric in info:
                    self.env_metric_buffers[i][metric].append(info[metric])
                else:
                    # 如果某个指标在当前步缺失，追加一个默认值（例如 0.0）
                    # 这样可以保证列表长度一致，避免潜在的问题
                    self.env_metric_buffers[i][metric].append(0.0)  # 或者 np.nan

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
        解码Agent输出的连续动作向量 (此方法保持不变)。
        """
        # clamp到[-1, 1]是一个好习惯，防止Agent输出越界
        action = np.clip(action, -1.0, 1.0)

        action_th = th.from_numpy(action).to(self.ue_embeddings_th.device)

        ue_vector_from_action = action_th[: self.ue_embedding_dim]
        distances = th.cdist(ue_vector_from_action.unsqueeze(0), self.ue_embeddings_th)
        discrete_action = th.argmin(distances, dim=1).item()

        original_continuous_space = self.env.action_space.spaces[1]
        low = original_continuous_space.low
        high = original_continuous_space.high

        continuous_action_normalized = action[self.ue_embedding_dim :]
        continuous_action_rescaled = low + (
            0.5 * (continuous_action_normalized + 1.0) * (high - low)
        )

        return (discrete_action, continuous_action_rescaled)

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, dict]:
        # 将Agent的连续动作解码为环境需要的混合动作
        hybrid_action = self._decode_action(action)
        return self.env.step(hybrid_action)


class SACWrapper(gym.Wrapper):
    """
    一个专门为标准 SAC 设计的包装器。
    它将 custom_env.py 的混合动作空间 (Tuple)
    转换为一个扁平的连续动作空间 (Box)。
    """

    def __init__(self, env: gym.Env):
        super().__init__(env)

        # 从原始环境中获取离散和连续动作空间
        discrete_space = self.env.action_space.spaces[0]
        continuous_space = self.env.action_space.spaces[1]

        # 定义新的、对 SAC Agent 可见的扁平化 Box 动作空间
        # 我们将离散动作的范围 [0, N-1] 也视为一个连续维度
        self.action_space = gym.spaces.Box(
            low=np.concatenate(([discrete_space.start], continuous_space.low)),
            high=np.concatenate(([discrete_space.n - 1], continuous_space.high)),
            dtype=np.float32,
        )
        print("SACWrapper applied: Action space converted from Tuple to Box.")

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, dict]:
        """
        在将动作传递给底层环境之前，对其进行解码。
        """
        # 1. 解码动作：将连续值转换为 (离散, 连续) 的元组
        #    这部分逻辑完全来自于你之前的 custom_env_sac.py

        # 离散部分：取整并裁剪
        ue_id = int(np.round(action[0]))
        ue_id = np.clip(ue_id, 0, self.env.action_space.spaces[0].n - 1)

        # 连续部分：直接使用
        continuous_parts = action[1:]

        # 2. 组装成环境期望的 Tuple 格式
        hybrid_action = (ue_id, continuous_parts)

        # 3. 将解码后的动作传递给原始环境
        return self.env.step(hybrid_action)

    # reset 方法不需要修改，它会自动调用底层环境的 reset


# =====================================================================================
# 新增函数: 两阶段训练法 - 阶段一
# =====================================================================================
def collect_teacher_buffer(
    config: DictConfig, buffer_save_path: str, seed: int, ue_num: int,wrapper_kwargs: dict
) -> None:
    """
    【单环境修正版】
    使用标准的SAC Agent（老师）在单个、非并行的环境中训练，以填充Replay Buffer。
    """
    print("\n--- [阶段一 - 单环境模式] 开始: 使用标准SAC收集高质量经验 ---")

    # (这部分参数获取逻辑保持不变)
    learning_rate = config.get("learning_rate", 3e-4)
    buffer_size = config.get("buffer_size", 1_000_000)
    n_envs = 4  # <--- 定义并行数
    original_timesteps = config.get("total_timesteps", 100000)

    # ========================== 核心修正：补偿Bug ==========================
    # 为了抵消那个“步数除以4”的Bug，我们在这里主动将目标步数乘以4
    collection_timesteps = original_timesteps * n_envs
    print(
        f"[补偿机制] 原始目标步数: {original_timesteps}, 补偿后传递给learn()的步数: {collection_timesteps}"
    )
    # =====================================================================
    env = make_vec_env(
        "UAVEnv-v1",
        n_envs=n_envs,
        vec_env_cls=SubprocVecEnv,
        seed=seed,
        wrapper_class=UAVEnvWrapper,
        wrapper_kwargs=wrapper_kwargs,
        env_kwargs={"ue_num": ue_num},
    )

    # 初始化标准的SAC模型作为“老师”
    teacher_agent = SAC(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=learning_rate,
        buffer_size=buffer_size,
        learning_starts=1000,
        gamma=config.get("gamma", 0.99),
        batch_size=config.get("batch_size", 256),
        tau=config.get("tau", 0.005),
    )

    # 训练老师Agent指定步数
    print(f"老师(SAC)将训练 {collection_timesteps} 步来填充经验池...")
    teacher_agent.learn(total_timesteps=collection_timesteps, log_interval=200)

    # 保存经验池到指定路径
    print(f"数据收集完成，将经验池保存到: {buffer_save_path}")
    teacher_agent.save_replay_buffer(buffer_save_path)

    print("--- [阶段一 - 单环境模式] 结束 ---")
    env.close()
    del teacher_agent


def run_experiment(config: dict, seed: int, ue_num: int):
    """
    根据给定的配置字典和种子，运行单个实验。

    Args:
        config (dict): 包含单个实验所有参数的字典。
        seed (int): 用于本次实验的随机种子。
        ue(int)：本次实验的UE数量
    """

    agent_type = config.get("agent_type")

    # --- 使用字典映射代替 if-elif ---
    AGENT_RUNNERS = {
        "sac": run_vanilla_sac,
        "ppo": run_ppo,
        "ddpg": run_ddpg,
        # 其他标准的agent也可以加在这里
    }

    if agent_type in AGENT_RUNNERS:
        runner = AGENT_RUNNERS[agent_type]
        runner(config, seed, ue_num)

    elif agent_type == "rule_based":
        print("Running Rule-Based Agent...")
        run_rule_based_agent(config.get("num_episodes", 5))
        print("Rule-Based Agent finished.")

    elif agent_type == "diffusion_sac":
        # Diffusion SAC 的两阶段逻辑比较特殊，可以单独处理
        if config.get("two_stage_training", False):
            print("检测到启用两阶段训练模式...")
            # 直接从配置中获取老师的配置对象，不再需要名字
            teacher_config = config.get("teacher_config") 
            if not teacher_config:
                # 使用 OmegaConf.select() 可以安全地访问，即使键不存在也不会报错
                raise ValueError("两阶段训练模式需要一个 'teacher_config' 配置块！")
            wrapper_kwargs_for_teacher = config.get("wrapper_kwargs", {}) # 从学生配置里拿

            # 定义预训练buffer的保存路径
            buffer_name = f"teacher_buffer_ue{ue_num}_seed{seed}_buffer.pkl"
            buffer_path = os.path.join("buffers", buffer_name)
            os.makedirs("buffers", exist_ok=True)

            # 阶段一：运行老师SAC来收集数据并保存buffer
            collect_teacher_buffer(teacher_config, buffer_path, seed, ue_num,wrapper_kwargs_for_teacher)

            # 阶段二：运行Diffusion SAC，并加载预训练的buffer
            run_diffusion_sac(
                config, seed, pretrained_buffer_path=buffer_path, ue_num=ue_num
            )
        else:
            # 如果不启用，则正常运行
            run_diffusion_sac(config, seed, ue_num=ue_num)
    else:
        raise ValueError(f"Unknown agent_type: {agent_type}")


def train_agent(
    agent_class,  # 传入Agent类，如 SAC, PPO
    config: dict,
    seed: int,
    ue_num: int,
):
    """
    一个通用的函数，负责处理所有Agent共享的训练流程。
    """
    # === [核心修改 1/3] ===
    # 从配置中获取基础名称，并创建一个本次运行的唯一名称
    base_experiment_name = config.get("agent_type", "default_agent") # 使用 agent_type 代替
    unique_experiment_name = f"{base_experiment_name}_ue{ue_num}_seed{seed}"
    print(f"\n{'='*50}\n--- Running: {unique_experiment_name} ---\n{'='*50}")

    total_timesteps = config["total_timesteps"]
    n_envs = config.get("n_envs", 4)  # PPO可能需要，其他agent也可以用

    # --- 1. 设置路径 (公共逻辑) ---
    # --- 1. 设置路径 (使用基础名称作为主目录) ---
    log_path = os.path.join("logs", unique_experiment_name) # 日志用唯一名称
    save_path = os.path.join("models", base_experiment_name) # 模型保存在基础名称目录下
    os.makedirs(log_path, exist_ok=True)
    os.makedirs(save_path, exist_ok=True)

    # === [核心修改 2/3] ===
    # --- 2. 初始化 Wandb (使用唯一名称) ---
    wandb.init(
        project="comparison-exp",
        name=unique_experiment_name, # <--- 在这里使用唯一名称
        notes=f"seed={seed}, ue_num={ue_num}",
        config=config,
        sync_tensorboard=True,
    )


    # --- 3. 创建环境 (公共逻辑) ---
    # 注意：这里假设了SAC和DDPG使用SACWrapper，PPO使用UAVEnvWrapper
    # 可以通过config来使其更灵活
    wrapper_map = {"SAC": SACWrapper, "DDPG": SACWrapper, "PPO": UAVEnvWrapper}
    wrapper_class = wrapper_map.get(agent_class.__name__, UAVEnvWrapper)

    env = make_vec_env(
        "UAVEnv-v1",
        n_envs=n_envs,
        vec_env_cls=SubprocVecEnv,
        seed=seed,
        wrapper_class=wrapper_class,
        env_kwargs={"ue_num": ue_num},
    )

    # --- 4. 提取模型超参数 ---
    # 从config中提取所有可能用于模型初始化的参数
    # SB3的构造函数会自动忽略不相关的参数，非常方便
    model_kwargs = config.copy()
    model_kwargs.pop("experiment_name", None)
    model_kwargs.pop("agent_type", None)
    model_kwargs.pop("total_timesteps", None)
    model_kwargs.pop("n_envs", None) 

    # --- 5. 初始化并训练模型 (核心部分) ---
    model = agent_class(
        "MlpPolicy",
        env,
        verbose=1,
        tensorboard_log=log_path,
        seed=seed,
        **model_kwargs,  # 将所有超参数一次性传入
    )

    print(f"--- [{agent_class.__name__}] 开始训练, 总步数: {total_timesteps} ---")
    model.learn(
        total_timesteps=total_timesteps,
        callback=ParallelEpisodeMetricCallback(verbose=1),  # 假设回调是通用的
        log_interval=10,
    )
    print(f"--- [{agent_class.__name__}] 训练完成 ---")

    # --- 6. 清理和保存 (公共逻辑) ---
    wandb.finish()

    # === [核心修改 3/3] ===
    # --- 6. 清理和保存 (使用唯一名称命名模型文件) ---
    timestamp = int(time.time())
    # 模型文件名包含所有关键信息
    model_name = f"{base_experiment_name}_ue{ue_num}_seed{seed}_{timestamp}.zip"
    final_save_path = os.path.join(save_path, model_name)
    model.save(final_save_path)
    print(f"--- [{agent_class.__name__}] 模型已保存至: {final_save_path} ---")

    env.close()


def run_vanilla_sac(config: dict, seed: int, ue_num: int):
    train_agent(SAC, config, seed, ue_num)


def run_ppo(config: dict, seed: int, ue_num: int):
    # 注意：PPO的环境创建可能需要不同的wrapper或参数，
    # 可以在通用函数中增加逻辑，或在这里单独处理环境创建
    train_agent(PPO, config, seed, ue_num)


def run_ddpg(config: dict, seed: int, ue_num: int):
    train_agent(DDPG, config, seed, ue_num)


def run_diffusion_sac(
    config: dict,
    seed: int,
    pretrained_buffer_path: Optional[str] = None,
    ue_num: int = 20,
):
    """
    运行Diffusion SAC实验。

    Args:
        config (dict): Diffusion SAC的配置字典。
        seed:随机种子
        pretrained_buffer_path (Optional[str]): 如果提供，则从该路径加载预训练的Replay Buffer。
        ue_num：本次训练环境的设备数量
    """

    # 要创建的并行环境数量
    num_envs = 4

    # Wrapper 和 Policy 的参数
    # 注意：wrapper_kwargs 是用来传递给 Wrapper 的构造函数的
    # === [核心修改 1/3] ===
    base_experiment_name = config.get("agent_type", "Diffusion-SAC-UAV") # 使用 agent_type 代替
    unique_experiment_name = f"{base_experiment_name}_ue{ue_num}_seed{seed}"
    print(f"\n{'='*50}\n--- Running: {unique_experiment_name} ---\n{'='*50}")

    wrapper_kwargs = config.get("wrapper_kwargs", {})
    policy_kwargs = OmegaConf.to_container(config.get("policy_kwargs", {}), resolve=True)
    lr_actor_initial = config.get("lr_actor_initial",5e-5)
    lr_critic_initial = config.get('lr_critic_initial',1.5e-4)
    lr_final = config.get('lr_final',1e-6)
    print(f'Now i have lr actor {lr_actor_initial},its type is {type(lr_actor_initial)}')
    print(f'Now i have lr critic {lr_critic_initial},its type is {type(lr_critic_initial)}')

    policy_kwargs["lr_actor_schedule"] = linear_schedule(lr_actor_initial,lr_final)
    policy_kwargs["lr_critic_schedule"] = linear_schedule(lr_critic_initial,lr_final)
    # policy_kwargs["optimizer_class "] = AdamW
    qne_k_samples = config.get("qne_k_samples", 8)
    total_timesteps = config.get("total_timesteps", 1_000_000)
    # learning_starts = config.get("learning_starts", 50000)  # 经验回放开始训练的步数
    # 关键修改：如果使用预训练buffer，我们希望Agent立即开始学习
    # 否则，使用配置文件中定义的learning_starts
    learning_starts = (
        0 if pretrained_buffer_path else config.get("learning_starts", 10000)
    )

    # learning_rate = config.get("learning_rate", 3e-4)
    qne_temperature = config.get(
        "qne_temperature", 5.0
    )  # QNE的温度参数，用于控制Softmax的平滑度
    # 保存log和model的路径
    # --- 设置路径 ---
    log_path = os.path.join("logs", unique_experiment_name)
    save_path = os.path.join("models", base_experiment_name)
    os.makedirs(log_path, exist_ok=True)
    os.makedirs(save_path, exist_ok=True)

    # 1. 创建您的原始无人机环境
    env = make_vec_env(
        "UAVEnv-v1",
        n_envs=num_envs,  # 创建多个并行环境
        wrapper_class=UAVEnvWrapper,  # 使用自定义的包装器
        # wrapper_kwargs=wrapper_kwargs,  # 传递包装器参数
        vec_env_cls=SubprocVecEnv,  # 使用SubprocVecEnv来真正利用多核CPU
        seed=seed,  # 设置随机种子以确保可重复性
        env_kwargs={"ue_num": ue_num},
    )
    # 2.初始化wandb
    wandb.init(
        project="comparison-exp",  # 项目名称（wandb 仪表盘中显示）
        name=unique_experiment_name, # <--- 在这里使用唯一名称
        notes=f"seed={seed}, ue_num={ue_num}",
        config={  # 记录超参数（可选）
            # "policy": "MlpPolicy",
            "learning_starts": learning_starts,
            "total_timesteps": total_timesteps,
            "qne_k_samples": qne_k_samples,
            "T_steps": policy_kwargs["T"],
            "ue_embedding_dim": wrapper_kwargs["ue_embedding_dim"],
            "learning_starts": learning_starts,  # 经验回放开始训练的步数
            "qne_temperature": qne_temperature,
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
        # batch_size=512,  # 经验回放的批量大小
        learning_starts=learning_starts,  # 经验回放开始训练的步数
        qne_k_samples=qne_k_samples,  # QNE中的K值
        policy_kwargs=policy_kwargs,  # 传入扩散模型和QNE所需的特定超参数
        # learning_rate=learning_rate,  # 学习率
        qne_temperature=qne_temperature,  # QNE的温度参数
        # ... 其他所有超参数 ...
    )
    # =====================================================================================
    # 新增逻辑: 加载预训练的 Replay Buffer
    # =====================================================================================
    if pretrained_buffer_path:
        print(f"\n--- [阶段二] 开始: 加载来自 '{pretrained_buffer_path}' 的经验池 ---")
        # 检查文件是否存在
        if not os.path.exists(pretrained_buffer_path):
            raise FileNotFoundError(
                f"预训练的经验池文件未找到: {pretrained_buffer_path}"
            )
        model.load_replay_buffer(pretrained_buffer_path)

        # --- 这里是修改的部分 ---

        # 1. 获取我们想要记录的信息
        buffer_size = model.replay_buffer.size()

        # 2. 打印到终端（可以保留，方便偶尔查看）
        print(f"经验池加载成功! 当前大小: {buffer_size}")

        # 3. 定义日志文件名
        log_filename = "experiment_log.txt"

        # 4. 获取当前时间，让日志更清晰
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # 5. 准备要写入文件的日志信息
        log_message = f"[{timestamp}] 经验池加载成功! 当前大小: {buffer_size}\n"

        # 6. 使用 'a' (append) 模式打开文件并写入日志
        # 'with open(...)' 是一种安全的文件操作方式，可以确保文件被正确关闭
        try:
            with open(log_filename, "a", encoding="utf-8") as f:
                f.write(log_message)
            print(f"该信息已成功记录到 {log_filename}")
        except Exception as e:
            print(f"写入日志文件时发生错误: {e}")

    print("--- 开始在您的自定义无人机环境上训练 DiffusionSACAgent ---")
    model.learn(
        total_timesteps=total_timesteps,
        log_interval=10,
        callback=metric_callback,
    )

    import time

    # 确保目标文件夹存在
    timestamp = int(time.time())
    model_name = f"{base_experiment_name}_ue{ue_num}_seed{seed}_{timestamp}.zip"
    final_save_path = os.path.join(save_path, model_name)
    model.save(final_save_path)
    print(f"--- [DiffusionSAC] 模型已保存至: {final_save_path} ---")


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


def get_sweep_config() -> dict:
    """
    返回 Wandb Sweep 的配置字典。
    """
    sweep_configuration = {
        "name": "DiffusionSAC-UAV-Optimization-Sweep",  # 给这次扫描起个名字
        "method": "bayes",  # 使用贝叶斯优化，它比随机搜索更高效
        "metric": {
            "name": "rollout/ep_rew_mean",  # 优化的目标指标
            "goal": "maximize",  # 目标是最大化这个指标
        },
        "parameters": {
            # --- 学习率: Actor vs Critic ---
            "lr_actor": {
                "distribution": "log_uniform_values",
                "min": 1e-5,
                "max": 4e-4,
            },
            "lr_critic": {
                "distribution": "log_uniform_values",
                "min": 1e-5,
                "max": 4e-4,
            },
            # --- 算法核心超参数 ---
            "qne_temperature": {
                "distribution": "uniform",
                "min": 0.01,
                "max": 0.5,
            },
            # "learning_starts": {"values": [10000, 25000, 30000]},
            # --- 扩散模型特定参数 ---
            "T_steps": {
                "values": [5, 10, 20]  # 扩散步数
            },
            "qne_k_samples": {
                "values": [8, 16, 32]  # "头脑风暴"样本数
            },
            # --- 网络结构 ---先不考虑吧
            "net_arch_actor": {"values": [[256, 256], [512, 512]]},
            "net_arch_critic": {"values": [[256, 256], [512, 512]]},
        },
        "early_terminate": {  # 提前终止不佳的实验，节省资源
            "type": "hyperband",
            "min_iter": 100000,  # 至少跑完100k步再做判断
        },
    }
    return sweep_configuration


def train_for_sweep():
    """
    为Wandb Sweep执行单次训练的函数。
    它会自动从 wandb.config 中读取超参数。
    """
    # 1. 初始化Wandb run
    # name可以由Wandb自动生成，也可以自定义
    run = wandb.init(
        project="SAC-find_out-sweep", reinit=True, sync_tensorboard=True, save_code=True
    )

    # 2. 从 wandb.config 中提取超参数
    #    这里我们给每个参数提供一个默认值，以防万一
    config = wandb.config

    num_envs = 4
    total_timesteps = 100000  # 在sweep中，可以适当减少总步数以加快速度

    # 从config中获取学习率
    lr_actor_initial = config.get("lr_actor", 3e-4)
    lr_critic_initial = config.get("lr_critic", 3e-4)

    # 创建独立的学习率调度器
    lr_actor_schedule_sweep = linear_schedule(lr_actor_initial)
    lr_critic_schedule_sweep = linear_schedule(lr_critic_initial)

    # 封装policy_kwargs
    policy_kwargs = {
        "T": config.get("T_steps", 5),
        "net_arch": {
            "pi": config.get("net_arch_actor", [256, 256]),
            "qf": config.get("net_arch_critic", [256, 256]),
        },
        "lr_actor_schedule": lr_actor_schedule_sweep,
        "lr_critic_schedule": lr_critic_schedule_sweep,
    }

    # 封装wrapper_kwargs (如果需要的话)
    wrapper_kwargs = {"ue_embedding_dim": 8}  # 假设这个是固定的

    # 3. 创建环境 (这部分与 run_diffusion_sac 相同)
    log_path = os.path.join("logs", f"sweep-{run.id}")
    os.makedirs(log_path, exist_ok=True)

    env = make_vec_env(
        "UAVEnv-v1",
        n_envs=num_envs,
        wrapper_class=UAVEnvWrapper,
        wrapper_kwargs=wrapper_kwargs,
        vec_env_cls=SubprocVecEnv,
        seed=SEED,
    )

    # 4. 创建并训练Agent (与 run_diffusion_sac 类似)
    model = DiffusionSACAgent(
        policy="DiffusionSACPolicy",
        env=env,
        tensorboard_log=log_path,
        verbose=0,  # 在sweep中通常关闭详细日志
        learning_starts=config.get("learning_starts", 10000),
        qne_k_samples=config.get("qne_k_samples", 16),
        qne_temperature=config.get("qne_temperature", 5.0),
        policy_kwargs=policy_kwargs,
        # 注意: 主学习率 learning_rate 不再需要，因为它被policy_kwargs中的调度器覆盖了
    )

    # 创建回调函数
    # 在Sweep中，WandbCallback会自动处理所有事情
    # ParallelEpisodeMetricCallback 依然有用，因为它计算了我们需要的 ep_rew_mean
    metric_callback = ParallelEpisodeMetricCallback()
    wandb_callback = WandbCallback(
        # 你可以配置梯度和模型保存，如果不需要可以留空
        # model_save_path=f"models/{run.id}",
        verbose=2
    )
    callback_list = CallbackList([metric_callback, wandb_callback])

    print(f"--- Starting Sweep Run: {run.name} with config: {dict(config)} ---")

    try:
        model.learn(
            total_timesteps=total_timesteps,
            log_interval=100,  # 可以增加log间隔
            callback=callback_list,
        )
    finally:
        # 确保每个run都正确结束
        run.finish()


def test_model():
    # --- 自动查找并加载最新的模型 ---
    import os
    import re

    model_dir = os.path.join("models", "Vanilla-SAC")
    if not os.path.isdir(model_dir):
        print(f"错误：模型目录 '{model_dir}' 不存在。")
        return

    # 1. 找到所有符合 'sac_model_*.zip' 格式的文件
    model_files = [
        f
        for f in os.listdir(model_dir)
        if f.startswith("Vanilla-SAC_") and f.endswith(".zip")
    ]

    if not model_files:
        print(f"错误：在目录 '{model_dir}' 中没有找到任何模型文件。")
        return

    # 2. 从文件名中解析时间戳，并找到最新的一个
    latest_timestamp = -1
    latest_model_file = ""
    for filename in model_files:
        # 使用正则表达式从 'sac_model_1751615810.zip' 中提取数字
        match = re.search(r"Vanilla-SAC_(\d+)\.zip", filename)
        if match:
            timestamp = int(match.group(1))
            if timestamp > latest_timestamp:
                latest_timestamp = timestamp
                latest_model_file = filename

    latest_model_path = os.path.join(model_dir, latest_model_file)

    # 加载模型
    model = SAC.load(latest_model_path)
    # 创建环境
    env = gym.make("UAVEnv-v1", render_mode="human")
    env = SACWrapper(env)  # 使用SAC专用的包装器

    # 确保TimeLimit生效的两种方式：
    # 方式1：直接包装（推荐）
    env = TimeLimit(env, max_episode_steps=40)

    # 方式2：检查是否已被TimeLimit包装
    if not isinstance(env, TimeLimit):
        env = TimeLimit(env, max_episode_steps=40)

    # 运行N个episodes进行测试
    num_episodes = 5
    reward_list = []
    for episode in range(num_episodes):
        obs, _ = env.reset()
        done = False
        total_reward = 0
        while not done:
            # 使用deterministic模式测试
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(action)
            total_reward += reward
            # 若达到步数限制则结束
            if truncated:
                done = True
        reward_list.append(total_reward)
        print(f"Episode {episode + 1} finished. Total reward: {total_reward:.2f}")
        env.render()

    env.close()

    print(f"all episode complete.The mean reward is {statistics.mean(reward_list)}")


def test_diffusion_sac():
    import re

    model_dir = os.path.join("models", "Diffusion-SAC")
    if not os.path.isdir(model_dir):
        print(f"错误：模型目录 '{model_dir}' 不存在。")
        return

    # 找到所有符合命名格式的模型文件
    model_files = [
        f
        for f in os.listdir(model_dir)
        if f.startswith("Diffusion-SAC") and f.endswith(".zip")
    ]
    if not model_files:
        print(f"错误：在目录 '{model_dir}' 中没有找到任何模型文件。")
        return

    # 从文件名中提取时间戳，选择最新的文件
    latest_timestamp = -1
    latest_model_file = ""
    for filename in model_files:
        # 示例文件名格式: "Diffusion-SAC-UAV_1751615810.zip"
        match = re.search(r"Diffusion-SAC_(\d+)\.zip", filename)
        if match:
            timestamp = int(match.group(1))
            if timestamp > latest_timestamp:
                latest_timestamp = timestamp
                latest_model_file = filename

    latest_model_path = os.path.join(model_dir, latest_model_file)
    print(f"加载最新模型：{latest_model_path}")

    # 加载模型
    model = DiffusionSACAgent.load(latest_model_path)

    # 构造环境：使用 "UAVEnv-v1" 并包装为带 TimeLimit 的环境
    env = gym.make("UAVEnv-v1", render_mode="human")
    env = UAVEnvWrapper(env, ue_embedding_dim=8)  # 使用自定义包装器
    env = TimeLimit(env, max_episode_steps=40)

    num_episodes = 5
    for episode in range(num_episodes):
        obs, _ = env.reset()
        done = False
        total_reward = 0
        while not done:
            # 使用deterministic模式测试
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(action)
            total_reward += reward
            # 若达到步数限制则结束
            if truncated:
                done = True
        print(f"Episode {episode + 1} finished. Total reward: {total_reward:.2f}")
        env.render()
    env.close()


def test_diffusion_sac_two():
    import re

    model_dir = os.path.join("models", "Diffusion-SAC-TwoStage")
    if not os.path.isdir(model_dir):
        print(f"错误：模型目录 '{model_dir}' 不存在。")
        return

    # 找到所有符合命名格式的模型文件
    model_files = [
        f
        for f in os.listdir(model_dir)
        if f.startswith("Diffusion-SAC-TwoStage") and f.endswith(".zip")
    ]
    if not model_files:
        print(f"错误：在目录 '{model_dir}' 中没有找到任何模型文件。")
        return

    # 从文件名中提取时间戳，选择最新的文件
    latest_timestamp = -1
    latest_model_file = ""
    for filename in model_files:
        # 示例文件名格式: "Diffusion-SAC-UAV_1751615810.zip"
        match = re.search(r"Diffusion-SAC-TwoStage_(\d+)\.zip", filename)
        if match:
            timestamp = int(match.group(1))
            if timestamp > latest_timestamp:
                latest_timestamp = timestamp
                latest_model_file = filename

    latest_model_path = os.path.join(model_dir, latest_model_file)
    print(f"加载最新模型：{latest_model_path}")

    # 加载模型
    model = DiffusionSACAgent.load(latest_model_path)

    # 构造环境：使用 "UAVEnv-v1" 并包装为带 TimeLimit 的环境
    env = gym.make("UAVEnv-v1", render_mode="human")
    env = UAVEnvWrapper(env, ue_embedding_dim=8)  # 使用自定义包装器
    env = TimeLimit(env, max_episode_steps=40)

    num_episodes = 5
    reward_list = []
    for episode in range(num_episodes):
        obs, _ = env.reset()
        done = False
        total_reward = 0
        while not done:
            # 使用deterministic模式测试
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(action)
            total_reward += reward
            # 若达到步数限制则结束
            if truncated:
                done = True
        print(f"Episode {episode + 1} finished. Total reward: {total_reward:.2f}")
        reward_list.append(total_reward)
        env.render()
    env.close()
    print(f"all episode complete.The mean reward is {statistics.mean(reward_list)}")


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
    # swanlab.finish()
    import time

    # 确保目标文件夹存在
    timestamp = int(time.time())
    save_dir = os.path.join("hybridSAC_v3_model", "models")
    os.makedirs(save_dir, exist_ok=True)
    model.save(os.path.join(save_dir, f"hybrid_sac_model_{timestamp}"))



# HYDRA: 这是新的程序入口。Hydra 会自动调用这个函数。
@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    """
    Hydra 会自动解析配置文件和命令行参数，
    并将最终合并后的配置作为 `cfg` 对象传入此函数。
    """
    # --- HYDRA: 打印最终配置，非常便于调试 ---
    print("--- Hydra Configuration ---")
    print(OmegaConf.to_yaml(cfg))
    print("--------------------------")

    # --- HYDRA: 替换 --sweep 逻辑 ---
    if cfg.sweep:
        sweep_id = None
        project_name = "SAC-find_out-sweep"

        if cfg.sweep == "new":
            print("--- 启动一个新的 Wandb Sweep ---")
            sweep_config = get_sweep_config()
            sweep_id = wandb.sweep(sweep=sweep_config, project=project_name)
            print(f"新的 Sweep 已创建, ID: {sweep_id}")
        else:
            sweep_id = cfg.sweep
            print(f"--- 继续已有的 Sweep, ID: {sweep_id} ---")

        print(f"启动 Agent, 将执行 {cfg.count} 次实验...")
        wandb.agent(
            sweep_id,
            function=train_for_sweep,
            count=cfg.count,
            project=project_name,
        )
        return # Sweep结束后直接退出

    # --- HYDRA: 替换 --evaluate 逻辑 ---
    if cfg.evaluate:
        if cfg.evaluate == "sac":
            print("Running test_model...")
            test_model()
        elif cfg.evaluate == "diffusion":
            print("Running test_diffusion...")
            test_diffusion_sac()
        elif cfg.evaluate == "diffusion-two-stage":
            print("Running test_diffusion_two_stage...")
            test_diffusion_sac_two()
        elif cfg.evaluate == "all":
            print("Running test_model...")
            test_model()
            print("Running test_diffusion...")
            test_diffusion_sac()
            print("Running test_diffusion_two_stage...")
            test_diffusion_sac_two()
        else:
            print(f"Error: Invalid evaluate option '{cfg.evaluate}'")
        return # 评估结束后直接退出

    # --- HYDRA: 运行主实验逻辑 ---
    # `cfg.experiment` 包含了从 conf/experiment/ 目录加载的实验配置
    # `cfg.seed` 和 `cfg.ue` 来自主 config.yaml 或命令行覆盖

    run_experiment(cfg.experiment, cfg.seed, cfg.ue)




if __name__ == "__main__":
    # 1. 创建一个命令行参数解析器
    main()
 