from SAC_test.entities.custom_env import CustomEnv
import gymnasium as gym
import optuna
from optuna.visualization import plot_optimization_history, plot_param_importances
from gymnasium.envs.registration import register
from stable_baselines3.common.env_checker import check_env
from stable_baselines3 import SAC, TD3
from stable_baselines3.common.callbacks import BaseCallback, CallbackList
from stable_baselines3.common.evaluation import evaluate_policy

import random
import wandb
from wandb.integration.sb3 import WandbCallback

import pandas as pd  # Optional, but helpful

import numpy as np
import torch
from typing import Dict, List, Optional

from gymnasium.wrappers import TimeLimit
import warnings

warnings.filterwarnings("ignore")


# 固定随机种子
SEED = 42


register(
    id="UAVEnv-v0",
    entry_point="SAC_test.entities.custom_env:CustomEnv",
    max_episode_steps=40,
)


class EpisodeMetricCallback(BaseCallback):
    def __init__(self, metrics_to_track: Optional[List[str]] = None, verbose: int = 0):
        """
        Args:
        metrics_to_track: List of metric names to extract from `info` (e.g., ["delay", "flying_energy"]).
                         If None, tracks all keys in `info`.

        """

        super().__init__(verbose)
        self.episode_counter = 0  # 记录episode数量
        self.metrics_to_track = metrics_to_track
        self.episode_metrics: Dict[str, List[float]] = {}  # 存储所有episode的均值
        self.current_episode_metrics: Dict[str, List[float]] = {}  # 当前episode的原始值

    def _on_step(self) -> bool:
        # 从info中提取信息
        info = self.locals["infos"][0]
        done = self.locals["dones"][0]

        # 首次运行时初始化字典
        # {} 是一个已经实例化的对象（有地址），只是内容为空。这种情况下if not还是会判断为true
        if not self.current_episode_metrics:
            self._init_metrics(info)

        for metric in self.current_episode_metrics.keys():
            self.current_episode_metrics[metric].append(info[metric])

        if done:
            self.episode_counter += 1  # 增加episode数量
            log_dict = {}  # 记录到wandb的字典
            for metric, values in self.current_episode_metrics.items():
                mean_value = np.mean(values) if values else 0.0
                self.episode_metrics[metric].append(mean_value)
                # self.logger.record(
                #     f"episode/{metric}", mean_value, self.episode_counter
                # )
                log_dict[f"episode/{metric}"] = mean_value  # 添加到 wandb 日志
                values.clear()  # 清空当前episode的值
                # self.current_episode_metrics[metric].clear()
            wandb.log(log_dict, step=self.episode_counter)  # 同步到 wandb

        return True

    def _init_metrics(self, info: Dict):
        if self.metrics_to_track is None:
            self.metrics_to_track = [
                k for k in info.keys() if not k.startswith("_")
            ]  # 排除内部字段

        for metric in self.metrics_to_track:
            self.episode_metrics[metric] = []  # 所有episode的均值
            self.current_episode_metrics[metric] = []  # 当前episode的原始值

    def get_metric(self, metric_name: str) -> List[float]:
        """获取某个指标的历史记录"""
        return self.episode_metrics.get(metric_name, [])

    def get_all_metrics(self) -> Dict[str, List[float]]:
        """获取所有指标的历史记录"""
        return self.episode_metrics


def objective(trial: optuna.Trial):
    params = {
        "learning_rate": trial.suggest_float("lr", 1e-5, 1e-3, log=True),
        "buffer_size": trial.suggest_categorical(
            "buffer_size", [50_000, 100_000, 1_000_000]
        ),
        "batch_size": trial.suggest_int("batch_size", 64, 512, step=64),
        "tau": trial.suggest_float("tau", 0.001, 0.1),
        "gamma": trial.suggest_float("gamma", 0.9, 0.9999),
        "ent_coef": trial.suggest_categorical("ent_coef", ["auto", 0.1, 0.2, 0.5]),
        "net_arch": trial.suggest_categorical("net_arch", ["small", "medium", "large"]),
    }

    net_arch_map = {
        "small": [64, 64],
        "medium": [256, 256],
        "large": [400, 300],
    }

    env = gym.make("UAVEnv-v0")  # 创建环境

    policy_kwargs = {"net_arch": net_arch_map[params.pop("net_arch")]}

    model = SAC(
        "MlpPolicy",
        env,
        verbose=0,
        seed=SEED,
        policy_kwargs=policy_kwargs,
        **params,
    )

    model.learn(total_timesteps=50_000)

    mean_reward, _ = evaluate_policy(model, env, n_eval_episodes=10, deterministic=True)

    del model
    env.close()  # 关闭环境

    return mean_reward


def find_best_hyperparameters_optuna():
    study = optuna.create_study(
        direction="maximize", sampler=optuna.samplers.TPESampler(seed=SEED)
    )
    study.enqueue_trial(
        {
            "lr": 3e-4,
            "buffer_size": 1_000_000,
            "batch_size": 256,
            "tau": 0.005,
            "gamma": 0.99,
            "ent_coef": "auto",
            "net_arch": "medium",
        }
    )
    study.optimize(objective, n_trials=10, timeout=600)  # 10次试验，超时600秒
    print(f"Best trial: {study.best_params}")
    print(f"Best value: {study.best_value}")
    # 可视化
    fig1 = plot_optimization_history(study)
    fig2 = plot_param_importances(study)
    fig1.show()
    fig2.show()


def SACtest():
    env1 = gym.make("UAVEnv-v0")
    env1.reset(seed=SEED)  # 设置随机种子以确保可重复性

    wandb.init(
        project="UAV-SAC_1",  # 项目名称（wandb 仪表盘中显示）
        name="experiment-SAC",  # 实验名称（可选）
        config={  # 记录超参数（可选）
            "policy": "MlpPolicy",
            "total_timesteps": 100000,
        },
        sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
    )

    metric_callback = EpisodeMetricCallback(verbose=1)

    # env1._get_obs()
    check_env(env1.unwrapped, skip_render_check=True)
    # 初始化 SAC 模型
    model = SAC(
        "MlpPolicy",  # 使用多层感知机策略
        env1,
        verbose=1,  # 打印训练日志
        tensorboard_log="./sac_logs",  # 保存日志用于TensorBoard可视化
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

    timestamp = int(time.time())
    model.save(f"sac_uav_model_{timestamp}")  # 使用时间戳保存模型


def TD3_test():
    print("当前模式:", "Sweep" if wandb.config else "普通训练")
    print("传入的 config:", wandb.config)  # 检查是否收到参数

    run = wandb.init(
        project="UAV-SAC_1",  # 项目名称（wandb 仪表盘中显示）
        name="experiment-TD3-2",  # 实验名称（可选）
        config=wandb.config,  # 接收sweep传入的参数
        sync_tensorboard=True,
        monitor_gym=True,  # 自动记录环境指标
    )

    env = gym.make("UAVEnv-v0")
    env.reset(seed=SEED)  # 设置随机种子以确保可重复性
    check_env(env.unwrapped, skip_render_check=True)

    # 使用config中的参数（sweep模式）或默认值（普通训练）
    params = {
        "gamma": wandb.config.gamma if wandb.config else 0.99,
        "batch_size": wandb.config.batch_size if wandb.config else 64,
        "learning_rate": wandb.config.learning_rate if wandb.config else 3e-4,
        "buffer_size": wandb.config.buffer_size if wandb.config else 300_000,
        "tau": wandb.config.tau if wandb.config else 0.003,
        "policy_delay": wandb.config.policy_delay if wandb.config else 4,
        "target_policy_noise": wandb.config.target_policy_noise
        if wandb.config
        else 0.3,
        "target_noise_clip": wandb.config.target_noise_clip if wandb.config else 0.5,
    }
    policy_kwargs = {}
    if wandb.config and hasattr(wandb.config, "net_arch"):
        policy_kwargs["net_arch"] = wandb.config.net_arch

    callbacks = [
        EpisodeMetricCallback(verbose=1),
    ]

    # 初始化 SAC 模型
    model = TD3(
        "MlpPolicy",  # 使用多层感知机策略
        env,
        verbose=1,  # 打印训练日志
        tensorboard_log=f"./TD3_logs/{run.id}",
        **params,
        policy_kwargs=policy_kwargs if policy_kwargs else None,
    )

    # 训练模型（带进度条）
    model.learn(
        total_timesteps=10000,
        callback=CallbackList(callbacks),  # 显示进度条
        log_interval=10,  # 每10步打印一次日志
    )

    # 保存模型
    model.save(f"TD3_uav_model_{run.id}")
    wandb.finish()


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


def TD3_useThebest():
    entity = "SACtest"
    # The name of your W&B project
    project = "UAV-TD3-Optimization"
    # The sweep ID you have (tn145nan)
    sweep_id = "zfinb4uk"
    # The metric you want to maximize (e.g., average episode reward)
    # Check your W&B run pages to confirm the exact name logged by SB3
    metric_to_optimize = "rollout/ep_rew_mean"
    api = wandb.Api()
    try:
        sweep = api.sweep(f"{entity}/{project}/{sweep_id}")
        print(f"Successfully accessed sweep: {sweep.name} ({sweep_id})")
    except Exception as e:
        print(f"Error accessing sweep {entity}/{project}/{sweep_id}: {e}")
        print("Please check your entity, project name, and sweep ID.")
        exit()  # Exit if sweep cannot be accessed

    print(f"Fetching runs for sweep...")

    best_run = None
    best_metric_value = -float("inf")  # Initialize for maximization

    runs_data = []  # To store data for potential DataFrame
    for run in sweep.runs:
        # Access summary metrics (usually contains the last value logged)
        summary = run.summary

        # Access configuration (hyperparameters)
        config = run.config

        # Check if the metric exists in the summary and is a number
        if metric_to_optimize in summary and isinstance(
            summary[metric_to_optimize], (int, float)
        ):
            metric_value = summary[metric_to_optimize]

            # print(f"Run {run.name} ({run.id}): {metric_to_optimize} = {metric_value}") # Uncomment to see each run's metric

            # Check if this run is better than the current best
            if metric_value > best_metric_value:
                best_metric_value = metric_value
                best_run = run

            # Store basic info and config
            runs_data.append(
                {
                    "run_id": run.id,
                    "run_name": run.name,
                    "metric_value": metric_value,
                    "config": config,
                }
            )
        # else:
        # print(f"Run {run.name} ({run.id}): Metric '{metric_to_optimize}' not found or not a number in summary.")

    if best_run:
        print("\n--- Best Run Found ---")
        print(f"Run Name: {best_run.name}")
        print(f"Run ID: {best_run.id}")
        print(f"Best {metric_to_optimize}: {best_metric_value}")
        print("\nHyperparameters (Config):")
        best_config = {}
        policy_kwargs = {}
        print(f"Policy Architecture: {policy_kwargs}")
        for key, value in best_run.config.items():
            # Skip wandb internal keys if necessary
            if not key.startswith("_"):
                if key == "net_arch":
                    policy_kwargs[key] = value
                    continue
                best_config[key] = value
                print(f"  {key}: {value}")

        if best_config:
            print("\n--- Training Final Model with Best Config ---")
            env = gym.make("UAVEnv-v0")
            env.reset(seed=SEED)  # 设置随机种子以确保可重复性
            check_env(env.unwrapped, skip_render_check=True)
            run = wandb.init(
                project="UAV-SAC_1",  # 项目名称（wandb 仪表盘中显示）
                name="experiment-TD3-2",
                sync_tensorboard=True,
            )  # 实验名称（可选）
            model = TD3(
                "MlpPolicy",
                env,
                verbose=1,
                tensorboard_log=f"./TD3_logs/{best_run.id}",
                **best_config,
                policy_kwargs=policy_kwargs if policy_kwargs else None,
            )
            model.learn(
                total_timesteps=100000,
                callback=EpisodeMetricCallback(verbose=1),
                log_interval=10,
            )
            model.save(f"TD3_uav_model_{best_run.id}")

        # The `best_config` dictionary now contains the hyperparameters
        # of the run that achieved the best value for your specified metric.

        # Optional: Create a DataFrame to inspect all runs' final metrics and config
    #     if runs_data:
    #         df = pd.DataFrame(runs_data)
    #         # Sort to see top runs
    #         df_sorted = df.sort_values(by="metric_value", ascending=False)
    #         print("\n--- Top 5 Runs by Metric ---")
    #         print(
    #             df_sorted[["run_name", "metric_value"]].head().to_markdown(index=False)
    #         )
    #         # df_sorted.to_csv("sweep_results.csv", index=False) # Save results to CSV

    # else:
    #     print(
    #         f"\nNo runs found in the sweep, or metric '{metric_to_optimize}' was not logged correctly in any run."
    #     )
    #     best_config = None  # Ensure best_config is None if no best run was found


def test_model():
    # 加载模型
    model = SAC.load("sac_uav_model_1746759492")
    # 创建环境
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


if __name__ == "__main__":
    # vv()
    # test_model()
    # TD3_test()
    TD3_useThebest()
    # find_best_hyperparameters()
    # find_best_hyperparameters_sweep()
