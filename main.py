from SAC_test.entities.custom_env import CustomEnv
import gymnasium as gym
import optuna
from optuna.visualization import plot_optimization_history, plot_param_importances
from gymnasium.envs.registration import register
from stable_baselines3.common.env_checker import check_env
from stable_baselines3 import SAC, TD3
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.evaluation import evaluate_policy

import wandb
from wandb.integration.sb3 import WandbCallback


import numpy as np
import torch
from typing import Dict, List, Optional

from gymnasium.wrappers import TimeLimit
import warnings

warnings.filterwarnings("ignore")


# 固定随机种子
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)


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


def find_best_hyperparameters():
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


def main():
    env1 = gym.make("UAVEnv-v0")
    env1.reset(seed=SEED)  # 设置随机种子以确保可重复性

    wandb.init(
        project="UAV-SAC_1",  # 项目名称（wandb 仪表盘中显示）
        name="experiment-2",  # 实验名称（可选）
        config={  # 记录超参数（可选）
            "policy": "MlpPolicy",
            "total_timesteps": 10000,
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
        total_timesteps=10000,
        callback=metric_callback,  # 显示进度条
        log_interval=10,  # 每10步打印一次日志
    )
    wandb.finish()

    # 保存模型

    import time

    timestamp = int(time.time())
    model.save(f"sac_uav_model_{timestamp}")  # 使用时间戳保存模型


def TD3_test():
    env1 = gym.make("UAVEnv-v0", seed=42)
    # env1.seed(42)  # 设置随机种子以确保可重复性

    metric_callback = EpisodeMetricCallback(verbose=1)
    progress_bar_callback = ProgressBarCallback()
    combined_callback = CallbackList(
        [
            metric_callback,
            progress_bar_callback,  # 显示进度条
        ]
    )

    # env1._get_obs()
    check_env(env1.unwrapped, skip_render_check=True)
    # 初始化 SAC 模型
    model = TD3(
        "MlpPolicy",  # 使用多层感知机策略
        env1,
        verbose=1,  # 打印训练日志
        tensorboard_log="./td3_logs",  # 保存日志用于TensorBoard可视化
        gamma=0.99,  # 折扣因子
        batch_size=256,  # 经验回放的批量大小
        learning_rate=3e-4,  # 学习率
    )

    # 训练模型（带进度条）
    model.learn(
        total_timesteps=100000,
        callback=combined_callback,  # 显示进度条
        log_interval=10,  # 每10步打印一次日志
    )

    # 保存模型

    model.save("sac_uav_model")


def test_model():
    # 加载模型
    model = SAC.load("sac_uav_model_1746280931")
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
    main()
    # test_model()
    # TD3_test()
    # find_best_hyperparameters()
