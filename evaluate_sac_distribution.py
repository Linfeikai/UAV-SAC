# evaluate_sac_distribution.py

import torch
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
import re
import yaml
import sys
import gymnasium as gym

# ==============================================================================
#                      从您的项目中导入必要的模块
# ==============================================================================
# ---【请根据您的项目结构，确认这些导入路径是否正确】---
from SAC_test.entities.custom_env import CustomEnv
from main import SACWrapper  # 从您的main.py导入SAC专用的Wrapper
# -------------------------------------------------------------

from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.evaluation import evaluate_policy


def run_sac_analysis(config_path: str, experiment_name: str, n_samples: int):
    """
    为原生SAC模型执行动作分布分析的核心函数。
    """
    print(f"--- 正在为原生SAC实验 '{experiment_name}' 运行分析 ---")

    # 1. --- 自动查找最新的模型 ---
    model_dir = os.path.join("models", experiment_name)
    if not os.path.isdir(model_dir):
        print(f"错误: 模型目录 '{model_dir}' 不存在。请先运行原生SAC训练。")
        return

    # 查找符合 'Vanilla-SAC_*.zip' 格式的文件 (根据您的 test_model 函数)
    model_files = [
        f
        for f in os.listdir(model_dir)
        if f.startswith(experiment_name) and f.endswith(".zip")
    ]
    if not model_files:
        print(f"错误: 在目录 '{model_dir}' 中没有找到任何模型文件。")
        return

    latest_timestamp = -1
    latest_model_file = ""
    for filename in model_files:
        match = re.search(r"_(\d+)\.zip", filename)
        if match:
            timestamp = int(match.group(1))
            if timestamp > latest_timestamp:
                latest_timestamp = timestamp
                latest_model_file = filename

    model_path = os.path.join(model_dir, latest_model_file)
    print(f"找到最新的原生SAC模型: {model_path}")

    # 2. --- 创建环境 (与您的 run_vanilla_sac 保持一致) ---
    # 评估时，我们只需要一个实例，并且不需要 SubprocVecEnv
    # 注意：原生SAC使用的是 UAVEnv-v1
    env = SACWrapper(CustomEnv())

    # 3. --- 加载训练好的模型 ---
    trained_model = SAC.load(model_path, env=env)
    print("训练好的原生SAC模型加载成功。")

    # --- 创建一个未经训练的、随机初始化的模型作为对比 ---
    untrained_model = SAC("MlpPolicy", env)
    print("未经训练的原生SAC模型创建成功。")

    # 4. --- 采样动作 ---
    # 获取一个固定的状态 s
    # 注意：SACWrapper 已经处理了 Tuple -> Box 的转换
    obs, _ = env.reset()

    # 从未经训练的模型采样
    untrained_actions = []
    for _ in range(n_samples):
        # model.predict 返回的是 numpy 数组
        action, _ = untrained_model.predict(obs, deterministic=False)
        untrained_actions.append(action)
    untrained_actions = np.array(untrained_actions)

    # 从训练好的模型采样
    trained_actions = []
    for _ in range(n_samples):
        action, _ = trained_model.predict(obs, deterministic=False)
        trained_actions.append(action)
    trained_actions = np.array(trained_actions)

    print(f"采样完成。共 {n_samples} 个动作点。")

    # 5. --- 可视化 ---
    # 在SACWrapper中，动作的维度是 [ue_id, angle, velocity_ratio, offloading_ratio]
    # 我们关心的是后面三个连续动作维度
    continuous_action_names = ["Angle", "Velocity Ratio", "Offloading Ratio"]

    # 提取连续部分
    untrained_continuous = untrained_actions[:, 1:]
    trained_continuous = trained_actions[:, 1:]

    def visualize_sac_actions(actions, title, n_samples):
        fig, axes = plt.subplots(1, 3, figsize=(24, 7), squeeze=False)
        fig.suptitle(f"{title}: SAC Action Distribution (N={n_samples})", fontsize=16)

        dims_to_plot = [(0, 1), (0, 2), (1, 2)]

        for i, (dim1, dim2) in enumerate(dims_to_plot):
            ax = axes.flatten()[i]
            ax.scatter(actions[:, dim1], actions[:, dim2], alpha=0.5, color="green")
            ax.set_title(
                f"{continuous_action_names[dim1]} vs {continuous_action_names[dim2]}"
            )
            ax.set_xlabel(f"{continuous_action_names[dim1]}")
            ax.set_ylabel(f"{continuous_action_names[dim2]}")
            ax.grid(True)

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        output_filename = f"sac_dist_{title.replace(' ', '_').lower()}.png"
        fig.savefig(output_filename)
        print(f"'{title}' 分布图已保存到: {output_filename}")
        plt.close(fig)

    visualize_sac_actions(untrained_continuous, "Before Training", n_samples)
    visualize_sac_actions(trained_continuous, "After Training", n_samples)

    print("\nSAC 分析完成。请查看生成的 PNG 图片文件。")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="分析原生 SAC 模型的动作分布。")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="指向实验配置 YAML 文件的路径 (例如, experiments.yaml)。",
    )
    parser.add_argument(
        "--name",
        type=str,
        required=True,
        help="要评估的原生SAC实验名称 (例如, Vanilla-SAC)。",
    )
    parser.add_argument(
        "--samples", type=int, default=200, help="为每个状态采样的动作数量。"
    )

    args = parser.parse_args()

    run_sac_analysis(
        config_path=args.config, experiment_name=args.name, n_samples=args.samples
    )
