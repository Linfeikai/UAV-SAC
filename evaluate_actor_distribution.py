# evaluate_actor_distribution.py

import torch
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
import re
import yaml
import sys

# ==============================================================================
#                      从您的项目中导入必要的模块
# ==============================================================================
# ---【请根据您的项目结构，确认这些导入路径是否正确】---
from SAC_test.entities.custom_env import CustomEnv
from diffusion_sac.diffusion_sac_agent import DiffusionSACAgent
from diffusion_sac.diffusion_sac_policy import DiffusionSACPolicy
from main import (
    UAVEnvWrapper,
    linear_schedule,
)  # 从您的main.py导入Wrapper和schedule函数
# -------------------------------------------------------------

from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

plt.rcParams["font.sans-serif"] = ["WenQuanYi Zen Hei"]
plt.rcParams["axes.unicode_minus"] = False  # 解决负号'-'显示为方块的问题
print("已将 Matplotlib 字体设置为 SimHei")


def run_analysis(
    config_path: str,
    experiment_name: str,
    n_samples: int,
):
    """
    执行动作分布分析的核心函数。
    """
    print("--- 正在运行 Actor 分布评估 ---")

    # 1. --- 加载与实验相关的 YAML 配置 ---
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            all_experiments = yaml.safe_load(f)
    except FileNotFoundError:
        print(f"错误: 配置文件未找到于 {config_path}")
        sys.exit(1)

    experiment_config = None
    for exp in all_experiments:
        if exp.get("experiment_name") == experiment_name:
            experiment_config = exp
            break

    if not experiment_config:
        print(f"错误: 在 {config_path} 中未找到名为 '{experiment_name}' 的实验配置。")
        sys.exit(1)

    print(f"已加载实验 '{experiment_name}' 的配置。")
    wrapper_kwargs = experiment_config.get("wrapper_kwargs", {})
    policy_kwargs = experiment_config.get("policy_kwargs", {})

    # 为策略初始化创建虚拟的学习率调度器 (加载时需要)
    policy_kwargs["lr_actor_schedule"] = linear_schedule(1e-4)
    policy_kwargs["lr_critic_schedule"] = linear_schedule(1e-4)

    # 2. --- 自动查找最新的模型和环境统计文件 ---
    model_dir = os.path.join("models", experiment_name)
    if not os.path.isdir(model_dir):
        print(f"错误: 模型目录 '{model_dir}' 不存在。请先训练模型。")
        return

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
    stats_path = os.path.join(model_dir, "vec_normalize.pkl")
    print(f"找到最新的模型: {model_path}")

    # 3. --- 加载环境 (与您的 main.py 完全一致) ---
    env = DummyVecEnv([lambda: UAVEnvWrapper(CustomEnv(), **wrapper_kwargs)])

    if os.path.exists(stats_path):
        print(f"加载环境统计数据: {stats_path}")
        env = VecNormalize.load(stats_path, env)
        env.training = False
        env.norm_reward = False
    else:
        print("警告: 未找到环境统计文件 (vec_normalize.pkl)，将使用未归一化的环境。")

    # 4. --- 加载训练好的模型和创建未训练的模型 ---
    # SB3 加载自定义策略时，需要提供 custom_objects
    custom_objects = {
        "policy_class": DiffusionSACPolicy,
        "policy_kwargs": policy_kwargs,
    }
    trained_model = DiffusionSACAgent.load(
        model_path, env=env, custom_objects=custom_objects
    )
    trained_actor = trained_model.policy.actor
    print("训练好的 Actor 加载成功。")

    # --- 创建一个未经训练的 Actor 作为对比 ---
    # 我们严格按照您 DiffusionSACPolicy 的 __init__ 方法来提供所有必需的参数
    print("正在创建未经训练的 Actor 作为对比...")

    # 从已加载的训练模型中获取所有必要的组件和参数
    observation_space = trained_model.observation_space
    action_space = trained_model.action_space
    features_extractor = trained_model.policy.features_extractor
    # 从 policy_kwargs 或默认值中获取 net_arch
    net_arch = policy_kwargs.get("net_arch", [256, 256])
    # 创建一个虚拟的学习率调度器，因为 __init__ 需要它
    lr_schedule_dummy = linear_schedule(1e-4)

    # 实例化一个全新的、未经训练的策略对象
    untrained_policy = DiffusionSACPolicy(
        observation_space=observation_space,
        action_space=action_space,
        lr_schedule=lr_schedule_dummy,  # 主 lr_schedule
        net_arch=net_arch,
        # features_extractor_class=type(features_extractor),
        # features_extractor_kwargs=features_extractor.state_dict(),
        # 传入您自定义的学习率调度器
        lr_actor_schedule=policy_kwargs["lr_actor_schedule"],
        lr_critic_schedule=policy_kwargs["lr_critic_schedule"],
        # 传入扩散模型特有的参数
        T=policy_kwargs.get("T", 5),
    )

    # 从这个新策略中获取 Actor，并确保它在正确的设备上
    untrained_actor = untrained_policy.actor.to(trained_model.device)
    print("未经训练的 Actor 创建成功。")

    # 5. --- 采样动作 ---
    obs = env.reset()
    obs_tensor = torch.as_tensor(obs).to(trained_model.device)

    # 从未经训练的Actor采样
    untrained_actions = []
    with torch.no_grad():
        untrained_actor.eval()
        for _ in range(n_samples):
            action = untrained_actor(obs_tensor, deterministic=False)
            untrained_actions.append(action.cpu().numpy().flatten())
    untrained_actions = np.array(untrained_actions)

    # 从训练好的Actor采样
    trained_actions = []
    with torch.no_grad():
        trained_actor.eval()
        for _ in range(n_samples):
            action = trained_actor(obs_tensor, deterministic=True)
            trained_actions.append(action.cpu().numpy().flatten())
    trained_actions = np.array(trained_actions)

    print(f"采样完成。共 {n_samples} 个动作点。")

    def visualize_decoded_actions(
        untrained_actions_flat, trained_actions_flat, wrapper_kwargs, n_samples
    ):
        """
        解码扁平动作，并可视化所有原始连续动作维度的两两配对关系。
        """
        import matplotlib.pyplot as plt
        import gymnasium as gym

        print("Decoding flat actions back to original continuous space...")

        # 创建一个临时的 Wrapper 实例，只为了使用它的 _decode_action 方法
        # 我们需要一个虚拟的环境实例来初始化它
        dummy_env = CustomEnv()
        action_decoder = UAVEnvWrapper(dummy_env, **wrapper_kwargs)

        # --- 解码动作 ---
        # _decode_action 返回 (discrete_action, continuous_action)
        # 我们只关心后面的 continuous_action 部分
        untrained_continuous = np.array(
            [action_decoder._decode_action(a)[1] for a in untrained_actions_flat]
        )
        trained_continuous = np.array(
            [action_decoder._decode_action(a)[1] for a in trained_actions_flat]
        )

        # 原始连续动作的维度名称
        continuous_action_names = ["Angle", "Velocity Ratio", "Offloading Ratio"]
        num_dims = len(continuous_action_names)

        # 定义我们要绘制的配对
        # 对于3个维度，有 C(3,2) = 3 种配对
        dims_to_plot = [(0, 1), (0, 2), (1, 2)]

        # --- 绘制训练前的分布图 ---
        fig_before, axes_before = plt.subplots(1, 3, figsize=(24, 7), squeeze=False)
        fig_before.suptitle(
            f"Before Training: Action Distribution (N={n_samples})", fontsize=16
        )

        for i, (dim1, dim2) in enumerate(dims_to_plot):
            ax = axes_before.flatten()[i]
            ax.scatter(
                untrained_continuous[:, dim1], untrained_continuous[:, dim2], alpha=0.5
            )
            ax.set_title(
                f"{continuous_action_names[dim1]} vs {continuous_action_names[dim2]}"
            )
            ax.set_xlabel(f"{continuous_action_names[dim1]}")
            ax.set_ylabel(f"{continuous_action_names[dim2]}")
            ax.grid(True)

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        output_filename_before = "decoded_dist_before_training.png"
        fig_before.savefig(output_filename_before)
        print(
            f"\n'Before Training' decoded distribution plot saved to: {output_filename_before}"
        )
        plt.close(fig_before)

        # --- 绘制训练后的分布图 ---
        fig_after, axes_after = plt.subplots(1, 3, figsize=(24, 7), squeeze=False)
        fig_after.suptitle(
            f"After Training: Action Distribution (N={n_samples})", fontsize=16
        )

        for i, (dim1, dim2) in enumerate(dims_to_plot):
            ax = axes_after.flatten()[i]
            ax.scatter(
                trained_continuous[:, dim1],
                trained_continuous[:, dim2],
                alpha=0.5,
                color="orange",
            )
            ax.set_title(
                f"{continuous_action_names[dim1]} vs {continuous_action_names[dim2]}"
            )
            ax.set_xlabel(f"{continuous_action_names[dim1]}")
            ax.set_ylabel(f"{continuous_action_names[dim2]}")
            ax.grid(True)

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        output_filename_after = "decoded_dist_after_training.png"
        fig_after.savefig(output_filename_after)
        print(
            f"'After Training' decoded distribution plot saved to: {output_filename_after}"
        )
        plt.close(fig_after)

        print("\nAnalysis complete. Please check the generated PNG image files.")

    visualize_decoded_actions(
        untrained_actions_flat=untrained_actions,
        trained_actions_flat=trained_actions,
        wrapper_kwargs=wrapper_kwargs,
        n_samples=n_samples,
    )


if __name__ == "__main__":
    # --- 设置命令行参数解析 ---
    parser = argparse.ArgumentParser(description="分析 Diffusion Actor 的动作分布。")
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
        help="要评估的实验名称 (来自 YAML 文件中的 experiment_name)。",
    )
    parser.add_argument(
        "--samples", type=int, default=200, help="为每个状态采样的动作数量。"
    )

    args = parser.parse_args()

    # --- 运行分析函数 ---
    run_analysis(
        config_path=args.config,
        experiment_name=args.name,
        n_samples=args.samples,
    )
