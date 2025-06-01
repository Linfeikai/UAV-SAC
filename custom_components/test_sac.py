import torch
import torch.nn as nn
import gymnasium as gym  # 使用 gymnasium
from gymnasium import spaces  # 确保从 gymnasium 导入 spaces
import numpy as np
import tempfile
import os
from typing import Dict, Any, Optional, Tuple, Union, Type, List

# 从 SB3 导入标准 SAC 和相关组件
from stable_baselines3 import SAC
from stable_baselines3.common.policies import BasePolicy
from stable_baselines3.common.buffers import ReplayBuffer  # SAC 使用 ReplayBuffer
from stable_baselines3.common.vec_env import DummyVecEnv, VecEnv
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.env_checker import check_env

# --- 1. 定义测试环境 (使用 Pendulum-v1，它有 Box 动作空间) ---
ENV_ID = "Pendulum-v1"
print(f"--- Using Standard Environment: {ENV_ID} ---")

# --- 2. 测试环境 (可选) ---
try:
    env_to_check = gym.make(ENV_ID)
    check_env(env_to_check, warn=True, skip_render_check=True)
    print("Standard environment check passed (or warnings printed).")
    env_to_check.close()
except Exception as e:
    print(f"Standard environment check FAILED: {e}")
    # exit() # 如果环境检查失败，可能无法继续

# --- 3. 定义测试参数 ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\nUsing device: {DEVICE}")

# 创建 VecEnv
vec_env: VecEnv = DummyVecEnv([lambda: gym.make(ENV_ID)])
print("DummyVecEnv created successfully.")
print(f"VecEnv Observation Space: {vec_env.observation_space}")
print(f"VecEnv Action Space: {vec_env.action_space}")  # 应该是 Box

# --- 4. 测试标准 SAC 算法初始化 ---
print("\n--- 4. Testing Standard SAC Algorithm Initialization ---")
agent_std_sac: Optional[SAC] = None
init_passed = False
try:
    agent_std_sac = SAC(
        policy="MlpPolicy",  # 标准策略
        env=vec_env,
        learning_starts=20,
        batch_size=4,
        buffer_size=200,
        verbose=1,
        device=DEVICE,
        seed=123,
        policy_kwargs=dict(net_arch=[32, 32]),
    )
    print(f"  Agent observation space: {agent_std_sac.observation_space}")
    print(f"  Agent action space: {agent_std_sac.action_space}")
    assert isinstance(agent_std_sac.action_space, spaces.Box), (
        "Standard SAC action space is not Box."
    )
    assert isinstance(
        agent_std_sac.policy, BasePolicy
    )  # MlpPolicy 继承自 BasePolicy (通常是 ActorCriticPolicy)
    assert isinstance(agent_std_sac.replay_buffer, ReplayBuffer)

    # 检查 _logger 属性是否存在以及初始值
    if hasattr(agent_std_sac, "_logger"):
        print(
            f"  Standard SAC agent has '_logger' attribute. Value: {agent_std_sac._logger}"
        )
    else:
        print(
            "  CRITICAL: Standard SAC agent does NOT have '_logger' attribute after __init__!"
        )
        # 如果这里 _logger 就不存在，说明 BaseAlgorithm.__init__ 没有被正确执行
        # 或者 SB3 的结构与我们假设的有所不同

    print("PASSED: Standard SAC Algorithm Initialization")
    init_passed = True
except Exception as e:
    print(f"FAILED: Standard SAC Algorithm Initialization - {e}")
    import traceback

    traceback.print_exc()

# --- 5. 测试数据收集 ---
print("\n--- 5. Testing Data Collection (Standard SAC) ---")
collection_passed = False
if agent_std_sac and init_passed:
    try:
        num_collection_steps = 30
        obs = agent_std_sac.env.reset()
        for step in range(num_collection_steps):
            action_np_batch, _ = agent_std_sac.predict(obs, deterministic=False)
            # predict 返回的是 NumPy 数组 (n_envs, action_dim)
            # VecEnv.step 需要一个动作列表或数组
            new_obs, rewards, dones, infos = agent_std_sac.env.step(action_np_batch)
            agent_std_sac.replay_buffer.add(
                obs, new_obs, action_np_batch, rewards, dones, infos
            )
            obs = new_obs
            for idx, done in enumerate(dones):
                if done:
                    if agent_std_sac.n_envs == 1:
                        obs = agent_std_sac.env.reset()

        assert (
            agent_std_sac.replay_buffer.size() > 0 or agent_std_sac.replay_buffer.full
        ), "Replay buffer is empty after collection."
        print(
            f"  Replay buffer current filled size: {agent_std_sac.replay_buffer.pos if not agent_std_sac.replay_buffer.full else agent_std_sac.replay_buffer.buffer_size}"
        )
        print("PASSED: Data Collection (Standard SAC)")
        collection_passed = True
    except Exception as e:
        print(f"FAILED: Data Collection (Standard SAC) - {e}")
        import traceback

        traceback.print_exc()
else:
    print("SKIPPED: Data Collection (Standard SAC Agent not initialized).")

# --- 6. 测试 train() 方法 (在 learn() 之前调用) ---
print("\n--- 6. Testing Standard SAC.train() method (before learn()) ---")
train_passed = False
if (
    agent_std_sac
    and collection_passed
    and agent_std_sac.replay_buffer.size() >= agent_std_sac.batch_size
):
    try:
        print(
            f"  Attempting {agent_std_sac.gradient_steps * 5} gradient steps with batch_size={agent_std_sac.batch_size}..."
        )

        # 关键：在调用 train 之前，检查 _logger 状态，并尝试手动配置（如果需要）
        print(
            f"  Before train(), agent_std_sac._logger is: {getattr(agent_std_sac, '_logger', 'Attribute NOT FOUND')}"
        )
        if hasattr(agent_std_sac, "_logger") and agent_std_sac._logger is None:
            if not getattr(
                agent_std_sac, "_custom_logger", False
            ):  # 检查是否已设置自定义logger
                from stable_baselines3.common.logger import configure

                print(
                    "  Manually configuring logger for standard SAC pre-learn train() test..."
                )
                agent_std_sac._logger = configure(
                    folder=agent_std_sac.tensorboard_log, format_strings=["stdout"]
                )  # 简化输出

        agent_std_sac.train(
            gradient_steps=agent_std_sac.gradient_steps * 5,
            batch_size=agent_std_sac.batch_size,
        )
        print(
            "PASSED: Standard SAC.train() ran without crashing (or with manual logger setup if _logger was None)."
        )
        train_passed = True
    except AttributeError as ae:
        print(f"FAILED: Standard SAC.train() - AttributeError: {ae}")
        if "_logger" in str(ae):
            print(
                "      This indicates '_logger' attribute was missing OR was None and record was called."
            )
        import traceback

        traceback.print_exc()
    except Exception as e:
        print(f"FAILED: Standard SAC.train() - Other Exception: {e}")
        import traceback

        traceback.print_exc()
elif agent_std_sac and collection_passed:
    print(
        f"SKIPPED: Standard SAC.train() (Not enough data in buffer: {agent_std_sac.replay_buffer.size()} < {agent_std_sac.batch_size})"
    )
else:
    print("SKIPPED: Standard SAC.train() (Agent not initialized or collection failed).")

# --- 7. 测试 learn() 方法 (短时间) ---
print("\n--- 7. Testing Standard SAC.learn() method (short run) ---")
if (
    agent_std_sac
):  # 即使 train() 测试部分失败，也尝试 learn()，因为 learn() 会设置 logger
    try:
        print("  Attempting to learn for 100 timesteps...")
        agent_std_sac.learn(total_timesteps=100, log_interval=2)
        print("PASSED: Standard SAC.learn() ran without crashing.")

        print("  Evaluating policy after short learning...")
        eval_env_std_sac = DummyVecEnv([lambda: gym.make(ENV_ID)])
        mean_reward, std_reward = evaluate_policy(
            agent_std_sac.policy,
            eval_env_std_sac,
            n_eval_episodes=2,
            warn=False,
            deterministic=True,
        )
        print(
            f"  Mean reward after learning (deterministic): {mean_reward:.2f} +/- {std_reward:.2f}"
        )
        eval_env_std_sac.close()

    except Exception as e:
        print(f"FAILED: Standard SAC.learn() - {e}")
        import traceback

        traceback.print_exc()
else:
    print("SKIPPED: Standard SAC.learn() (Agent not initialized).")

if vec_env:
    vec_env.close()

print("\n--- Standard SB3 SAC Testing Complete ---")
