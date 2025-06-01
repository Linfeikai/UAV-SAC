import torch
import torch.nn as nn
import gymnasium  # 完整导入 gymnasium

# from gymnasium import spaces # 或者直接用 gymnasium.spaces
import numpy as np

# 假设你的自定义组件可以被导入
from hybrid_critic import HybridCritic  # 被测试的类

# 从 SB3 导入相关组件
from stable_baselines3.common.torch_layers import (
    FlattenExtractor,
    BaseFeaturesExtractor,
)
from stable_baselines3.common.policies import BaseModel  # HybridCritic 继承自 BaseModel

# --- 0. 定义测试用的参数 ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# 定义观测空间和混合动作空间 (与 HybridActor 测试一致)
obs_shape = (4,)
observation_space = gymnasium.spaces.Box(
    low=-np.inf, high=np.inf, shape=obs_shape, dtype=np.float32
)

discrete_action_dim_env = 3  # 环境中的离散动作数量
continuous_action_dim_env = 2  # 环境中的连续动作维度
action_space = gymnasium.spaces.Tuple(
    (
        gymnasium.spaces.Discrete(discrete_action_dim_env),
        gymnasium.spaces.Box(
            low=-1, high=1, shape=(continuous_action_dim_env,), dtype=np.float32
        ),
    )
)
print(f"Test Observation Space: {observation_space}")
print(f"Test Action Space: {action_space}")

# Critic 参数
net_arch_critic = [64, 64]
activation_fn_critic = nn.ReLU
n_critics_val = 2
share_features_extractor_val = True  # 测试共享的情况

# 创建一个简单的特征提取器实例
# HybridCritic 的 __init__ 期望一个 features_extractor 实例
# 并且我们假设它会从中获取 features_dim (如果 __init__ 已按此修改)
features_extractor_instance_critic = FlattenExtractor(observation_space)
# features_dim_manual_critic = np.prod(obs_shape).astype(int) # 如果需要显式 features_dim

# 模拟一批观测数据和动作数据
batch_size = 5
example_obs_np_critic = np.array(
    [observation_space.sample() for _ in range(batch_size)], dtype=np.float32
)
example_obs_th_critic = torch.tensor(
    example_obs_np_critic, device=DEVICE, dtype=torch.float32
)

# 模拟动作元组 (离散动作是整数索引, 连续动作是浮点数)
example_discrete_actions_np = np.random.randint(
    0, discrete_action_dim_env, size=(batch_size,)
)
example_continuous_actions_np = np.random.uniform(
    -1, 1, size=(batch_size, continuous_action_dim_env)
).astype(np.float32)

example_discrete_actions_th = torch.tensor(
    example_discrete_actions_np, device=DEVICE, dtype=torch.long
)
example_continuous_actions_th = torch.tensor(
    example_continuous_actions_np, device=DEVICE, dtype=torch.float32
)
example_actions_tuple_th = (example_discrete_actions_th, example_continuous_actions_th)


# --- 1. 测试 HybridCritic 初始化 ---
print("\n--- 1. Testing HybridCritic Initialization ---")
hybrid_critic_instance = None
try:
    # 假设 HybridCritic 的 __init__ 遵循了从 features_extractor 获取 features_dim 的建议:
    hybrid_critic_instance = HybridCritic(
        observation_space=observation_space,
        action_space=action_space,
        net_arch=net_arch_critic,
        features_extractor=features_extractor_instance_critic.to(
            DEVICE
        ),  # 确保在正确设备
        # features_dim 参数已移除，会从 features_extractor 获取
        activation_fn=activation_fn_critic,
        n_critics=n_critics_val,
        share_features_extractor=share_features_extractor_val,
    )
    hybrid_critic_instance.to(DEVICE)  # 确保整个 critic 在设备上

    assert len(hybrid_critic_instance.q_networks) == n_critics_val, (
        "Incorrect number of Q networks."
    )
    for q_net in hybrid_critic_instance.q_networks:
        assert isinstance(q_net, nn.Sequential), "A Q network is not nn.Sequential."

    assert hybrid_critic_instance.discrete_action_dim == discrete_action_dim_env
    assert hybrid_critic_instance.one_hot_discrete_action_dim == discrete_action_dim_env
    assert hybrid_critic_instance.continuous_action_dim == continuous_action_dim_env
    assert (
        hybrid_critic_instance.share_features_extractor == share_features_extractor_val
    )
    # 检查 self.features_dim 是否被正确设置 (由 BaseModel 或 HybridCritic 自身)
    assert (
        hybrid_critic_instance.features_dim
        == features_extractor_instance_critic.features_dim
    ), "features_dim not set correctly from features_extractor."

    print("PASSED: HybridCritic Initialization")
except Exception as e:
    print(f"FAILED: HybridCritic Initialization - {e}")
    import traceback

    traceback.print_exc()

# --- 2. 测试 forward(obs, actions_tuple) ---
print("\n--- 2. Testing HybridCritic.forward ---")
critic_forward_passed = False
if hybrid_critic_instance:
    try:
        # 测试 share_features_extractor = True (默认)
        # 检查特征提取器参数的 requires_grad 状态 (比较难直接在 forward 中测，但可以确保代码运行)
        # 我们主要关注输出
        q_values_tuple = hybrid_critic_instance.forward(
            example_obs_th_critic, example_actions_tuple_th
        )

        assert isinstance(q_values_tuple, tuple), "Forward output is not a tuple."
        assert len(q_values_tuple) == n_critics_val, (
            f"Forward output tuple length mismatch. Expected {n_critics_val}, Got {len(q_values_tuple)}."
        )

        for i, q_val_tensor in enumerate(q_values_tuple):
            assert isinstance(q_val_tensor, torch.Tensor), (
                f"Q value {i} is not a Tensor."
            )
            assert q_val_tensor.shape == (batch_size, 1), (
                f"Q value {i} shape mismatch. Expected {(batch_size, 1)}, Got {q_val_tensor.shape}."
            )
            assert not torch.isnan(q_val_tensor).any(), f"Q value {i} contains NaN."
            assert not torch.isinf(q_val_tensor).any(), f"Q value {i} contains Inf."
        print("  forward output shape and type checks PASSED.")

        # 测试当 share_features_extractor = False 时 (需要重新创建一个 Critic 实例)
        # print("  Testing with share_features_extractor = False")
        # features_extractor_critic_separate = FlattenExtractor(observation_space)
        # critic_no_share = HybridCritic(
        #     observation_space=observation_space,
        #     action_space=action_space,
        #     net_arch=net_arch_critic,
        #     features_extractor=features_extractor_critic_separate.to(DEVICE),
        #     activation_fn=activation_fn_critic,
        #     n_critics=n_critics_val,
        #     share_features_extractor=False # 设置为 False
        # ).to(DEVICE)
        # q_values_no_share = critic_no_share.forward(example_obs_th_critic, example_actions_tuple_th)
        # # 类似地检查输出...
        # print("  forward (share_features_extractor=False) basic run PASSED.")
        # (这个部分的测试可以根据需要添加，主要确保在不同设置下都能运行)

        print("PASSED: HybridCritic.forward")
        critic_forward_passed = True
    except Exception as e:
        print(f"FAILED: HybridCritic.forward - {e}")
        import traceback

        traceback.print_exc()
else:
    print("SKIPPED: HybridCritic.forward (HybridCritic not initialized).")


# --- 3. 测试 q1_forward(obs, actions_tuple) ---
print("\n--- 3. Testing HybridCritic.q1_forward ---")
if hybrid_critic_instance and critic_forward_passed:
    try:
        q1_value = hybrid_critic_instance.q1_forward(
            example_obs_th_critic, example_actions_tuple_th
        )

        assert isinstance(q1_value, torch.Tensor), "q1_forward output is not a Tensor."
        assert q1_value.shape == (batch_size, 1), (
            f"q1_forward output shape mismatch. Expected {(batch_size, 1)}, Got {q1_value.shape}."
        )
        assert not torch.isnan(q1_value).any(), "q1_forward output contains NaN."
        assert not torch.isinf(q1_value).any(), "q1_forward output contains Inf."

        # 验证 q1_forward 的输出是否与 forward() 的第一个Q值相同
        q_values_from_forward = hybrid_critic_instance.forward(
            example_obs_th_critic, example_actions_tuple_th
        )
        assert torch.allclose(q1_value, q_values_from_forward[0], atol=1e-7), (
            "q1_forward output does not match the first Q value from forward()."
        )
        print("  q1_forward output matches forward()[0].")

        print("PASSED: HybridCritic.q1_forward")
    except Exception as e:
        print(f"FAILED: HybridCritic.q1_forward - {e}")
        import traceback

        traceback.print_exc()
else:
    print(
        "SKIPPED: HybridCritic.q1_forward (HybridCritic not initialized or forward test failed)."
    )

# --- 4. 测试 _get_constructor_parameters (如果实现了) ---
# print("\n--- 4. Testing HybridCritic._get_constructor_parameters ---")
# if hybrid_critic_instance:
#     try:
#         params = hybrid_critic_instance._get_constructor_parameters()
#         assert isinstance(params, dict), "_get_constructor_parameters did not return a dict."
#         # 检查是否包含了必要的参数，例如：
#         # assert "net_arch" in params # 如果 HybridCritic 自己保存
#         # assert "n_critics" in params
#         # assert "share_features_extractor" in params
#         # BasePolicy/BaseModel 保存的参数 (obs_space, action_space, normalize_images) 也应该在 super() 调用后存在
#         assert "observation_space" in params
#         assert "action_space" in params
#         print("PASSED: _get_constructor_parameters (basic check)")
#     except Exception as e:
#         print(f"FAILED: _get_constructor_parameters - {e}")
# else:
#    print("SKIPPED: _get_constructor_parameters (HybridCritic not initialized).")


print("\n--- HybridCritic Testing Complete ---")
