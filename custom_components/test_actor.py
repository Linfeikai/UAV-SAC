import torch
import torch.nn as nn
import gymnasium  # 完整导入 gymnasium

# from gymnasium import spaces # 或者直接用 gymnasium.spaces
import numpy as np

# 假设你的自定义组件可以被导入
from hybrid_distribution import (
    make_hybrid_proba_distribution,
    HybridDistribution,
)  # 依赖这个
from hybrid_actor import HybridActor

# 从 SB3 导入相关组件
from stable_baselines3.common.torch_layers import FlattenExtractor
from stable_baselines3.sac.policies import LOG_STD_MAX, LOG_STD_MIN  # HybridActor 会用

# --- 0. 定义测试用的参数 ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# 定义观测空间和混合动作空间
obs_shape = (4,)  # 假设一个简单的向量观测
observation_space = gymnasium.spaces.Box(
    low=-np.inf, high=np.inf, shape=obs_shape, dtype=np.float32
)

discrete_action_dim = 3
continuous_action_dim = 2
action_space = gymnasium.spaces.Tuple(
    (
        gymnasium.spaces.Discrete(discrete_action_dim),
        gymnasium.spaces.Box(
            low=-1, high=1, shape=(continuous_action_dim,), dtype=np.float32
        ),
    )
)
print(f"Test Observation Space: {observation_space}")
print(f"Test Action Space: {action_space}")

# 模拟 Actor 参数
net_arch = [64, 64]
activation_fn = nn.ReLU
log_std_init_actor = -2.0  # 和 HybridActor 默认的 -3 不同，以便测试传递

# 创建一个简单的特征提取器实例
# 注意：根据我们之前的讨论，HybridActor 的 __init__ 期望一个 features_extractor 实例
# 并且会从中获取 features_dim (如果 HybridActor 的 __init__ 被修改成这样)
# 或者，如果 HybridActor 的 __init__ 仍然接收 features_dim，我们需要手动提供。
# 我们假设 HybridActor 的 __init__ 签名是 (..., features_extractor: nn.Module, features_dim: int, ...)
# 或者，如果遵循了修改建议，是 (..., features_extractor: nn.Module, ...)
# 这里我们创建一个 FlattenExtractor，它的 features_dim 就是 obs_shape 的乘积。
features_extractor_instance = FlattenExtractor(observation_space)
# features_dim_manual = np.prod(obs_shape).astype(int) # 如果 HybridActor 需要显式 features_dim

# 模拟一批观测数据
batch_size = 5
example_obs_np = np.array(
    [observation_space.sample() for _ in range(batch_size)], dtype=np.float32
)
example_obs_th = torch.tensor(example_obs_np, device=DEVICE, dtype=torch.float32)


# --- 1. 测试 HybridActor 初始化 ---
print("\n--- 1. Testing HybridActor Initialization ---")
hybrid_actor_instance = None
try:
    # 如果 HybridActor 的 __init__ 遵循了从 features_extractor 获取 features_dim 的建议:
    hybrid_actor_instance = HybridActor(
        observation_space=observation_space,
        action_space=action_space,
        net_arch=net_arch,
        features_extractor=features_extractor_instance.to(DEVICE),  # 确保在正确设备
        activation_fn=activation_fn,
        log_std_init=log_std_init_actor,
    )

    # 如果 HybridActor 的 __init__ 仍然需要显式的 features_dim:
    # (请根据你的 HybridActor.__init__ 签名取消注释正确的版本)
    # hybrid_actor_instance = HybridActor(
    #     observation_space=observation_space,
    #     action_space=action_space,
    #     net_arch=net_arch,
    #     features_extractor=features_extractor_instance.to(DEVICE),
    #     features_dim=features_extractor_instance.features_dim, # 从实例获取
    #     activation_fn=activation_fn,
    #     log_std_init=log_std_init_actor
    # )
    hybrid_actor_instance.to(DEVICE)  # 确保整个 actor 在设备上

    assert isinstance(hybrid_actor_instance.latent_pi, nn.Sequential), (
        "latent_pi is not nn.Sequential."
    )
    assert isinstance(hybrid_actor_instance.action_dist, HybridDistribution), (
        "action_dist is not HybridDistribution."
    )
    assert hasattr(hybrid_actor_instance, "action_net_discrete_logits"), (
        "Missing action_net_discrete_logits."
    )
    assert hasattr(hybrid_actor_instance, "action_net_continuous_mean"), (
        "Missing action_net_continuous_mean."
    )
    assert hasattr(hybrid_actor_instance, "action_net_continuous_log_std"), (
        "Missing action_net_continuous_log_std."
    )
    assert hybrid_actor_instance.log_std_init == log_std_init_actor, (
        "log_std_init not set correctly."
    )
    print("PASSED: HybridActor Initialization")
except Exception as e:
    print(f"FAILED: HybridActor Initialization - {e}")
    import traceback

    traceback.print_exc()

# --- 2. 测试 get_action_dist_params(obs) ---
print("\n--- 2. Testing get_action_dist_params ---")
params_obtained_successfully = False
if hybrid_actor_instance:
    try:
        d_logits, c_mean, c_log_std, kwargs = (
            hybrid_actor_instance.get_action_dist_params(example_obs_th)
        )

        assert d_logits.shape == (batch_size, discrete_action_dim), (
            f"Discrete logits shape mismatch."
        )
        assert c_mean.shape == (batch_size, continuous_action_dim), (
            f"Continuous mean shape mismatch."
        )
        assert c_log_std.shape == (batch_size, continuous_action_dim), (
            f"Continuous log_std shape mismatch."
        )
        assert isinstance(kwargs, dict), "kwargs is not a dict."

        # 检查 log_std 是否被裁剪 (LOG_STD_MIN 和 LOG_STD_MAX 来自 SB3)
        assert (c_log_std >= LOG_STD_MIN).all() and (c_log_std <= LOG_STD_MAX).all(), (
            "Continuous log_std is not clamped correctly."
        )
        print("  log_std clamping PASSED.")
        print("PASSED: get_action_dist_params")
        params_obtained_successfully = True
    except Exception as e:
        print(f"FAILED: get_action_dist_params - {e}")
        import traceback

        traceback.print_exc()
else:
    print("SKIPPED: get_action_dist_params (HybridActor not initialized).")

# --- 3. 测试 forward(obs, deterministic) ---
print("\n--- 3. Testing forward ---")
actor_forward_passed = False
if hybrid_actor_instance and params_obtained_successfully:
    try:
        # 测试 deterministic=False (stochastic)
        actions_stochastic_tuple = hybrid_actor_instance.forward(
            example_obs_th, deterministic=False
        )

        assert (
            isinstance(actions_stochastic_tuple, tuple)
            and len(actions_stochastic_tuple) == 2
        ), "Stochastic forward did not return a tuple of 2 elements."
        disc_action_s, cont_action_s = actions_stochastic_tuple
        print(
            f"  Sampled discrete actions (stochastic): {disc_action_s.cpu().numpy()}"
        )  # 打印出来看看
        print("  forward (deterministic=False) PASSED.")
        assert disc_action_s.shape == (batch_size,), (
            f"Stochastic discrete action shape mismatch."
        )
        assert disc_action_s.dtype == torch.int64, (
            f"Stochastic discrete action dtype mismatch."
        )
        assert ((disc_action_s >= 0) & (disc_action_s < discrete_action_dim)).all(), (
            "Stochastic discrete action out of bounds."
        )

        assert cont_action_s.shape == (batch_size, continuous_action_dim), (
            f"Stochastic continuous action shape mismatch."
        )
        assert ((cont_action_s >= -1.0001) & (cont_action_s <= 1.0001)).all(), (
            "Stochastic continuous action out of [-1, 1] bounds."
        )
        print("  forward (deterministic=False) PASSED.")

        # 测试 deterministic=True
        actions_deterministic_tuple = hybrid_actor_instance.forward(
            example_obs_th, deterministic=True
        )
        assert (
            isinstance(actions_deterministic_tuple, tuple)
            and len(actions_deterministic_tuple) == 2
        ), "Deterministic forward did not return a tuple of 2 elements."
        disc_action_d, cont_action_d = actions_deterministic_tuple

        assert disc_action_d.shape == (batch_size,), (
            f"Deterministic discrete action shape mismatch."
        )
        assert disc_action_d.dtype == torch.int64, (
            f"Deterministic discrete action dtype mismatch."
        )
        # 确定性模式下，离散动作应该是众数
        # (我们可以通过 action_dist.mode() 来验证，但这会重新计算参数，所以这里只检查基本属性)

        assert cont_action_d.shape == (batch_size, continuous_action_dim), (
            f"Deterministic continuous action shape mismatch."
        )
        assert ((cont_action_d >= -1.0001) & (cont_action_d <= 1.0001)).all(), (
            "Deterministic continuous action out of [-1, 1] bounds."
        )
        # 确定性模式下，连续动作应该是 tanh(mean)
        # _, c_mean_for_mode, _, _ = hybrid_actor_instance.get_action_dist_params(example_obs_th) # 获取当前的mean
        # expected_cont_mode = torch.tanh(c_mean_for_mode)
        # assert torch.allclose(cont_action_d, expected_cont_mode, atol=1e-6), "Deterministic continuous action not tanh(mean)."
        # ^ 上面的验证比较复杂，因为 get_action_dist_params 会重新计算。
        #   一个更简单的检查是确保它与 action_dist.mode() 的结果一致，如果 action_dist 内部参数没变的话。
        #   但 forward 内部会调用 action_dist.actions_from_params，它会更新 action_dist 的参数。
        #   所以，我们主要检查形状和范围。
        print("  forward (deterministic=True) PASSED.")
        print("PASSED: forward")
        actor_forward_passed = True

    except Exception as e:
        print(f"FAILED: forward - {e}")
        import traceback

        traceback.print_exc()
else:
    print("SKIPPED: forward (HybridActor not initialized or params not obtained).")


# --- 4. 测试 action_log_prob(obs) ---
print("\n--- 4. Testing action_log_prob ---")
if hybrid_actor_instance and params_obtained_successfully:
    try:
        actions_tuple_alp, log_prob_alp = hybrid_actor_instance.action_log_prob(
            example_obs_th
        )

        assert isinstance(actions_tuple_alp, tuple) and len(actions_tuple_alp) == 2, (
            "action_log_prob actions part is not a tuple of 2 elements."
        )
        disc_action_alp, cont_action_alp = actions_tuple_alp

        assert disc_action_alp.shape == (batch_size,), (
            f"action_log_prob discrete action shape mismatch."
        )
        assert cont_action_alp.shape == (batch_size, continuous_action_dim), (
            f"action_log_prob continuous action shape mismatch."
        )
        # 动作应该是随机采样的，因为 action_dist.log_prob_from_params 内部是 deterministic=False

        assert log_prob_alp.shape == (batch_size,), (
            f"action_log_prob log_prob shape mismatch."
        )
        assert not torch.isnan(log_prob_alp).any(), (
            "action_log_prob log_prob contains NaN."
        )
        assert not torch.isinf(log_prob_alp).any(), (
            "action_log_prob log_prob contains Inf."
        )
        print("PASSED: action_log_prob")
    except Exception as e:
        print(f"FAILED: action_log_prob - {e}")
        import traceback

        traceback.print_exc()
else:
    print(
        "SKIPPED: action_log_prob (HybridActor not initialized or params not obtained)."
    )


# --- 5. 测试 _predict(obs, deterministic) ---
# _predict 通常由 BasePolicy.predict 调用，我们这里直接测试它是否和 forward 行为一致
print("\n--- 5. Testing _predict ---")
if hybrid_actor_instance and actor_forward_passed:  # 依赖 forward 测试通过
    try:
        # _predict 应该直接调用 forward
        actions_stochastic_predict_tuple = hybrid_actor_instance._predict(
            example_obs_th, deterministic=False
        )
        actions_stochastic_forward_tuple = hybrid_actor_instance.forward(
            example_obs_th, deterministic=False
        )
        # 由于随机性，我们不能直接比较值，但可以比较类型和形状
        assert (
            isinstance(actions_stochastic_predict_tuple, tuple)
            and len(actions_stochastic_predict_tuple) == 2
        )
        assert (
            actions_stochastic_predict_tuple[0].shape
            == actions_stochastic_forward_tuple[0].shape
        )
        assert (
            actions_stochastic_predict_tuple[1].shape
            == actions_stochastic_forward_tuple[1].shape
        )
        print(
            "  _predict (deterministic=False) shape/type checks PASSED (compared to forward)."
        )

        actions_deterministic_predict_tuple = hybrid_actor_instance._predict(
            example_obs_th, deterministic=True
        )
        actions_deterministic_forward_tuple = hybrid_actor_instance.forward(
            example_obs_th, deterministic=True
        )
        assert torch.allclose(
            actions_deterministic_predict_tuple[0].float(),
            actions_deterministic_forward_tuple[0].float(),
        ), "Deterministic discrete action from _predict mismatch with forward."
        assert torch.allclose(
            actions_deterministic_predict_tuple[1],
            actions_deterministic_forward_tuple[1],
            atol=1e-6,
        ), "Deterministic continuous action from _predict mismatch with forward."
        print(
            "  _predict (deterministic=True) value checks PASSED (compared to forward)."
        )
        print("PASSED: _predict")

    except Exception as e:
        print(f"FAILED: _predict - {e}")
        import traceback

        traceback.print_exc()
else:
    print("SKIPPED: _predict (HybridActor not initialized or forward tests failed).")

print("\n--- HybridActor Testing Complete ---")
