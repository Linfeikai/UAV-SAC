import torch
import torch.nn as nn
import gymnasium  # 完整导入 gymnasium

# from gymnasium import spaces # 或者直接用 gymnasium.spaces
import numpy as np
import tempfile
import os
from typing import Dict, Any, Optional, Tuple, Union  # 确保导入

# 假设你的自定义组件可以被导入
from hybrid_sac_policy import HybridSACPolicy  # 被测试的类
from hybrid_actor import HybridActor
from hybrid_critic import HybridCritic
# from hybrid_distribution import HybridDistribution

# 从 SB3 导入相关组件
from stable_baselines3.common.torch_layers import (
    FlattenExtractor,
    BaseFeaturesExtractor,
)
from stable_baselines3.common.utils import get_schedule_fn
from stable_baselines3.common.type_aliases import Schedule  # 确保导入 Schedule

# --- 0. 定义测试用的参数 ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# 定义观测空间
obs_shape = (4,)
observation_space = gymnasium.spaces.Box(
    low=-np.inf, high=np.inf, shape=obs_shape, dtype=np.float32
)
print(f"Test Observation Space: {observation_space}")

# Policy 参数
lr_schedule_fn: Schedule = get_schedule_fn(1e-3)
net_arch_policy: Optional[Union[list[int], Dict[str, list[int]]]] = [
    32,
    32,
]  # 使用小一点的网络以便快速测试
activation_fn_policy: type[nn.Module] = nn.ReLU
log_std_init_policy: float = -2.0  # Actor 的 log_std 初始化
features_extractor_class_policy: type[BaseFeaturesExtractor] = FlattenExtractor
features_extractor_kwargs_policy: Optional[Dict[str, Any]] = None
n_critics_policy: int = 2
share_features_extractor_policy: bool = True
optimizer_class_policy: type[torch.optim.Optimizer] = torch.optim.Adam
optimizer_kwargs_policy: Optional[Dict[str, Any]] = None

# 定义动作空间维度常量，以便在脚本中一致使用
DISCRETE_ACTION_DIM = 3
CONTINUOUS_ACTION_DIM = 2


def create_policy_for_action_space(
    action_space_to_test: gymnasium.spaces.Tuple,
) -> Optional[HybridSACPolicy]:
    """辅助函数：创建并返回一个 HybridSACPolicy 实例"""
    print(f"\n--- Creating Policy for Action Space: {action_space_to_test} ---")
    policy = None
    try:
        policy = HybridSACPolicy(
            observation_space=observation_space,
            action_space=action_space_to_test,  # 使用传入的动作空间
            lr_schedule=lr_schedule_fn,
            net_arch=net_arch_policy,
            activation_fn=activation_fn_policy,
            log_std_init=log_std_init_policy,
            features_extractor_class=features_extractor_class_policy,
            features_extractor_kwargs=features_extractor_kwargs_policy,
            optimizer_class=optimizer_class_policy,
            optimizer_kwargs=optimizer_kwargs_policy,
            n_critics=n_critics_policy,
            share_features_extractor=share_features_extractor_policy,
        )
        policy.to(DEVICE)
        print("Policy instance created successfully.")
        return policy
    except Exception as e:
        print(
            f"FAILED to create Policy instance for action space {action_space_to_test}: {e}"
        )
        import traceback

        traceback.print_exc()
        return None


def test_predict_method(
    policy: HybridSACPolicy,
    test_action_space: gymnasium.spaces.Tuple,
    test_obs_np_single: np.ndarray,
    test_obs_np_batch: np.ndarray,
    is_unscaled_env: bool,  # 标记环境是否需要反向缩放 (即Box不是[-1,1])
):
    """辅助函数：测试给定策略的 predict 方法"""
    if policy is None:
        print("SKIPPING predict test as policy instance is None.")
        return

    print(
        f"\n--- Testing predict() for Action Space: {test_action_space} (Unscaled Env: {is_unscaled_env}) ---"
    )

    # 提取动作空间的细节
    # discrete_dim = test_action_space.spaces[0].n # 使用全局常量
    # continuous_dim = test_action_space.spaces[1].shape[0] # 使用全局常量
    discrete_dim = DISCRETE_ACTION_DIM
    continuous_dim = CONTINUOUS_ACTION_DIM
    continuous_low = test_action_space.spaces[1].low
    continuous_high = test_action_space.spaces[1].high

    # --- 测试 predict() with single observation ---
    try:
        print("  Testing predict() with single observation (deterministic=False)...")
        (disc_action_s_np, cont_action_s_np), _ = policy.predict(
            test_obs_np_single, deterministic=False
        )

        assert isinstance(disc_action_s_np, np.ndarray), (
            "Single obs, stochastic discrete action is not np.ndarray."
        )
        # 离散动作对于单个观测，squeeze后应该是标量或 () 形的数组
        assert (
            disc_action_s_np.ndim == 0
            or disc_action_s_np.shape == ()
            or disc_action_s_np.shape == (1,)
        ), (
            f"Single obs, stochastic discrete action shape mismatch: {disc_action_s_np.shape}"
        )
        current_discrete_action_val = (
            disc_action_s_np.item() if disc_action_s_np.ndim > 0 else disc_action_s_np
        )
        assert 0 <= current_discrete_action_val < discrete_dim, (
            f"Single obs, stochastic discrete action out of bounds. Got {current_discrete_action_val}, expected < {discrete_dim}"
        )

        assert isinstance(cont_action_s_np, np.ndarray), (
            "Single obs, stochastic continuous action is not np.ndarray."
        )
        assert cont_action_s_np.shape == (continuous_dim,), (
            f"Single obs, stochastic continuous action shape mismatch: {cont_action_s_np.shape}"
        )
        assert np.all(cont_action_s_np >= continuous_low - 1e-5) and np.all(
            cont_action_s_np <= continuous_high + 1e-5
        ), (
            f"Single obs, stochastic continuous action out of env bounds [{continuous_low}, {continuous_high}]. Got: {cont_action_s_np}"
        )
        print("    Single obs, stochastic: PASSED")

        print("  Testing predict() with single observation (deterministic=True)...")
        (disc_action_d_np, cont_action_d_np), _ = policy.predict(
            test_obs_np_single, deterministic=True
        )
        assert isinstance(disc_action_d_np, np.ndarray)
        current_discrete_action_val_det = (
            disc_action_d_np.item() if disc_action_d_np.ndim > 0 else disc_action_d_np
        )
        assert 0 <= current_discrete_action_val_det < discrete_dim, (
            f"Single obs, deterministic discrete action out of bounds. Got {current_discrete_action_val_det}, expected < {discrete_dim}"
        )
        assert isinstance(cont_action_d_np, np.ndarray) and cont_action_d_np.shape == (
            continuous_dim,
        )
        assert np.all(cont_action_d_np >= continuous_low - 1e-5) and np.all(
            cont_action_d_np <= continuous_high + 1e-5
        ), (
            f"Single obs, deterministic continuous action out of env bounds [{continuous_low}, {continuous_high}]. Got: {cont_action_d_np}"
        )
        print("    Single obs, deterministic: PASSED")

    except Exception as e:
        print(f"    Single obs predict FAILED: {e}")
        import traceback

        traceback.print_exc()

    # --- 测试 predict() with batched observation ---
    try:
        print("  Testing predict() with batched observation (deterministic=False)...")
        (disc_actions_batch_s_np, cont_actions_batch_s_np), _ = policy.predict(
            test_obs_np_batch, deterministic=False
        )

        assert isinstance(disc_actions_batch_s_np, np.ndarray), (
            "Batch obs, stochastic discrete action is not np.ndarray."
        )
        # 对于批量离散动作，期望形状是 (batch_size,)
        assert disc_actions_batch_s_np.shape == (test_obs_np_batch.shape[0],), (
            f"Batch obs, stochastic discrete action shape mismatch: Expected ({test_obs_np_batch.shape[0]},), Got {disc_actions_batch_s_np.shape}"
        )
        assert np.all(
            (disc_actions_batch_s_np >= 0) & (disc_actions_batch_s_np < discrete_dim)
        ), "Batch obs, stochastic discrete action out of bounds."

        assert isinstance(cont_actions_batch_s_np, np.ndarray), (
            "Batch obs, stochastic continuous action is not np.ndarray."
        )
        assert cont_actions_batch_s_np.shape == (
            test_obs_np_batch.shape[0],
            continuous_dim,
        ), (
            f"Batch obs, stochastic continuous action shape mismatch: Expected ({test_obs_np_batch.shape[0]}, {continuous_dim}), Got {cont_actions_batch_s_np.shape}"
        )
        assert np.all(cont_actions_batch_s_np >= continuous_low - 1e-5) and np.all(
            cont_actions_batch_s_np <= continuous_high + 1e-5
        ), (
            f"Batch obs, stochastic continuous action out of env bounds [{continuous_low}, {continuous_high}]. Got sample: {cont_actions_batch_s_np[0] if test_obs_np_batch.shape[0] > 0 else 'N/A'}"
        )
        print("    Batch obs, stochastic: PASSED")

        print("  Testing predict() with batched observation (deterministic=True)...")
        (disc_actions_batch_d_np, cont_actions_batch_d_np), _ = policy.predict(
            test_obs_np_batch, deterministic=True
        )
        assert isinstance(
            disc_actions_batch_d_np, np.ndarray
        ) and disc_actions_batch_d_np.shape == (test_obs_np_batch.shape[0],)
        assert np.all(
            (disc_actions_batch_d_np >= 0) & (disc_actions_batch_d_np < discrete_dim)
        )
        assert isinstance(
            cont_actions_batch_d_np, np.ndarray
        ) and cont_actions_batch_d_np.shape == (
            test_obs_np_batch.shape[0],
            continuous_dim,
        )
        assert np.all(cont_actions_batch_d_np >= continuous_low - 1e-5) and np.all(
            cont_actions_batch_d_np <= continuous_high + 1e-5
        ), (
            f"Batch obs, deterministic continuous action out of env bounds [{continuous_low}, {continuous_high}]. Got sample: {cont_actions_batch_d_np[0] if test_obs_np_batch.shape[0] > 0 else 'N/A'}"
        )
        print("    Batch obs, deterministic: PASSED")

        print(f"PASSED: predict() tests for Action Space: {test_action_space}")

    except Exception as e:
        print(f"    Batch obs predict FAILED: {e}")
        import traceback

        traceback.print_exc()


# --- 模拟观测数据 ---
batch_size_policy_test = 3  # 使用小批量进行测试
single_obs_np = observation_space.sample().astype(np.float32)
batch_obs_np = np.array(
    [observation_space.sample() for _ in range(batch_size_policy_test)],
    dtype=np.float32,
)


# --- 场景1: 连续动作空间是 [-1, 1] (不需要反向缩放) ---
action_space_scaled = gymnasium.spaces.Tuple(
    (
        gymnasium.spaces.Discrete(DISCRETE_ACTION_DIM),  # 使用全局常量
        gymnasium.spaces.Box(
            low=-1.0, high=1.0, shape=(CONTINUOUS_ACTION_DIM,), dtype=np.float32
        ),  # 使用全局常量
    )
)
policy_scaled_env = create_policy_for_action_space(action_space_scaled)
if policy_scaled_env:
    test_predict_method(
        policy_scaled_env,
        action_space_scaled,
        single_obs_np,
        batch_obs_np,
        is_unscaled_env=False,
    )

# --- 场景2: 连续动作空间不是 [-1, 1] (例如 [0, 5], 需要反向缩放) ---
action_space_unscaled = gymnasium.spaces.Tuple(
    (
        gymnasium.spaces.Discrete(DISCRETE_ACTION_DIM),  # 使用全局常量
        gymnasium.spaces.Box(
            low=np.array([0.0, -2.0]),
            high=np.array([5.0, 2.0]),
            shape=(CONTINUOUS_ACTION_DIM,),
            dtype=np.float32,
        ),  # 使用全局常量
    )
)
policy_unscaled_env = create_policy_for_action_space(action_space_unscaled)
if policy_unscaled_env:
    test_predict_method(
        policy_unscaled_env,
        action_space_unscaled,
        single_obs_np,
        batch_obs_np,
        is_unscaled_env=True,
    )


# --- 测试保存和加载 (使用其中一个策略实例) ---
print("\n--- Testing HybridSACPolicy Save and Load (using policy_scaled_env) ---")
if policy_scaled_env:  # 确保 policy_scaled_env 成功创建
    try:
        with tempfile.TemporaryDirectory() as tmpdirname:
            policy_path = os.path.join(tmpdirname, "hybrid_sac_policy_predict_test.zip")

            # 为了比较，获取加载前的预测 (单个观测，确定性)
            (orig_disc, orig_cont), _ = policy_scaled_env.predict(
                single_obs_np, deterministic=True
            )

            policy_scaled_env.save(policy_path)
            print(f"  Policy saved to {policy_path}")

            loaded_policy = HybridSACPolicy.load(policy_path, device=DEVICE)
            print(f"  Policy loaded from {policy_path}")

            assert isinstance(loaded_policy, HybridSACPolicy), (
                "Loaded policy is not HybridSACPolicy instance."
            )
            assert isinstance(loaded_policy.actor, HybridActor)
            assert isinstance(loaded_policy.critic, HybridCritic)

            # 比较加载后的策略的输出
            (loaded_disc, loaded_cont), _ = loaded_policy.predict(
                single_obs_np, deterministic=True
            )

            assert np.allclose(orig_disc, loaded_disc, atol=1e-7), (
                f"Discrete actions differ after loading. Orig: {orig_disc}, Loaded: {loaded_disc}"
            )
            assert np.allclose(orig_cont, loaded_cont, atol=1e-7), (
                f"Continuous actions differ after loading. Orig: {orig_cont}, Loaded: {loaded_cont}"
            )
            print("  Policy output matches after loading (deterministic, single obs).")

        print("PASSED: HybridSACPolicy Save and Load")
    except Exception as e:
        print(f"FAILED: HybridSACPolicy Save and Load - {e}")
        import traceback

        traceback.print_exc()
else:
    print("SKIPPED: HybridSACPolicy Save and Load (policy_scaled_env not initialized).")


# --- 5. 关于 HybridSACPolicy.predict() (公共接口) 的讨论 ---
# (这部分保持不变，作为提醒)
print("\n--- 5. Discussion on HybridSACPolicy.predict() (Public API) ---")
print(
    "REMINDER: The standard BasePolicy.predict() method needs to be overridden in HybridSACPolicy"
)
print(
    "to correctly handle the tuple action output from _predict(), convert it to NumPy,"
)
print(
    "and perform any necessary unscaling for the continuous part if environment expects unscaled actions."
)
print(
    "The tests above focus on _predict(). You'll need to implement and test predict() separately."
)
print(
    "Example structure for predict override (ensure return type matches what your environment/algorithm expects):"
)
print("""
# In HybridSACPolicy:
# def predict(
#     self,
#     observation: Union[np.ndarray, Dict[str, np.ndarray]],
#     state: Optional[Tuple[np.ndarray, ...]] = None,
#     episode_start: Optional[np.ndarray] = None,
#     deterministic: bool = False,
# ) -> Tuple[Tuple[np.ndarray, np.ndarray], Optional[Tuple[np.ndarray, ...]]]: # Modified to return Tuple[np.ndarray, np.ndarray] for actions
#     self.set_training_mode(False)
#     obs_tensor, vectorized_env = self.obs_to_tensor(observation) # from BasePolicy
#     with th.no_grad():
#         # self._predict returns (discrete_th, continuous_th)
#         discrete_actions_th, continuous_actions_th = self._predict(obs_tensor, deterministic=deterministic)
# 
#     disc_np = discrete_actions_th.cpu().numpy()
#     cont_np = continuous_actions_th.cpu().numpy()
# 
#     cont_space = self.action_space.spaces[1] # This is a gymnasium.spaces.Box
#     if self.squash_output: # Assuming squash_output applies to the continuous part (typical for SAC)
#         if not (np.allclose(cont_space.low, -1.0) and np.allclose(cont_space.high, 1.0)):
#             low = cont_space.low
#             high = cont_space.high
#             unscaled_cont_np = low + (0.5 * (cont_np + 1.0) * (high - low))
#         else:
#             unscaled_cont_np = cont_np # No unscaling needed if env expects [-1, 1]
#     else: # Not squashed, clip to bounds
#         unscaled_cont_np = np.clip(cont_np, cont_space.low, cont_space.high)
# 
#     if not vectorized_env: # If the observation was not batched
#         disc_np_final = disc_np.squeeze(axis=0)
#         # Ensure discrete is scalar if original space is Discrete()
#         if isinstance(self.action_space.spaces[0], gymnasium.spaces.Discrete) and disc_np_final.ndim == 0:
#             pass # Already scalar
#         elif disc_np_final.ndim == 1 and disc_np_final.shape[0] == 1 and isinstance(self.action_space.spaces[0], gymnasium.spaces.Discrete):
#             disc_np_final = disc_np_final[0]

#         unscaled_cont_np_final = unscaled_cont_np.squeeze(axis=0)
#         actions_output = (disc_np_final, unscaled_cont_np_final)
#     else: # If the observation was batched
#         actions_output = (disc_np, unscaled_cont_np) # Return tuple of batched actions
# 
#     return actions_output, state
""")

print("\n--- HybridSACPolicy predict() focused testing complete ---")
