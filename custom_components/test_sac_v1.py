import torch
import torch.nn as nn
import gymnasium  # 完整导入 gymnasium
from gymnasium import spaces
import numpy as np
import tempfile
import os
from typing import Dict, Any, Optional, Tuple, Union, Type, List  # 确保导入 List

# 假设你的自定义组件可以被导入
# 这些路径需要根据你的项目结构进行调整
from hybrid_sac_agent import HybridSAC  # 被测试的算法类
from hybrid_sac_policy import HybridSACPolicy
from hybrid_replay_buffer import HybridReplayBuffer
from hybrid_actor import HybridActor  # 用于类型检查
from hybrid_critic import HybridCritic  # 用于类型检查

# 从 SB3 导入相关组件
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import DummyVecEnv, VecEnv
from stable_baselines3.common.type_aliases import GymEnv, Schedule
from stable_baselines3.common.policies import BasePolicy


os.system("cls")


# --- 1. 定义一个简单的自定义混合动作环境 ---
class SimpleHybridEnv(
    gymnasium.Env[np.ndarray, Tuple[Union[int, np.ndarray], np.ndarray]]
):  # 动作类型提示更精确
    metadata = {"render_modes": ["human"], "render_fps": 30}

    def __init__(
        self,
        discrete_n=3,
        continuous_dim=2,
        continuous_low=None,
        continuous_high=None,
        max_steps=50,
    ):
        super(SimpleHybridEnv, self).__init__()
        self.obs_dim = 4
        self.discrete_action_n = discrete_n
        self.continuous_action_dim = continuous_dim

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.obs_dim,), dtype=np.float32
        )

        if continuous_low is None:
            continuous_low = np.array([-0.5] * continuous_dim, dtype=np.float32)
        if continuous_high is None:
            continuous_high = np.array([0.5] * continuous_dim, dtype=np.float32)

        self.action_space = spaces.Tuple(
            (
                spaces.Discrete(self.discrete_action_n),
                spaces.Box(
                    low=continuous_low,
                    high=continuous_high,
                    shape=(self.continuous_action_dim,),
                    dtype=np.float32,
                ),
            )
        )

        self.state: Optional[np.ndarray] = None
        self.current_step = 0
        self.max_steps = max_steps

    def reset(
        self, seed: Optional[int] = None, options: Optional[dict] = None
    ) -> Tuple[np.ndarray, dict]:
        super().reset(seed=seed)
        self.state = self.observation_space.sample().astype(np.float32)
        self.current_step = 0
        return self.state, {}

    def step(
        self, action: Tuple[Union[int, np.ndarray], np.ndarray]
    ) -> Tuple[np.ndarray, float, bool, bool, dict]:
        print(
            f"DEBUG_ENV_STEP: Received action = {action}, type = {type(action)}"
        )  # <--- 添加这个打印

        action_is_valid = self.action_space.contains(action)
        if not action_is_valid:
            # 修改后的简化 print 语句
            type_action0_str = str(type(action[0]))
            dtype_action0_str = "N/A"
            if isinstance(action[0], np.ndarray):
                dtype_action0_str = str(action[0].dtype)

            type_action1_str = str(type(action[1]))
            dtype_action1_str = "N/A"
            if isinstance(action[1], np.ndarray):
                dtype_action1_str = str(action[1].dtype)

            print(
                f"Warning: Action {action} (discrete_type: {type_action0_str}, discrete_dtype: {dtype_action0_str}; continuous_type: {type_action1_str}, continuous_dtype: {dtype_action1_str}) is not in action space {self.action_space}"
            )

            if not self.action_space.spaces[0].contains(action[0]):
                print(
                    f"  Discrete part {action[0]} (type: {type(action[0])}) not in {self.action_space.spaces[0]}"
                )
            if not self.action_space.spaces[1].contains(action[1]):
                print(
                    f"  Continuous part {action[1]} (type: {type(action[1])}) not in {self.action_space.spaces[1]}"
                )
                print(
                    f"    Continuous bounds: low={self.action_space.spaces[1].low}, high={self.action_space.spaces[1].high}"
                )
            # raise ValueError(f"Received invalid action={action} for space={self.action_space}")

        discrete_action_val, continuous_action_val = action

        if isinstance(discrete_action_val, np.ndarray):
            if discrete_action_val.ndim > 0:
                discrete_action_val = discrete_action_val.item()

        reward = 0.0
        if discrete_action_val == 1:
            reward += 1.0

        target_continuous = np.array([0.2, 0.8], dtype=np.float32)
        reward -= np.sum(np.square(continuous_action_val - target_continuous)) * 0.1

        if self.state is not None:
            self.state = (
                self.state + np.random.randn(self.obs_dim).astype(np.float32) * 0.05
            )
            self.state = np.clip(
                self.state, self.observation_space.low, self.observation_space.high
            ).astype(np.float32)
        else:
            self.state = self.observation_space.sample().astype(np.float32)

        self.current_step += 1
        terminated = self.current_step >= self.max_steps
        truncated = False

        return self.state, float(reward), terminated, truncated, {}

    def render(self):
        pass

    def close(self):
        pass


# --- 2. 测试环境 ---
print("--- Checking Custom Environment ---")
env_params_for_check = {
    "discrete_n": 3,
    "continuous_dim": 2,
    "continuous_low": np.array([-0.5, 0.0], dtype=np.float32),
    "continuous_high": np.array([0.5, 1.0], dtype=np.float32),
}
env_to_check = SimpleHybridEnv(**env_params_for_check)
try:
    check_env(env_to_check, warn=True, skip_render_check=True)
    print("Custom environment check passed (or warnings printed).")
except Exception as e:
    print(f"Custom environment check FAILED: {e}")
    print(
        "This failure in check_env for Tuple action spaces is often expected and may not block algorithm testing."
    )

# --- 3. 定义测试参数 ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\nUsing device: {DEVICE}")

vec_env: VecEnv = DummyVecEnv(
    [
        lambda: SimpleHybridEnv(
            discrete_n=3,
            continuous_dim=2,
            continuous_low=np.array([-0.5, 0.0], dtype=np.float32),
            continuous_high=np.array([0.5, 1.0], dtype=np.float32),
        )
    ]
)
print("DummyVecEnv created successfully.")
print(f"VecEnv Observation Space: {vec_env.observation_space}")
print(f"VecEnv Action Space: {vec_env.action_space}")


# --- 4. 测试 HybridSAC 算法初始化 ---
print("\n--- 4. Testing HybridSAC Algorithm Initialization ---")
agent: Optional[HybridSAC] = None
init_passed = False
try:
    agent = HybridSAC(
        policy="HybridSACPolicy",
        env=vec_env,
        learning_starts=20,
        batch_size=4,
        buffer_size=200,
        tau=0.01,
        gamma=0.98,
        gradient_steps=1,
        verbose=0,
        device=DEVICE,
        seed=123,
        policy_kwargs=dict(net_arch=[32, 32]),
    )
    print(f"  Agent observation space: {agent.observation_space}")
    print(f"  Agent action space: {agent.action_space}")
    assert isinstance(agent.action_space, spaces.Tuple), (
        "Agent action space is not Tuple."
    )
    assert isinstance(agent.policy, HybridSACPolicy), (
        "Agent policy is not HybridSACPolicy."
    )
    assert isinstance(agent.replay_buffer, HybridReplayBuffer), (
        "Agent replay_buffer is not HybridReplayBuffer."
    )
    assert hasattr(agent, "actor") and isinstance(agent.actor, HybridActor), (
        "Agent actor is not HybridActor or not set."
    )
    assert hasattr(agent, "critic") and isinstance(agent.critic, HybridCritic), (
        "Agent critic is not HybridCritic or not set."
    )
    print("PASSED: HybridSAC Algorithm Initialization")
    init_passed = True
except Exception as e:
    print(f"FAILED: HybridSAC Algorithm Initialization - {e}")
    import traceback

    traceback.print_exc()


def split_combined_actions(actions_iterable) -> Tuple[np.ndarray, np.ndarray]:
    # 这个函数和上面的 reformat_actions_batch是一样的，理论上没啥用
    # print()
    if isinstance(actions_iterable, (List, tuple)) and not actions_iterable:
        # 如果是空的或者不是列表或者元组
        return np.array([], dtype=np.int64), np.array([], dtype=np.float32)
    discrete_actions_list = []
    continuous_actions_list = []
    item_idx = 0
    for action in actions_iterable:
        # action 应该是一个元组 (discrete_action, continuous_action)
        if not (isinstance(action, tuple) and len(action) == 2):
            raise ValueError(
                f"Action at index {item_idx} must be a tuple of (discrete_action, continuous_action)."
            )
        # 取出离散和连续动作
        discrete_action, continuous_action = action
        if not isinstance(continuous_action, Tuple):
            raise ValueError(
                f"输入数据中位置 {item_idx} 的元素 '{action}' 的第二个组成部分 '{continuous_action}' 格式不正确。"
                "它必须是一个元组 (tuple)。"
            )
        discrete_actions_list.append(discrete_action)
        continuous_actions_list.append(continuous_action)
        item_idx += 1
    if not discrete_actions_list:
        # 如果是空的或者不是列表或者元组
        return np.array([], dtype=np.int64), np.array([], dtype=np.float32)
    # 将列表转换为 numpy 数组
    discrete_actions = np.array(discrete_actions_list, dtype=np.int64)
    continuous_actions = np.array(continuous_actions_list, dtype=np.float32)
    return (discrete_actions, continuous_actions)


# --- 5. 测试数据收集和 Replay Buffer ---
print("\n--- 5. Testing Data Collection and Replay Buffer ---")
collection_passed = False
if agent and init_passed:
    try:
        num_collection_steps = 30
        obs = agent.env.reset()

        print(f"  Initial obs shape from vec_env.reset(): {obs.shape}")

        for step in range(num_collection_steps):
            action_tuple_np_batch_predict, _ = agent.predict(obs, deterministic=False)

            # 这段代码是假定返回的动作是一个元组，其中第一个元素是离散动作，第二个元素是连续动作
            # 但是我在policy里面把动作组合起来了，现在一个元组的一个元素就是离散+连续 所以注释掉
            # actions_for_vec_env: List[Tuple[Any, np.ndarray]] = []
            # for i in range(agent.n_envs):
            #     disc_action_for_env_i = action_tuple_np_batch_predict[0][i]
            #     cont_action_for_env_i = action_tuple_np_batch_predict[1][i]
            #     actions_for_vec_env.append(
            #         (disc_action_for_env_i, cont_action_for_env_i)
            #     )

            new_obs, rewards, dones, infos = agent.env.step(
                action_tuple_np_batch_predict
            )

            action_tuple_np_batch_predict = split_combined_actions(
                action_tuple_np_batch_predict
            )  # 分离动作
            agent.replay_buffer.add(
                obs, new_obs, action_tuple_np_batch_predict, rewards, dones, infos
            )
            obs = new_obs

            for idx, done in enumerate(dones):
                if done:
                    if agent.n_envs == 1:
                        obs = agent.env.reset()
                    else:  # Should not happen with DummyVecEnv([lambda: ...]) as n_envs is 1
                        reset_output = agent.env.reset()
                        obs[idx] = reset_output[idx]

        assert agent.replay_buffer.size() > 0 or agent.replay_buffer.full, (
            "Replay buffer is empty after collection."
        )
        print(
            f"  Replay buffer current filled size: {agent.replay_buffer.pos if not agent.replay_buffer.full else agent.replay_buffer.buffer_size}"
        )
        print(f"  Replay buffer capacity: {agent.replay_buffer.buffer_size}")
        print(
            f"  Number of transitions collected (approx): {agent.replay_buffer.pos * agent.n_envs if not agent.replay_buffer.full else agent.replay_buffer.buffer_size * agent.n_envs}"
        )
        print("PASSED: Data Collection and Replay Buffer")
        collection_passed = True
    except Exception as e:
        print(f"FAILED: Data Collection and Replay Buffer - {e}")
        import traceback

        traceback.print_exc()
else:
    print("SKIPPED: Data Collection (Agent not initialized).")


# collection_passed = True  # For testing learn() method, we assume collection passed
# --- 6. 测试 learn() 方法 (这是主要的集成测试) ---
print("\n--- 6. Testing HybridSAC.learn() method (short run) ---")
learn_ran = False
if agent and init_passed and collection_passed:  # Added collection_passed
    try:
        print(
            "  Attempting to learn for 100 timesteps (enough for learning_starts and a few training steps)..."
        )
        total_learn_timesteps = (
            agent.learning_starts + agent.batch_size + (agent.gradient_steps * 10)
        )
        if total_learn_timesteps < 100:
            total_learn_timesteps = 100

        actor_params_before_learn = [
            p.clone().detach() for p in agent.actor.parameters() if p.requires_grad
        ]
        critic_params_before_learn = [
            p.clone().detach() for p in agent.critic.parameters() if p.requires_grad
        ]

        agent.learn(total_timesteps=total_learn_timesteps, log_interval=5)

        actor_params_after_learn = [
            p.clone().detach() for p in agent.actor.parameters() if p.requires_grad
        ]
        critic_params_after_learn = [
            p.clone().detach() for p in agent.critic.parameters() if p.requires_grad
        ]

        actor_changed_learn = False
        if actor_params_before_learn:
            actor_changed_learn = any(
                not torch.allclose(p_before, p_after, atol=1e-6)
                for p_before, p_after in zip(
                    actor_params_before_learn, actor_params_after_learn
                )
            )

        critic_changed_learn = False
        if critic_params_before_learn:
            critic_changed_learn = any(
                not torch.allclose(p_before, p_after, atol=1e-6)
                for p_before, p_after in zip(
                    critic_params_before_learn, critic_params_after_learn
                )
            )

        assert actor_changed_learn or not actor_params_before_learn, (
            "Actor parameters did not change after learn()."
        )
        assert critic_changed_learn or not critic_params_before_learn, (
            "Critic parameters did not change after learn()."
        )
        print("  Actor and Critic parameters changed after learn() (as expected).")
        print("PASSED: HybridSAC.learn() ran without crashing and parameters changed.")
        learn_ran = True

        print("  Evaluating policy after short learning...")
        eval_env_params = {
            "discrete_n": 3,
            "continuous_dim": 2,
            "continuous_low": np.array([-0.5, 0.0], dtype=np.float32),
            "continuous_high": np.array([0.5, 1.0], dtype=np.float32),
            "max_steps": 100,
        }
        eval_env = DummyVecEnv([lambda: SimpleHybridEnv(**eval_env_params)])
        mean_reward, std_reward = evaluate_policy(
            agent.policy, eval_env, n_eval_episodes=5, warn=False, deterministic=True
        )
        print(
            f"  Mean reward after learning (deterministic): {mean_reward:.2f} +/- {std_reward:.2f}"
        )
        eval_env.close()

    except Exception as e:
        print(f"FAILED: HybridSAC.learn() - {e}")
        import traceback

        traceback.print_exc()
else:
    print(
        "SKIPPED: HybridSAC.learn() (Agent not initialized or collection/init failed)."
    )


# --- 7. 测试保存与加载算法 ---
print("\n--- 7. Testing HybridSAC Algorithm Save and Load ---")
if agent and learn_ran:
    try:
        with tempfile.TemporaryDirectory() as tmpdirname:
            model_path = os.path.join(tmpdirname, "hybrid_sac_model.zip")

            # 从 agent 实例获取观测空间
            if agent.observation_space is None:
                raise ValueError(
                    "agent.observation_space is None. Cannot sample for save/load test."
                )
            current_test_obs_space = agent.observation_space
            test_single_obs_np_algo = current_test_obs_space.sample().astype(np.float32)

            # agent.predict() for a single obs
            # 现在我们期望它直接返回 (discrete_action_part, continuous_action_part)
            original_action_output_combined, _ = agent.predict(
                test_single_obs_np_algo, deterministic=True
            )

            # 断言 original_action_output_combined 是一个长度为2的元组
            assert (
                isinstance(original_action_output_combined, tuple)
                and len(original_action_output_combined) == 2
            ), (
                f"Predict output format mismatch for single obs. Expected (discrete_part, continuous_part), got type: {type(original_action_output_combined)} with len {len(original_action_output_combined) if isinstance(original_action_output_combined, tuple) else 'N/A'}"
            )

            orig_disc_algo_part = original_action_output_combined[0]
            orig_cont_algo_part = original_action_output_combined[
                1
            ]  # 这应该是 NumPy 数组

            # 进一步检查类型
            assert isinstance(orig_disc_algo_part, (int, np.integer, np.ndarray)), (
                f"Original discrete action part is not int or np.ndarray. Got type: {type(orig_disc_algo_part)}"
            )
            assert isinstance(orig_cont_algo_part, np.ndarray), (
                f"Original continuous action part is not np.ndarray. Got type: {type(orig_cont_algo_part)}"
            )

            # 如果离散部分是0维数组，转换为标量以便打印和比较
            orig_disc_algo_scalar = (
                orig_disc_algo_part.item()
                if isinstance(orig_disc_algo_part, np.ndarray)
                and orig_disc_algo_part.ndim == 0
                else orig_disc_algo_part
            )
            orig_cont_algo_np = orig_cont_algo_part  # 它已经是 NumPy 数组了

            print(
                f"  Original predict output (Save/Load Test): Discrete={orig_disc_algo_scalar}, Continuous={orig_cont_algo_np}"
            )

            agent.save(model_path)
            print(f"  Agent saved to {model_path}")

            original_policy_id = id(agent.policy)
            del agent
            print("  Original agent deleted.")

            vec_env_for_load: VecEnv = DummyVecEnv(
                [
                    lambda: SimpleHybridEnv(
                        discrete_n=3,
                        continuous_dim=2,
                        continuous_low=np.array([-0.5, 0.0], dtype=np.float32),
                        continuous_high=np.array([0.5, 1.0], dtype=np.float32),
                    )
                ]
            )

            loaded_agent = HybridSAC.load(
                model_path, env=vec_env_for_load, device=DEVICE
            )
            print(f"  Agent loaded from {model_path}")

            assert id(loaded_agent.policy) != original_policy_id, (
                "Loaded policy is the same instance as original (should be new)."
            )
            assert isinstance(loaded_agent.policy, HybridSACPolicy)
            assert isinstance(loaded_agent.actor, HybridActor)
            assert isinstance(loaded_agent.critic, HybridCritic)

            loaded_action_output_combined, _ = loaded_agent.predict(
                test_single_obs_np_algo, deterministic=True
            )

            assert (
                isinstance(loaded_action_output_combined, tuple)
                and len(loaded_action_output_combined) == 2
            ), (
                f"Loaded predict output format mismatch. Expected (discrete_part, continuous_part), got {type(loaded_action_output_combined)}"
            )
            loaded_disc_algo_part = loaded_action_output_combined[0]
            loaded_cont_algo_part = loaded_action_output_combined[1]

            assert isinstance(loaded_disc_algo_part, (int, np.integer, np.ndarray)), (
                f"Loaded discrete action part is not int or np.ndarray. Got type: {type(loaded_disc_algo_part)}"
            )
            assert isinstance(loaded_cont_algo_part, np.ndarray), (
                f"Loaded continuous action part is not np.ndarray. Got type: {type(loaded_cont_algo_part)}"
            )

            loaded_disc_algo_scalar = (
                loaded_disc_algo_part.item()
                if isinstance(loaded_disc_algo_part, np.ndarray)
                and loaded_disc_algo_part.ndim == 0
                else loaded_disc_algo_part
            )
            loaded_cont_algo_np = loaded_cont_algo_part

            print(
                f"  Loaded predict output (Save/Load Test): Discrete={loaded_disc_algo_scalar}, Continuous={loaded_cont_algo_np}"
            )

            assert np.allclose(
                np.array(orig_disc_algo_scalar).astype(np.int64),
                np.array(loaded_disc_algo_scalar).astype(np.int64),
            ), (
                f"Discrete actions differ after loading agent. Original: {orig_disc_algo_scalar}, Loaded: {loaded_disc_algo_scalar}"
            )

            assert np.allclose(orig_cont_algo_np, loaded_cont_algo_np, atol=1e-7), (
                f"Continuous actions differ after loading agent. Original: {orig_cont_algo_np}, Loaded: {loaded_cont_algo_np}"
            )
            print("  Agent output matches after loading (deterministic, single obs).")

            print("  Attempting to learn with loaded agent...")
            loaded_agent.learn(total_timesteps=20, reset_num_timesteps=False)
            print("  Learning with loaded agent ran without crashing.")

            vec_env_for_load.close()

        print("PASSED: HybridSAC Algorithm Save and Load")
    except Exception as e:
        print(f"FAILED: HybridSAC Algorithm Save and Load - {e}")
        import traceback

        traceback.print_exc()
else:
    print(
        "SKIPPED: HybridSAC Algorithm Save and Load (Agent not initialized or learn test failed)."
    )


if "vec_env" in locals() and vec_env is not None:
    vec_env.close()

print("\n--- HybridSAC Algorithm Testing Complete (Focus on learn()) ---")
