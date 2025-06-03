# 在你的 custom_components 目录下，例如 hybrid_replay_buffer.py

import numpy as np
import torch as th
import warnings
from gymnasium import spaces
from typing import Any, Dict, List, Optional, Tuple, Union

from stable_baselines3.common.buffers import ReplayBuffer, BaseBuffer
from stable_baselines3.common.type_aliases import ReplayBufferSamples
from stable_baselines3.common.vec_env import VecNormalize

from stable_baselines3.common.utils import get_device
from stable_baselines3.common.preprocessing import (
    get_obs_shape,
    get_action_dim,
)  # 需要这个

try:
    # Check memory used by replay buffer when possible
    import psutil
except ImportError:
    psutil = None


class HybridReplayBufferSamples(ReplayBufferSamples):
    # ReplayBufferSamples 已经是一个 NamedTuple，可以直接使用。
    # 如果需要，可以扩展它，但通常不需要，因为动作本身就是元组形式的数据。
    # 我们只需要确保在创建 ReplayBufferSamples 实例时，
    # actions 字段是一个包含 (discrete_actions_tensor, continuous_actions_tensor) 的元组。
    # 但标准的 ReplayBufferSamples 期望 actions 是一个单独的 th.Tensor。
    # 这意味着 _get_samples 返回的 ReplayBufferSamples.actions 将是一个元组。
    # 训练循环中的代码（例如 SAC.train()）在消费这个 actions 时需要知道它是一个元组。

    # 另一种更符合 SB3 结构的做法是，在 HybridReplayBufferSamples 中将 actions 定义为一个元组类型
    # 但这会破坏与期望单个动作张量的下游代码的兼容性。

    # 更好的方法是，ReplayBufferSamples.actions 仍然是一个张量，
    # 但这个张量是拼接后的。然后在 Actor 和 Critic 中分解它。
    # 或者，如果想保持分离，那么下游的 SAC.train() 方法需要修改。

    # 鉴于我们之前的 HybridContinuousCritic 期望一个元组动作输入，
    # 让 _get_samples 返回一个包含动作元组的结构是合理的。
    # 我们需要创建一个新的 Samples 类型或修改下游。

    # 为了最小化对 SAC 算法本身的修改，一个策略可以是：
    # 1. 在 HybridReplayBuffer 中分别存储离散和连续动作。
    # 2. 在 _get_samples 中，将它们作为元组的 PyTorch 张量返回。
    # 3. 修改 ReplayBufferSamples 使其能接受动作元组，或者创建一个新的 Samples 类。
    #    或者，让 SAC.train() 直接处理从 buffer.sample() 返回的动作元组。

    # 让我们创建一个新的 Samples 类，以保持清晰。
    pass  # ReplayBufferSamples 本身可以被用来传递 actions=tuple_of_tensors


class HybridReplayBuffer(ReplayBuffer):
    def __init__(
        self,
        buffer_size: int,
        observation_space: spaces.Space,
        action_space: spaces.Tuple,  # 期望一个元组空间
        device: Union[th.device, str] = "auto",
        n_envs: int = 1,
        optimize_memory_usage: bool = False,  # 通常 DictReplayBuffer 不支持
        handle_timeout_termination: bool = True,
    ):
        # 调用 ReplayBuffer 的父类 BaseBuffer 的 __init__
        # 我们只手动初始化 BaseBuffer 需要的部分，避免调用 get_action_dim

        # 手动初始化 BaseBuffer 的属性 (除了 action_dim)
        super(BaseBuffer, self).__init__()  # 这只是 ABC.__init__()，不做事
        self.buffer_size = max(buffer_size // n_envs, 1)  # ReplayBuffer 的逻辑
        self.observation_space = observation_space
        self.action_space = action_space  # 存储原始的 Tuple 空间
        self.obs_shape = get_obs_shape(observation_space)
        self.device = get_device(device)
        self.n_envs = n_envs
        self.pos = 0
        self.full = False

        # Check that the replay buffer can fit into the memory
        if psutil is not None:
            mem_available = psutil.virtual_memory().available

        if not isinstance(action_space, spaces.Tuple) or len(action_space.spaces) != 2:
            raise ValueError(
                "HybridReplayBuffer expects a Tuple action space with 2 elements (Discrete, Box)."
            )
        print(action_space.spaces)

        self.discrete_action_space = action_space.spaces[0]
        self.continuous_action_space = action_space.spaces[1]

        assert isinstance(self.discrete_action_space, spaces.Discrete), (
            "First action space must be Discrete"
        )
        assert isinstance(self.continuous_action_space, spaces.Box), (
            "Second action space must be Box"
        )

        # 不能使用 optimize_memory_usage，因为它假设 obs 和 next_obs 可以共享内存，
        # 并且它与 DictReplayBuffer 的结构不兼容 (我们更接近 DictReplayBuffer 的思想)
        if optimize_memory_usage:
            raise ValueError(
                "HybridReplayBuffer does not support optimize_memory_usage."
            )
        self.optimize_memory_usage = optimize_memory_usage  # 应该为 False

        # 定义我们实际使用的维度信息 (不再需要 BaseBuffer 的 self.action_dim)
        self.discrete_action_dim = get_action_dim(self.discrete_action_space)
        self.continuous_action_dim = get_action_dim(self.continuous_action_space)

        # 存储原始的 action_space
        self.action_space = action_space

        # 为离散和连续动作分别创建存储
        self.actions_discrete = np.zeros(
            (
                self.buffer_size,
                self.n_envs,
                self.discrete_action_dim,
            ),
            dtype=self.discrete_action_space.dtype,
        )
        self.actions_continuous = np.zeros(
            (self.buffer_size, self.n_envs, self.continuous_action_dim),
            dtype=self._maybe_cast_dtype(self.continuous_action_space.dtype),
        )

        # 其他存储 (observations, next_observations, rewards, dones, timeouts)
        # 可以从 ReplayBuffer 继承，如果它们不需要特殊处理 Dict 类型的 obs
        # 如果你的 observation_space 是字典，你需要基于 DictReplayBuffer 来做修改
        # 这里假设 observation_space 是简单的 Box，所以 ReplayBuffer 的 obs 存储方式适用

        # 如果 observation_space 是字典:
        if isinstance(observation_space, spaces.Dict):
            self.observations: Dict[str, np.ndarray] = {  # type: ignore[no-redef]
                key: np.zeros(
                    (self.buffer_size, self.n_envs, *_obs_shape),
                    dtype=observation_space[key].dtype,
                )
                for key, _obs_shape in self.obs_shape.items()  # type: ignore
            }
            # 当 optimize_memory_usage 为 False (我们的情况)
            self.next_observations: Dict[str, np.ndarray] = {  # type: ignore[no-redef]
                key: np.zeros(
                    (self.buffer_size, self.n_envs, *_obs_shape),
                    dtype=observation_space[key].dtype,
                )
                for key, _obs_shape in self.obs_shape.items()  # type: ignore
            }
        else:  # 如果是 Box 观察空间
            self.observations = np.zeros(
                (self.buffer_size, self.n_envs, *self.obs_shape),
                dtype=observation_space.dtype,
            )  # type: ignore
            self.next_observations = np.zeros(
                (self.buffer_size, self.n_envs, *self.obs_shape),
                dtype=observation_space.dtype,
            )  # type: ignore

        self.rewards = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.dones = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.handle_timeout_termination = handle_timeout_termination
        self.timeouts = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)

        # psutil 检查
        if psutil is not None:
            total_memory_usage = 0.0

            # 计算观察空间内存
            if isinstance(self.observations, dict):
                for key in self.observations:
                    total_memory_usage += self.observations[key].nbytes
                    total_memory_usage += self.next_observations[
                        key
                    ].nbytes  # 因为 optimize_memory_usage=False
            else:  # Box 或其他非字典类型
                total_memory_usage += self.observations.nbytes
                total_memory_usage += (
                    self.next_observations.nbytes
                )  # 因为 optimize_memory_usage=False

            # 计算动作空间内存 (使用我们自定义的数组)
            total_memory_usage += self.actions_discrete.nbytes
            total_memory_usage += self.actions_continuous.nbytes

            # 计算奖励、完成标志、超时标志内存
            total_memory_usage += self.rewards.nbytes
            total_memory_usage += self.dones.nbytes
            total_memory_usage += self.timeouts.nbytes

            # --- 比较并警告 ---
            if total_memory_usage > mem_available:
                # 转换为 GB
                total_memory_usage /= 1e9
                mem_available /= 1e9
                import warnings  # 确保导入 warnings

                warnings.warn(
                    "This system does not have apparently enough memory to store the complete "
                    f"replay buffer {total_memory_usage:.2f}GB > {mem_available:.2f}GB"
                )

    def split_combined_actions(
        self, actions_iterable: Union[np.ndarray, List[np.ndarray]]
    ) -> Tuple[np.ndarray, np.ndarray]:
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

    def add(
        self,
        obs: Union[np.ndarray, Dict[str, np.ndarray]],
        next_obs: Union[np.ndarray, Dict[str, np.ndarray]],
        action: Tuple[np.ndarray, np.ndarray],  # 动作现在是一个元组
        reward: np.ndarray,
        done: np.ndarray,
        infos: List[Dict[str, Any]],
    ) -> None:
        # 检查动作元组的结构
        # 这个判断语句默认 action 是一个元组，包含离散和连续动作
        # 但是我们已经把action的离散和连续，合成为一个元素。所以不能再这样。
        # 诶，但是我们可以把他再拆分开？
        # split_actions = self.split_combined_actions(action)
        #
        if not (isinstance(action, tuple) and len(action) == 2):
            raise ValueError(
                "Action must be a tuple of (discrete_action, continuous_action)."
            )

        discrete_action, continuous_action = action

        # 处理观察 (如果不是字典，确保它是正确的形状)
        if isinstance(
            self.observation_space, spaces.Discrete
        ):  # 这是一个例子，你的 obs 可能是 Box 或 Dict
            obs = obs.reshape((self.n_envs, *self.obs_shape))  # type: ignore
            next_obs = next_obs.reshape((self.n_envs, *self.obs_shape))  # type: ignore
        elif isinstance(self.observation_space, spaces.Dict):
            for key in self.observations.keys():  # type: ignore
                if isinstance(self.observation_space.spaces[key], spaces.Discrete):
                    obs[key] = obs[key].reshape((self.n_envs,) + self.obs_shape[key])  # type: ignore
                self.observations[key][self.pos] = np.array(obs[key])  # type: ignore
            for key in self.next_observations.keys():  # type: ignore
                if isinstance(self.observation_space.spaces[key], spaces.Discrete):
                    next_obs[key] = next_obs[key].reshape(
                        (self.n_envs,) + self.obs_shape[key]
                    )  # type: ignore
                self.next_observations[key][self.pos] = np.array(next_obs[key])  # type: ignore
        else:  # Box
            self.observations[self.pos] = np.array(obs)  # type: ignore
            self.next_observations[self.pos] = np.array(next_obs)  # type: ignore

        # 存储离散和连续动作
        # discrete_action 应该是 (n_envs,) 或 (n_envs, 1)
        # continuous_action 应该是 (n_envs, continuous_dim)
        self.actions_discrete[self.pos] = discrete_action.reshape(
            (self.n_envs, self.discrete_action_dim)
        )
        self.actions_continuous[self.pos] = continuous_action.reshape(
            (self.n_envs, self.continuous_action_dim)
        )

        self.rewards[self.pos] = np.array(reward)
        self.dones[self.pos] = np.array(done)

        if self.handle_timeout_termination:
            self.timeouts[self.pos] = np.array(
                [info.get("TimeLimit.truncated", False) for info in infos]
            )

        self.pos += 1
        if self.pos == self.buffer_size:
            self.full = True
            self.pos = 0

    def _get_samples(
        self, batch_inds: np.ndarray, env: Optional[VecNormalize] = None
    ) -> ReplayBufferSamples:
        # Sample randomly the env idx
        env_indices = np.random.randint(0, high=self.n_envs, size=(len(batch_inds),))

        # --- 处理观察 ---
        if isinstance(self.observation_space, spaces.Dict):
            # 从 DictReplayBuffer._get_samples 借鉴
            obs_batch = {
                key: obs[batch_inds, env_indices, :]
                for key, obs in self.observations.items()  # type: ignore
            }
            next_obs_batch = {
                key: obs[batch_inds, env_indices, :]
                for key, obs in self.next_observations.items()  # type: ignore
            }
            obs_normalized = self._normalize_obs(obs_batch, env)
            next_obs_normalized = self._normalize_obs(next_obs_batch, env)

            # 转换为 torch 张量字典
            observations = {
                key: self.to_torch(obs) for key, obs in obs_normalized.items()
            }  # type: ignore
            next_observations = {
                key: self.to_torch(obs) for key, obs in next_obs_normalized.items()
            }  # type: ignore
        else:  # Box
            observations = self.to_torch(
                self._normalize_obs(self.observations[batch_inds, env_indices, :], env)
            )  # type: ignore
            next_observations = self.to_torch(
                self._normalize_obs(
                    self.next_observations[batch_inds, env_indices, :], env
                )
            )  # type: ignore

        # --- 处理动作 ---
        # 获取离散和连续动作，并转换为 PyTorch 张量
        actions_discrete_th = self.to_torch(
            self.actions_discrete[batch_inds, env_indices, :]
        )
        actions_continuous_th = self.to_torch(
            self.actions_continuous[batch_inds, env_indices, :]
        )

        # 将动作打包成元组，这是下游 Critic 和 Actor 期望的
        actions_tuple = (actions_discrete_th, actions_continuous_th)

        return ReplayBufferSamples(
            observations=observations,
            actions=actions_tuple,  # 这里的 actions 现在是一个元组的张量
            next_observations=next_observations,
            dones=self.to_torch(
                (
                    self.dones[batch_inds, env_indices]
                    * (1 - self.timeouts[batch_inds, env_indices])
                ).reshape(-1, 1)
            ),
            rewards=self.to_torch(
                self._normalize_reward(
                    self.rewards[batch_inds, env_indices].reshape(-1, 1), env
                )
            ),
        )

    # sample() 方法可以从 ReplayBuffer 继承，因为它最终会调用 _get_samples()
    # 如果 optimize_memory_usage 为 True (我们禁用了它)，则 sample() 有特殊逻辑，但我们不需要担心。
