from typing import Any, Optional, Union, Tuple, Dict, List, Type

import torch as th
from gymnasium import spaces
from torch import nn
import numpy as np

from stable_baselines3.common.distributions import (
    SquashedDiagGaussianDistribution,
    StateDependentNoiseDistribution,
)
from stable_baselines3.common.policies import BasePolicy, ContinuousCritic
from stable_baselines3.common.preprocessing import get_action_dim
from stable_baselines3.common.torch_layers import (
    BaseFeaturesExtractor,
    CombinedExtractor,
    FlattenExtractor,
    NatureCNN,
    create_mlp,
    get_actor_critic_arch,
)
from stable_baselines3.common.type_aliases import PyTorchObs, Schedule

from .hybrid_actor import HybridActor
from .hybrid_critic import HybridCritic


# CAP the standard deviation of the actor
LOG_STD_MAX = 2
LOG_STD_MIN = -20


class HybridSACPolicy(BasePolicy):
    actor: HybridActor
    critic: HybridCritic
    critic_target: HybridCritic

    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Tuple,  # 期望一个元组空间
        lr_schedule: Schedule,
        net_arch: Optional[Union[List[int], Dict[str, List[int]]]] = None,
        activation_fn: type[nn.Module] = nn.ReLU,
        use_sde: bool = False,
        log_std_init: float = -3,
        use_expln: bool = False,
        clip_mean: float = 2.0,
        features_extractor_class: type[BaseFeaturesExtractor] = FlattenExtractor,
        features_extractor_kwargs: Optional[dict[str, Any]] = None,
        normalize_images: bool = True,
        optimizer_class: type[th.optim.Optimizer] = th.optim.Adam,
        # 这里可以接收到我们传入的优化器类和参数
        optimizer_kwargs: Optional[dict[str, Any]] = None,
        n_critics: int = 2,
        share_features_extractor: bool = True,  # 注意这里 SACPolicy 默认为 False，但通常共享更高效
        lr_actor_schedule: Optional[Schedule] = None,
        lr_critic_schedule: Optional[Schedule] = None,
    ):
        super().__init__(
            observation_space,
            action_space,  # BasePolicy 会存储这个
            features_extractor_class,
            features_extractor_kwargs,
            optimizer_class=optimizer_class,
            optimizer_kwargs=optimizer_kwargs,
            squash_output=True,  # SAC actor 的输出是 squash 过的
            normalize_images=normalize_images,
        )
        if not isinstance(action_space, spaces.Tuple) or len(action_space.spaces) != 2:
            raise ValueError(
                "HybridSACPolicy requires a Tuple action space with 2 elements (Discrete, Box)."
            )
        if not isinstance(action_space.spaces[0], spaces.Discrete):
            raise ValueError("First element of action_space Tuple must be Discrete.")
        if not isinstance(action_space.spaces[1], spaces.Box):
            raise ValueError("Second element of action_space Tuple must be Box.")

        if net_arch is None:
            net_arch = [256, 256]  # 默认网络结构

        self.lr_actor_schedule = (
            lr_actor_schedule if lr_actor_schedule is not None else lr_schedule
        )
        self.lr_critic_schedule = (
            lr_critic_schedule if lr_critic_schedule is not None else lr_schedule
        )

        # 从 net_arch 中分离 actor 和 critic 的网络结构
        # get_actor_critic_arch 是一个辅助函数，如果 net_arch 是列表，则两者共享；
        # 如果是字典 {'pi': [...], 'vf': [...]} (或 'actor', 'critic')，则分别使用。
        actor_arch, critic_arch = get_actor_critic_arch(net_arch)

        # 存储构造参数，用于 _get_constructor_parameters
        self.net_arch = net_arch
        self.activation_fn = activation_fn
        self.log_std_init = log_std_init
        self.n_critics = n_critics
        self.share_features_extractor = share_features_extractor
        # features_extractor_class 和 kwargs 已经由 BasePolicy 的 __init__ 存储为
        # self.features_extractor_class 和 self.features_extractor_kwargs

        # 用于创建 Actor 和 Critic 的参数字典
        # net_args 包含了会被 Actor 和 Critic 共享的参数 (如果它们自己的 kwargs 中没有覆盖)
        self.net_args = {
            "observation_space": self.observation_space,
            "action_space": self.action_space,  # HybridActor/Critic 会处理这个 Tuple
            "activation_fn": self.activation_fn,
            "normalize_images": normalize_images,
        }
        self.actor_kwargs = self.net_args.copy()
        self.actor_kwargs.update(
            {
                "net_arch": actor_arch,
                "log_std_init": self.log_std_init,
            }
        )

        # 如果使用 SDE，添加 SDE 相关参数
        sde_kwargs = {
            "use_sde": use_sde,
            "log_std_init": log_std_init,
            "use_expln": use_expln,
            "clip_mean": clip_mean,
        }
        self.actor_kwargs.update(sde_kwargs)

        self.critic_kwargs = self.net_args.copy()
        self.critic_kwargs.update(
            {
                "net_arch": critic_arch,
                "n_critics": self.n_critics,
                "share_features_extractor": self.share_features_extractor,  # Critic 需要知道是否共享
            }
        )

        # 调用 _build 来创建网络和优化器
        self._build()

    def _build(self) -> None:
        if self.features_extractor is None:  # 检查 BasePolicy 是否已经创建了
            self.features_extractor = self.make_features_extractor()

        # 2. 创建 Actor
        #    make_actor 方法会处理 features_extractor 的传递
        self.actor = self.make_actor()  # 使用 self.actor_kwargs
        self.actor.optimizer = self.optimizer_class(
            self.actor.parameters(),
            # lr=lr_schedule(1),  # type: ignore[call-arg]
            lr=self.lr_actor_schedule(1),  # type: ignore[call-arg]
            **self.optimizer_kwargs,
        )
        # 3. 创建 Critic
        if self.share_features_extractor:
            # 如果共享，将 Actor 的 features_extractor 传递给 Critic
            self.critic = self.make_critic(
                features_extractor=self.actor.features_extractor
            )
            # 确保 Critic 的优化器不更新共享的 features_extractor 的参数
            critic_parameters = [
                param
                for name, param in self.critic.named_parameters()
                if "features_extractor" not in name
            ]
        else:
            # 如果不共享，Critic 会创建自己的 features_extractor (make_critic 内部处理)
            self.critic = self.make_critic(features_extractor=None)
            critic_parameters = list(self.critic.parameters())

        self.critic.optimizer = self.optimizer_class(
            critic_parameters,
            # lr=lr_schedule(1),  # type: ignore[call-arg]
            lr=self.lr_critic_schedule(1),  # type: ignore[call-arg]
            **self.optimizer_kwargs,
        )
        # 4. 创建 Target Critic 网络
        # Target Critic 通常不共享 features_extractor (或者说，它有自己独立的副本)
        # 并且它的参数是从主 Critic 网络复制而来，不进行梯度更新。
        self.critic_target = self.make_critic(features_extractor=None)  # 创建独立的 FE
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_target.set_training_mode(False)  # 目标网络总是在评估模式

    def make_actor(
        self, features_extractor: Optional[BaseFeaturesExtractor] = None
    ) -> HybridActor:
        actor_kwargs = self._update_features_extractor(
            self.actor_kwargs, features_extractor
        )
        # 如果 HybridActor 的 __init__ 不需要 features_dim，则不需要在这里手动添加
        # actor_kwargs["features_dim"] = actor_kwargs["features_extractor"].features_dim
        return HybridActor(**actor_kwargs).to(self.device)

    def make_critic(
        self, features_extractor: Optional[BaseFeaturesExtractor] = None
    ) -> HybridCritic:
        critic_kwargs = self._update_features_extractor(
            self.critic_kwargs, features_extractor
        )
        # 如果 HybridCritic 的 __init__ 不需要 features_dim，则不需要在这里手动添加
        # critic_kwargs["features_dim"] = critic_kwargs["features_extractor"].features_dim
        return HybridCritic(**critic_kwargs).to(self.device)

    def forward(
        self, obs: PyTorchObs, deterministic: bool = False
    ) -> Tuple[th.Tensor, th.Tensor]:
        # BasePolicy 的 _predict 最终会调用 Actor 的 forward
        # 我们需要确保 HybridActor 的 forward 返回的是 (discrete_action, continuous_action)
        return self.actor.forward(obs, deterministic=deterministic)

    def _predict(
        self, observation: PyTorchObs, deterministic: bool = False
    ) -> Tuple[th.Tensor, th.Tensor]:
        # 直接调用 actor 的 forward (或 predict)
        return self.actor(
            observation, deterministic=deterministic
        )  # 或者 actor.forward

    def _get_constructor_parameters(self) -> Dict[str, Any]:
        # 收集并返回一个字典，这个字典包含了重建当前策略对象（例如你的 HybridSACPolicy 实例）所必需的所有构造函数参数及其对应的值。
        data = super()._get_constructor_parameters()  # 获取 BasePolicy 保存的参数
        # BasePolicy 的 _get_constructor_parameters 通常包含：
        # observation_space, action_space, normalize_images,
        # optimizer_class, optimizer_kwargs,
        # features_extractor_class, features_extractor_kwargs (这些由 BasePolicy 自身保存)

        data.update(
            dict(
                net_arch=self.net_arch,
                activation_fn=self.net_args["activation_fn"],
                use_sde=self.actor_kwargs["use_sde"],
                log_std_init=self.actor_kwargs["log_std_init"],
                use_expln=self.actor_kwargs["use_expln"],
                clip_mean=self.actor_kwargs["clip_mean"],
                n_critics=self.n_critics,  # Critic 特有
                share_features_extractor=self.share_features_extractor,
                lr_schedule=self._dummy_schedule,  # 像 SACPolicy 那样
                # optimizer_class 和 optimizer_kwargs 应该由 super() 处理了
                # features_extractor_class 和 features_extractor_kwargs 也由 super() 处理
                optimizer_class=self.optimizer_class,
                optimizer_kwargs=self.optimizer_kwargs,
                features_extractor_class=self.features_extractor_class,
                features_extractor_kwargs=self.features_extractor_kwargs,
            )
        )
        return data

    def set_training_mode(self, mode: bool) -> None:
        """Put the policy in either training or evaluation mode."""
        self.actor.set_training_mode(mode)
        self.critic.set_training_mode(mode)
        self.training = mode  # BasePolicy 有这个属性

    def predict(
        self,
        observation: Union[np.ndarray, Dict[str, np.ndarray]],
        state: Optional[Tuple[np.ndarray, ...]] = None,
        episode_start: Optional[np.ndarray] = None,
        deterministic: bool = False,
    ) -> Tuple[Tuple[np.ndarray, np.ndarray], Optional[Tuple[np.ndarray, ...]]]:
        """
        Get the policy's action from an observation for a hybrid action space.
        Overrides BasePolicy.predict to handle tuple actions.

        :param observation: The input observation.
        :param state: The last hidden states (can be None, used in recurrent policies).
        :param episode_start: The last masks (can be None, used in recurrent policies).
        :param deterministic: Whether or not to return deterministic actions.
        :return: A tuple containing:
            - actions: A tuple (discrete_action_np, continuous_action_np).
            - state: The next hidden state (used in recurrent policies).
        """
        # 1. 设置评估模式 (与 BasePolicy 相同)
        self.set_training_mode(False)

        # 2. 将观测转换为 PyTorch 张量 (与 BasePolicy 相同)
        # obs_to_tensor 内部会处理字典观测和图像归一化等
        obs_tensor, vectorized_env = self.obs_to_tensor(observation)

        # 3. 从 self._predict 获取 PyTorch 张量形式的动作元组
        # self._predict 应该返回 (discrete_actions_th, continuous_actions_th)
        with th.no_grad():
            actions = self._predict(obs_tensor, deterministic=deterministic)

        # 4. 将 PyTorch 张量转换为 NumPy 数组
        actions_np = (actions[0].cpu().numpy(), actions[1].cpu().numpy())

        # 5. 对连续动作部分进行反向缩放 (unscaling)
        continuous_actions_np = actions_np[1]

        # SAC Actor (使用 SquashedDiagGaussianDistribution) 输出的连续动作在 [-1, 1] 范围内。
        # 如果环境的连续动作空间 (self.action_space.spaces[1]) 的边界不是 [-1, 1]，
        # 我们需要将动作从 [-1, 1] 映射回环境期望的 [low, high] 范围。
        # self.squash_output 在 BasePolicy 初始化时被设为 True (因为 SAC Actor 输出是压缩的)

        # 获取连续动作的子空间
        continuous_space = self.action_space.spaces[1]
        assert isinstance(continuous_space, spaces.Box), (
            "The continuous part of the action space must be a Box."
        )

        if self.squash_output:
            # Unscale to the correct range
            low, high = continuous_space.low, continuous_space.high
            continuous_actions_np = low + (
                0.5 * (continuous_actions_np + 1.0) * (high - low)
            )
        # 如果不是矢量化环境，移除批处理维度
        if not vectorized_env:
            discrete_action = actions_np[0].squeeze(axis=0)
            continuous_action = continuous_actions_np.squeeze(axis=0)
            # 确保离散动作是标量整数
            if discrete_action.ndim == 0:
                actions_output = (int(discrete_action), continuous_action)
            else:
                actions_output = (discrete_action, continuous_action)
        else:
            # 如果是矢量化环境，返回两个NumPy数组的元组
            actions_output = (actions_np[0], continuous_actions_np)

        # # 6. 处理批处理维度 (如果输入不是批量观测)
        # if not vectorized_env:
        #     # 移除批处理维度 (通常是第0维)
        #     # 确保此时它们是 NumPy 数组
        #     assert isinstance(discrete_actions_np, np.ndarray)
        #     assert isinstance(continuous_actions_np, np.ndarray)

        #     # discrete_actions_np 的形状可能是 (1,) 或 (1, 1)
        #     # continuous_actions_np 的形状是 (1, continuous_dim)
        #     # squeeze() 会移除所有大小为1的维度，如果只想移除第一个，用 squeeze(axis=0)
        #     final_discrete_action = discrete_actions_np.squeeze(axis=0)
        #     final_continuous_action = continuous_actions_np.squeeze(axis=0)

        #     # 确保离散动作是标量整数（如果原始空间是 Discrete() 而不是 MultiDiscrete）
        #     if (
        #         isinstance(self.action_space.spaces[0], spaces.Discrete)
        #         and final_discrete_action.ndim == 0
        #     ):
        #         pass  # 已经是标量了
        #     elif (
        #         final_discrete_action.ndim == 1
        #         and final_discrete_action.shape[0] == 1
        #         and isinstance(self.action_space.spaces[0], spaces.Discrete)
        #     ):
        #         final_discrete_action = final_discrete_action[
        #             0
        #         ]  # 如果是 (1,) 形状的数组，取其元素

        #     actions_output = (final_discrete_action, final_continuous_action)
        # else:
        #     # 如果是批量观测 (vectorized_env is True)，
        #     # discrete_actions_np 的形状是 (batch_size,) 或 (batch_size, 1)
        #     # continuous_actions_np 的形状是 (batch_size, continuous_dim)
        #     # 我们直接返回这两个批量的 NumPy 数组组成的元组。
        #     # 调用者 (例如 VecEnv) 需要知道如何处理这个动作元组。
        #     # actions_output = (discrete_actions_np, continuous_actions_np)
        #     # 在源代码的/common/vec_env/dummy_vec_env.py 的step_wait的函数中
        #     # obs, self.buf_rews[env_idx], terminated, truncated, self.buf_infos[env_idx] = self.envs[env_idx].step(  # type: ignore[assignment]
        #     #     self.actions[env_idx]
        #     # )
        #     # 他只能对actions取到第一个元素 所以我们返回的元素，必须一个元素就是一个动作元组
        #     # 所以对他进行修改
        #     actions_output = self.combine_specific_actions(
        #         discrete_actions_np, continuous_actions_np
        #     )
        # # 7. 返回动作和状态

        return actions_output, state

    def combine_specific_actions(self, discrete_actions_np, continuous_actions_np):
        """
        将 discrete_actions_np 中的一个元素与 continuous_actions_np 中对应的一行元素
        （continuous_dim 个）组合成元组。

        参数:
        discrete_actions_np (np.ndarray): 形状为 (batch_size,) 或 (batch_size, 1) 的离散动作数组。
        continuous_actions_np (np.ndarray): 形状为 (batch_size, continuous_dim) 的连续动作数组。

        返回:
        tuple: 一个包含 (1 + continuous_dim) 元素元组的大元组。
            每个小元组的第一个元素来自 discrete_actions_np，
            其余元素来自 continuous_actions_np 的对应行。
        """
        # 检查输入是否为 NumPy 数组
        if not isinstance(discrete_actions_np, np.ndarray) or not isinstance(
            continuous_actions_np, np.ndarray
        ):
            raise TypeError(
                "输入 discrete_actions_np 和 continuous_actions_np 都必须是 NumPy ndarray。"
            )

        # 检查 batch_size (第一个维度) 是否一致
        batch_size_discrete = discrete_actions_np.shape[0]
        batch_size_continuous = continuous_actions_np.shape[0]

        if batch_size_discrete != batch_size_continuous:
            raise ValueError(
                f"两个数组的 batch_size (第一个维度) 必须一致: "
                f"discrete_actions_np 为 {batch_size_discrete}, "
                f"continuous_actions_np 为 {batch_size_continuous}。"
            )
        batch_size = batch_size_discrete

        # 如果 batch_size 为 0，返回空元组
        if batch_size == 0:
            return ()

        # 检查 continuous_actions_np 的维度是否为 2
        if continuous_actions_np.ndim != 2:
            raise ValueError(
                f"continuous_actions_np 的形状应为 (batch_size, continuous_dim)，"
                f"但当前维度为 {continuous_actions_np.ndim}。"
            )
        # continuous_dim = continuous_actions_np.shape[1] # continuous_dim

        # 处理 discrete_actions_np 的形状，统一为 (batch_size,)
        if discrete_actions_np.ndim == 2:
            if discrete_actions_np.shape[1] == 1:
                # 从 (batch_size, 1) 转换为 (batch_size,)
                discrete_elements = discrete_actions_np.squeeze(axis=1)
            else:
                raise ValueError(
                    "如果 discrete_actions_np 是二维数组, 其形状应为 (batch_size, 1)，"
                    f"但当前形状为 {discrete_actions_np.shape}。"
                )
        elif discrete_actions_np.ndim == 1:
            discrete_elements = discrete_actions_np
        else:
            raise ValueError(
                "discrete_actions_np 的形状应为 (batch_size,) 或 (batch_size, 1)，"
                f"但当前维度为 {discrete_actions_np.ndim}。"
            )

        list_of_combined_tuples = []
        # continuous_actions_np 的每一行已经是 (continuous_dim,) 的1D数组
        for discrete_action, continuous_action_row in zip(
            discrete_elements, continuous_actions_np
        ):
            # 将离散动作（标量）和连续动作的行（元组化）连接起来
            combined_tuple = (discrete_action, tuple(continuous_action_row))
            list_of_combined_tuples.append(combined_tuple)

        return tuple(list_of_combined_tuples)
