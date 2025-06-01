import torch as th
import torch.nn as nn
from gymnasium import spaces
from typing import Tuple, List, Type  # Python 3.9+可以直接用list, tuple, type

from stable_baselines3.common.torch_layers import BaseFeaturesExtractor, create_mlp
from stable_baselines3.common.preprocessing import get_action_dim  #
from stable_baselines3.common.policies import (
    BaseModel,
)  # BaseModel在较新版本SB3中，或者可以直接继承nn.Module


class HybridCritic(BaseModel):  # 或者 nn.Module
    """
    Critic network(s) for SAC/TD3 with hybrid action spaces (Discrete + Continuous).
    It represents the action-state value function (Q-value function).
    It takes the state, discrete action, and continuous action as input.

    :param observation_space: Observation space
    :param action_space: Action space (expected to be Tuple(Discrete, Box))
    :param net_arch: Network architecture
    :param features_extractor: Network to extract features
    :param features_dim: Number of features
    :param activation_fn: Activation function
    :param normalize_images: Whether to normalize images or not
    :param n_critics: Number of critic networks to create.
    :param share_features_extractor: Whether the features extractor is shared or not
    """

    features_extractor: BaseFeaturesExtractor  # type: ignore[assignment]

    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Tuple,  # 明确为 Tuple 类型
        net_arch: List[int],  # Python 3.9+可以用 list[int]
        features_extractor: BaseFeaturesExtractor,
        features_dim: int,
        activation_fn: Type[nn.Module] = nn.ReLU,  # Python 3.9+可以用 type[nn.Module]
        normalize_images: bool = True,
        n_critics: int = 2,
        share_features_extractor: bool = True,
    ):
        super().__init__(
            observation_space,
            action_space,  # 父类可能不直接支持Tuple，但我们在这里处理
            features_extractor=features_extractor,
            normalize_images=normalize_images,
        )

        if not isinstance(action_space, spaces.Tuple) or len(action_space.spaces) != 2:
            raise ValueError(
                "HybridCritic requires a Tuple action space with 2 elements (Discrete, Box)."
            )

        self.discrete_action_space = action_space.spaces[0]
        self.continuous_action_space = action_space.spaces[1]
        self.features_dim = features_dim

        if not isinstance(self.discrete_action_space, spaces.Discrete):
            raise ValueError("First element of action_space Tuple must be Discrete.")
        if not isinstance(self.continuous_action_space, spaces.Box):
            raise ValueError("Second element of action_space Tuple must be Box.")

        # 维度计算
        self.discrete_action_dim = int(self.discrete_action_space.n)
        # 使用one-hot编码，所以离散动作的输入维度是其类别数
        self.one_hot_discrete_action_dim = self.discrete_action_dim
        self.continuous_action_dim = get_action_dim(self.continuous_action_space)

        self.share_features_extractor = share_features_extractor
        self.n_critics = n_critics
        self.q_networks: List[nn.Module] = []  # Python 3.9+可以用 list[nn.Module]

        # 总的动作相关输入维度 = one-hot离散动作维度 + 连续动作维度
        total_action_related_dim = (
            self.one_hot_discrete_action_dim + self.continuous_action_dim
        )

        for idx in range(n_critics):
            # Q网络的输入维度 = 状态特征维度 + 总的动作相关输入维度
            q_net_input_dim = self.features_dim + total_action_related_dim
            q_net_list = create_mlp(q_net_input_dim, 1, net_arch, activation_fn)
            q_net = nn.Sequential(*q_net_list)
            self.add_module(f"qf{idx}", q_net)
            self.q_networks.append(q_net)

    def forward(
        self, obs: th.Tensor, actions: Tuple[th.Tensor, th.Tensor]
    ) -> Tuple[th.Tensor, ...]:
        """
        :param obs: Observations
        :param actions: Tuple of (discrete_actions, continuous_actions)
                        discrete_actions: Tensor of shape (batch_size, 1) or (batch_size,) with integer action indices.
                        continuous_actions: Tensor of shape (batch_size, continuous_action_dim).
        :return: Tuple of Q-values from each critic network.
        """
        discrete_actions, continuous_actions = actions

        # 确保离散动作是 LongTensor 以用于 one_hot
        discrete_actions = discrete_actions.long()
        if (
            discrete_actions.ndim == 1
        ):  # 如果是 (batch_size,)，增加一个维度变为 (batch_size, 1)
            discrete_actions = discrete_actions.unsqueeze(1)

        # 将离散动作转换为 one-hot 编码
        # discrete_actions 应该是 (batch_size, 1) 包含类别索引
        # self.discrete_action_dim 是类别的数量
        discrete_actions_one_hot = th.nn.functional.one_hot(
            discrete_actions.squeeze(1), num_classes=self.discrete_action_dim
        ).float()
        # discrete_actions_one_hot 的形状将是 (batch_size, self.discrete_action_dim)

        with th.set_grad_enabled(not self.share_features_extractor):
            features = self.extract_features(
                obs, self.features_extractor
            )  # 在 SB3 3.0+ self.features_extractor 参数已移除

        # 拼接: 状态特征, one-hot离散动作, 连续动作
        qvalue_input = th.cat(
            [features, discrete_actions_one_hot, continuous_actions], dim=1
        )

        return tuple(q_net(qvalue_input) for q_net in self.q_networks)

    def q1_forward(
        self, obs: th.Tensor, actions: Tuple[th.Tensor, th.Tensor]
    ) -> th.Tensor:
        """
        Only predict the Q-value using the first network.
        """
        discrete_actions, continuous_actions = actions
        discrete_actions = discrete_actions.long()
        if discrete_actions.ndim == 1:
            discrete_actions = discrete_actions.unsqueeze(1)

        discrete_actions_one_hot = th.nn.functional.one_hot(
            discrete_actions.squeeze(1), num_classes=self.discrete_action_dim
        ).float()

        with (
            th.no_grad()
        ):  # 在SB3 3.0+中，extract_features通常在no_grad上下文之外，因为它可能被训练
            features = self.extract_features(
                obs, self.features_extractor
            )  # 在 SB3 3.0+ self.features_extractor 参数已移除

        qvalue_input = th.cat(
            [features, discrete_actions_one_hot, continuous_actions], dim=1
        )
        return self.q_networks[0](qvalue_input)

    # 如果你的BaseModel没有实现_get_constructor_parameters，你可能需要添加它
    # 或者确保你的SAC算法的policy_kwargs能正确传递参数给这个Critic
    def _get_constructor_parameters(self) -> dict:
        data = super()._get_constructor_parameters()  # 如果父类有这个方法
        # data.update(dict( # 根据父类实现来更新
        #    net_arch=self.net_arch, # 这些通常由父类处理或在policy中管理
        #    activation_fn=self.activation_fn,
        #    n_critics=self.n_critics,
        #    share_features_extractor=self.share_features_extractor,
        # ))
        return data
