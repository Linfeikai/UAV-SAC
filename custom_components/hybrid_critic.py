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
        embedding_dim: int = 32,  # 修改一：嵌入层，添加这个参数
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

        discrete_action_dim = self.discrete_action_space.n
        continuous_action_dim = get_action_dim(self.continuous_action_space)
        self.share_features_extractor = share_features_extractor


        if not isinstance(self.discrete_action_space, spaces.Discrete):
            raise ValueError("First element of action_space Tuple must be Discrete.")
        if not isinstance(self.continuous_action_space, spaces.Box):
            raise ValueError("Second element of action_space Tuple must be Box.")

        # --- 核心修改 2: 创建一个可学习的嵌入层 ---
        # 这是我们的“智能档案袋”
        self.action_embedding = nn.Embedding(discrete_action_dim, embedding_dim)

        self.q_networks: List[nn.Module] = []

        # --- 核心修改 3: 计算新的Q网络输入维度 ---
        # 新的维度 = 状态特征 + 动作嵌入 + 连续动作
        q_net_input_dim = features_dim + embedding_dim + continuous_action_dim

        for idx in range(n_critics):
            # 使用新的输入维度创建Q网络
            q_net_list = create_mlp(q_net_input_dim, 1, net_arch, activation_fn)
            q_net = nn.Sequential(*q_net_list)
            self.add_module(f"qf{idx}", q_net)
            self.q_networks.append(q_net)

    def forward(
        self, obs: th.Tensor, actions: Tuple[th.Tensor, th.Tensor]
    ) -> Tuple[th.Tensor, ...]:
        """
        :param obs: 观测
        :param actions: 动作元组 (discrete_actions, continuous_actions)
        :return: 每个Q网络的Q值输出元组
        """
        # --- 核心修改 4: 全新的前向传播逻辑 ---

        # 1. 解包动作
        discrete_actions, continuous_actions = actions

        # 确保离散动作为Long类型以便查询嵌入层
        # .squeeze(-1)是为了处理(B, 1) -> (B,)的形状，以匹配embedding层的输入要求
        discrete_actions_long = discrete_actions.long().squeeze(-1)

        # 2. 提取状态特征 (使用父类的方法)
        with th.set_grad_enabled(not self.share_features_extractor):
            features = self.extract_features(obs, self.features_extractor)

        # 3. 查询离散动作的嵌入向量
        action_embeds = self.action_embedding(discrete_actions_long)

        # 4. 拼接所有信息作为Q网络的输入
        qvalue_input = th.cat([features, action_embeds, continuous_actions], dim=1)

        # 5. 将整合后的特征送入两个Q网络并返回结果
        return tuple(q_net(qvalue_input) for q_net in self.q_networks)

    def q1_forward(
        self, obs: th.Tensor, actions: Tuple[th.Tensor, th.Tensor]
    ) -> th.Tensor:
        """只用第一个Q网络进行前向传播，主要用于Actor更新。"""
        # 这里的逻辑与forward完全一致，只是只返回第一个网络的输出
        discrete_actions, continuous_actions = actions
        discrete_actions_long = discrete_actions.long().squeeze(-1)

        with th.no_grad():
            features = self.extract_features(obs, self.features_extractor)
            action_embeds = self.action_embedding(discrete_actions_long)

        qvalue_input = th.cat([features, action_embeds, continuous_actions], dim=1)
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
