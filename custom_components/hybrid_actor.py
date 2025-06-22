import torch as th
import torch.nn as nn
from gymnasium import spaces
from typing import Tuple, List, Type, Any, Dict

from stable_baselines3.common.policies import BasePolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor, create_mlp
from stable_baselines3.common.distributions import (
    CategoricalDistribution,
    SquashedDiagGaussianDistribution,
)
from stable_baselines3.common.type_aliases import PyTorchObs
from stable_baselines3.sac.policies import LOG_STD_MAX, LOG_STD_MIN



class HybridActor(BasePolicy):
    """
    最终版：采用“动作嵌入”架构的Actor网络。
    它提供了一个与标准SAC兼容的action_log_prob接口。
    """

    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Tuple,
        net_arch: List[int],
        features_extractor: BaseFeaturesExtractor,
        features_dim: int,
        activation_fn: Type[nn.Module] = nn.Tanh,
        log_std_init: float = -2.0,
        embedding_dim: int = 32,
        normalize_images: bool = True,
        use_sde: bool = False,
        full_std: bool = True,
        use_expln: bool = False,
        clip_mean: float = 2.0,
    ):
        super().__init__(
            observation_space,
            action_space,
            features_extractor=features_extractor,
            squash_output=True,
            normalize_images=normalize_images,
        )

        self.discrete_dim = self.action_space.spaces[0].n
        self.continuous_dim = self.action_space.spaces[1].shape[0]

        self.use_sde = use_sde
        self.sde_features_extractor = None
        self.net_arch = net_arch
        self.features_dim = features_dim
        self.log_std_init = log_std_init
        self.activation_fn = activation_fn
        self.use_expln = use_expln
        self.full_std = full_std
        self.clip_mean = clip_mean

        # 共享主体网络
        latent_pi_net = create_mlp(features_dim, -1, net_arch, activation_fn)
        self.latent_pi = nn.Sequential(*latent_pi_net)
        last_layer_dim = net_arch[-1] if net_arch else features_dim

        # 定义网络头部
        self.discrete_head = nn.Linear(last_layer_dim, self.discrete_dim)
        self.action_embedding = nn.Embedding(self.discrete_dim, embedding_dim)

        unified_continuous_head_input_dim = last_layer_dim + embedding_dim
        self.mean_net = nn.Linear(
            unified_continuous_head_input_dim, self.continuous_dim
        )
        # 为了稳定性，让log_std成为一个独立的可学习参数，而不是网络的输出
        self.log_std = nn.Parameter(
            th.ones(self.continuous_dim) * log_std_init, requires_grad=True
        )

    def _get_dists(
        self, latent_pi: th.Tensor, discrete_action: th.Tensor
    ) -> Tuple[CategoricalDistribution, SquashedDiagGaussianDistribution]:
        """一个辅助方法，用于获取两个底层的分布对象"""
        discrete_logits = self.discrete_head(latent_pi)
        categorical_dist = CategoricalDistribution(action_logits=discrete_logits)

        action_emb = self.action_embedding(discrete_action.long())
        conditional_input = th.cat([latent_pi, action_emb], dim=1)
        mean = self.mean_net(conditional_input)

        log_std = th.clamp(self.log_std, LOG_STD_MIN, LOG_STD_MAX)
        gaussian_dist = SquashedDiagGaussianDistribution(self.continuous_dim)
        gaussian_dist.proba_distribution(mean, log_std)  # 设置高斯分布的参数

        return categorical_dist, gaussian_dist

    def get_action_dist_from_obs(
        self, obs: PyTorchObs, discrete_action: th.Tensor = None
    ) -> Tuple[CategoricalDistribution, SquashedDiagGaussianDistribution, th.Tensor]:
        """
        一个统一的辅助方法，从观测obs中计算并返回两个分布。
        如果提供了discrete_action，则连续分布基于它生成；否则，基于新采样的生成。
        """
        # 1. 提取特征，计算logits (只计算一次)
        features = self.extract_features(
            obs, self.features_extractor
        )  # features_extractor 在 BasePolicy 中定义
        latent_pi = self.latent_pi(features)
        discrete_logits = self.discrete_head(latent_pi)

        # 2. 创建离散分布
        categorical_dist = CategoricalDistribution(self.discrete_dim)
        categorical_dist.proba_distribution(action_logits=discrete_logits)

        # 3. 如果没有提供离散动作，就从刚创建的分布中采样一个
        if discrete_action is None:
            discrete_action = categorical_dist.sample()

        # 4. 根据离散动作，计算连续分布
        action_emb = self.action_embedding(discrete_action.long())
        conditional_input = th.cat([latent_pi, action_emb], dim=1)
        mean = self.mean_net(conditional_input)
        log_std = th.clamp(self.log_std, LOG_STD_MIN, LOG_STD_MAX)

        gaussian_dist = SquashedDiagGaussianDistribution(self.continuous_dim)
        gaussian_dist.proba_distribution(mean, log_std)

        return categorical_dist, gaussian_dist, discrete_action

    def action_log_prob(
        self, obs: PyTorchObs
    ) -> Tuple[Tuple[th.Tensor, th.Tensor], th.Tensor]:
        """
        这是被SAC.train()调用的核心方法。
        它采样一个动作，并返回这个动作及其对数概率。
        """
        categorical_dist, continuous_dist, discrete_action = (
            self.get_action_dist_from_obs(obs)
        )
        continuous_action = continuous_dist.sample()
        actions = (discrete_action, continuous_action)

        # 计算这个被采样出的动作的对数概率
        log_prob_discrete = categorical_dist.log_prob(discrete_action)
        log_prob_continuous = continuous_dist.log_prob(continuous_action)
        total_log_prob = log_prob_discrete + log_prob_continuous

        return actions, total_log_prob

    def forward(
        self, obs: PyTorchObs, deterministic: bool = False
    ) -> Tuple[th.Tensor, th.Tensor]:
        """
        forward只负责生成一个动作，用于与环境交互。
        """
        # 把obs传入，设为distribution的参数
        categorical_dist, continuous_dist, discrete_action = (
            self.get_action_dist_from_obs(obs)
        )
        # 如果需要确定性动作，我们还需要重新计算连续分布
        if deterministic:
            # 离散动作取众数
            discrete_action = categorical_dist.mode()
            # 重新获取对应的连续分布
            _, continuous_dist, _ = self.get_action_dist_from_obs(obs, discrete_action)
            # 连续动作取众数
            continuous_action = continuous_dist.mode()
        else:
            # 随机采样连续动作
            continuous_action = continuous_dist.sample()

        return (discrete_action, continuous_action)

    def _predict(
        self, observation: PyTorchObs, deterministic: bool = False
    ) -> Tuple[th.Tensor, th.Tensor]:
        return self.forward(observation, deterministic)

    def _get_constructor_parameters(self) -> Dict[str, Any]:  # 使用 Dict
        data = super()._get_constructor_parameters()
        data.update(
            dict(
                net_arch=self.net_arch,
                # features_dim=self.features_dim,  # 这个应该由父类的 features_extractor 确定
                activation_fn=self.activation_fn,
                log_std_init=self.log_std_init,
                # features_extractor 在 BasePolicy 中处理
            )
        )
        return data
