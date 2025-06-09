from stable_baselines3.common.policies import BasePolicy
from stable_baselines3.common.torch_layers import (
    create_mlp,
    BaseFeaturesExtractor,
)  # 移除了 get_actor_critic_arch

# 从 SB3 的 SAC 策略中导入 LOG_STD_MIN/MAX
from stable_baselines3.sac.policies import LOG_STD_MAX, LOG_STD_MIN, get_action_dim
from stable_baselines3.common.type_aliases import PyTorchObs  # 用于类型提示
import torch as th
import torch.nn as nn
from typing import Tuple, Any, Dict, List, Type
from gymnasium import spaces
from .hybrid_distribution import make_hybrid_proba_distribution


class HybridActor(BasePolicy):  # 为了更多控制，继承自 BasePolicy
    action_space: spaces.Tuple  # 期望一个元组空间

    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Tuple,  # 期望一个元组空间
        net_arch: List[int],  # 使用 List 而不是 list
        features_extractor: BaseFeaturesExtractor,
        features_dim: int,
        activation_fn: Type[nn.Module] = nn.ReLU,  # 使用 Type
        normalize_images: bool = True,
        log_std_init: float = -3,  # 用于连续部分
        use_sde: bool = False,
        full_std: bool = True,
        use_expln: bool = False,
        clip_mean: float = 2.0,
    ):
        # 依赖传入的 features_extractor来获取这个值
        # self.features_dim = features_extractor.features_dim
        super().__init__(
            observation_space,
            action_space,
            features_extractor=features_extractor,
            normalize_images=normalize_images,
            squash_output=True,  # 对 SAC 连续部分很重要
        )

        if not isinstance(action_space, spaces.Tuple) or len(action_space.spaces) != 2:
            raise ValueError(
                "HybridActor 需要一个包含2个元素（Discrete, Box）的元组动作空间。"
            )
        self.discrete_space = action_space.spaces[0]
        self.continuous_space = action_space.spaces[1]
        if not isinstance(self.discrete_space, spaces.Discrete):
            raise ValueError("动作空间元组的第一个元素必须是 Discrete。")
        if not isinstance(self.continuous_space, spaces.Box):
            raise ValueError("动作空间元组的第二个元素必须是 Box。")

        # 目前是冗余的，先注释掉吧。
        self.discrete_dim = self.discrete_space.n
        self.continuous_dim = get_action_dim(self.continuous_space)

        self.use_sde = use_sde
        self.sde_features_extractor = None
        self.net_arch = net_arch
        self.features_dim = features_dim
        self.log_std_init = log_std_init
        self.net_arch = net_arch
        self.activation_fn = activation_fn
        self.use_expln = use_expln
        self.full_std = full_std
        self.clip_mean = clip_mean

        # 共享主体
        actor_arch_actual = net_arch  # 比如[256,256]
        latent_pi_net = create_mlp(
            self.features_dim, -1, actor_arch_actual, activation_fn
        )
        # latent_pi就是我们的主体网络
        self.latent_pi = nn.Sequential(*latent_pi_net)
        last_layer_dim = (
            actor_arch_actual[-1] if len(actor_arch_actual) > 0 else self.features_dim
        )

        # actor会有一个action_dist属性，这个属性就是hybrid_distribution实例
        self.action_dist = make_hybrid_proba_distribution(action_space)

        # 核心修改1：直接创建分离的头部网络

        # 1.离散动作头（保持不变，只有一个）
        # 根据状态特征，决定选择每个离散动作的倾向性
        self.action_net_discrete_logits = nn.Linear(last_layer_dim, self.discrete_dim)

        # 连续动作头

        # 我们为每个离散动作创建一个专属的 mean 和 log_std 网络头
        self.mean_heads = nn.ModuleList(
            [
                nn.Linear(last_layer_dim, self.continuous_dim)
                for _ in range(self.discrete_dim)
            ]
        )
        self.log_std_heads = nn.ModuleList(
            [
                nn.Linear(last_layer_dim, self.continuous_dim)
                for _ in range(self.discrete_dim)
            ]
        )
        self.action_dist = make_hybrid_proba_distribution(action_space)

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

    def get_action_dist_params(
        self, obs: PyTorchObs
    ) -> Tuple[th.Tensor, List[th.Tensor], List[th.Tensor]]:
        """
        获取动作分布的参数。
        返回的是一个离散logits张量,和N个连续mean/log_std张量的列表。
        """
        # 首先提取特征
        features = self.extract_features(
            obs, self.features_extractor
        )  # features_extractor 在 BasePolicy 中定义
        latent_pi = self.latent_pi(features)

        # 当前这一批数据的动作分布参数。
        discrete_logits = self.action_net_discrete_logits(latent_pi)
        means = []
        log_stds = []
        for i in range(self.discrete_dim):
            # latent_pi 直接送入每个头，不再有任何拼接
            mean = self.mean_heads[i](latent_pi)
            log_std = self.log_std_heads[i](latent_pi)

            # 对 log_std 进行 clamp 是SAC的常规操作，保持稳定性
            log_stds.append(th.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX))
            means.append(mean)

        # 返回新的参数结构
        return discrete_logits, means, log_stds

    def forward(
        self, obs: PyTorchObs, deterministic: bool = False
    ) -> Tuple[th.Tensor, th.Tensor]:
        discrete_logits, means, log_stds, kwargs = self.get_action_dist_params(obs)
        # 其实就是我们先传入actor网络，得到动作分布参数，然后传入action_dist
        # 这里的action_dist是一个hybrid_distribution实例
        return self.action_dist.actions_from_params(
            discrete_logits,
            continuous_mean=means,
            continuous_log_std=log_stds,
            deterministic=deterministic,
            **kwargs,
        )

    def action_log_prob(
        self, obs: PyTorchObs
    ) -> Tuple[Tuple[th.Tensor, th.Tensor], th.Tensor]:
        discrete_logits, means, log_stds, kwargs = self.get_action_dist_params(obs)
        # log_prob_from_params 将采样动作并返回这些动作及其对数概率
        actions, log_prob = self.action_dist.log_prob_from_params(
            discrete_logits,
            continuous_mean=means,
            continuous_log_std=log_stds,
            **kwargs,
        )
        return actions, log_prob

    def _predict(
        self, observation: PyTorchObs, deterministic: bool = False
    ) -> Tuple[th.Tensor, th.Tensor]:
        return self.forward(observation, deterministic=deterministic)
