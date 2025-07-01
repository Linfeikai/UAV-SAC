# =====================================================================================
# 文件名: diffusion_sac_policy.py (最终修复版本)
# 描述: 通过重写_build方法，完全掌控Actor和Critic的创建过程，彻底解决参数传递问题。
# =====================================================================================

import torch as th
import torch.nn as nn
from gymnasium import spaces
from typing import Dict, Any, List, Type, Optional, Union
import numpy as np

# 导入我们新定义的Actor和Critic
from hybrid_actor import DiffusionPolicyActor
from hybrid_critic import ContinuousCritic

# 从SB3导入必要的基类和类型提示
from stable_baselines3.common.policies import BasePolicy
from stable_baselines3.common.torch_layers import (
    BaseFeaturesExtractor,
    get_actor_critic_arch,
    FlattenExtractor,
)
from stable_baselines3.common.type_aliases import Schedule, PyTorchObs


# =====================================================================================
# 新策略: DiffusionSACPolicy
# 描述: 这是一个专门为 DiffusionSACAgent 设计的策略。
#       它将 DiffusionPolicyActor 和 ContinuousCritic 组合在一起。
#       这个策略只处理连续动作空间 (Box)。
# =====================================================================================
class DiffusionSACPolicy(BasePolicy):
    """
    基于扩散模型的SAC策略 (仅支持连续动作空间).
    """

    actor: DiffusionPolicyActor
    critic: ContinuousCritic
    critic_target: ContinuousCritic

    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Box,  # <-- 明确指定为Box
        lr_schedule: Schedule,
        net_arch: Optional[Union[List[int], Dict[str, List[int]]]] = None,
        activation_fn: Type[nn.Module] = nn.ReLU,
        normalize_images: bool = True,
        n_critics: int = 2,
        features_extractor_class: type[BaseFeaturesExtractor] = FlattenExtractor,
        features_extractor_kwargs: Optional[dict[str, Any]] = None,
        share_features_extractor: bool = True,
        # --- Diffusion Actor 特有的参数 ---
        T: int = 5,
        beta_schedule: str = "linear",
    ):
        # 调用父类构造函数，但只传递它能安全处理的核心参数
        super().__init__(
            observation_space,
            action_space,
            features_extractor_class,
            features_extractor_kwargs,
        )

        # --- 手动设置所有其他必要的属性 ---
        # 绕过在super().__init__中传递它们时可能引发的TypeError
        self.normalize_images = normalize_images
        self.share_features_extractor = share_features_extractor

        # SAC/DiffusionSAC 策略需要将动作压缩到 [-1, 1] 范围
        # 'squash_output' 是一个只读属性，所以我们必须设置底层的私有变量
        self._squash_output = True

        if net_arch is None:
            net_arch = [256, 256]
        self.activation_fn = activation_fn
        self.n_critics = n_critics

        # -- 保存 Diffusion Actor 的参数 --
        self.T = T
        self.beta_schedule = beta_schedule

        # actor和critic共同的参数
        self.net_args = {
            "observation_space": self.observation_space,
            "action_space": self.action_space,
            "activation_fn": self.activation_fn,
        }
        self.actor_kwargs = self.net_args.copy()
        self.critic_kwargs = self.net_args.copy()

        # 从 net_arch 中分离 actor 和 critic 的网络结构
        # get_actor_critic_arch 是一个辅助函数，如果 net_arch 是列表，则两者共享；
        # 如果是字典 {'pi': [...], 'vf': [...]} (或 'actor', 'critic')，则分别使用。
        actor_arch, critic_arch = get_actor_critic_arch(net_arch)
        # 更新 actor 和 critic 各自的参数
        self.actor_kwargs.update(
            {
                "net_arch": actor_arch,
                "T_steps": self.T,
            }
        )
        self.critic_kwargs.update(
            {
                "net_arch": critic_arch,
            }
        )

        self._build(lr_schedule)

    def _build(self, lr_schedule: Schedule) -> None:
        """
        创建actor, critic, 和它们的优化器.
        """
        # Note: features_extractor is already initialized at this point
        if self.share_features_extractor:
            self.actor = self.make_actor(features_extractor=self.features_extractor)
            # The features extractor is shared, so the critic will use the same one
            self.critic = self.make_critic(features_extractor=self.features_extractor)
        else:
            # Create separate features extractors for actor and critic
            self.actor = self.make_actor(features_extractor=None)
            self.critic = self.make_critic(features_extractor=None)

        self.critic_target = self.make_critic(
            features_extractor=self.critic.features_extractor
        )
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_target.set_training_mode(False)  # Freeze target network

        # Setup optimizers
        self.actor.optimizer = self.optimizer_class(
            self.actor.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs
        )
        self.critic.optimizer = self.optimizer_class(
            self.critic.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs
        )

        self.to(self.device)

    def make_actor(
        self, features_extractor: Optional[BaseFeaturesExtractor] = None
    ) -> DiffusionPolicyActor:
        """
        创建 Diffusion Actor.
        """
        # 把features_extractor和features_dim参数更新到actor的参数dict里面
        actor_kwargs = self._update_features_extractor(
            self.actor_kwargs, features_extractor
        )
        return DiffusionPolicyActor(**actor_kwargs).to(self.device)

    def make_critic(
        self, features_extractor: Optional[BaseFeaturesExtractor] = None
    ) -> ContinuousCritic:
        """
        创建 Critic 网络.
        """
        critic_kwargs = self._update_features_extractor(
            self.critic_kwargs, features_extractor
        )
        return ContinuousCritic(**critic_kwargs).to(self.device)

    def forward(self, obs: PyTorchObs, deterministic: bool = False) -> th.Tensor:
        """
        前向传播.
        """
        return self._predict(obs, deterministic=deterministic)

    def _predict(
        self, observation: PyTorchObs, deterministic: bool = False
    ) -> th.Tensor:
        """
        获取动作. 这是 `predict()` 的核心.
        """
        return self.actor(observation, deterministic)

    def set_training_mode(self, mode: bool) -> None:
        """
        将策略及其子模块设置为训练模式或评估模式.
        """
        self.actor.set_training_mode(mode)
        self.critic.set_training_mode(mode)
        self.training = mode

    def _get_constructor_parameters(self):
        return super()._get_constructor_parameters()
