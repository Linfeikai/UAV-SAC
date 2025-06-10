import numpy as np
import torch as th
from torch import nn
from gymnasium import spaces
from typing import Tuple, Optional, List, Any, Dict

from stable_baselines3.common.distributions import (
    Distribution,
    CategoricalDistribution,
    SquashedDiagGaussianDistribution,
)
from stable_baselines3.sac.policies import LOG_STD_MAX, LOG_STD_MIN


class HybridDistribution(Distribution):
    """
    为多头Actor设计的混合动作空间概率分布。
    管理一个离散分类分布和N个独立的连续高斯分布。
    """

    def __init__(self, discrete_dim: int, continuous_dim: int):
        super().__init__()
        self.discrete_dim = discrete_dim
        self.continuous_dim = continuous_dim

        # 离散部分保持不变
        self.cat_dist = CategoricalDistribution(discrete_dim)

        # --- 核心修改 1: 创建N个独立的连续分布 ---
        # 原先是一个实例，现在是一个包含N个实例的ModuleList
        self.gauss_dists = [
            SquashedDiagGaussianDistribution(continuous_dim)
            for _ in range(discrete_dim)
        ]

    def proba_distribution(
        self,
        discrete_logits: th.Tensor,
        # --- 核心修改 2: 接收列表形式的连续参数 ---
        continuous_means: List[th.Tensor],
        continuous_log_stds: List[th.Tensor],
    ) -> "HybridDistribution":
        """
        根据多头Actor输出的参数，设置分布。
        """
        # 1. 设置离散分布的参数 (不变)
        self.cat_dist.proba_distribution(discrete_logits)

        # 2. 循环设置N个连续分布的参数
        for i in range(self.discrete_dim):
            mean = continuous_means[i]
            log_std = continuous_log_stds[i]
            # Clamp log_std for stability
            clamped_log_std = th.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)
            self.gauss_dists[i].proba_distribution(mean, clamped_log_std)

        return self

    def log_prob(self, actions: Tuple[th.Tensor, th.Tensor]) -> th.Tensor:
        """
        计算给定混合动作的对数概率。
        这是技术上最复杂的部分，需要矢量化处理。
        """
        discrete_action, continuous_action = actions

        # 1. 计算离散动作的对数概率 (不变)
        log_prob_discrete = self.cat_dist.log_prob(discrete_action)

        # --- 核心修改 3: 矢量化计算连续动作的对数概率 ---
        # a. 计算所有N个高斯分布对于给定continuous_action的log_prob
        #    这将得到一个 [B, N] 的张量，B是批量大小
        all_continuous_log_probs = th.stack(
            [dist.log_prob(continuous_action) for dist in self.gauss_dists], dim=1
        )

        # b. 使用离散动作作为索引，从 [B, N] 的张量中选出正确的log_prob
        #    discrete_action的形状是 [B] or [B, 1]，需要是LongTensor作为索引
        discrete_action_long = discrete_action.long().reshape(-1, 1)

        # th.gather会根据discrete_action_long中的索引，在dim=1上选取元素
        log_prob_continuous = th.gather(
            all_continuous_log_probs, 1, discrete_action_long
        ).squeeze(-1)

        return log_prob_discrete + log_prob_continuous

    def sample(self) -> Tuple[th.Tensor, th.Tensor]:
        """
        从混合分布中采样一个动作。
        """
        # 1. 采样离散动作 (不变)
        discrete_sample = self.cat_dist.sample()

        # --- 核心修改 4: 矢量化采样连续动作 ---
        # a. 从所有N个高斯分布中都采样一次
        #    这将得到一个 [N, B, continuous_dim] 的张量
        all_continuous_samples = th.stack(
            [dist.sample() for dist in self.gauss_dists], dim=0
        )

        # b. 我们需要根据离散样本，为批次中的每个元素挑选出正确的连续样本
        #    这需要一些索引技巧
        batch_size = discrete_sample.shape[0]
        # 创建一个从0到B-1的索引
        batch_indices = th.arange(batch_size, device=discrete_sample.device)
        # 使用离散样本和批次索引来定位正确的连续样本
        continuous_sample = all_continuous_samples[
            discrete_sample.long(), batch_indices, :
        ]

        return discrete_sample, continuous_sample

    def mode(self) -> Tuple[th.Tensor, th.Tensor]:
        """
        返回最可能的动作（确定性模式）。
        """
        discrete_mode = self.cat_dist.mode()

        # 与sample()的逻辑相同，只是调用.mode()
        all_continuous_modes = th.stack(
            [dist.mode() for dist in self.gauss_dists], dim=0
        )
        batch_size = discrete_mode.shape[0]
        batch_indices = th.arange(batch_size, device=discrete_mode.device)
        continuous_mode = all_continuous_modes[discrete_mode.long(), batch_indices, :]

        return discrete_mode, continuous_mode

    # --- 核心修改 5: 更新辅助函数的签名 ---
    # 这两个函数是给SAC算法主体调用的，必须更新它们的接口
    def actions_from_params(
        self,
        discrete_logits: th.Tensor,
        continuous_means: List[th.Tensor],
        continuous_log_stds: List[th.Tensor],
        deterministic: bool = False,
        **kwargs,
    ) -> Tuple[th.Tensor, th.Tensor]:
        self.proba_distribution(discrete_logits, continuous_means, continuous_log_stds)
        return self.get_actions(deterministic=deterministic)

    def log_prob_from_params(
        self,
        discrete_logits: th.Tensor,
        continuous_means: List[th.Tensor],
        continuous_log_stds: List[th.Tensor],
        **kwargs,
    ) -> Tuple[Tuple[th.Tensor, th.Tensor], th.Tensor]:
        actions = self.actions_from_params(
            discrete_logits, continuous_means, continuous_log_stds, deterministic=False
        )
        log_prob = self.log_prob(actions)
        return actions, log_prob

    # --- 核心修改 6: 删除不再需要的方法 ---
    # 这个方法现在应该在Actor类中实现，而不是在分布中
    # def proba_distribution_net(...):
    #     ...
    # 我们现在直接在actor中新建网络
    # 我们直接删掉它，以避免混淆

    # --- 因为我们是从sb3的distribution里面继承的，所以需要实现如下方法 实际上我们并不会用到 ---
    def proba_distribution_net(self, *args, **kwargs) -> None:
        """
        This method is required by the base Distribution class.
        In our multi-head architecture, the network heads are created directly
        in the HybridActor class, so this method is not used.
        We add it here as an empty placeholder to satisfy the inheritance requirement.
        """
        # We do nothing here because the heads are managed by the Actor.
        pass

    # --- 同上 ---
    def entropy(self) -> None:
        """
        The entropy for a SquashedDiagGaussianDistribution is not analytically
        computable and returns None in the SB3 implementation. Therefore, the total
        entropy for our hybrid distribution is also not defined and we return None.
        """
        # The entropy is not defined for the squashed gaussian,
        # and therefore not for the whole distribution.
        return None


# Helper function 保持不变，它只负责创建我们的HybridDistribution实例
def make_hybrid_proba_distribution(
    action_space: spaces.Tuple,
) -> HybridDistribution:
    if not isinstance(action_space, spaces.Tuple) or len(action_space.spaces) != 2:
        raise ValueError("Requires a Tuple action space (Discrete, Box).")

    discrete_space, continuous_space = action_space.spaces

    if not isinstance(discrete_space, spaces.Discrete) or not isinstance(
        continuous_space, spaces.Box
    ):
        raise ValueError("Tuple must contain Discrete and Box spaces.")

    return HybridDistribution(
        discrete_dim=int(discrete_space.n),
        continuous_dim=int(np.prod(continuous_space.shape)),
    )


# --- 新的测试代码 ---
if __name__ == "__main__":
    batch_size = 32
    discrete_dim = 5
    continuous_dim = 8

    # 1. 创建分布实例
    hybrid_dist = HybridDistribution(discrete_dim, continuous_dim)

    # 2. 模拟多头Actor的输出
    # 离散部分的logits
    mock_discrete_logits = th.randn(batch_size, discrete_dim)
    # 连续部分的N个mean和log_std列表
    mock_continuous_means = [
        th.randn(batch_size, continuous_dim) for _ in range(discrete_dim)
    ]
    mock_continuous_log_stds = [
        th.randn(batch_size, continuous_dim) for _ in range(discrete_dim)
    ]

    # 3. 设置分布参数
    hybrid_dist.proba_distribution(
        mock_discrete_logits, mock_continuous_means, mock_continuous_log_stds
    )
    print("Distribution parameters set successfully.")

    # 4. 测试采样
    discrete_action, continuous_action = hybrid_dist.sample()
    print("\n--- Testing sample() ---")
    print(f"Sampled discrete action shape: {discrete_action.shape}")  # 应该为 [32]
    print(
        f"Sampled continuous action shape: {continuous_action.shape}"
    )  # 应该为 [32, 8]
    assert discrete_action.shape == (batch_size,)
    assert continuous_action.shape == (batch_size, continuous_dim)
    print("Sample shapes are correct. ✅")

    # 5. 测试对数概率
    log_p = hybrid_dist.log_prob((discrete_action, continuous_action))
    print("\n--- Testing log_prob() ---")
    print(f"Log probability shape: {log_p.shape}")  # 应该为 [32]
    assert log_p.shape == (batch_size,)
    print("Log probability shape is correct. ✅")
