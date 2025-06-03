# Place this code in your custom_components/hybrid_distribution.py file
import numpy as np
import torch as th
from torch import nn
from gymnasium import spaces
from typing import (
    Tuple,
    Optional,
    Union,
    List,
    Type,
    Any,
    Dict,
)  # Import necessary types

# Import from the provided SB3 distributions file or library
from stable_baselines3.common.distributions import (
    Distribution,
    CategoricalDistribution,
    SquashedDiagGaussianDistribution,
    DiagGaussianDistribution,  # Import base Gaussian if needed
    sum_independent_dims,
)

# Import log_std constants if needed (e.g., from sac.policies)
from stable_baselines3.sac.policies import LOG_STD_MAX, LOG_STD_MIN
# Define them here if not easily importable
# 确实通过限制分布参数的范围来避免由极端参数值引发的数值问题和梯度问题，从而保证了策略学习的稳定性。


class HybridDistribution(Distribution):
    """
    Distribution for hybrid action spaces (Discrete + Continuous).
    It assumes the action space is a Tuple(Discrete, Box).

    :param discrete_dim: Dimension of the discrete action space (number of actions).
    :param continuous_dim: Dimension of the continuous action space.
    """

    def __init__(self, discrete_dim: int, continuous_dim: int):
        super().__init__()
        self.discrete_dim = discrete_dim
        self.continuous_dim = continuous_dim

        # Use SB3's existing distributions internally
        self.cat_dist = CategoricalDistribution(discrete_dim)
        # For SAC, we need the squashed version for the continuous part
        self.squashed_gauss_dist = SquashedDiagGaussianDistribution(continuous_dim)

        # Placeholders for the actual torch distributions after parameters are set
        self._cur_cat_dist: Optional[th.distributions.Categorical] = None
        self._cur_squashed_gauss_dist: Optional[SquashedDiagGaussianDistribution] = None

    def proba_distribution_net(
        self, latent_dim: int, log_std_init: float = -2.0
    ) -> Tuple[nn.Module, nn.Module, nn.Module]:
        """
        Create the layers for the distribution parameters.
        Outputs:
            - Discrete logits layer
            - Continuous mean layer
            - Continuous log_std layer

        :param latent_dim: Dimension of the latent features.
        :param log_std_init: Initial value for the continuous log_std.
        :return: Tuple of networks (discrete_logits, continuous_mean, continuous_log_std)
        """
        # Discrete part: uses CategoricalDistribution's network logic
        # 每次调用都会创建一个新的网络
        discrete_logits_net = self.cat_dist.proba_distribution_net(
            latent_dim=latent_dim
        )

        # Continuous part: uses SquashedDiagGaussianDistribution's network logic
        # Need mean and log_std networks. SquashedGaussian inherits from DiagGaussian.
        # DiagGaussian's proba_distribution_net returns (mean_net, log_std_param)
        # We need two separate networks coming from the latent_dim
        continuous_mean_net = nn.Linear(latent_dim, self.continuous_dim)
        continuous_log_std_net = nn.Linear(latent_dim, self.continuous_dim)

        # Initialize log_std layer's bias (optional but common)
        if (
            hasattr(continuous_log_std_net, "bias")
            and continuous_log_std_net.bias is not None
        ):
            nn.init.constant_(continuous_log_std_net.bias, log_std_init)

        return discrete_logits_net, continuous_mean_net, continuous_log_std_net

    def proba_distribution(
        self,
        discrete_logits: th.Tensor,
        continuous_mean: th.Tensor,
        continuous_log_std: th.Tensor,
    ) -> "HybridDistribution":
        """
        Set the parameters of the distribution.

        :param discrete_logits: Logits for the discrete part.
        :param continuous_mean: Mean for the continuous part.
        :param continuous_log_std: Log standard deviation for the continuous part.
        :return: self
        """

        # print("Inside HybridDistribution.proba_distribution:")  # DEBUG PRINT
        # print("Received discrete_logits:", discrete_logits)  # DEBUG PRINT
        # print(
        #     "Received discrete_logits grad_fn:", discrete_logits.grad_fn
        # )  # DEBUG PRINT

        # Clamp log_std for stability (important for SAC)
        clamped_log_std = th.clamp(continuous_log_std, LOG_STD_MIN, LOG_STD_MAX)

        # Set internal distributions using SB3 helpers
        self.cat_dist.proba_distribution(discrete_logits)
        # print(
        #     "Logits in self.cat_dist.distribution AFTER call:",
        #     self.cat_dist.distribution.logits,
        # )  # DEBUG PRINT
        # print(
        #     "Logits grad_fn in self.cat_dist.distribution AFTER call:",
        #     self.cat_dist.distribution.logits.grad_fn,
        # )  # DEBUG PRINT

        self.squashed_gauss_dist.proba_distribution(continuous_mean, clamped_log_std)

        # Store the underlying torch distributions for direct use if needed
        self._cur_cat_dist = self.cat_dist.distribution
        # SquashedGaussian stores the original Gaussian in self.distribution
        self._cur_squashed_gauss_dist = self.squashed_gauss_dist

        return self

    def log_prob(self, actions: Tuple[th.Tensor, th.Tensor]) -> th.Tensor:
        """
        Returns the log likelihood of the hybrid action.

        :param actions: Tuple (discrete_action, continuous_action)
        :return: The log likelihood.
        """
        if self._cur_cat_dist is None or self._cur_squashed_gauss_dist is None:
            raise ValueError("proba_distribution must be called first")

        discrete_action, continuous_action = actions
        log_prob_discrete = self.cat_dist.log_prob(discrete_action)

        # SquashedGaussian log_prob needs careful handling of potential NaNs
        # It takes the squashed action (tanh output)
        log_prob_continuous = self.squashed_gauss_dist.log_prob(continuous_action)

        # Check shapes - log_prob_discrete should be (batch_size,), log_prob_continuous should be (batch_size,)
        # sum_independent_dims is already handled inside SquashedDiagGaussianDistribution.log_prob

        return log_prob_discrete + log_prob_continuous

    # 这个函数只是为了接口的完整性存在
    def entropy(self) -> Optional[th.Tensor]:
        """
        Returns Shannon's entropy.
        Note: Entropy for SquashedGaussian is not analytically computable (returns None).
        The total entropy might need to be estimated via sampling (-log_prob.mean()).
        Here we return the sum if both are computable, otherwise None.
        """
        if self._cur_cat_dist is None or self._cur_squashed_gauss_dist is None:
            raise ValueError("proba_distribution must be called first")

        entropy_discrete = self.cat_dist.entropy()
        entropy_continuous = self.squashed_gauss_dist.entropy()  # This will be None

        if entropy_discrete is None or entropy_continuous is None:
            return None
        # This line will likely not be reached due to entropy_continuous being None
        return entropy_discrete + entropy_continuous

    def sample(self) -> Tuple[th.Tensor, th.Tensor]:
        """
        Returns samples from the distributions.
        :return: Tuple (discrete_sample, continuous_sample)
        """
        if self._cur_cat_dist is None or self._cur_squashed_gauss_dist is None:
            raise ValueError("proba_distribution must be called first")

        discrete_sample = self.cat_dist.sample()
        continuous_sample = self.squashed_gauss_dist.sample()
        return discrete_sample, continuous_sample

    def mode(self) -> Tuple[th.Tensor, th.Tensor]:
        """
        Returns the most likely actions (deterministic output).
        :return: Tuple (discrete_mode, continuous_mode)
        """
        if self._cur_cat_dist is None or self._cur_squashed_gauss_dist is None:
            raise ValueError("proba_distribution must be called first")

        discrete_mode = self.cat_dist.mode()
        continuous_mode = self.squashed_gauss_dist.mode()
        return discrete_mode, continuous_mode

    # get_actions is inherited from Distribution base class and should work

    def actions_from_params(
        self,
        discrete_logits: th.Tensor,
        continuous_mean: th.Tensor,
        continuous_log_std: th.Tensor,
        deterministic: bool = False,
        **kwargs,  # Accept potential extra kwargs
    ) -> Tuple[th.Tensor, th.Tensor]:
        """
        Returns samples from the distribution given its parameters.

        :param discrete_logits: Logits for the discrete part.
        :param continuous_mean: Mean for the continuous part.
        :param continuous_log_std: Log standard deviation for the continuous part.
        :param deterministic: Whether to sample or use the mode.
        :return: Tuple of actions (discrete_action, continuous_action)
        """
        self.proba_distribution(discrete_logits, continuous_mean, continuous_log_std)
        return self.get_actions(deterministic=deterministic)

    def log_prob_from_params(
        self,
        discrete_logits: th.Tensor,
        continuous_mean: th.Tensor,
        continuous_log_std: th.Tensor,
        **kwargs,  # Accept potential extra kwargs
    ) -> Tuple[Tuple[th.Tensor, th.Tensor], th.Tensor]:
        """
        Returns samples and the associated log probabilities.

        :param discrete_logits: Logits for the discrete part.
        :param continuous_mean: Mean for the continuous part.
        :param continuous_log_std: Log standard deviation for the continuous part.
        :return: Tuple containing:
            - Tuple of actions (discrete_action, continuous_action)
            - Combined log probability tensor
        """
        # Sample actions NON-deterministically
        actions = self.actions_from_params(
            discrete_logits, continuous_mean, continuous_log_std, deterministic=False
        )
        # 这里用false是因为这是为了评估策略的随机行为。
        # 因为我们关心的是在当前策略的随机性下，采取这些动作能得到的Q值以及这些动作本身的概率。
        # Calculate log prob of these sampled actions
        log_prob = self.log_prob(actions)
        return actions, log_prob


# ==================================
# Helper Function
# ==================================


def make_hybrid_proba_distribution(
    action_space: spaces.Tuple,
    dist_kwargs: Optional[Dict[str, Any]] = None,  # Allow passing kwargs like epsilon
) -> HybridDistribution:
    """
    Return an instance of HybridDistribution.

    :param action_space: The action space (must be Tuple(Discrete, Box)).
    :param dist_kwargs: Keyword arguments to pass to the probability distribution constructor.
    :return: The appropriate Distribution object.
    """
    if dist_kwargs is None:
        dist_kwargs = {}

    if not isinstance(action_space, spaces.Tuple) or len(action_space.spaces) != 2:
        raise ValueError(
            "HybridDistribution r equires a Tuple action space with 2 elements."
        )

    discrete_space, continuous_space = action_space.spaces

    if not isinstance(discrete_space, spaces.Discrete):
        raise ValueError("First element of Tuple must be Discrete.")
    if not isinstance(continuous_space, spaces.Box):
        raise ValueError("Second element of Tuple must be Box.")

    # Extract dimensions
    discrete_dim = int(discrete_space.n)
    continuous_dim = int(
        np.prod(continuous_space.shape)
    )  # Use np.prod for multi-dim Box

    return HybridDistribution(discrete_dim, continuous_dim, **dist_kwargs)


# ==================================
# Optional: Register in make_proba_distribution if modifying SB3 directly
# (Not recommended, better to use your custom setup)
# ==================================
# You could theoretically modify the main make_proba_distribution in SB3
# to recognize Tuple spaces, but it's cleaner to use your custom setup.
# Example modification (for illustration, not direct use):
#
# def make_proba_distribution_modified(...):
#     ...
#     elif isinstance(action_space, spaces.Tuple):
#         # Add checks for Discrete+Box structure
#         return make_hybrid_proba_distribution(action_space, dist_kwargs)
#     ...
