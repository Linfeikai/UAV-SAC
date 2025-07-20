# =====================================================================================
# 文件名: diffusion_policy_actor.py (原 hybrid_actor.py)
# 描述: 实现了基于扩散模型的Actor，用于在SAC等算法中学习多模态策略。
#       该Actor将取代原有的、基于高斯分布和分离头的HybridActor。
# =====================================================================================

import torch
import torch.nn as nn
import math
from gymnasium import spaces
from typing import Tuple, Dict, Any, List

# 从SB3导入必要的基类和类型提示
from stable_baselines3.common.policies import BasePolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.type_aliases import PyTorchObs

# -------------------------------------------------------------------------------------
# 辅助模块和函数
# -------------------------------------------------------------------------------------


def _precompute_diffusion_schedule(
    T_steps: int, beta_start: float = 0.0001, beta_end: float = 0.02
) -> Dict[str, torch.Tensor]:
    """
    预先计算扩散过程所需的常量系数 (betas, alphas, ...)，避免在训练中重复计算。
    """
    betas = torch.linspace(beta_start, beta_end, T_steps, dtype=torch.float32)
    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)  # alpha_bar
    alphas_cumprod_prev = torch.cat(
        [torch.tensor([1.0]), alphas_cumprod[:-1]], dim=0
    )  # alpha_bar_{t-1}

    # 计算用于前向加噪过程的系数: sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * noise
    sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
    sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod)

    # 计算用于反向去噪过程的系数
    posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)

    return {
        "betas": betas,
        "alphas_cumprod": alphas_cumprod,
        "sqrt_alphas_cumprod": sqrt_alphas_cumprod,
        "sqrt_one_minus_alphas_cumprod": sqrt_one_minus_alphas_cumprod,
        "posterior_variance": posterior_variance,
    }


class _SinusoidalTimestepEmbedding(nn.Module):
    """
    将标量时间步 t 编码成一个高维向量。
    """

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, timestep: torch.Tensor) -> torch.Tensor:
        device = timestep.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = timestep.float() * embeddings
        sin_embeddings = torch.sin(embeddings)
        cos_embeddings = torch.cos(embeddings)
        final_embeddings = torch.cat((sin_embeddings, cos_embeddings), dim=-1)
        if self.dim % 2 == 1:
            final_embeddings = torch.nn.functional.pad(final_embeddings, (0, 1))
        return final_embeddings


# -------------------------------------------------------------------------------------
# 核心网络：EpsilonNet
# -------------------------------------------------------------------------------------
class _EpsilonNet(nn.Module):
    """
    噪声预测网络的具体实现。作为 DiffusionPolicyActor 的一个内部模块。
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        time_embedding_dim: int = 64,
        hidden_dim: int = 256,
    ):
        super().__init__()

        # 时间步编码器
        self.time_encoder = _SinusoidalTimestepEmbedding(time_embedding_dim)
        self.time_mlp = nn.Sequential(
            nn.Linear(time_embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # 状态和动作的编码器
        self.state_encoder = nn.Sequential(nn.Linear(state_dim, hidden_dim), nn.ReLU())
        self.action_encoder = nn.Sequential(
            nn.Linear(action_dim, hidden_dim), nn.ReLU()
        )

        # 融合信息的主干网络
        combined_dim = hidden_dim + hidden_dim + hidden_dim
        self.backbone = nn.Sequential(
            nn.Linear(combined_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        # 输出层
        self.output_layer = nn.Linear(hidden_dim, action_dim)

    def forward(
        self, state: torch.Tensor, noisy_action: torch.Tensor, timestep: torch.Tensor
    ) -> torch.Tensor:
        # 分别编码
        time_features = self.time_mlp(self.time_encoder(timestep))
        state_features = self.state_encoder(state)
        action_features = self.action_encoder(noisy_action)

        # 融合并处理
        combined_features = torch.cat(
            [state_features, action_features, time_features], dim=-1
        )
        backbone_output = self.backbone(combined_features)

        # 输出预测的噪声
        return self.output_layer(backbone_output)


# =====================================================================================
# 主类：DiffusionPolicyActor (替换原有的 HybridActor)
# =====================================================================================
class DiffusionPolicyActor(BasePolicy):
    """
    基于扩散模型的Actor。
    它将动作空间视为一个单一的、可通过去噪过程生成的多模态连续空间。
    """

    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Box,  # 动作空间必须是 Box
        net_arch: List[int],  # 这里的 net_arch 可以用于 EpsilonNet 的 hidden_dim
        features_extractor: BaseFeaturesExtractor,
        features_dim: int,
        # 扩散模型特定超参数
        T_steps: int = 50,  # 扩散总步数
        log_prob_monte_carlo_samples: int = 10,  # 计算log_prob时的蒙特卡洛采样数
        normalize_images: bool = True,
        **kwargs,  # 忽略其他来自Policy的参数
    ):
        super().__init__(
            observation_space=observation_space,
            action_space=action_space,
            features_extractor=features_extractor,
            normalize_images=normalize_images,
            # squash_output 对扩散模型不直接适用，因为输出范围由数据本身决定
        )

        # 验证动作空间是否为我们期望的 Box 类型
        if not isinstance(action_space, spaces.Box):
            raise ValueError("DiffusionPolicyActor 要求动作空间为 spaces.Box。")

        self.action_dim = action_space.shape[0]
        self.features_dim = features_dim
        self.T = T_steps
        self.N_log_prob = log_prob_monte_carlo_samples

        # --- 实例化核心组件 ---
        # 1. 噪声预测网络 EpsilonNet
        hidden_dim = net_arch[0] if net_arch else 256  # 使用 net_arch 定义隐藏层维度
        self._epsilon_net = _EpsilonNet(
            state_dim=self.features_dim,
            action_dim=self.action_dim,
            hidden_dim=hidden_dim,
        )

        # 2. 预计算扩散调度系数
        diffusion_schedule = _precompute_diffusion_schedule(self.T)
        # 将这些系数注册为 buffer，以便能随模型移动 (例如 to(device))
        # "我们先预先计算出扩散模型需要的所有数学常量，然后通过register_buffer方法，将这些常量作为模型的一部分固定下来。
        # 这样做既能保证它们不被优化器错误地训练，又能让它们随着模型自动在CPU和GPU之间切换，还能在保存和加载模型时保持一致。"
        for key, value in diffusion_schedule.items():
            self.register_buffer(key, value)

    def _get_constructor_parameters(self) -> Dict[str, Any]:
        # 返回创建此Actor所需的参数
        return dict(
            observation_space=self.observation_space,
            action_space=self.action_space,
            net_arch=[self._epsilon_net.backbone[0].in_features // 3],  # 示例
            features_extractor=self.features_extractor,
            features_dim=self.features_dim,
            T_steps=self.T,
            log_prob_monte_carlo_samples=self.N_log_prob,
        )

    def forward(self, obs: PyTorchObs, deterministic: bool = False) -> torch.Tensor:
        """
        生成动作的核心逻辑，即反向扩散(采样)过程。
        """
        # 1. 提取状态特征
        features = self.extract_features(obs, self.features_extractor)
        batch_size = features.shape[0]

        # 2. 从纯噪声开始
        action_t = torch.randn((batch_size, self.action_dim), device=self.device)

        # 3. 从 T 到 1 循环去噪
        for t_step in reversed(range(1, self.T + 1)):
            t = torch.full(
                (batch_size, 1), t_step, device=self.device, dtype=torch.long
            )

            # 预测噪声
            predicted_noise = self._epsilon_net(features, action_t, t)

            # 计算去噪一步后的动作 a_{t-1}
            alpha_t = self.alphas_cumprod.gather(0, t.squeeze(-1) - 1).reshape(-1, 1)

            beta_t = self.betas.gather(0, t.squeeze(-1) - 1).reshape(-1, 1)

            # 使用 DDPM 论文中的去噪公式
            # DDPM指的是：<Denoising Diffusion Probabilistic Models>
            term1 = 1.0 / torch.sqrt(1.0 - beta_t)
            term2 = action_t - (beta_t / torch.sqrt(1 - alpha_t)) * predicted_noise
            action_t_minus_1 = term1 * term2

            # 如果不是最后一步，并且非确定性模式，则加回一些随机性
            if t_step > 1 and not deterministic:
                variance = self.posterior_variance.gather(0, t.squeeze(-1) - 1).reshape(
                    -1, 1
                )
                action_t_minus_1 += torch.sqrt(variance) * torch.randn_like(action_t)

            action_t = action_t_minus_1

        # 4. 返回最终的干净动作
        return torch.tanh(action_t)  # <-- 关键修正！

    # def action_log_prob(self, obs: PyTorchObs) -> Tuple[torch.Tensor, torch.Tensor]:
    #     """
    #     这个方法在SAC中用于计算Actor Loss。
    #     对于扩散模型，Actor Loss的计算方式完全不同 (使用QNE)。
    #     因此，这个方法主要在训练Critic时，提供下一个动作的log_prob。
    #     """
    #     # 1. 提取状态特征
    #     features = self.extract_features(obs, self.features_extractor)

    #     # 2. 生成一个动作样本 (与 forward 逻辑相同)
    #     # 注意：这里生成的动作是用于计算 Critic loss 的下一个动作 (next_action)
    #     sampled_action = self.forward(obs, deterministic=False)

    #     # 3. 计算这个生成动作的对数概率 (使用数值积分)
    #     # 这是一个计算密集型操作
    #     batch_size = features.shape[0]

    #     # 准备一个变量来累积所有时间步的误差项
    #     total_mse_terms = torch.zeros(batch_size, 1, device=self.device)

    #     # 在所有时间步上进行近似积分
    #     for t_step in range(1, self.T + 1):
    #         t = torch.full(
    #             (batch_size, 1), t_step, device=self.device, dtype=torch.long
    #         )

    #         # 蒙特卡洛采样来估计期望误差
    #         mse_at_t = 0
    #         for _ in range(self.N_log_prob):
    #             epsilon = torch.randn_like(sampled_action)

    #             # 加噪
    #             sqrt_alpha_bar = self.sqrt_alphas_cumprod.gather(
    #                 0, t.squeeze(-1) - 1
    #             ).reshape(-1, 1)
    #             sqrt_one_minus_alpha_bar = self.sqrt_one_minus_alphas_cumprod.gather(
    #                 0, t.squeeze(-1) - 1
    #             ).reshape(-1, 1)
    #             noisy_action = (
    #                 sqrt_alpha_bar * sampled_action + sqrt_one_minus_alpha_bar * epsilon
    #             )

    #             # 预测噪声并计算误差
    #             predicted_noise = self._epsilon_net(features, noisy_action, t)
    #             mse_at_t += torch.sum(
    #                 (epsilon - predicted_noise) ** 2, dim=-1, keepdim=True
    #             )

    #         average_mse_at_t = mse_at_t / self.N_log_prob

    #         # 根据论文公式累加项 (这是一个简化的表达，精确公式更复杂)
    #         # 这里的权重依赖于 alpha 和 beta
    #         weight = (self.betas[t_step - 1] ** 2) / (
    #             2
    #             * (1.0 - self.alphas_cumprod[t_step - 1])
    #             * (1.0 - self.betas[t_step - 1])
    #         )
    #         total_mse_terms += weight * average_mse_at_t

    #     # 最终的log_prob是这些项的负和，加上一个常数
    #     # 注意: 精确的log_prob计算非常复杂，这里提供的是一个近似思路
    #     # 在很多实现中，可能会用更简化的方式处理熵项
    #     log_prob = -total_mse_terms
    #     # log_prob = torch.zeros_like(log_prob)  # 这里可以根据实际需要调整

    #     return sampled_action, log_prob
    def action_log_prob(self, obs: PyTorchObs) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        【已修正】这个方法在SAC中用于计算 Critic Loss 的目标值。
        我们简化它，使其只提供动作和稳定的log_prob，以保证Critic学习的稳定性。
        """
        # 1. 调用修正后的 forward 方法来生成一个被正确挤压到 [-1, 1] 的动作
        sampled_action = self.forward(obs, deterministic=False)

        # 2. 返回一个稳定的 log_prob (暂时设为0)
        # 这会有效地在 Critic 更新时忽略熵项，是稳定训练的关键一步。
        batch_size = sampled_action.shape[0]
        log_prob = torch.zeros(batch_size, 1, device=self.device)

        return sampled_action, log_prob

    def _predict(
        self, observation: PyTorchObs, deterministic: bool = False
    ) -> torch.Tensor:
        """
        在SB3框架中用于推理/评估的方法。
        """
        # 在推理时不计算梯度，以提高效率
        with torch.no_grad():
            return self.forward(observation, deterministic=deterministic)
