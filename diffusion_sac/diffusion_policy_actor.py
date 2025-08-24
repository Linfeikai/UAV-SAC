# =====================================================================================
# 文件名: diffusion_policy_actor.py (已修改)
# 描述: 实现了基于扩散模型的Actor。
#       新增了 approximate_log_prob 方法，用于精确计算 log π(a|s)。
# =====================================================================================

import torch
import torch.nn as nn
import math
from gymnasium import spaces
from typing import Tuple, Dict, Any, List, Optional

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
    betas = torch.linspace(beta_start, beta_end, T_steps, dtype=torch.float32)
    alphas = 1.0 - betas  # <--- 保留原始 alpha_t
    alphas_cumprod = torch.cumprod(alphas, dim=0)  # \bar{alpha}_t
    alphas_cumprod_prev = torch.cat([torch.tensor([1.0]), alphas_cumprod[:-1]], dim=0)

    sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
    sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod)

    posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)

    return {
        "betas": betas,
        "alphas": alphas,  # <--- 新增
        "alphas_cumprod": alphas_cumprod,
        "sqrt_alphas_cumprod": sqrt_alphas_cumprod,
        "sqrt_one_minus_alphas_cumprod": sqrt_one_minus_alphas_cumprod,
        "posterior_variance": posterior_variance,
        "alphas_cumprod_prev": alphas_cumprod_prev,
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
            nn.Mish(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # 状态和动作的编码器
        self.state_encoder = nn.Sequential(nn.Linear(state_dim, hidden_dim), nn.Mish())
        self.action_encoder = nn.Sequential(
            nn.Linear(action_dim, hidden_dim), nn.Mish()
        )

        # 融合信息的主干网络
        combined_dim = hidden_dim + hidden_dim + hidden_dim
        self.backbone = nn.Sequential(
            nn.Linear(combined_dim, hidden_dim),
            nn.Mish(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Mish(),
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
        T_steps: int = 20,  # 扩散总步数 T_total
        log_prob_n_steps: int = 20,  # 数值积分步数 T_log_prob
        log_prob_n_samples: int = 50,  # 蒙特卡洛采样数 N
        normalize_images: bool = True,
        entropy_scale: float = 1.0,  # 用于熵代理的缩放系数
        t_min: float = 1e-3,  # 积分区间的下界
        t_max: float = 1.0 - 1e-3,  # 积分区间的上界
        **kwargs,
    ):
        super().__init__(
            observation_space=observation_space,
            action_space=action_space,
            features_extractor=features_extractor,
            normalize_images=normalize_images,
        )

        if not isinstance(action_space, spaces.Box):
            raise ValueError("DiffusionPolicyActor 要求动作空间为 spaces.Box。")

        self.action_dim = action_space.shape[0]
        self.features_dim = features_dim
        self.T = T_steps
        self.T_log_prob = log_prob_n_steps
        self.N_log_prob = log_prob_n_samples
        self.entropy_scale = entropy_scale
        self.t_min = t_min
        self.t_max = t_max

        hidden_dim = net_arch[0] if net_arch else 256
        self._epsilon_net = _EpsilonNet(
            state_dim=self.features_dim,
            action_dim=self.action_dim,
            hidden_dim=hidden_dim,
        )

        diffusion_schedule = _precompute_diffusion_schedule(self.T)
        for key, value in diffusion_schedule.items():
            self.register_buffer(key, value)

    def _get_constructor_parameters(self) -> Dict[str, Any]:
        data = super()._get_constructor_parameters()
        data.update(
            net_arch=[self._epsilon_net.backbone[0].in_features // 3],
            T_steps=self.T,
            log_prob_n_steps=self.T_log_prob,
            log_prob_n_samples=self.N_log_prob,
            entropy_scale=self.entropy_scale,
            t_min=self.t_min,
            t_max=self.t_max,
        )
        return data

    def _sample_from_noise(
        self, features: torch.Tensor, deterministic: bool = False
    ) -> torch.Tensor:
        """
        一个辅助函数，执行完整的去噪过程并返回最终的无界动作。
        """
        batch_size = features.shape[0]
        action_t = torch.randn((batch_size, self.action_dim), device=self.device)

        # DDPM的反向采样过程
        # --- 关键修复：_sample_from_noise() 的反向步 ---
        for t_step in reversed(range(1, self.T + 1)):
            t = torch.full(
                (batch_size,), t_step - 1, device=self.device, dtype=torch.long
            )

            predicted_noise = self._epsilon_net(features, action_t, t.unsqueeze(1))

            # 取出本步所需的标量（按 batch 广播）
            alpha_bar_t = self.alphas_cumprod.gather(0, t)  # \bar{alpha}_t
            alpha_bar_prev = self.alphas_cumprod_prev.gather(0, t)  # \bar{alpha}_{t-1}
            alpha_t = self.alphas.gather(0, t)  # \alpha_t
            beta_t = self.betas.gather(0, t)  # \beta_t

            # 先用 \bar{alpha}_t 还原 x0（这是正确做法）
            pred_x0 = (
                action_t - torch.sqrt(1.0 - alpha_bar_t).view(-1, 1) * predicted_noise
            ) / torch.sqrt(alpha_bar_t).view(-1, 1)
            pred_x0 = torch.clamp(pred_x0, -1.0, 1.0)

            # 正确的 posterior mean 系数
            denom = (1.0 - alpha_bar_t).view(-1, 1)
            coef_x0 = (
                beta_t.view(-1, 1) * torch.sqrt(alpha_bar_prev).view(-1, 1)
            ) / denom
            coef_xt = (
                torch.sqrt(alpha_t).view(-1, 1) * (1.0 - alpha_bar_prev).view(-1, 1)
            ) / denom

            posterior_mean = coef_x0 * pred_x0 + coef_xt * action_t

            if t_step > 1 and not deterministic:
                posterior_variance = self.posterior_variance.gather(0, t)
                noise = torch.randn_like(action_t)
                action_t = (
                    posterior_mean + torch.sqrt(posterior_variance).view(-1, 1) * noise
                )
            else:
                action_t = posterior_mean

        # 返回最终的无界动作(这里是[-1,1]的)
        return torch.atanh(action_t.clamp(-1 + 1e-6, 1 - 1e-6))

    def forward(self, obs: PyTorchObs, deterministic: bool = False) -> torch.Tensor:
        features = self.extract_features(obs, self.features_extractor)
        unbounded_action = self._sample_from_noise(features, deterministic)
        return torch.tanh(unbounded_action)

    def _predict(
        self, observation: PyTorchObs, deterministic: bool = False
    ) -> torch.Tensor:
        with torch.no_grad():
            return self.forward(observation, deterministic=deterministic)

    def action_log_prob(
        self, obs: PyTorchObs, use_entropy_proxy: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        生成动作并计算其对数概率。
        :param obs: 观察值
        :param use_entropy_proxy: 如果为True，则使用简单的熵代理；否则使用精确的数值积分。
        :return: (压缩到[-1,1]的动作, 对数概率)
        """
        features = self.extract_features(obs, self.features_extractor)
        unbounded_action = self._sample_from_noise(features, deterministic=False)
        policy_action = torch.tanh(unbounded_action)

        if use_entropy_proxy:
            # 使用简单、稳健的熵代理
            log_prob = -torch.sum(unbounded_action**2, dim=-1, keepdim=True)
            log_prob = log_prob * self.entropy_scale
        else:
            # 使用精确的数值积分计算
            log_prob = self.approximate_log_prob(features, policy_action)

        return policy_action, log_prob

    # def action_log_prob(self, obs, use_entropy_proxy: bool = True):
    #     features = self.extract_features(obs, self.features_extractor)
    #     unbounded_action = self._sample_from_noise(features, deterministic=False)
    #     policy_action = torch.tanh(unbounded_action)

    #     # 稳健代理：负二范数作为熵近似（永远 ≤ 0）
    #     log_prob = -torch.sum(unbounded_action**2, dim=-1, keepdim=True)
    #     log_prob = log_prob * self.entropy_scale  # 建议先设 1.0 或 0.5

    #     return policy_action, log_prob

    def approximate_log_prob(
        self, features: torch.Tensor, action: torch.Tensor
    ) -> torch.Tensor:
        """
        根据论文公式21，使用数值积分精确地近似log_prob。
        """
        # 0. 准备参数
        B, A = action.shape  # Batch size, Action dimension

        # 将动作从 [-1, 1] 映射回无界空间，因为扩散过程定义在无界空间
        # 注意：这里假设 forward 采样后，action是tanh()的结果
        unbounded_action = torch.atanh(action.clamp(-1 + 1e-6, 1 - 1e-6))

        # 1. 准备积分所需的时间步和alpha_bar值
        # 在 [t_min, t_max] 之间均匀取 T_log_prob+1 个点来定义 T_log_prob 个积分区间
        time_steps = torch.linspace(
            self.t_min, self.t_max, self.T_log_prob + 1, device=self.device
        )

        # 将连续时间 t (0-1) 映射到离散的 schedule 索引 (0 to T-1)
        discrete_indices = (time_steps * (self.T - 1)).long()

        # 获取 alpha_bar_{t_{i-1}} 和 alpha_bar_{t_i}
        # alphas_cumprod 是预先计算好的 alpha_bar schedule
        alpha_hat_t_minus_1 = self.alphas_cumprod[discrete_indices[:-1]]
        alpha_hat_t = self.alphas_cumprod[discrete_indices[1:]]

        # 2. 向量化计算所有时间步的误差
        # 扩展维度以进行批处理计算: [B, A] -> [B, N, T, A]
        action_expanded = (
            unbounded_action.unsqueeze(1)
            .unsqueeze(1)
            .expand(B, self.N_log_prob, self.T_log_prob, A)
        )
        features_expanded = (
            features.unsqueeze(1)
            .unsqueeze(1)
            .expand(B, self.N_log_prob, self.T_log_prob, -1)
        )

        # 扩展时间步相关参数: [T] -> [1, 1, T, 1]
        alpha_hats_expanded = alpha_hat_t.view(1, 1, self.T_log_prob, 1)

        # 采样N次噪声: [B, N, T, A]
        epsilon = torch.randn_like(action_expanded)

        # 计算加噪动作 a_t
        noisy_actions = (
            torch.sqrt(alpha_hats_expanded) * action_expanded
            + torch.sqrt(1.0 - alpha_hats_expanded) * epsilon
        )

        # 扩展离散时间步用于epsilon-net输入: [T] -> [1, 1, T, 1]
        discrete_t_expanded = (
            discrete_indices[1:]
            .view(1, 1, self.T_log_prob, 1)
            .expand(B, self.N_log_prob, -1, -1)
        )

        # 预测噪声 ε_φ
        # reshape for batch matmul: [B*N*T, Dim]
        predicted_noise = self._epsilon_net(
            features_expanded.reshape(-1, self.features_dim),
            noisy_actions.reshape(-1, A),
            discrete_t_expanded.reshape(-1, 1),
        ).reshape(B, self.N_log_prob, self.T_log_prob, A)

        # 计算平均MSE误差 ε̃φ
        error_mse = torch.sum(
            (epsilon - predicted_noise) ** 2, dim=-1
        )  # Sum over action dim: [B, N, T]
        average_mse = torch.mean(error_mse, dim=1)  # Mean over N samples: [B, T]

        # 3. 计算积分权重 w_ti
        # [T] -> [1, T] for broadcasting
        alpha_hats_t_reshaped = alpha_hat_t.view(1, -1)
        alpha_hat_t_minus_1_reshaped = alpha_hat_t_minus_1.view(1, -1)

        # VP SDE 下 σ(α) = α, σ(-α) = 1-α
        # 这里使用论文中的公式
        sigma_at = alpha_hats_t_reshaped
        sigma_neg_at = 1.0 - alpha_hats_t_reshaped
        sigma_at_prev = alpha_hat_t_minus_1_reshaped

        # w_ti = (σ(α_ti-1) - σ(α_ti)) / (σ(α_ti) * σ(-α_ti))
        integration_weight = (sigma_at_prev - sigma_at) / (
            sigma_at * sigma_neg_at + 1e-8
        )

        # 4. 计算求和项 (d * σ(α_ti) - ε̃φ)
        term_inside_sum = A * sigma_at - average_mse

        # 5. 最终求和并加上常数 c'
        # [B, T] * [1, T] -> [B, T], then sum over T dim
        weighted_term = term_inside_sum * integration_weight
        sum_term = torch.sum(weighted_term, dim=-1)  # Result shape: [B]

        # c' = -d/2 * log(2πε)
        c_prime = -0.5 * A * math.log(2 * math.pi * math.e)

        # log p ≈ c' + 1/2 * Σ(...)
        log_prob = c_prime + 0.5 * sum_term

        return log_prob.unsqueeze(-1)  # Ensure shape is [B, 1]
