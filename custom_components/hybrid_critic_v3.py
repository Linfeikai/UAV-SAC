import torch
import torch.nn as nn
from gymnasium import spaces
from typing import List, Type, Tuple

# --- 核心修改 1: 导入我们需要的SB3基类和工具 ---
from stable_baselines3.common.policies import BaseModel
from stable_baselines3.common.preprocessing import get_action_dim
from stable_baselines3.common.torch_layers import (
    BaseFeaturesExtractor,
    create_mlp,
    FlattenExtractor,
)


# QHead 类保持不变，它是一个标准的nn.Module，设计得很好
class QHead(nn.Module):
    def __init__(
        self,
        input_dim: int,
        net_arch: List[int],
        activation_fn: Type[nn.Module] = nn.ReLU,
    ):
        super().__init__()
        self.q1 = nn.Sequential(*create_mlp(input_dim, 1, net_arch, activation_fn))
        self.q2 = nn.Sequential(*create_mlp(input_dim, 1, net_arch, activation_fn))

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.q1(x), self.q2(x)


# --- 核心修改 2: 让 MultiHeadCritic 继承自 BaseModel ---
class MultiHeadCritic(BaseModel):
    def __init__(
        self,
        # --- 核心修改 3: 更新 __init__ 的签名以匹配SB3规范 ---
        observation_space: spaces.Space,
        action_space: spaces.Tuple,
        net_arch: List[int],
        features_extractor: BaseFeaturesExtractor,
        features_dim: int,  # 这个维度由Policy传入，是features_extractor的输出维度
        activation_fn: Type[nn.Module] = nn.ReLU,
        # 其他参数 (如 normalize_images) 会被父类处理
        **kwargs,
    ):
        # 调用父类的构造函数，并把 features_extractor 传给它
        super().__init__(
            observation_space=observation_space,
            action_space=action_space,
            features_extractor=features_extractor,
            **kwargs,
        )

        # 从action_space中解析维度
        discrete_action_dim = action_space.spaces[0].n
        continuous_action_dim = get_action_dim(action_space.spaces[1])

        self.discrete_action_dim = discrete_action_dim

        # Q头的输入维度现在基于 features_dim，而不是 state_dim
        q_head_input_dim = features_dim + continuous_action_dim

        self.q_networks = nn.ModuleList(
            [
                QHead(
                    input_dim=q_head_input_dim,
                    net_arch=net_arch,
                    activation_fn=activation_fn,
                )
                for _ in range(self.discrete_action_dim)
            ]
        )

    def forward(
        self,
        obs: torch.Tensor,  # --- 核心修改 4: forward输入现在是原始观测 obs ---
        discrete_action: torch.Tensor,
        continuous_action: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # --- 核心修改 5: 使用继承来的 self.extract_features 方法 ---
        # 不再直接使用obs，而是先提取特征
        with torch.set_grad_enabled(False):  # 通常Critic的特征提取不计算梯度
            features = self.extract_features(obs, self.features_extractor)

        # 后续逻辑保持不变，只是输入从 state 变为 features
        input_features = torch.cat([features, continuous_action], dim=1)
        batch_size = features.shape[0]

        q1_all = torch.zeros(batch_size, 1, device=obs.device)
        q2_all = torch.zeros(batch_size, 1, device=obs.device)

        for i in range(self.discrete_action_dim):
            batch_indices = (discrete_action.squeeze(-1) == i).nonzero(as_tuple=True)[0]
            if batch_indices.numel() > 0:
                selected_inputs = input_features[batch_indices]
                q1, q2 = self.q_networks[i](selected_inputs)
                q1_all[batch_indices] = q1
                q2_all[batch_indices] = q2

        return q1_all, q2_all


# --- 核心修改 6: 更新测试代码以反映新的实例化方式 ---
if __name__ == "__main__":
    # 1. 定义模拟的环境空间
    obs_space = spaces.Box(low=-1, high=1, shape=(128,))
    act_space = spaces.Tuple(
        (spaces.Discrete(5), spaces.Box(low=-1, high=1, shape=(8,)))
    )
    discrete_action_dim = act_space.spaces[0].n
    continuous_action_dim = get_action_dim(act_space.spaces[1])

    # 2. 创建一个特征提取器实例
    # 在真实使用场景中，这个对象是由Policy创建并传递过来的
    features_extractor = FlattenExtractor(obs_space)
    features_dim = features_extractor.features_dim

    # 3. 用新的方式实例化Critic
    critic = MultiHeadCritic(
        observation_space=obs_space,
        action_space=act_space,
        net_arch=[256, 256],
        features_extractor=features_extractor,
        features_dim=features_dim,
    )

    # 4. 创建模拟数据
    batch_size = 32
    # 输入现在是 observation，而不是 state
    obs_batch = torch.randn(batch_size, 128)
    continuous_action_batch = torch.randn(batch_size, continuous_action_dim)
    discrete_action_batch = torch.randint(0, discrete_action_dim, (batch_size, 1))

    # 5. 调用新的forward方法
    q1_outputs, q2_outputs = critic(
        obs_batch, discrete_action_batch, continuous_action_batch
    )

    print("--- Critic (继承自BaseModel) 测试 ---")
    print(f"Q1 output shape: {q1_outputs.shape}")
    print(f"Q2 output shape: {q2_outputs.shape}")
    print(f"Q1 mean value: {q1_outputs.mean().item()}")
    print("测试通过！✅")
