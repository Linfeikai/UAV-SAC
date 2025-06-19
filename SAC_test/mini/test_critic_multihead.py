import torch
import torch.nn as nn
import torch.nn.functional as F
from stable_baselines3.common.torch_layers import create_mlp
from typing import Optional, List, Type, Tuple


class QHead(nn.Module):
    def __init__(
        self,
        input_dim: int,
        net_arch: List[int],
        activation_fn: Type[nn.Module] = nn.ReLU,
    ):
        super().__init__()
        # SAC使用两个Q网络来缓解Q值过高估计问题
        self.q1 = nn.Sequential(*create_mlp(input_dim, 1, net_arch, activation_fn))
        self.q2 = nn.Sequential(*create_mlp(input_dim, 1, net_arch, activation_fn))

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.q1(x), self.q2(x)


# MultiHeadCritic 是我们修改的重点
class MultiHeadCritic(nn.Module):
    def __init__(
        self,
        state_dim: int,
        continuous_action_dim: int,
        discrete_action_dim: int,
        net_arch: List[int] = [256, 256],
        activation_fn: Type[nn.Module] = nn.ReLU,
    ):
        super().__init__()
        self.discrete_action_dim = discrete_action_dim

        # --- 核心修改 1: 修正Q头的输入维度 ---
        # Q值函数的输入必须包含状态和连续动作
        q_head_input_dim = state_dim + continuous_action_dim

        # --- 核心修改 2: 简化网络列表 ---
        # 我们的QHead已经包含了n_critics=2的功能（q1, q2）。
        # 所以这里我们只需要一个ModuleList，长度等于离散动作的数量。
        self.q_networks = nn.ModuleList(
            [
                QHead(
                    input_dim=q_head_input_dim,
                    net_arch=net_arch,
                    activation_fn=activation_fn,
                )
                for _ in range(discrete_action_dim)
            ]
        )

    def forward(
        self,
        state: torch.Tensor,
        discrete_action: torch.Tensor,
        continuous_action: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        这个forward方法现在是“矢量化”的，可以处理一个批次的数据，
        其中每个样本的离散动作可能都不同。

        :param state: 状态批次, [B, state_dim]
        :param discrete_action: 离散动作索引批次, [B, 1] or [B,]
        :param continuous_action: 连续动作批次, [B, continuous_action_dim]
        :return: (q1_values, q2_values) for the entire batch, [B, 1] each
        """
        # 首先，为所有样本准备好Q网络的输入特征
        input_features = torch.cat([state, continuous_action], dim=1)
        batch_size = state.shape[0]

        # 初始化用于存放最终结果的张量
        q1_all = torch.zeros(batch_size, 1, device=state.device)
        q2_all = torch.zeros(batch_size, 1, device=state.device)

        # --- 核心修改 3: 矢量化的选择与计算 ---
        # 我们遍历所有的Q头（离散动作的数量通常不大，所以这个循环是高效的）
        for i in range(self.discrete_action_dim):
            # 找到当前批次中，离散动作等于i的那些样本的索引
            # .squeeze()是为了处理[B,1]和[B,]两种形状
            # .nonzero()返回非零元素的索引
            batch_indices = (discrete_action.squeeze() == i).nonzero(as_tuple=True)[0]

            # 如果这个批次中存在执行了动作i的样本
            if batch_indices.numel() > 0:
                # 提取出这些样本对应的输入特征
                selected_inputs = input_features[batch_indices]

                # 将这些特征送入第i个Q头进行计算
                q1, q2 = self.q_networks[i](selected_inputs)

                # 将计算得到Q值放回最终结果张量的正确位置
                q1_all[batch_indices] = q1
                q2_all[batch_indices] = q2

        return q1_all, q2_all


# 我们需要更新测试代码来匹配新的forward接口
if __name__ == "__main__":
    # 定义模型参数
    batch_size = 32
    state_dim = 128
    continuous_action_dim = 8
    discrete_action_dim = 5  # 假设有5个离散动作

    # 创建多头Critic实例
    critic = MultiHeadCritic(
        state_dim=state_dim,
        continuous_action_dim=continuous_action_dim,
        discrete_action_dim=discrete_action_dim,
        net_arch=[256, 256],
    )

    # 创建一个批次的模拟数据
    state_batch = torch.randn(batch_size, state_dim)
    continuous_action_batch = torch.randn(batch_size, continuous_action_dim)

    # 创建一个批次的离散动作，包含不同的动作索引
    # 例如: [0, 1, 2, 3, 4, 0, 1, ...]
    discrete_action_batch = torch.randint(0, discrete_action_dim, (batch_size, 1))

    print("--- 输入数据形状 ---")
    print(f"State batch shape: {state_batch.shape}")
    print(f"Discrete action batch shape: {discrete_action_batch.shape}")
    print(f"Continuous action batch shape: {continuous_action_batch.shape}")
    print("-" * 20)

    # 调用新的forward方法
    q1_outputs, q2_outputs = critic(
        state_batch, discrete_action_batch, continuous_action_batch
    )

    print("--- 输出结果 ---")
    print(f"Q1 output shape: {q1_outputs.shape}")
    print(f"Q2 output shape: {q2_outputs.shape}")
    print(f"Q1 mean value: {q1_outputs.mean().item()}")
    print(f"Q2 mean value: {q2_outputs.mean().item()}")
    print("-" * 20)

    # 验证：检查一个特定离散动作的输出是否正确
    action_to_check = 2
    mask = discrete_action_batch.squeeze() == action_to_check
    print(f"有 {mask.sum()} 个样本的离散动作是 {action_to_check}")
    if mask.sum() > 0:
        # 手动计算这部分样本的Q值
        manual_q1, manual_q2 = critic.q_networks[action_to_check](
            torch.cat([state_batch[mask], continuous_action_batch[mask]], dim=1)
        )
        # 比较手动计算结果和forward输出结果
        print(f"手动计算的Q1均值: {manual_q1.mean().item()}")
        print(f"从总输出中提取的Q1均值: {q1_outputs[mask].mean().item()}")
        assert torch.allclose(manual_q1, q1_outputs[mask])
        print("验证通过！✅")
