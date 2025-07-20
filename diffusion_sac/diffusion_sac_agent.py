# =====================================================================================
# 文件名: diffusion_sac_agent.py (原 hybrid_sac_agent.py)
# 描述: 实现了基于扩散策略的SAC Agent (MaxEntDP)。
#       核心修改在于 train 方法，使用 QNE 来更新 Actor。
# =====================================================================================

import numpy as np
import torch as th
from torch.nn import functional as F
from gymnasium import spaces
from typing import Any, ClassVar, Optional, Type, TypeVar, Union, Dict, List, Tuple

# 导入我们新定义的 Policy
from .diffusion_sac_policy import DiffusionSACPolicy

# 导入我们新定义的 Actor 和 Critic
from .diffusion_policy_actor import DiffusionPolicyActor
from .diffusion_policy_critic import ContinuousCritic

# 导入标准的 ReplayBuffer 和 SB3 的核心组件
from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.off_policy_algorithm import OffPolicyAlgorithm
from stable_baselines3.common.policies import BasePolicy
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, Schedule
from stable_baselines3.common.utils import polyak_update

SelfDiffusionSACAgent = TypeVar("SelfDiffusionSACAgent", bound="DiffusionSACAgent")


class DiffusionSACAgent(OffPolicyAlgorithm):
    """
    基于扩散策略的软演员-评论家 (Soft Actor-Critic) 算法。
    它使用Q加权噪声估计(QNE)来训练一个扩散模型作为策略。
    """

    # --- 指定新的默认策略、Actor和Critic类型 ---
    policy: DiffusionSACPolicy
    actor: DiffusionPolicyActor
    critic: ContinuousCritic
    critic_target: ContinuousCritic

    # 定义策略名称到类的映射
    policy_aliases: ClassVar[Dict[str, Type[BasePolicy]]] = {
        "DiffusionSACPolicy": DiffusionSACPolicy,
    }

    def __init__(
        self,
        policy: Union[str, Type[DiffusionSACPolicy]],
        env: Union[GymEnv, str],
        learning_rate: Union[float, Schedule] = 3e-5,
        buffer_size: int = 1_000_000,
        learning_starts: int = 10000,
        batch_size: int = 256,
        tau: float = 0.005,
        gamma: float = 0.99,
        train_freq: Union[int, Tuple[int, str]] = 1,
        gradient_steps: int = 1,
        # --- 移除了 HybridSAC 特有的、不再需要的参数 ---
        replay_buffer_class: Optional[
            Type[ReplayBuffer]
        ] = ReplayBuffer,  # <-- 使用标准ReplayBuffer
        replay_buffer_kwargs: Optional[Dict[str, Any]] = None,
        ent_coef: Union[str, float] = "auto",
        target_update_interval: int = 1,
        target_entropy: Union[str, float] = "auto",
        # --- 新增的扩散模型特定参数 ---
        qne_k_samples: int = 32,  # QNE中的K值，即"头脑风暴"的样本数
        policy_kwargs: Optional[Dict[str, Any]] = None,
        qne_temperature: float = 0.1,
        # --- 其他标准参数 ---
        tensorboard_log: Optional[str] = None,
        verbose: int = 0,
        seed: Optional[int] = None,
        device: Union[th.device, str] = "auto",
        _init_setup_model: bool = True,
    ):
        self.qne_k_samples = qne_k_samples

        super().__init__(
            policy=policy,
            env=env,
            learning_rate=learning_rate,
            buffer_size=buffer_size,
            learning_starts=learning_starts,
            batch_size=batch_size,
            tau=tau,
            gamma=gamma,
            train_freq=train_freq,
            gradient_steps=gradient_steps,
            action_noise=None,  # SAC不使用外部噪声
            replay_buffer_class=replay_buffer_class,
            replay_buffer_kwargs=replay_buffer_kwargs,
            policy_kwargs=policy_kwargs,
            stats_window_size=100,
            tensorboard_log=tensorboard_log,
            verbose=verbose,
            device=device,
            seed=seed,
            supported_action_spaces=(spaces.Box,),  # <-- 关键：现在只支持Box空间
            support_multi_env=True,
        )

        self.target_entropy = target_entropy
        self.log_ent_coef: Optional[th.Tensor] = None
        self.ent_coef = ent_coef
        self.target_update_interval = target_update_interval
        self.ent_coef_optimizer: Optional[th.optim.Adam] = None
        self.qne_temperature = qne_temperature

        if _init_setup_model:
            self._setup_model()

    def _setup_model(self) -> None:
        # P-TODO: 移除 "use_sde"，因为它不被 DiffusionSACPolicy 支持
        if "use_sde" in self.policy_kwargs:
            self.policy_kwargs.pop("use_sde")

        super()._setup_model()
        self._create_aliases()

        # 自动计算目标熵，现在逻辑非常简单
        if self.target_entropy == "auto":
            self.target_entropy = float(
                -np.prod(self.action_space.shape).astype(np.float32)
            )
        else:
            self.target_entropy = float(self.target_entropy)

        # 设置熵系数(alpha)的优化器
        if isinstance(self.ent_coef, str) and self.ent_coef.startswith("auto"):
            init_value = 1.0
            if "_" in self.ent_coef:
                init_value = float(self.ent_coef.split("_")[1])
            self.log_ent_coef = th.log(
                th.ones(1, device=self.device) * init_value
            ).requires_grad_(True)
            self.ent_coef_optimizer = th.optim.Adam(
                [self.log_ent_coef], lr=self.lr_schedule(1)
            )
        else:
            self.ent_coef_tensor = th.tensor(float(self.ent_coef), device=self.device)

    def _create_aliases(self) -> None:
        self.actor = self.policy.actor
        self.critic = self.policy.critic
        self.critic_target = self.policy.critic_target

    def train(self, gradient_steps: int, batch_size: int = 256) -> None:
        """
        训练循环。这是算法的核心，Actor的更新逻辑将在这里被彻底改变。
        """
        self.policy.set_training_mode(True)
        optimizers = [self.critic.optimizer, self.actor.optimizer]
        if self.ent_coef_optimizer is not None:
            optimizers.append(self.ent_coef_optimizer)

        self._update_learning_rate(optimizers)

        actor_losses, critic_losses, ent_coef_losses = [], [], []
        ent_coefs = []

        for gradient_step in range(gradient_steps):
            # 1. 从Replay Buffer采样
            replay_data = self.replay_buffer.sample(
                batch_size, env=self._vec_normalize_env
            )
            if (
                th.isinf(replay_data.observations).any()
                or th.isnan(replay_data.observations).any()
            ):
                raise ValueError("NaN or Inf detected in observations!")
            if (
                th.isinf(replay_data.actions).any()
                or th.isnan(replay_data.actions).any()
            ):
                raise ValueError("NaN or Inf detected in actions!")
            if (
                th.isinf(replay_data.rewards).any()
                or th.isnan(replay_data.rewards).any()
            ):
                raise ValueError("NaN or Inf detected in rewards!")

            # 这里是为了得到ent_coef的值，也就是α系数。用来控制熵的权重
            if self.log_ent_coef is not None and self.ent_coef_optimizer is not None:
                # 如果是自动调优模式，通过exp()获取当前alpha的值
                ent_coef_tensor = self.log_ent_coef.exp()
            else:
                # 如果是固定值模式，直接使用预先创建的张量
                ent_coef_tensor = self.ent_coef_tensor

            # 2. --- Critic Loss 计算 (与标准SAC非常相似) ---
            with th.no_grad():
                # 使用 Actor 生成下一状态的动作及其对数概率
                # next_actions, next_log_prob = self.actor.action_log_prob(
                #     replay_data.next_observations
                # )
                next_actions = self.actor.forward(
                    replay_data.next_observations, deterministic=True
                )
                next_log_prob = th.zeros(next_actions.shape[0], 1, device=self.device)

                # 在这里，我们需要插入第一个诊断点，检查Actor的输出是否合理
                # 【上一轮建议的“行动一”】
                self.logger.record(
                    "debug/next_actions_mean", th.mean(next_actions).item()
                )
                self.logger.record(
                    "debug/next_actions_std", th.std(next_actions).item()
                )
                self.logger.record(
                    "debug/next_log_prob_mean", th.mean(next_log_prob).item()
                )

                # 用目标Critic网络评估下一状态-动作对的Q值
                qf_next_target = th.cat(
                    self.critic_target(replay_data.next_observations, next_actions),
                    dim=1,
                )
                min_qf_next_target, _ = th.min(qf_next_target, dim=1, keepdim=True)

                # 在这里，可以诊断目标Q网络给出的分数
                self.logger.record(
                    "debug/qf_next_target_mean", th.mean(min_qf_next_target).item()
                )

                # 加上熵项，计算最终的目标Q值
                min_qf_next_target -= ent_coef_tensor * next_log_prob
                next_q_value = (
                    replay_data.rewards
                    + (1 - replay_data.dones) * self.gamma * min_qf_next_target
                )
                self.logger.record(
                    "debug/target_q_value_mean", th.mean(next_q_value).item()
                )

            # 计算当前Critic的Q值
            qf_values = self.critic(replay_data.observations, replay_data.actions)
            # 计算Critic的MSE损失
            critic_loss = 0.5 * sum(F.mse_loss(q, next_q_value) for q in qf_values)
            critic_losses.append(critic_loss.item())

            # 优化Critic
            self.critic.optimizer.zero_grad()
            critic_loss.backward()
            self.critic.optimizer.step()

            # --- 3. Actor Loss 计算 (QNE核心逻辑) ---
            # 这里的逻辑完全取代了原始SAC的Actor Loss

            # (a) 准备计算Actor Loss所需的数据
            # 我们需要从replay_data中获取干净的动作`a` (即 `replay_data.actions`)
            # 和对应的状态`s` (即 `replay_data.observations`)
            clean_actions_from_buffer = replay_data.actions
            states_from_buffer = replay_data.observations

            # (b) 随机采样扩散时间步t和真实噪声epsilon
            # 随机生成大小为 (batch_size, 1) 的时间步t
            t = th.randint(1, self.actor.T + 1, (batch_size, 1), device=self.device)
            epsilon = th.randn_like(clean_actions_from_buffer)

            # (c) 根据公式创建加噪动作 a_t
            sqrt_alpha_bar = self.actor.sqrt_alphas_cumprod.gather(
                0, t.squeeze(-1) - 1
            ).reshape(-1, 1)
            sqrt_one_minus_alpha_bar = self.actor.sqrt_one_minus_alphas_cumprod.gather(
                0, t.squeeze(-1) - 1
            ).reshape(-1, 1)
            noisy_actions_t = (
                sqrt_alpha_bar * clean_actions_from_buffer
                + sqrt_one_minus_alpha_bar * epsilon
            )
            # 加噪动作=从buffer里采样的干净动作*一个系数+噪声*一个系数

            # (d) 通过QNE计算"目标噪声"epsilon*
            #    i. "头脑风暴" K 个候选动作
            k_candidate_actions = []
            k_noises = th.randn(
                batch_size,
                self.qne_k_samples,
                self.actor.action_dim,
                device=self.device,
            )  # shape: [B, K, A_dim]

            # 使用去噪公式反向生成K个候选的干净动作
            # 这是一个批处理操作，效率很高
            a_t_expanded = noisy_actions_t.unsqueeze(1).expand(
                -1, self.qne_k_samples, -1
            )  # [B, 1, A_dim] -> [B, K, A_dim]
            sqrt_alpha_bar_exp = sqrt_alpha_bar.unsqueeze(1).expand(
                -1, self.qne_k_samples, -1
            )
            sqrt_one_minus_alpha_bar_exp = sqrt_one_minus_alpha_bar.unsqueeze(1).expand(
                -1, self.qne_k_samples, -1
            )
            k_candidate_actions = (
                a_t_expanded - sqrt_one_minus_alpha_bar_exp * k_noises
            ) / sqrt_alpha_bar_exp  # [B, K, A_dim]

            #    ii. Critic打分
            states_expanded = states_from_buffer.unsqueeze(1).expand(
                -1, self.qne_k_samples, -1
            )  # [B, 1, S_dim] -> [B, K, S_dim]

            # ======================= 在这里加入关键的修正代码 =======================
            # 在送入Critic之前，将所有候选动作裁剪到有效范围 [-1, 1]
            # action_space.low 和 high 通常是 -1 和 1，这里用它们来确保通用性
            action_low = self.action_space.low[0]
            action_high = self.action_space.high[0]
            clipped_k_candidate_actions = th.clamp(
                k_candidate_actions.reshape(batch_size * self.qne_k_samples, -1),
                action_low,
                action_high,
            )
            # 【上一轮建议的“行动二”】
            self.logger.record(
                "debug/qne_candidate_actions_mean",
                th.mean(clipped_k_candidate_actions).item(),
            )
            self.logger.record(
                "debug/qne_candidate_actions_std",
                th.std(clipped_k_candidate_actions).item(),
            )

            # =====================================================================

            # 将 [B, K, Dim] 的形状展平为 [B*K, Dim] 以便输入网络
            q_values_k = self.critic.q1_forward(
                states_expanded.reshape(batch_size * self.qne_k_samples, -1),
                clipped_k_candidate_actions,
            )
            q_values_k = q_values_k.reshape(
                batch_size, self.qne_k_samples, 1
            )  # [B*K, 1] -> [B, K, 1]
            with th.no_grad():  # 在no_grad环境下计算，以防影响梯度
                # 计算这 K*B 个Q值的均值和标准差
                q_values_k_mean = th.mean(q_values_k).item()
                q_values_k_std = th.std(q_values_k).item()

                # 使用 SB3 的 logger 记录下来
                # 可以在 TensorBoard 中看到名为 "train/qne_q_mean" 和 "train/qne_q_std" 的图表
                self.logger.record("train/qne_q_mean", q_values_k_mean)
                self.logger.record("train/qne_q_std", q_values_k_std)

                #    iii. Softmax加权合成Target_Noise
            # softmax_weights = F.softmax(
            #     q_values_k / ent_coef_tensor.detach(), dim=1
            # )  # [B, K, 1]
            # q_values_stable = q_values_k - th.max(q_values_k, dim=1, keepdim=True)[0]
            # softmax_weights = F.softmax(
            #     q_values_stable / self.qne_temperature, dim=1
            # )  # [B, K, 1]
            # ====================== 这是修正方案 ======================
            with th.no_grad():
                # 1. 计算当前批次中所有K个候选Q值的均值和标准差
                q_mean = th.mean(q_values_k)
                q_std = th.std(q_values_k) + 1e-6  # 加上一个很小的数防止除以零

                # 2. 对Q值进行归一化，使其分布在均值为0，标准差为1左右
                normalized_q_values = (q_values_k - q_mean) / q_std

                # 在TensorBoard中监控归一化后的Q值，它们应该稳定得多
                self.logger.record(
                    "train/qne_q_normalized_mean", th.mean(normalized_q_values).item()
                )
                self.logger.record(
                    "train/qne_q_normalized_std", th.std(normalized_q_values).item()
                )

            # 3. 在归一化的Q值上应用温度系数和Softmax
            softmax_weights = F.softmax(
                normalized_q_values / self.qne_temperature, dim=1
            )  # [B, K, 1]
            # Target_Noise 是对那K个随机噪声的加权和
            # 在这里，我们可以诊断Softmax的输出
            self.logger.record(
                "debug/softmax_weights_std", th.std(softmax_weights).item()
            )

            target_noise = th.sum(softmax_weights * k_noises, dim=1)  # [B, A_dim]

            # (e) Actor进行噪声预测
            predicted_noise = self.actor._epsilon_net(
                states_from_buffer, noisy_actions_t, t
            )

            # (f) 计算最终的Actor Loss (MSE)
            actor_loss = F.mse_loss(
                predicted_noise, target_noise.detach()
            )  # target_noise不反向传播
            actor_losses.append(actor_loss.item())

            # --- 4. 优化Actor ---
            self.actor.optimizer.zero_grad()
            actor_loss.backward()
            self.actor.optimizer.step()

            # --- 5. 熵系数(alpha)更新 (这部分逻辑可以保留，以稳定训练) ---
            if self.ent_coef_optimizer is not None:
                with th.no_grad():
                    _, log_prob = self.actor.action_log_prob(replay_data.observations)
                ent_coef_loss = (
                    -self.log_ent_coef.exp() * (log_prob + self.target_entropy)
                ).mean()
                ent_coef_losses.append(ent_coef_loss.item())

                self.ent_coef_optimizer.zero_grad()
                ent_coef_loss.backward()
                self.ent_coef_optimizer.step()
                ent_coefs.append(self.log_ent_coef.exp().item())
            else:
                ent_coefs.append(self.ent_coef_tensor.item())

            # --- 6. 更新目标网络 ---
            if gradient_step % self.target_update_interval == 0:
                polyak_update(
                    self.critic.parameters(), self.critic_target.parameters(), self.tau
                )

        self._n_updates += gradient_steps
        actor_lr = self.actor.optimizer.param_groups[0]["lr"]
        critic_lr = self.critic.optimizer.param_groups[0]["lr"]
        # 使用 logger.record 将它们记录下来，以便在 WandB 中显示
        self.logger.record("train/actor_lr", actor_lr)
        self.logger.record("train/critic_lr", critic_lr)

        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/actor_loss", np.mean(actor_losses))
        self.logger.record("train/critic_loss", np.mean(critic_losses))
        self.logger.record("train/ent_coef", np.mean(ent_coefs))
        if len(ent_coef_losses) > 0:
            self.logger.record("train/ent_coef_loss", np.mean(ent_coef_losses))

    def learn(self: SelfDiffusionSACAgent, **kwargs) -> SelfDiffusionSACAgent:
        # 重写learn方法，主要是为了类型提示
        return super().learn(**kwargs)

    # _excluded_save_params 和 _get_torch_save_params 可以从父类继承或根据需要微调
    # 由于我们现在使用标准的Actor/Critic，父类的实现可能已经足够

    def _update_learning_rate(self, optimizers: List[th.optim.Optimizer]) -> None:
        """
        重写此方法以支持 Actor 和 Critic 的独立学习率。
        """
        # 1. 计算当前的训练进度
        progress = self._current_progress_remaining

        # 2. 从 Policy 中获取各自的 schedule，并计算新的学习率
        new_actor_lr = self.policy.lr_actor_schedule(progress)
        new_critic_lr = self.policy.lr_critic_schedule(progress)

        # 3. 手动为 Actor 和 Critic 的优化器设置新的学习率
        self.actor.optimizer.param_groups[0]["lr"] = new_actor_lr
        self.critic.optimizer.param_groups[0]["lr"] = new_critic_lr

        # 4. (可选但推荐) 同时更新熵系数优化器的学习率
        # 它通常使用 Agent 的主学习率
        if self.ent_coef_optimizer is not None:
            new_ent_lr = self.lr_schedule(progress)
            self.ent_coef_optimizer.param_groups[0]["lr"] = new_ent_lr

    # --- 新增方法结束 ---
