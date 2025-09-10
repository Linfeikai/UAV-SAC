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
from sklearn.mixture import GaussianMixture


# 修改后 (推荐):
from torch.amp import autocast
from torch.cuda.amp import GradScaler

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
        tau: float = 0.001,
        gamma: float = 0.99,
        train_freq: Union[int, Tuple[int, str]] = 1,
        gradient_steps: int = 1,
        # --- 移除了 HybridSAC 特有的、不再需要的参数 ---
        replay_buffer_class: Optional[
            Type[ReplayBuffer]
        ] = ReplayBuffer,  # <-- 使用标准ReplayBuffer
        replay_buffer_kwargs: Optional[Dict[str, Any]] = None,
        ent_coef: Union[str, float] = 0.1,
        target_update_interval: int = 1,
        policy_kwargs: Optional[Dict[str, Any]] = None,
        max_grad_norm: float = 1.0,  # <-- 确保有这个参数
        # --- 新增DACER熵调节机制的参数 ---
        alpha_lr: float = 3e-2,  # alpha的学习率
        initial_log_alpha: float = 1.0,  # log_alpha 的初始值 (对应 JAX 代码里的 math.log(3) or math.log(5))
        delay_alpha_update: int = 10000,  # 多少步更新一次alpha
        gmm_samples: int = 200,  # GMM估计时采样的动作数量
        gmm_components: int = 3,  # GMM的成分数量
        target_entropy: Union[str, float] = "auto",  # 目标熵
        # --- 其他标准参数 ---
        tensorboard_log: Optional[str] = None,
        verbose: int = 0,
        seed: Optional[int] = None,
        device: Union[th.device, str] = "auto",
        _init_setup_model: bool = True,
    ):
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

        self.delay_alpha_update = delay_alpha_update
        self.gmm_samples = gmm_samples
        self.gmm_components = gmm_components

        # 定义目标熵，如果为'auto'则自动计算
        if target_entropy == "auto":
            # 这里的 target_entropy 是一个负值
            self.target_entropy = float(
                -np.prod(self.action_space.shape).astype(np.float32)
            )
        else:
            self.target_entropy = float(target_entropy)

        # 定义可学习的探索参数 log_alpha (对应论文里的 a)
        self.log_alpha = th.log(
            th.ones(1, device=self.device) * np.exp(initial_log_alpha)
        ).requires_grad_(True)
        # 为 log_alpha 创建独立的优化器
        self.alpha_optimizer = th.optim.Adam([self.log_alpha], lr=alpha_lr)

        # 记录上一次计算的熵，用于非更新步骤
        self.last_estimated_entropy = 0.0

        # 确保 policy_delay 和 target_update_interval 已定义
        self.policy_delay = 2  # 示例值
        self.target_update_interval = 2  # 示例值

        self.target_update_interval = target_update_interval
        self.max_grad_norm = max_grad_norm
        self.critic_updates_per_step = 4
        self.policy_delay = 2

        if _init_setup_model:
            self._setup_model()

        # === 在这里添加AMP相关的初始化 ===
        # 仅当设备为CUDA时才创建Scaler
        self.scaler = GradScaler() if self.device.type == "cuda" else None

    def _setup_model(self) -> None:
        # P-TODO: 移除 "use_sde"，因为它不被 DiffusionSACPolicy 支持
        if "use_sde" in self.policy_kwargs:
            self.policy_kwargs.pop("use_sde")

        super()._setup_model()
        self._create_aliases()

    def _create_aliases(self) -> None:
        self.actor = self.policy.actor
        self.critic = self.policy.critic
        self.critic_target = self.policy.critic_target

    # === NEW: 显式的动作缩放/反缩放，确保策略空间([-1,1])与环境空间一致 ===
    def _to_policy_space(self, env_actions: th.Tensor) -> th.Tensor:
        low = th.as_tensor(self.action_space.low, device=self.device)
        high = th.as_tensor(self.action_space.high, device=self.device)
        return 2.0 * (env_actions - low) / (high - low) - 1.0

    def _to_env_space(self, policy_actions: th.Tensor) -> th.Tensor:
        low = th.as_tensor(self.action_space.low, device=self.device)
        high = th.as_tensor(self.action_space.high, device=self.device)
        return (policy_actions + 1.0) * 0.5 * (high - low) + low

    def train(self, gradient_steps: int, batch_size: int = 256) -> None:
        """
        训练循环。这是算法的核心，Actor的更新逻辑将在这里被彻底改变。
        """
        self.policy.set_training_mode(True)
        optimizers = [self.critic.optimizer, self.actor.optimizer]

        self._update_learning_rate(optimizers)

        actor_losses, critic_losses = [], []
        # ent_coefs = []
        # 只有在达到指定间隔时才进行昂贵的熵估计和alpha更新
        if self._n_updates % self.delay_alpha_update == 0:
            self._estimate_and_update_alpha(batch_size)

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

            with autocast(
                device_type=self.device.type,
                dtype=th.float16,
                enabled=(self.scaler is not None),
            ):
                # 2. --- Critic Loss 计算 (与标准SAC非常相似) ---
                with th.no_grad():
                    # 使用 Actor 生成下一状态的动作及其对数概率

                    next_actions = self.actor(
                        replay_data.next_observations, deterministic=False
                    )

                    # 【上一轮建议的“行动一”】
                    self.logger.record(
                        "debug/next_actions_mean", th.mean(next_actions).item()
                    )
                    self.logger.record(
                        "debug/next_actions_std", th.std(next_actions).item()
                    )

                    # 用目标Critic网络评估下一状态-动作对的Q值
                    qf_next_target = th.cat(
                        self.critic_target(replay_data.next_observations, next_actions),
                        dim=1,
                    )
                    min_qf_next_target, _ = th.min(qf_next_target, dim=1, keepdim=True)

                    # 在这里，可以诊断目标Q网络给出的分数
                    self.logger.record(
                        "debug/qf_next_target_mean",
                        th.mean(min_qf_next_target).item(),
                    )

                    next_q_value = (
                        replay_data.rewards
                        + (1 - replay_data.dones) * self.gamma * min_qf_next_target
                    )
                    self.logger.record(
                        "debug/target_q_value_mean", th.mean(next_q_value).item()
                    )

                # 计算当前Critic的Q值
                qf_values = self.critic(replay_data.observations, replay_data.actions)

                # 计算 TD 误差
                # 注意：qf_values 通常会返回多个 Q 值（例如，SAC 中有两个 Critic 网络），
                # 你需要选择一个来计算 TD 误差，或者计算每个 Q 值的 TD 误差的平均值或范数。
                # 假设你希望记录第一个 Critic 的 TD 误差，或者所有 Critic 误差的平均值
                td_error_per_qf = [
                    q - next_q_value for q in qf_values
                ]  # 这是一个列表，包含每个Q头的TD误差

                # 为了记录，你可以取其平均绝对值、均方根 (RMS) 或L2范数。
                # 这里我们取所有 Critic 头的 TD 误差的平均绝对值作为示例：
                # 先将所有批次样本和所有Q头的误差压平
                td_errors_flat = th.cat([error.flatten() for error in td_error_per_qf])
                # 计算平均绝对误差
                mean_abs_td_error = th.mean(th.abs(td_errors_flat)).item()

                # 记录 TD 误差
                self.logger.record("debug/td_error_mean_abs", mean_abs_td_error)

                # 计算Critic的MSE损失
                critic_loss = 0.5 * sum(F.mse_loss(q, next_q_value) for q in qf_values)

            critic_losses.append(critic_loss.item())

            # 优化Critic
            self.critic.optimizer.zero_grad()
            # 【修改】使用 scaler.scale() 来缩放loss
            if self.scaler is not None:
                self.scaler.scale(critic_loss).backward()
                # 【修改】在 unscale 之后进行梯度裁剪
                self.scaler.unscale_(self.critic.optimizer)

                critic_grad_norm = 0.0
                for p in self.critic.parameters():
                    if p.grad is not None:
                        param_norm = p.grad.data.norm(2)
                        critic_grad_norm += param_norm.item() ** 2
                critic_grad_norm = critic_grad_norm**0.5
                self.logger.record("train/critic_grad_norm_raw", critic_grad_norm)

                th.nn.utils.clip_grad_norm_(
                    self.critic.parameters(), self.max_grad_norm
                )
                # 【修改】使用 scaler.step() 来更新优化器
                self.scaler.step(self.critic.optimizer)
            else:  # 如果不用GPU，则按原样执行
                critic_loss.backward()

                critic_grad_norm = 0.0
                for p in self.critic.parameters():
                    if p.grad is not None:
                        param_norm = p.grad.data.norm(2)
                        critic_grad_norm += param_norm.item() ** 2
                critic_grad_norm = critic_grad_norm**0.5
                self.logger.record("train/critic_grad_norm_raw", critic_grad_norm)

                th.nn.utils.clip_grad_norm_(
                    self.critic.parameters(), self.max_grad_norm
                )
                self.critic.optimizer.step()

            # --- 3. Actor Loss 计算 (QNE核心逻辑) ---
            # 这里的逻辑完全取代了原始SAC的Actor Loss
            if (self._n_updates + 1) % self.policy_delay == 0:
                # (a) 准备计算Actor Loss所需的数据
                # 我们需要从replay_data中获取干净的动作`a` (即 `replay_data.actions`)
                # 和对应的状态`s` (即 `replay_data.observations`)
                with autocast(
                    device_type=self.device.type,
                    dtype=th.float16,
                    enabled=(self.scaler is not None),
                ):
                    # Actor生成动作 -> Critic评估 -> 最小化负Q值”的直接优化流程。

                    # (a) Actor根据当前状态(obs)生成新动作
                    #    这是关键一步，必须重新生成，不能用buffer里的
                    new_actions_from_policy = self.actor(
                        replay_data.observations, deterministic=False
                    )

                    # (b) Critic评估这些新动作的Q值
                    #    我们只用第一个Critic来指导策略更新，以节省计算
                    q_values_for_new_actions = self.critic.q1_forward(
                        replay_data.observations, new_actions_from_policy
                    )

                    # (c) Actor的损失是最大化Q值，等价于最小化负Q值
                    actor_loss = -q_values_for_new_actions.mean()

                # --- 4. 优化Actor ---
                self.actor.optimizer.zero_grad()
                # 使用 scaler
                if self.scaler is not None:
                    self.scaler.scale(actor_loss).backward()
                    self.scaler.unscale_(self.actor.optimizer)

                    actor_grad_norm = 0.0
                    for p in self.actor.parameters():
                        if p.grad is not None:
                            param_norm = p.grad.data.norm(2)
                            actor_grad_norm += param_norm.item() ** 2
                    actor_grad_norm = actor_grad_norm**0.5
                    self.logger.record("train/actor_grad_norm_raw", actor_grad_norm)

                    th.nn.utils.clip_grad_norm_(
                        self.actor.parameters(), self.max_grad_norm
                    )
                    # ==========================================================

                    self.scaler.step(self.actor.optimizer)
                else:
                    actor_loss.backward()

                    actor_grad_norm = 0.0
                    for p in self.actor.parameters():
                        if p.grad is not None:
                            param_norm = p.grad.data.norm(2)
                            actor_grad_norm += param_norm.item() ** 2
                    actor_grad_norm = actor_grad_norm**0.5
                    self.logger.record("train/actor_grad_norm_raw", actor_grad_norm)

                    th.nn.utils.clip_grad_norm_(
                        self.actor.parameters(), self.max_grad_norm
                    )
                    # ==========================================================

                    self.actor.optimizer.step()

            # 【新增】在所有优化器步骤之后，更新scaler
            if self.scaler is not None:
                self.scaler.update()

            # --- 6. 更新目标网络 ---
            if gradient_step % self.target_update_interval == 0:
                polyak_update(
                    self.critic.parameters(), self.critic_target.parameters(), self.tau
                )

            self._n_updates += 1
        actor_lr = self.actor.optimizer.param_groups[0]["lr"]
        critic_lr = self.critic.optimizer.param_groups[0]["lr"]
        # 使用 logger.record 将它们记录下来，以便在 WandB 中显示
        self.logger.record("train/actor_lr", actor_lr)
        self.logger.record("train/critic_lr", critic_lr)

        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record(
            "train/alpha", self.log_alpha.exp().item()
        )  # 记录当前的alpha值

        if len(actor_losses) > 0:
            self.logger.record("train/actor_loss", np.mean(actor_losses))
        # self.logger.record("train/actor_loss", np.mean(actor_losses))
        self.logger.record("train/critic_loss", np.mean(critic_losses))
        # # 【新增】记录 ent_coef (β) 的固定值，方便在实验中追踪
        # self.logger.record("train/ent_coef", self.ent_coef_tensor.item())

        # self.logger.record("train/ent_coef", np.mean(ent_coefs))
        # if len(ent_coef_losses) > 0:
        #     self.logger.record("train/ent_coef_loss", np.mean(ent_coef_losses))

    def learn(self: SelfDiffusionSACAgent, **kwargs) -> SelfDiffusionSACAgent:
        # 重写learn方法，主要是为了类型提示
        return super().learn(**kwargs)

    def predict(
        self,
        observation: np.ndarray,
        state: Optional[Tuple[np.ndarray, ...]] = None,
        episode_start: Optional[np.ndarray] = None,
        deterministic: bool = False,
    ) -> Tuple[np.ndarray, Optional[Tuple[np.ndarray, ...]]]:
        """
        重写 predict 方法以在训练时注入由 alpha 控制的探索噪声。

        :param observation: 观察值
        :param state: (RNNs使用) 在这里为 None
        :param episode_start: (RNNs使用) 在这里为 None
        :param deterministic: 如果为True，则不添加任何噪声，用于评估。
        :return: (动作, 下一个RNN状态)
        """
        # 1. 如果是确定性模式（评估模式），则调用原始的、不加噪声的预测逻辑
        if deterministic:
            # self.policy.predict() 最终会调用 actor.forward(..., deterministic=True)
            # 它会返回一个没有额外噪声的动作
            return self.policy.predict(
                observation, state, episode_start, deterministic=True
            )

        # 2. 如果是训练模式 (deterministic=False)，执行我们的探索逻辑
        #    首先，我们需要将NumPy的观察值转换为PyTorch张量
        self.policy.set_training_mode(
            False
        )  # 切换到评估模式以关闭dropout等，但我们仍然会手动加噪声
        obs_tensor, _ = self.policy.obs_to_tensor(observation)

        with th.no_grad():
            # 3. 调用Actor的核心forward方法，得到一个“干净”的动作
            #    这对应JAX代码中的 self.diffusion.p_sample(...) 的结果
            base_action = self.policy.actor(
                obs_tensor, deterministic=False
            )  # 使用随机采样

            # 4. === 在这里注入由 exploration_alpha 控制的噪声 ===
            #    首先，获取当前 alpha 的值
            #    你需要确保 self.exploration_alpha 是一个在 __init__ 中定义的可学习参数
            current_alpha = th.exp(self.log_alpha).item()  # .item() 转换成标量

            #    然后，创建一个与动作形状相同的高斯噪声
            noise = th.randn_like(base_action) * current_alpha

            #    最后，将噪声添加到基础动作上
            final_action = base_action + noise

            #    将动作裁剪到 [-1, 1] 范围内
            final_action = th.clamp(final_action, -1.0, 1.0)

        # 5. 将PyTorch张量转换回NumPy数组以返回
        #    注意：SB3希望动作被反归一化到环境的真实动作空间
        final_action_np = final_action.cpu().numpy()
        unscaled_action = self.policy.unscale_action(final_action_np)

        self.policy.set_training_mode(True)  # 别忘了切换回训练模式

        return unscaled_action, state

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
    def _estimate_and_update_alpha(self, batch_size: int) -> None:
        """
        周期性地执行熵估计和alpha更新。
        这个函数不应该在JIT或autocast的上下文中被直接调用，因为它包含NumPy/sklearn操作。
        """
        # --- 1. 熵估计 ---
        # (a) 从buffer中采样一批状态用于估计
        #     我们只需要观察值，所以可以复用 sample 方法
        replay_data = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)
        obs_for_entropy = replay_data.observations

        # (b) 为每个观察值采样 gmm_samples 个动作
        #    这个过程计算量大，所以在no_grad下进行
        with th.no_grad():
            # obs_for_entropy: [B, obs_dim] -> [B, 1, obs_dim] -> [B, N, obs_dim]
            # N是gmm_samples
            expanded_obs = obs_for_entropy.unsqueeze(1).expand(-1, self.gmm_samples, -1)
            # 压平以便批处理: [B*N, obs_dim]
            obs_dim = obs_for_entropy.shape[-1]
            flat_obs = expanded_obs.reshape(-1, obs_dim)

            # actor 一次性生成所有动作
            sampled_actions = self.actor(
                flat_obs, deterministic=False
            )  # [B*N, act_dim]
            # 恢复形状: [B, N, act_dim]
            sampled_actions = sampled_actions.view(batch_size, self.gmm_samples, -1)

        # (c) 使用GMM估计熵
        #    这部分必须在CPU上用NumPy执行
        actions_np = sampled_actions.cpu().numpy()
        total_entropy = []
        for action_samples_for_one_state in actions_np:
            try:
                gmm = GaussianMixture(
                    n_components=self.gmm_components,
                    covariance_type="full",
                    random_state=np.random,
                )
                gmm.fit(action_samples_for_one_state)

                # 计算GMM熵的解析解
                weights = gmm.weights_
                log_determinants = np.linalg.slogdet(gmm.covariances_)[1]

                entropy_gaussians = (
                    0.5 * self.action_space.shape[0] * (1 + np.log(2 * np.pi))
                    + 0.5 * log_determinants
                )
                entropy_weights = -np.sum(weights * np.log(weights + 1e-8))

                estimated_entropy = entropy_weights + np.sum(
                    weights * entropy_gaussians
                )
                total_entropy.append(estimated_entropy)
            except ValueError:
                # 如果协方差矩阵是奇异的，GMM拟合会失败，跳过这个样本
                continue

        if not total_entropy:  # 如果所有GMM都失败了
            return

        final_entropy = np.mean(total_entropy)
        self.last_estimated_entropy = final_entropy  # 缓存结果
        self.logger.record("train/estimated_entropy", self.last_estimated_entropy)

        # --- 2. Alpha 更新 ---
        # (a) 计算alpha的损失函数
        #    注意：我们希望最小化这个损失，所以当熵小时，损失为负，梯度会使log_alpha增大
        alpha_loss = (
            self.log_alpha * (self.last_estimated_entropy - self.target_entropy)
        ).mean()

        # (b) 优化alpha
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()

        self.logger.record("train/alpha_loss", alpha_loss.item())
