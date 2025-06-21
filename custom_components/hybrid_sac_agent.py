from typing import Any, ClassVar, Optional, Type, TypeVar, Union, Dict, Tuple, List
import warnings

import numpy as np
import torch as th
from torch.nn import functional as F
from gymnasium import spaces

# 导入你的新 Policy 和 ReplayBuffer
# The leading dot . indicates a relative import from the current package.
from .hybrid_sac_policy import HybridSACPolicy
from .hybrid_replay_buffer import HybridReplayBuffer  # 确保这个类已正确实现

from .hybrid_actor import HybridActor
from .hybrid_critic import HybridCritic


from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.noise import ActionNoise
from stable_baselines3.common.off_policy_algorithm import OffPolicyAlgorithm
from stable_baselines3.common.policies import BasePolicy
from stable_baselines3.common.type_aliases import (
    GymEnv,
    MaybeCallback,
    Schedule,
    RolloutReturn,
    TrainFreq,
    TrainFrequencyUnit,
)
from stable_baselines3.common.utils import (
    get_parameters_by_name,
    polyak_update,
    get_schedule_fn,
    should_collect_more_steps,
)
from stable_baselines3.common.vec_env import VecEnv

# 为了类型提示
SelfHybridSAC = TypeVar("SelfHybridSAC", bound="HybridSAC")


class HybridSAC(
    OffPolicyAlgorithm
):  # 继承自SAC的父类OffPolicyAlgorithm 并override了他的一些方法
    """
    Soft Actor-Critic (SAC) for Hybrid Action Spaces.
    """

    # 指定新的默认策略
    policy: HybridSACPolicy  # 这个类型提示需要调整，因为父类有具体的 SACPolicy
    actor: HybridActor  # 父类有 Actor
    critic: HybridCritic  # 父类是nn.module
    critic_target: HybridCritic  # 父类是nn.module

    # 注册你的 HybridSACPolicy
    policy_aliases: ClassVar[Dict[str, Type[BasePolicy]]] = {
        # “声明一个类属性 policy_aliases，它是一个从字符串到策略类的映射（字典），
        # 并且这些策略类都继承自 BasePolicy。”
        "HybridSACPolicy": HybridSACPolicy,
    }

    def __init__(
        self,
        policy: Union[str, Type[HybridSACPolicy]],  # <--- 策略类型改为 HybridSACPolicy
        env: Union[GymEnv, str],
        learning_rate: Union[float, Schedule] = 3e-4,
        buffer_size: int = 1_000_000,
        learning_starts: int = 100,
        batch_size: int = 256,
        tau: float = 0.005,
        gamma: float = 0.99,
        train_freq: Union[int, Tuple[int, str]] = 1,
        gradient_steps: int = 1,
        action_noise: Optional[Any] = None,  # SAC 通常不用外部 action_noise
        replay_buffer_class: Optional[
            Type[ReplayBuffer]
        ] = HybridReplayBuffer,  # <--- 默认使用 HybridReplayBuffer
        replay_buffer_kwargs: Optional[Dict[str, Any]] = None,
        optimize_memory_usage: bool = False,  # HybridReplayBuffer 可能不支持
        ent_coef: Union[str, float] = "auto",
        target_update_interval: int = 1,
        target_entropy: Union[
            str, float
        ] = "auto",  # <--- 这个可能需要调整 会在下面的代码进行修改
        # use_sde, sde_sample_freq, use_sde_at_warmup: HybridActor 不支持SDE，可以移除或报错
        stats_window_size: int = 100,
        tensorboard_log: Optional[str] = None,
        policy_kwargs: Optional[Dict[str, Any]] = None,
        verbose: int = 0,
        seed: Optional[int] = None,
        device: Union[th.device, str] = "auto",
        _init_setup_model: bool = True,
    ):
        # 在调用 super().__init__ 之前，确保 action_space 是我们期望的类型
        # 或者在 super().__init__ 之后覆盖/修改相关部分。
        # 父类 SAC 的 __init__ 有 supported_action_spaces=(spaces.Box,) 检查。
        # 我们需要绕过或修改这个。

        # 一个技巧是，在调用父类构造函数时，临时用一个Box替代，然后在_setup_model中修正。
        # 或者更干净的方式是，不完全依赖父类的__init__进行所有设置。

        # 为了简单起见，我们先尝试直接调用，并准备处理 supported_action_spaces 的问题。
        # 我们可能需要覆盖 _setup_model 或 __init__ 的部分逻辑。

        # 关键: 父类 SAC 的 __init__ 会检查 supported_action_spaces
        # 我们需要确保这个检查能通过或者被我们的逻辑覆盖。
        # 最好的方式可能是直接在我们的 HybridSAC 中重写 _setup_model 的部分逻辑。
        # 或者，如果 HybridSACPolicy 已经正确处理了 Tuple action space，
        # 那么在 OffPolicyAlgorithm 的 _setup_model 中，对 action_space 的使用可能就没问题了。

        super().__init__(
            policy=policy,  # 这里会传 "HybridSACPolicy" 字符串或类
            env=env,
            learning_rate=learning_rate,
            buffer_size=buffer_size,
            learning_starts=learning_starts,
            batch_size=batch_size,
            tau=tau,
            gamma=gamma,
            train_freq=train_freq,
            gradient_steps=gradient_steps,
            action_noise=action_noise,  # SAC 通常为 None
            replay_buffer_class=replay_buffer_class,  # 使用 HybridReplayBuffer
            replay_buffer_kwargs=replay_buffer_kwargs,
            policy_kwargs=policy_kwargs,
            stats_window_size=stats_window_size,
            tensorboard_log=tensorboard_log,
            verbose=verbose,
            device=device,
            seed=seed,
            use_sde=False,  # HybridActor 不支持 SDE
            sde_sample_freq=-1,  # HybridActor 不支持 SDE
            use_sde_at_warmup=False,  # HybridActor 不支持 SDE
            optimize_memory_usage=optimize_memory_usage,  # HybridReplayBuffer 可能不支持
            supported_action_spaces=(spaces.Tuple,),  # <--- 关键：声明支持 Tuple 空间
            support_multi_env=True,  # 支持多环境
        )

        self.target_entropy = target_entropy
        self.log_ent_coef: Optional[th.Tensor] = None
        self.ent_coef = ent_coef  # string or float
        self.target_update_interval = target_update_interval
        self.ent_coef_optimizer: Optional[th.optim.Adam] = None

        if _init_setup_model:
            self._setup_model()  # 现在调用，此时 self.action_space 应该是正确的 Tuple

    def _setup_model(self) -> None:
        super()._setup_model()  # 这会调用 self.policy_class(...) 创建策略实例
        self._create_aliases()  # 创建别名，确保 self.actor, self.critic, self.critic_target 被正确设置

        # batch_norm才需要，一般不用，这里为了保持完整性
        self.batch_norm_stats = get_parameters_by_name(self.critic, ["running_"])
        self.batch_norm_stats_target = get_parameters_by_name(
            self.critic_target, ["running_"]
        )

        # 确保 replay_buffer 是 HybridReplayBuffer
        if (
            not isinstance(self.replay_buffer, HybridReplayBuffer)
            and self.replay_buffer_class is HybridReplayBuffer
        ):
            # 如果父类基于 optimize_memory_usage=False 自动选择了 ReplayBuffer，
            # 我们需要强制替换为 HybridReplayBuffer。
            # (通常，如果 replay_buffer_class 被指定，父类会使用它)
            replay_buffer_args = self._get_replay_buffer_kwargs()
            self.replay_buffer = HybridReplayBuffer(
                self.buffer_size,
                self.observation_space,
                self.action_space,  # 应该已经是 Tuple
                self.device,
                n_envs=self.n_envs,
                # optimize_memory_usage 应为 False， HybridReplayBuffer 构造函数会处理
                handle_timeout_termination=self.handle_timeout_termination,
                **replay_buffer_args,
            )

        # 确保 actor 和 critic 的类型是我们期望的
        # (这一步更多是用于开发者确认和调试，如果 policy 创建正确，类型应该匹配)
        if not isinstance(self.actor, HybridActor):
            warnings.warn(f"Expected actor to be HybridActor, got {type(self.actor)}")
        if not isinstance(self.critic, HybridCritic):
            warnings.warn(
                f"Expected critic to be HybridCritic, got {type(self.critic)}"
            )
        if not isinstance(self.critic_target, HybridCritic):
            warnings.warn(
                f"Expected critic_target to be HybridCritic, got {type(self.critic_target)}"
            )

        # 调整 Target Entropy 的计算
        if self.target_entropy == "auto":
            # 原始 SAC: self.target_entropy = float(-np.prod(self.env.action_space.shape)...)
            # 对于 Tuple(Discrete(n), Box(m,)):
            # 离散部分的熵最大为 log(n)
            # 连续部分的 target_entropy 通常是 -action_dim_continuous (即 -m)
            # 我们可以将目标总熵设为这两者之和，或者只关注连续部分然后让 alpha 学习。
            # 一个简单的做法是只基于连续部分计算，因为离散部分的熵通常较小且固定。
            # 或者，完全手动设置一个目标熵数值。
            if isinstance(self.action_space, spaces.Tuple):
                discrete_space = self.action_space.spaces[0]
                continuous_space = self.action_space.spaces[1]

                # 选项1: 主要关注连续部分 (像原始SAC那样)
                target_entropy_continuous = -float(
                    np.prod(continuous_space.shape).astype(np.float32)
                )

                # 选项2: 尝试包含离散部分的最大熵 (可选)
                # max_entropy_discrete = float(np.log(discrete_space.n))
                # self.target_entropy = target_entropy_continuous + max_entropy_discrete

                self.target_entropy = target_entropy_continuous  # 使用选项1
                print(
                    f"Auto-calculated target entropy for HybridSAC: {self.target_entropy} (based on continuous part)"
                )

            else:  # Fallback or error if action_space is not Tuple as expected
                raise ValueError(
                    "HybridSAC expects a Tuple action space for auto target entropy calculation."
                )

        else:
            self.target_entropy = float(self.target_entropy)

            # ent_coef 学习设置 (copied from SB3 SAC._setup_model)
        if isinstance(self.ent_coef, str) and self.ent_coef.startswith("auto"):
            init_value = 1.0
            if "_" in self.ent_coef:
                init_value = float(self.ent_coef.split("_")[1])
                assert init_value > 0.0, (
                    "The initial value of ent_coef must be greater than 0"
                )

            self.log_ent_coef = th.log(
                th.ones(1, device=self.device) * init_value
            ).requires_grad_(True)
            self.ent_coef_optimizer = th.optim.Adam(
                [self.log_ent_coef], lr=self.lr_schedule(1)
            )
        else:
            self.ent_coef_tensor = th.tensor(float(self.ent_coef), device=self.device)

    def _create_aliases(self) -> None:
        """Create aliases for direct access to policy's actor and critic attributes."""
        if (
            not hasattr(self.policy, "actor")
            or not hasattr(self.policy, "critic")
            or not hasattr(self.policy, "critic_target")
        ):
            raise AttributeError(
                "The policy (HybridSACPolicy) must have 'actor', 'critic', and 'critic_target' attributes."
            )

        # 类型断言 (可选，但有助于静态分析和IDE)
        assert isinstance(self.policy.actor, HybridActor), (
            "Policy's actor is not a HybridActor"
        )
        assert isinstance(self.policy.critic, HybridCritic), (
            "Policy's critic is not a HybridCritic"
        )
        assert isinstance(self.policy.critic_target, HybridCritic), (
            "Policy's critic_target is not a HybridCritic"
        )

        self.actor = self.policy.actor
        self.critic = self.policy.critic
        self.critic_target = self.policy.critic_target

    def reformat_actions_batch(self, actions_tuple_batch):
        """
        将包含离散和连续动作NumPy数组的元组，转换为一个新的结构。
        新的结构是一个大元组，每个元素是一个小元组，格式为：
        (离散动作列表, 连续动作列表)

        参数:
        actions_tuple_batch (tuple): 一个包含两个NumPy数组的元组。
                                     第一个数组是离散动作 (例如 shape (batch_size, discrete_dim))。
                                     第二个数组是连续动作 (例如 shape (batch_size, continuous_dim))。

        返回:
        tuple: 一个大元组，格式为 ((discrete_list_1, continuous_list_1),
                                   (discrete_list_2, continuous_list_2), ...)。
        """

        # 1. 检查输入类型和结构
        if not (
            isinstance(actions_tuple_batch, tuple) and len(actions_tuple_batch) == 2
        ):
            raise TypeError("输入 actions_tuple_batch 必须是一个包含两个元素的元组。")

        discrete_actions_array = actions_tuple_batch[0]
        continuous_actions_array = actions_tuple_batch[1]

        if not isinstance(discrete_actions_array, np.ndarray) or not isinstance(
            continuous_actions_array, np.ndarray
        ):
            raise TypeError("actions_tuple_batch 中的两个元素都必须是 NumPy 数组。")

        # 2. 检查 batch_size 是否一致
        if discrete_actions_array.shape[0] != continuous_actions_array.shape[0]:
            raise ValueError(
                f"离散动作数组和连续动作数组的 batch_size (第一个维度) 必须一致。"
                f"当前分别为: {discrete_actions_array.shape[0]} 和 {continuous_actions_array.shape[0]}"
            )

        batch_size = discrete_actions_array.shape[0]

        # 如果批处理大小为0，则返回空元组
        if batch_size == 0:
            return ()

        # 确保输入数组至少是二维的，以便按行切片
        if discrete_actions_array.ndim < 2 or continuous_actions_array.ndim < 2:
            raise ValueError(
                "离散和连续动作数组都应当至少是二维的 (batch_size, features_dim)。"
                f"当前维度: discrete={discrete_actions_array.ndim}, continuous={continuous_actions_array.ndim}"
            )

        combined_actions_list = []
        # 3. 遍历批处理中的每个项目
        for i in range(batch_size):
            # 4a. 提取当前项目的离散动作部分 (1D NumPy数组切片)
            discrete_slice_np = discrete_actions_array[i]
            # 将其转换为元组 (例如 np.array([1]) 变为 (1,))
            discrete_part_tuple = tuple(discrete_slice_np)

            # 4b. 提取当前项目的连续动作部分 (1D NumPy数组切片)
            continuous_slice_np = continuous_actions_array[i]
            # 将其转换为元组 (例如 np.array([-0.003..., 0.137...]) 变为 (-0.003..., 0.137...))
            continuous_part_tuple = tuple(continuous_slice_np)

            # 4c. 组成小元组 (现在内部元素也是元组)
            inner_tuple = (discrete_part_tuple, continuous_part_tuple)
            combined_actions_list.append(inner_tuple)

        # 5. 将小元组的列表转换为最终的大元组
        return tuple(combined_actions_list)

    def collect_rollouts(
        self,
        env: VecEnv,
        callback: BaseCallback,
        train_freq: TrainFreq,
        replay_buffer: HybridReplayBuffer,  # Type hint to HybridReplayBuffer
        action_noise: Optional[ActionNoise] = None,
        learning_starts: int = 0,
        log_interval: Optional[int] = None,
    ) -> RolloutReturn:
        """
        Collect experiences and store them into a ``ReplayBuffer``.
        This method is overridden to correctly format actions for VecEnv.step().

        :param env: The training environment
        :param callback: Callback that will be called at each step
            (and at the beginning and end of the rollout)
        :param train_freq: How much experience to collect
            by doing rollouts of current policy. Do not train during rollout.
        :param replay_buffer:
        :param action_noise: Action noise that will be used for exploration
            Required for DDPG / TD3 / SAC.
        :param learning_starts: Number of steps before learning for the warm-up phase.
        :param log_interval: Log data every ``log_interval`` episodes
        :return:
        """
        # Switch to eval mode (this affects batch norm / dropout)
        self.policy.set_training_mode(False)

        num_collected_steps, num_collected_episodes = 0, 0

        assert isinstance(env, VecEnv), "You must pass a VecEnv"
        assert train_freq.frequency > 0, "Should at least collect one step or episode."

        if env.num_envs > 1:
            assert train_freq.unit == TrainFrequencyUnit.STEP, (
                "You must use only TrainFrequencyUnit.STEP for multi-env training"
            )

        if self.use_sde:  # Should be False for HybridSAC
            self.actor.reset_noise(env.num_envs)

        callback.on_rollout_start()
        continue_training = True
        while should_collect_more_steps(
            train_freq, num_collected_steps, num_collected_episodes
        ):
            if (
                self.use_sde
                and self.sde_sample_freq > 0
                and num_collected_steps % self.sde_sample_freq == 0
            ):
                # Sample a new noise matrix
                self.actor.reset_noise(env.num_envs)

            # Select action randomly or according to policy
            # actions_tuple_batch is ( (batch_disc_np), (batch_cont_np) )
            actions_tuple_batch, buffer_actions_tuple_batch = self._sample_action(
                learning_starts, action_noise, env.num_envs
            )

            # print(f"actions_tuple_batch = {actions_tuple_batch}")

            # Reshape actions for env.step()
            # env.step() expects a list of actions, one for each environment.
            # For hybrid actions, each action in the list should be a tuple (discrete_action, continuous_action).
            # 2025/5/31 13:08 如果上面这三行注释是正确的话，那么在我修改完sample_action后，返回的
            # actions_tuple_batch 已经是符合要求的数据形式((每个环境的动作元组(离散动作, 连续动作)), (每个环境的动作元组(离散动作, 连续动作)))
            actions_for_env_step: List[Tuple[Any, np.ndarray]] = []
            for i in range(env.num_envs):
                # discrete part might be (n_envs, 1), continuous (n_envs, cont_dim)
                # We need scalar discrete for Discrete space, and (cont_dim,) for Box
                disc_action_i = (
                    actions_tuple_batch[0][i].item()
                    if actions_tuple_batch[0][i].ndim > 0
                    else actions_tuple_batch[0][i]
                )

                cont_action_i = actions_tuple_batch[1][i]
                actions_for_env_step.append((disc_action_i, cont_action_i))

            # print(
            #     f"DEBUG_COLLECT_ROLLOUTS: actions_for_env_step = {actions_for_env_step}"
            # )
            new_obs, rewards, dones, infos = env.step(actions_for_env_step)

            self.num_timesteps += env.num_envs
            num_collected_steps += 1

            # Give access to local variables
            callback.update_locals(locals())
            # Only stop training if return value is False, not when it is None.
            if callback.on_step() is False:
                return RolloutReturn(
                    num_collected_steps * env.num_envs,
                    num_collected_episodes,
                    continue_training=False,
                )

            # Retrieve reward and episode length if using Monitor wrapper
            self._update_info_buffer(infos, dones)

            # Store data in replay buffer (normalized action and unnormalized observation)
            self._store_transition(
                replay_buffer,
                buffer_actions_tuple_batch,
                new_obs,
                rewards,
                dones,
                infos,
            )

            self._update_current_progress_remaining(
                self.num_timesteps, self._total_timesteps
            )

            # For DQN, check if the target network should be updated
            # and update the exploration schedule
            # Not directly applicable to SAC, but OffPolicyAlgorithm has it.
            # self._on_step()

            for idx, done in enumerate(dones):
                if done:
                    # Update stats
                    num_collected_episodes += 1
                    self._episode_num += 1
                    if (
                        log_interval is not None
                        and self._episode_num % log_interval == 0
                    ):
                        self._dump_logs()

        callback.on_rollout_end()
        return RolloutReturn(
            num_collected_steps * env.num_envs,
            num_collected_episodes,
            continue_training,
        )

    def train(self, gradient_steps: int, batch_size: int = 64) -> None:
        # 我们需要修改的是从 replay_buffer 获取数据后，如何将动作传递给 critic

        # 切换到训练模式
        self.policy.set_training_mode(True)

        #  黄金标准：在训练循环的开始，手动更新学习率 ---
        # 1. 计算当前剩余进度
        self._update_current_progress_remaining(
            self.num_timesteps, self._total_timesteps
        )
        progress_remaining = self._current_progress_remaining

        # 2. 从保存在Policy中的、独立的调度函数获取当前的学习率
        new_lr_actor = self.policy.lr_actor_schedule(progress_remaining)
        new_lr_critic = self.policy.lr_critic_schedule(progress_remaining)

        # 3. 将新学习率应用到优化器
        for param_group in self.actor.optimizer.param_groups:
            param_group["lr"] = new_lr_actor
        for param_group in self.critic.optimizer.param_groups:
            param_group["lr"] = new_lr_critic
        # --- 学习率更新逻辑结束 ---

        # 更新优化器学习率
        optimizers = [self.actor.optimizer, self.critic.optimizer]
        if self.ent_coef_optimizer is not None:
            optimizers += [self.ent_coef_optimizer]
        self._update_learning_rate(optimizers)

        ent_coef_losses, ent_coefs = [], []
        actor_losses, critic_losses = [], []

        for gradient_step in range(gradient_steps):
            # 从 HybridReplayBuffer 采样数据
            # replay_data.actions 现在应该是一个 (discrete_actions_th, continuous_actions_th) 的元组
            replay_data = self.replay_buffer.sample(
                batch_size, env=self._vec_normalize_env
            )

            # --- Actor 和 ent_coef 更新 (这部分与 SAC 基本一致，因为 Actor 的输出和 log_prob 已经适配) ---
            # actions_pi 和 log_prob_pi 来自 Actor 对当前观测的评估
            actions_pi_tuple, log_prob_pi = self.actor.action_log_prob(
                replay_data.observations
            )
            log_prob_pi = log_prob_pi.reshape(-1, 1)

            ent_coef_loss = None

            # 这部分代码处理的是自动学习熵系数的情况
            if self.ent_coef_optimizer is not None and self.log_ent_coef is not None:
                ent_coef = th.exp(self.log_ent_coef.detach())
                ent_coef_loss = -(
                    self.log_ent_coef * (log_prob_pi + self.target_entropy).detach()
                ).mean()
                # 记录熵系数损失
                ent_coef_losses.append(ent_coef_loss.item())
            else:
                ent_coef = self.ent_coef_tensor  # 使用确定的 ent_coef

            # 记录熵系数
            ent_coefs.append(ent_coef.item())

            # Optimize entropy coefficient, also called
            # entropy temperature or alpha in the paper
            if ent_coef_loss is not None:
                self.ent_coef_optimizer.zero_grad()
                ent_coef_loss.backward()
                self.ent_coef_optimizer.step()

            # --- Critic 更新 ---
            with th.no_grad():
                # 从当前 Actor 获取下一状态的动作和 log_prob (这部分不变，因为 Actor 返回混合动作元组)
                # 我们从replay中获得s,a,r,s'.这里就是把s'传入Actor，获得下一个动作和log_prob
                next_actions_tuple, next_log_prob = self.actor.action_log_prob(
                    replay_data.next_observations
                )
                next_log_prob = next_log_prob.reshape(-1, 1)

                # 目标 Critic 计算 Q 值 (需要传递动作元组)
                # 我们相信目标critic，他是一个基准。先计算Q(s',a')，然后计算目标Q值
                next_q_values_tuple = self.critic_target(
                    replay_data.next_observations, next_actions_tuple
                )  # 传递元组
                next_q_values = th.cat(next_q_values_tuple, dim=1)
                next_q_values, _ = th.min(next_q_values, dim=1, keepdim=True)
                next_q_values = next_q_values - ent_coef * next_log_prob

                target_q_values = (
                    replay_data.rewards
                    + (1 - replay_data.dones) * self.gamma * next_q_values
                )

            # 当前 Critic 计算 Q 值 (replay_data.actions 是元组)
            # 当前的critic是我们要更新的，为了更新他我们要知道他现在的值是多少
            current_q_values_tuple = self.critic(
                replay_data.observations, replay_data.actions
            )  # 传递元组

            # Critic 损失 (这部分计算不变)
            critic_loss = 0.5 * sum(
                F.mse_loss(current_q, target_q_values)
                for current_q in current_q_values_tuple
            )  # 使用 current_q_values_tuple
            critic_losses.append(critic_loss.item())

            self.critic.optimizer.zero_grad()
            critic_loss.backward()
            th.nn.utils.clip_grad_norm_(
                self.critic.parameters(), max_norm=10.0
            )  # 裁剪Critic梯度

            self.critic.optimizer.step()

            # 计算 Actor 损失
            # 我们先先计算当前transition里面的Q(s,a)
            q_values_pi_tuple = self.critic(
                replay_data.observations, actions_pi_tuple
            )  # 传递元组
            q_values_pi = th.cat(q_values_pi_tuple, dim=1)
            min_qf_pi, _ = th.min(q_values_pi, dim=1, keepdim=True)
            actor_loss = (ent_coef * log_prob_pi - min_qf_pi).mean()
            actor_losses.append(actor_loss.item())

            self.actor.optimizer.zero_grad()
            actor_loss.backward()
            # Clip gradient norm to prevent gradients from exploding
            th.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=1.0)
            self.actor.optimizer.step()

            # 更新目标网络
            if gradient_step % self.target_update_interval == 0:
                polyak_update(
                    self.critic.parameters(), self.critic_target.parameters(), self.tau
                )
                polyak_update(self.batch_norm_stats, self.batch_norm_stats_target, 1.0)

                # batch_norm_stats 的更新可以保留，如果你的 Critic 中有 BatchNorm
                # polyak_update(self.batch_norm_stats, self.batch_norm_stats_target, 1.0)

        self._n_updates += gradient_steps
        # --- 在这里添加新的日志记录 ---
        # 从优化器中直接获取当前的学习率
        # 这是标准的PyTorch做法
        current_lr_actor = self.actor.optimizer.param_groups[0]["lr"]
        current_lr_critic = self.critic.optimizer.param_groups[0]["lr"]

        # 日志记录部分可以基本保持不变
        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        # 记录我们关心的两个独立学习率
        self.logger.record("train/lr_actor", current_lr_actor)
        self.logger.record("train/lr_critic", current_lr_critic)

        self.logger.record(
            "train/ent_coef", np.mean(ent_coefs)
        )  # <--- 使用 np.mean(ent_coefs)
        self.logger.record("train/actor_loss", np.mean(actor_losses))
        self.logger.record("train/critic_loss", np.mean(critic_losses))
        if len(ent_coef_losses) > 0:
            self.logger.record("train/ent_coef_loss", np.mean(ent_coef_losses))

    # learn, _excluded_save_params, _get_torch_save_params 可以从父类 SAC 继承
    # 因为它们主要处理的是通用流程和参数名，不直接与动作的内部结构打交道。
    # _excluded_save_params 中可能需要确保 "actor", "critic", "critic_target" 指的是你的新类型，
    # 但如果它们是 self.policy 的属性，那么保存 self.policy 就能包含它们。
    def learn(
        self: SelfHybridSAC,
        total_timesteps: int,
        callback: MaybeCallback = None,
        log_interval: int = 4,
        tb_log_name: str = "HybridSAC",  # Changed default name
        reset_num_timesteps: bool = True,
        progress_bar: bool = False,
    ) -> SelfHybridSAC:
        return super().learn(
            total_timesteps=total_timesteps,
            callback=callback,
            log_interval=log_interval,
            tb_log_name=tb_log_name,
            reset_num_timesteps=reset_num_timesteps,
            progress_bar=progress_bar,
        )

    def _excluded_save_params(self) -> list[str]:
        return super()._excluded_save_params() + ["actor", "critic", "critic_target"]  # noqa: RUF005

    def _get_torch_save_params(self) -> tuple[list[str], list[str]]:
        state_dicts = ["policy", "actor.optimizer", "critic.optimizer"]
        if self.ent_coef_optimizer is not None:
            saved_pytorch_variables = ["log_ent_coef"]
            state_dicts.append("ent_coef_optimizer")
        else:
            saved_pytorch_variables = ["ent_coef_tensor"]
        return state_dicts, saved_pytorch_variables

    def split_combined_actions(self, actions_iterable) -> Tuple[np.ndarray, np.ndarray]:
        # 这个函数和上面的 reformat_actions_batch是一样的，理论上没啥用
        if isinstance(actions_iterable, (List, tuple)) and not actions_iterable:
            # 如果是空的或者不是列表或者元组
            return np.array([], dtype=np.int64), np.array([], dtype=np.float32)
        discrete_actions_list = []
        continuous_actions_list = []
        item_idx = 0
        for action in actions_iterable:
            # action 应该是一个元组 (discrete_action, continuous_action)
            if not (isinstance(action, tuple) and len(action) == 2):
                raise ValueError(
                    f"Action at index {item_idx} must be a tuple of (discrete_action, continuous_action)."
                )
            # 取出离散和连续动作
            discrete_action, continuous_action = action
            if not isinstance(continuous_action, Tuple):
                raise ValueError(
                    f"输入数据中位置 {item_idx} 的元素 '{action}' 的第二个组成部分 '{continuous_action}' 格式不正确。"
                    "它必须是一个元组 (tuple)。"
                )
            discrete_actions_list.append(discrete_action)
            continuous_actions_list.append(continuous_action)
            item_idx += 1
        if not discrete_actions_list:
            # 如果是空的或者不是列表或者元组
            return np.array([], dtype=np.int64), np.array([], dtype=np.float32)
        # 将列表转换为 numpy 数组
        discrete_actions = np.array(discrete_actions_list, dtype=np.int64)
        continuous_actions = np.array(continuous_actions_list, dtype=np.float32)
        return (discrete_actions, continuous_actions)

    def _sample_action(
        self,
        learning_starts: int,
        action_noise: Optional[ActionNoise] = None,  # SAC doesn't use this
        n_envs: int = 1,
    ) -> Tuple[
        Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]
    ]:  # Return type changed
        """
        Sample an action randomly or according to the policy.
        This method is overridden to handle Tuple action spaces correctly for random sampling.
        Returns actions to be performed and actions to be stored in the buffer.
        For HybridSAC, these are the same and are a tuple of (discrete_batch_np, continuous_batch_np).
        """
        # Select action randomly or according to policy
        if self.num_timesteps < learning_starts and not (
            self.use_sde and self.use_sde_at_warmup
        ):
            # Warmup phase: sample random actions
            if (
                not isinstance(self.action_space, spaces.Tuple)
                or len(self.action_space.spaces) != 2
            ):
                raise ValueError(
                    "HybridSAC._sample_action expects a Tuple action space with a Discrete and a Box subspace."
                )

            discrete_space: spaces.Discrete = self.action_space.spaces[0]
            continuous_space: spaces.Box = self.action_space.spaces[1]

            # Sample n_envs actions
            discrete_actions_list = []
            continuous_actions_list = []
            for _ in range(n_envs):
                # self.action_space.sample() returns (int, np.ndarray) for each env
                sampled_action_tuple = self.action_space.sample()
                discrete_actions_list.append(sampled_action_tuple[0])
                continuous_actions_list.append(sampled_action_tuple[1])

            # Convert lists to batched NumPy arrays
            # Discrete actions: shape (n_envs,) or (n_envs, 1) if discrete_action_dim is 1
            # .sample() for Discrete gives an int. We need to ensure it's an array.
            # The replay buffer (and policy input if not using predict) might expect (n_envs, discrete_action_dim_buffer)
            # where discrete_action_dim_buffer is usually 1 for simple Discrete.
            unscaled_discrete_actions_np = np.array(
                discrete_actions_list, dtype=discrete_space.dtype
            ).reshape(n_envs, -1)

            # Continuous actions: shape (n_envs, continuous_action_dim)
            unscaled_continuous_actions_np = np.array(
                continuous_actions_list, dtype=continuous_space.dtype
            )
            # Ensure correct shape if continuous_space.shape is (dim,) vs (dim,1) etc.
            # np.array of list of (dim,) arrays should result in (n_envs, dim)
            if unscaled_continuous_actions_np.shape != (
                n_envs,
                *continuous_space.shape,
            ):
                unscaled_continuous_actions_np = unscaled_continuous_actions_np.reshape(
                    n_envs, *continuous_space.shape
                )

            # "unscaled_action" for hybrid space is a tuple of these batched arrays
            unscaled_action = (
                unscaled_discrete_actions_np,
                unscaled_continuous_actions_np,
            )
        else:
            # Use policy to predict actions
            # self.predict() should return Tuple[Tuple[np.ndarray, np.ndarray], Optional[Tuple[np.ndarray, ...]]]
            # The first element is the action tuple (batch_disc_np, batch_cont_np)
            assert self._last_obs is not None, "self._last_obs was not set"
            # predict() already handles unscaling for the environment if necessary
            unscaled_action, _ = self.predict(self._last_obs, deterministic=False)
            # unscaled_action = self.split_combined_actions(unscaled_action)
            # unscaled_action is already ( (batch_disc_np), (batch_cont_np) )

        # For SAC, action_noise is typically None.
        # The actions from predict() are already in the environment's scale due to HybridSACPolicy.predict override.
        # Randomly sampled actions from self.action_space.sample() are also in the environment's scale.
        # Therefore, for SAC, the action to be executed ("actions") and the action to be stored ("buffer_actions")
        # are usually the same.
        actions = unscaled_action
        buffer_actions = unscaled_action

        return actions, buffer_actions

    # Add _store_transition if it's not in OffPolicyAlgorithm or if custom logic is needed
    # For HybridReplayBuffer, buffer_actions is already Tuple[np.ndarray, np.ndarray]
    def _store_transition(
        self,
        replay_buffer: HybridReplayBuffer,
        buffer_action: Tuple[
            np.ndarray, np.ndarray
        ],  # This is (batch_disc_np, batch_cont_np)
        new_obs: Union[np.ndarray, Dict[str, np.ndarray]],
        reward: np.ndarray,
        done: np.ndarray,
        infos: List[Dict[str, Any]],
    ) -> None:
        # Store only the unnormalized version
        if self._vec_normalize_env is not None:
            new_obs_ = self._vec_normalize_env.get_original_obs()
            reward_ = self._vec_normalize_env.get_original_reward()
        else:
            # Avoid changing the original ones
            self._last_original_obs, new_obs_, reward_ = self._last_obs, new_obs, reward

        # As self.replay_buffer is HybridReplayBuffer, its add method expects
        # action to be Tuple[np.ndarray, np.ndarray]
        replay_buffer.add(
            self._last_original_obs, new_obs_, buffer_action, reward_, done, infos
        )  # type: ignore[union-attr]

        self._last_obs = new_obs
        # Save the unnormalized observation
        if self._vec_normalize_env is not None:
            self._last_original_obs = new_obs_
