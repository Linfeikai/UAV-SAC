import time
import torch
from torch.distributions import Categorical, Normal
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
from state_normalization import StateNormalizer
from UAV_env import Env
from tqdm import tqdm
import os
import matplotlib.pyplot as plt


# hyperparameters
GAMMA = 0.99
TAU = 0.005  # soft update
ALPHA = 0.4  # 熵正则化参数 这个值大说明更重视探索
LR = 1e-4  # 学习率
BATCH_SIZE = 32  # 训练一批数据的大小
Train_threshold = 256  # 当replaybuffer里有多少条经验时就可以开始训练了
MEMORY_CAPACITY = 50000  # 经验回访池
MAX_EPISODES = 1000


class PolicyNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, max_action, n_ues):
        super(PolicyNetwork, self).__init__()
        # 共享特征提取层
        self.shared = nn.Sequential(
            nn.Linear(state_dim, 512),
            nn.LayerNorm(512),  # 归一化
            nn.Tanh(),  # 激活函数
            nn.Linear(512, 256),
            nn.LayerNorm(256),  # 归一化
            nn.Tanh(),  # 激活函数
        )
        # relu:max(0,x)

        # 离散动作头
        # self.ue_head = nn.Linear(256, n_ues)
        self.ue_head = nn.Sequential(
            nn.Linear(256, 256),
            nn.Tanh(),  # 激活函数
            nn.Linear(256, n_ues),
        )

        # 连续动作头
        self.ratio_head = nn.Linear(256, 2)  # 均值和log_std
        self.direction_head = nn.Linear(256, 2)  # 均值和log_std
        self.speed_head = nn.Linear(256, 2)  # 均值和log_std

        # 在__init__()末尾添加：

        for layer in self.shared:
            if isinstance(layer, nn.Linear):
                nn.init.orthogonal_(layer.weight, gain=0.5)  # 比xavier更稳定
                nn.init.constant_(layer.bias, 0.1)

        nn.init.orthogonal_(self.ue_head[-1].weight, gain=0.01)
        nn.init.constant_(self.ue_head[-1].bias, 0.0)

        print("Actor第一层权重统计：")
        print("权重均值:", self.shared[0].weight.data.mean())
        print(
            "权重极值:",
            self.shared[0].weight.data.min(),
            self.shared[0].weight.data.max(),
        )
        assert not torch.any(torch.isnan(self.shared[0].weight)), "权重包含NaN！"

    # 构建网络，在sample就是想要生成action的时候会调用。
    # forward is used for generating parameters for the distribution
    def forward(self, state):
        assert not torch.isnan(state).any(), "输入包含NaN!"

        # forward天然支持批量输入
        if state.dim() == 1:  # 单样本输入
            state = state.unsqueeze(0)  # [state_dim] -> [1, state_dim]

        # x = self.shared(state)
        x = self.shared[0](state)  # 第一层：Linear
        x = self.shared[1](x)  # LayerNorm
        x = self.shared[2](x)  # ReLU
        # print("After first layer:", x[0, :5])  # 查看部分输出

        x = self.shared[3](x)  # 第二层：Linear
        x = self.shared[4](x)  # LayerNorm
        x = self.shared[5](x)  # ReLU
        # print("After second layer:", x[0, :5])

        # 检查UE头输出
        ue_logits = self.ue_head(x)
        # print("UE logits before clamp:", ue_logits)
        ue_logits = torch.clamp(self.ue_head(x), min=-10, max=10)  # 防止softmax爆炸
        # with open("ue_logits.txt", "a") as f:
        #     f.write(f"  The ue_logit is {ue_logits} ==========")
        # # 在训练循环中添加：
        # print(
        #     f"UE logits - 均值:{ue_logits.mean():.4f}, 标准差:{ue_logits.std():.4f}, 极值:[{ue_logits.min():.4f}, {ue_logits.max():.4f}]"
        # )
        # print("ue_logits:", ue_logits)
        if torch.any(torch.isnan(ue_logits)):
            print("NaNs detected in ue_logits!")

        # 连续动作
        ratio_mu, ratio_logstd = self.ratio_head(x).chunk(2, dim=-1)
        dir_sin, dir_cos = torch.tanh(self.direction_head(x)).chunk(2, dim=-1)
        speed_mu, speed_logstd = self.speed_head(x).chunk(2, dim=-1)

        return (
            ue_logits,
            ratio_mu,
            ratio_logstd,
            dir_sin,
            dir_cos,
            speed_mu,
            speed_logstd,
        )

    def sample_action(self, state, deterministic=False):
        state = torch.as_tensor(state, dtype=torch.float32)  # 转为tensor
        if state.dim() == 1:  # 单样本输入
            state = state.unsqueeze(0)  # [state_dim] -> [1, state_dim]

        # 获取网络输出
        ue_logits, ratio_mu, ratio_logstd, dir_sin, dir_cos, speed_mu, speed_logstd = (
            self.forward(state)
        )
        # print("ue_logits:", ue_logits.shape)  # [batch_size, n_ues]
        # print("ratio_mu:", ratio_mu.shape)
        # print("ratio_logstd:", ratio_logstd.shape)
        # print("dir_sin:", dir_sin.shape)  # [batch_size, 2]
        # print("dir_cos:", dir_cos.shape)  # [batch_size, 2]
        # print("speed_mu:", speed_mu.shape)  # [batch_size, 2]
        # print("speed_logstd:", speed_logstd.shape)  # [batch_size, 2]

        # 1. 离散动作（UE选择）

        # 将logits转换为概率分布（自动完成softmax）logits是一个大小为[batch_size, n_ues]的张量
        ue_dist = Categorical(logits=ue_logits)
        ue_action = (
            ue_dist.probs.argmax(dim=-1) if deterministic else ue_dist.sample()
        )  # 按概率采样动作
        # ue_action = ue_dist.sample()  # 按概率采样动作
        ue_log_prob = ue_dist.log_prob(ue_action)  # 计算选择ue的log概率

        # 2. 连续动作
        # 卸载比率
        if deterministic:
            ratio_action = torch.sigmoid(ratio_mu)
        else:
            ratio_std = torch.exp(ratio_logstd).clamp(0.1, 2.0)  # 防止方差爆炸
            ratio_dist = Normal(ratio_mu, ratio_std)
            ratio_action = torch.sigmoid(
                ratio_dist.rsample()  # sigmoid 将值映射到[0, 1],rsample 是重参数化技巧，使得梯度能够回传
            )

        ratio_log_prob = ratio_dist.log_prob(ratio_action).sum(
            dim=-1
        )  # 计算动作的log概率
        ratio_log_prob = torch.clamp(ratio_log_prob, min=-50, max=0)  # 限制范围

        # 飞行方向
        angle = torch.atan2(dir_sin, dir_cos)  # [-π, π]
        final_angle = (angle + 2 * np.pi) % (2 * np.pi)  # [0, 2π]

        # 飞行速度
        if deterministic:
            speed_action = torch.sigmoid(speed_mu)
        else:
            speed_std = torch.exp(speed_logstd).clamp(0.1, 2.0)  # 防止方差爆炸
            speed_dist = Normal(speed_mu, speed_std)
            speed_action = torch.sigmoid(speed_dist.rsample())

        speed_log_prob = speed_dist.log_prob(speed_action).sum(
            dim=-1
        )  # 计算动作的log概率
        speed_log_prob = torch.clamp(speed_log_prob, min=-50, max=0)

        # combine actions
        ue_action = ue_action.unsqueeze(-1)  # [batch_size, 1]
        # ratio_action = ratio_action.squeeze(-1)  # [batch_size, 1]
        # final_angle = final_angle.squeeze(-1)  # [batch_size, 1]
        # speed_action = speed_action.squeeze(-1)  # [batch_size, 1]
        # print("ratio_action:", ratio_action.shape)  # [batch_size, 1]
        # print("final_angle:", final_angle.shape)  # [batch_size, 1]
        combined_action = torch.cat(
            [ue_action, final_angle, speed_action, ratio_action], dim=1
        )
        # print(
        #     f"UE log_prob: {ue_log_prob.mean()}, Ratio log_prob: {ratio_log_prob.mean()}, Speed log_prob: {speed_log_prob.mean()}"
        # )
        # Combine log probabilities
        log_probs = ue_log_prob + ratio_log_prob + speed_log_prob
        log_probs = torch.clamp(log_probs, -50, 0)  # 绝对约束

        return (combined_action, log_probs)  # 返回动作和log概率


class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(QNetwork, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, 512),
            nn.LayerNorm(512),  # 归一化
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.LayerNorm(256),  # 归一化
            nn.ReLU(),
            nn.Linear(256, 1),
        )

    def forward(self, state, action):
        x = torch.cat((state, action), -1)
        x = self.net(x).squeeze(-1)
        return x


class ReplayBuffer(object):
    def __init__(self, capacity: int, state_dim: int, action_dim: int):
        self.capacity = capacity
        self.state_dim = state_dim
        self.action_dim = action_dim
        # 使用numpy数组预分配空间
        self.states = np.zeros((self.capacity, state_dim), dtype=np.float32)
        self.actions = np.zeros((self.capacity, action_dim), dtype=np.float32)
        self.rewards = np.zeros(self.capacity, dtype=np.float32)
        self.next_states = np.zeros((self.capacity, state_dim), dtype=np.float32)
        self.dones = np.zeros(self.capacity, dtype=np.float32)

        self.size = 0  # 当前存储的数据量
        self.pos = 0  # 当前插入位置

    def store_transition(self, state, action, reward, next_state, done):
        """
        添加一条经验到缓冲区
        """
        if not isinstance(state, np.ndarray) or not isinstance(action, np.ndarray):
            raise TypeError("State and action must be of type np.ndarray")
        if not isinstance(reward, (float, int, np.float32)):
            raise TypeError("Reward must be a float, int, or numpy.float32")
        if not isinstance(done, (bool, int, float)):  # 兼容布尔和数字
            raise TypeError("Done must be a boolean or numeric value")
        if self.capacity == 0:
            raise ValueError("Max size of the replay buffer must be greater than 0")

        # 存储数据到当前指针位置
        self.states[self.pos] = state
        self.actions[self.pos] = action
        self.rewards[self.pos] = reward
        self.next_states[self.pos] = next_state
        self.dones[self.pos] = done

        # 更新指针和大小
        self.pos = (self.pos + 1) % self.capacity  # 循环覆盖
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size):
        # 随机采样一批数据

        # 随机选择索引
        indices = np.random.choice(self.size, size=batch_size, replace=False)
        # 从缓冲区取出数据,states先不转换为tensor,因为后面要归一化
        states = self.states[indices]
        actions = torch.FloatTensor(self.actions[indices])
        rewards = torch.FloatTensor(self.rewards[indices])
        next_states = self.next_states[indices]
        # dones = self.dones[indices]
        dones = torch.FloatTensor(self.dones[indices])
        return states, actions, rewards, next_states, dones

    def __len__(self):
        return self.size


class SACAgent:
    def __init__(self, state_dim: int, action_dim: int, max_action, ue_num):
        self.state_dim = state_dim
        self.action_dim = action_dim

        # 初始化状态归一器
        self.state_normalizer = StateNormalizer(state_dim)
        # 初始化网络
        self.actor = PolicyNetwork(state_dim, action_dim, max_action, ue_num)
        self.critic1 = QNetwork(state_dim, action_dim)
        self.critic2 = QNetwork(state_dim, action_dim)
        self.target_critic1 = QNetwork(state_dim, action_dim)
        self.target_critic2 = QNetwork(state_dim, action_dim)

        # 同步目标网络
        self._hard_update(self.target_critic1, self.critic1)
        self._hard_update(self.target_critic2, self.critic2)

        # 初始化优化器
        self.actor_optim = torch.optim.Adam(self.actor.parameters(), lr=LR)
        self.critic1_optim = torch.optim.Adam(self.critic1.parameters(), lr=LR)
        self.critic2_optim = torch.optim.Adam(self.critic2.parameters(), lr=LR)

        # self.value_net_optimizer = torch.optim.Adam(self.value_net.parameters(), lr=LR)

        # 初始化缓冲区
        self.replay_buffer = ReplayBuffer(
            MEMORY_CAPACITY, self.state_dim, self.action_dim
        )

        self.maxaction = max_action

        # 和log_alpha相关的变量

        # 温度系数
        self.alpha = ALPHA
        self.target_entropy = -4  # 目标熵是动作空间大小的负数
        self.log_alpha = torch.zeros(
            1, requires_grad=True
        )  # log_alpha是一个可学习的参数
        self.alpha_optim = torch.optim.Adam(
            [self.log_alpha],
            lr=LR,
        )  # 学习率
        self.gamma = 0.99  # 折扣因子

    # 把原网络的参数复制到目标网络中
    def _hard_update(self, target_net, source_net):
        target_net.load_state_dict(source_net.state_dict())

    # 用于选择动作
    def select_action(self, state, deterministic=False):
        state = torch.FloatTensor(state).unsqueeze(
            0
        )  # Convert to tensor with batch dimension
        assert not torch.any(torch.isnan(state)), f"Tensor contains NaN: {state}"
        assert not torch.any(torch.isinf(state)), f"Tensor contains Inf: {state}"
        # print(
        #     "输入值统计 - 均值:", state.mean(), "极值:", state.min(), state.max()
        # )  # 新增
        assert torch.all(state.abs() < 1e5), f"输入值超出安全范围: {state}"

        with torch.no_grad():
            (
                ue_logits,
                ratio_mu,
                ratio_logstd,
                dir_sin,
                dir_cos,
                speed_mu,
                speed_logstd,
            ) = self.actor(state)  # 调用actor的forward方法

            # 1. 离散动作（UE选择）
            ue_dist = Categorical(logits=ue_logits)
            ue_action = (
                ue_dist.probs.argmax(dim=-1) if deterministic else ue_dist.sample()
            )

            # 2. 连续动作
            ratio_action = (
                torch.sigmoid(ratio_mu)
                if deterministic
                else torch.sigmoid(Normal(ratio_mu, torch.exp(ratio_logstd)).rsample())
            )
            angle = torch.atan2(dir_sin, dir_cos)  # [-π, π]
            final_angle = (angle + 2 * np.pi) % (2 * np.pi)  # [0, 2π]
            speed_action = (
                speed_mu
                if deterministic
                else torch.clamp(
                    Normal(speed_mu, torch.exp(speed_logstd)).rsample(), 0, 1
                )
            )

        return {
            "ue": ue_action.item(),
            "ratio": ratio_action.squeeze().numpy(),
            "angle": final_angle.squeeze().numpy(),
            "speed": speed_action.squeeze().numpy(),
        }

    # 软更新目标网络
    def _soft_update(self, target_net, source_net, tau=TAU):
        """
        Soft update the target network parameters using the source network parameters.
        target_param = tau * source_param + (1 - tau) * target_param
        """
        for target_param, source_param in zip(
            target_net.parameters(), source_net.parameters()
        ):
            target_param.data.copy_(
                tau * source_param.data + (1 - tau) * target_param.data
            )

    def train(self, stateNormalizer=None):
        if len(self.replay_buffer) < Train_threshold:
            return
        # 从回放缓冲区采样出一个batch的transition,取出来的是numpy格式
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(
            BATCH_SIZE
        )
        if stateNormalizer is not None:
            states_tensor = torch.tensor(
                [stateNormalizer.normalize(s) for s in states], dtype=torch.float32
            )
            next_states_tensor = torch.tensor(
                [stateNormalizer.normalize(s) for s in next_states], dtype=torch.float32
            )
        # print
        # ----1.更新critic网络
        with torch.no_grad():
            # 采样下一动作和他的log_pob
            next_actions, next_log_probs = self.actor.sample_action(next_states_tensor)
            next_log_probs = torch.clamp(next_log_probs, min=-10, max=10)
            # 计算下一个s,a的值
            next_q1 = self.target_critic1(next_states_tensor, next_actions)
            next_q2 = self.target_critic2(next_states_tensor, next_actions)
            min_next_q = torch.min(next_q1, next_q2)
            # print(
            #     f"Next_Q范围: {min_next_q.min().item():.4f}~{min_next_q.max().item():.4f}"
            # )  # 检查是否出现1e6等异常值
            q_target = rewards + self.gamma * (1 - dones) * (
                min_next_q - self.alpha * next_log_probs  # 目标熵一般设为-dim(A)
            )
            # print(q_target.shape)  # [batch_size, 1]

        # 计算两个Q网络的损失
        current_q1 = self.critic1(states_tensor, actions)  # 当前q值
        current_q2 = self.critic2(states_tensor, actions)
        huber_loss = torch.nn.SmoothL1Loss()
        critic1_loss = huber_loss(current_q1, q_target)
        critic2_loss = huber_loss(current_q2, q_target)

        # 梯度传播 更新参数
        self.critic1_optim.zero_grad()
        critic1_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            self.critic1.parameters(), max_norm=0.3
        )  # 从0.5降到0.3
        self.critic1_optim.step()

        self.critic2_optim.zero_grad()
        critic2_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            self.critic2.parameters(), max_norm=0.3
        )  # 从0.5降到0.3
        self.critic2_optim.step()

        # --- 2. 更新Actor网络 ---
        new_actions, new_log_probs = self.actor.sample_action(states_tensor)
        # print(
        #     f"new_log_probs: {new_log_probs.min().item()} - {new_log_probs.max().item()}"
        # )

        min_q = torch.min(
            self.critic1(states_tensor, new_actions),
            self.critic2(states_tensor, new_actions),
        )
        # print(f"min_q: {min_q.min().item()} - {min_q.max().item()}")

        actor_loss = 0.1 * (self.alpha * new_log_probs - min_q).mean()

        self.actor_optim.zero_grad()
        actor_loss.backward()
        # 进行梯度裁剪
        torch.nn.utils.clip_grad_norm_(
            self.actor.parameters(), max_norm=0.5
        )  # 推荐0.5-1.0
        # torch.nn.utils.clip_grad_value_(self.actor.parameters(), clip_value=1.0)

        self.actor_optim.step()
        # --- 3. 更新温度系数α ---

        safe_entropy_diff = torch.clamp(
            new_log_probs.detach() + self.target_entropy, -10, 10
        )
        alpha_loss = -(self.log_alpha * safe_entropy_diff).mean()

        # 这里的target_entropy是一个常数，表示我们希望的熵值，log_alpha是一个可学习的参数
        # 通过最小化这个损失函数来调整α，使得熵值接近target_entropy
        # .mean()是对当前batch样本的平均值

        self.alpha_optim.zero_grad()
        # 清空优化器中的历史梯度
        alpha_loss.backward()
        self.alpha_optim.step()

        # 最后把log_alpha转换回alpha
        self.alpha = self.log_alpha.exp()

        # --- 可视化损失 ---
        if hasattr(self, "loss_history"):
            self.loss_history["critic1"].append(critic1_loss.item())
            self.loss_history["critic2"].append(critic2_loss.item())
            self.loss_history["actor"].append(actor_loss.item())
            self.loss_history["alpha"].append(alpha_loss.item())
        else:
            self.loss_history = {
                "critic1": [critic1_loss.item()],
                "critic2": [critic2_loss.item()],
                "actor": [actor_loss.item()],
                "alpha": [alpha_loss.item()],
            }

        # 可选：打印损失值
        # print(
        #     f"Critic1 Loss: {critic1_loss.item():.4f}, Critic2 Loss: {critic2_loss.item():.4f}, "
        #     f"Actor Loss: {actor_loss.item():.4f}, Alpha Loss: {alpha_loss.item():.4f}"
        # )

        # --- 4. 软更新目标网络 ---
        self._soft_update(self.target_critic1, self.critic1)
        self._soft_update(self.target_critic2, self.critic2)

        return critic1_loss.item(), critic2_loss.item(), actor_loss.item()

    def save(self, filename):
        os.makedirs(os.path.dirname(filename), exist_ok=True)

        torch.save(self.actor.state_dict(), filename + "_actor.pth")
        torch.save(self.critic1.state_dict(), filename + "_q1.pth")
        torch.save(self.critic2.state_dict(), filename + "_q2.pth")
        # torch.save(self.value_net.state_dict(), filename + "_value.pth")

    def load(self, filename):
        self.actor.load_state_dict(torch.load(filename + "_actor.pth"))
        self.q1.load_state_dict(torch.load(filename + "_q1.pth"))
        self.q2.load_state_dict(torch.load(filename + "_q2.pth"))
        self.value_net.load_state_dict(torch.load(filename + "_value.pth"))


def anothermain():
    np.random.seed(48)  # 设置随机种子
    torch.manual_seed(48)  # 设置随机种子
    env = Env()
    agent = SACAgent(
        state_dim=env.state_dim,
        action_dim=env.action_dim,
        max_action=env.max_action,
        ue_num=env.ue_num,
    )
    s_normalizer = StateNormalizer(env.state_dim)
    start_time = time.time()
    ep_reward_list = []
    avg_ratio_list = []  # 用于存储平均卸载比率

    # 使用 tqdm 包裹外层循环（总训练进度）
    with tqdm(total=MAX_EPISODES, desc="Training", unit="episode") as pbar:
        for i in range(MAX_EPISODES):
            s = np.array(env.reset())
            episode_reward = 0
            flying_trajectory = []

            # 内层循环（单个 episode 的步骤）
            for j in range(env.slot_num):
                flying_trajectory.append(env.uav.loc.copy())  # 记录飞行轨迹

                s = s_normalizer.normalize(s)  # 归一化状态
                assert not np.any(np.isnan(s)), (
                    f"State contains NaN after normalization: {s}"
                )
                assert not np.any(np.isinf(s)), (
                    f"State contains Inf after normalization: {s}"
                )

                a = agent.select_action(s, deterministic=False)  # 选择动作
                s_, r, is_terminal = env.step(a)  # 执行动作
                # print(f"现在的奖励是{r}")
                s_ = np.array(s_, dtype=np.float32)  # 将下一个状态转换为 numpy 格式
                store_a = np.array(
                    list(a.values()), dtype=np.float32
                )  # 将动作转换为 numpy 格式（因为现在 agent 返回的动作是字典格式）
                agent.replay_buffer.store_transition(
                    s, store_a, r, s_, is_terminal
                )  # 存储经验
                agent.train(s_normalizer)  # 训练 agent
                s = s_  # 更新状态
                episode_reward += r

                # 如果到达终止条件，提前结束当前 episode
                if j == env.slot_num - 1 or is_terminal:
                    ep_reward_list.append(episode_reward)
                    print(f"Episode {i + 1} finished with reward: {episode_reward:.2f}")

                    # 每 10 个 episode 可视化一次
                    if (i + 1) % 10 == 0:
                        env.visualize_nodes(flying_trajectory, i)

                    # 记录日志
                    with open("output.txt", "a") as f:
                        f.write(f"Episode #{i + 1} Finished  ==========")
                        f.write(
                            f"This episode finished within {j} steps , accumulated reward: {episode_reward:.2f}"
                        )
                    break

            # 更新进度条（每完成一个 episode）
            pbar.update(1)

            # 动态更新进度条描述（显示最新 10 个 episode 的平均奖励）
            if len(ep_reward_list) >= 10:
                avg_reward = np.mean(ep_reward_list[-10:])
                pbar.set_postfix({"Avg Reward": f"{avg_reward:.2f}"})
            else:
                pbar.set_postfix({"Avg Reward": f"{np.mean(ep_reward_list):.2f}"})

            # 每 100 个 episode 保存模型
            if (i + 1) % 100 == 0:
                agent.save(f"model/episode_{i + 1}")
                print(f"Model saved at episode {i + 1}")

    end_time = time.time()

    print(f"Running time: {end_time - start_time:.2f} s")
    print(f"Average reward per episode: {np.mean(ep_reward_list):.2f}")
    plt.plot(ep_reward_list)
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title("Episode vs Reward")
    plt.savefig("reward_plot.png")
    plt.show()


def testEnv():
    policy_network = PolicyNetwork(state_dim=87, action_dim=4, max_action=1, n_ues=20)
    single_state = torch.randn(87)
    testEnv = Env()


def main():
    env = Env()
    agent = SACAgent(
        state_dim=env.state_dim,
        action_dim=env.action_dim,
        max_action=env.max_action,
        ue_num=env.ue_num,
    )
    s_normalizer = StateNormalizer(env.state_dim)
    start_time = time.time()
    ep_reward_list = []
    for i in range(MAX_EPISODES):
        # Reset at the beginning of each episode
        s = np.array(env.reset())
        episode_reward = 0
        flying_trajectory = []
        # 记录飞行轨迹
        for j in range(env.slot_num):
            flying_trajectory.append(env.uav.loc.copy())
            a = agent.select_action(s_normalizer.normalize(s), deterministic=False)
            s_, r, is_terminal = env.step(a)
            # print(f"现在的奖励是{r}")
            s_ = np.array(s_, dtype=np.float32)  # 把env返回的下一个状态变成numpy格式
            store_a = np.array(
                list(a.values()), dtype=np.float32
            )  # 把动作变成numpy格式（因为现在agent返回的动作是字典格式）
            agent.replay_buffer.store_transition(s, store_a, r, s_, is_terminal)
            agent.train()
            s = s_
            episode_reward += r
            if j == env.slot_num - 1 or is_terminal:
                ep_reward_list.append(episode_reward)
                # 记录飞行轨迹
                print(f"Episode {i + 1} finished with reward: {episode_reward:.2f}")
                if (i + 1) % 10 == 0:
                    env.visualize_nodes(flying_trajectory, i)
                with open("output.txt", "a") as f:
                    f.write(f"\n =========== Episode #{i + 1} Finished  ==========")
                    f.write(
                        f"\n This episode finished within {j} steps , accmulated reward: {episode_reward:.2f}"
                    )
                break
        # 每100个episode保存一次模型
        # if (i + 1) % 100 == 0:
        #     agent.save(f"model/episode_{i + 1}")
        #     print(f"Model saved at episode {i + 1}")

    end_time = time.time()
    print(f"Running time: {end_time - start_time:.2f} s")
    print(f"Average reward per episode: {np.mean(ep_reward_list):.2f}")
    plt.plot(ep_reward_list)
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title("Episode vs Reward")
    plt.savefig("reward_plot.png")
    plt.show()


def testPolicy():
    # 测试policy
    """
    Test function for the PolicyNetwork class.

    Creates a PolicyNetwork instance and tests single state action sampling by:
    - Initializing a policy network with state_dim=87, action_dim=4
    - Generating a random state tensor
    - Sampling actions (UE ID, angle, distance, offloading ratio)
    - Printing the sampled action values and their shapes

    The commented code also includes batch testing functionality.
    """
    policy_network = PolicyNetwork(state_dim=87, action_dim=4, max_action=1, n_ues=20)
    single_state = torch.randn(87)
    action_dict, _ = policy_network.sample_action(single_state)
    action_dict = action_dict.squeeze(0)  # [1, 4] -> [4]
    ue_id, uav_angle, uav_distance, offloading_ratio = action_dict
    # print("UE ID:", ue_id)  # torch.Size([1])
    # print("Angle:", uav_angle)  # torch.Size([1])
    # print("Distance:", uav_distance)  # torch.Size([1])
    # print("Offloading Ratio:", offloading_ratio)  # torch.Size([1, 2])
    # 测试批量状态
    # batch_states = torch.randn(32, 87)
    # batch_actions = policy_network.sample_action(batch_states)
    # print("Batch Actions:", batch_actions["ue"])  # torch.Size([32])
    # print("Batch Angles:", batch_actions["angle"])  # torch.Size([32])
    # print("Batch Speeds:", batch_actions["speed"])  # torch.Size([32])
    # print("Batch Actions:", batch_actions["ratio"])  # torch.Size([32, 2])


if __name__ == "__main__":
    anothermain()

    # testPolicy()
