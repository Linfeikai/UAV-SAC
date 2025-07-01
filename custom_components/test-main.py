import gymnasium as gym
import numpy as np
import torch as th
from typing import Tuple

# 导入我们全新设计的Agent和Policy
# 确保所有 .py 文件都在同一个目录下
from hybrid_sac_agent import DiffusionSACAgent
from hybrid_sac_policy import DiffusionSACPolicy


# =====================================================================================
# 关键部分：环境包装器 (Wrapper) 示例
# 这是您将我们设计的Agent应用到您自己无人机环境所需要遵循的模式。
# 它负责将Agent输出的"统一连续动作"解码为您环境需要的"混合动作"。
# =====================================================================================
class UAVEnvWrapper(gym.Wrapper):
    """
    一个包装器，将一个需要混合动作(Tuple)的无人机环境，
    伪装成一个接受连续动作(Box)的环境，以便与我们的DiffusionSACAgent兼容。
    """

    def __init__(self, env, ue_embeddings: np.ndarray):
        super().__init__(env)

        # 1. 保存UE的嵌入向量
        self.ue_embeddings = (
            th.from_numpy(ue_embeddings)
            .float()
            .to("cuda" if th.cuda.is_available() else "cpu")
        )
        self.num_ues = ue_embeddings.shape[0]
        self.ue_embedding_dim = ue_embeddings.shape[1]

        # 2. 从原始环境中获取连续动作空间的维度
        original_continuous_space = self.env.action_space.spaces[1]
        self.continuous_dim = original_continuous_space.shape[0]

        # 3. 定义新的、统一的、对Agent可见的动作空间 (Box)
        # 新空间维度 = UE嵌入维度 + 原始连续动作维度
        new_action_dim = self.ue_embedding_dim + self.continuous_dim
        self.action_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(new_action_dim,), dtype=np.float32
        )
        print(f"包装器已创建：")
        print(f"  - 原始混合动作空间: {self.env.action_space}")
        print(f"  - 新的统一连续动作空间: {self.action_space}")

    def action(self, action: np.ndarray) -> Tuple[int, np.ndarray]:
        """
        这个方法负责"解码"Agent输出的连续动作。
        :param action: Agent输出的、单一的连续动作向量。
        :return: 原始环境可以理解的混合动作元组 (离散UE索引, 连续飞行参数)。
        """
        # 将numpy动作转为torch张量以便计算
        action_th = th.from_numpy(action).float().to(self.ue_embeddings.device)

        # a. 解码离散部分 (选择UE)
        #    取出动作向量中代表UE嵌入的部分
        ue_vector_from_action = action_th[: self.ue_embedding_dim]
        #    计算它与所有预定义UE嵌入向量的距离（这里用负的L2距离作为相似度）
        distances = -th.sum((self.ue_embeddings - ue_vector_from_action) ** 2, dim=1)
        #    选择距离最近（相似度最高）的那个UE的索引
        discrete_action = th.argmax(distances).item()

        # b. 解码连续部分 (飞行参数)
        #    直接取出动作向量的剩余部分
        continuous_action = action[self.ue_embedding_dim :]

        # c. 返回原始环境能理解的混合动作元组
        return (discrete_action, continuous_action)

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, dict]:
        """
        重写step方法。在将动作传递给真实环境之前，先进行解码。
        """
        # 调用我们上面定义的action方法进行解码
        hybrid_action = self.action(action)
        return self.env.step(hybrid_action)


# =====================================================================================
# 主函数：运行测试
# =====================================================================================
if __name__ == "__main__":
    # --- 步骤1: 创建一个标准测试环境 ---
    # 我们使用 Pendulum-v1，因为它是一个简单的、纯连续动作的环境。
    # 这可以帮助我们快速验证算法本身是否能工作，而不用立即处理您复杂的无人机环境。
    # 它的动作空间是 Box(1,)，观察空间是 Box(3,)。
    env = gym.make("Pendulum-v1")

    # --- 如何适配您自己的无人机环境 (示例) ---
    # 当您想在自己的项目上测试时，您会像下面这样使用包装器：
    #
    # 1. 先创建您的原始无人机环境
    #    uav_env_original = gym.make("YourUAVEnv-v0")
    #
    # 2. 定义UE嵌入向量
    #    NUM_UES = 8
    #    UE_EMBEDDING_DIM = 4
    #    ue_embeddings = np.random.randn(NUM_UES, UE_EMBEDDING_DIM).astype(np.float32)
    #
    # 3. 用包装器包裹您的环境
    #    env = UAVEnvWrapper(uav_env_original, ue_embeddings)
    #
    # 这样，变量 'env' 对于我们的Agent来说，就是一个标准的、拥有Box动作空间的环境了。

    # --- 步骤2: 实例化我们全新的 Agent ---
    # 请注意，我们传入的是我们自己定义的 DiffusionSACPolicy
    model = DiffusionSACAgent(
        policy=DiffusionSACPolicy,  # 直接传入我们定义的Policy类
        env=env,
        verbose=1,  # 打印训练信息
        tensorboard_log="./diffusion_sac_pendulum_tensorboard/",  # Tensorboard日志路径
        # 为Agent和Policy设置一些用于快速测试的超参数
        learning_starts=1000,  # 1000步后开始学习
        batch_size=128,
        # 传入扩散模型和QNE所需的特定超参数
        qne_k_samples=32,  # QNE中的K值，可以先设小一点以加快速度
        policy_kwargs=dict(
            T=5,  # 扩散总步数，新的参数名为T
            beta_schedule="linear",  # 可以指定beta的调度方式
            net_arch=[64, 64],  # EpsilonNet和Critic的隐藏层维度
        ),
    )

    # --- 步骤3: 开始训练 ---
    # 我们先只训练一小段时间，进行"点火测试"，看看是否能跑通
    print("--- 开始使用 DiffusionSACAgent进行训练 ---")
    model.learn(total_timesteps=20000, log_interval=4)

    print("\n--- 训练完成！---")
    print(
        "如果程序没有报错，并且在Tensorboard日志中能看到loss在下降，说明我们的新Agent已成功运行！"
    )

    # # --- 步骤4: 测试训练好的模型 (可选) ---
    # print("\n--- 测试训练好的模型 ---")
    # obs, _ = env.reset()
    # for _ in range(200):
    #     action, _states = model.predict(obs, deterministic=True)
    #     obs, reward, terminated, truncated, info = env.step(action)
    #     if terminated or truncated:
    #         obs, _ = env.reset()

    # env.close()
