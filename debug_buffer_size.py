import gymnasium as gym
from stable_baselines3 import SAC
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.buffers import ReplayBuffer
import os
import pickle

# --- 参数设置 ---
N_ENVS = 4
TOTAL_TIMESTEPS = 100000
LEARNING_STARTS = 1000
BUFFER_SIZE = 1000000
BUFFER_SAVE_PATH = "debug_buffer.pkl"


def run_test():
    """
    一个最小化的测试，用于验证SB3在并行环境下填充Replay Buffer的行为。
    """
    print("--- 启动最小化复现实验 ---")
    print(f"并行环境数量 (N_ENVS): {N_ENVS}")
    print(f"总训练步数 (TOTAL_TIMESTEPS): {TOTAL_TIMESTEPS}")
    print(f"学习开始步数 (LEARNING_STARTS): {LEARNING_STARTS}")
    print("-" * 30)

    # 1. 创建一个标准的、不会出错的Gym环境 (Pendulum-v1)
    #    我们使用SubprocVecEnv来完全模拟你之前的设置
    env = make_vec_env("Pendulum-v1", n_envs=N_ENVS, vec_env_cls=SubprocVecEnv)

    # 2. 创建一个标准的SAC模型
    model = SAC(
        "MlpPolicy",
        env,
        buffer_size=BUFFER_SIZE,
        learning_starts=LEARNING_STARTS,
        verbose=1,
    )

    # 3. 运行learn方法
    model.learn(total_timesteps=TOTAL_TIMESTEPS * N_ENVS)

    # 4. 保存Replay Buffer
    print(f"\n训练完成，保存经验池到 {BUFFER_SAVE_PATH}...")
    model.save_replay_buffer(BUFFER_SAVE_PATH)
    print("保存成功！")

    env.close()


def check_buffer_size():
    """
    检查保存的debug_buffer.pkl的大小。
    """
    if not os.path.exists(BUFFER_SAVE_PATH):
        print(f"错误: 找不到 {BUFFER_SAVE_PATH}。请先运行run_test()。")
        return

    with open(BUFFER_SAVE_PATH, "rb") as f:
        buffer = pickle.load(f)

    expected_size = TOTAL_TIMESTEPS - LEARNING_STARTS
    actual_size = buffer.size()

    print("\n" + "=" * 30)
    print("--- 最终Buffer大小检查 ---")
    print(f"理论期望大小: {TOTAL_TIMESTEPS} - {LEARNING_STARTS} = {expected_size}")
    print(f"实际文件中大小: {actual_size}")

    if actual_size == expected_size:
        print("\n[结论] ✅ 结果符合预期！SB3核心功能正常。")
        print("这强烈表明问题出在你的自定义环境(UAVEnv-v1)或包装器(UAVEnvWrapper)中。")
    else:
        print("\n[结论] ❌ 结果不符合预期！")
        print(
            f"这表明在你当前环境中，SB3的行为存在异常，可能与版本或依赖有关。实际大小为期望的 {actual_size / expected_size:.2f} 倍。"
        )
    print("=" * 30)


if __name__ == "__main__":
    # 先运行测试来生成buffer文件
    run_test()
    # 然后检查生成的文件
    check_buffer_size()
