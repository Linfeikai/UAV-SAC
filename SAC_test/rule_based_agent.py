import math

import numpy as np

from .entities.custom_env_sac import CustomEnv


class RuleBasedAgent:
    """
    一个简单的、基于规则的智能体。

    该智能体在每个时间步都执行一个贪心策略：
    1. 找到当前任务队列数据量最大的UE。
    2. 全速飞向该UE。
    3. 将该UE的所有任务都进行卸载计算。
    """

    def predict(self, obs: np.ndarray, env: CustomEnv) -> tuple[np.ndarray, None]:
        """
        根据简单的启发式规则预测一个动作。

        Args:
            obs: 环境的当前观察值 (此智能体未使用)。
            env: 环境实例，用于访问内部状态（如UE列表和UAV位置）。

        Returns:
            一个包含动作和None（用于状态）的元组。
        """
        # 1. 找到任务队列最大的UE (使用NumPy进行优化，效率更高)
        # env.ue_cache_sizes 是一个NumPy数组，存储了所有UE的当前缓存大小
        ue_id = np.argmax(env.ue_cache_sizes)
        most_demanding_ue = env.nodeList[ue_id]

        # 2. 计算朝向该UE的方向角度
        uav_loc = env.uav.loc
        ue_loc = most_demanding_ue.loc
        dx = ue_loc[0] - uav_loc[0]
        dy = ue_loc[1] - uav_loc[1]
        angle = math.atan2(dy, dx)

        # 3. 构造动作：永远全速、永远完全卸载
        action = np.array([ue_id, angle, 1.0, 1.0], dtype=np.float32)

        return action, None
