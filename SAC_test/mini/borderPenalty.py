import numpy as np


def calculate_boundary_penalty(current_positions, X_min, X_max, W):
    """
    计算多无人机系统的边界越界惩罚

    参数：
        current_positions: 无人机当前位置列表 [N x D] (N为无人机数，D为维度)
        X_min: 允许区域最小边界 [D,] 或标量
        X_max: 允许区域最大边界 [D,] 或标量
        W: 归一化常数（建议取区域最大跨度）

    返回：
        p_o: 系统级越界惩罚值
        d_m: 各无人机越界距离 [N,]
    """
    # 转换为numpy数组
    q = np.array(current_positions)
    X_min = np.array(X_min)
    X_max = np.array(X_max)

    # 计算裁剪后的位置
    clipped_pos = np.clip(q, X_min, X_max)
    print(f"裁剪后位置: {clipped_pos}")

    # 计算各无人机越界距离（欧氏距离）
    d_m = np.linalg.norm(q - clipped_pos, axis=-1)  # [N,]
    print(f"各无人机越界距离: {d_m}")
    # 计算单机惩罚项
    penalty_m = 1 + d_m / W

    # 系统级平均惩罚
    p_o = np.mean(penalty_m)
    p_o = -float(p_o)  # 转换为标量并取负值

    return p_o, d_m


# 场景参数设置
X_min = [0, 0]  # 二维区域最小边界
X_max = [1000, 500]  # 二维区域最大边界
W = np.linalg.norm(np.array(X_max) - np.array(X_min))  # 区域对角线长度 ≈1118

# 假设有3架无人机的当前位置
uav_positions = [
    [1200, 400],  # x方向越界200
]

# 计算惩罚
p_o, d_m = calculate_boundary_penalty(uav_positions, X_min, X_max, W)

print(f"各无人机越界距离: {d_m}")  # 输出: [  0. 200. 100.]
print(f"系统惩罚值: {p_o:.3f}")  # 输出: 1.089
print(f"数据类型： {type(p_o)}")  # 输出: <class 'numpy.float64'>
