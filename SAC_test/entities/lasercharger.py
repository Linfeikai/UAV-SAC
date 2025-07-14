from dataclasses import dataclass, field
import numpy as np
from typing import TYPE_CHECKING

# 为了让类型检查器（如 mypy）能够识别 UAVNode 类型，
# 但在直接运行时不执行导入，从而避免 ImportError。
if TYPE_CHECKING:
    from .uav import UAVNode


import numpy as np
import math
from dataclasses import field, dataclass
import logging

# Forward declaration for type hinting
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .uav import UAVNode


@dataclass
class LaserCharger:
    """
    Represents a dedicated laser charger for the UAV.

    The power decay model is based on the paper 'Green Laser-Powered UAV...',
    but its output is scaled by 'k_charge_scale' to create a meaningful energy
    economy for this specific simulation environment.
    """

    # 将充电桩位置设在400x400地图的中心
    loc: np.ndarray = field(
        default_factory=lambda: np.array([200, 200], dtype=np.float32)
    )
    # --- 核心物理参数 (源自参考论文) ---
    # These parameters define the SHAPE of the power decay curve.
    Pb: float = 1000.0
    a1: float = 0.445
    b1: float = -0.75
    a2: float = 0.5414
    b2: float = -0.2313
    alpha_per_m: float = 0.000237

    # --- 仿真环境校准参数 (我们新增和修改的部分) ---

    # 【关键修改点 1】: 引入充电功率缩放系数
    # 这个系数将论文公式的输出值放大，使其在我们的仿真中具有实际意义。
    k_charge_scale: float = 1.5

    # 【关键修改点 2】: 简化充电范围为一个最大半径
    # 对于专用充电桩，我们只关心其最大有效作用范围。
    effective_charge_radius: float = 250.0  # 最大有效充电距离 (米)

    def _calculate_base_power(self, distance: float) -> float:
        """
        根据论文的原始公式计算【基础】接收功率(Pu)，单位为瓦特(W)。
        这个函数完全保留了你之前的物理模型，我们不改动它。
        """
        # 1. 计算发射端效率 (eta_el)
        eta_el = self.a1 + self.b1 / self.Pb

        # 2. 计算路径损耗 (eta_lr)
        eta_lr = np.exp(-self.alpha_per_m * distance)

        # 3. 求解关于 Pu 的二次方程: A*Pu^2 + B*Pu + C = 0
        A = 1
        B = -(eta_el * eta_lr * self.a2 * self.Pb)
        C = -(eta_el * eta_lr * self.b2 * self.Pb)

        discriminant = B**2 - 4 * A * C
        if discriminant < 0:
            return 0.0

        pu = (-B + np.sqrt(discriminant)) / (2 * A)
        return pu

    def charge(self, uav: "UAVNode", charge_time: float) -> float:
        """
        为无人机充电，返回实际获得的能量。
        能量 = (基础功率 * 缩放系数) × 时间。
        """
        # 1. 计算无人机与充电桩的直线距离
        dx = self.loc[0] - uav.loc[0]
        dy = self.loc[1] - uav.loc[1]
        dh = uav.flying_height
        distance_uav_ap = np.sqrt(dx**2 + dy**2 + dh**2)

        # 2. 检查无人机是否在有效的圆形充电区域内
        if not self._is_uav_in_range(distance_uav_ap):
            return 0.0  # 超出有效范围，无法充电

        # 3. 使用原始公式计算基础功率
        base_power = self._calculate_base_power(distance_uav_ap)

        # 【关键修改点 3】: 将基础功率进行等比例缩放
        scaled_power = base_power * self.k_charge_scale

        if scaled_power > 0:
            # 4. 基于【缩放后】的功率计算获得的能量 (单位: 焦耳)
            E_harvest = scaled_power * charge_time

            # 5. 更新无人机电量
            uav.receive_energy(E_harvest)
            return E_harvest
        else:
            return 0.0

    def _is_uav_in_range(self, distance: float) -> bool:
        """
        【关键修改点 4】: 检查无人机是否在最大有效充电半径内（圆形区域）。
        """
        # 不再有最小距离限制，只检查是否小于最大距离
        return distance <= self.effective_charge_radius


if __name__ == "__main__":
    # 为了能独立运行此文件进行测试，我们需要处理相对导入的问题。
    # 我们将定义一个简单的 UAVNode 模拟类，以防导入失败。
    try:
        # 尝试从相对路径导入，这在作为模块导入时会成功
        from .uav import UAVNode
    except ImportError:
        # 如果直接运行此文件，上述导入会失败，此时我们使用下面的模拟类
        print("未能从 .uav 导入 UAVNode。为测试目的，将使用一个模拟（Mock）UAVNode类。")
        from dataclasses import dataclass, field

        @dataclass
        class UAVNode:
            loc: list = field(default_factory=lambda: [200, 200])
            battery_capacity: float = 100000.0
            e_battery: float = 50000.0  # 假设初始电量为一半
            flying_height: float = 60.0

            def receive_energy(self, energy):
                """接收能量并更新电池状态"""
                actual_energy = min(energy, self.battery_capacity - self.e_battery)
                self.e_battery += actual_energy
                return actual_energy

    print("--- 正在测试 LaserCharger 功能 (新版模型) ---")
    charger = LaserCharger()
    print(f"充电桩位置: {charger.loc}")
    print(f"最大有效充电距离 (3D): {charger.effective_charge_radius}m")
    print(f"充电功率缩放系数: {charger.k_charge_scale}")
    print("-" * 50)

    # 定义一系列与充电桩的水平距离进行测试
    # 注意：实际判断的是3D距离，所以我们会基于水平距离和飞行高度计算3D距离
    test_horizontal_distances = [0, 40, 80, 150, 240, 260]
    charge_duration = 1.0  # 假设充电持续时间为 1 秒

    for h_dist in test_horizontal_distances:
        # 将无人机放置在相对于充电桩 [200, 200] 的位置
        uav = UAVNode()
        uav.loc = [200 + h_dist, 200]
        uav.e_battery = 50000.0

        # 为打印信息，计算3D距离
        dx = charger.loc[0] - uav.loc[0]
        dy = charger.loc[1] - uav.loc[1]
        distance_3d = np.sqrt(dx**2 + dy**2 + uav.flying_height**2)

        print(f"测试场景: 无人机水平距离 = {abs(dx):.0f}m")
        print(f"无人机固定飞行高度 = {uav.flying_height}m")
        print(f"计算出的3D直线距离 = {distance_3d:.2f}m")
        print(f"无人机初始电量: {uav.e_battery:.2f} J")

        harvested_energy = charger.charge(uav, charge_duration)

        if harvested_energy > 0:
            print(f"结果: 在 {charge_duration}s 内成功充电!")
            print(f"  -> 收集到的能量: {harvested_energy:.4f} J")
            print(f"  -> 无人机最终电量: {uav.e_battery:.2f} J")
        else:
            print("结果: 未充电 (原因: 无人机与充电桩的3D距离超出了最大有效充电半径)")
        print("-" * 50)

    print("\n--- 额外测试: 计算不同3D距离下的接收功率 ---")
    power_test_distances = [50, 100, 150, 200, 250, 260]
    for dist in power_test_distances:
        base_power = charger._calculate_base_power(dist)
        scaled_power = base_power * charger.k_charge_scale
        if charger._is_uav_in_range(dist):
            print(
                f"  - 3D距离 {dist}m: 基础功率 = {base_power:.4f} W, 缩放后功率 = {scaled_power:.4f} W"
            )
        else:
            print(f"  - 3D距离 {dist}m: 超出有效范围，功率为 0 W")
