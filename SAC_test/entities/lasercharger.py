from dataclasses import dataclass, field
from .uav import UAVNode
import numpy as np


@dataclass
class LaserCharger:
    loc: list = field(default_factory=lambda: [250, 250])  # 初始坐标 [0, 0]
    # 从表1初始化参数
    A = 1e-2  # 激光接收器望远镜或收集透镜的面积 (m^2)
    D = 0.05  # 初始激光束大小 (m)
    theta = 0.2  # 综合发射接收光效率
    alpha = 1e-6  # 信道介质衰减系数 (/m)
    beta = 3.4e-5  # 角扩展 The angular spread
    eta_u = 0.8  # 能量转换效率
    P_0 = 1  # 60dBm转换为瓦特为1w
    effective_range = 50  # 有效充电半径

    def charge(self, uav: UAVNode, charge_time):
        # 执行充电操作
        if self._is_uav_in_range(uav):
            # 计算信息增益
            # 1.计算距离
            dx = self.loc[0] - uav.loc[0]
            dy = self.loc[1] - uav.loc[1]
            dh = uav.flying_height
            distance_uav_ap = np.sqrt(dx * dx + dy * dy + dh * dh)
            numerator = self.A * self.theta * np.exp(-self.alpha * distance_uav_ap)
            denominator = (self.D + self.beta * distance_uav_ap) ** 2  # 分母
            g_ap = numerator / denominator
            E_harvest = self.eta_u * g_ap * self.P_0 * charge_time * charge_time
            uav.receive_energy(E_harvest)  # 更新无人机电量
            return E_harvest
        else:
            return 0  # 超出有效范围，无法充电

    def _is_uav_in_range(self, uav: UAVNode):
        dx = self.loc[0] - uav.loc[0]
        dy = self.loc[1] - uav.loc[1]
        dh = uav.flying_height
        distance_uav_ap = np.sqrt(dx * dx + dy * dy + dh * dh)
        return distance_uav_ap <= self.effective_range
