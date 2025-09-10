import numpy as np
import math
from collections import defaultdict
import json  # 用于打印美观的字典
import tqdm

# 确保可以导入你的环境和配置
from SAC_test.entities.custom_env import CustomEnv
from SAC_test.entities.all_config import (
    UE_NUM,
    TASK_TYPE_DISTRIBUTION,
    DATA_SIZE_RANGES,
    WORKLOAD_INTENSITY,
    CPU_RANGES,
)

"""
======================================================================
 Part 1: 理论静态分析模块
======================================================================
 根据 all_config.py 计算环境的理论压力，作为动态测试前的基准。
"""


def perform_theoretical_analysis():
    """
    (V2 - 更新版)
    通过读取配置文件，对环境进行静态的理论压力估算。
    新增了对每种UE类型的独立负荷分析。
    """
    print("\n" + "=" * 60)
    print("         Part 1: 理论静态分析 (Theoretical Static Analysis)")
    print("=" * 60)

    # --- Step 1: 初始化变量 ---
    total_workload_rate_per_sec = 0
    total_local_capacity = 0
    per_type_analysis = {}  # 用于存储每种类型UE的分析结果
    avg_arrival_per_ue_per_step = 2.0  # 平均每个UE在一个step内产生2个任务

    # --- Step 2: 遍历所有UE类型，分别计算并累加需求和供给 ---
    for ue_type, dist in TASK_TYPE_DISTRIBUTION.items():
        type_name = ue_type.name  # 获取枚举的名称，如 'NORMAL'

        # 计算该类型UE的数量
        num_ues_of_type = UE_NUM * dist
        if num_ues_of_type == 0:
            continue

        # 计算需求 (Demand)
        avg_size = np.mean(DATA_SIZE_RANGES[ue_type])
        intensity = WORKLOAD_INTENSITY[ue_type]
        workload_per_step = (
            num_ues_of_type * avg_arrival_per_ue_per_step * avg_size * intensity
        )
        demand_per_sec = workload_per_step / 8.0  # 8秒一个step
        total_workload_rate_per_sec += demand_per_sec

        # 计算供给 (Supply)
        avg_cpu = np.mean(CPU_RANGES[ue_type])
        supply_per_sec = num_ues_of_type * avg_cpu
        total_local_capacity += supply_per_sec

        # 计算该类型的负荷系数 rho
        rho_type = (
            demand_per_sec / supply_per_sec if supply_per_sec > 0 else float("inf")
        )

        # 存储该类型的结果
        per_type_analysis[type_name] = {
            "demand": demand_per_sec,
            "supply": supply_per_sec,
            "rho": rho_type,
        }

    # --- Step 3: 计算最终的系统级指标 ---
    rho_system_local = (
        total_workload_rate_per_sec / total_local_capacity
        if total_local_capacity > 0
        else float("inf")
    )

    # --- Step 4: 生成并打印报告 ---
    print("--- 系统总体分析 (System-wide Analysis) ---")
    print(f"  - 理论任务生成速率: {total_workload_rate_per_sec / 1e9:.3f} G-cycles/sec")
    print(f"  - 理论UE本地总算力: {total_local_capacity / 1e9:.3f} G-cycles/sec")
    print(f"  - 理论系统本地负荷 (ρ_system_local): {rho_system_local:.3f}")
    if rho_system_local < 1:
        print(
            "    -> 诊断: 系统总体负荷 < 1，表明UE本地总算力有盈余。压力的来源在于分布不均。"
        )
    else:
        print("    -> 诊断: 系统总体负荷 > 1，表明UE本地总算力不足，UAV是绝对必要的。")

    print("\n--- 各类型UE负荷细分 (Per-Type Load Breakdown) ---")
    for type_name, metrics in per_type_analysis.items():
        rho_value = metrics["rho"]
        diagnosis = ""
        if rho_value < 0.5:
            diagnosis = "-> 诊断: 非常清闲，几乎无压力。"
        elif rho_value < 0.9:
            diagnosis = "-> 诊断: 压力可控，但有一定负荷。"
        elif rho_value <= 1.1:
            diagnosis = "-> 诊断: 饱和区！系统瓶颈，对UAV服务高度敏感。"
        else:
            diagnosis = "-> 诊断: 已过载！任务会持续积压，是UAV救援的最高优先级目标。"

        print(f"  - {type_name:<8} 负荷系数 (ρ_local): {rho_value:.3f}")
        print(f"    {diagnosis}")

    print("=" * 60)


"""
======================================================================
 Part 2: 智能基线探测策略
======================================================================
 采用更高级的、有生存意识和动态决策的贪心策略。
"""


def greedy_probe_policy(obs, env: CustomEnv):
    battery_ratio = float(obs[0])
    uav_xy = np.array(
        [obs[1] * env.ground_width, obs[2] * env.ground_height], dtype=np.float32
    )

    # 规则：电量低于30%就去充电；否则选缓存占用最高的UE
    if battery_ratio < 0.3:
        target_loc = env.charger.loc
        off_ratio = 0.2  # 低电量时减少计算，多充电
        ue_id_for_action = int(
            np.argmin(np.linalg.norm(env.ue_locs - target_loc, axis=1))
        )
    else:
        # 解析UE状态：obs[6:] 包含了所有UE的信息
        # 每个UE占4个维度: dx, dy, cache_ratio, cap_ratio
        ue_states = obs[6 : 6 + env.ue_num * 4].reshape(env.ue_num, 4)
        cache_ratios = ue_states[:, 2]
        target_ue_id = int(np.argmax(cache_ratios))
        target_loc = env.ue_locs[target_ue_id]
        off_ratio = 0.7 if battery_ratio > 0.5 else 0.5
        ue_id_for_action = target_ue_id

    # 计算飞行角度和速度
    vec = target_loc - uav_xy
    angle = float(np.arctan2(vec[1], vec[0]))
    desired_v = 12.0
    vr = np.clip(
        (desired_v - env.uav.flying_speed) / (env.uav.max_acceleration * env.t_fly),
        -1.0,
        1.0,
    )

    return (ue_id_for_action, np.array([angle, vr, off_ratio], dtype=np.float32))


"""
======================================================================
 Part 3: 环境审计器核心类
======================================================================
"""


class EnvironmentAuditor:
    def __init__(self, env: CustomEnv):
        self.env = env
        self.ep_metrics = []  # 存储每个回合的最终指标

    def run_simulation(self, n_episodes: int, seed: int):
        print("\n" + "=" * 60)
        print(f"       Part 2: 动态探测分析 (Running for {n_episodes} Episodes)")
        print("=" * 60)
        rng = np.random.default_rng(seed)

        for ep in range(n_episodes):
            obs, _ = self.env.reset(seed=int(rng.integers(0, 1e9)))
            step_logs = []

            for _ in range(self.env.slot_num):
                action = greedy_probe_policy(obs, self.env)
                obs, reward, terminated, truncated, info = self.env.step(action)
                step_logs.append(info)
                if terminated or truncated:
                    break

            self._analyze_episode(step_logs, self.env)
            print(f"  - Episode {ep + 1}/{n_episodes} completed.")

    def _analyze_episode(self, logs: list, env_instance: CustomEnv):
        # [MODIFIED] 直接从环境实例中读取最终的累积计数值
        A = max(env_instance.ep_arrived, 1)
        completion_rate = env_instance.ep_completed / A
        drop_rate = env_instance.ep_dropped / A
        final_backlog = sum(len(n.task_queue) for n in env_instance.nodeList)
        backlog_ratio = final_backlog / A

        consume = np.array(
            [d.get("flying_energy", 0.0) + d.get("consumed_energy", 0.0) for d in logs]
        )
        harvest = np.array([d.get("harvested_energy", 0.0) for d in logs])
        alpha = np.array(
            [
                d.get("t_tr", 0.0) / self.env.t_com
                for d in logs
                if d.get("t_tr", 0.0) > 0
            ]
        )

        # 统计丢弃和完成的任务数
        # 注意: 需确保你的info字典能提供这些值
        # dropped = np.sum([d.get("num_tasks_dropped", 0) for d in logs])
        # processed = np.sum([d.get("num_tasks_processed", 0) for d in logs])

        # drop_rate = float(dropped) / max(dropped + processed, 1)
        eta_E = np.sum(harvest) / max(np.sum(consume), 1e-9)  # 使用总和比率，更稳定
        alive = env_instance.current_step == env_instance.slot_num

        self.ep_metrics.append(
            {
                "eta_E": float(eta_E),
                "alpha_med": float(np.median(alpha) if alpha.size > 0 else 0.0),
                "drop_rate": float(drop_rate),
                "survive": 1.0 if alive else 0.0,
                "completion_rate": float(completion_rate),
                "backlog_ratio": float(backlog_ratio),
                "fairness": float(env_instance.previous_fairness_index),
            }
        )

    def _calculate_difficulty_score(self, m: dict) -> float:
        # ... 此函数与上一版完全相同，无需修改 ...
        s = 0.0
        s += 40.0 * (1.0 - m["survive"])
        s += 25.0 * min(abs(np.log(max(m["eta_E"], 1e-6))), 1.5)
        s += 15.0 * max(0.0, abs(m["alpha_med"] - 0.4) / 0.4)
        s += 20.0 * m["drop_rate"]
        s += 10.0 * m["backlog_ratio"]
        s += 5.0 * max(0.0, 1.0 - m["fairness"])
        return s

    def generate_report(self):
        scores = [self._calculate_difficulty_score(m) for m in self.ep_metrics]

        report = {
            "episodes": len(self.ep_metrics),
            "difficulty_score_mean": float(np.mean(scores)),
            "difficulty_score_std": float(np.std(scores)),
            "survival_rate": float(np.mean([m["survive"] for m in self.ep_metrics])),
            "completion_rate_median": float(
                np.median([m["completion_rate"] for m in self.ep_metrics])
            ),
            "drop_rate_median": float(
                np.median([m["drop_rate"] for m in self.ep_metrics])
            ),
            "backlog_ratio_median": float(
                np.median([m["backlog_ratio"] for m in self.ep_metrics])
            ),
            "eta_E_median": float(np.median([m["eta_E"] for m in self.ep_metrics])),
            "alpha_median": float(np.median([m["alpha_med"] for m in self.ep_metrics])),
            "fairness_median": float(
                np.median([m["fairness"] for m in self.ep_metrics])
            ),
        }

        print("\n" + "=" * 60)
        print("      Part 3: 综合审计报告 (Comprehensive Audit Report)")
        print("=" * 60)
        print(json.dumps(report, indent=4))

        print("\n--- 诊断与建议 ---")
        score = report["difficulty_score_mean"]
        print(
            f"综合难度评分: {score:.2f} (标准差: {report['difficulty_score_std']:.2f})"
        )
        if score < 20:
            print(" -> 结论: 环境可能过于简单。智能体可能缺乏挑战，学习信号弱。")
        elif score > 60:
            print(
                " -> 结论: 环境可能过于困难。智能体可能难以生存或完成任务，导致无法有效学习。"
            )
        else:
            print(" -> 结论: 环境处在一个良好、具有挑战性的难度区间。")

        print("\n细项分析:")
        if report["survival_rate"] < 0.9:
            print(
                f"  - 生存率较低 ({report['survival_rate']:.0%})，请检查能量模型 (η_E={report['eta_E_median']:.2f})。"
            )
        if not (0.2 < report["alpha_median"] < 0.6):
            print(
                f"  - 通信瓶颈系数 α ({report['alpha_median']:.2f}) 不在理想区间(0.2-0.6)，检查带宽或UE分布。"
            )
        if report["drop_rate_median"] > 0.1:
            print(
                f"  - 任务丢弃率较高 ({report['drop_rate_median']:.0%})，主要瓶颈可能在于UE缓存大小不足。"
            )
        if report["backlog_ratio_median"] > 0.1:
            print(
                f"  - 期末积压率较高 ({report['backlog_ratio_median']:.0%})，主要瓶颈可能在于系统总算力不足。"
            )

        print("=" * 60)


def heuristic_agent_policy(env: CustomEnv, observation: np.ndarray) -> tuple:
    """
    一个简单的启发式策略（基于规则的智能体）.
    - 目标选择: 飞向缓存占用率最高的UE.
    - 飞行决策: 以中等速度（max_speed * 0.6）朝目标飞去.
    - 卸载决策: 根据自身电量决定卸载率. 电量高时多卸载，电量低时少卸载.
    """
    # --- 1. 目标选择：找到缓存最满的UE ---
    # 从观测中解析出缓存占用率
    # 观察空间结构: 6个UAV状态 + 20个UE * (2个位置 + 1个缓存率 + 1个算力率) = 6 + 20*4 = 86
    # 缓存率在每个UE块的第3个位置 (索引是2)
    cache_ratios = observation[6:86:4]
    target_ue_id = np.argmax(cache_ratios)

    # --- 2. 飞行决策：计算飞向目标UE的方向和速度 ---
    # 获取目标UE和UAV的归一化位置
    uav_loc = observation[1:3]  # [x, y]
    ue_locs_flat = observation[6:86].reshape((env.ue_num, 4))
    target_ue_loc = (
        ue_locs_flat[target_ue_id, 0:2] + uav_loc
    )  # 相对位置 + uav位置 = 绝对位置

    # 计算方向
    delta_loc = target_ue_loc - uav_loc
    angle = np.arctan2(delta_loc[1], delta_loc[0])

    # 设定一个恒定的速度比率 (例如，使用最大加速度的60%)
    velocity_ratio = 0.6

    # --- 3. 卸载决策：根据电量调整 ---
    battery_ratio = observation[0]
    # 电量 > 70%, 积极卸载; 电量 < 30%, 保守卸载 (多在本地计算)
    if battery_ratio > 0.7:
        offloading_ratio = 0.9
    elif battery_ratio < 0.3:
        offloading_ratio = 0.1
    else:
        offloading_ratio = 0.5

    # 组合成最终动作
    continuous_action = np.array(
        [angle, velocity_ratio, offloading_ratio], dtype=np.float32
    )

    return (target_ue_id, continuous_action)


def run_calibration(num_episodes=100, agent_type="heuristic"):
    """
    运行基准测试来校准归一化常数.

    Args:
        num_episodes (int): 要运行的回合总数.
        agent_type (str): 'random' 或 'heuristic'.
    """
    print(f"\n--- 开始校准，使用 '{agent_type}' 智能体，运行 {num_episodes} 个回合 ---")

    env = CustomEnv()

    # 用于存储每个step记录的关键指标
    all_delays = []
    all_energy_consumptions = []
    all_energy_gained = []
    all_flying_distances = []  # 用于 PBRS 归一化

    # tqdm提供一个可视化的循环进度条
    for episode in tqdm.tqdm(range(num_episodes)):
        obs, info = env.reset()
        terminated = False
        truncated = False

        while not terminated and not truncated:
            # 根据选择的智能体类型来决定动作
            if agent_type == "random":
                action = env.action_space.sample()
            elif agent_type == "heuristic":
                action = heuristic_agent_policy(env, obs)
            else:
                raise ValueError("未知的 agent_type. 请选择 'random' 或 'heuristic'.")

            # 与环境交互
            obs, reward, terminated, truncated, info = env.step(action)

            # --- 核心：记录当前step的原始数据 ---
            # info 字典包含了我们需要的所有原始值
            raw_delay = info.get("delay", 0.0)
            raw_flying_energy = info.get("flying_energy", 0.0)
            raw_computing_energy = info.get("consumed_energy", 0.0)
            raw_total_consumption = raw_flying_energy + raw_computing_energy
            raw_harvested_energy = info.get("harvested_energy", 0.0)
            raw_flying_distance = info.get(
                "flying_distance", 0.0
            )  # 这是PBRS奖励的原始值

            all_delays.append(raw_delay)
            all_energy_consumptions.append(raw_total_consumption)
            all_energy_gained.append(raw_harvested_energy)
            all_flying_distances.append(raw_flying_distance)

    print("\n--- 校准完成！正在计算统计数据... ---\n")

    # --- 计算统计数据并打印建议 ---
    def print_stats(name, data, percentile=95):
        if not data:
            print(f"指标 '{name}' 没有收集到数据。")
            return

        data_np = np.array(data)
        # 过滤掉0值，因为我们关心的是发生时的量级
        data_np_filtered = data_np[data_np > 1e-6]
        if len(data_np_filtered) == 0:
            print(f"指标 '{name}' 的所有值都接近于零。")
            return

        mean_val = np.mean(data_np_filtered)
        max_val = np.max(data_np_filtered)
        p_val = np.percentile(data_np_filtered, percentile)

        print("=" * 60)
        print(f"指标: {name}")
        print("=" * 60)
        print(f"  - 平均值 (非零): {mean_val:.4f}")
        print(f"  - 最大值:         {max_val:.4f}")
        print(f"  - {percentile}th 百分位数:  {p_val:.4f}")
        print(f"\n  >>> 建议的 NORM 值 (使用 {percentile}th 百分位数): {p_val:.1f}")
        print("\n")

    print_stats("NORM_DELAY", all_delays)
    print_stats("NORM_ENERGY_CONSUMED", all_energy_consumptions)
    print_stats("NORM_ENERGY_GAINED", all_energy_gained)
    print_stats("NORM_FLYING_DISTANCE", all_flying_distances)

    print("--- 请将上述建议值更新到你的 CustomEnv.Config 类中 ---")


"""
======================================================================
 Part 4: 主程序入口
======================================================================
"""
if __name__ == "__main__":
    # 1. 执行静态理论分析
    perform_theoretical_analysis()

    # 2. 初始化环境和审计器
    env = CustomEnv()
    auditor = EnvironmentAuditor(env)

    # 3. 运行动态探测
    # 更多回合数结果更稳定，但耗时更长。建议从20-50个开始。
    auditor.run_simulation(n_episodes=100, seed=42)

    # 4. 生成最终的综合报告
    auditor.generate_report()
    # run_calibration()
