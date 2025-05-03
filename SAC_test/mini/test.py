import numpy as np

# 假设状态和动作的定义
class State:
    def __init__(self, x, y, battery, services_completed, time_step):
        self.x = x
        self.y = y
        self.battery = battery
        self.services_completed = services_completed
        self.time_step = time_step

class Action:
    def __init__(self, unload_ratio, ue_index, flight_distance, flight_direction):
        self.unload_ratio = unload_ratio
        self.ue_index = ue_index
        self.flight_distance = flight_distance
        self.flight_direction = flight_direction

# 状态转移函数（简化示例）
def update_state(state, action):
    # 位置更新
    if action.flight_direction == 'north':
        state.x += action.flight_distance
    elif action.flight_direction == 'south':
        state.x -= action.flight_distance
    elif action.flight_direction == 'east':
        state.y += action.flight_distance
    elif action.flight_direction == 'west':
        state.y -= action.flight_distance
    
    # 电量更新
    state.battery -= action.flight_distance * 0.1  # 假设每单位距离消耗0.1电量
    
    # 服务完成更新
    if state.x == action.ue_index:  # 假设到达UE位置
        state.services_completed += action.unload_ratio
    
    return state

# 计算相似性因子
def compute_similarity_factor(state, actions):
    similarity_matrix = np.zeros((len(actions), len(actions)))
    for i in range(len(actions)):
        for j in range(len(actions)):
            # 计算动作之间的相似性因子（简化示例）
            if actions[i].flight_direction == actions[j].flight_direction:
                similarity_matrix[i, j] = 0  # 相似
            else:
                similarity_matrix[i, j] = 1  # 不相似
    
    return similarity_matrix

# 动作聚类
def cluster_actions(similarity_matrix, epsilon=0.5):
    clusters = {}
    for i in range(len(actions)):
        cluster_id = None
        for j in range(len(actions)):
            if similarity_matrix[i, j] < epsilon:
                if cluster_id is None:
                    cluster_id = j
                else:
                    # 合并聚类
                    clusters[cluster_id].extend(clusters.pop(j, []))
                    break
        if cluster_id is not None and i not in clusters[cluster_id]:
            clusters[cluster_id].append(i)
    
    clustered_actions = []
    for cluster in clusters.values():
        representative = cluster[0]
        clustered_actions.append(actions[representative])
    
    return clustered_actions

# 示例使用
state = State(x=0, y=0, battery=100, services_completed=0, time_step=0)
actions = [
    Action(unload_ratio=0.5, ue_index=5, flight_distance=10, flight_direction='north'),
    Action(unload_ratio=0.5, ue_index=5, flight_distance=10, flight_direction='south'),
    Action(unload_ratio=0.6, ue_index=6, flight_distance=10, flight_direction='east'),
    Action(unload_ratio=0.6, ue_index=6, flight_distance=10, flight_direction='west')
]

# 计算相似性因子
similarity_matrix = compute_similarity_factor(state, actions)

# 动作聚类
clustered_actions = cluster_actions(similarity_matrix)

print("Original Actions:", [(action.flight_direction, action.unload_ratio) for action in actions])
print("Clustered Actions:", [(action.flight_direction, action.unload_ratio) for action in clustered_actions])