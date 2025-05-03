import numpy as np


class StateNormalizer:
    def __init__(self, num_features, kappa=1e-8):
        self.num_features = num_features
        self.kappa = kappa
        self.mean = np.zeros(num_features)
        self.std = np.zeros(num_features)
        self.t = 0

    def normalize(self, state):
        if self.t == 0:
            self.mean = state
            self.std = np.zeros_like(state)
        else:
            self.mean = self.mean + (state - self.mean) / (self.t + 1)
            self.std = np.sqrt(
                (self.t * self.std**2 + (state - self.mean) * (state - self.mean))
                / (self.t + 1)
            )
        self.t += 1
        normalized_state = (state - self.mean) / (self.std + self.kappa)
        return normalized_state


if __name__ == "__main__":
    num_features = 4  # 假设状态有4个特征
    normalizer = StateNormalizer(num_features)

    # 模拟一些状态数据
    states = np.array(
        [[1.0, 2.0, 3.0, 4.0], [2.0, 3.0, 4.0, 5.0], [3.0, 4.0, 5.0, 6.0]]
    )

    # 对每个状态进行归一化
    for state in states:
        normalized_state = normalizer.normalize(state)
        print("Normalized State:", normalized_state)
