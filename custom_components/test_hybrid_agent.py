import pytest
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import torch as th

from hybrid_sac_agent import HybridSAC
from hybrid_sac_policy import HybridSACPolicy
from hybrid_replay_buffer import HybridReplayBuffer


class SimpleHybridEnv(gym.Env):
    """A simple environment with hybrid action space for testing."""

    def __init__(self):
        # Define action space: Tuple(Discrete(3), Box(-1, 1, (2,)))
        self.action_space = spaces.Tuple(
            (
                spaces.Discrete(3),  # 3 discrete actions
                spaces.Box(
                    low=-1, high=1, shape=(2,), dtype=np.float32
                ),  # 2D continuous actions
            )
        )
        # Define observation space: Box(-inf, inf, (4,))
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(4,), dtype=np.float32
        )
        self.state = None
        self.steps = 0
        self.max_steps = 100

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.state = np.random.randn(4).astype(np.float32)
        self.steps = 0
        return self.state, {}

    def step(self, action):
        discrete_action, continuous_action = action

        # Simple reward function
        reward = (
            -0.1 * np.sum(np.square(self.state))  # Penalize large state values
            + 0.1 * discrete_action  # Reward for higher discrete actions
            - 0.1
            * np.sum(np.square(continuous_action))  # Penalize large continuous actions
        )

        # Update state
        self.state = self.state + 0.1 * np.random.randn(4).astype(np.float32)

        # Update steps
        self.steps += 1
        done = self.steps >= self.max_steps

        return self.state, reward, done, False, {}


@pytest.fixture
def env():
    """Create a test environment."""
    return SimpleHybridEnv()


@pytest.fixture
def model(env):
    """Create a HybridSAC model."""
    return HybridSAC(
        policy="HybridSACPolicy",
        env=env,
        learning_rate=3e-4,
        buffer_size=10000,
        learning_starts=100,
        batch_size=64,
        tau=0.005,
        gamma=0.99,
        train_freq=1,
        gradient_steps=1,
        ent_coef="auto",
        target_update_interval=1,
        target_entropy="auto",
        verbose=1,
        seed=42,
    )


def test_model_initialization(model, env):
    """Test if the model initializes correctly."""
    assert isinstance(model, HybridSAC)
    assert isinstance(model.policy, HybridSACPolicy)
    assert isinstance(model.replay_buffer, HybridReplayBuffer)
    assert model.action_space == env.action_space
    assert model.observation_space == env.observation_space


def test_predict(model, env):
    """Test if the model can predict actions."""
    obs, _ = env.reset()
    action, _ = model.predict(obs, deterministic=True)

    # Check action structure
    assert isinstance(action, tuple)
    assert len(action) == 2
    assert isinstance(action[0], np.ndarray)  # discrete action
    assert isinstance(action[1], np.ndarray)  # continuous action

    # Check action shapes
    assert action[0].shape == (1,)  # batch size of 1
    assert action[1].shape == (1, 2)  # batch size of 1, 2D continuous action


def test_learn(model, env):
    """Test if the model can learn."""
    # Train for a small number of steps
    model.learn(total_timesteps=1000, log_interval=10)

    # Check if the model has been updated
    assert model.num_timesteps > 0
    assert model._n_updates > 0


def test_save_load(model, tmp_path):
    """Test if the model can be saved and loaded."""
    # Save the model
    save_path = tmp_path / "hybrid_sac_test"
    model.save(save_path)

    # Load the model
    loaded_model = HybridSAC.load(save_path)

    # Check if the loaded model has the same attributes
    assert isinstance(loaded_model, HybridSAC)
    assert loaded_model.action_space == model.action_space
    assert loaded_model.observation_space == model.observation_space


def test_collect_rollouts(model, env):
    """Test if the model can collect rollouts."""
    # Create a vectorized environment
    from stable_baselines3.common.vec_env import DummyVecEnv

    vec_env = DummyVecEnv([lambda: env])

    # Collect some rollouts
    from stable_baselines3.common.callbacks import BaseCallback

    class DummyCallback(BaseCallback):
        def __init__(self, verbose=0):
            super().__init__(verbose)
            self.n_calls = 0

        def _on_step(self):
            self.n_calls += 1
            return True

    callback = DummyCallback()

    # Test collect_rollouts
    from stable_baselines3.common.type_aliases import TrainFreq

    train_freq = TrainFreq(frequency=1, unit="step")

    rollout_return = model.collect_rollouts(
        env=vec_env,
        callback=callback,
        train_freq=train_freq,
        replay_buffer=model.replay_buffer,
        learning_starts=0,
    )

    assert rollout_return.num_collected_steps > 0
    assert callback.n_calls > 0


def test_train_step(model):
    """Test if the model can perform a training step."""
    # Add some transitions to the replay buffer
    obs = np.random.randn(4).astype(np.float32)
    next_obs = np.random.randn(4).astype(np.float32)
    discrete_action = np.array([1])
    continuous_action = np.array([[0.5, -0.5]])
    reward = 1.0
    done = False

    model.replay_buffer.add(
        obs, next_obs, (discrete_action, continuous_action), reward, done, {}
    )

    # Perform a training step
    model.train(gradient_steps=1, batch_size=1)

    # Check if the model has been updated
    assert model._n_updates > 0


if __name__ == "__main__":
    pytest.main([__file__])
