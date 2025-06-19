import optuna
import gymnasium as gym
from stable_baselines3 import SAC
from stable_baselines3.common.evaluation import evaluate_policy
from optuna.visualization import plot_optimization_history, plot_param_importances

# 固定随机种子
SEED = 42


def objective(trial: optuna.Trial):
    params = {
        "learning_rate": trial.suggest_float("lr", 1e-5, 1e-3, log=True),
        "buffer_size": trial.suggest_categorical(
            "buffer_size", [50_000, 100_000, 1_000_000]
        ),
        "batch_size": trial.suggest_int("batch_size", 64, 512, step=64),
        "tau": trial.suggest_float("tau", 0.001, 0.1),
        "gamma": trial.suggest_float("gamma", 0.9, 0.9999),
        "ent_coef": trial.suggest_categorical("ent_coef", ["auto", 0.1, 0.2, 0.5]),
        "net_arch": trial.suggest_categorical("net_arch", ["small", "medium", "large"]),
    }

    net_arch_map = {
        "small": [64, 64],
        "medium": [256, 256],
        "large": [400, 300],
    }

    env = gym.make("UAVEnv-v0")  # 创建环境

    policy_kwargs = {"net_arch": net_arch_map[params.pop("net_arch")]}

    model = SAC(
        "MlpPolicy",
        env,
        verbose=0,
        seed=SEED,
        policy_kwargs=policy_kwargs,
        **params,
    )

    model.learn(total_timesteps=50_000)

    mean_reward, _ = evaluate_policy(model, env, n_eval_episodes=10, deterministic=True)

    del model
    env.close()  # 关闭环境

    return mean_reward


def find_best_hyperparameters_optuna():
    study = optuna.create_study(
        direction="maximize", sampler=optuna.samplers.TPESampler(seed=SEED)
    )
    study.enqueue_trial(
        {
            "lr": 3e-4,
            "buffer_size": 1_000_000,
            "batch_size": 256,
            "tau": 0.005,
            "gamma": 0.99,
            "ent_coef": "auto",
            "net_arch": "medium",
        }
    )
    study.optimize(objective, n_trials=10, timeout=600)  # 10次试验，超时600秒
    print(f"Best trial: {study.best_params}")
    print(f"Best value: {study.best_value}")
    # 可视化
    fig1 = plot_optimization_history(study)
    fig2 = plot_param_importances(study)
    fig1.show()
    fig2.show()
