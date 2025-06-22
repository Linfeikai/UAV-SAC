import wandb

sweep_configuration = {
    'name': 'Hybrid_SAC_Sweep',
    'method': "bayes",
    "metric": {
            "name": "rollout/ep_rew_mean",  # SB3自动记录的平均回合奖励
            "goal": "maximize",
        },
    'parameters': {
        "learning_rate": {"min": 1e-5, "max": 1e-3, "distribution": "log_uniform"},
        "batch_size": {"values": [256, 512, 1024]},
        "gamma": {"values": [0.95, 0.99, 0.995]},
        "ent_coef": {"min": 0.01, "max": 0.2},
        "tau": {"min": 0.001, "max": 0.1},
        "net_arch": {
                "values": [
                    [64, 64],  # 简单双隐藏层
                    [128, 128],
                    [256, 256],
                    {"pi": [64], "qf": [128]},  # 异构结构
                    {"pi": [128, 128], "qf": [256, 256]},
                ]
            },
    },
     "early_terminate": {"type": "hyperband", "min_iter": 10, "eta": 3},

}
sweep_id = wandb.sweep(sweep=sweep_configuration, project="UAV-SAC-Optimization",entity="SACtest")  # 创建Sweep
wandb.agent(sweep_id, function=TD3_test, entity="SACtest", count=30)  # 运行30次实验
