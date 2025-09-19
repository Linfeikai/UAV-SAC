#!/bin/bash

# set -e: 脚本中任何命令执行失败（返回非零退出码），则立即退出脚本。
# 这对于自动化脚本非常重要，可以防止错误累积。
set -e

# --- 实验配置 (Hydra 版本) ---

# 1. 定义要遍历的参数数组
SEEDS=(42 43 44)
UE_NUMS=(10 15 20 25)

# 【修改1】这里的名字必须与 conf/experiment/ 目录下的 *文件名* 对应（不带.yaml）
#           建议使用小写和下划线，以符合通常的命名习惯。
EXPERIMENT_NAMES=("diffusion_sac" "diffusion_sac_twostage")

# 【修改2】下面这个变量不再需要，Hydra会自动在 conf/ 目录中查找
# CONFIG_FILE="experiments.yaml" 
ERROR_LOG_FILE="experiments_error.log" # 定义错误日志文件名 (这个可以保留)

# --- 实验循环 ---

echo "Starting batch of experiments with Hydra..."
echo "Errors will be logged to: $ERROR_LOG_FILE"

# 遍历所有实验 (变量名从 agent 改为 experiment 更贴切)
for experiment in "${EXPERIMENT_NAMES[@]}"; do
  # 遍历所有UE数量
  for ue in "${UE_NUMS[@]}"; do
    # 遍历所有Seed
    for seed in "${SEEDS[@]}"; do
      
      # 打印当前正在运行的实验信息，方便追踪进度
      echo ""
      echo "========================================================================"
      echo "Running Experiment: $experiment, UE_Num: $ue, Seed: $seed"
      echo "========================================================================"
      
      # --- 【修改3】【核心修改】 ---
      # 执行Python脚本，使用 Hydra 的覆盖语法
      # 旧的 --config, --name, --ue, --seed 参数
      # 全部换成 key=value 的格式
      python main.py \
        experiment="$experiment" \
        ue="$ue" \
        seed="$seed" 2>> "$ERROR_LOG_FILE"
      
      echo "--- Experiment Finished ---"
      
    done
  done
done

echo ""
echo "All experiments completed successfully!"
echo "Check '$ERROR_LOG_FILE' for any potential errors during the run."