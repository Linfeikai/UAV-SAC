#!/bin/bash

# set -e: 脚本中任何命令执行失败（返回非零退出码），则立即退出脚本。
set -e

# --- 实验配置 ---
SEEDS=(42 43 44)
UE_NUMS=(10 15 20 25)
EXPERIMENT_NAME="diffusion_sac_twostage"
ERROR_LOG_FILE="diffusion_sac_twostage_error.log" # 为这个脚本设置独立的日志文件

# --- 实验循环 ---

echo "Starting batch of experiments for: $EXPERIMENT_NAME"
echo "Errors will be logged to: $ERROR_LOG_FILE"

# 遍历所有UE数量
for ue in "${UE_NUMS[@]}"; do
  # 遍历所有Seed
  for seed in "${SEEDS[@]}"; do
    
    # 打印当前正在运行的实验信息，方便追踪进度
    echo ""
    echo "========================================================================"
    echo "Running Experiment: $EXPERIMENT_NAME, UE_Num: $ue, Seed: $seed"
    echo "========================================================================"
    
    # 执行Python脚本
    python main.py \
      experiment="$EXPERIMENT_NAME" \
      ue="$ue" \
      seed="$seed" 2>> "$ERROR_LOG_FILE"
    
    echo "--- Experiment Finished ---"
    
  done
done

echo ""
echo "All experiments for $EXPERIMENT_NAME completed successfully!"
echo "Check '$ERROR_LOG_FILE' for any potential errors during the run."