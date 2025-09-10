python main.py --config experiments.yaml --name Diffusion-SAC-TwoStage
echo "--> 正在清理环境，请稍候..."
rm SAC*.xlsx
rm diffusion-two-stage*.xlsx
rm *.png
echo "--> 清理完成！"
echo "" # 打印一个空行，让输出更好看
echo "--> 开始运行评估程序..."

python main.py --evaluate "sac"
echo "" # 打印一个空行，让输出更好看
echo "--> 开始运行sac的表格合并..."
python process_rewards.py -p SAC
python main.py --evaluate "diffusion-two-stage"
echo "" # 打印一个空行，让输出更好看
python process_rewards.py -p diffusion-two-stage
echo "--> 评估程序运行结束。"
echo "" # 打印一个空行，让输出更好看
