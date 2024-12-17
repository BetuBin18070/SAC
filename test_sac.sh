#!/bin/bash

# 参数设置
env_ids=("Hopper-v3" "Walker2d-v3" "HalfCheetah-v3" "Ant-v3")  # 设置环境ID列表
seed_ids=(1 2 3 4 5)  # 设置种子ID列表
device_ids=(4 5 6 6)  # 设置设备ID列表

# 创建tmux窗口并执行命令
for ((i=0; i<${#env_ids[@]}; i++)); do
    env_id=${env_ids[i]}
    device_id=${device_ids[i]}
    for seed_id in "${seed_ids[@]}"; do
        tmux new -d -s "test_sac_env${env_id}_seed${seed_id}"  # 创建带有特定名称的tmux会话
        tmux send-keys -t "test_sac_env${env_id}_seed${seed_id}" "conda activate baseRL" C-m  # 激活conda环境

        tmux send-keys -t "test_sac_env${env_id}_seed${seed_id}" "cd /home/users/liqingbin/Research/PureRLCode/SAC/" C-m  # 激活conda环境
        tmux send-keys -t "test_sac_env${env_id}_seed${seed_id}" "CUDA_VISIBLE_DEVICES=$device_id python save_res.py --env_name $env_id --seed $seed_id --cuda" C-m  # 执行Python命令
    done
done