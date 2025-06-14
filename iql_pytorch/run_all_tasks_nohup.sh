#!/bin/bash

# conda 환경 활성화
source ~/anaconda3/etc/profile.d/conda.sh
conda activate _diffuser_og

# 각 task별 GPU 할당
declare -A gpu_map
gpu_map[1]=0  # task 1 -> GPU 0
gpu_map[2]=1  # task 2 -> GPU 1
gpu_map[3]=2  # task 3 -> GPU 2
gpu_map[4]=3  # task 4 -> GPU 3
gpu_map[5]=4  # task 5 -> GPU 4

# 각 task별 프로세스를 백그라운드로 실행
for task_id in 2 3; do
    gpu_id=${gpu_map[$task_id]}
    echo "Starting Task $task_id on GPU $gpu_id"
    
    # 각 task별 로그 파일 생성
    log_file="nohup_task${task_id}_gpu${gpu_id}.log"
    
    # CUDA_VISIBLE_DEVICES를 설정하고 nohup으로 실행
    export CUDA_VISIBLE_DEVICES=$gpu_id
    nohup python train_ogbench.py \
        --task_id $task_id \
        --save_video \
        --wandb_project "ogbench-iql-all-tasks" \
        --max_timesteps 1000000 \
        --eval_freq 10000 \
        --batch_size 256 \
        --temperature 3.0 \
        --expectile 0.7 \
        --tau 0.005 \
        --discount 0.99 \
        > $log_file 2>&1 &
    
    # 프로세스 ID 저장
    pids[$task_id]=$!
    echo "Task $task_id started with PID ${pids[$task_id]}"
    
    # 각 task 시작 사이에 약간의 딜레이
    sleep 5
done

echo "모든 task가 nohup으로 시작되었습니다. 로그 파일을 확인하세요:"
for task_id in 2 3 4 5; do
    echo "Task $task_id: nohup_task${task_id}_gpu${gpu_map[$task_id]}.log"
done

# PID 저장
echo "실행 중인 프로세스 ID:"
for task_id in 2 3 4 5; do
    echo "Task $task_id: ${pids[$task_id]}"
done > running_pids.txt

echo "모든 task가 백그라운드에서 실행 중입니다. running_pids.txt 파일에서 PID를 확인할 수 있습니다."