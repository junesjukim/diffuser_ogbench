#!/bin/bash

# 각 task별 GPU 할당
declare -A gpu_map
gpu_map[2]=0  # task 2 -> GPU 0
gpu_map[3]=1  # task 3 -> GPU 1
gpu_map[4]=2  # task 4 -> GPU 2
gpu_map[5]=3  # task 5 -> GPU 3

# 각 task별 프로세스를 백그라운드로 실행
for task_id in 2 3 4 5; do
    gpu_id=${gpu_map[$task_id]}
    echo "Starting Task $task_id on GPU $gpu_id"
    
    # 각 task별 로그 파일 생성
    log_file="task${task_id}_gpu${gpu_id}.log"
    
    # CUDA_VISIBLE_DEVICES를 설정하고 백그라운드로 실행
    CUDA_VISIBLE_DEVICES=$gpu_id python train_ogbench.py \
        --task_id $task_id \
        --save_video \
        --wandb_project "ogbench-iql-all-tasks" \
        --max_timesteps 1e6 \
        --eval_freq 1e4 \
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

echo "모든 task가 시작되었습니다. 로그 파일을 확인하세요:"
for task_id in 2 3 4 5; do
    echo "Task $task_id: task${task_id}_gpu${gpu_map[$task_id]}.log"
done

# 모든 프로세스가 완료될 때까지 대기
echo "모든 task가 완료될 때까지 대기 중..."
for task_id in 2 3 4 5; do
    wait ${pids[$task_id]}
    echo "Task $task_id 완료"
done

echo "모든 task가 완료되었습니다!" 