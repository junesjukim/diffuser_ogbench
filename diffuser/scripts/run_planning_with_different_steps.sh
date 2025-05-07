#!/bin/bash

# 여러 diffusion step으로 계획을 생성하는 스크립트

# 기본 경로 설정
DATASET="walker2d-medium-replay-v2"
SAVE_DIR="results/different_diffusion_steps"

# 저장 디렉토리 생성
mkdir -p $SAVE_DIR

# 다양한 diffusion step에 대한 계획 실행
for STEPS in 16 8 4 2 1
do
    echo "Running planning with n_sample_timesteps = $STEPS"
    
    # 결과 저장 경로
    OUTPUT_DIR="$SAVE_DIR/steps_$STEPS"
    mkdir -p $OUTPUT_DIR
    
    # 계획 실행
    python scripts/plan_guided.py \
        --dataset $DATASET \
        --n_sample_timesteps $STEPS \
        --savepath $OUTPUT_DIR
    
    echo "Finished planning with n_sample_timesteps = $STEPS"
    echo "Results saved to $OUTPUT_DIR"
    echo ""
done

echo "All planning runs completed!" 