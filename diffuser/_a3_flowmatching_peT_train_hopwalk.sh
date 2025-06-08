#!/bin/bash
# Create output directories if they don't exist

prefix_str="peT_hopwalk"

mkdir -p output
mkdir -p output/flowmatching_${prefix_str}

n_diffusion_steps=20
seed=0

# GPU 장치 배열 정의
declare -a GPU_DEVICES=(0 1)
# 데이터셋 배열 정의  
declare -a DATASETS=(
  "walker2d-medium-replay-v2"
  "hopper-medium-replay-v2"
)

# 각 GPU에서 작업 실행
pids=()
for i in "${!GPU_DEVICES[@]}"; do
  OMP_NUM_THREADS=24 CUDA_VISIBLE_DEVICES=${GPU_DEVICES[$i]} python scripts/train.py \
    --predict_epsilon True \
    --dataset ${DATASETS[$i]} \
    --logbase logs \
    --horizon 4 \
    --n_diffusion_steps ${n_diffusion_steps} \
    --seed $seed \
    --prefix 'flowmatching/flowmatcher_peT' > output/flowmatching_${prefix_str}/output_${GPU_DEVICES[$i]}_seed_${seed}.log 2>&1 &

  pids+=($!)
  echo "Started job for seed $seed on GPU ${GPU_DEVICES[$i]}"
done

# Wait for all background jobs to finish
wait "${pids[@]}"

echo "All jobs have been completed."
