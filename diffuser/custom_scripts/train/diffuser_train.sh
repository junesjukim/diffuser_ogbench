#!/bin/bash
# Create output directories if they don't exist
mkdir -p output
mkdir -p output/diffuser_pen_kitchen

n_diffusion_steps=16
seed=0

# GPU 장치 배열 정의
declare -a GPU_DEVICES=(0 1)
# 데이터셋 배열 정의  
declare -a DATASETS=(
  "pen-cloned-v0"
  "kitchen-partial-v0"
)

# 각 GPU에서 작업 실행
pids=()
for i in "${!GPU_DEVICES[@]}"; do
  OMP_NUM_THREADS=24 CUDA_VISIBLE_DEVICES=${GPU_DEVICES[$i]} python scripts/train.py \
    --dataset ${DATASETS[$i]} \
    --normalizer 'DebugNormalizer' \
    --logbase logs \
    --horizon 32 \
    --n_diffusion_steps ${n_diffusion_steps} \
    --seed $seed \
    --prefix 'diffusion/diffuser' > output/diffuser_pen_kitchen/output_${GPU_DEVICES[$i]}_seed_${seed}.log 2>&1 &

  pids+=($!)
  echo "Started job for seed $seed on GPU ${GPU_DEVICES[$i]}"
done

# Wait for all background jobs to finish
wait "${pids[@]}"

echo "All jobs have been completed."
