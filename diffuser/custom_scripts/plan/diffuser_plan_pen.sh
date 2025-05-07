#!/bin/bash
#test
# 변수 정의
n_diffusion_steps=16
prefix_path="diffusion_plan/pen"

# Create output directory if it doesn't exist
mkdir -p output
mkdir -p output/${prefix_path}

# GPU 장치 배열 정의
declare -a GPU_DEVICES=(4 5 6 7 8)

# 데이터셋 배열 정의
declare -a DATASETS=(
  #pen-cloned-v0"
  "pen-cloned-v0"
  "pen-cloned-v0"
  "pen-cloned-v0"
  "pen-cloned-v0"
  "pen-cloned-v0"
)

# n_sample_timesteps 변수 정의
declare -a n_sample_timesteps=(
  16
  8
  4
  2
  1
)


# Loop over seed values from 0 to 149
for seed in {0..149}
do
  # 각 GPU에서 작업 실행
  pids=()
  for i in "${!GPU_DEVICES[@]}"; do
    OMP_NUM_THREADS=24 CUDA_VISIBLE_DEVICES=${GPU_DEVICES[$i]} python scripts/plan_guided.py \
      --dataset ${DATASETS[$i]} \
      --logbase logs \
      --diffusion_loadpath "f:diffusion/diffuser_H32_T${n_diffusion_steps}_S0" \
      --value_loadpath "f:values/diffusion_H32_T${n_diffusion_steps}_S0_d0.99" \
      --horizon 32 \
      --n_diffusion_steps ${n_diffusion_steps} \
      --seed $seed \
      --n_sample_timesteps ${n_sample_timesteps[$i]} \
      --discount 0.99 \
      --prefix ${prefix_path} > output/${prefix_path}/output_${GPU_DEVICES[$i]}_seed_${seed}.log 2>&1 &

    pids+=($!)
    echo "----------------------------------------"
    echo "[작업 시작] GPU ${GPU_DEVICES[$i]}"
    echo "- 데이터셋: ${DATASETS[$i]}"
    echo "- 시드: $seed" 
    echo "- Diffusion Steps: ${n_diffusion_steps}"
    echo "- Sample Timesteps: ${n_sample_timesteps[$i]}"
    echo "- PID: $!"
    echo "- Prefix: ${prefix_path}"
    echo "----------------------------------------"
  done

  # Wait for all background jobs to finish before moving to the next seed
  wait "${pids[@]}"
  echo "시드 $seed에 대한 모든 작업이 완료되었습니다"
done

echo "모든 작업이 완료되었습니다."

