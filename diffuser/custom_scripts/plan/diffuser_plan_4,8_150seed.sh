#!/bin/bash

# Create output directory if it doesn't exist
mkdir -p output

# Loop over seed values from 0 to 149
for seed in {0..149}
do
  # Run first job on GPU 0
  OMP_NUM_THREADS=24 CUDA_VISIBLE_DEVICES=0 python scripts/plan_guided.py \
    --dataset halfcheetah-medium-expert-v2 \
    --logbase logs \
    --diffusion_loadpath 'f:diffusion/defaults_H4_T4_S0' \
    --value_loadpath 'f:values/diffuser_H4_T4_S0_d0.99' \
    --horizon 4 \
    --n_diffusion_steps 4 \
    --seed $seed \
    --discount 0.99 > output/output_0_seed_${seed}.log 2>&1 &

  pid1=$!  # Capture process ID of first job

  # Run second job on GPU 1
  OMP_NUM_THREADS=24 CUDA_VISIBLE_DEVICES=1 python scripts/plan_guided.py \
    --dataset halfcheetah-medium-expert-v2 \
    --logbase logs \
    --diffusion_loadpath 'f:diffusion/defaults_H4_T8_S0' \
    --value_loadpath 'f:values/diffuser_H4_T8_S0_d0.99' \
    --horizon 4 \
    --n_diffusion_steps 8 \
    --seed $seed \
    --discount 0.99 > output/output_1_seed_${seed}.log 2>&1 &

  pid2=$!  # Capture process ID of second job

  # Run third job on GPU 2
  OMP_NUM_THREADS=24 CUDA_VISIBLE_DEVICES=2 python scripts/plan_guided.py \
    --dataset halfcheetah-medium-replay-v2 \
    --logbase logs \
    --diffusion_loadpath 'f:diffusion/defaults_H4_T4_S0' \
    --value_loadpath 'f:values/diffuser_H4_T4_S0_d0.99' \
    --horizon 4 \
    --n_diffusion_steps 4 \
    --seed $seed \
    --discount 0.99 > output/output_2_seed_${seed}.log 2>&1 &

  pid3=$!  # Capture process ID of third job

  # Run fourth job on GPU 3
  OMP_NUM_THREADS=24 CUDA_VISIBLE_DEVICES=3 python scripts/plan_guided.py \
    --dataset halfcheetah-medium-replay-v2 \
    --logbase logs \
    --diffusion_loadpath 'f:diffusion/defaults_H4_T8_S0' \
    --value_loadpath 'f:values/diffuser_H4_T8_S0_d0.99' \
    --horizon 4 \
    --n_diffusion_steps 8 \
    --seed $seed \
    --discount 0.99 > output/output_3_seed_${seed}.log 2>&1 &

  pid4=$!  # Capture process ID of fourth job

  echo "Started jobs for seed $seed on GPU 0, GPU 1, GPU 2, and GPU 3"

  # Wait for all background jobs to finish before moving to the next seed
  wait $pid1 $pid2 $pid3 $pid4

  echo "Completed jobs for seed $seed"
done

echo "All jobs have been completed."
