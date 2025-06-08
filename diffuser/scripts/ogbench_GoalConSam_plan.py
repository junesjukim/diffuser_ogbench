import ogbench
import imageio
import numpy as np
import os
import torch
import diffuser.utils as utils
from diffuser.models.diffusion import set_model_mode
from diffuser.sampling.policies import GuidedPolicy
from diffuser.sampling.functions import n_step_guided_p_sample
import subprocess
import time

subprocess.run(['pkill', 'Xvfb'])

# Xvfb를 사용하여 가상 디스플레이 설정
subprocess.Popen(['Xvfb', ':100', '-screen', '0', '1024x768x24', '-ac'])
os.environ['DISPLAY'] = ':100.0'
os.environ['MUJOCO_GL'] = 'egl'
os.environ['PYOPENGL_PLATFORM'] = 'egl'
os.environ['EGL_DEVICE_ID'] = '0'
time.sleep(2)  # Xvfb가 시작될 때까지 대기 시간 증가

# 모델 모드 설정
set_model_mode('diffusion')

# Make an environment and datasets (they will be automatically downloaded).
dataset_name = 'scene-play-singletask-task2-v0'
env, train_dataset, val_dataset = ogbench.make_env_and_datasets(
    dataset_name,
    render_mode='rgb_array',  # rgb_array 모드 사용
    width=640,  # 해상도 낮춤
    height=480  # 해상도 낮춤
)

# Load diffusion model
diffusion_experiment = utils.load_diffusion(
    'logs',  # 로드베이스 경로 수정
    dataset_name,
    'diffusion/flowmatching_H32_T20_SNone',  # diffusion 모델 경로
    epoch='latest',
    n_sample_timesteps=20,
    renderer=env  # 환경의 렌더러 사용
)

# Initialize policy without value guide
diffusion = diffusion_experiment.ema
policy = GuidedPolicy(
    guide=None,  # value guide 없이 사용
    diffusion_model=diffusion,
    normalizer=diffusion_experiment.dataset.normalizer,
    preprocess_fns=[],
    n_guide_steps=0,  # guide steps를 0으로 설정
    scale=0.0,  # scale을 0으로 설정
    t_stopgrad=0,  # t_stopgrad를 0으로 설정
    sample_fn=n_step_guided_p_sample,  # n_step_guided_p_sample 사용
)

# Train your offline goal-conditioned RL agent on the dataset.
# ...

# Evaluate the agent.
for task_id in [1, 2, 3, 4, 5]:
    # Reset the environment and set the evaluation task.
    ob, info = env.reset(
        options=dict(
            task_id=task_id,  # Set the evaluation task. Each environment provides five
                              # evaluation goals, and `task_id` must be in [1, 5].
            render_goal=False,  # Set to `True` to get a rendered goal image (optional).
        )
    )

    goal = info['goal']  # Get the goal observation to pass to the agent.
    #goal_rendered = info['goal_rendered']  # Get the rendered goal image (optional).

    # Initialize video writer
    video_writer = imageio.get_writer(f'task_{task_id}.mp4', fps=30)

    done = False
    while not done:
        # Use diffuser model to generate action
        conditions = {
            0: ob,  # 현재 상태
            diffusion.horizon - 1: goal,  # 목표 상태
        }
        action, trajectories = policy(conditions, batch_size=1)
        
        ob, reward, terminated, truncated, info = env.step(action)  # action은 이미 numpy array
        done = terminated or truncated
        frame = env.render()  # Render the current frame
        video_writer.append_data(frame)  # Save frame to video

    video_writer.close()  # Close the video writer
    print(f"Task {task_id} completed")
    success = info['success']  # Whether the agent reached the goal (0 or 1).
                               # `terminated` also indicates this.

# Xvfb 프로세스 종료
subprocess.run(['pkill', 'Xvfb'])