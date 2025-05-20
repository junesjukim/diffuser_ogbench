import sys
import os
import numpy as np
import torch
import imageio

# diffuser_ogbench 경로를 Python path에 추가
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

import diffuser.utils as utils
from diffuser.models.diffusion import set_model_mode
from diffuser.datasets.ogbench import OGBenchGoalDataset
from diffuser.sampling.policies import GuidedPolicy

class Parser(utils.Parser):
    dataset: str = 'scene-play-singletask-task2-v0'
    config: str = 'config.locomotion'
    horizon: int = 64
    n_diffusion_steps: int = 100
    batch_size: int = 32
    normalizer: str = 'LimitsNormalizer'
    use_padding: bool = True
    max_path_length: int = 1000
    # sampling specific parameters
    n_samples: int = 1
    guide_scale: float = 1.0
    verbose: bool = True

args = Parser().parse_args('value_guided_sampling')
set_model_mode(args.prefix)

# 데이터셋 설정
dataset_config = utils.Config(
    OGBenchGoalDataset,
    savepath=(args.savepath, 'dataset_config.pkl'),
    env_name=args.dataset,
    horizon=args.horizon,
    normalizer=args.normalizer,
    preprocess_fns=args.preprocess_fns,
    use_padding=args.use_padding,
    max_path_length=args.max_path_length,
)

# 환경 및 데이터셋 초기화
dataset = dataset_config()
env = dataset.env

# Behavior Policy (Diffusion Model) 로드
model_config = utils.Config(
    args.model,
    savepath=(args.savepath, 'model_config.pkl'),
    horizon=args.horizon,
    transition_dim=dataset.observation_dim + dataset.action_dim,
    cond_dim=dataset.observation_dim * 2,  # 현재 상태 + 목표 상태
    dim_mults=args.dim_mults,
    attention=args.attention,
    device=args.device,
)

diffusion_config = utils.Config(
    args.diffusion,
    savepath=(args.savepath, 'diffusion_config.pkl'),
    horizon=args.horizon,
    observation_dim=dataset.observation_dim,
    action_dim=dataset.action_dim,
    n_timesteps=args.n_diffusion_steps,
    device=args.device,
)

# Value Model 로드
value_model_config = utils.Config(
    args.model,
    savepath=(args.savepath, 'value_model_config.pkl'),
    horizon=args.horizon,
    transition_dim=dataset.observation_dim + dataset.action_dim,
    cond_dim=dataset.observation_dim * 2,  # 현재 상태 + 목표 상태
    dim_mults=args.dim_mults,
    attention=args.attention,
    device=args.device,
)

value_diffusion_config = utils.Config(
    args.diffusion,
    savepath=(args.savepath, 'value_diffusion_config.pkl'),
    horizon=args.horizon,
    observation_dim=dataset.observation_dim,
    action_dim=dataset.action_dim,
    n_timesteps=args.n_diffusion_steps,
    device=args.device,
)

# 모델 초기화
model = model_config()
diffusion = diffusion_config(model)
value_model = value_model_config()
value_diffusion = value_diffusion_config(value_model)

# Guided policy 초기화
policy = GuidedPolicy(
    guide=value_diffusion,
    diffusion_model=diffusion,
    normalizer=dataset.normalizer,
    preprocess_fns=args.preprocess_fns,
    guide_scale=args.guide_scale,
)

# 평가 루프
for task_id in range(1, 6):
    # 환경 초기화
    obs, info = env.reset(
        options=dict(
            task_id=task_id,
            render_goal=True,
        )
    )
    
    goal = info['goal']
    goal_rendered = info['goal_rendered']
    
    # 비디오 저장 설정
    video_writer = imageio.get_writer(f'task_{task_id}_value_guided.mp4', fps=30)
    
    done = False
    while not done:
        # 현재 상태와 목표 상태를 조건으로 사용
        conditions = {
            0: obs,  # 현재 상태
            args.horizon - 1: goal  # 목표 상태
        }
        
        # Value guided policy로 액션 생성
        action, trajectories = policy(
            conditions,
            batch_size=args.n_samples,
            verbose=args.verbose
        )
        
        # 환경에서 액션 실행
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        
        # 프레임 저장
        frame = env.render()
        video_writer.append_data(frame)
    
    video_writer.close()
    print(f"Task {task_id} completed")
    print(f"Success: {info['success']}") 