import sys
import os
import numpy as np
import torch
import imageio
import subprocess
import time
import pickle
from diffuser.sampling.functions import n_step_guided_p_sample

# diffuser_ogbench 경로를 Python path에 추가
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

import diffuser.utils as utils
from diffuser.models.diffusion import set_model_mode
from diffuser.datasets.ogbench import OGBenchGoalDataset
from diffuser.sampling.policies import GuidedPolicy
import ogbench

# Xvfb를 사용하여 가상 디스플레이 설정
subprocess.run(['pkill', 'Xvfb'])
subprocess.Popen(['Xvfb', ':100', '-screen', '0', '1024x768x24', '-ac'])
os.environ['DISPLAY'] = ':100.0'
os.environ['MUJOCO_GL'] = 'egl'
os.environ['PYOPENGL_PLATFORM'] = 'egl'
os.environ['EGL_DEVICE_ID'] = '0'
time.sleep(2)  # Xvfb가 시작될 때까지 대기

task_id = 3
class Parser(utils.Parser):
    dataset: str = f'scene-play-singletask-task{task_id}-v0'
    config: str = 'config.locomotion'
    horizon: int = 64
    n_diffusion_steps: int = 100
    batch_size: int = 32
    normalizer: str = 'LimitsNormalizer'
    use_padding: bool = True
    max_path_length: int = 1000
    # sampling specific parameters
    n_samples: int = 32
    guide_scale: float = 1.0
    verbose: bool = True
    # 추가된 설정
    
args = Parser().parse_args('plan')
set_model_mode(args.prefix)

# 현재 작업 디렉토리의 절대 경로 설정
args.loadbase = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'logs')

# 환경 및 데이터셋 초기화
env, train_dataset, val_dataset = ogbench.make_env_and_datasets(
    args.dataset,
    render_mode='rgb_array',
    width=640,
    height=480
)

# render_config.pkl 파일 생성
render_config = {
    'width': 640,
    'height': 480,
    'render_mode': 'rgb_array',
    'camera_id': 0,
    'camera_name': 'default',
    'camera_pos': [0, 0, 0],
    'camera_rot': [0, 0, 0],
}

# diffusion과 value 함수를 위한 render_config.pkl 파일 저장
paths = [
    os.path.join(args.loadbase, args.dataset, args.diffusion_loadpath, 'render_config.pkl'),
    os.path.join(args.loadbase, args.dataset, args.value_loadpath, 'render_config.pkl')
]

for path in paths:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'wb') as f:
        pickle.dump(render_config, f)
    print(f"Created render_config.pkl at: {path}")

# Diffusion 모델과 Value 함수 로드
print(f"========== n_sample_timesteps: {args.n_sample_timesteps} ==========", flush=True)
diffusion_experiment = utils.load_diffusion(
    args.loadbase, args.dataset, args.diffusion_loadpath,
    epoch=args.diffusion_epoch, seed=args.seed,
    n_sample_timesteps=args.n_sample_timesteps,
    renderer=env  # 환경의 렌더러 사용
)

value_experiment = utils.load_diffusion(
    args.loadbase, args.dataset, args.value_loadpath,
    epoch=args.value_epoch, seed=args.seed,
    n_sample_timesteps=args.n_sample_timesteps,
    renderer=env  # 환경의 렌더러 사용
)

# 모델 호환성 확인
utils.check_compatibility(diffusion_experiment, value_experiment)

diffusion = diffusion_experiment.ema
value_function = value_experiment.ema
renderer = diffusion_experiment.renderer

# Value guide 초기화
guide_config = utils.Config(args.guide, model=value_function, verbose=False)
guide = guide_config()

# Guided policy 초기화
policy_config = utils.Config(
    args.policy,
    guide=guide,
    scale=args.scale,
    diffusion_model=diffusion,
    normalizer=diffusion_experiment.dataset.normalizer,
    preprocess_fns=args.preprocess_fns,
    n_guide_steps=args.n_guide_steps,
    t_stopgrad=args.t_stopgrad,
    scale_grad_by_std=args.scale_grad_by_std,
    verbose=False,
    sample_fn=n_step_guided_p_sample,  # guide를 지원하는 샘플링 함수 명시
)

policy = policy_config()

# 평가 루프 - task 2에 대해 10번 실행
  # task 2만 실행
for run_id in range(10):  # 10번 실행
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
    video_path = os.path.join(args.savepath, f'task_{task_id}_run_{run_id}_value_guided.mp4')
    video_writer = imageio.get_writer(video_path, fps=30)
    
    max_steps = 1000  # 최대 step 제한
    done = False
    step = 0  # step 카운터 추가
    while not done and step < max_steps:
        # 현재 상태와 목표 상태를 조건으로 사용
        conditions = {
            0: obs,  # 현재 상태
            #diffusion.horizon - 1: goal,  # 목표 상태
        }
        
        # Value guided policy로 액션 생성
        action, samples = policy(
            conditions,
            batch_size=args.n_samples,
            verbose=args.verbose
        )
        
        # 환경에서 액션 실행
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        
        # step이 100의 배수일 때 출력
        if step % 100 == 0:
            print(f"\n{'='*50}")
            print(f"[Task {task_id}, Run {run_id+1}/10] Step: {step}")
            print(f"{'='*50}\n")
        
        # 프레임 저장
        frame = env.render()
        video_writer.append_data(frame)
        step += 1
    
    video_writer.close()
    print(f"Task {task_id}, Run {run_id+1}/10 completed")
    print(f"Success: {info['success']}")
    print(f"Total frames saved: {step}")  # 실제 저장된 프레임 수 출력

# Xvfb 프로세스 종료
subprocess.run(['pkill', 'Xvfb']) 