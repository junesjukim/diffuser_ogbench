import numpy as np
import torch
import gymnasium as gym
import argparse
import os
from tqdm import trange
import time
import json
import utils
from utils import VideoRecorder
import IQL
import wandb
from log import Logger
import ogbench
import subprocess

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def setup_rendering():
    """렌더링을 위한 가상 디스플레이 설정"""
    subprocess.run(['pkill', 'Xvfb'])
    subprocess.Popen(['Xvfb', ':100', '-screen', '0', '1024x768x24', '-ac'])
    os.environ['DISPLAY'] = ':100.0'
    os.environ['MUJOCO_GL'] = 'egl'
    os.environ['PYOPENGL_PLATFORM'] = 'egl'
    os.environ['EGL_DEVICE_ID'] = '0'
    time.sleep(2)  # Xvfb 시작 대기

def load_ogbench_dataset(task_id):
    """OGBench 데이터셋 로드"""
    # X11 디스플레이 설정
    os.environ['DISPLAY'] = ':0'
    
    # make_env_and_datasets를 사용하여 데이터셋 로드
    env, train_dataset, val_dataset = ogbench.make_env_and_datasets(
        dataset_name=f'scene-play-singletask-task{task_id}-v0',
        render_mode='rgb_array',
        width=640,
        height=480,
        max_episode_steps=1000
    )
    
    # 데이터셋이 튜플로 반환되는 경우를 처리
    if isinstance(train_dataset, tuple):
        observations, actions, rewards, next_observations, dones = train_dataset
        dataset = {
            'observations': observations,
            'actions': actions,
            'rewards': rewards,
            'next_observations': next_observations,
            'dones': dones
        }
    
    # 데이터셋 크기 확인
    print(f"Dataset shape: {train_dataset['observations'].shape}")
    
    return train_dataset, val_dataset

def eval_policy(args, iter, video: VideoRecorder, policy, env_name, seed, mean, std, seed_offset=100, eval_episodes=10):
    eval_env, _, _ = ogbench.make_env_and_datasets(
        dataset_name=env_name
    )

    lengths = []
    returns = []
    avg_reward = 0.
    success_count = 0
    
    # 결과를 저장할 디렉토리 생성
    results_dir = os.path.join(args.work_dir, 'eval_results')
    os.makedirs(results_dir, exist_ok=True)
    
    for _ in range(eval_episodes):
        video.init(enabled=(args.save_video and _ == 0))
        state, info = eval_env.reset()
        video.record(eval_env)
        steps = 0
        episode_return = 0
        done = False
        truncated = False
        while not (done or truncated):
            state = (np.array(state).reshape(1, -1) - mean)/std
            action = policy.select_action(state)
            state, reward, terminated, truncated, info = eval_env.step(action)
            done = terminated
            video.record(eval_env)
            avg_reward += reward
            episode_return += reward
            steps += 1
            
            # 성공 조건 체크 (info에서 success 정보 확인)
            if info.get('success', False):
                success_count += 1
                break
                
        lengths.append(steps)
        returns.append(episode_return)
        video.save(f'eval_s{iter}_r{str(episode_return)}.mp4')

    avg_reward /= eval_episodes
    success_rate = (success_count / eval_episodes) * 100

    # 결과를 txt 파일에 저장
    result_file = os.path.join(results_dir, f'eval_results_{iter}.txt')
    with open(result_file, 'w') as f:
        f.write(f"Evaluation at step {iter}\n")
        f.write(f"Average reward: {avg_reward:.3f}\n")
        f.write(f"Success rate: {success_rate:.2f}%\n")
        f.write(f"Average episode length: {np.mean(lengths):.2f}\n")
        f.write(f"Number of successful episodes: {success_count}/{eval_episodes}\n")

    print("---------------------------------------")
    print(f"Evaluation over {eval_episodes} episodes:")
    print(f"Average reward: {avg_reward:.3f}")
    print(f"Success rate: {success_rate:.2f}%")
    print(f"Average episode length: {np.mean(lengths):.2f}")
    print(f"Successful episodes: {success_count}/{eval_episodes}")
    print("---------------------------------------")
    
    return avg_reward, success_rate

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Experiment
    parser.add_argument("--policy", default="IQL")
    parser.add_argument("--task_id", default=1, type=int)
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--eval_freq", default=1e4, type=int)
    parser.add_argument("--max_timesteps", default=1e6, type=int)
    parser.add_argument("--save_model", action="store_true", default=True)
    parser.add_argument('--eval_episodes', default=10, type=int)
    parser.add_argument('--save_video', default=False, action='store_true')
    parser.add_argument("--normalize", default=True, action='store_true')
    parser.add_argument("--wandb_project", default="ogbench-iql-test", type=str)
    parser.add_argument("--wandb_entity", default=None, type=str)
    
    # IQL
    parser.add_argument("--batch_size", default=256, type=int)
    parser.add_argument("--temperature", default=3.0, type=float)
    parser.add_argument("--expectile", default=0.7, type=float)
    parser.add_argument("--tau", default=0.005, type=float)
    parser.add_argument("--discount", default=0.99, type=float)
    
    # Work dir
    parser.add_argument('--work_dir', default='ogbench_runs', type=str)
    args = parser.parse_args()

    # 렌더링 설정
    if args.save_video:
        setup_rendering()

    # Build work dir
    base_dir = 'runs'
    utils.make_dir(base_dir)
    base_dir = os.path.join(base_dir, args.work_dir)
    utils.make_dir(base_dir)
    args.work_dir = os.path.join(base_dir, f'task{args.task_id}')
    utils.make_dir(args.work_dir)

    # make directory
    ts = time.gmtime()
    ts = time.strftime("%m-%d-%H-%M", ts)
    exp_name = f'test-task{args.task_id}-{ts}-bs{args.batch_size}-s{args.seed}'
    if args.policy == 'IQL':
        exp_name += f'-t{args.temperature}-e{args.expectile}'
    args.work_dir = args.work_dir + '/' + exp_name
    utils.make_dir(args.work_dir)

    args.model_dir = os.path.join(args.work_dir, 'model')
    utils.make_dir(args.model_dir)
    args.video_dir = os.path.join(args.work_dir, 'video')
    utils.make_dir(args.video_dir)

    with open(os.path.join(args.work_dir, 'args.json'), 'w') as f:
        json.dump(vars(args), f, sort_keys=True, indent=4)

    print("---------------------------------------")
    print(f"Policy: {args.policy}, Task: {args.task_id}, Seed: {args.seed}")
    print("---------------------------------------")

    # 데이터셋 로드
    train_dataset, val_dataset = load_ogbench_dataset(args.task_id)
    
    state_dim = train_dataset['observations'].shape[1]
    action_dim = train_dataset['actions'].shape[1]
    max_action = 1.0  # OGBench의 action space는 [-1, 1]

    kwargs = {
        "state_dim": state_dim,
        "action_dim": action_dim,
        "discount": args.discount,
        "tau": args.tau,
        "temperature": args.temperature,
        "expectile": args.expectile,
    }

    # Initialize policy
    policy = IQL.IQL(**kwargs)

    # Replay buffer 설정
    replay_buffer = utils.ReplayBuffer(state_dim, action_dim)
    replay_buffer.load_dataset(train_dataset)
    
    if args.normalize:
        mean, std = replay_buffer.normalize_states()
    else:
        mean, std = 0, 1

    video = VideoRecorder(dir_name=args.video_dir)
    logger = Logger(args.work_dir, use_tb=False)  # tensorboard는 사용하지 않음

    # wandb 초기화
    wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        config=vars(args),
        name=exp_name
    )

    # 학습 시작
    pbar = trange(int(args.max_timesteps), desc='Training')
    eval_results = []  # 평가 결과를 저장할 리스트
    
    for t in pbar:
        policy.train(replay_buffer, args.batch_size, logger=logger)
        
        # wandb에 학습 메트릭 로깅
        wandb.log({
            "train/step": t,
            "train/critic_loss": policy.critic_loss,
            "train/actor_loss": policy.actor_loss,
            "train/value_loss": policy.value_loss
        })
        
        # 진행 상황 업데이트
        pbar.set_postfix({
            'critic_loss': f'{policy.critic_loss:.3f}',
            'actor_loss': f'{policy.actor_loss:.3f}',
            'value_loss': f'{policy.value_loss:.3f}'
        })
        
        # Evaluate
        if (t + 1) % args.eval_freq == 0:
            eval_episodes = 100 if t+1 == int(args.max_timesteps) else args.eval_episodes
            avg_reward, success_rate = eval_policy(args, t+1, video, policy, f'scene-play-singletask-task{args.task_id}-v0',
                                   args.seed, mean, std, eval_episodes=eval_episodes)
            
            eval_results.append((t+1, avg_reward, success_rate))
            
            # wandb에 평가 메트릭 로깅
            wandb.log({
                "eval/step": t + 1,
                "eval/avg_reward": avg_reward,
                "eval/success_rate": success_rate
            })
            
            if args.save_model:
                policy.save(args.model_dir)
                print(f"  Model saved to {args.model_dir}")

    wandb.finish()
