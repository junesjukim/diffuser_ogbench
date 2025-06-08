import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
import subprocess
import time
import imageio
import ogbench
import gymnasium as gym
import traceback

# diffuser_ogbench 경로를 Python path에 추가
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

import diffuser.utils as utils
from diffuser.datasets.ogbench import OGBenchGoalDataset

class Parser(utils.Parser):
    dataset: str = 'scene-play-singletask-task1-v0'
    config: str = 'config.locomotion'
    horizon: int = 64
    normalizer: str = 'LimitsNormalizer'
    use_padding: bool = True
    max_path_length: int = 1000

def get_episode_indices(terminals, episode_idx):
    """Returns (start, end) indices for the requested episode from the terminals array"""
    ends = np.where(terminals)[0]
    if episode_idx == 0:
        start = 0
    else:
        start = ends[episode_idx - 1] + 1
    end = ends[episode_idx] + 1  # end is exclusive
    return start, end

def create_stats_figure(observations_history, actions_history, rewards_history, step, obs_dim, action_dim, env_rewards_history=None):
    """Create time-series visualization of observations, actions, and rewards up to current step"""
    
    # reward가 있는지 확인
    has_rewards = rewards_history is not None
    
    # Render Matplotlib charts in memory
    num_plots = 3 if has_rewards else 2
    fig, axes = plt.subplots(num_plots, 1, figsize=(8, 4 * num_plots))
    
    # Time steps for x-axis
    timesteps = np.arange(step + 1)
    
    # Calculate y-axis limits with some padding
    obs_min = np.min(observations_history[:step+1, :min(3, obs_dim)]) - 0.1
    obs_max = np.max(observations_history[:step+1, :min(3, obs_dim)]) + 0.1
    action_min = np.min(actions_history[:step+1, :min(3, action_dim)]) - 0.1
    action_max = np.max(actions_history[:step+1, :min(3, action_dim)]) + 0.1
    
    if has_rewards:
        reward_min = np.min(rewards_history[:step+1]) - 0.1
        reward_max = np.max(rewards_history[:step+1]) + 0.1
    
    # Observations chart (time series)
    ax_obs = axes[0]
    for i in range(min(3, obs_dim)):
        ax_obs.plot(timesteps, observations_history[:step+1, i], 
                    label=f'Obs {i}', 
                    linewidth=2)
    
    # Add current values as text
    current_obs = observations_history[step, :min(3, obs_dim)]
    obs_text = ", ".join([f"Obs{i}: {val:.3f}" for i, val in enumerate(current_obs)])
    ax_obs.text(0.02, 0.05, obs_text, transform=ax_obs.transAxes, 
                fontsize=10, bbox=dict(facecolor='white', alpha=0.7))
    
    ax_obs.set_title(f'Observations History (Current Step: {step})')
    ax_obs.set_ylim([obs_min, obs_max])
    ax_obs.legend(loc='upper right')
    ax_obs.grid(True, linestyle='--', alpha=0.7)
    
    # Actions chart (time series)
    ax_act = axes[1]
    for i in range(min(3, action_dim)):
        ax_act.plot(timesteps, actions_history[:step+1, i], 
                    label=f'Action {i}', 
                    linewidth=2)
    
    # Add current values as text
    current_actions = actions_history[step, :min(3, action_dim)]
    action_text = ", ".join([f"Act{i}: {val:.3f}" for i, val in enumerate(current_actions)])
    ax_act.text(0.02, 0.05, action_text, transform=ax_act.transAxes, 
                fontsize=10, bbox=dict(facecolor='white', alpha=0.7))
    
    ax_act.set_title(f'Actions History (Current Step: {step})')
    ax_act.set_ylim([action_min, action_max])
    ax_act.legend(loc='upper right')
    ax_act.grid(True, linestyle='--', alpha=0.7)
    
    # Rewards chart (time series) - 있는 경우에만 표시
    if has_rewards:
        ax_rew = axes[2]
        
        # 직접 계산한 reward만 표시 (초록색)
        ax_rew.plot(timesteps, rewards_history[:step+1], 
                   label='Computed Reward', 
                   color='green', 
                   linewidth=2)
        
        # 현재 reward 값 텍스트로 표시
        current_reward = rewards_history[step]
        reward_text = f"Computed Reward: {current_reward:.3f}"
        ax_rew.text(0.02, 0.05, reward_text, transform=ax_rew.transAxes, 
                   fontsize=10, bbox=dict(facecolor='white', alpha=0.7))
        
        ax_rew.set_title(f'Computed Reward History (Current Step: {step})')
        ax_rew.set_ylim([reward_min, reward_max])
        ax_rew.legend(loc='upper right')
        ax_rew.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    
    # Convert graph image to NumPy array
    fig.canvas.draw()
    stats_img = np.array(fig.canvas.renderer.buffer_rgba())
    
    # Convert from RGBA to RGB
    stats_img = cv2.cvtColor(stats_img, cv2.COLOR_RGBA2RGB)
    
    plt.close(fig)
    
    return stats_img

def combine_frames(env_frame, stats_frame):
    """Merge environment frame and stats frame horizontally"""
    
    # Frames might have different heights, so adjust to the larger one
    h1, w1 = env_frame.shape[:2]
    h2, w2 = stats_frame.shape[:2]
    
    # Set height
    max_height = max(h1, h2)
    
    # Resize environment frame
    env_resized = cv2.resize(env_frame, (int(w1 * max_height / h1), max_height))
    
    # Resize stats frame
    stats_resized = cv2.resize(stats_frame, (int(w2 * max_height / h2), max_height))
    
    # Concatenate images horizontally
    combined = np.hstack((env_resized, stats_resized))
    
    return combined

def render_episode_with_stats(episode_idx, dataset, env, output_dir='rendered_episodes_with_stats'):
    """Render specific episode and save as video with stats"""
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Calculate episode segment
    terminals = dataset['terminals']
    start, end = get_episode_indices(terminals, episode_idx)
    print(f"Episode {episode_idx} - Start index: {start}, End index: {end}")
    observations = dataset['observations'][start:end]
    actions = dataset['actions'][start:end]
    
    # 데이터셋에서 qpos, qvel, button_states 직접 가져오기
    qpos_all = dataset.get('qpos', None)
    qvel_all = dataset.get('qvel', None)
    button_states_all = dataset.get('button_states', None)
    
    if qpos_all is not None and qvel_all is not None and button_states_all is not None:
        qpos_episode = qpos_all[start:end]
        qvel_episode = qvel_all[start:end]
        button_states_episode = button_states_all[start:end]
        print("데이터셋에서 qpos, qvel, button_states를 직접 로드했습니다.")
    else:
        print("데이터셋에 qpos, qvel, button_states가 없습니다. observations에서 추정합니다.")
        qpos_episode = None
        qvel_episode = None
        button_states_episode = None
    
    # Check dimensions of observations and actions
    obs_dim = observations.shape[1]
    action_dim = actions.shape[1]
    
    # Get dataset name from environment
    dataset_name = env.unwrapped.spec.id
    
    # Initialize video writer with dataset name in filename
    video_writer = imageio.get_writer(
        os.path.join(output_dir, f'{dataset_name}_episode_{episode_idx}_with_stats.mp4'),
        fps=30
    )
    
    # Reset the environment with the initial state from the dataset
    try:
        # 환경의 reset 메서드로 초기화
        obs, info = env.reset()
        
        # TimeLimit 래퍼를 제거하고 내부 환경에 접근
        inner_env = env.unwrapped
        
        # 환경의 model.nq와 model.nv 값 확인
        print(f"model.nq: {inner_env.model.nq}")
        print(f"model.nv: {inner_env.model.nv}")
        
        # 기본 reset을 먼저 수행하여 환경을 초기화
        _, info = inner_env.reset()
        
        # 데이터셋에서 직접 qpos, qvel, button_states 사용하기
        if qpos_episode is not None and qvel_episode is not None and button_states_episode is not None:
            # 첫 번째 타임스텝의 상태 사용
            qpos = qpos_episode[0]
            qvel = qvel_episode[0]
            button_states = button_states_episode[0]
            
            print("\n데이터셋에서 직접 가져온 초기 상태:")
            print(f"qpos: {qpos}")
            print(f"qvel: {qvel}")
            print(f"button_states: {button_states}")
        else:
            # 이전 방식: observations에서 qpos, qvel, button_states 추정
            # 데이터셋의 첫 번째 관측값에서 qpos, qvel, button_states 추출
            print(f"observations shape: {observations[0].shape}")
            print(f"observations[0]: {observations[0]}")
            
            # qpos, qvel, button_states 초기화
            qpos = np.zeros(inner_env.model.nq)
            qvel = np.zeros(inner_env.model.nv)
            button_states = np.zeros(3)
            
            # qpos 채우기 - 첫 번째 관측값에서 첫 model.nq 개의 값을 사용
            qpos_end = min(inner_env.model.nq, len(observations[0]))
            qpos[:qpos_end] = observations[0][:qpos_end]
            
            # qvel 채우기 - 기본적으로 관측값의 qpos 다음 부분이 qvel인 경우가 많음
            # model.nq 이후부터 model.nv 개의 값을 사용하려고 시도
            if len(observations[0]) > inner_env.model.nq:
                qvel_start = inner_env.model.nq
                qvel_end = min(qvel_start + inner_env.model.nv, len(observations[0]))
                available_values = qvel_end - qvel_start
                qvel[:available_values] = observations[0][qvel_start:qvel_end]
            
            # button_states는 일반적으로 버튼의 on/off 상태를 나타냄
            # SceneEnv 클래스에서는 각 버튼의 상태를 binary array로 저장
            # 예: [0, 1, 0]은 두 번째 버튼만 켜져 있음을 의미
            try:
                if 'info' in dataset and episode_idx in dataset['info'] and 'button_states' in dataset['info'][episode_idx]:
                    button_states = dataset['info'][episode_idx]['button_states']
                else:
                    # 데이터셋에 button_states가 없으면 기본값 사용
                    button_states = np.array([0, 0, 0])
            except:
                # 오류 발생 시 기본값 사용
                button_states = np.array([0, 0, 0])
        
        # 이미 올바른 shape를 가지고 있는지 확인
        print(f"qpos shape: {qpos.shape}")
        print(f"qvel shape: {qvel.shape}")
        print(f"button_states shape: {button_states.shape}")
        
        print("\n상세 분석:")
        print(f"qpos: {qpos}")
        print(f"qvel: {qvel}")
        print(f"button_states: {button_states}")
        
        # 환경 정보 출력
        print("\n환경 정보:")
        print(f"model.nq: {inner_env.model.nq}")
        print(f"model.nv: {inner_env.model.nv}")
        
        # 환경 상태 설정
        try:
            inner_env.set_state(qpos, qvel, button_states)
            print("set_state 성공!")
            
            # 새로운 관측값 가져오기
            try:
                obs = inner_env._get_obs()
            except:
                try:
                    obs = inner_env.get_obs()
                except:
                    obs = np.zeros(40)  # 관측값 가져오기 실패 시 임시 값
            print(f"새로운 관측값: {obs[:5]}...")
        except Exception as e:
            print(f"set_state 실패: {str(e)}")
            traceback.print_exc()  # 자세한 에러 추적 정보 출력
            
            # 실패하면 기본 reset 메서드 사용
            obs, info = env.reset()
            print(f"기본 reset 사용, 초기 관측값: {obs[:5]}...")
    except (TypeError, KeyError, AttributeError) as e:
        # set_state 메서드가 지원되지 않는 경우 기본 reset 사용
        obs, info = env.reset()
        print(f"Warning: Environment does not support state setting: {str(e)}")
        print("Initial observation from dataset:", observations[0])
        print("Initial observation from environment:", obs)
    
    # Play episode
    print(f"Episode {episode_idx} - Number of timesteps: {actions.shape[0]}")
    
    # computed reward를 저장할 배열
    computed_rewards = np.zeros(actions.shape[0])
    prev_obs = obs  # 이전 관측값 저장 (첫 번째 action 적용 전 상태)
    
    for t in range(actions.shape[0]):
        action = actions[t]
        obs, _, terminated, truncated, info = env.step(action)
        
        # computed reward 계산
        try:
            if hasattr(inner_env, '_compute_successes'):
                cube_successes, button_successes, drawer_success, window_success = inner_env._compute_successes()
                successes = cube_successes + button_successes + [drawer_success, window_success]
                computed_reward = float(sum(successes) - len(successes))
                computed_rewards[t] = computed_reward
                
                if t % 100 == 0:  # 로그 출력 빈도 조절
                    print(f"Step {t}: Computed Reward = {computed_reward}")
        except Exception as e:
            if t % 100 == 0:
                print(f"_compute_successes 호출 실패: {str(e)}")
        
        # 이전 관측값 업데이트
        prev_obs = obs
        
        # 데이터셋에서 직접 가져온 qpos, qvel, button_states가 있는 경우
        # 각 스텝마다 환경 상태를 설정할 수 있음
        if qpos_episode is not None and t+1 < len(qpos_episode):
            try:
                inner_env.set_state(qpos_episode[t+1], qvel_episode[t+1], button_states_episode[t+1])
            except Exception as e:
                # 오류 발생 시 무시하고 계속 진행 (기본 env.step 결과 사용)
                if t % 100 == 0:  # 너무 많은 오류 메시지 출력 방지
                    print(f"Step {t}: set_state 실패: {str(e)}")
        
        # Print progress every 100 timesteps
        if t % 100 == 0:
            print(f"Episode {episode_idx} - Progress: {t}/{actions.shape[0]} timesteps")
        
        # Render environment frame
        env_frame = env.render()
        
        # Add timestep information
        env_frame_cv = cv2.cvtColor(env_frame, cv2.COLOR_RGB2BGR)
        text = f"t: {t}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.7
        font_color = (255, 255, 255)  # White
        font_thickness = 2
        
        # Text position (top right)
        text_size = cv2.getTextSize(text, font, font_scale, font_thickness)[0]
        text_x = env_frame.shape[1] - text_size[0] - 10  # 10 pixel margin from right
        text_y = text_size[1] + 10  # 10 pixel margin from top
        
        # Text background (for better readability)
        cv2.rectangle(env_frame_cv, 
                    (text_x - 5, text_y - text_size[1] - 5), 
                    (text_x + text_size[0] + 5, text_y + 5), 
                    (0, 0, 0), -1)
        
        # Add text
        cv2.putText(env_frame_cv, text, (text_x, text_y), font, font_scale, 
                  font_color, font_thickness, cv2.LINE_AA)
        
        # Convert back to RGB
        env_frame = cv2.cvtColor(env_frame_cv, cv2.COLOR_BGR2RGB)
        
        # Create stats frame with history
        stats_frame = create_stats_figure(
            observations, 
            actions, 
            computed_rewards[:t+1],  # computed reward만 사용
            t,
            obs_dim,
            action_dim
        )
        
        # Merge frames
        combined_frame = combine_frames(env_frame, stats_frame)
        
        # Add frame to video
        video_writer.append_data(combined_frame)
        
        if terminated or truncated:
            print(f"Episode {episode_idx} ended: terminated={terminated}, truncated={truncated}")
            break
    
    # Close video writer
    video_writer.close()
    
    # Close environment
    env.close()
    print(f"Episode {episode_idx} video saved to {output_dir}")

def main():
    # Set up virtual display with Xvfb
    subprocess.run(['pkill', 'Xvfb'])
    subprocess.Popen(['Xvfb', ':100', '-screen', '0', '1024x768x24', '-ac'])
    os.environ['DISPLAY'] = ':100.0'
    os.environ['MUJOCO_GL'] = 'egl'
    os.environ['PYOPENGL_PLATFORM'] = 'egl'
    os.environ['EGL_DEVICE_ID'] = '0'
    time.sleep(2)  # Wait for Xvfb to start
    
    # Get settings from parser
    args = Parser().parse_args('diffusion')
    
    # Create environment and dataset with add_info=True 옵션 추가
    env, train_dataset, val_dataset = ogbench.make_env_and_datasets(
        args.dataset,
        render_mode='rgb_array',
        width=640,
        height=480,
        max_episode_steps=1000,
        add_info=True  # qpos, qvel, button_states 데이터 포함
    )
    
    # Check and modify max episode length
    print(f"Original max_episode_steps: {env._max_episode_steps}")
    env._max_episode_steps = 1000
    print(f"Modified max_episode_steps: {env._max_episode_steps}")
    
    # Print dataset structure
    print('train_dataset keys:', train_dataset.keys())
    for k in train_dataset:
        print(f"{k}: {type(train_dataset[k])}, shape: {getattr(train_dataset[k], 'shape', None)}")
    
    # Render a few episodes (e.g., first 3)
    num_episodes = 3
    for i in range(num_episodes):
        print(f"\nRendering: Episode {i} (with stats)")
        render_episode_with_stats(i, train_dataset, env)
    
    # Kill Xvfb process
    subprocess.run(['pkill', 'Xvfb'])

if __name__ == "__main__":
    main() 