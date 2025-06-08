import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# diffuser_ogbench 경로를 Python path에 추가
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

from diffuser.datasets.ogbench import OGBenchGoalDataset

def visualize_episode(episode, save_path=None):
    """에피소드의 관측값을 시각화하는 함수"""
    observations = episode['observations']
    actions = episode['actions']
    rewards = episode['rewards']
    # masks가 있는지 확인
    has_masks = 'masks' in episode
    masks = episode.get('masks', None)
    
    # 관측값의 차원 확인
    obs_dim = observations.shape[1]
    action_dim = actions.shape[1]
    
    # 시각화를 위한 figure 생성 (masks가 있으면 subplot 개수 증가)
    num_subplots = 4 if has_masks else 3
    fig, axes = plt.subplots(num_subplots, 1, figsize=(12, 3 * num_subplots))
    
    # 관측값을 표시할 subplot
    obs_lines = []
    for i in range(min(3, obs_dim)):  # 처음 3개의 관측값 차원만 표시
        line, = axes[0].plot([], [], label=f'Obs {i}')
        obs_lines.append(line)
    axes[0].set_xlim(0, len(observations))
    axes[0].set_ylim(observations.min(), observations.max())
    axes[0].legend()
    axes[0].set_title('Observations')
    axes[0].set_xlabel('Steps')
    
    # 행동을 표시할 subplot
    action_lines = []
    for i in range(min(3, action_dim)):  # 처음 3개의 행동 차원만 표시
        line, = axes[1].plot([], [], label=f'Action {i}')
        action_lines.append(line)
    axes[1].set_xlim(0, len(actions))
    axes[1].set_ylim(actions.min(), actions.max())
    axes[1].legend()
    axes[1].set_title('Actions')
    axes[1].set_xlabel('Steps')
    
    # 보상을 표시할 subplot
    reward_line, = axes[2].plot([], [], label='Reward')
    axes[2].set_xlim(0, len(rewards))
    axes[2].set_ylim(rewards.min(), rewards.max())
    axes[2].legend()
    axes[2].set_title('Rewards')
    axes[2].set_xlabel('Steps')
    
    # masks가 있으면 masks를 표시할 subplot 추가
    mask_line = None
    if has_masks:
        mask_line, = axes[3].plot([], [], label='Mask')
        axes[3].set_xlim(0, len(masks))
        axes[3].set_ylim(0, 1.1)  # masks는 보통 0 또는 1
        axes[3].legend()
        axes[3].set_title('Masks')
        axes[3].set_xlabel('Steps')
    
    def init():
        for line in obs_lines:
            line.set_data([], [])
        for line in action_lines:
            line.set_data([], [])
        reward_line.set_data([], [])
        if mask_line:
            mask_line.set_data([], [])
        return obs_lines + action_lines + [reward_line] + ([mask_line] if mask_line else [])
    
    def animate(i):
        # 관측값 업데이트
        for j, line in enumerate(obs_lines):
            line.set_data(range(i+1), observations[:i+1, j])
        
        # 행동 업데이트
        for j, line in enumerate(action_lines):
            line.set_data(range(i+1), actions[:i+1, j])
        
        # 보상 업데이트
        reward_line.set_data(range(i+1), rewards[:i+1])
        
        # masks 업데이트
        if mask_line:
            mask_line.set_data(range(i+1), masks[:i+1])
        
        return obs_lines + action_lines + [reward_line] + ([mask_line] if mask_line else [])
    
    # 애니메이션 생성
    anim = FuncAnimation(fig, animate, init_func=init, frames=len(observations),
                        interval=50, blit=True)
    
    if save_path:
        anim.save(save_path, writer='pillow')
    else:
        plt.show()
    
    plt.close()

def main():
    # 데이터셋 초기화
    dataset = OGBenchGoalDataset(
        env_name='scene-play-singletask-task2-v0',
        horizon=64,
        normalizer='LimitsNormalizer',
        use_padding=True,
        max_path_length=1000
    )
    
    # 첫 번째 에피소드 시각화
    first_episode = dataset.episodes[0]
    print("\n=== 첫 번째 에피소드 시각화 ===")
    print("Observations shape:", first_episode['observations'].shape)
    print("Actions shape:", first_episode['actions'].shape)
    print("Rewards shape:", first_episode['rewards'].shape)
    if 'masks' in first_episode:
        print("Masks shape:", first_episode['masks'].shape)
    
    # 시각화 실행
    visualize_episode(first_episode, save_path='episode_visualization.gif')

if __name__ == "__main__":
    main() 