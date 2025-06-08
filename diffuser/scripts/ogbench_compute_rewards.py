import sys
import os
import numpy as np
import ogbench
import gymnasium as gym
from tqdm import tqdm

# diffuser_ogbench 경로를 Python path에 추가
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

def compute_rewards_for_dataset(dataset_path, task_id):
    """
    데이터셋의 reward를 계산하는 함수
    
    Args:
        dataset_path: npz 파일 경로
        task_id: task 번호 (1-5)
    """
    print(f"\n=== Computing rewards for Task {task_id} ===")
    
    # 환경 초기화
    env_name = f'scene-singletask-task{task_id}'
    env = gym.make(env_name, render_mode='rgb_array')
    
    # 내부 환경에 접근
    inner_env = env.unwrapped
    
    # 데이터셋 로드
    data = np.load(dataset_path)
    
    # 에피소드 경계 찾기
    terminals = data['terminals']
    terminal_indices = np.where(terminals)[0]
    
    # 모든 에피소드에 대해 reward 계산
    all_rewards = []
    all_observations = []
    all_actions = []
    all_qpos = []
    all_qvel = []
    all_button_states = []
    
    start = 0
    for episode_idx, end in enumerate(tqdm(terminal_indices, desc=f'Task {task_id} Episodes')):
        # 에피소드 데이터 추출
        observations = data['observations'][start:end+1]
        actions = data['actions'][start:end+1]
        qpos = data['qpos'][start:end+1]
        qvel = data['qvel'][start:end+1]
        button_states = data['button_states'][start:end+1]
        
        # reward 계산
        rewards = np.zeros(len(observations))
        for t in range(len(observations)):
            try:
                # 환경 상태 설정
                inner_env.set_state(qpos[t], qvel[t], button_states[t])
                # 성공 여부 계산
                if hasattr(inner_env, '_compute_successes'):
                    cube_successes, button_successes, drawer_success, window_success = inner_env._compute_successes()
                    successes = cube_successes + button_successes + [drawer_success, window_success]
                    rewards[t] = float(sum(successes) - len(successes))
                    
                    # if t % 1000 == 0:  # 로그 출력 빈도 조절
                    #     print(f"\nEpisode {episode_idx}, Step {t}:")
                    #     print(f"  Cube successes: {cube_successes}")
                    #     print(f"  Button successes: {button_successes}")
                    #     print(f"  Drawer success: {drawer_success}")
                    #     print(f"  Window success: {window_success}")
                    #     print(f"  Total reward: {rewards[t]:.3f}")
                else:
                    print(f"Step {t}: _compute_successes 메소드가 없습니다.")
                    rewards[t] = 0.0
            except Exception as e:
                print(f"Step {t}: reward 계산 실패: {str(e)}")
                rewards[t] = 0.0
        
        # 에피소드 데이터 저장
        all_rewards.extend(rewards)
        all_observations.extend(observations)
        all_actions.extend(actions)
        all_qpos.extend(qpos)
        all_qvel.extend(qvel)
        all_button_states.extend(button_states)
        
        start = end + 1
    
    # 결과를 numpy 배열로 변환
    all_rewards = np.array(all_rewards)
    all_observations = np.array(all_observations)
    all_actions = np.array(all_actions)
    all_qpos = np.array(all_qpos)
    all_qvel = np.array(all_qvel)
    all_button_states = np.array(all_button_states)
    
    # 결과 저장
    output_path = f'rewards_task{task_id}.npz'
    np.savez(
        output_path,
        rewards=all_rewards,
        observations=all_observations,
        actions=all_actions,
        qpos=all_qpos,
        qvel=all_qvel,
        button_states=all_button_states
    )
    print(f"\nRewards saved to {output_path}")
    print(f"Reward statistics:")
    print(f"  Mean: {np.mean(all_rewards):.4f}")
    print(f"  Min: {np.min(all_rewards):.4f}")
    print(f"  Max: {np.max(all_rewards):.4f}")
    print(f"  Std: {np.std(all_rewards):.4f}")

if __name__ == "__main__":
    # 데이터셋 경로
    dataset_path = './iql-pytorch/data/ogbench/scene-play-v0.npz'
    
    # 각 task에 대해 reward 계산 (1-5)
    for task_id in range(1, 6):  # 1부터 5까지
        compute_rewards_for_dataset(dataset_path, task_id) 