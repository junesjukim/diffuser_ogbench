import sys
import os
import numpy as np

# diffuser_ogbench 경로를 Python path에 추가
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

from diffuser.datasets.ogbench import OGBenchGoalDataset

def analyze_dataset(dataset):
    """데이터셋의 구조와 통계를 분석하는 함수"""
    print("\n=== 데이터셋 기본 정보 ===")
    print(f"Observation 차원: {dataset.observation_dim}")
    print(f"Action 차원: {dataset.action_dim}")
    print(f"에피소드 수: {dataset.n_episodes}")
    print(f"최대 경로 길이: {dataset.max_path_length}")
    
    # 첫 번째 에피소드 분석
    first_episode = dataset.episodes[0]
    print("\n=== 첫 번째 에피소드 상세 정보 ===")
    print("Observations shape:", first_episode['observations'].shape)
    print("Actions shape:", first_episode['actions'].shape)
    print("Rewards shape:", first_episode['rewards'].shape)
    
    # Reward 통계
    rewards = first_episode['rewards']
    print("\nReward 통계:")
    print(f"평균: {np.mean(rewards):.4f}")
    print(f"최소: {np.min(rewards):.4f}")
    print(f"최대: {np.max(rewards):.4f}")
    print(f"표준편차: {np.std(rewards):.4f}")
    
    # Value 계산 예시
    print("\n=== Value 계산 예시 (처음 5개 샘플) ===")
    for i in range(5):
        value = dataset._compute_value(i)
        print(f"샘플 {i}의 value: {value:.4f}")
    
    # 전체 에피소드의 reward 통계
    all_rewards = []
    for episode in dataset.episodes:
        all_rewards.extend(episode['rewards'])
    all_rewards = np.array(all_rewards)
    
    print("\n=== 전체 데이터셋 Reward 통계 ===")
    print(f"평균: {np.mean(all_rewards):.4f}")
    print(f"최소: {np.min(all_rewards):.4f}")
    print(f"최대: {np.max(all_rewards):.4f}")
    print(f"표준편차: {np.std(all_rewards):.4f}")

    

def main():
    # 데이터셋 초기화
    dataset = OGBenchGoalDataset(
        env_name='scene-play-singletask-task2-v0',
        horizon=64,
        normalizer='LimitsNormalizer',
        use_padding=True,
        max_path_length=1000
    )
    
    # 데이터셋 분석
    analyze_dataset(dataset)

if __name__ == "__main__":
    main() 