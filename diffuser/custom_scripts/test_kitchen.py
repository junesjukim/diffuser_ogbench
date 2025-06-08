import d4rl
import gym

print("테스트 시작")
env = gym.make('pen-cloned-v0')
dataset = env.get_dataset()

# 환경 정보 출력
print(f"최대 타임스텝: {env._max_episode_steps}")
print(f"관측 공간: {env.observation_space}")
print(f"행동 공간: {env.action_space}")

# 데이터셋 정보
print("\n데이터셋 정보:")
print(f"총 타임스텝 수: {len(dataset['observations'])}")
print(f"터미널 상태 수: {dataset['terminals'].sum()}")  # 에피소드 종료 횟수

# 리워드 정보 출력
print("\n리워드 정보:")
print(f"총 리워드 합계: {dataset['rewards'].sum()}")
print(f"평균 리워드: {dataset['rewards'].mean()}")
print(f"최대 리워드: {dataset['rewards'].max()}")
print(f"최소 리워드: {dataset['rewards'].min()}")
print(f"리워드 표준편차: {dataset['rewards'].std()}")

# 에피소드별 리워드 분석
# print("\n에피소드별 리워드 분석:")
# episode_rewards = []
# current_episode_reward = 0

# for i in range(len(dataset['rewards'])):
#     current_episode_reward += dataset['rewards'][i]
    
#     if dataset['terminals'][i]:
#         episode_rewards.append(current_episode_reward)
#         current_episode_reward = 0
        
#         # 마지막 에피소드의 리워드 변화 상세 출력 (예시로 마지막 에피소드만)
#         if len(episode_rewards) == len(episode_rewards):
#             print(f"\n마지막 에피소드의 리워드 변화:")
#             start_idx = i - sum(dataset['rewards'][j] != 0 for j in range(i, -1, -1) if not dataset['terminals'][j])
#             for step in range(start_idx, i+1):
#                 if dataset['rewards'][step] != 0:
#                     print(f"스텝 {step-start_idx}: {dataset['rewards'][step]:.3f}")

# print(f"\n총 에피소드 수: {len(episode_rewards)}")
# print(f"에피소드당 평균 리워드: {sum(episode_rewards)/len(episode_rewards):.3f}")
# print(f"최고 에피소드 리워드: {max(episode_rewards):.3f}")
# print(f"최저 에피소드 리워드: {min(episode_rewards):.3f}")

print("환경 로드 완료")