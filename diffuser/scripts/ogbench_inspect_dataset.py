import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import ogbench

# diffuser_ogbench 경로를 Python path에 추가
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

from diffuser.datasets.ogbench import OGBenchGoalDataset

def inspect_env_and_dataset(env_name='scene-play-singletask-task2-v0'):
    """ogbench 환경과 데이터셋의 정보를 출력"""
    print(f"\n==== 환경 정보: {env_name} ====")
    
    # 환경과 데이터셋 로드
    env, train_dataset, val_dataset = ogbench.make_env_and_datasets(
        env_name,
        render_mode='rgb_array'
    )
    
    # 환경 정보
    print(f"환경 타입: {type(env)}")
    print(f"관측 공간: {env.observation_space}")
    print(f"행동 공간: {env.action_space}")
    
    # 데이터셋 정보
    print("\n==== 데이터셋 정보 ====")
    print("학습 데이터셋 키:", train_dataset.keys())
    
    for key in train_dataset:
        data = train_dataset[key]
        print(f"{key}: 타입={type(data)}, 형태={getattr(data, 'shape', None)}")
    
    # 에피소드 구조 분석
    terminals = train_dataset['terminals']
    terminal_indices = np.where(terminals)[0]
    episode_lengths = np.diff(np.concatenate([[-1], terminal_indices]))
    
    print(f"\n총 에피소드 수: {len(terminal_indices)}")
    print(f"평균 에피소드 길이: {np.mean(episode_lengths):.1f}")
    print(f"최소/최대 에피소드 길이: {np.min(episode_lengths)}/{np.max(episode_lengths)}")
    
    return env, train_dataset, val_dataset

def analyze_ogbench_dataset(dataset_obj):
    """OGBenchGoalDataset 객체의 데이터 처리 방식 분석"""
    print("\n==== OGBenchGoalDataset 분석 ====")
    print(f"전체 에피소드 수: {dataset_obj.n_episodes}")
    print(f"에피소드 샘플링을 위한 인덱스 수: {len(dataset_obj.indices)}")
    print(f"설정된 horizon: {dataset_obj.horizon}")
    
    # 정규화 방식
    print(f"\n정규화 방식: {type(dataset_obj.normalizer).__name__}")
    
    # 에피소드 구조
    first_episode = dataset_obj.episodes[0]
    print("\n첫 번째 에피소드 구조:")
    for key, value in first_episode.items():
        print(f"  {key}: 형태={getattr(value, 'shape', None)}")
    
    # 배치 샘플 확인
    batch = dataset_obj[0]
    print("\n배치 데이터 구조:")
    print(f"배치 타입: {type(batch).__name__}")
    print(f"궤적 형태: {batch.trajectories.shape}")
    print(f"조건 구조: {batch.conditions}")
    
    # Mask 분포 분석
    print("\n==== Mask 분포 분석 ====")
    total_masks = 0
    zero_masks = 0
    one_masks = 0
    
    for episode in dataset_obj.episodes:
        if 'masks' in episode:
            masks = episode['masks']
            total_masks += masks.size
            zero_masks += np.sum(masks == 0)
            one_masks += np.sum(masks == 1)
    
    print(f"전체 mask 수: {total_masks}")
    print(f"0의 개수: {zero_masks} ({zero_masks/total_masks*100:.2f}%)")
    print(f"1의 개수: {one_masks} ({one_masks/total_masks*100:.2f}%)")

    

def analyze_rendering_episode(train_dataset):
    """ogbench_render_episode.py에서 사용되는 에피소드 렌더링 방식 분석"""
    print("\n==== 렌더링 에피소드 분석 ====")
    
    # terminals 배열에서 에피소드 경계 찾기
    terminals = train_dataset['terminals']
    terminal_indices = np.where(terminals)[0]
    
    # 첫 번째 에피소드의 인덱스 범위 계산
    if len(terminal_indices) > 0:
        start = 0
        end = terminal_indices[0] + 1
        
        # 에피소드 데이터 확인
        observations = train_dataset['observations'][start:end]
        actions = train_dataset['actions'][start:end]
        rewards = train_dataset['rewards'][start:end]
        
        print(f"첫 번째 에피소드 범위: {start} ~ {end-1} (길이: {end-start})")
        print(f"관측 형태: {observations.shape}")
        print(f"행동 형태: {actions.shape}")
        print(f"보상 형태: {rewards.shape}")
        print(f"에피소드 종료 여부: {terminals[end-1]}")
        
        # 보상 통계
        print(f"총 보상: {np.sum(rewards):.4f}")
        print(f"평균 보상: {np.mean(rewards):.4f}")
    else:
        print("에피소드 경계를 찾을 수 없습니다.")

def analyze_visualization():
    """ogbench_visualize_dataset.py에서 시각화하는 데이터 분석"""
    print("\n==== 시각화 데이터 분석 ====")
    
    # OGBenchGoalDataset 객체 생성
    dataset = OGBenchGoalDataset(
        env_name='scene-play-singletask-task2-v0',
        horizon=64,
        normalizer='LimitsNormalizer',
        use_padding=True,
        max_path_length=1000
    )
    
    # 첫 번째 에피소드 확인
    first_episode = dataset.episodes[0]
    
    print("시각화되는 데이터: dataset.episodes[0] (첫 번째 에피소드)")
    print(f"관측 형태: {first_episode['observations'].shape}")
    print(f"행동 형태: {first_episode['actions'].shape}")
    print(f"보상 형태: {first_episode['rewards'].shape}")
    
    # 시각화 내용 설명
    print("\n시각화 내용:")
    print("1. 관측값 (처음 3개 차원)")
    print("2. 행동 (처음 3개 차원)")
    print("3. 보상")
    
    return first_episode

def inspect_dataset_info(train_dataset, episode_idx=0):
    """데이터셋의 info 필드에서 qpos, qvel 등의 정보 확인"""
    print("\n==== Dataset Info 필드 분석 ====")
    
    # info 키가 있는지 확인
    if 'info' in train_dataset:
        print("Info 키 존재함")
        
        # info의 구조 확인
        info = train_dataset['info']
        print(f"Info 타입: {type(info)}")
        
        # info가 딕셔너리인 경우 키 확인
        if isinstance(info, dict):
            print(f"Info 키: {list(info.keys())}")
            
            # 특정 에피소드 정보 확인
            if episode_idx in info:
                episode_info = info[episode_idx]
                print(f"\n에피소드 {episode_idx}의 info 키: {list(episode_info.keys())}")
                
                # qpos, qvel 확인
                if 'qpos' in episode_info:
                    qpos = episode_info['qpos']
                    print(f"qpos 존재함: 형태={getattr(qpos, 'shape', None)}")
                    print(f"qpos 샘플: {qpos[:5] if hasattr(qpos, '__getitem__') else qpos}")
                else:
                    print("qpos 키가 없음")
                
                if 'qvel' in episode_info:
                    qvel = episode_info['qvel']
                    print(f"qvel 존재함: 형태={getattr(qvel, 'shape', None)}")
                    print(f"qvel 샘플: {qvel[:5] if hasattr(qvel, '__getitem__') else qvel}")
                else:
                    print("qvel 키가 없음")
                
                # button_states 확인
                if 'button_states' in episode_info:
                    button_states = episode_info['button_states']
                    print(f"button_states 존재함: {button_states}")
                else:
                    print("button_states 키가 없음")
                
                return episode_info
            else:
                print(f"에피소드 {episode_idx}에 대한 정보 없음")
        else:
            print("Info가 딕셔너리 형태가 아님")
    else:
        print("Info 키가 없음")
        
        # 대안적으로 terminals를 이용한 에피소드 분석
        if 'terminals' in train_dataset and 'observations' in train_dataset:
            print("\n대안: 관측값으로부터 qpos/qvel 추정")
            terminals = train_dataset['terminals']
            observations = train_dataset['observations']
            
            # 에피소드 경계 찾기
            terminal_indices = np.where(terminals)[0]
            if len(terminal_indices) > episode_idx:
                start = 0 if episode_idx == 0 else terminal_indices[episode_idx-1] + 1
                end = terminal_indices[episode_idx]
                
                # 첫 번째 관측값 확인
                first_obs = observations[start]
                print(f"에피소드 {episode_idx}의 첫 번째 관측값: {first_obs[:10]}...")  # 처음 10개 요소만 출력
                print(f"관측값 크기: {first_obs.shape}")
                
                print("\n관측값에서 qpos와 qvel을 추출하려면 모델의 구조를 알아야 합니다.")
                print("가능한 추정 방법:")
                print("1. 전체 관측값이 [qpos, qvel, ...] 순서로 구성된 경우")
                print("2. SceneEnv에서는 관측값의 초반 부분이 관절 위치와 속도를 포함")
                
                # 힌트 제공
                print("\n코드에 다음과 같은 함수를 추가하여 실험해보세요:")
                print("""
def estimate_qpos_qvel(observation, model_nq=None, model_nv=None):
    \"\"\"관측값에서 qpos와 qvel을 추정하는 시도\"\"\"
    # 모델 차원을 모른다면 관측값 크기를 기반으로 추정
    if model_nq is None:
        model_nq = len(observation) // 3  # 관측값의 1/3이 qpos라고 가정
    if model_nv is None:
        model_nv = len(observation) // 3  # 관측값의 1/3이 qvel이라고 가정
    
    # 추정된 qpos와 qvel
    qpos_est = observation[:model_nq]
    qvel_est = observation[model_nq:model_nq+model_nv]
    
    return qpos_est, qvel_est
                """)
    
    return None

def main():
    # 1. 환경 및 데이터셋 정보 확인
    env, train_dataset, val_dataset = inspect_env_and_dataset()
    
    # 2. OGBenchGoalDataset 분석
    dataset = OGBenchGoalDataset(
        env_name='scene-play-singletask-task2-v0',
        horizon=64,
        normalizer='LimitsNormalizer',
        use_padding=True,
        max_path_length=1000
    )
    analyze_ogbench_dataset(dataset)
    
    # 3. 렌더링 에피소드 분석
    analyze_rendering_episode(train_dataset)
    
    # 4. 시각화 데이터 분석
    analyze_visualization()
    
    # 5. 데이터셋의 info 필드 분석 (추가된 부분)
    episode_info = inspect_dataset_info(train_dataset)
    
    # 6. 만약 info 필드에서 정보를 찾지 못했다면 환경에서 직접 확인
    if episode_info is None and env is not None:
        print("\n==== 환경에서 직접 상태 정보 확인 ====")
        try:
            # 환경을 리셋하고 내부 상태에 접근
            obs, _ = env.reset()
            inner_env = env.unwrapped
            
            # qpos와 qvel 크기 확인
            if hasattr(inner_env, 'model'):
                print(f"model.nq: {inner_env.model.nq}")
                print(f"model.nv: {inner_env.model.nv}")
                
                # 샘플 관측값에서 qpos와 qvel 추출 시도
                if hasattr(train_dataset, 'observations') and len(train_dataset['observations']) > 0:
                    first_obs = train_dataset['observations'][0]
                    print("\n첫 번째 관측값에서 qpos와 qvel 추정:")
                    qpos_est, qvel_est = estimate_qpos_qvel(first_obs, inner_env.model.nq, inner_env.model.nv)
                    print(f"추정된 qpos: {qpos_est}")
                    print(f"추정된 qvel: {qvel_est}")
            else:
                print("환경에 model 속성이 없습니다")
        except Exception as e:
            print(f"환경에서 정보 확인 중 오류 발생: {str(e)}")

if __name__ == "__main__":
    main()