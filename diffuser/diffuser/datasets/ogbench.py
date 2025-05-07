import numpy as np
import torch
from .sequence import SequenceDataset, Batch
from .normalization import DatasetNormalizer
from .buffer import ReplayBuffer

class OGBenchGoalDataset(SequenceDataset):
    def __init__(
        self,
        env_name='scene-play-singletask-task2-v0',
        horizon=64,
        normalizer='LimitsNormalizer',
        preprocess_fns=[],
        max_path_length=1000,
        max_n_episodes=10000,
        termination_penalty=0,
        use_padding=True,
        seed=None
    ):
        # OGBench 환경 로드
        import ogbench
        self.env, self.train_dataset, self.val_dataset = ogbench.make_env_and_datasets(
            env_name,
            render_mode='rgb_array'
        )
        
        # 데이터셋 정보
        self.observation_dim = self.train_dataset['observations'].shape[-1]  # 40
        self.action_dim = self.train_dataset['actions'].shape[-1]  # 5
        self.max_path_length = max_path_length
        self.horizon = horizon
        self.use_padding = use_padding
        
        # 에피소드 분할
        self.episodes = self._split_episodes()
        
        # 버퍼 초기화
        self.fields = ReplayBuffer(
            max_n_episodes,
            max_path_length,
            termination_penalty
        )
        
        # 에피소드 데이터 추가
        for episode in self.episodes:
            self.fields.add_path(episode)
        self.fields.finalize()
        
        # 정규화 설정
        self.normalizer = DatasetNormalizer(
            self.fields,
            normalizer,
            path_lengths=self.fields['path_lengths']
        )
        
        # 인덱스 생성
        self.indices = self.make_indices(
            self.fields.path_lengths,
            horizon
        )
        
        # 필요한 속성들 추가
        self.n_episodes = self.fields.n_episodes
        self.path_lengths = self.fields.path_lengths
        
        # 정규화 수행
        self.normalize()
        
        print(self.fields)
        
    def _split_episodes(self):
        """
        연속된 데이터를 에피소드 단위로 분할
        """
        episodes = []
        terminals = self.train_dataset['terminals']
        start_idx = 0
        
        for i in range(len(terminals)):
            if terminals[i]:
                episode = {
                    'observations': self.train_dataset['observations'][start_idx:i+1],
                    'actions': self.train_dataset['actions'][start_idx:i+1],
                    'rewards': self.train_dataset['rewards'][start_idx:i+1],
                    'terminals': self.train_dataset['terminals'][start_idx:i+1],
                    'next_observations': self.train_dataset['next_observations'][start_idx:i+1],
                    'masks': self.train_dataset['masks'][start_idx:i+1],
                    'timeouts': np.zeros_like(self.train_dataset['terminals'][start_idx:i+1])
                }
                episodes.append(episode)
                start_idx = i + 1
                
        return episodes
    
    def get_conditions(self, observations):
        """
        목표 상태를 조건으로 추가
        """
        return {
            0: observations[0],  # 현재 상태
            self.horizon - 1: observations[-1]  # 마지막 상태를 목표로 사용
        }
    
    def normalize(self, keys=['observations', 'actions']):
        """
        데이터 정규화
        """
        for key in keys:
            array = self.fields[key].reshape(self.n_episodes*self.max_path_length, -1)
            normed = self.normalizer(array, key)
            self.fields[f'normed_{key}'] = normed.reshape(self.n_episodes, self.max_path_length, -1)
    
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx, eps=1e-4):
        path_ind, start, end = self.indices[idx]
        
        observations = self.fields.normed_observations[path_ind, start:end]
        actions = self.fields.normed_actions[path_ind, start:end]
        
        conditions = self.get_conditions(observations)
        trajectories = np.concatenate([actions, observations], axis=-1)
        
        return Batch(trajectories, conditions) 