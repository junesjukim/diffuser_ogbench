from collections import namedtuple
import numpy as np
import torch
from .sequence import SequenceDataset, Batch, ValueBatch
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
        seed=None,
        discount=0.99,  # value 계산을 위한 할인율
        normed=False    # value 정규화 여부
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
        
        # value 관련 설정
        self.discount = discount
        self.discounts = self.discount ** np.arange(self.max_path_length)[:,None]
        self.normed = False
        if normed:
            self.vmin, self.vmax = self._get_bounds()
            self.normed = True
        
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
    
    def _get_bounds(self):
        """value의 최소/최대값 계산"""
        print('[ datasets/sequence ] Getting value dataset bounds...', end=' ', flush=True)
        vmin = np.inf
        vmax = -np.inf
        
        total = len(self.indices)
        for i in range(total):
            if i % 100 == 0:  # 100개마다 진행상황 출력
                print(f'\r[ datasets/sequence ] Getting value dataset bounds... {i}/{total}', end='', flush=True)
            
            # _compute_value를 사용하여 value 계산
            value = self._compute_value(i)
            vmin = min(value, vmin)
            vmax = max(value, vmax)
        
        print(f'\r[ datasets/sequence ] Getting value dataset bounds... {total}/{total} ✓')
        return vmin, vmax

    def normalize_value(self, value):
        """value 정규화"""
        ## [0, 1]
        normed = (value - self.vmin) / (self.vmax - self.vmin)
        ## [-1, 1]
        normed = normed * 2 - 1
        return normed

    def _compute_value(self, idx):
        """raw value 계산 (정규화 없음)"""
        path_ind, start, end = self.indices[idx]
        
        # Get observations and actions for the trajectory
        observations = self.fields.observations[path_ind, start:end]
        actions = self.fields.actions[path_ind, start:end]
        
        # Calculate Q-values and V-values
        with torch.no_grad():
            obs_tensor = torch.FloatTensor(observations).to(self.device)
            act_tensor = torch.FloatTensor(actions).to(self.device)
            
            # Q-network는 두 개의 Q-값을 반환하므로 min을 사용
            q1, q2 = self.q_network(obs_tensor, act_tensor)
            q_values = torch.min(q1, q2)
            
            # V-network는 상태에 대한 가치를 반환
            v_values = self.v_network(obs_tensor)
            
            # Calculate advantages
            advantages = q_values - v_values
            
            # Calculate discounted advantage sum
            discounts = torch.FloatTensor(self.discounts[:len(advantages)]).to(self.device)
            advantage_sum = (discounts * advantages).sum()
            
            return advantage_sum.cpu().item()
    
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx, eps=1e-4):
        path_ind, start, end = self.indices[idx]
        
        observations = self.fields.normed_observations[path_ind, start:end]
        actions = self.fields.normed_actions[path_ind, start:end]
        
        conditions = self.get_conditions(observations)
        trajectories = np.concatenate([actions, observations], axis=-1)
        
        # value 계산
        value = self._compute_value(idx)
        value = np.array([value], dtype=np.float32)
        
        # 기존 ogbench_train.py와의 호환성을 위해 Batch 반환
        if not hasattr(self, 'is_value_training') or not self.is_value_training:
            return Batch(trajectories, conditions)
        # value 학습 시에는 ValueBatch 반환
        else:
            return ValueBatch(trajectories, conditions, value)

class OGBenchValueDataset(OGBenchGoalDataset):
    """OGBenchGoalDataset을 상속하여 value 계산 기능을 추가한 데이터셋"""
    
    def __init__(
        self,
        env_name,
        horizon,
        normalizer,
        preprocess_fns,
        max_path_length,
        max_n_episodes,
        termination_penalty,
        use_padding,
        seed=None,
        discount=0.99,
        normed=False,
        q_network=None,
        v_network=None,
        device='cuda'
    ):
        super().__init__(
            env_name=env_name,
            horizon=horizon,
            normalizer=normalizer,
            preprocess_fns=preprocess_fns,
            max_path_length=max_path_length,
            max_n_episodes=max_n_episodes,
            termination_penalty=termination_penalty,
            use_padding=use_padding,
            seed=seed,
            discount=discount,
            normed=False  # 부모 클래스에서는 normed를 False로 설정
        )
        
        self.q_network = q_network
        self.v_network = v_network
        self.device = device
        self.normed = normed
        self.vmin = None
        self.vmax = None
    
    def to(self, device):
        """데이터셋을 지정된 device로 이동"""
        self.device = device
        if self.q_network is not None:
            self.q_network = self.q_network.to(device)
        if self.v_network is not None:
            self.v_network = self.v_network.to(device)
        return self
    
    def _compute_value(self, idx):
        """raw value 계산 (정규화 없음)"""
        path_ind, start, end = self.indices[idx]
        
        # Get observations and actions for the trajectory
        observations = self.fields.observations[path_ind, start:end]
        actions = self.fields.actions[path_ind, start:end]
        
        # Calculate Q-values and V-values
        with torch.no_grad():
            obs_tensor = torch.FloatTensor(observations).to(self.device)
            act_tensor = torch.FloatTensor(actions).to(self.device)
            
            # Q-network는 두 개의 Q-값을 반환하므로 min을 사용
            q1, q2 = self.q_network(obs_tensor, act_tensor)
            q_values = torch.min(q1, q2)
            
            # V-network는 상태에 대한 가치를 반환
            v_values = self.v_network(obs_tensor)
            
            # Calculate advantages
            advantages = q_values - v_values
            
            # Calculate discounted advantage sum
            discounts = torch.FloatTensor(self.discounts[:len(advantages)]).to(self.device)
            advantage_sum = (discounts * advantages).sum()
            
            return advantage_sum.cpu().item()
    
    def __getitem__(self, idx):
        batch = super().__getitem__(idx)
        
        # value 계산
        value = self._compute_value(idx)
        
        # 정규화가 필요한 경우에만 정규화
        if self.normed:
            # bounds가 계산되지 않은 경우에만 계산
            if self.vmin is None or self.vmax is None:
                print('[ datasets/sequence ] Getting value dataset bounds...', end=' ', flush=True)
                values = []
                total = len(self.indices)
                for i in range(total):
                    if i % 100 == 0:  # 100개마다 진행상황 출력
                        print(f'\r[ datasets/sequence ] Getting value dataset bounds... {i}/{total}', end='', flush=True)
                    values.append(self._compute_value(i))
                
                self.vmin = np.min(values)
                self.vmax = np.max(values)
                print(f'\r[ datasets/sequence ] Getting value dataset bounds... {total}/{total} ✓')
            
            value = self.normalize_value(value)
        
        value = np.array([value], dtype=np.float32)
        value_batch = ValueBatch(*batch, value)
        return value_batch 