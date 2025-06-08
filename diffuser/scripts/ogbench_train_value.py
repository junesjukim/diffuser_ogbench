import sys
import os

# diffuser_ogbench 경로를 Python path에 추가
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

import diffuser.utils as utils
import torch
import ogbench
from diffuser.models.diffusion import set_model_mode
from diffuser.datasets.ogbench import OGBenchValueDataset
from torch.utils.data import DataLoader

from iql_pytorch.critic import Critic, ValueCritic

class Parser(utils.Parser):
    dataset: str = 'scene-play-singletask-task1-v0'
    config: str = 'config.locomotion'
    horizon: int = 64
    n_diffusion_steps: int = 100
    batch_size: int = 32
    learning_rate: float = 2e-4
    normalizer: str = 'LimitsNormalizer'
    use_padding: bool = True
    max_path_length: int = 1000
    q_path: str = '/home/junseolee/intern/diffuser_ogbench/iql_pytorch/runs/ogbench_runs/task1/test-task1-06-08-16-31-bs256-s0-t3.0-e0.7/model/critic_s700000.pth'
    v_path: str = '/home/junseolee/intern/diffuser_ogbench/iql_pytorch/runs/ogbench_runs/task1/test-task1-06-08-16-31-bs256-s0-t3.0-e0.7/model/value_s700000.pth'

args = Parser().parse_args('values')
set_model_mode(args.prefix)

#-----------------------------------------------------------------------------#
#---------------------------------- dataset ----------------------------------#
#-----------------------------------------------------------------------------#

# ogbench 환경 로드
env, _, _ = ogbench.make_env_and_datasets(
    args.dataset,
    render_mode='rgb_array'
)
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]

# Q와 V network 로드
q_network = Critic(state_dim, action_dim).to(args.device)
v_network = ValueCritic(state_dim, 256, 3).to(args.device)

# state_dict 로드
q_network.load_state_dict(torch.load(args.q_path, map_location=args.device))
v_network.load_state_dict(torch.load(args.v_path, map_location=args.device))

# 평가 모드로 설정
q_network.eval()
v_network.eval()

dataset_config = utils.Config(
    OGBenchValueDataset,
    savepath=(args.savepath, 'dataset_config.pkl'),
    env_name=args.dataset,
    horizon=args.horizon,
    normalizer=args.normalizer,
    preprocess_fns=args.preprocess_fns,
    use_padding=args.use_padding,
    max_path_length=args.max_path_length,
    max_n_episodes=1000,  # 최대 에피소드 수
    termination_penalty=0,  # 종료 페널티
    q_network=q_network,
    v_network=v_network,
    discount=0.99,
    normed=False,
    device=args.device
)

dataset = dataset_config()

observation_dim = dataset.observation_dim
action_dim = dataset.action_dim

#-----------------------------------------------------------------------------#
#------------------------------ model & trainer ------------------------------#
#-----------------------------------------------------------------------------#

# Value model 설정
value_model_config = utils.Config(
    args.model,
    savepath=(args.savepath, 'value_model_config.pkl'),
    horizon=args.horizon,
    transition_dim=observation_dim + action_dim,
    cond_dim=observation_dim,  # 현재 상태만을 조건으로 사용
    dim_mults=args.dim_mults,
    device=args.device,
)

# Value diffusion 설정
value_diffusion_config = utils.Config(
    args.diffusion,
    savepath=(args.savepath, 'value_diffusion_config.pkl'),
    horizon=args.horizon,
    observation_dim=observation_dim,
    action_dim=action_dim,
    n_timesteps=args.n_diffusion_steps,
    loss_type=args.loss_type,
    device=args.device,
)

# Value trainer 설정
value_trainer_config = utils.Config(
    utils.Trainer,
    savepath=(args.savepath, 'value_trainer_config.pkl'),
    train_batch_size=args.batch_size,
    gradient_accumulate_every=args.gradient_accumulate_every,
    ema_decay=args.ema_decay,
    sample_freq=args.sample_freq,
    save_freq=args.save_freq,
    label_freq=int(args.n_train_steps // args.n_saves),
    save_parallel=args.save_parallel,
    results_folder=args.savepath,
    bucket=args.bucket,
    n_reference=args.n_reference,
)

# Value 모델 초기화
value_model = value_model_config()
value_diffusion = value_diffusion_config(value_model)
value_trainer = value_trainer_config(value_diffusion, dataset, None)

# Forward pass 테스트
print('Testing forward...', end=' ', flush=True)
batch = utils.batchify(dataset[0])
loss, _ = value_diffusion.loss(*batch)
loss.backward()
print('✓')

# 학습 시작
n_epochs = int(args.n_train_steps // args.n_steps_per_epoch)
for i in range(n_epochs):
    print(f'Epoch {i} / {n_epochs} | {args.savepath}')
    value_trainer.train(n_train_steps=args.n_steps_per_epoch)

# Create dataloader
dataloader = DataLoader(
    dataset,
    batch_size=args.batch_size,
    shuffle=True,
    num_workers=4,
    multiprocessing_context='spawn'  # CUDA 초기화 문제 해결을 위해 추가
)