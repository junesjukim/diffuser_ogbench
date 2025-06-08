import diffuser.utils as utils
import pdb
import torch
import torch.nn as nn

from diffuser.models.diffusion import set_model_mode
from diffuser.datasets.d4rl import load_environment
#-----------------------------------------------------------------------------#
#-------------------------- conda env test -----------------------------------#
#-----------------------------------------------------------------------------#

import os

conda_env = os.environ.get("CONDA_DEFAULT_ENV", "Conda environment not detected")
print("Active conda environment:", conda_env)

#-----------------------------------------------------------------------------#
#----------------------------------- setup -----------------------------------#
#-----------------------------------------------------------------------------#

class Parser(utils.Parser):
    dataset: str = 'walker2d-medium-replay-v2'
    config: str = 'config.locomotion'
    q_path: str = '/home/junseolee/intern/diffuser_ogbench/iql-pytorch/runs/ogbench_runs/task1/test-task1-06-08-16-31-bs256-s0-t3.0-e0.7/model/critic_s470000.pth'
    v_path: str = '/home/junseolee/intern/diffuser_ogbench/iql-pytorch/runs/ogbench_runs/task1/test-task1-06-08-16-31-bs256-s0-t3.0-e0.7/model/value_s470000.pth'

args = Parser().parse_args('values')
set_model_mode(args.prefix)

#-----------------------------------------------------------------------------#
#---------------------------------- dataset ----------------------------------#
#-----------------------------------------------------------------------------#

# 모델 구조 정의
from iql_pytorch.critic import Critic, ValueCritic

# 환경 정보 가져오기
env = load_environment(args.dataset)
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]

# 모델 구조 정의
q_network = Critic(state_dim, action_dim).to(args.device)
v_network = ValueCritic(state_dim, 256, 3).to(args.device)

# state_dict 로드
q_network.load_state_dict(torch.load(args.q_path, map_location=args.device))
v_network.load_state_dict(torch.load(args.v_path, map_location=args.device))

# 평가 모드로 설정
q_network.eval()
v_network.eval()

# ValueDataset에 Q-network와 V-network 전달
dataset_config = utils.Config(
    args.loader,
    savepath=(args.savepath, 'dataset_config.pkl'),
    env=args.dataset,
    horizon=args.horizon,
    normalizer=args.normalizer,
    preprocess_fns=args.preprocess_fns,
    use_padding=args.use_padding,
    max_path_length=args.max_path_length,
    ## value-specific kwargs
    discount=args.discount,
    termination_penalty=args.termination_penalty,
    normed=args.normed,
    q_network=q_network,
    v_network=v_network,
)

render_config = utils.Config(
    args.renderer,
    savepath=(args.savepath, 'render_config.pkl'),
    env=args.dataset,
)

dataset = dataset_config()
renderer = render_config()

observation_dim = dataset.observation_dim
action_dim = dataset.action_dim

#-----------------------------------------------------------------------------#
#------------------------------ model & trainer ------------------------------#
#-----------------------------------------------------------------------------#

model_config = utils.Config(
    args.model,
    savepath=(args.savepath, 'model_config.pkl'),
    horizon=args.horizon,
    transition_dim=observation_dim + action_dim,
    cond_dim=observation_dim,
    dim_mults=args.dim_mults,
    device=args.device,
)

diffusion_config = utils.Config(
    args.diffusion,
    savepath=(args.savepath, 'diffusion_config.pkl'),
    horizon=args.horizon,
    observation_dim=observation_dim,
    action_dim=action_dim,
    n_timesteps=args.n_diffusion_steps,
    loss_type=args.loss_type,
    device=args.device,
)

trainer_config = utils.Config(
    utils.Trainer,
    savepath=(args.savepath, 'trainer_config.pkl'),
    train_batch_size=args.batch_size,
    train_lr=args.learning_rate,
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

#-----------------------------------------------------------------------------#
#-------------------------------- instantiate --------------------------------#
#-----------------------------------------------------------------------------#

model = model_config()

diffusion = diffusion_config(model)

trainer = trainer_config(diffusion, dataset, renderer)

#-----------------------------------------------------------------------------#
#------------------------ test forward & backward pass -----------------------#
#-----------------------------------------------------------------------------#

print('Testing forward...', end=' ', flush=True)
batch = utils.batchify(dataset[0])

loss, _ = diffusion.loss(*batch)
loss.backward()
print('✓')

#-----------------------------------------------------------------------------#
#--------------------------------- main loop ---------------------------------#
#-----------------------------------------------------------------------------#

n_epochs = int(args.n_train_steps // args.n_steps_per_epoch)

for i in range(n_epochs):
    print(f'Epoch {i} / {n_epochs} | {args.savepath}')
    trainer.train(n_train_steps=args.n_steps_per_epoch)
