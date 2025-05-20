import sys
import os

# diffuser_ogbench 경로를 Python path에 추가
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

# 직접 OGBenchGoalDataset import
from diffuser.datasets.ogbench import OGBenchGoalDataset
from diffuser.utils.setup import Parser
from diffuser.models.diffusion import set_model_mode
import diffuser.utils as utils

class Config:
    def __init__(self, cls, **kwargs):
        self.cls = cls
        self.kwargs = kwargs
        self._instance = None

    def __call__(self, *args, **kwargs):
        if self._instance is None:
            self._instance = self.cls(*args, **kwargs, **self.kwargs)
        return self._instance

class Parser(Parser):
    dataset: str = 'scene-play-singletask-task2-v0'
    config: str = 'config.locomotion'
    horizon: int = 64
    n_diffusion_steps: int = 100
    batch_size: int = 32
    learning_rate: float = 2e-4
    normalizer: str = 'LimitsNormalizer'
    use_padding: bool = True
    max_path_length: int = 1000

args = Parser().parse_args('diffusion')
set_model_mode(args.prefix)

# 데이터셋 설정
dataset_config = utils.Config(
    OGBenchGoalDataset,
    savepath=(args.savepath, 'dataset_config.pkl'),
    env_name=args.dataset,
    horizon=args.horizon,
    normalizer=args.normalizer,
    preprocess_fns=args.preprocess_fns,
    use_padding=args.use_padding,
    max_path_length=args.max_path_length,
)

# 데이터셋 초기화
dataset = dataset_config()

observation_dim = dataset.observation_dim
action_dim = dataset.action_dim

# 모델 설정
model_config = utils.Config(
    args.model,
    savepath=(args.savepath, 'model_config.pkl'),
    horizon=args.horizon,
    transition_dim=observation_dim + action_dim,
    cond_dim=observation_dim * 2,  # 현재 상태 + 목표 상태
    dim_mults=args.dim_mults,
    attention=args.attention,
    device=args.device,
)

# Diffusion 설정
diffusion_config = utils.Config(
    args.diffusion,
    savepath=(args.savepath, 'diffusion_config.pkl'),
    horizon=args.horizon,
    observation_dim=observation_dim,
    action_dim=action_dim,
    n_timesteps=args.n_diffusion_steps,
    loss_type=args.loss_type,
    clip_denoised=args.clip_denoised,
    predict_epsilon=args.predict_epsilon,
    action_weight=args.action_weight,
    loss_weights=args.loss_weights,
    loss_discount=args.loss_discount,
    device=args.device,
)

# 트레이너 설정
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

# 모델 초기화
model = model_config()
diffusion = diffusion_config(model)
trainer = trainer_config(diffusion, dataset, None)  # renderer를 None으로 설정

# 학습 시작
n_epochs = int(args.n_train_steps // args.n_steps_per_epoch)
for i in range(n_epochs):
    print(f'Epoch {i} / {n_epochs} | {args.savepath}')
    trainer.train(n_train_steps=args.n_steps_per_epoch) 